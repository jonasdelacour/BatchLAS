// Native batched POSV, the fused register-resident tier for order n <= 32, nrhs <= 4:
// potrf_tiny.cc's unblocked Cholesky followed, in the same kernel, by both triangular
// solves with L still in registers. Zero local memory, zero barriers, every cross-lane
// value a sub-group shuffle. evidence: docs/perf/potrf.md#the-fused-posv-tier
//
// THE ASYMMETRY THAT DECIDES THE DESIGN, and it is the one place the plan's text is
// wrong. P2 says the backward solve needs no transpose "because every lane holds a full
// row". It does not follow: lane r holds ROW r of L, so `L y = b` reads L(r, i) = rA[i]
// locally and is a cheap right-looking sweep, but `L^H x = y` needs L(i, r) -- COLUMN
// access -- which no lane has. This kernel buys that with an explicit on-the-fly
// transpose: at step i, lane i broadcasts rA[0..i-1] and lane r keeps the one element
// where c == r. It costs N(N-1)/2 shuffles, independent of nrhs, i.e. the same order as
// the factorization itself. evidence: docs/perf/potrf.md#posv-the-backward-solve-costs-a-transpose

#include "solve_native.hh"

#include "potrf_native.hh"
#include "tiny_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/error.hh>
#include <batchlas/util/mempool.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace batchlas {

// At namespace scope: a kernel name must not name an internal-linkage entity.
template <typename T, int N, int NR>
class PosvTinyKernel;

namespace sycl_posv {

namespace {

namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

constexpr int kTinyWg = tn::kTinyWgSize;

// A launch ABORT, not a slowdown, so it is encoded to fail at COMPILE time. A BOUND TO
// RE-PROBE, not a measurement: potrf_tiny's worst probed count plus the 2*NR scalars this
// tier adds. evidence: docs/perf/potrf.md#p2-the-register-bound-is-assumed-not-probed
constexpr int kWorstRegsPerThread = 256;
static_assert(kTinyWg * kWorstRegsPerThread <= 65536,
              "re-run scripts/register_probe.sh before raising the tiny work-group size");

// The same flat compile-time ceiling potrf_tiny carries, for the same reason: the tier
// allocates no local memory, so no budget walk applies to it.
template <typename T>
constexpr int tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

constexpr int tiny_rhs_bucket(int nrhs) {
    if (nrhs < 1) return 0;
    if (nrhs <= 1) return 1;
    if (nrhs <= kPosvTinyMaxRhs) return kPosvTinyMaxRhs;
    return 0;
}

template <typename T, int N, int NR>
Event posv_tiny_launch(Queue& ctx,
                       T* a_ptr, int lda, int stride_a,
                       T* b_ptr, int ldb, int stride_b,
                       bool upper, int n, int nrhs, int batch, int32_t* info_ptr) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(tn::tiny_n_is_legal(N), "the tiny ladder is {8, 16, 32}");
    static_assert(NR == 1 || NR == kPosvTinyMaxRhs, "the RHS ladder is {1, 4}");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const bp = reinterpret_cast<D*>(b_ptr);

    constexpr int kMpw = resident::pack_matrices_per_wg(
        /*bytes_per_matrix=*/1u, N, /*wg_slm_budget_bytes=*/~std::size_t(0),
        kTinyWg, kTinyWg, /*max_pack=*/kTinyWg / N);
    static_assert(kMpw * N == kTinyWg, "the tiny launch must fill its work-group exactly");

    const int num_wg = (batch + kMpw - 1) / kMpw;
    const std::ptrdiff_t ldbp = static_cast<std::ptrdiff_t>(ldb);
    const std::ptrdiff_t strbp = static_cast<std::ptrdiff_t>(stride_b);

    // A Hermitian diagonal's imaginary part is contractually ignored; for a real type
    // the transform is the identity.
    constexpr bool real_diag = DM::is_complex;

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<PosvTinyKernel<T, N, NR>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(kTinyWg)),
                              sycl::range<1>(static_cast<std::size_t>(kTinyWg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<N>(sg);
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int lane = static_cast<int>(part.get_local_linear_id());
                const int prob_id = wg_id * kMpw + tn::tiny_partition_id(sg, part);

                // NO EARLY RETURN: tiny_device.hh's third invariant.
                const bool live = (prob_id < batch);
                const bool row_live = live && (lane < n);
                const int b = live ? prob_id : 0;

                const D* __restrict Ag = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
                const D* __restrict Bg = bp + static_cast<std::ptrdiff_t>(b) * strbp;

                // Upper is a LOAD TRANSFORM, not a second algorithm: A = U^H U is the
                // same recurrence on S(i,c) = conj(A(c,i)), so everything below reads
                // "lane r holds row r of L" whichever triangle the caller owns.
                D rA[N];
                if (upper) {
                    tn::tiny_load_upper<D, N>(rA, Ag, lda, lane, row_live, real_diag);
                } else {
                    tn::tiny_load_lower<D, N>(rA, Ag, lda, lane, row_live, real_diag);
                }

                D rB[NR];   // b, then y
                D rX[NR];   // y on the pivot lane, then x
#pragma unroll
                for (int k = 0; k < NR; ++k) {
                    // A BRANCH, not a `?:` over a 16-byte aggregate; see tiny_device.hh.
                    D v = D{};
                    if (row_live && k < nrhs) {
                        v = Bg[static_cast<std::ptrdiff_t>(lane) +
                               static_cast<std::ptrdiff_t>(k) * ldbp];
                    }
                    rB[k] = v;
                    rX[k] = D{};
                }

                bool alive = live;
                int32_t linfo = 0;

                // --- 1. potrf_tiny's recurrence, verbatim. The order guard is a
                // PREDICATE, never a `break`: a runtime break defeats the unroll at
                // N >= 16, after which rA[j] is a dynamic index.
#pragma unroll
                for (int j = 0; j < N; ++j) {
                    const bool col_live = (j < n);
                    const R akk = select_from_group(part, sd::dev_real(rA[j]),
                                                    static_cast<uint32_t>(j));
                    // `!(akk > 0)`, not `akk <= 0`, so NaN is rejected too, as LAPACK does.
                    const bool bad = !(akk > R(0));
                    if (alive && col_live && bad) {
                        linfo = j + 1;   // 1-based; sticky, so FIRST FAILURE WINS
                        alive = false;
                    }
                    // Every failure effect is a value substitution, never control flow:
                    // `alive` is partition-uniform but NOT sub-group-uniform.
                    const bool act = alive && col_live;
                    const R dkk = act ? sycl::sqrt(akk) : R(1);
                    const R rinv = R(1) / dkk;   // NOT rsqrt: rsqrt.approx is not the reference

                    rA[j] = tn::tiny_select(
                        act,
                        tn::tiny_select(lane == j, sd::dev_from_real<D>(dkk),
                                        sd::dev_mul_real(rA[j], rinv)),
                        rA[j]);

#pragma unroll
                    for (int k = j + 1; k < N; ++k) {
                        const D vk = tn::tiny_bcast<D>(part, rA[j], static_cast<uint32_t>(k));
                        const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                        // `k <= lane` keeps rA[k] for k > lane exactly zero for the
                        // kernel's whole life, which is what makes the padding inert and
                        // what both solves below rely on.
                        rA[k] = tn::tiny_select(act && k < n && k <= lane, upd, rA[k]);
                    }
                }

                // --- 2. forward solve L y = b, right-looking. L(r, i) is rA[i] on lane r,
                // so the only cross-lane traffic is the pivot row's own values.
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (i >= n) continue;
                    const D lii = tn::tiny_bcast<D>(part, rA[i], static_cast<uint32_t>(i));
                    const bool zero = sd::dev_is_zero(lii);
                    const D rc = sd::dev_recip(lii);
                    const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
#pragma unroll
                    for (int k = 0; k < NR; ++k) {
                        const D bi = tn::tiny_bcast<D>(part, rB[k], static_cast<uint32_t>(i));
                        // A non-positive-definite leading minor is reported through
                        // `info` and LAPACK leaves X undefined; zero is chosen over the
                        // inf a bare divide would write, so a residual check reports the
                        // failure and not a NaN that propagates into every statistic.
                        const D yi = tn::tiny_select(
                            zero, D{},
                            tn::tiny_select(use_mul, sd::dev_mul(bi, rc),
                                            sd::dev_div(bi, lii)));
                        // Lane i writes rX, never rB: it is this step's shuffle source.
                        rX[k] = tn::tiny_select(lane == i, yi, rX[k]);
                        const D upd = sd::dev_sub(rB[k], sd::dev_mul(rA[i], yi));
                        rB[k] = tn::tiny_select(lane > i, upd, rB[k]);
                    }
                }

                // --- 3. backward solve L^H x = y. rX now holds y on lane i; rB is reused
                // for x. The transpose read is the header's subject.
#pragma unroll
                for (int i = N - 1; i >= 0; --i) {
                    if (i >= n) continue;

                    // conj(L(i, lane)) for lanes below i, assembled from lane i's row.
                    // The collective is OUTSIDE the lane guard, as it must be.
                    D lir = D{};
#pragma unroll
                    for (int c = 0; c < N; ++c) {
                        if (c >= i) continue;   // compile-time once both loops are unrolled
                        const D v = tn::tiny_bcast<D>(part, rA[c], static_cast<uint32_t>(i));
                        lir = tn::tiny_select(c == lane, v, lir);
                    }

                    const D lii = tn::tiny_bcast<D>(part, rA[i], static_cast<uint32_t>(i));
                    const D dii = sd::dev_conj(lii);   // real in exact arithmetic; not assumed
                    const bool zero = sd::dev_is_zero(dii);
                    const D rc = sd::dev_recip(dii);
                    const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);

#pragma unroll
                    for (int k = 0; k < NR; ++k) {
                        const D yi = tn::tiny_bcast<D>(part, rX[k], static_cast<uint32_t>(i));
                        const D xi = tn::tiny_select(
                            zero, D{},
                            tn::tiny_select(use_mul, sd::dev_mul(yi, rc),
                                            sd::dev_div(yi, dii)));
                        // Lane i is the source of rX[k] and of every rA read above, and
                        // writes only rB; lanes below i update rX in place.
                        rB[k] = tn::tiny_select(lane == i, xi, rB[k]);
                        const D upd = sd::dev_sub(rX[k], sd::dev_mul(sd::dev_conj(lir), xi));
                        rX[k] = tn::tiny_select(lane < i, upd, rX[k]);
                    }
                }

                D* __restrict Aout = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
                if (upper) {
                    tn::tiny_store_upper<D, N>(rA, Aout, lda, lane, row_live);
                } else {
                    tn::tiny_store_lower<D, N>(rA, Aout, lda, lane, row_live);
                }

                if (row_live) {
                    D* const dstB = bp + static_cast<std::ptrdiff_t>(b) * strbp;
#pragma unroll
                    for (int k = 0; k < NR; ++k) {
                        if (k >= nrhs) continue;
                        // No permutation: x_i is the i-th unknown and lane i holds it.
                        dstB[static_cast<std::ptrdiff_t>(lane) +
                             static_cast<std::ptrdiff_t>(k) * ldbp] = rB[k];
                    }
                }
                if (live && part.leader()) info_ptr[b] = linfo;
            });
    });
    return ctx.get_event();
}

Span<int32_t> posv_tiny_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
int posv_tiny_max_n() {
    return tiny_cap<T>();
}

template <typename T>
std::size_t posv_tiny_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B) {
    static_cast<void>(B);
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& p) {
        return posv_tiny_layout(ctx, p, batch);
    });
}

// Every supports() gate is re-applied here; there is no vendor posv to fall through to.
template <typename T>
Event posv_tiny_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Uplo uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info_out) {
    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (A.rows() != A.cols()) {
        throw batchlas::invalid_argument("posv_tiny: A must be square");
    }
    if (n < 1 || nrhs < 1 || batch < 1) {
        throw batchlas::invalid_argument("posv_tiny: degenerate extents");
    }
    if (B.rows() != A.rows()) {
        throw batchlas::invalid_argument("posv_tiny: B must have A.rows() rows");
    }
    if (B.batch_size() != A.batch_size()) {
        throw batchlas::invalid_argument("posv_tiny: A and B must share a batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        throw batchlas::invalid_argument("posv_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("posv_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never MAX_SUB_GROUP_SIZE >= 32, which reports entry [0].
        throw batchlas::unsupported(
            "posv_tiny: device does not offer sub-group size 32, which the kernel requires");
    }
    if (static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) < kTinyWg) {
        throw batchlas::unsupported(
            "posv_tiny: the device's maximum work-group size is below the tier's " +
            std::to_string(kTinyWg));
    }
    const int bucket = tn::tiny_bucket_ge(n);
    if (bucket < 1 || bucket > tiny_cap<T>()) {
        throw batchlas::unsupported(
            "posv_tiny: order " + std::to_string(n) +
            " is above this type's register-resident ceiling of " +
            std::to_string(tiny_cap<T>()));
    }
    const int rbucket = tiny_rhs_bucket(nrhs);
    if (rbucket < 1) {
        throw batchlas::unsupported(
            "posv_tiny: nrhs " + std::to_string(nrhs) + " is above the tier's " +
            std::to_string(kPosvTinyMaxRhs));
    }

    BumpAllocator pool(workspace);
    // An empty or short caller span means "not requested" and draws pool scratch.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : posv_tiny_layout(ctx, pool, batch);

    const bool upper = (uplo == Uplo::Upper);

#define BATCHLAS_POSV_TINY_ARM(NN, RR)                                                 \
    if constexpr (tiny_cap<T>() >= (NN)) {                                             \
        return posv_tiny_launch<T, NN, RR>(ctx, A.data_ptr(), A.ld(), A.stride(),       \
                                           B.data_ptr(), B.ld(), B.stride(), upper, n,  \
                                           nrhs, batch, info.data());                   \
    }

    if (rbucket == 1) {
        switch (bucket) {
            case 8:  BATCHLAS_POSV_TINY_ARM(8, 1)  break;
            case 16: BATCHLAS_POSV_TINY_ARM(16, 1) break;
            case 32: BATCHLAS_POSV_TINY_ARM(32, 1) break;
            default: break;
        }
    } else {
        switch (bucket) {
            case 8:  BATCHLAS_POSV_TINY_ARM(8, kPosvTinyMaxRhs)  break;
            case 16: BATCHLAS_POSV_TINY_ARM(16, kPosvTinyMaxRhs) break;
            case 32: BATCHLAS_POSV_TINY_ARM(32, kPosvTinyMaxRhs) break;
            default: break;
        }
    }
#undef BATCHLAS_POSV_TINY_ARM

    throw batchlas::unsupported(
        "posv_tiny: no instantiation for order " + std::to_string(n) +
        " with nrhs " + std::to_string(nrhs));
}

// Per scalar type only; the switch above pulls the <T, N, NR> cross-product in implicitly.
#define BATCHLAS_POSV_TINY_INSTANTIATE(T)                                                   \
    template int posv_tiny_max_n<T>();                                                      \
    template std::size_t posv_tiny_buffer_size<T>(                                          \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                  \
        const MatrixView<T, MatrixFormat::Dense>&);                                         \
    template Event posv_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                         const MatrixView<T, MatrixFormat::Dense>&, Uplo,   \
                                         Span<std::byte>, Span<int32_t>);

BATCHLAS_POSV_TINY_INSTANTIATE(float)
BATCHLAS_POSV_TINY_INSTANTIATE(double)
BATCHLAS_POSV_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_POSV_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POSV_TINY_INSTANTIATE

}  // namespace sycl_posv
}  // namespace batchlas
