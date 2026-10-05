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
template <typename T, int N, int NC, int NR, int MinBlocks>
class PosvTinyKernel;

namespace sycl_posv {

namespace {

namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

constexpr int kTinyWg = tn::kTinyWgSize;
static_assert(kTinyWg == kPosvTinyWgSize, "src/ops/posv/posv.cc's can_run reads kPosvTinyWgSize");

// A launch ABORT, not a slowdown, so it is encoded to fail at COMPILE time. NOW PROBED, and
// the old assumed 256 was a tell: it is above the 255-register ISA ceiling, so no kernel could
// ever have reported it. Measured worst is 255 (cdouble N=16 NR=4), which SPILLS.
// evidence: docs/perf/potrf.md#the-posv-tiny-tier-is-not-register-resident-for-cdouble
constexpr int kWorstRegsPerThread = 255;
// The bound is per SUB-PARTITION: 64 lanes is 2 warps in one of the four, 1 x 32 x 256 =
// 8,192 of its 16,384. evidence: docs/perf/lu.md#the-register-cap-that-binds-is-per-sub-partition
static_assert(resident::sm89_fits(kWorstRegsPerThread, kTinyWg),
              "re-run scripts/register_probe.sh before raising the tiny work-group size");

// The same flat compile-time ceiling potrf_tiny carries, for the same reason: the tier
// allocates no local memory, so no budget walk applies to it.
template <typename T>
constexpr int tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

// The RHS ladder is {1, 2, 4}; every RHS loop stops at nrhs, see gesv_tiny.cc.
constexpr int tiny_rhs_bucket(int nrhs) {
    if (nrhs < 1) return 0;
    if (nrhs <= 2) return nrhs;
    if (nrhs <= kPosvTinyMaxRhs) return kPosvTinyMaxRhs;
    return 0;
}

// A FUNCTOR so the launch bound can be spelled (refused on a lambda); see getrf_tiny.cc,
// whose column bucket NC this shares. Slm moves both cross-lane reads that scale with n
// into local memory: the Cholesky column (every lane stores one element) and the
// backward solve's transposed row (lane i stores its row, each lane loads ONE element).
// evidence: docs/perf/potrf.md#the-posv-local-memory-transpose
template <typename T, int N, int NC, int NR, int Mpw, int MinBlocks, bool Slm>
struct PosvTinyBody {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static constexpr bool real_diag = DM::is_complex;
    using R4 = tn::tiny_r4<R>;
    static constexpr int kCpx = static_cast<int>(sizeof(D) / sizeof(R));
    static constexpr int kW = tn::tiny_r4_width<D>();
    static constexpr int kNV = (N + kW - 1) / kW;   // vectors per buffer: one per lane
    D* ap;
    D* bp;
    int lda, stride_a;
    std::ptrdiff_t ldbp, strbp;
    bool upper;
    int n, nrhs, batch;
    int32_t* info_ptr;
    sycl::local_accessor<R4, 1> slm;   // 2 phases x 2 parities x Mpw; one element when !Slm

    [[sycl::reqd_sub_group_size(32), BATCHLAS_LAUNCH_BOUNDS(kTinyWg, MinBlocks)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int kMpw = Mpw;
        const auto sg = it.get_sub_group();
        const auto part = make_partition<N>(sg);
        const int wg_id = static_cast<int>(it.get_group_linear_id());
        const int lane = static_cast<int>(part.get_local_linear_id());
        const int pidx = tn::tiny_partition_id(sg, part);
        const int prob_id = wg_id * kMpw + pidx;

        // NO EARLY RETURN: tiny_device.hh's third invariant.
        const bool live = (prob_id < batch);
        const bool row_live = live && (lane < n);
        const int b = live ? prob_id : 0;

        const D* __restrict Ag = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
        const D* __restrict Bg = bp + static_cast<std::ptrdiff_t>(b) * strbp;

        // Upper is a LOAD TRANSFORM, not a second algorithm: A = U^H U is the
        // same recurrence on S(i,c) = conj(A(c,i)), so everything below reads
        // "lane r holds row r of L" whichever triangle the caller owns.
        D rA[NC];
        if (upper) {
            tn::tiny_load_upper<D, NC>(rA, Ag, lda, lane, row_live, real_diag);
        } else {
            tn::tiny_load_lower<D, NC>(rA, Ag, lda, lane, row_live, real_diag);
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

        // --- 1. potrf_tiny's recurrence. The order guard is a `continue`, never a
        // `break`: a runtime break defeats the unroll at N >= 16, after which rA[j] is a
        // dynamic index. Skipping the collectives is legal only because n is
        // kernel-uniform (the entry point rejects a heterogeneous batch).
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            if (j >= n) continue;
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

            // Column j of L to every lane: each lane stores ITS element, one barrier,
            // vector loads. Double-buffered by parity like getrf_tiny's pivot row.
            if constexpr (Slm) {
                R4* const sp = &slm[((j & 1) * kMpw + pidx) * kNV];
                R* const rp = reinterpret_cast<R*>(sp);
                if constexpr (kCpx == 2) {
                    rp[2 * lane] = rA[j].re;
                    rp[2 * lane + 1] = rA[j].im;
                } else {
                    rp[lane] = rA[j];
                }
                sycl::group_barrier(sg);
#pragma unroll
                for (int v = 0; v < kNV; ++v) {
                    if ((v + 1) * kW <= j + 1) continue;
                    if (v * kW >= n || v * kW >= NC) continue;
                    const R4 x = sp[v];
#pragma unroll
                    for (int e = 0; e < kW; ++e) {
                        const int k = v * kW + e;
                        if (k <= j || k >= NC || k >= n) continue;
                        const D vk = tn::tiny_r4_get<D, R>(x, e);
                        const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                        rA[k] = tn::tiny_select(act && k < n && k <= lane, upd, rA[k]);
                    }
                }
            } else {
#pragma unroll
                for (int k = j + 1; k < NC; ++k) {
                    if (k >= n) continue;
                    const D vk = tn::tiny_bcast<D>(part, rA[j], static_cast<uint32_t>(k));
                    const D upd = sd::dev_sub(rA[k], sd::dev_mul(rA[j], sd::dev_conj(vk)));
                    // `k <= lane` keeps rA[k] for k > lane exactly zero for the
                    // kernel's whole life, which is what makes the padding inert and
                    // what both solves below rely on.
                    rA[k] = tn::tiny_select(act && k < n && k <= lane, upd, rA[k]);
                }
            }
        }

        // --- 2. forward solve L y = b, right-looking. L(r, i) is rA[i] on lane r,
        // so the only cross-lane traffic is the pivot row's own values.
#pragma unroll
        for (int i = 0; i < NC; ++i) {
            if (i >= n) continue;
            const D lii = tn::tiny_bcast<D>(part, rA[i], static_cast<uint32_t>(i));
            const bool zero = sd::dev_is_zero(lii);
            const D rc = sd::dev_recip(lii);
            const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
#pragma unroll
            for (int k = 0; k < NR; ++k) {
                if (k >= nrhs) continue;   // kernel-uniform
                const D bi = tn::tiny_bcast<D>(part, rB[k], static_cast<uint32_t>(i));
                // A non-positive-definite leading minor is reported through
                // `info` and LAPACK leaves X undefined; zero is chosen over the
                // inf a bare divide would write, so a residual check reports the
                // failure and not a NaN that propagates into every statistic. A
                // BRANCH on partition-uniform flags: a select pays the divide always.
                D yi = D{};
                if (use_mul) {
                    yi = sd::dev_mul(bi, rc);
                } else if (!zero) {
                    yi = sd::dev_div(bi, lii);
                }
                // Lane i writes rX, never rB: it is this step's shuffle source.
                rX[k] = tn::tiny_select(lane == i, yi, rX[k]);
                const D upd = sd::dev_sub(rB[k], sd::dev_mul(rA[i], yi));
                rB[k] = tn::tiny_select(lane > i, upd, rB[k]);
            }
        }

        // --- 3. backward solve L^H x = y. rX now holds y on lane i; rB is reused
        // for x. The transpose read is the header's subject.
#pragma unroll
        for (int i = NC - 1; i >= 0; --i) {
            if (i >= n) continue;

            // conj(L(i, lane)) for lanes below i, assembled from lane i's row.
            // The collective is OUTSIDE the lane guard, as it must be.
            D lir = D{};
            if constexpr (Slm) {
                R4* const sp = &slm[((2 + (i & 1)) * kMpw + pidx) * kNV];
                if (lane == i) {
#pragma unroll
                    for (int v = 0; v < kNV; ++v) {
                        if (v * kW >= i) continue;
                        sp[v] = BATCHLAS_TINY_R4_PACK(D, R, NC, rA, v);
                    }
                }
                sycl::group_barrier(sg);
                if (lane < i) {
                    const R* const rp = reinterpret_cast<const R*>(sp);
                    if constexpr (kCpx == 2) {
                        lir = D{rp[2 * lane], rp[2 * lane + 1]};
                    } else {
                        lir = rp[lane];
                    }
                }
            } else {
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    if (c >= i) continue;   // compile-time once both loops are unrolled
                    const D v = tn::tiny_bcast<D>(part, rA[c], static_cast<uint32_t>(i));
                    lir = tn::tiny_select(c == lane, v, lir);
                }
            }

            const D lii = tn::tiny_bcast<D>(part, rA[i], static_cast<uint32_t>(i));
            const D dii = sd::dev_conj(lii);   // real in exact arithmetic; not assumed
            const bool zero = sd::dev_is_zero(dii);
            const D rc = sd::dev_recip(dii);
            const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);

#pragma unroll
            for (int k = 0; k < NR; ++k) {
                if (k >= nrhs) continue;   // kernel-uniform
                const D yi = tn::tiny_bcast<D>(part, rX[k], static_cast<uint32_t>(i));
                D xi = D{};
                if (use_mul) {
                    xi = sd::dev_mul(yi, rc);
                } else if (!zero) {
                    xi = sd::dev_div(yi, dii);
                }
                // Lane i is the source of rX[k] and of every rA read above, and
                // writes only rB; lanes below i update rX in place.
                rB[k] = tn::tiny_select(lane == i, xi, rB[k]);
                const D upd = sd::dev_sub(rX[k], sd::dev_mul(sd::dev_conj(lir), xi));
                rX[k] = tn::tiny_select(lane < i, upd, rX[k]);
            }
        }

        D* __restrict Aout = ap + static_cast<std::ptrdiff_t>(b) * stride_a;
        if (upper) {
            tn::tiny_store_upper<D, NC>(rA, Aout, lda, lane, row_live);
        } else {
            tn::tiny_store_lower<D, NC>(rA, Aout, lda, lane, row_live);
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
    }
};

// The local-memory reads are for the 32-bit types at N >= 16, where they were measured.
template <typename T, int N>
constexpr bool posv_tiny_slm() {
    return (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) && N >= 16;
}

template <typename T, int N, int NC, int NR, int MinBlocks>
Event posv_tiny_launch_b(Queue& ctx,
                         T* a_ptr, int lda, int stride_a,
                         T* b_ptr, int ldb, int stride_b,
                         bool upper, int n, int nrhs, int batch, int32_t* info_ptr) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(tn::tiny_n_is_legal(N), "the tiny ladder is {8, 16, 32}");
    static_assert(NR == 1 || NR == 2 || NR == kPosvTinyMaxRhs, "the RHS ladder is {1, 2, 4}");
    static_assert(NC <= N && NC % 2 == 0, "the column bucket is even and within the lanes");

    constexpr int kMpw = resident::pack_matrices_per_wg(
        /*bytes_per_matrix=*/1u, N, /*wg_slm_budget_bytes=*/~std::size_t(0),
        kTinyWg, kTinyWg, /*max_pack=*/kTinyWg / N);
    static_assert(kMpw * N == kTinyWg, "the tiny launch must fill its work-group exactly");
    constexpr bool kSlm = posv_tiny_slm<T, N>();
    using Body = PosvTinyBody<T, N, NC, NR, kMpw, MinBlocks, kSlm>;

    const int num_wg = (batch + kMpw - 1) / kMpw;
    const std::size_t slm_elems = kSlm ? std::size_t(4 * kMpw * Body::kNV) : 1;

    ctx->submit([&](sycl::handler& h) {
        const Body body{
            reinterpret_cast<D*>(a_ptr), reinterpret_cast<D*>(b_ptr), lda, stride_a,
            static_cast<std::ptrdiff_t>(ldb), static_cast<std::ptrdiff_t>(stride_b),
            upper, n, nrhs, batch, info_ptr,
            sycl::local_accessor<typename Body::R4, 1>(sycl::range<1>(slm_elems), h)};
        h.parallel_for<PosvTinyKernel<T, N, NC, NR, MinBlocks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(kTinyWg)),
                              sycl::range<1>(static_cast<std::size_t>(kTinyWg))),
            body);
    });
    return ctx.get_event();
}

// The column bucket: {12, 16} under N = 16 and {20, 24, 28, 32} under N = 32.
template <typename T, int N>
constexpr bool posv_nc_legal(int nc) {
    if constexpr (!posv_tiny_slm<T, N>()) return nc == N;
    return N == 16 ? (nc == 12 || nc == 16) : (nc == 20 || nc == 24 || nc == 28 || nc == 32);
}
inline int posv_nc_of(int n, int N) {
    if (N == 16) return n <= 12 ? 12 : 16;
    if (N == 32) return n <= 20 ? 20 : tn::tiny_col_bucket(n, 4);
    return N;
}

// The launch bound per (type, N, NC, NR), each the argmin of a batch-32768 sweep over
// {8, 12, 16}; float is flat within 2% and takes 16 throughout. N = 8 keeps the earlier
// table. evidence: docs/perf/potrf.md#the-posv-local-memory-transpose
template <typename T, int N, int NC, int NR>
constexpr int posv_tiny_min_blocks() {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (N == 8) return NR == 1 ? 1 : 16;
        return 16;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if constexpr (N == 8) return NR == 1 ? 16 : 1;
        if constexpr (N == 16) return (NR == 4 || (NR == 2 && NC == 16)) ? 12 : 16;
        if constexpr (NC >= 28 || (NC == 24 && NR == 4)) return 8;
        return 12;
    } else {
        return 1;
    }
}

template <typename T, int N, int NC, int NR>
Event posv_tiny_launch_nc(Queue& ctx, T* a_ptr, int lda, int stride_a, T* b_ptr, int ldb,
                          int stride_b, bool upper, int n, int nrhs, int batch,
                          int32_t* info_ptr) {
    return posv_tiny_launch_b<T, N, NC, NR, posv_tiny_min_blocks<T, N, NC, NR>()>(
        ctx, a_ptr, lda, stride_a, b_ptr, ldb, stride_b, upper, n, nrhs, batch, info_ptr);
}

template <typename T, int N, int NR>
Event posv_tiny_launch(Queue& ctx,
                       T* a_ptr, int lda, int stride_a,
                       T* b_ptr, int ldb, int stride_b,
                       bool upper, int n, int nrhs, int batch, int32_t* info_ptr) {
    switch (posv_nc_of(n, N)) {
#define BATCHLAS_POSV_TINY_NC(NCC)                                                           \
        case NCC:                                                                            \
            if constexpr (NCC <= N && posv_nc_legal<T, N>(NCC)) {                            \
                return posv_tiny_launch_nc<T, N, NCC, NR>(ctx, a_ptr, lda, stride_a, b_ptr,  \
                                                          ldb, stride_b, upper, n, nrhs,     \
                                                          batch, info_ptr);                  \
            }                                                                                \
            break;
        BATCHLAS_POSV_TINY_NC(8) BATCHLAS_POSV_TINY_NC(12) BATCHLAS_POSV_TINY_NC(16)
        BATCHLAS_POSV_TINY_NC(20) BATCHLAS_POSV_TINY_NC(24) BATCHLAS_POSV_TINY_NC(28)
        BATCHLAS_POSV_TINY_NC(32)
#undef BATCHLAS_POSV_TINY_NC
        default:
            break;
    }
    return posv_tiny_launch_nc<T, N, N, NR>(ctx, a_ptr, lda, stride_a, b_ptr, ldb, stride_b,
                                            upper, n, nrhs, batch, info_ptr);
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

// Every can_run gate is re-applied here; direct callers reach this without the selector.
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
    } else if (rbucket == 2) {
        switch (bucket) {
            case 8:  BATCHLAS_POSV_TINY_ARM(8, 2)  break;
            case 16: BATCHLAS_POSV_TINY_ARM(16, 2) break;
            case 32: BATCHLAS_POSV_TINY_ARM(32, 2) break;
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
