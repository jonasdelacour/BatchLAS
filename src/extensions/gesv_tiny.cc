// Native batched GESV, the fused register-resident tier for order n <= 32, nrhs <= 4:
// getrf_tiny.cc's elimination with the RHS carried in `D rB[NR]` alongside `D rA[N]`, so
// forward substitution happens inside the elimination loop and back substitution follows
// it in the same kernel. One launch, one read of A, no ipiv round trip.
// evidence: docs/perf/lu.md#the-fused-gesv-tier
//
// EVERY INVARIANT OF getrf_tiny.cc STILL HOLDS and two extend to the RHS:
//   1. The lane that wins column j's argmax is, after the relabel, the lane whose rowid
//      IS j, so `act = (rowid > j)` is false for it: it writes neither rA[k] nor rB[k]
//      while it is the source of every shuffle in that iteration.
//   2. B is permuted by the SAME lazy relabel as A, at zero cost, so no laswp is needed.
// The solution vector is NOT permuted -- x_i is the i-th unknown -- so the store writes
// rX at row `rowid`, exactly as the factor store does.

#include "solve_native.hh"

#include "getrf_native.hh"
#include "getrf_cta_device.hh"
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
template <typename T, int N, int NR, int MinBlocks>
class GesvTinyKernel;

namespace sycl_gesv {

namespace {

namespace gn = ::batchlas::getrf_native;
namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

constexpr int kTinyWg = tn::kTinyWgSize;

// A launch ABORT, not a slowdown, so it is encoded to fail at COMPILE time. NOW PROBED: worst
// is cdouble N=16 NR=4 at 182 registers, frame 0, spill 0; every gesv tiny kernel is clean. The
// 224 below is the HEADROOM a new instantiation may use before re-probing, not a measurement.
// evidence: docs/perf/lu.md#p2-the-register-bound-is-assumed-not-probed
constexpr int kWorstRegsPerThread = 224;   // per SUB-PARTITION: 32 x 224 = 7,168 of 16,384
static_assert(resident::sm89_fits(kWorstRegsPerThread, kTinyWg),
              "re-run scripts/register_probe.sh before raising the tiny work-group size");

// cdouble stops at 16 for the same reason getrf_tiny does: 32 rows of a 16-byte scalar
// is already 128 registers before the RHS. evidence: docs/perf/lu.md#d3-why-cdouble-stops-at-n16
template <typename T>
constexpr int tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

// The RHS ladder is {1, 2, 4}: nrhs 3 runs in NR = 4 with one zero column. Every RHS loop
// stops at nrhs (kernel-uniform): a zero pad column sends each divide down its slow path.
// evidence: docs/perf/lu.md#the-rhs-pad-column-cost. 0 means "above the tier".
constexpr int tiny_rhs_bucket(int nrhs) {
    if (nrhs < 1) return 0;
    if (nrhs <= 2) return nrhs;
    if (nrhs <= kGesvTinyMaxRhs) return kGesvTinyMaxRhs;
    return 0;
}

// A FUNCTOR so the launch bound can be spelled (refused on a lambda); see getrf_tiny.cc,
// whose column bucket NC and local-memory pivot row this body shares. The published row
// carries the pivot row of B after that of A, so forward substitution rides the same
// barrier. evidence: docs/perf/lu.md#the-local-memory-broadcast
template <typename D, int N, int NC, int NR, int Mpw, int MinBlocks, bool Slm>
struct GesvTinyBody {
    using R = gn::real_of<D>;
    using R4 = tn::tiny_r4<R>;
    static constexpr int kW = tn::tiny_r4_width<D>();
    static constexpr int kNCV = (NC + kW - 1) / kW;
    static constexpr int kNRV = (NR + kW - 1) / kW;
    static constexpr int kRowV = kNCV + kNRV;   // vectors per published row
    D* ap;
    D* bp;
    int n;
    int nrhs;
    int batch;
    std::ptrdiff_t ldap, ldbp, strap, strbp;
    int* piv_ptr;
    int32_t* info_ptr;
    sycl::local_accessor<R4, 1> slm;   // 2 parities x Mpw rows; one element when !Slm

    [[sycl::reqd_sub_group_size(32), BATCHLAS_LAUNCH_BOUNDS(kTinyWg, MinBlocks)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int kMpw = Mpw;
        const auto sg = it.get_sub_group();
        const auto part = make_partition<N>(sg);
        const int wg_id = static_cast<int>(it.get_group_linear_id());
        const int lane = static_cast<int>(part.get_local_linear_id());
        const int pidx = tn::tiny_partition_id(sg, part);
        const int prob_id = wg_id * kMpw + pidx;

        // CLAMP, DO NOT RETURN: tiny_device.hh's third invariant -- an
        // early-exited lane still sits in the shuffle mask.
        const bool live = (prob_id < batch);
        const int b = live ? prob_id : 0;
        const D* const srcA = ap + static_cast<std::ptrdiff_t>(b) * strap;
        const D* const srcB = bp + static_cast<std::ptrdiff_t>(b) * strbp;

        D rA[NC];  // top level, never a parameter: tiny_device.hh invariant 1
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            rA[c] = tn::tiny_load_pad_identity<D>(srcA, lane, c, n, ldap, live);
        }

        // The pad columns of the RHS are ZERO and never touched: every RHS loop below
        // stops at nrhs, so no column nobody asked for can manufacture a NaN.
        D rB[NR];
        D rX[NR];
#pragma unroll
        for (int k = 0; k < NR; ++k) {
            // A BRANCH, not a `?:` over a 16-byte aggregate: LLVM will not build
            // a `select` of one, SROA then declines to promote the array and it
            // leaves the register file. tiny_device.hh records the same rule.
            D v = D{};
            if (live && (lane < n) && (k < nrhs)) {
                v = srcB[static_cast<std::ptrdiff_t>(lane) +
                         static_cast<std::ptrdiff_t>(k) * ldbp];
            }
            rB[k] = v;
            rX[k] = D{};
        }

        int rowid = lane;      // WHICH matrix row this lane currently owns
        int my_piv = lane;     // lane j accumulates ipiv[j]
        int row_lane = lane;   // lane j records WHICH lane ends up holding row j
        int32_t linfo = 0;     // partition-uniform: from the broadcast pivot

        // Full unroll needs the TU's raised -pragma-unroll-threshold; see getrf_tiny.cc.
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            // `continue`, NOT `break`: a break makes the trip count
            // data-dependent, the unroll is declined and rA/rB leave the register
            // file with zero spill and green tests.
            // evidence: docs/perf/lu.md#the-register-probe-and-the-unroll-that-decides-it
            if (j >= n) continue;

            // --- 1. argmax over live rows, exactly getrf_tiny's: the `mag == mag`
            // map is load-bearing, an unmapped NaN leaves lanes disagreeing.
            const R mag = gn::lu_cabs1<D>(rA[j]);
            const bool cand = (rowid >= j) && (rowid < n);
            R a = (cand && (mag == mag)) ? mag : R(-1);
            int key = tn::tiny_key(cand ? rowid : (rowid + N), lane);
            tn::tiny_argmax_pair<N>(part, a, key);
            const int p = tn::tiny_key_order(key);
            const uint32_t pl = tn::tiny_key_lane(key);

            // THE INVERSE PERMUTATION, FOR FREE. The winner sets rowid = j below
            // and no later step touches a rowid < j, so lane `pl` holds row j for
            // the rest of the kernel. Back substitution needs exactly that map and
            // would otherwise cost an N-step search with no other purpose.
            if (lane == j) { my_piv = p; row_lane = static_cast<int>(pl); }

            if (rowid == p) rowid = j;  // --- 2. lazy swap; no row moves lanes
            else if (rowid == j) rowid = p;

            // --- 3. publish the pivot row of [A | B], double-buffered by parity.
            R4* sp = nullptr;
            if constexpr (Slm) {
                sp = &slm[((j & 1) * kMpw + pidx) * kRowV];
                if (lane == static_cast<int>(pl)) {
#pragma unroll
                    for (int v = 0; v < kNCV; ++v) {
                        if (v < j / kW) continue;
                        sp[v] = BATCHLAS_TINY_R4_PACK(D, R, NC, rA, v);
                    }
#pragma unroll
                    for (int v = 0; v < kNRV; ++v) {
                        if (v * kW >= nrhs) continue;
                        sp[kNCV + v] = BATCHLAS_TINY_R4_PACK(D, R, NR, rB, v);
                    }
                }
                sycl::group_barrier(sg);
            }

            D piv;   // --- 4. ?GETF2
            if constexpr (Slm) {
                piv = tn::tiny_r4_get<D, R>(sp[j / kW], j % kW);
            } else {
                piv = tn::tiny_bcast<D>(part, rA[j], pl);
            }
            const bool zero = sd::dev_is_zero(piv);   // EXACT zero, no epsilon
            if (zero && linfo == 0) linfo = static_cast<int32_t>(j + 1);
            const D rc = sd::dev_recip(piv);
            const bool use_mul =
                !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
            const bool act = (rowid > j);   // NEW rowid: the pivot row is excluded
            if (act) {
                if (use_mul) {
                    rA[j] = sd::dev_mul(rA[j], rc);
                } else if (!zero) {
                    rA[j] = sd::dev_div(rA[j], piv);   // ?GETF2's sfmin arm
                }
            }

            // --- 5. rank-1 update and, THE FUSION, forward substitution: the same
            // update applied to the RHS with the multiplier step 4 just produced, so
            // L y = P b costs no launch, no reload and no barrier of its own.
            if constexpr (Slm) {
#pragma unroll
                for (int v = 0; v < kNCV; ++v) {
                    if (v < j / kW || v * kW >= n) continue;
                    const R4 x = sp[v];
#pragma unroll
                    for (int e = 0; e < kW; ++e) {
                        const int k = v * kW + e;
                        if (k <= j || k >= NC || k >= n) continue;
                        const D u = tn::tiny_r4_get<D, R>(x, e);
                        if (act) rA[k] = sd::dev_sub(rA[k], sd::dev_mul(rA[j], u));
                    }
                }
#pragma unroll
                for (int v = 0; v < kNRV; ++v) {
                    if (v * kW >= nrhs) continue;
                    const R4 x = sp[kNCV + v];
#pragma unroll
                    for (int e = 0; e < kW; ++e) {
                        const int k = v * kW + e;
                        if (k >= NR || k >= nrhs) continue;
                        const D ub = tn::tiny_r4_get<D, R>(x, e);
                        if (act) rB[k] = sd::dev_sub(rB[k], sd::dev_mul(rA[j], ub));
                    }
                }
            } else {
#pragma unroll
                for (int k = j + 1; k < NC; ++k) {
                    if (k >= n) continue;
                    const D u = tn::tiny_bcast<D>(part, rA[k], pl);
                    if (act) rA[k] = sd::dev_sub(rA[k], sd::dev_mul(rA[j], u));
                }
#pragma unroll
                for (int k = 0; k < NR; ++k) {
                    if (k >= nrhs) continue;   // kernel-uniform
                    const D ub = tn::tiny_bcast<D>(part, rB[k], pl);
                    if (act) rB[k] = sd::dev_sub(rB[k], sd::dev_mul(rA[j], ub));
                }
            }
        }

        // --- 6. back substitution, U x = y. `il` is the lane holding row i, read
        // from lane i's `row_lane` with a partition-UNIFORM source, as
        // tiny_device.hh's tiny_bcast contract requires.
#pragma unroll
        for (int i = NC - 1; i >= 0; --i) {
            if (i >= n) continue;

            const uint32_t il = static_cast<uint32_t>(
                select_from_group(part, row_lane, static_cast<uint32_t>(i)));
            const D dii = tn::tiny_bcast<D>(part, rA[i], il);
            const bool zero = sd::dev_is_zero(dii);
            const D rc = sd::dev_recip(dii);
            const bool use_mul =
                !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);

#pragma unroll
            for (int k = 0; k < NR; ++k) {
                if (k >= nrhs) continue;   // kernel-uniform
                const D yi = tn::tiny_bcast<D>(part, rB[k], il);
                // A singular U is reported through `info` and LAPACK leaves X
                // undefined; zero is chosen over the inf a bare divide would
                // write. A BRANCH on the partition-uniform flags, not a select: a
                // select pays the divide on every step. evidence: docs/perf/lu.md#the-rhs-pad-column-cost
                D xi = D{};
                if (use_mul) {
                    xi = sd::dev_mul(yi, rc);
                } else if (!zero) {
                    xi = sd::dev_div(yi, dii);
                }

                // Lane `il` writes rX, never rB: it is this step's shuffle source
                // and a self-update would race every read above it.
                rX[k] = tn::tiny_select(rowid == i, xi, rX[k]);
                const D upd = sd::dev_sub(rB[k], sd::dev_mul(rA[i], xi));
                rB[k] = tn::tiny_select(rowid < i, upd, rB[k]);
            }
        }

        if (live && lane < n) {  // not `rowid < n`: equivalent, loop-invariant
            D* const dstA = ap + static_cast<std::ptrdiff_t>(b) * strap;
            D* const dstB = bp + static_cast<std::ptrdiff_t>(b) * strbp;
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                if (k >= n) continue;
                dstA[static_cast<std::ptrdiff_t>(rowid) +
                     static_cast<std::ptrdiff_t>(k) * ldap] = rA[k];
            }
#pragma unroll
            for (int k = 0; k < NR; ++k) {
                if (k >= nrhs) continue;
                dstB[static_cast<std::ptrdiff_t>(rowid) +
                     static_cast<std::ptrdiff_t>(k) * ldbp] = rX[k];
            }
            // 1-BASED and GLOBAL, as LAPACK defines ipiv.
            piv_ptr[static_cast<std::ptrdiff_t>(b) * n + lane] = my_piv + 1;
        }
        if (live && part.leader()) info_ptr[b] = linfo;
    }
};

// The local-memory row is for the 32-bit types at N >= 16, where it was measured.
template <typename T, int N>
constexpr bool gesv_tiny_slm() {
    return (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) && N >= 16;
}

template <typename T, int N, int NC, int NR, int MinBlocks>
Event gesv_tiny_launch_b(Queue& ctx,
                         T* a_ptr, int lda, int stride_a,
                         T* b_ptr, int ldb, int stride_b,
                         int n, int nrhs, int batch,
                         int* piv_ptr, int32_t* info_ptr) {
    // Re-typed HERE: std::complex's Annex-G operator* costs an isnan branch and a libcall.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(tn::tiny_n_is_legal(N), "the tiny ladder is {8, 16, 32}");
    static_assert(NR == 1 || NR == 2 || NR == kGesvTinyMaxRhs, "the RHS ladder is {1, 2, 4}");
    static_assert(NC <= N && NC % 2 == 0, "the column bucket is even and within the lanes");

    constexpr int kMpw = resident::pack_matrices_per_wg(
        /*bytes_per_matrix=*/1u, N, /*wg_slm_budget_bytes=*/~std::size_t(0),
        kTinyWg, kTinyWg, /*max_pack=*/kTinyWg / N);
    static_assert(kMpw * N == kTinyWg, "the tiny launch must fill its work-group exactly");
    constexpr bool kSlm = gesv_tiny_slm<T, N>();
    using Body = GesvTinyBody<D, N, NC, NR, kMpw, MinBlocks, kSlm>;

    const int num_wg = (batch + kMpw - 1) / kMpw;
    const std::size_t slm_elems = kSlm ? std::size_t(2 * kMpw * Body::kRowV) : 1;

    ctx->submit([&](sycl::handler& h) {
        const Body body{
            reinterpret_cast<D*>(a_ptr), reinterpret_cast<D*>(b_ptr), n, nrhs, batch,
            static_cast<std::ptrdiff_t>(lda), static_cast<std::ptrdiff_t>(ldb),
            static_cast<std::ptrdiff_t>(stride_a), static_cast<std::ptrdiff_t>(stride_b),
            piv_ptr, info_ptr,
            sycl::local_accessor<typename Body::R4, 1>(sycl::range<1>(slm_elems), h)};
        h.parallel_for<GesvTinyKernel<T, N, NC * 8 + NR, MinBlocks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(kTinyWg)),
                              sycl::range<1>(static_cast<std::size_t>(kTinyWg))),
            body);
    });
    return ctx.get_event();
}

// The column bucket, coarser than getrf's: every NC here is paid three times (NR).
template <typename T, int N>
constexpr bool gesv_nc_legal(int nc) {
    if constexpr (!gesv_tiny_slm<T, N>()) return nc == N;
    return N == 16 ? (nc == 12 || nc == 16) : (nc == 20 || nc == 24 || nc == 28 || nc == 32);
}
inline int gesv_nc_of(int n, int N) {
    if (N == 16) return n <= 12 ? 12 : 16;
    if (N == 32) return n <= 20 ? 20 : tn::tiny_col_bucket(n, 4);
    return N;
}

// The launch bound per (type, N, NC, NR), each the argmin of a batch-32768 sweep over
// {8, 12, 16}; float is flat within 2% except NR = 1 at NC >= 28.
// evidence: docs/perf/lu.md#the-column-bucket
template <typename T, int N, int NC, int NR>
constexpr int gesv_tiny_min_blocks() {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (N == 32) return (NR == 1 && NC >= 28) ? 12 : 16;
        return N == 16 ? 16 : 1;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if constexpr (N == 16) return (NC == 16 && NR == 4) ? 12 : 16;
        if constexpr (N == 32) {
            if constexpr (NC == 20) return NR == 1 ? 16 : 12;
            if constexpr (NC == 24) return 12;
            if constexpr (NC == 28) return NR == 2 ? 12 : 8;
            return 8;
        }
        return 1;
    } else {
        return 1;
    }
}

template <typename T, int N, int NC, int NR>
Event gesv_tiny_launch_nc(Queue& ctx, T* a_ptr, int lda, int stride_a, T* b_ptr, int ldb,
                          int stride_b, int n, int nrhs, int batch, int* piv_ptr,
                          int32_t* info_ptr) {
    return gesv_tiny_launch_b<T, N, NC, NR, gesv_tiny_min_blocks<T, N, NC, NR>()>(
        ctx, a_ptr, lda, stride_a, b_ptr, ldb, stride_b, n, nrhs, batch, piv_ptr, info_ptr);
}

template <typename T, int N, int NR>
Event gesv_tiny_launch(Queue& ctx,
                       T* a_ptr, int lda, int stride_a,
                       T* b_ptr, int ldb, int stride_b,
                       int n, int nrhs, int batch,
                       int* piv_ptr, int32_t* info_ptr) {
    switch (gesv_nc_of(n, N)) {
#define BATCHLAS_GESV_TINY_NC(NCC)                                                           \
        case NCC:                                                                            \
            if constexpr (NCC <= N && gesv_nc_legal<T, N>(NCC)) {                            \
                return gesv_tiny_launch_nc<T, N, NCC, NR>(ctx, a_ptr, lda, stride_a, b_ptr,  \
                                                          ldb, stride_b, n, nrhs, batch,     \
                                                          piv_ptr, info_ptr);                \
            }                                                                                \
            break;
        BATCHLAS_GESV_TINY_NC(8) BATCHLAS_GESV_TINY_NC(12) BATCHLAS_GESV_TINY_NC(16)
        BATCHLAS_GESV_TINY_NC(20) BATCHLAS_GESV_TINY_NC(24) BATCHLAS_GESV_TINY_NC(28)
        BATCHLAS_GESV_TINY_NC(32)
#undef BATCHLAS_GESV_TINY_NC
        default:
            break;
    }
    return gesv_tiny_launch_nc<T, N, N, NR>(ctx, a_ptr, lda, stride_a, b_ptr, ldb, stride_b,
                                            n, nrhs, batch, piv_ptr, info_ptr);
}

Span<int32_t> gesv_tiny_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
int gesv_tiny_max_n() {
    return tiny_cap<T>();
}

template <typename T>
std::size_t gesv_tiny_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B) {
    static_cast<void>(B);
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& p) {
        return gesv_tiny_layout(ctx, p, batch);
    });
}

// Every supports() gate is re-applied here. For gesv the usual consequence of missing one
// is worse than elsewhere: there is no batched vendor gesv to fall through to, so a route
// this entry point refuses has nowhere to land.
template <typename T>
Event gesv_tiny_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Span<int64_t> pivots,
                         Span<std::byte> workspace,
                         Span<int32_t> info_out) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || nrhs < 1 || batch < 1) {
        throw batchlas::invalid_argument("gesv_tiny: degenerate extents");
    }
    if (m != n) {
        throw batchlas::invalid_argument("gesv_tiny: A must be square");
    }
    if (B.rows() != A.rows()) {
        throw batchlas::invalid_argument("gesv_tiny: B must have A.rows() rows");
    }
    if (B.batch_size() != A.batch_size()) {
        throw batchlas::invalid_argument("gesv_tiny: A and B must share a batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        // Not merely unsupported: the unrolled body's `if (j >= n) continue` skips a
        // collective, which is legal only because n is kernel-uniform.
        throw batchlas::invalid_argument("gesv_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("gesv_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports entry [0].
        throw batchlas::unsupported(
            "gesv_tiny: device does not offer sub-group size 32, which the kernel requires");
    }
    if (static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) < kTinyWg) {
        throw batchlas::unsupported(
            "gesv_tiny: the tier launches a work-group of " + std::to_string(kTinyWg) +
            ", which exceeds this device's maximum");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("gesv_tiny: pivot span is shorter than n * batch");
    }

    const int bucket = tn::tiny_bucket_ge(n);
    if (bucket < 1 || bucket > tiny_cap<T>()) {
        throw batchlas::unsupported(
            "gesv_tiny: order " + std::to_string(n) +
            " is above this type's register-resident ceiling of " +
            std::to_string(tiny_cap<T>()));
    }
    const int rbucket = tiny_rhs_bucket(nrhs);
    if (rbucket < 1) {
        throw batchlas::unsupported(
            "gesv_tiny: nrhs " + std::to_string(nrhs) + " is above the tier's " +
            std::to_string(kGesvTinyMaxRhs));
    }

    BumpAllocator pool(workspace);
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : gesv_tiny_layout(ctx, pool, batch);

    auto piv_i32 = pivots.as_span<int>();

    // A nested switch rather than a table: the <T, N, NR> cross-product is instantiated
    // implicitly from here, so a missing arm is a link error, not a silent fallback.
#define BATCHLAS_GESV_TINY_ARM(NN, RR)                                                 \
    if constexpr (tiny_cap<T>() >= (NN)) {                                             \
        return gesv_tiny_launch<T, NN, RR>(ctx, A.data_ptr(), A.ld(), A.stride(),       \
                                           B.data_ptr(), B.ld(), B.stride(), n, nrhs,   \
                                           batch, piv_i32.data(), info.data());         \
    }

    if (rbucket == 1) {
        switch (bucket) {
            case 8:  BATCHLAS_GESV_TINY_ARM(8, 1)  break;
            case 16: BATCHLAS_GESV_TINY_ARM(16, 1) break;
            case 32: BATCHLAS_GESV_TINY_ARM(32, 1) break;
            default: break;
        }
    } else if (rbucket == 2) {
        switch (bucket) {
            case 8:  BATCHLAS_GESV_TINY_ARM(8, 2)  break;
            case 16: BATCHLAS_GESV_TINY_ARM(16, 2) break;
            case 32: BATCHLAS_GESV_TINY_ARM(32, 2) break;
            default: break;
        }
    } else {
        switch (bucket) {
            case 8:  BATCHLAS_GESV_TINY_ARM(8, kGesvTinyMaxRhs)  break;
            case 16: BATCHLAS_GESV_TINY_ARM(16, kGesvTinyMaxRhs) break;
            case 32: BATCHLAS_GESV_TINY_ARM(32, kGesvTinyMaxRhs) break;
            default: break;
        }
    }
#undef BATCHLAS_GESV_TINY_ARM

    // Unreachable: thrown rather than falling through to a partial solve.
    throw batchlas::unsupported(
        "gesv_tiny: no instantiation for order " + std::to_string(n) +
        " with nrhs " + std::to_string(nrhs));
}

// Per scalar type only; the switch above pulls the <T, N, NR> cross-product in implicitly.
#define BATCHLAS_GESV_TINY_INSTANTIATE(T)                                                   \
    template int gesv_tiny_max_n<T>();                                                      \
    template std::size_t gesv_tiny_buffer_size<T>(                                          \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                  \
        const MatrixView<T, MatrixFormat::Dense>&);                                         \
    template Event gesv_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                         const MatrixView<T, MatrixFormat::Dense>&,         \
                                         Span<int64_t>, Span<std::byte>, Span<int32_t>);

BATCHLAS_GESV_TINY_INSTANTIATE(float)
BATCHLAS_GESV_TINY_INSTANTIATE(double)
BATCHLAS_GESV_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_GESV_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GESV_TINY_INSTANTIATE

}  // namespace sycl_gesv
}  // namespace batchlas
