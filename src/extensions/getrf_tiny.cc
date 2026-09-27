// Native batched GETRF, the register-resident tier for order n <= 32: one matrix per
// SubGroupPartition<N>, row r in lane r of a `D rA[N]`, pivoting by lazy relabel.
// evidence: docs/perf/lu.md#the-tiny-tier
// THE INVARIANT THAT REMOVES EVERY BARRIER: the lane that wins column j's argmax is, after
// the relabel, the lane whose rowid IS j, so `act = (rowid > j)` is false for it and it never
// writes rA[k] that iteration; every other lane therefore reads a value the source lane is
// not concurrently changing. Letting the pivot lane update itself races every shuffle below.
// The pivot metric is LAPACK's cabs1, not cuBLAS's modulus, so a pivot test needs a HOST
// oracle; ipiv is 1-based and GLOBAL, int32 in the public int64 span.

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
namespace sycl_getrf {

// At TU namespace scope: a kernel name must not name an internal-linkage entity.
template <typename T, int N, int NC, int MinBlocks> class GetrfTinyKernel;

namespace {

namespace gn = ::batchlas::getrf_native;
namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

// The shared tiny-tier work-group. A wider group makes the launch tail COARSER, not finer.
// evidence: docs/perf/lu.md#the-work-group-ab
constexpr int kTinyWg = tn::kTinyWgSize;

// A launch ABORT, not a slowdown, so it is encoded to fail at COMPILE time; per SUB-PARTITION,
// not per block. evidence: docs/perf/lu.md#the-register-probe-at-wg--64
constexpr int kWorstRegsPerThread = 143;   // 64 lanes = 2 warps: 32 x 144 = 4,608 of 16,384
static_assert(resident::sm89_fits(kWorstRegsPerThread, kTinyWg),
              "re-run scripts/register_probe.sh out.log '' batchlas_extensions_cta "
              "before raising the tiny tier's work-group size");

// 0 means "above the tier" -- `unsupported`, never a silent leading-submatrix factorisation.
constexpr int tiny_bucket(int n) {
    if (n < 1) return 0;
    if (n <= 4) return 4;   // eight matrices per sub-group; n = 4 half-fills N = 8
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    if (n <= 32) return 32;
    return 0;
}

// evidence: docs/perf/lu.md#d3-why-cdouble-stops-at-n16
template <typename T>
constexpr int tiny_cap() {
    return std::is_same_v<T, std::complex<double>> ? 16 : 32;
}

// A FUNCTOR, not a lambda, only so the launch bound can be spelled: the attribute that
// emits ptxas's .minnctapersm is refused on a lambda. MinBlocks caps the registers the
// allocator may take (65,536 / (64 * MinBlocks)); 1 is no cap. NC <= N is the column
// bucket (tiny_device.hh); Slm publishes the pivot row through local memory instead of
// one shuffle per element (two per complex). evidence: docs/perf/lu.md#the-local-memory-broadcast
template <typename D, int N, int NC, int Mpw, int MinBlocks, bool Slm>
struct GetrfTinyBody {
    using R = gn::real_of<D>;
    using R4 = tn::tiny_r4<R>;
    static constexpr int kW = tn::tiny_r4_width<D>();
    static constexpr int kNCV = (NC + kW - 1) / kW;   // vectors per published row
    D* ap;
    int n;
    int batch;
    std::ptrdiff_t ldp;
    std::ptrdiff_t strp;
    int* piv_ptr;
    int32_t* info_ptr;
    sycl::local_accessor<R4, 1> slm;   // 2 parities x Mpw rows; one element when !Slm

    [[sycl::reqd_sub_group_size(32), intel::max_work_group_size(1, 1, kTinyWg),
      intel::min_work_groups_per_cu(MinBlocks)]]
    void operator()(sycl::nd_item<1> it) const {
        constexpr int kMpw = Mpw;
        const auto sg = it.get_sub_group();
        const auto part = make_partition<N>(sg);
        const int wg_id = static_cast<int>(it.get_group_linear_id());
        const int lane = static_cast<int>(part.get_local_linear_id());
        const int pidx = tn::tiny_partition_id(sg, part);
        const int prob_id = wg_id * kMpw + pidx;

        // CLAMP, DO NOT RETURN (as steqr_cta.cc does too): tiny_device.hh's
        // third invariant -- an early-exited lane still sits in the shuffle mask.
        const bool live = (prob_id < batch);
        const int b = live ? prob_id : 0;
        const D* const src = ap + static_cast<std::ptrdiff_t>(b) * strp;

        D rA[NC];  // top level, never a parameter: tiny_device.hh invariant 1
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            rA[c] = tn::tiny_load_pad_identity<D>(src, lane, c, n, ldp, live);
        }

        int rowid = lane;    // WHICH matrix row this lane currently owns
        int my_piv = lane;   // lane j accumulates ipiv[j]
        int32_t linfo = 0;   // partition-uniform: from the broadcast pivot

        // The unroll is FULL only under the TU's raised -pragma-unroll-threshold
        // (src/CMakeLists.txt): above LLVM's default it is declined silently and rA[]
        // lands on the stack with zero spill. evidence: docs/perf/lu.md#the-column-bucket
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            // `continue`, NOT `break`. A break makes the trip count data-dependent,
            // the toolchain declines the unroll, rA becomes dynamically indexed and
            // ptxas relocates it to the stack -- with ZERO spill and green tests, so
            // only the probe's stack-frame column shows it. Skipping a collective
            // here is legal ONLY because n is kernel-uniform, which holds only
            // because the entry point rejects a heterogeneous batch; relax that and
            // every collective below is undefined -- a wrong answer, not a crash.
            // evidence: docs/perf/lu.md#the-register-probe-and-the-unroll-that-decides-it
            if (j >= n) continue;

            // --- 1. argmax over live rows. A pad row is kept out by the tie-break
            // DIRECTION, not by the `rowid < n` mask; the `mag == mag` map IS
            // load-bearing -- an unmapped NaN leaves lanes disagreeing on the winner.
            // evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
            const R mag = gn::lu_cabs1<D>(rA[j]);
            const bool cand = (rowid >= j) && (rowid < n);
            R a = (cand && (mag == mag)) ? mag : R(-1);
            // `rowid + N`: every non-candidate sorts strictly BELOW every candidate.
            int key = tn::tiny_key(cand ? rowid : (rowid + N), lane);
            tn::tiny_argmax_pair<N>(part, a, key);
            const int p = tn::tiny_key_order(key);        // winner's rowid
            const uint32_t pl = tn::tiny_key_lane(key);   // winner's lane
            if (lane == j) my_piv = p;

            if (rowid == p) rowid = j;  // --- 2. lazy swap; no row moves lanes
            else if (rowid == j) rowid = p;

            // --- 3. publish the pivot row. DOUBLE-BUFFERED by step parity: one barrier
            // per step orders this store after every lane's reads two steps back.
            R4* sp = nullptr;
            if constexpr (Slm) {
                sp = &slm[((j & 1) * kMpw + pidx) * kNCV];
                if (lane == static_cast<int>(pl)) {
#pragma unroll
                    for (int v = 0; v < kNCV; ++v) {
                        if (v < j / kW) continue;
                        sp[v] = BATCHLAS_TINY_R4_PACK(D, R, NC, rA, v);
                    }
                }
                sycl::group_barrier(sg);
            }

            // --- 4. ?GETF2
            D piv;
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

            // --- 5. rank-1 update, collective OUTSIDE the guard. A zero pivot was
            // the argmax, so every multiplier is zero: LAPACK's ?GER as a no-op.
            // `k = j + 1` is a COMPILE-TIME lower bound once j is unrolled.
            // evidence: docs/perf/lu.md#why-the-rank-1-update-starts-at-k--j--1
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
            } else {
#pragma unroll
                for (int k = j + 1; k < NC; ++k) {
                    if (k >= n) continue;      // kernel-uniform, as above
                    const D u = tn::tiny_bcast<D>(part, rA[k], pl);
                    if (act) rA[k] = sd::dev_sub(rA[k], sd::dev_mul(rA[j], u));
                }
            }
        }

        if (live) {
            D* const dst = ap + static_cast<std::ptrdiff_t>(b) * strp;
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                if (k >= n) continue;
                if (lane < n) {  // not `rowid < n`: equivalent, and loop-invariant
                    dst[static_cast<std::ptrdiff_t>(rowid) +
                        static_cast<std::ptrdiff_t>(k) * ldp] = rA[k];
                }
            }
            if (lane < n) {
                // 1-BASED and GLOBAL, as LAPACK defines ipiv.
                piv_ptr[static_cast<std::ptrdiff_t>(b) * n + lane] = my_piv + 1;
            }
            if (part.leader()) info_ptr[b] = linfo;
        }
    }
};

// The local-memory broadcast is for the 32-bit types, where it was measured; fp64 runs at
// 1/64 rate on this part and keeps the shuffles.
template <typename T>
constexpr bool getrf_tiny_slm() {
    return std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
}

template <typename T, int N, int NC, int MinBlocks>
Event getrf_tiny_launch(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                        int* piv_ptr, int32_t* info_ptr) {
    // Re-typed HERE: std::complex's Annex-G operator* costs an isnan branch and a libcall.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(N == 4 || N == 8 || N == 16 || N == 32, "the tiny ladder is {4, 8, 16, 32}");
    static_assert(NC <= N && NC % 2 == 0, "the column bucket is even and within the lanes");

    // Via the shared helper, not kTinyWg / N: the guarantees stay where they are owned.
    constexpr int kMpw = resident::pack_matrices_per_wg(
        /*bytes_per_matrix=*/1u, N, /*wg_slm_budget_bytes=*/~std::size_t(0),
        kTinyWg, kTinyWg, /*max_pack=*/kTinyWg / N);
    static_assert(kMpw * N == kTinyWg, "the tiny launch must fill its work-group exactly");
    constexpr bool kSlm = getrf_tiny_slm<T>() && N >= 16;
    using Body = GetrfTinyBody<D, N, NC, kMpw, MinBlocks, kSlm>;
    const int num_wg = (batch + kMpw - 1) / kMpw;
    const std::size_t slm_elems = kSlm ? std::size_t(2 * kMpw * Body::kNCV) : 1;

    ctx->submit([&](sycl::handler& h) {
        const Body body{reinterpret_cast<D*>(a_ptr), n, batch, static_cast<std::ptrdiff_t>(ld),
                        static_cast<std::ptrdiff_t>(stride), piv_ptr, info_ptr,
                        sycl::local_accessor<typename Body::R4, 1>(sycl::range<1>(slm_elems), h)};
        h.parallel_for<GetrfTinyKernel<T, N, NC, MinBlocks>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(kTinyWg)),
                              sycl::range<1>(static_cast<std::size_t>(kTinyWg))),
            body);
    });
    return ctx.get_event();
}

// The launch bound per (type, column bucket), each the argmin of a batch-32768 sweep over
// {8, 12, 16, 20} (N = 32) or {1, 8, 12, 16} (N = 16); fp64 unmeasured, no cap.
// evidence: docs/perf/lu.md#the-column-bucket
template <typename T, int N, int NC>
constexpr int getrf_tiny_min_blocks() {
    if constexpr (std::is_same_v<T, float>) {
        if constexpr (N == 32) {
            return (NC == 18 || NC == 22 || NC == 24) ? 20 : NC == 26 ? 12 : 16;
        }
        return N == 16 ? 16 : 1;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        if constexpr (N == 32) {
            return (NC == 22 || NC == 24) ? 16 : NC >= 30 ? 8 : 12;
        }
        return N == 16 ? 16 : 1;
    } else {
        return 1;
    }
}

template <typename T, int N, int NC>
Event getrf_tiny_launch_nc(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                           int* piv_ptr, int32_t* info_ptr) {
    return getrf_tiny_launch<T, N, NC, getrf_tiny_min_blocks<T, N, NC>()>(
        ctx, a_ptr, ld, stride, n, batch, piv_ptr, info_ptr);
}

// Order -> column bucket inside lane bucket N: every even order for the 32-bit types (one
// instantiation per bucket; the step is the measured cost), the lane count for fp64.
template <typename T, int N>
Event getrf_tiny_launch_mb(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                           int* piv_ptr, int32_t* info_ptr) {
    if constexpr (getrf_tiny_slm<T>() && N >= 16) {
        switch (tn::tiny_col_bucket(n, 2)) {
#define BATCHLAS_GETRF_TINY_NC(NCC)                                                            \
            case NCC:                                                                          \
                if constexpr (NCC <= N && NCC > N / 2) {                                       \
                    return getrf_tiny_launch_nc<T, N, NCC>(ctx, a_ptr, ld, stride, n, batch,   \
                                                           piv_ptr, info_ptr);                 \
                }                                                                              \
                break;
            BATCHLAS_GETRF_TINY_NC(10) BATCHLAS_GETRF_TINY_NC(12) BATCHLAS_GETRF_TINY_NC(14)
            BATCHLAS_GETRF_TINY_NC(16) BATCHLAS_GETRF_TINY_NC(18) BATCHLAS_GETRF_TINY_NC(20)
            BATCHLAS_GETRF_TINY_NC(22) BATCHLAS_GETRF_TINY_NC(24) BATCHLAS_GETRF_TINY_NC(26)
            BATCHLAS_GETRF_TINY_NC(28) BATCHLAS_GETRF_TINY_NC(30) BATCHLAS_GETRF_TINY_NC(32)
#undef BATCHLAS_GETRF_TINY_NC
            default:
                break;
        }
    }
    return getrf_tiny_launch_nc<T, N, N>(ctx, a_ptr, ld, stride, n, batch, piv_ptr, info_ptr);
}

// An empty OR SHORT `info` span means "not requested". A may carry a null data_ptr().
Span<int32_t> getrf_tiny_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

// The ONE place the ceiling is spelled: a fork lets supports() promise what the launcher refuses.
template <typename T>
int getrf_tiny_max_n() {
    return tiny_cap<T>();
}

template <typename T>
std::size_t getrf_tiny_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& p) {
        return getrf_tiny_layout(ctx, p, batch);
    });
}

// Every supports() gate is re-applied here: a forced route that fails one falls through
// to the vendor and passes green regardless.
template <typename T>
Event getrf_tiny_dispatch(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          Span<int64_t> pivots,
                          Span<std::byte> workspace,
                          Span<int32_t> info_out) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("getrf_tiny: degenerate extents");
    }
    if (m != n) {
        throw batchlas::invalid_argument(
            "getrf_tiny: A must be square (route_getrf.hh's supports() refuses m != n)");
    }
    if (A.is_heterogeneous()) {
        // Not merely unsupported: the unrolled body's `if (j >= n) continue` skips a
        // collective, which is legal only because n is kernel-uniform.
        throw batchlas::invalid_argument("getrf_tiny: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("getrf_tiny: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports entry [0].
        throw batchlas::unsupported(
            "getrf_tiny: device does not offer sub-group size 32, which the kernel requires");
    }
    if (static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)) < kTinyWg) {
        throw batchlas::unsupported(
            "getrf_tiny: the tier launches a work-group of " + std::to_string(kTinyWg) +
            ", which exceeds this device's maximum");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("getrf_tiny: pivot span is shorter than n * batch");
    }

    const int bucket = tiny_bucket(n);
    if (bucket < 1 || bucket > tiny_cap<T>()) {
        throw batchlas::unsupported(
            "getrf_tiny: order " + std::to_string(n) +
            " is above this type's register-resident ceiling of " +
            std::to_string(tiny_cap<T>()));
    }

    BumpAllocator pool(workspace);
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : getrf_tiny_layout(ctx, pool, batch);

    // No zero pre-fill, so this body cannot serve as a blocked-driver panel leaf.

    auto piv_i32 = pivots.as_span<int>();

    switch (bucket) {
        case 4:
            return getrf_tiny_launch_mb<T, 4>(ctx, A.data_ptr(), A.ld(), A.stride(), n, batch,
                                           piv_i32.data(), info.data());
        case 8:
            return getrf_tiny_launch_mb<T, 8>(ctx, A.data_ptr(), A.ld(), A.stride(), n, batch,
                                           piv_i32.data(), info.data());
        case 16:
            return getrf_tiny_launch_mb<T, 16>(ctx, A.data_ptr(), A.ld(), A.stride(), n, batch,
                                            piv_i32.data(), info.data());
        case 32:
            if constexpr (tiny_cap<T>() >= 32) {
                return getrf_tiny_launch_mb<T, 32>(ctx, A.data_ptr(), A.ld(), A.stride(), n,
                                                batch, piv_i32.data(), info.data());
            }
            break;
        default:
            break;
    }
    // Unreachable: thrown rather than falling through to a leading-submatrix factorisation.
    throw batchlas::unsupported(
        "getrf_tiny: no instantiation for order " + std::to_string(n) +
        " (ceiling " + std::to_string(tiny_cap<T>()) + ")");
}

// Per scalar type only; the switch above pulls the <T, N> cross-product in implicitly.
#define BATCHLAS_GETRF_TINY_INSTANTIATE(T)                                                     \
    template int getrf_tiny_max_n<T>();                                                        \
    template std::size_t getrf_tiny_buffer_size<T>(Queue&,                                     \
                                                   const MatrixView<T, MatrixFormat::Dense>&); \
    template Event getrf_tiny_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,   \
                                          Span<int64_t>, Span<std::byte>, Span<int32_t>);

BATCHLAS_GETRF_TINY_INSTANTIATE(float)
BATCHLAS_GETRF_TINY_INSTANTIATE(double)
BATCHLAS_GETRF_TINY_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRF_TINY_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRF_TINY_INSTANTIATE

}  // namespace sycl_getrf
}  // namespace batchlas
