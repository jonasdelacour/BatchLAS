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
template <typename T, int N> class GetrfTinyKernel;

namespace {

namespace gn = ::batchlas::getrf_native;
namespace tn = ::batchlas::tiny_native;
namespace sd = ::batchlas::sycl_device;

// The shared tiny-tier work-group. A wider group makes the launch tail COARSER, not finer.
// evidence: docs/perf/lu.md#the-work-group-ab
constexpr int kTinyWg = tn::kTinyWgSize;

// A launch ABORT, not a slowdown, so it is encoded to fail at COMPILE time.
// evidence: docs/perf/lu.md#the-register-probe-at-wg--64
constexpr int kWorstRegsPerThread = 143;
static_assert(kTinyWg * kWorstRegsPerThread <= 65536,
              "re-run scripts/register_probe.sh out.log '' batchlas_extensions_cta "
              "before raising the tiny tier's work-group size");

// 0 means "above the tier" -- `unsupported`, never a silent leading-submatrix factorisation.
constexpr int tiny_bucket(int n) {
    if (n < 1) return 0;
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

template <typename T, int N>
Event getrf_tiny_launch(Queue& ctx, T* a_ptr, int ld, int stride, int n, int batch,
                        int* piv_ptr, int32_t* info_ptr) {
    // Re-typed HERE: std::complex's Annex-G operator* costs an isnan branch and a libcall.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(N == 8 || N == 16 || N == 32, "the tiny ladder is {8, 16, 32}");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    // Via the shared helper, not kTinyWg / N: the guarantees stay where they are owned.
    constexpr int kMpw = resident::pack_matrices_per_wg(
        /*bytes_per_matrix=*/1u, N, /*wg_slm_budget_bytes=*/~std::size_t(0),
        kTinyWg, kTinyWg, /*max_pack=*/kTinyWg / N);
    static_assert(kMpw * N == kTinyWg, "the tiny launch must fill its work-group exactly");
    const int num_wg = (batch + kMpw - 1) / kMpw;
    const std::ptrdiff_t ldp = static_cast<std::ptrdiff_t>(ld);
    const std::ptrdiff_t strp = static_cast<std::ptrdiff_t>(stride);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<GetrfTinyKernel<T, N>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(kTinyWg)),
                              sycl::range<1>(static_cast<std::size_t>(kTinyWg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<N>(sg);
                const int wg_id = static_cast<int>(it.get_group_linear_id());
                const int lane = static_cast<int>(part.get_local_linear_id());
                const int prob_id = wg_id * kMpw + tn::tiny_partition_id(sg, part);

                // CLAMP, DO NOT RETURN (steqr_cta.cc:88 does the opposite): tiny_device.hh's
                // third invariant -- an early-exited lane still sits in the shuffle mask.
                const bool live = (prob_id < batch);
                const int b = live ? prob_id : 0;
                const D* const src = ap + static_cast<std::ptrdiff_t>(b) * strp;

                D rA[N];  // top level, never a parameter: tiny_device.hh invariant 1
#pragma unroll
                for (int c = 0; c < N; ++c) {
                    rA[c] = tn::tiny_load_pad_identity<D>(src, lane, c, n, ldp, live);
                }

                int rowid = lane;    // WHICH matrix row this lane currently owns
                int my_piv = lane;   // lane j accumulates ipiv[j]
                int32_t linfo = 0;   // partition-uniform: from the broadcast pivot

#pragma unroll
                for (int j = 0; j < N; ++j) {
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

                    const D piv = tn::tiny_bcast<D>(part, rA[j], pl);  // --- 3. ?GETF2
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

                    // --- 4. rank-1 update, collective OUTSIDE the guard. A zero pivot was
                    // the argmax, so every multiplier is zero: LAPACK's ?GER as a no-op.
#pragma unroll
                    for (int k = j + 1; k < N; ++k) {
                        // `k = j + 1` is a COMPILE-TIME lower bound once j is unrolled.
                        // evidence: docs/perf/lu.md#why-the-rank-1-update-starts-at-k--j--1
                        if (k >= n) continue;      // kernel-uniform, as above
                        const D u = tn::tiny_bcast<D>(part, rA[k], pl);
                        if (act) rA[k] = sd::dev_sub(rA[k], sd::dev_mul(rA[j], u));
                    }
                }

                if (live) {
                    D* const dst = ap + static_cast<std::ptrdiff_t>(b) * strp;
#pragma unroll
                    for (int k = 0; k < N; ++k) {
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
            });
    });
    return ctx.get_event();
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
        case 8:
            return getrf_tiny_launch<T, 8>(ctx, A.data_ptr(), A.ld(), A.stride(), n, batch,
                                           piv_i32.data(), info.data());
        case 16:
            return getrf_tiny_launch<T, 16>(ctx, A.data_ptr(), A.ld(), A.stride(), n, batch,
                                            piv_i32.data(), info.data());
        case 32:
            if constexpr (tiny_cap<T>() >= 32) {
                return getrf_tiny_launch<T, 32>(ctx, A.data_ptr(), A.ld(), A.stride(), n,
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
