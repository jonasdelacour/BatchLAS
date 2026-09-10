#pragma once

// GETRS routing table (docs/perf/lu.md). {Native, CTA} is the fused narrow-RHS kernel
// (getrs_fused.cc): one work-group per matrix, permutation and both substitutions in a
// single launch. {Native, Blocked} is the composition (getrs_native.cc): laswp + 2 trsm.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GetrsShape : OpShape {
    bool blocked_available = false;

    // Enumerated from sub_group_sizes: the fused kernels carry
    // [[sycl::reqd_sub_group_size(32)]], so a {64}-only device cannot launch them.
    bool has_sg32 = false;

    // Bounds n*nrhs, not n: the fused kernel holds the whole RHS block in local memory.
    int64_t fused_max_elems = 0;

    // The widest nrhs the fused kernel is instantiated for: a build fact, not a device one.
    int64_t fused_max_nrhs = 0;

    int64_t order() const { return m; }
    int64_t nrhs() const { return n; }
};

inline constexpr Route kGetrsOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getrs, T> {
    // Correctness only: a forced route bypasses preferred() but never supports(), so a
    // speed gate here would send a pinned `native:cta` to the vendor and pass green.
    static bool supports(Route r, const GetrsShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        if (r.algo == Algorithm::CTA) {
            if (s.fused_max_elems <= 0 || s.fused_max_nrhs <= 0) return false;
        } else if (r.algo == Algorithm::Blocked) {
            if (!s.blocked_available) return false;
        }

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        if (s.heterogeneous_batch) return false;

        if (s.order() < 1 || s.nrhs() < 1 || s.batch < 1) return false;

        // Pivot format: the GPU arms pack 1-based int32 into the low half of the int64
        // span, netlib writes true int64; mixing them is silently wrong (info stays 0).
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                if (s.order() * s.nrhs() > s.fused_max_elems) return false;
                if (s.nrhs() > s.fused_max_nrhs) return false;
                return true;
            case Algorithm::Blocked:
                return true;
            default:
                return false;
        }
    }

    // The measured window, native vs vendor: CTA at order >= 32 with nrhs <= 2 (all
    // types) and <= 4 (float); the composition at batch >= 128 with nrhs >= 64 (float)
    // / >= 128 (double), and NO order floor -- clause C's axis is nrhs and its own
    // batch floor, and it was never measured below order 32 in the first place.
    // The CTA order floor is documented against its grid at the clause itself below.
    // evidence: docs/perf/lu.md#getrs-fused-window-evidence, #getrs-composition-window-evidence,
    //           docs/perf/small-n-baseline.md#getrs
    static bool preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return false;

        if (r.algo == Algorithm::Blocked) {
            // Conservative: at nrhs = 128 the composition still wins at batch 32-64.
            if (s.batch < 128) return false;
            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;
            if constexpr (std::is_same_v<T, double>) return s.nrhs() >= 128;
            return false;   // cfloat and cdouble earn nothing at any width
        }

        if (r.algo != Algorithm::CTA) return false;

        // ORDER FLOOR, and it is a fix rather than a tuning knob.
        //
        // Clauses A and B shipped with no order bound at all, on a grid whose
        // smallest order was 32. Re-measured down to n = 4 they lose in live
        // routed traffic at every type, worst 0.234 (cdouble n=4, nrhs=1,
        // batch 32768), and the loss deepens with batch rather than washing
        // out: float n=4 nrhs=1 reads 1.50 / 1.07 / 0.71 / 0.52 at batch
        // 8192 / 16384 / 32768 / 65536.
        //
        // The mechanism is that the fused kernel gives one work-group to a
        // matrix whose whole solve is a few dozen flops, so the work-group is
        // the cost; cuBLAS getrsBatched keeps its per-item work in one kernel
        // and pays no such floor.
        //
        // Worst ratio over the batch ladder, nrhs = 1, by order:
        //
        //   T         n=4    n=8    n=16   n=17   n=24   n=32   n=48
        //   float     0.71   1.12   1.04   0.76   0.95   2.29   1.94
        //   cfloat    0.59   0.69   3.76   1.38   1.27   1.40   1.60
        //   double    0.52   0.98   0.99   0.87   3.66   3.77   3.74
        //   cdouble   0.23   0.41   2.00   2.02   1.93   2.13   2.46
        //
        // 32 is the first order where all four types clear the flip gate AND
        // stay clear above it; the band below is non-monotone (float passes at
        // 8, fails at 9, 16, 17 and 24), so no lower floor is defensible.
        // Clause B is measured on the same grid and takes the same floor:
        // float nrhs=4 reads 0.45 / 0.46 at n = 4 / 8 and 1.07 / 1.10 at
        // 17 / 24, clearing only from 32 (1.30).
        //
        // P2 of the small-n plan is what reclaims this band: a fused
        // factor-and-solve kernel that holds the matrix in registers has no
        // work-group floor to pay.
        // evidence: docs/perf/small-n-baseline.md#getrs
        if (s.order() < 32) return false;

        if (s.nrhs() <= 2) return true;                  // clause A

        if constexpr (std::is_same_v<T, float>) {        // clause B
            if (s.nrhs() <= 4) return true;
        }
        return false;
    }

    // Native-vs-native tie-break for the vendor-free walk: CTA leads the composition
    // everywhere inside supports(); raising kGetrsFusedMaxRhs needs a window here.
    static bool native_tier_preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return true;
        static_cast<void>(s);
        switch (r.algo) {
            case Algorithm::CTA:     return true;
            case Algorithm::Blocked: return false;
            default:                 return true;
        }
    }

    static constexpr const Route* order_begin() { return kGetrsOrder; }
    static constexpr const Route* order_end() {
        return kGetrsOrder + (sizeof(kGetrsOrder) / sizeof(kGetrsOrder[0]));
    }
};

template <typename T>
inline Route resolve_getrs_route(Route forced, const GetrsShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getrs, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
