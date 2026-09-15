#pragma once

// POTRF's routing table. evidence: docs/perf/potrf.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <type_traits>

namespace batchlas::dispatch {

struct PotrfShape : OpShape {
    int cta_max_n = 0;    // device-queried local-memory ceiling; 0 = tier absent from this build
    int tiny_max_n = 0;   // compile-time ceiling (the tier owns no local memory); 0 = absent
    int lpanel_max_n = 0;  // SLM slice AND MAX_WORK_GROUP_SIZE (one work-item per row)

    bool blocked_available = false;

    // MUST come from sycl::info::device::sub_group_sizes: OpShape::max_sub_group reports
    // entry [0], not the max, so it admits a device that rejects the sg32 launch.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

inline constexpr Route kPotrfOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::LPanel},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::potrf, T> {
    // Correctness only: a speed threshold here removes potrf's vendor-free route.
    static bool supports(Route r, const PotrfShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        if (s.m != s.n) return false;

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        if (s.heterogeneous_batch) return false;

        if (s.order() < 1 || s.batch < 1) return false;

        // No uplo gate on Tiny/CTA: Upper is the same recurrence on S(i,c) = conj(A(c,i)).
        switch (r.algo) {
            case Algorithm::Tiny:
                if (s.tiny_max_n < 1) return false;
                return s.order() <= s.tiny_max_n;

            case Algorithm::CTA:
                if (s.cta_max_n < 1) return false;
                return s.order() <= s.cta_max_n;

            case Algorithm::LPanel:
                // uplo IS correctness, as on Blocked: the update reads the LOWER triangle.
                if (s.uplo != Uplo::Lower) return false;
                if (s.lpanel_max_n < 1) return false;
                return s.order() <= s.lpanel_max_n;

            case Algorithm::Blocked:
                // uplo IS correctness: the driver is Lower-only and handed Upper overwrites
                // the caller's triangle. No order floor, or a forced `blocked` falls through.
                if (s.uplo != Uplo::Lower) return false;
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                return false;
        }
    }

    // Two windows: the register tier at n <= 32, LPanel at 32 < n <= 256 (Lower, fp32 only).
    // evidence: docs/perf/potrf.md#the-measured-lpanel-window
    static bool preferred(Route r, const PotrfShape& s) {
        if (!is_native(r)) return false;

        // The only window measured on Upper or for fp64, so it precedes both gates below.
        if (tiny_window(s)) return r.algo == Algorithm::Tiny;

        if (s.uplo != Uplo::Lower || !lpanel_types()) return false;
        if (s.order() <= 32 || s.order() > 256) return false;

        // R8b: exactly ONE tier may answer true, because automatic() returns on the first
        // supports && preferred hit and never consults the tier hook.
        const Route best = best_native_tier(s);
        if (best.origin == Origin::Auto) return false;

        // On a device that cannot HOLD the tier the grid names, the walk lands on one
        // measured LOSING to the vendor, so the window must not fire at all.
        const Algorithm measured = (s.order() <= cta_last_order()) ? Algorithm::CTA
                                                                   : Algorithm::LPanel;
        if (best.algo != measured) return false;
        return r == best;
    }

    // evidence: docs/perf/potrf.md#the-tiny-potrf-window-extended-to-upper-and-to-fp64
    static bool tiny_window(const PotrfShape& s) {
        if (s.tiny_max_n < 1) return false;
        const int64_t n = s.order();
        if (n < 1 || n > static_cast<int64_t>(s.tiny_max_n) || n > 32) return false;
        if (s.uplo == Uplo::Upper) return true;       // whole tier, every type: 1.42-28.73x
        if constexpr (lpanel_types()) return true;    // Lower, float/cfloat: the whole tier
        else return (n >= 2 && n <= 8) || (n >= 12 && n <= 16);   // Lower fp64: fill splits it
    }

    // The tier the walk lands on; Auto/Auto means none. Resolves FIT before the hook.
    static Route best_native_tier(const PotrfShape& s) {
        Route first_supported{};
        bool found = false;
        for (const Route* it = order_begin(); it != order_end(); ++it) {
            if (!is_native(*it) || !supports(*it, s)) continue;
            if (!found) { first_supported = *it; found = true; }
            if (native_tier_preferred(*it, s)) return *it;
        }
        return found ? first_supported : Route{};
    }

    // Native-vs-native tie-break. evidence: docs/perf/potrf.md#native_tier_preferred
    static bool native_tier_preferred(Route r, const PotrfShape& s) {
        if (!is_native(r)) return true;

        // Enumerate EVERY tier: `default:` answers true, so an omitted arm takes every shape.
        // evidence: docs/perf/potrf.md#every-tier-is-enumerated-explicitly
        const bool cta_holds = (s.cta_max_n >= 1) && (s.order() <= s.cta_max_n);
        // LPanel takes a shape only where it was MEASURED: double and cdouble have no grid.
        const bool lpanel_holds = lpanel_types() && (s.uplo == Uplo::Lower) &&
                                  (s.lpanel_max_n >= 1) && (s.order() <= s.lpanel_max_n) &&
                                  (s.order() > cta_last_order());
        switch (r.algo) {
            case Algorithm::Tiny:    return tiny_window(s);
            case Algorithm::CTA:     return !tiny_window(s) && cta_holds && !lpanel_holds;
            case Algorithm::LPanel:  return lpanel_holds;
            case Algorithm::Blocked: return !cta_holds && !lpanel_holds;
            default:                 return true;
        }
    }

    // Measured types and the measured CTA/LPanel boundary inside them.
    // evidence: docs/perf/potrf.md#the-measured-lpanel-window
    static constexpr bool lpanel_types() {
        return std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
    }
    static constexpr int64_t cta_last_order() {
        return std::is_same_v<T, float> ? 35 : 32;
    }

    static constexpr const Route* order_begin() { return kPotrfOrder; }
    static constexpr const Route* order_end() {
        return kPotrfOrder + (sizeof(kPotrfOrder) / sizeof(kPotrfOrder[0]));
    }
};

// vendor_available is solver_vendor_available<B>, NOT the factorization one: differ on CUDA.
template <typename T>
inline Route resolve_potrf_route(Route forced, const PotrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::potrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
