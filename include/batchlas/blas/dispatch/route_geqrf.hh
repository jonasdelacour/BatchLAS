#pragma once

// Routing table for geqrf's two native arms (CTA, blocked) and the vendor.
// Evidence: docs/perf/qr.md. supports() is correctness only: a speed cutoff
// here deletes the native arm from vendor-free builds (route_resolve.hh:60-63).

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GeqrfShape : OpShape {
    // Capacity is the AREA m*n -- the panel is resident in local memory -- and is
    // asked of the device, never of device_limits.hh (whose 49152 is 2.06x wrong).
    int cta_max_m = 0;
    int64_t cta_max_elems = 0;

    // Must describe the BUILD: a Blocked route that is not linked throws.
    bool blocked_available = false;

    // From sycl::info::device::sub_group_sizes, not OpShape::max_sub_group, which
    // is sub_group_sizes()[0] and so admits a device that aborts the sg32 launch.
    bool has_sg32 = false;

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    // min(rows, cols): the only spelling of k here, so a predicate cannot mean m.
    int64_t reflectors() const { return k; }
};

inline constexpr Route kGeqrfOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::geqrf, T> {
    static bool supports(Route r, const GeqrfShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything
        if (!is_native(r)) return false;

        // No squareness gate, deliberately -- rectangular A is the point of
        // geqrf. But on a wide view the trailing update runs off the panel.
        if (s.m < s.n) return false;

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        // One launch, one (m, n, ld, stride) tuple, no batch walker: per-item
        // active dims would factorise the wrong extents for every item but the first.
        if (s.heterogeneous_batch) return false;

        if (s.m < 1 || s.n < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                if (s.cta_max_m < 1 || s.cta_max_elems < 1) return false;
                return s.m <= static_cast<int64_t>(s.cta_max_m) &&
                       s.m * s.n <= s.cta_max_elems;

            case Algorithm::Blocked:
                // Inherits CTA's presence gate (its leaf IS that kernel), not its
                // capacity. No lower bound: one sends a forced `blocked` to the vendor.
                return s.blocked_available && s.cta_max_m >= 1 && s.cta_max_elems >= 1;

            default:
                return false;
        }
    }

    // All-false, deliberately: Auto still takes the vendor. Un-preferred is not
    // unroutable -- vendor-free builds still resolve to a supported native arm.
    static bool preferred(Route r, const GeqrfShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // Native-vs-native, and only on the vendor-free walk; both arms stay supported().
    static bool native_tier_preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return true;

        // Crossover in n, not m*n: the work-group comes from n alone. Square-only
        // sweep. evidence: docs/perf/qr.md#cta-vs-blocked-crossover
        const int64_t cta_max_cols = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>) {
                return 96;      // bracketed on both sides
            } else if constexpr (std::is_same_v<T, double>) {
                return 48;      // a tie, given to CTA: its workspace is zero
            } else {
                // No measured crossover for either complex type; the fit gate rules.
                return 1 << 30;
            }
        }();

        switch (r.algo) {
            case Algorithm::CTA:
                return s.cols() <= cta_max_cols;
            case Algorithm::Blocked:
                return s.cols() > cta_max_cols;
            default:
                return true;
        }
    }

    static constexpr const Route* order_begin() { return kGeqrfOrder; }
    static constexpr const Route* order_end() {
        return kGeqrfOrder + (sizeof(kGeqrfOrder) / sizeof(kGeqrfOrder[0]));
    }
};

// Pass `vendor_available` EXPLICITLY -- it is factorization_vendor_available<B>,
// not solver_vendor_available, and the `= true` default skips the vendor-free walk.
template <typename T>
inline Route resolve_geqrf_route(Route forced, const GeqrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::geqrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
