#pragma once

// GEQRF's routing table. evidence: docs/perf/qr.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GeqrfShape : OpShape {
    // CTA capacity is the AREA m*n and must come from the device, not device_limits.hh.
    int cta_max_m = 0;
    int64_t cta_max_elems = 0;

    int tiny_max_n = 0;   // an ORDER, not an area: the tier owns no local memory; 0 = absent

    // Must describe the BUILD: a Blocked route that is not linked throws.
    bool blocked_available = false;

    // MUST come from sycl::info::device::sub_group_sizes: OpShape::max_sub_group reports
    // entry [0], not the max, so it admits a device that rejects the sg32 launch.
    bool has_sg32 = false;

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    int64_t reflectors() const { return k; }   // k is min(rows, cols)
};

// Walk order is this array, never Algorithm's numeric value; Tiny first, the narrowest tier.
inline constexpr Route kGeqrfOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::geqrf, T> {
    // Correctness only: a speed cutoff here deletes the native arm from vendor-free builds.
    static bool supports(Route r, const GeqrfShape& s) {
        if (is_vendor(r)) return true;
        if (!is_native(r)) return false;

        // Only m < n is rejected: on a wide view the trailing update runs off the panel.
        if (s.m < s.n) return false;

        if (!s.is_gpu) return false;

        if (!s.has_sg32) return false;

        // One launch, one (m, n, ld, stride) tuple: per-item dims break all but item 0.
        if (s.heterogeneous_batch) return false;

        if (s.m < 1 || s.n < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Tiny:
                // Square only: the register array IS the matrix; a tall panel is CTA's.
                if (s.tiny_max_n < 1) return false;
                return s.m == s.n && s.n <= static_cast<int64_t>(s.tiny_max_n);

            case Algorithm::CTA:
                if (s.cta_max_m < 1 || s.cta_max_elems < 1) return false;
                return s.m <= static_cast<int64_t>(s.cta_max_m) &&
                       s.m * s.n <= s.cta_max_elems;

            case Algorithm::Blocked:
                // Inherits CTA's presence gate (its leaf IS that kernel), not its capacity.
                return s.blocked_available && s.cta_max_m >= 1 && s.cta_max_elems >= 1;

            default:
                return false;
        }
    }

    // A per-type order floor plus a tall-panel clause. Both constants below are window
    // EDGES, not knobs. evidence: docs/perf/qr.md#the-geqrf-order-floor-and-the-tall-panel-clause
    static bool preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return false;

        const int64_t floor_n = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>)  return 64;
            if constexpr (std::is_same_v<T, double>) return 96;
            if constexpr (std::is_same_v<T, std::complex<float>>)  return 48;
            if constexpr (std::is_same_v<T, std::complex<double>>) return 256;
            return (1 << 30);
        }();

        const int64_t tall_aspect = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>) return 4;
            if constexpr (std::is_same_v<T, std::complex<float>>) return 4;
            return 8;
        }();

        const bool in_window =
            s.cols() >= floor_n ||
            (s.rows() >= 128 && s.cols() >= 32 && s.rows() >= tall_aspect * s.cols());
        if (!in_window) return false;

        // Exactly ONE native tier may answer true: automatic() returns on the first
        // supports && preferred hit, before native_tier_preferred is consulted at all.
        // evidence: docs/perf/qr.md#why-the-window-answers-for-exactly-one-tier
        const Route best = best_native_tier(s);
        return best.origin != Origin::Auto && r == best;
    }

    // The tier the native walk lands on; Auto/Auto means none. Resolves FIT before the hook.
    static Route best_native_tier(const GeqrfShape& s) {
        Route first_supported{};
        bool found = false;
        for (const Route* it = order_begin(); it != order_end(); ++it) {
            if (!is_native(*it) || !supports(*it, s)) continue;
            if (!found) { first_supported = *it; found = true; }
            if (native_tier_preferred(*it, s)) return *it;
        }
        return found ? first_supported : Route{};
    }

    // Crossover in n, not m*n: the work-group comes from n alone.
    // evidence: docs/perf/qr.md#cta-vs-blocked-crossover
    static bool native_tier_preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return true;

        const int64_t cta_max_cols = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>) {
                return 96;
            } else if constexpr (std::is_same_v<T, double>) {
                return 48;
            } else {
                // No measured crossover for complex; the fit gate rules.
                return 1 << 30;
            }
        }();

        switch (r.algo) {
            // EXPLICIT: `default:` returns TRUE and Tiny leads the order array. Flips with
            // preferred(). evidence: docs/perf/qr.md#why-the-tiny-arm-is-spelled-out
            case Algorithm::Tiny:
                return false;
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

// Pass vendor_available (factorization_vendor_available<B>): the default skips the walk.
template <typename T>
inline Route resolve_geqrf_route(Route forced, const GeqrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::geqrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
