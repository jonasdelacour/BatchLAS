#pragma once

// Routing table for geqrf's two native arms (CTA, blocked) and the vendor.
// Evidence: docs/perf/qr.md. supports() is correctness only: a speed cutoff
// here deletes the native arm from vendor-free builds (route_resolve.hh).

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <complex>
#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GeqrfShape : OpShape {
    // Capacity is the AREA m*n and must come from the device, not device_limits.hh.
    int cta_max_m = 0;
    int64_t cta_max_elems = 0;

    // Must describe the BUILD: a Blocked route that is not linked throws.
    bool blocked_available = false;

    // From sycl::info::device::sub_group_sizes, not OpShape::max_sub_group, which
    // is sub_group_sizes()[0] and so admits a device that aborts the sg32 launch.
    bool has_sg32 = false;

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    // k is min(rows, cols).
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

    // Native above a per-type order floor. cuBLAS geqrfBatched is unblocked and
    // saturates at ~380 GFLOP/s (float) REGARDLESS of n, so the native arm pulls
    // away as n grows; below the floor the reverse holds, because the native
    // panel kernel is the whole cost at a size where there is no trailing work
    // to amortise it.
    //
    // The floors are the first order clearing the repository's flip gate,
    // t_native <= 0.90 t_vendor (ratio >= 1.11), at the top of the measured
    // batch ladder, with the order below it measured as a loss:
    //
    //   T         floor   ratio at floor   bracketing loss below
    //   float      64        1.71            48: 1.02   33: 0.76
    //   cfloat     48        1.74            33: 0.69   32: 0.62
    //   double     96        1.16            65: 0.66   64: 0.58
    //   cdouble   256        1.50           192: 1.06  129: 0.58
    //
    // float n=48 (1.02) and cdouble n=192 (1.06) are inside the gate's dead
    // band and are DELIBERATELY excluded: both are single cells that do not
    // clear 1.11, and a window edge without a clearing measurement is a guess.
    //
    // TALL PANELS CROSS OVER EARLIER, and they are the shape the callers
    // actually issue: sytrd_sy2sb.cc:509 and band_reduction.cc:603 factorise
    // an m x kd panel with kd typically 32, where every square cell loses. A
    // floor on cols() alone leaves that shape on the vendor while the
    // measurement says native is 1.6-3.8x there:
    //
    //   shape      aspect   float   cfloat   double   cdouble
    //   128 x 32      4x     2.23    3.79     0.68     0.68
    //   512 x 32     16x     2.68    3.39     1.58     2.16
    //   512 x 64      8x     3.17    3.70      -        -
    //   1024 x 128    8x     7.16    5.07     6.42      -
    //
    // Hence the second clause, on the panel's ASPECT RATIO, and the ratio is
    // per type. 128 x 32 is the whole reason: at 4x the 32-bit types win
    // 2.23-3.79 and the 64-bit types LOSE at 0.68, so a type-independent
    // aspect floor of 4 would route two measured losses native. The 64-bit
    // floor is 8x, which is bracketed on both sides -- 1024 x 128 (8x) wins
    // 6.42 for double, 128 x 32 (4x) loses 0.68 for both.
    //
    // The 32-bit floor of 4x is NOT bracketed below: no tall cell narrower
    // than 4x was measured for any type, so 4 is the smallest measured
    // aspect and not a demonstrated boundary. Neither is `rows() >= 128`,
    // for the same reason -- 128 x 32 is simply the shortest tall panel in
    // the grid. Both edges are recorded as debts in
    // docs/perf/small-n-baseline.md rather than dressed up as evidence.
    //
    // THE TIER HOOK MUST NOT BE PRE-EMPTED. resolve_route's automatic() walks
    // kGeqrfOrder testing `supports(r) && preferred(r)` and RETURNS on the
    // first hit, before native_tier_preferred is consulted at all. CTA leads
    // that order, so a window that answers true for CTA hands it every shape
    // it can hold, whatever the tier hook says. Measured cost of getting this
    // wrong, at the double floor: CTA 65.19 ms against blocked 47.70 and
    // vendor 55.27, i.e. the window shipped 0.848x -- a LOSS against the arm
    // it replaced -- while float n=128 took CTA's 2.12x instead of blocked's
    // 3.62x. tests/geqrf_tests.cc G9b exists for exactly this and caught it.
    // evidence: docs/perf/small-n-baseline.md#geqrf, docs/perf/qr.md:33
    static bool preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return false;

        const int64_t floor_n = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>)  return 64;
            if constexpr (std::is_same_v<T, double>) return 96;
            if constexpr (std::is_same_v<T, std::complex<float>>)  return 48;
            if constexpr (std::is_same_v<T, std::complex<double>>) return 256;
            return (1 << 30);
        }();

        // 4x for the 32-bit types, 8x for the 64-bit ones: see the table above.
        const int64_t tall_aspect = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>) return 4;
            if constexpr (std::is_same_v<T, std::complex<float>>) return 4;
            return 8;
        }();

        const bool in_window =
            s.cols() >= floor_n ||
            (s.rows() >= 128 && s.cols() >= 32 && s.rows() >= tall_aspect * s.cols());
        if (!in_window) return false;

        // Answer true for exactly ONE native tier -- the one the vendor-free
        // walk would land on -- so the window cannot pre-empt the tier hook.
        //
        // Not simply `native_tier_preferred(r, s)`: for the complex types that
        // hook returns a 1<<30 column cap, meaning "CTA wherever it FITS", so
        // its Blocked arm is false at every real order. Composing it directly
        // would answer false for both tiers at, say, cfloat 256x256 -- where
        // CTA cannot hold the tile and Blocked measures 7.51x -- and hand a
        // large measured win back to the vendor. best_native_tier resolves the
        // fit first and consults the hook only among tiers that can serve.
        const Route best = best_native_tier(s);
        return best.origin != Origin::Auto && r == best;
    }

    // The tier the native walk lands on: the first supported native route the
    // tier hook prefers, else the first supported one. Auto/Auto means none.
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

    static bool native_tier_preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return true;

        // Crossover in n, not m*n: the work-group comes from n alone.
        // evidence: docs/perf/qr.md#cta-vs-blocked-crossover
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

// Pass `vendor_available` explicitly: it is factorization_vendor_available<B>, and
// the `= true` default skips the vendor-free walk.
template <typename T>
inline Route resolve_geqrf_route(Route forced, const GeqrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::geqrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
