#pragma once

// POTRF's routing table. evidence: docs/perf/potrf.md

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct PotrfShape : OpShape {
    int cta_max_n = 0;    // device-queried local-memory ceiling; 0 = tier absent from this build
    int tiny_max_n = 0;   // compile-time ceiling (the tier owns no local memory); 0 = absent

    bool blocked_available = false;

    // MUST come from sycl::info::device::sub_group_sizes: OpShape::max_sub_group reports
    // entry [0], not the max, so it admits a device that rejects the sg32 launch.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

inline constexpr Route kPotrfOrder[] = {
    {Origin::Native, Algorithm::Tiny},
    {Origin::Native, Algorithm::CTA},
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

            case Algorithm::Blocked:
                // uplo IS correctness: the driver is Lower-only and handed Upper overwrites
                // the caller's triangle. No order floor, or a forced `blocked` falls through.
                if (s.uplo != Uplo::Lower) return false;
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                return false;
        }
    }

    // Empty window: Auto takes the vendor everywhere, a vendor-free build still resolves.
    // evidence: docs/perf/potrf.md#preferred-is-false-everywhere
    static bool preferred(Route r, const PotrfShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // Native-vs-native tie-break, consulted only in the vendor-free walk.
    // evidence: docs/perf/potrf.md#native_tier_preferred
    static bool native_tier_preferred(Route r, const PotrfShape& s) {
        if (!is_native(r)) return true;

        // Enumerate EVERY tier: `default:` answers true, so an omitted arm takes every shape.
        // evidence: docs/perf/potrf.md#every-tier-is-enumerated-explicitly
        const bool cta_holds = (s.cta_max_n >= 1) && (s.order() <= s.cta_max_n);
        switch (r.algo) {
            case Algorithm::Tiny:    return false;
            case Algorithm::CTA:     return cta_holds;
            case Algorithm::Blocked: return !cta_holds;
            default:                 return true;
        }
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
