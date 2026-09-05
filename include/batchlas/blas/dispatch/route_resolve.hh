#pragma once

// The one route resolver, shared by every op's RouteTable; it reads only its
// arguments. See docs/design/vendor-independence.md#the-resolver

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

template <Op O, typename T>
struct RouteTable;

// Optional third predicate for ops with several native routes: among the native
// routes that can serve s, is r the better one? A speed threshold in supports()
// would break forcing, and in preferred() would move vendor-present traffic too;
// an absent hook means `true`. evidence: docs/perf/qr.md#the-third-predicate
template <typename Table, typename Shape>
inline bool native_tier_preferred_or_default(Route r, const Shape& s) {
    if constexpr (requires { Table::native_tier_preferred(r, s); }) {
        return Table::native_tier_preferred(r, s);
    } else {
        return true;
    }
}

// `Shape` is deduced: an op routing on more than OpShape passes a derived struct.
template <Op O, typename T, typename Shape>
inline Route resolve_route_uninstrumented(Route forced, const Shape& s,
                                          bool vendor_available = true) {
    using Table = RouteTable<O, T>;

    // Preference first; a merely supported native route only when no vendor is
    // left. Vendor when nothing serves the shape, so the caller can diagnose.
    auto automatic = [&]() -> Route {
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (Table::supports(*r, s) && Table::preferred(*r, s)) return *r;
        }
        if (!vendor_available) {
            // Two passes: the tie-break, then the plain walk a table without the hook needs.
            for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
                if (is_native(*r) && Table::supports(*r, s) &&
                    native_tier_preferred_or_default<Table>(*r, s)) {
                    return *r;
                }
            }
            for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
                if (is_native(*r) && Table::supports(*r, s)) return *r;
            }
        }
        return Route{Origin::Vendor, Algorithm::Auto};
    };

    if (forced.origin == Origin::Auto) {
        return automatic();
    }

    // A requested vendor still has to exist: GEMM's unset default IS Vendor, so
    // an ordinary call arrives here rather than at `automatic` above.
    if (is_vendor(forced)) {
        return vendor_available ? forced : automatic();
    }

    // A bare origin must resolve to a concrete route: {Native, Auto} maps to no kernel.
    if (forced.algo == Algorithm::Auto) {
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (r->origin == forced.origin && Table::supports(*r, s) && Table::preferred(*r, s)) {
                return *r;
            }
        }
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (r->origin == forced.origin && Table::supports(*r, s)) return *r;
        }
        return automatic();
    }

    if (Table::supports(forced, s)) return forced;

    // Forced but unsupported falls back to automatic(), not to the vendor: a
    // forced `cta` too big for the CTA path must still reach the blocked one.
    return automatic();
}

// The instrumented entry point, and the ONLY one ops should call; `s` is sliced
// to OpShape on purpose. evidence: docs/design/vendor-independence.md#the-coverage-instrument
template <Op O, typename T, typename Shape>
inline Route resolve_route(Route forced, const Shape& s, bool vendor_available = true) {
    const Route chosen = resolve_route_uninstrumented<O, T, Shape>(forced, s, vendor_available);

    if (coverage::dynamic_enabled()) {
        using Table = RouteTable<O, T>;
        bool native_existed = false;
        bool native_supported = false;
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (!is_native(*r)) continue;
            native_existed = true;
            if (Table::supports(*r, s)) native_supported = true;
        }
        coverage::record_if_enabled(s.op, s.scalar, s.backend, static_cast<const OpShape&>(s),
                                    chosen, native_existed, native_supported);
    }

    return chosen;
}

} // namespace batchlas::dispatch
