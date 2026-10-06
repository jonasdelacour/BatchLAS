#pragma once

/// @file
/// @brief The one route resolver, shared by every op's RouteTable.
///
/// The resolver and every table read only their arguments (no `getenv`, no SYCL
/// query, no operand data), which is what makes an op and its `*_buffer_size`
/// query reach the same route by construction.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#the-resolver

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

/// @brief Per-op routing table; each op specialises it in `route_<op>.hh`.
///
/// A specialisation provides, as static members:
/// - `order_begin()` / `order_end()`: the walk order, a constexpr array of Route.
///   The order, not Algorithm's numeric value, decides ties.
/// - `bool supports(Route r, const Shape& s)`: **correctness only.** False means
///   `r` would compute a wrong answer, or index out of bounds, for `s`. A forced
///   route bypasses preferred() but never supports(). Never put a speed
///   threshold here: it makes a pinned route silently fall through to the
///   automatic choice, and in a vendor-free build it leaves a working shape with
///   no supported route at all.
/// - `bool preferred(Route r, const Shape& s)`: **a measured speed window,
///   vendor included.** False means merely slower; the route stays eligible.
///   Consulted on every automatic walk, in every build, so it must not be used
///   to choose between two native tiers. Where several native tiers exist,
///   exactly one may answer true for a given shape, because the walk returns on
///   the first `supports && preferred` hit.
/// - optionally `bool native_tier_preferred(Route r, const Shape& s)`: **native
///   against native**, consulted only on the vendor-free walk and on a bare
///   `native` pin. Absent means `true`.
///
/// Every table must be pure. Its window and evidence are on the op's
/// `docs/perf/` page, named in each specialisation's brief.
/// @see resolve_route(), @ref md_docs_2design_2vendor-independence
template <Op O, typename T>
struct RouteTable;

/// @brief The table's `native_tier_preferred(r, s)` if it declares one, else `true`.
///
/// Defaults to `true`, not `false`, so a table that has not thought about the
/// question keeps its old answer: its two vendor-free passes become identical.
// evidence: docs/perf/qr.md#the-third-predicate
template <typename Table, typename Shape>
inline bool native_tier_preferred_or_default(Route r, const Shape& s) {
    if constexpr (requires { Table::native_tier_preferred(r, s); }) {
        return Table::native_tier_preferred(r, s);
    } else {
        return true;
    }
}

/// @brief Pure route resolution, without the coverage record; call resolve_route() instead.
///
/// Rules, in order:
/// 1. `forced.origin == Auto`: the automatic walk. The first route in the table's
///    order that is both supported and preferred wins. Only when
///    `vendor_available` is false does the walk then accept a merely supported
///    native route: first one that native_tier_preferred() also accepts, then any.
///    Falling through returns `{Vendor, Auto}`, the "nothing serves this" answer
///    the caller turns into a diagnostic.
/// 2. A forced vendor route is returned when `vendor_available` and supports()
///    hold, else the automatic walk.
/// 3. A forced bare origin (`{Native, Auto}`) resolves to that origin's routes in
///    order: supported and preferred, then supported and native_tier_preferred,
///    then supported; else the automatic walk.
/// 4. Any other forced route is returned when supported, else the automatic walk,
///    never the vendor directly.
/// @tparam Shape  deduced: OpShape, or the op's derived shape
/// @param forced            the request from the environment or options; default = no opinion
/// @param s                 the call's shape
/// @param vendor_available  the `*_vendor_available<B>` of the op's library group
/// @return the chosen route; `{Vendor, Auto}` may mean "no route" when the vendor is absent
/// @trap Rule 4 is silent: a pin the shape cannot take resolves to the automatic
///       choice, which in a vendor-present build is usually the vendor. Confirm a pin
///       from the resolved route, never from the exit status.
template <Op O, typename T, typename Shape>
inline Route resolve_route_uninstrumented(Route forced, const Shape& s,
                                          bool vendor_available = true) {
    using Table = RouteTable<O, T>;

    // Returns Vendor when nothing serves the shape, so the caller can diagnose it.
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

    // A forced vendor must exist, and must pass supports() like any forced route:
    // is_vendor() also admits {Vendor, FusedDevice}, and a table may one day gate
    // its vendor arm on a real capability.
    if (is_vendor(forced)) {
        return (vendor_available && Table::supports(forced, s)) ? forced : automatic();
    }

    // A bare origin must resolve to a concrete route: {Native, Auto} maps to no kernel.
    if (forced.algo == Algorithm::Auto) {
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (r->origin == forced.origin && Table::supports(*r, s) && Table::preferred(*r, s)) {
                return *r;
            }
        }
        // The tie-break belongs here too: without it a bare `native` pin on a
        // vendor-present box lands on a different tier than the vendor-free build
        // takes wherever preferred() is false, and measures a route that never ships.
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (r->origin == forced.origin && Table::supports(*r, s) &&
                native_tier_preferred_or_default<Table>(*r, s)) {
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

/// @brief Resolves the route for one call of `O` on scalar `T`; the only resolver ops should call.
///
/// Same result as resolve_route_uninstrumented(). When coverage is on
/// (`$BATCHLAS_COVERAGE_OUT` set) it also records one `reached` row, with `s`
/// sliced to OpShape on purpose.
/// @param forced            the request; a default Route means "no opinion"
/// @param s                 the call's shape (OpShape or the op's derived shape)
/// @param vendor_available  whether the op's vendor library is compiled in for this backend
/// @return the chosen route
/// @see RouteTable, coverage::record_if_enabled()
// evidence: docs/design/vendor-independence.md#vendor-independence-the-coverage-instrument
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
