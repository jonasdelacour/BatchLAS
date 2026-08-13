#pragma once

// One resolver, shared by every op's RouteTable.
//
// This body was written for GEMM (route_gemm.hh) and is factored out here
// unchanged, because the three questions it answers are not GEMM-specific:
//
//   * a forced route bypasses `preferred` -- that is what forcing is for --
//     but never bypasses `supports`, because that hands back a wrong answer
//     rather than a slow one;
//   * a requested VENDOR still has to exist;
//   * "supported but not preferred" is reached only when there is nothing
//     better left, which is the vendor-off configuration this work package is
//     building toward.
//
// Each op supplies a RouteTable<Op, T> with `supports`, `preferred`, and an
// order. Everything here reads only its arguments -- no getenv, no SYCL query
// -- which is what makes an op and its *_buffer_size query reach the same route
// by construction rather than by a comment asking them to.

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

// Declared here so every per-op header specialises the same template.
template <Op O, typename T>
struct RouteTable;

template <Op O, typename T>
inline Route resolve_route(Route forced, const OpShape& s, bool vendor_available = true) {
    using Table = RouteTable<O, T>;

    // Where everything that cannot get what it asked for ends up.
    //
    // The obvious body -- "take the first merely SUPPORTED route" -- is wrong,
    // and the GEMM equivalence test catches it: the orders list the native
    // routes first, so a tiny problem far outside every measured window would
    // select a native kernel where today it goes to the vendor, silently
    // inverting the default.
    //
    // Returning Vendor when nothing at all can serve the shape is deliberate:
    // it is the honest "this needs a vendor and there isn't one" signal, and
    // the caller turns that into a diagnostic rather than a wrong answer.
    auto fallback = [&]() -> Route {
        if (!vendor_available) {
            for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
                if (is_native(*r) && Table::supports(*r, s)) return *r;
            }
        }
        return Route{Origin::Vendor, Algorithm::Auto};
    };

    if (forced.origin != Origin::Auto) {
        // A REQUESTED VENDOR STILL HAS TO EXIST. Not hypothetical: GEMM's unset
        // default IS Vendor (legacy_unset_default), so an ordinary call with
        // nothing set arrives here rather than at the preference walk below.
        if (is_vendor(forced)) {
            return vendor_available ? forced : fallback();
        }
        return Table::supports(forced, s) ? forced : fallback();
    }

    for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
        if (Table::supports(*r, s) && Table::preferred(*r, s)) return *r;
    }
    return fallback();
}

} // namespace batchlas::dispatch
