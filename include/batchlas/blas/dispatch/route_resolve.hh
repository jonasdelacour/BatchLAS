#pragma once

// One resolver, shared by every op's RouteTable.
//
// The body was written for GEMM (route_gemm.hh) and factored out here, because
// the questions it answers are not GEMM-specific:
//
//   * a forced route bypasses `preferred` -- that is what forcing is for --
//     but never bypasses `supports`, because that hands back a wrong answer
//     rather than a slow one;
//   * a forced route that cannot serve the shape falls back to the ORDINARY
//     automatic choice, not straight to the vendor;
//   * a requested VENDOR still has to exist;
//   * "supported but not preferred" is reached only when there is nothing
//     better left, which is the vendor-off configuration this work package is
//     building toward.
//
// Each op supplies a RouteTable<Op, T> with `supports`, `preferred`, and an
// order. Everything here reads only its arguments -- no getenv, no SYCL query
// -- which is what makes an op and its *_buffer_size query reach the same route
// by construction rather than by a comment asking them to.

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch {

// Declared here so every per-op header specialises the same template.
template <Op O, typename T>
struct RouteTable;

// ---------------------------------------------------------------------------
// THE NATIVE-VS-NATIVE TIE-BREAK, AND WHY IT IS A THIRD PREDICATE RATHER THAN A
// REUSE OF EITHER OF THE OTHER TWO.
//
// An op with more than one native route -- geqrf and potrf both have {CTA,
// Blocked} -- has a question the existing pair cannot express:
//
//   * supports() cannot answer it. It is CORRECTNESS ONLY. Putting "n is past
//     the CTA crossover so CTA should be false" there is the exact defect
//     route_potrf.hh:284-296 and route_geqrf.hh:300-310 both warn against: a
//     forced route bypasses preferred() but NEVER supports() (:101 here), so a
//     speed threshold in supports() makes a pinned route fall through to
//     automatic() and the test that pinned it measures something else entirely.
//
//   * preferred() cannot answer it EITHER, and this is the part that is easy to
//     get wrong. preferred() is consulted by the loop above the vendor-free
//     walk, which runs REGARDLESS of vendor_available. So a window written to
//     fix the vendor-free tier choice necessarily also moves vendor-PRESENT
//     traffic onto that tier -- including at shapes where the vendor beats both
//     natives. The two questions genuinely differ:
//
//         preferred(r, s)            -> "r is the best route available, vendor
//                                        included". Flipping it moves the
//                                        default in EVERY build.
//         native_tier_preferred(r,s) -> "among the NATIVE routes that can serve
//                                        s, r is the better one". Consulted
//                                        only where there is no vendor left, so
//                                        flipping it moves nothing in a
//                                        vendor-present build.
//
// WITHOUT this, the vendor-free choice is decided entirely by the ORDER array,
// which is static and therefore cannot follow a crossover. Measured cost of
// that for geqrf (docs/perf/qr.md#cta-vs-blocked-crossover): the order lists
// CTA first, and CTA is 1.37x SLOWER than the blocked driver in the same build
// at double n=96, 1.43x at float n=128 -- a pure loss, in the one build this
// work package exists for, with the better route already linked in.
//
// NEUTRAL BY CONSTRUCTION FOR EVERY OTHER TABLE. The hook is OPTIONAL: a
// RouteTable that does not declare it gets `true`, which makes the first pass
// below identical to the second and therefore identical to the single walk this
// replaced. That is why this is not a flag day for gemm, trsm, potrf and gesvd.
// It is also why the default is `true` and not `false`: a table that has not
// thought about the question must keep its old answer.
// ---------------------------------------------------------------------------
template <typename Table, typename Shape>
inline bool native_tier_preferred_or_default(Route r, const Shape& s) {
    if constexpr (requires { Table::native_tier_preferred(r, s); }) {
        return Table::native_tier_preferred(r, s);
    } else {
        return true;
    }
}

// `Shape` is deduced. Most ops pass OpShape; an op whose routing reads
// something OpShape has no field for -- gesvd needs jobu/jobvh and the
// Hermitian flag -- passes a struct deriving from it, rather than growing
// OpShape into a union of every op's arguments.
template <Op O, typename T, typename Shape>
inline Route resolve_route_uninstrumented(Route forced, const Shape& s,
                                          bool vendor_available = true) {
    using Table = RouteTable<O, T>;

    // The choice with nobody forcing anything.
    //
    // Preference decides first: a native route wins only where it is BOTH able
    // to serve the shape and measured to be the better choice for it.
    //
    // Then -- and only if there is no vendor left to fall back to -- a merely
    // SUPPORTED native route. Reaching for that unconditionally is wrong, and
    // the GEMM equivalence test catches it: the orders list the native routes
    // first, so a tiny problem far outside every measured window would select a
    // native kernel where today it goes to the vendor, silently inverting the
    // default.
    //
    // Returning Vendor when nothing at all can serve the shape is deliberate:
    // it is the honest "this needs a vendor and there isn't one" signal, which
    // the caller turns into a diagnostic rather than a wrong answer.
    auto automatic = [&]() -> Route {
        for (const Route* r = Table::order_begin(); r != Table::order_end(); ++r) {
            if (Table::supports(*r, s) && Table::preferred(*r, s)) return *r;
        }
        if (!vendor_available) {
            // TWO PASSES, not one. The first honours the optional native-vs-native
            // tie-break described above; the second is the original walk and is
            // what a table without the hook -- or a shape outside every tier
            // window it declares -- falls back to. A table with no hook makes the
            // two passes identical, so this is a no-op for every op but geqrf.
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

    // A REQUESTED VENDOR STILL HAS TO EXIST. Not hypothetical: GEMM's unset
    // default IS Vendor (legacy_unset_default), so an ordinary call with
    // nothing set arrives here rather than at `automatic` above. Honouring the
    // request unconditionally made the vendor-off degradation unreachable
    // through the real call path -- provable only at the pure layer, on an
    // Origin::Auto input the adapter never produces.
    if (is_vendor(forced)) {
        return vendor_available ? forced : automatic();
    }

    // A BARE ORIGIN LEAVES THE ALGORITHM FREE, so "native" has to resolve to a
    // specific one. For gemm and ormqr there is only one native route and the
    // distinction is invisible; gesvd has three, and returning {Native, Auto}
    // verbatim would hand the caller a route no dispatch tail can map to a
    // kernel. Walk the order restricted to the requested origin -- preference
    // first, then mere support, since the caller has already said it wants this
    // origin whatever the cost.
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

    // FORCED, BUT UNABLE TO SERVE THIS SHAPE. It falls back to the ordinary
    // automatic choice rather than to the vendor. That distinction is real and
    // came from choose_gesvd_provider, whose forced arm ended in
    // `chosen = Provider::Auto;` and re-entered the order walk: a forced `cta`
    // on a real 64x64 matrix is unsupported by the CTA path but must still
    // reach the blocked one, not the vendor. GEMM is unaffected -- with a
    // vendor present its automatic choice for an unsupported shape IS the
    // vendor -- which is why the equivalence test still holds.
    return automatic();
}

// The instrumented entry point, and the ONLY one ops should call.
//
// Every op reaches its route through here, so one call site records all of
// them -- no per-op plumbing, and no way for a new op to be added and silently
// go unmeasured. `record_if_enabled` compiles to nothing unless the build was
// configured with -DBATCHLAS_ENABLE_COVERAGE=ON, so the default build pays a
// dead `if constexpr (false)` and nothing else.
//
// The two extra facts recorded are what make a `reached` row worth having:
// `native_existed` says this op HAS a native route at all, and
// `native_supported` says one could have served THIS shape. A row where the
// vendor was chosen and native_supported is true is a tuning question; one
// where native_existed is false is a missing kernel. Those are different work
// items and the row has to distinguish them.
//
// `s` is sliced to OpShape on purpose: GesvdShape and SyevShape carry extra
// routing inputs that the CSV has no column for, and the shape_class bucket is
// computed from the base fields anyway.
template <Op O, typename T, typename Shape>
inline Route resolve_route(Route forced, const Shape& s, bool vendor_available = true) {
    const Route chosen = resolve_route_uninstrumented<O, T, Shape>(forced, s, vendor_available);

    // Runtime-gated, not compiled out: see the note in coverage.hh on why the
    // macro version silently recorded nothing. The extra order walk below is
    // inside the branch, so an ordinary call pays one predicted test.
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
