#pragma once

// What this build can actually do, and what a test run actually reached.
//
// After S6 the vendor-free build runs and throws NoRouteError where it has no
// implementation. VENDOR_FREE_BASELINE.md records which test SUITES fail, which
// is the right unit for spotting regressions and the wrong unit for planning:
// "ortho_tests fails" does not say whether the gap is potrf, trsm, geqrf, orgqr
// or gemv, and one missing kernel fails a dozen suites.
//
// Two tables answer different halves of that, and both are needed:
//
//   STATIC  -- what is LINKED. Iterates (Op x Backend x ScalarKind) over the
//              route predicates with no kernel run at all, so it is exact,
//              instant, and available in a build with no GPU present. This is
//              the planning input: it names the ops with no native kernel.
//
//   DYNAMIC -- what a run REACHED, per shape. Counts (op, scalar, backend,
//              shape_class) and records whether the chosen route was native or
//              vendor. This is the burn-down input: it says which shapes real
//              callers actually hit, so WP3-WP8 can cover those first.
//
// The dynamic half is COMPILED OUT unless -DBATCHLAS_ENABLE_COVERAGE=ON. The
// spec has it always on, arguing the counters are cheap; they are cheap per
// call, but gemm is called in inner loops and "cheap" is not "free". Making it
// a build option keeps the measurement honest -- a coverage build is not the
// build you benchmark -- and costs nothing in the default one, where
// record_route() compiles to nothing.

#include <cstdint>
#include <string>

#include <batchlas/backend_config.h>

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch::coverage {

// Emitted to $BATCHLAS_COVERAGE_OUT (CSV) from an atexit handler that touches
// no SYCL object -- the standing static-destruction rule in this tree.
void record(Op op, ScalarKind scalar, Backend backend, const OpShape& shape,
            Route chosen, bool native_route_existed, bool native_route_supported);

// A call that found no route at all. Recorded separately because it is the row
// that matters most: it is a gap, not a preference.
void record_miss(Op op, ScalarKind scalar, Backend backend, const char* library);

// The static table: what routes this build contains, independent of any run.
// Returns CSV with a header row.
std::string static_table();

// True when the dynamic half is compiled in.
inline constexpr bool dynamic_enabled =
#ifdef BATCHLAS_ENABLE_COVERAGE
    true;
#else
    false;
#endif

// Zero-cost in the default build.
inline void record_if_enabled(Op op, ScalarKind scalar, Backend backend,
                              const OpShape& shape, Route chosen,
                              bool native_existed, bool native_supported) {
    if constexpr (dynamic_enabled) {
        record(op, scalar, backend, shape, chosen, native_existed, native_supported);
    }
}

} // namespace batchlas::dispatch::coverage
