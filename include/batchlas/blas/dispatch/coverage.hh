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
// The dynamic half is gated at RUNTIME on $BATCHLAS_COVERAGE_OUT -- the same
// variable that decides whether anything is written at all -- so recording and
// emission cannot disagree about whether coverage is on.
//
// IT WAS A COMPILE-TIME GATE FIRST, AND THAT WAS THE WRONG CALL, TWICE OVER.
//
// The stated reason was "the counters are cheap per call, but gemm is called in
// inner loops and cheap is not free". That does not survive reading the code
// it guards: resolve_route runs ONCE PER OP INVOCATION, not per element, and it
// already walks the route order calling supports() and preferred() on every
// entry. A predicted branch on a global bool is far cheaper than the work
// sitting next to it, so the gate was optimising something that was never the
// cost.
//
// The second problem is the one that actually bit. resolve_route is an inline
// function template, so EVERY TU that routes instantiates its own weak copy.
// ELF resolves the executable's weak symbols ahead of a shared library's, so a
// test compiled without the macro interposes its uninstrumented copy over the
// library's instrumented one -- and the library's own calls then record
// nothing. Observed exactly that: build with -DBATCHLAS_ENABLE_COVERAGE=ON,
// confirm libbatchlas_backends.so references coverage::record, run gemm_tests,
// get a coverage file with a correct header and ZERO `reached` rows, because
// gemm_tests carried its own `resolve_route_uninstrumented<Op::gemm, float>`.
//
// A compile-time switch on an inline function in a header is only sound if
// EVERY TU in the process agrees on it, which is not something a library can
// enforce on its consumers. The runtime gate has no such requirement.

#include <cstdint>
#include <string>

#include <batchlas/backend_config.h>

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch::coverage {

// Emitted to $BATCHLAS_COVERAGE_OUT (CSV) from an atexit handler that touches
// no SYCL object -- the standing static-destruction rule in this tree.
// `native_route_supported` is a TRI-STATE: 1 yes, 0 no, -1 the call site could
// not tell. The third value exists because a gate that merely says "not this
// route" cannot distinguish "nothing native serves this shape" from "something
// does, but the vendor was preferred" -- and recording either as a definite
// answer would be a claim the caller cannot support.
void record(Op op, ScalarKind scalar, Backend backend, const OpShape& shape,
            Route chosen, bool native_route_existed, int native_route_supported);

// A call that found no route at all. Recorded separately because it is the row
// that matters most: it is a gap, not a preference.
void record_miss(Op op, ScalarKind scalar, Backend backend, const char* library);

// The static table: what routes this build contains, independent of any run.
// Returns CSV with a header row.
std::string static_table();

// Set once, from $BATCHLAS_COVERAGE_OUT, by a dynamic initialiser in
// coverage.cc. A plain bool rather than a function-local static so the hot path
// is a load and a predictable branch, with no guard-variable acquire.
//
// It lives in exactly one TU, so unlike the macro it cannot differ between the
// library and its callers -- which is the whole point of the change.
extern bool g_dynamic_enabled;

inline bool dynamic_enabled() { return g_dynamic_enabled; }

// Called from resolve_route -- the single choke point every op passes through,
// so adding an op cannot silently skip coverage.
inline void record_if_enabled(Op op, ScalarKind scalar, Backend backend,
                              const OpShape& shape, Route chosen,
                              bool native_existed, int native_supported) {
    if (g_dynamic_enabled) {
        record(op, scalar, backend, shape, chosen, native_existed, native_supported);
    }
}

} // namespace batchlas::dispatch::coverage
