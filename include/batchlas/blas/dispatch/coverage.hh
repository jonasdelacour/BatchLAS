#pragma once

/// @file
/// @brief The coverage instrument: what this build can do, and what a run actually reached.
///
/// Two tables. **static** (`linked` rows) iterates Op x Backend x ScalarKind over
/// the route predicates with no kernel run: exact, instant, no GPU needed, and
/// the planning input. **dynamic** (`reached` rows) counts, per call, the
/// (op, scalar, backend, shape_class) and whether the chosen route was native or
/// vendor: the burn-down input. `miss` rows record calls nothing could serve.
///
/// The dynamic half is gated at RUNTIME on `$BATCHLAS_COVERAGE_OUT`, the same
/// variable that decides whether anything is written, so recording and emission
/// cannot disagree.
/// @invariant Never a compile-time gate: resolve_route() is an inline template, so
///            every TU carries its own weak copy, and an executable's uninstrumented
///            copy interposes over the library's and records nothing.
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#vendor-independence-the-coverage-instrument

#include <batchlas/export.hh>
#include <cstdint>
#include <string>

#include <batchlas/backend_config.h>

#include <batchlas/blas/dispatch/route.hh>

namespace batchlas::dispatch::coverage {

/// @brief Records one resolved call as a `reached` row.
///
/// Rows are emitted to `$BATCHLAS_COVERAGE_OUT` (CSV) from an atexit handler that
/// touches no SYCL object.
/// @param native_route_existed    the op's table lists any native route
/// @param native_route_supported  TRI-STATE: 1 yes, 0 no, -1 the call site could not
///        tell. A gate that merely says "not this route" cannot distinguish "nothing
///        native serves this shape" from "something does, but the vendor was
///        preferred", and recording either as definite would be a claim it cannot support.
/// @ingroup dispatch
BATCHLAS_API void record(Op op, ScalarKind scalar, Backend backend, const OpShape& shape,
                         Route chosen, bool native_route_existed, int native_route_supported);

/// @brief Records a call that found no route at all: a gap, not a preference.
/// @param library  the missing library, as named in the diagnostic
/// @ingroup dispatch
BATCHLAS_API void record_miss(Op op, ScalarKind scalar, Backend backend, const char* library);

/// @brief The static table: which routes this build contains, independent of any run.
/// @return CSV text with a header row
/// @ingroup dispatch
BATCHLAS_API std::string static_table();

/// @brief Whether the dynamic table records; set once from `$BATCHLAS_COVERAGE_OUT` by a
///        dynamic initialiser in coverage.cc.
///
/// A plain bool rather than a function-local static, so the hot path is a load and
/// a predictable branch with no guard-variable acquire. It lives in exactly one TU,
/// so it cannot differ between the library and its callers.
/// @ingroup dispatch
extern bool g_dynamic_enabled;

/// @brief Whether dynamic coverage is on.
/// @ingroup dispatch
inline bool dynamic_enabled() { return g_dynamic_enabled; }

/// @brief record() when dynamic coverage is on; called from resolve_route(), the single
///        choke point every op passes through, so adding an op cannot silently skip coverage.
/// @ingroup dispatch
inline void record_if_enabled(Op op, ScalarKind scalar, Backend backend,
                              const OpShape& shape, Route chosen,
                              bool native_existed, int native_supported) {
    if (g_dynamic_enabled) {
        record(op, scalar, backend, shape, chosen, native_existed, native_supported);
    }
}

} // namespace batchlas::dispatch::coverage
