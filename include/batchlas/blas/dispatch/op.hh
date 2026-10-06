#pragma once

#include <utility>

namespace batchlas {

/// @brief Tags a call that is a pure wrapper around an external library, and invokes it.
///
/// Currently a no-op that returns `f()`; it exists as the single place to add
/// tracing or instrumentation of direct vendor calls later.
/// @param f  the call to make
/// @return whatever `f()` returns
/// @ingroup dispatch
// evidence: docs/design/vendor-independence.md#what-is-still-open-architecturally
template <class F>
decltype(auto) op_external(const char* /*name*/, F&& f) {
    return std::forward<F>(f)();
}

} // namespace batchlas
