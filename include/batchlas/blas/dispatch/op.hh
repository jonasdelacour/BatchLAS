#pragma once

#include <utility>

namespace batchlas {

// Lightweight tag for operations that are pure wrappers around external libraries.
// This is currently a no-op, but provides a single place to add tracing/
// instrumentation later.
template <class F>
decltype(auto) op_external(const char* /*name*/, F&& f) {
    return std::forward<F>(f)();
}

} // namespace batchlas
