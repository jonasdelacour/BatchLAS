#pragma once

// Backend-neutral helpers for the level-3 tile and expansion kernels; names no CUDA type.

#include "../math-helpers.hh"

#include <sycl/sycl.hpp>

#include <batchlas/settings.hh>

namespace batchlas::backend::detail {

inline int ceil_div(int value, int divisor) {
    return internal::ceil_div(value, divisor);
}

} // namespace batchlas::backend::detail
