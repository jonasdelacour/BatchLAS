#pragma once

// Backend-neutral queue helpers for the level-3 tile and expansion kernels. Nothing here names
// a CUDA type. Kernel choice for symm/syrk/syr2k/trmm is flat selection (src/ops/<op>).

#include "../math-helpers.hh"

#include <sycl/sycl.hpp>

#include <batchlas/settings.hh>

namespace batchlas::backend::detail {

inline int ceil_div(int value, int divisor) {
    return internal::ceil_div(value, divisor);
}

inline bool is_gpu_queue(const Queue& ctx) {
    return ctx.device().type == DeviceType::GPU;
}

} // namespace batchlas::backend::detail
