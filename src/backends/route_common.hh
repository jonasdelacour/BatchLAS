#pragma once

// Backend-neutral route-selection helpers: nothing here may name a CUDA type. `should_use_cublasdx`
// decides vendor vs ANY custom route; the name is historical. evidence: docs/perf/dispatch.md#the-environment-vocabulary

#include "../math-helpers.hh"

#include <sycl/sycl.hpp>

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace batchlas::backend::detail {

inline int ceil_div(int value, int divisor) {
    return internal::ceil_div(value, divisor);
}

inline bool is_gpu_queue(const Queue& ctx) {
    return ctx.device().type == DeviceType::GPU;
}

template <typename Variant>
bool should_use_cublasdx(const Queue& ctx,
                        Variant request,
                        Variant vendor_variant,
                        Variant custom_variant,
                        bool problem_supported,
                        bool heuristic_preferred) {
    if (request == custom_variant) {
        return true;
    }
    if (!is_gpu_queue(ctx) || !problem_supported) {
        return false;
    }
    if (request == vendor_variant) {
        return false;
    }
    return heuristic_preferred;
}

[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view env_var,
                                                           std::string_view op_name,
                                                           const std::string& reason) {
    throw batchlas::unsupported(std::string(env_var) + "=cublasdx requested, but fused cuBLASDx " +
                             std::string(op_name) + " is unavailable: " + reason);
}

} // namespace batchlas::backend::detail
