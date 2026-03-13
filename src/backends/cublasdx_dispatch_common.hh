#pragma once

#include "gemm_variant.hh"
#include "../math-helpers.hh"

#include <cuda_runtime_api.h>
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

template <typename Variant>
Variant parse_cublasdx_variant_request(const char* env_var,
                                      Variant vendor_variant,
                                      Variant custom_variant,
                                      Variant auto_variant) {
    const char* raw = std::getenv(env_var);
    if (!raw) {
        return auto_variant;
    }

    std::string value(raw);
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }

    if (value == "vendor") {
        return vendor_variant;
    }
    if (value == "cublasdx" || value == "dx" || value == "custom") {
        return custom_variant;
    }
    if (value == "auto") {
        return auto_variant;
    }

    return auto_variant;
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

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

inline bool cublasdx_variant_needs_fallback(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                            bool fused_kernel_available) {
    return variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback ||
           !cublasdx_gemm_variant_available(variant) ||
           !fused_kernel_available;
}

[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view env_var,
                                                           std::string_view op_name,
                                                           const std::string& reason) {
    throw std::runtime_error(std::string(env_var) + "=cublasdx requested, but fused cuBLASDx " +
                             std::string(op_name) + " is unavailable: " + reason);
}

} // namespace batchlas::backend::detail