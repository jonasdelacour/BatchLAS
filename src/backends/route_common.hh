#pragma once

// Backend-neutral route-selection helpers.
//
// These were carved out of cublasdx_dispatch_common.hh, which includes
// <cuda_runtime_api.h> so that cuda_stream_from_queue() can name cudaStream_t.
// That one include made the whole header CUDA-only, and with it every consumer
// -- including triangular_expand.hh and the symm/syrk/syr2k/trmm route
// selectors, which are portable SYCL and have no business being confined to a
// CUDA build. Nothing below names a CUDA type; the CUDA-specific helpers stay
// where they were.
//
// Names are deliberately unchanged from their previous spellings so that this
// split is a pure relocation with no call-site churn. `should_use_cublasdx` in
// particular now reads oddly, since it decides between a vendor route and *any*
// custom route rather than a cuBLASDx one specifically -- renaming it is worth
// doing, but as its own commit.

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

// Parses a BATCHLAS_<OP>_VARIANT env var into one of three route requests.
//
// Note the default: an UNSET variable yields `auto_variant`, not
// `vendor_variant`. So the custom level-3 routes are live by default wherever
// their heuristics fire -- unlike GEMM, whose gemm_variant_request() defaults to
// Vendor. That asymmetry is easy to misread and is load-bearing for anyone
// reasoning about what runs out of the box.
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

[[noreturn]] inline void throw_forced_cublasdx_unavailable(std::string_view env_var,
                                                           std::string_view op_name,
                                                           const std::string& reason) {
    throw std::runtime_error(std::string(env_var) + "=cublasdx requested, but fused cuBLASDx " +
                             std::string(op_name) + " is unavailable: " + reason);
}

} // namespace batchlas::backend::detail
