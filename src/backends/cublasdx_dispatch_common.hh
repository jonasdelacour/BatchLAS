#pragma once

// CUDA-specific route helpers.
//
// The backend-neutral half of what used to live here now sits in
// route_common.hh; this header keeps only what genuinely needs CUDA, and
// re-includes the portable half so existing consumers see the same set of names
// as before. Anything portable that needs ceil_div / parse_cublasdx_variant_request
// / is_gpu_queue / should_use_cublasdx / throw_forced_cublasdx_unavailable should
// include route_common.hh directly rather than this file, or it will drag in
// <cuda_runtime_api.h> and become CUDA-only for no reason.

#include "route_common.hh"

#include "gemm_variant.hh"

#include <cuda_runtime_api.h>
#include <sycl/sycl.hpp>

namespace batchlas::backend::detail {

inline cudaStream_t cuda_stream_from_queue(const Queue& ctx) {
    return sycl::get_native<sycl::backend::ext_oneapi_cuda>(*ctx);
}

inline bool cublasdx_variant_needs_fallback(cublasdx_gemm::CuBLASDxGemmVariant variant,
                                            bool fused_kernel_available) {
    return variant == cublasdx_gemm::CuBLASDxGemmVariant::VendorFallback ||
           !cublasdx_gemm_variant_available(variant) ||
           !fused_kernel_available;
}

} // namespace batchlas::backend::detail
