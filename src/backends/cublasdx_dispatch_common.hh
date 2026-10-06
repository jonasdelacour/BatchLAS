#pragma once

// CUDA-only route helpers. Portable code includes route_common.hh instead: this header
// pulls in <cuda_runtime_api.h>. evidence: docs/perf/dispatch.md#the-environment-vocabulary

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
