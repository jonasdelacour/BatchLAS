#pragma once

#include "gemm_cublasdx_kernels.hh"

#include <cuda_runtime_api.h>

namespace batchlas::backend::cublasdx_gemm {

bool available();

bool variant_supported(CuBLASDxGemmVariant variant,
                       const GemmLaunchDescriptor& desc);

cudaError_t launch_float(CuBLASDxGemmVariant variant,
                         const GemmLaunchDescriptor& desc,
                         cudaStream_t stream);

} // namespace batchlas::backend::cublasdx_gemm