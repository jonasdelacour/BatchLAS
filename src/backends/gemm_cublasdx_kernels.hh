#pragma once

#include <batchlas/blas/enums.hh>

#include <cuda_runtime_api.h>

namespace batchlas::backend::cublasdx_gemm {

enum class CuBLASDxGemmVariant {
    VendorFallback,
    CuBLASDx32x32x32NN,
    CuBLASDx32x32x32TN,
    CuBLASDx32x32x32NT,
    CuBLASDx32x32x32TT,
    CuBLASDx64x64x32NN,
    CuBLASDx64x64x32TN,
    CuBLASDx64x64x32NT,
    CuBLASDx64x64x32TT,
};

} // namespace batchlas::backend::cublasdx_gemm
