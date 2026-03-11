#pragma once

#include <blas/enums.hh>

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

struct GemmLaunchDescriptor {
    const float* a_ptr;
    const float* b_ptr;
    float* c_ptr;
    int lda;
    int ldb;
    int ldc;
    int stride_a;
    int stride_b;
    int stride_c;
    int m;
    int n;
    int k;
    int batch;
    float alpha;
    float beta;
    bool packet_a;
    bool packet_b;
    bool aligned_fast_path;
};

cudaError_t launch_float(CuBLASDxGemmVariant variant,
                         const GemmLaunchDescriptor& desc,
                         cudaStream_t stream);

} // namespace batchlas::backend::cublasdx_gemm