#pragma once

#include "gemm_cublasdx_kernels.hh"

#include <blas/enums.hh>

#include <cuda_runtime_api.h>

namespace batchlas::backend::syr2k_cublasdx {

struct Syr2kLaunchDescriptor {
    const float* a_ptr;
    const float* b_ptr;
    float* c_ptr;
    int lda;
    int ldb;
    int ldc;
    int stride_a;
    int stride_b;
    int stride_c;
    int n;
    int k;
    int batch;
    float alpha;
    float beta;
};

bool available();

cudaError_t launch_float(cublasdx_gemm::CuBLASDxGemmVariant variant,
                         const Syr2kLaunchDescriptor& desc,
                         Uplo uplo,
                         Transpose transA,
                         cudaStream_t stream);

} // namespace batchlas::backend::syr2k_cublasdx