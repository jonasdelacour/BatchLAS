#pragma once

#include "gemm_cublasdx_kernels.hh"

#include <blas/enums.hh>

#include <cuda_runtime_api.h>

namespace batchlas::backend::symm_cublasdx {

struct SymmLaunchDescriptor {
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
};

bool available();

cudaError_t launch_float(cublasdx_gemm::CuBLASDxGemmVariant variant,
                         const SymmLaunchDescriptor& desc,
                         Side side,
                         Uplo uplo,
                         cudaStream_t stream);

} // namespace batchlas::backend::symm_cublasdx