#pragma once

#include "gemm_cublasdx_kernels.hh"

#include <blas/enums.hh>

#include <cuda_runtime_api.h>

namespace batchlas::backend::syrk_cublasdx {

struct SyrkLaunchDescriptor {
    const float* a_ptr;
    float* c_ptr;
    int lda;
    int ldc;
    int stride_a;
    int stride_c;
    int n;
    int k;
    int batch;
    float alpha;
    float beta;
};

bool available();

cudaError_t launch_float(cublasdx_gemm::CuBLASDxGemmVariant variant,
                         const SyrkLaunchDescriptor& desc,
                         Uplo uplo,
                         Transpose transA,
                         cudaStream_t stream);

} // namespace batchlas::backend::syrk_cublasdx