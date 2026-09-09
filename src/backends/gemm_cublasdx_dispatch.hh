#pragma once

#include <batchlas/backend_config.h>

#include "gemm_cublasdx_kernels.hh"

#include "../queue.hh"

#include <batchlas/blas/matrix.hh>

namespace batchlas::backend {

const char* cublasdx_gemm_trace_name(cublasdx_gemm::CuBLASDxGemmVariant variant);

bool cublasdx_gemm_has_forced_variant();

// gemm_cublasdx_dispatch.cc is compiled only when cuBLAS is present (it calls
// gemm_vendor_cuda_raw), so in a vendor-free build this has no definition. It is
// nonetheless the predicate every cuBLASDx test uses to SKIP itself, so the
// honest answer there is a compile-time `false` rather than a link error:
// "no cuBLASDx variant is available" is exactly true in that build.
#if BATCHLAS_HAS_CUBLAS
bool cublasdx_gemm_variant_available(cublasdx_gemm::CuBLASDxGemmVariant variant);
#else
inline bool cublasdx_gemm_variant_available(cublasdx_gemm::CuBLASDxGemmVariant) {
    return false;
}
#endif

cublasdx_gemm::CuBLASDxGemmVariant forced_cublasdx_gemm_variant();

cublasdx_gemm::CuBLASDxGemmVariant cublasdx_gemm_select_variant(
    const MatrixView<float, MatrixFormat::Dense>& A,
    const MatrixView<float, MatrixFormat::Dense>& B,
    const MatrixView<float, MatrixFormat::Dense>& C,
    Transpose transA,
    Transpose transB);

Event gemm_cublasdx(Queue& ctx,
                    const MatrixView<float, MatrixFormat::Dense>& A,
                    const MatrixView<float, MatrixFormat::Dense>& B,
                    const MatrixView<float, MatrixFormat::Dense>& C,
                    float alpha,
                    float beta,
                    Transpose transA,
                    Transpose transB,
                    ComputePrecision precision);

Event gemm_vendor_cuda_raw(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Transpose transA,
                           Transpose transB,
                           ComputePrecision precision);

} // namespace batchlas::backend