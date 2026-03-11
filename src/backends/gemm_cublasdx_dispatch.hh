#pragma once

#include "gemm_cublasdx_kernels.hh"

#include "../queue.hh"

#include <blas/matrix.hh>

namespace batchlas::backend {

const char* cublasdx_gemm_trace_name(cublasdx_gemm::CuBLASDxGemmVariant variant);

bool cublasdx_gemm_has_forced_variant();

bool cublasdx_gemm_variant_available(cublasdx_gemm::CuBLASDxGemmVariant variant);

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