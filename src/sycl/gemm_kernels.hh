#pragma once

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemm {

// Trace identities only (BATCHLAS_KERNEL_TRACE names). Which kernel runs is decided by
// src/ops/gemm/gemm.cc; each launcher below names the instantiation it took.
enum class KernelVariant {
    Direct,
    Tiled16,
    Tiled32x32Register,
    Tiled64x64Register,
    Tiled64x64RegisterK16,
    Tiled64x64RegisterK16TN,
    Tiled64x64RegisterK16NT,
    Tiled64x64RegisterK16TT,
    Tiled128x32RegisterK16,
    Tiled128x32RegisterK16TN,
    Tiled128x32RegisterK16NT,
    Tiled128x32RegisterK16TT,
    Tiled128x32RegisterK32TN,
    Tiled128x32RegisterK32NT,
    Tiled128x32RegisterK32TT,
    Tiled128x64RegisterK16TN,
    Tiled128x64RegisterK16NT,
    Tiled128x64RegisterK16TT,
    Tiled128x32RegisterK32S2U1Aligned,  // the NN 128x32x32 tile's two legs
    Tiled128x32RegisterK32S2U1Generic,
    Tiled128x64RegisterK32Large,
    Tiled128x64RegisterK32LargeU2,
    Tiled128x128RegisterK8,
    Tiled64x64RegisterK16Wide,
    Tiled64x64RegisterK16WideCN,
    Tiled64x64RegisterK16WideNC,
    Tiled128x32RegisterK16WideNC,
    Tiled32x128RegisterK16WideCN,
    Tiled32x128RegisterK16,
    Tiled32x128RegisterK16TN,
    Tiled32x128RegisterK16TT,
    SmallBatched,
    Tiled32x32RegisterK16Wide,
    Tiled16x16RegisterK16Wide,
};

// One launcher per gemm family (src/ops/gemm/choice.hh). Each serves exactly what its family's
// can_run admits and throws batchlas::unsupported otherwise -- never a silent fallback to another
// kernel. The transpose instantiation and the aligned/predicated leg are derived here.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_direct(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& B,
                                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                        Transpose transA, Transpose transB);

template <typename T>
BATCHLAS_INTERNAL_API Event gemm_tiled(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                       Transpose transA, Transpose transB);

// Real scalars, max(m, n, k) <= 64.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_small(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B,
                                       const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                       Transpose transA, Transpose transB);

// float only; (tm, tn, tk, u) must be a row of ops::gemm::reg_configs instantiated for the form.
BATCHLAS_INTERNAL_API Event gemm_reg(Queue& ctx, int tm, int tn, int tk, int u,
                                     const MatrixView<float, MatrixFormat::Dense>& A,
                                     const MatrixView<float, MatrixFormat::Dense>& B,
                                     const MatrixView<float, MatrixFormat::Dense>& C, float alpha, float beta,
                                     Transpose transA, Transpose transB);

// (tm, tn, tk) must be a row of ops::gemm::wide_configs instantiated for the form.
template <typename T>
BATCHLAS_INTERNAL_API Event gemm_wide(Queue& ctx, int tm, int tn, int tk, const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta,
                                      Transpose transA, Transpose transB);

} // namespace batchlas::sycl_gemm
