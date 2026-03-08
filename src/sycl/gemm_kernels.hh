#pragma once

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas::sycl_gemm {

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
    Tiled128x32RegisterK32,
    Tiled128x32RegisterK32S2U1,
    Tiled128x32RegisterK32S2U2,
    Tiled128x32RegisterK32S1U4,
    Tiled128x64RegisterK32Large,
    Tiled32x128RegisterK16,
    Tiled32x128RegisterK16TN,
    Tiled32x128RegisterK16TT,
};

template <typename T>
KernelVariant select_kernel_variant(const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Transpose transA,
                                    Transpose transB);

template <typename T>
Event gemm_custom(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha,
                  T beta,
                  Transpose transA,
                  Transpose transB,
                  ComputePrecision precision);

} // namespace batchlas::sycl_gemm