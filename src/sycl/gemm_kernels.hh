#pragma once

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas::sycl_gemm {

enum class KernelVariant {
    Direct,
    Tiled16,
    Tiled32x32Register,
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