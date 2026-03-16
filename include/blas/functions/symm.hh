#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

template <Backend Ba, RealScalar T>
Event symm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo);

template <Backend Ba, RealScalar T>
inline Event symm(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& A,
                  const Matrix<T, MatrixFormat::Dense>& Bmat,
                  const Matrix<T, MatrixFormat::Dense>& Cmat,
                  T alpha,
                  T beta,
                  Side side,
                  Uplo uplo) {
    return symm<Ba, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(Bmat),
                       MatrixView<T, MatrixFormat::Dense>(Cmat),
                       alpha,
                       beta,
                       side,
                       uplo);
}

}  // namespace batchlas