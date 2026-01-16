#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

template <Backend Ba, typename T>
Event trmm(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                const MatrixView<T, MatrixFormat::Dense>& C,
                T alpha,
                Side side,
                Uplo uplo,
                Transpose transA,
                Diag diag);

template <Backend Ba, typename T>
inline Event trmm(Queue& ctx,
                                 const Matrix<T, MatrixFormat::Dense>& A,
                                 const Matrix<T, MatrixFormat::Dense>& Bmat,
                                 const Matrix<T, MatrixFormat::Dense>& Cmat,
                                 T alpha,
                                 Side side,
                                 Uplo uplo,
                                 Transpose transA,
                                 Diag diag) {
        return trmm<Ba,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, side, uplo, transA, diag);
}

}  // namespace batchlas
