#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

template <Backend Ba, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA);

template <Backend Ba, RealScalar T>
inline Event syr2k(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   T alpha,
                   T beta,
                   Uplo uplo,
                   Transpose transA) {
    return syr2k<Ba, T>(ctx,
                        MatrixView<T, MatrixFormat::Dense>(A),
                        MatrixView<T, MatrixFormat::Dense>(Bmat),
                        MatrixView<T, MatrixFormat::Dense>(Cmat),
                        alpha,
                        beta,
                        uplo,
                        transA);
}

} // namespace batchlas