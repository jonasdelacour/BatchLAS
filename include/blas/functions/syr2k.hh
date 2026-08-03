#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using syr2k = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    T, T, Uplo, Transpose);
}  // namespace sig


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

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(syr2k)

}  // namespace batchlas
