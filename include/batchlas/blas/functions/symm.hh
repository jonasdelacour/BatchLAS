#pragma once

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using symm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Side, Uplo);
}  // namespace sig


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

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(symm)

}  // namespace batchlas
