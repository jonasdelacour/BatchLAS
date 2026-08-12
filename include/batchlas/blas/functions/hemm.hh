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
using hemm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Side, Uplo);
}  // namespace sig


// C = alpha * A * B + beta * C (Side::Left) or alpha * B * A + beta * C
// (Side::Right), with A Hermitian: only the triangle named by `uplo` is read,
// the opposite one is taken to be its conjugate transpose, and the imaginary
// part of the diagonal is taken to be zero whatever is stored there.
//
// Constrained to complex scalars, which is the whole of the difference from
// symm -- for a real matrix "Hermitian" and "symmetric" are the same statement,
// and BLAS has no ?hemm for real types.
template <Backend Ba, ComplexScalar T>
Event hemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo);

template <Backend Ba, ComplexScalar T>
inline Event hemm(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& A,
                  const Matrix<T, MatrixFormat::Dense>& Bmat,
                  const Matrix<T, MatrixFormat::Dense>& Cmat,
                  T alpha,
                  T beta,
                  Side side,
                  Uplo uplo) {
    return hemm<Ba, T>(ctx,
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

BATCHLAS_DISPATCH_ON_QUEUE(hemm)

}  // namespace batchlas
