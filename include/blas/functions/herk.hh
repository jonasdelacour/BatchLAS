#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using herk = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   float_t<T>, float_t<T>, Uplo, Transpose);
}  // namespace sig


// C = alpha * A * A^H + beta * C (Transpose::NoTrans, A is n x k) or
// C = alpha * A^H * A + beta * C (Transpose::ConjTrans, A is k x n), with C
// Hermitian n x n: only the triangle named by `uplo` is written, and the
// diagonal comes out real -- A A^H is Hermitian and alpha and beta are real, so
// an imaginary part on C's diagonal is neither read nor produced.
//
// alpha and beta are real, not T. That is the BLAS signature rather than an
// approximation of it: a complex alpha would make alpha * A * A^H
// non-Hermitian, so there is no such operation to express. cublas?herk and
// cblas_?herk both take the real scalar directly.
//
// Constrained to complex scalars; the real spelling of this is syrk.
template <Backend Ba, ComplexScalar T>
Event herk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha,
           float_t<T> beta,
           Uplo uplo,
           Transpose transA);

template <Backend Ba, ComplexScalar T>
inline Event herk(Queue& ctx,
                  const Matrix<T, MatrixFormat::Dense>& A,
                  const Matrix<T, MatrixFormat::Dense>& Cmat,
                  float_t<T> alpha,
                  float_t<T> beta,
                  Uplo uplo,
                  Transpose transA) {
    return herk<Ba, T>(ctx,
                       MatrixView<T, MatrixFormat::Dense>(A),
                       MatrixView<T, MatrixFormat::Dense>(Cmat),
                       alpha,
                       beta,
                       uplo,
                       transA);
}

} // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(herk)

}  // namespace batchlas
