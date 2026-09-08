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
using trmm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, Side, Uplo, Transpose, Diag);
}  // namespace sig


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

}  // namespace batchlas

namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(trmm)

BATCHLAS_DISPATCH_ON_QUEUE(trmm)

}  // namespace batchlas
