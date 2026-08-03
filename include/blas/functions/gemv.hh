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
using gemv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const VectorView<T>&,
                   const VectorView<T>&,
                   T, T, Transpose);
}  // namespace sig


template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA);

template <Backend B, typename T>
inline Event gemv(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const Vector<T>& X,
            const Vector<T>& Y,
            T alpha,
            T beta,
            Transpose transA) {
    return gemv<B, T>(ctx, A,
                         static_cast<VectorView<T>>(X),
                         static_cast<VectorView<T>>(Y),
                         alpha, beta, transA);
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(gemv)

}  // namespace batchlas
