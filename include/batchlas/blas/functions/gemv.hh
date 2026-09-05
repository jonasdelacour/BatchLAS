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
using gemv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const VectorView<T>&,
                   const VectorView<T>&,
                   T, T, Transpose);

// backend::gemv_vendor's signature. NOT an alias for sig::gemv: the vendor
// parameter order can differ from the public one -- trsm's alpha moves to
// the end -- so each is spelled out from the definition it describes.
template <typename T>
using gemv_vendor = Event(Queue&,
                          const MatrixView<T,MatrixFormat::Dense>&,
                          const VectorView<T>&,
                          const VectorView<T>&,
                          T,
                          T,
                          Transpose);
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


namespace batchlas::backend {

// The vendor path for gemv.
//
// DECLARATION ONLY. The public `gemv<Back, T>` used to be DEFINED inside each
// vendor TU, so dropping a vendor library dropped the public entry point along
// with the vendor path. WP0 S5 moves that definition to
// src/dispatch/entry_points/level3.cc; what stays behind is the vendor
// implementation, named as such. Each vendor wrapper TU defines this primary
// template for its own Backend value and instantiates it there.
template <Backend B, typename T>
Event gemv_vendor(Queue& ctx,
                  const MatrixView<T,MatrixFormat::Dense>& A,
                  const VectorView<T>& X,
                  const VectorView<T>& Y,
                  T alpha,
                  T beta,
                  Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(gemv)

}  // namespace batchlas
