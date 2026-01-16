#pragma once

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

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
