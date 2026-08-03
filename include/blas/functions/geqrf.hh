#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using geqrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<T>, Span<std::byte>);

template <typename T>
using geqrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<T>);
}  // namespace sig


template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event geqrf(Queue& ctx,
                        const Matrix<T,MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> work_space) {
        return geqrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, work_space);
}

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t geqrf_buffer_size(Queue& ctx,
                                                 const Matrix<T,MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return geqrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(geqrf)
BATCHLAS_DISPATCH_ON_QUEUE(geqrf_buffer_size)

}  // namespace batchlas
