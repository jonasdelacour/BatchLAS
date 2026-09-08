#pragma once

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using orgqr = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<T>, Span<std::byte>);

template <typename T>
using orgqr_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Span<T>);
}  // namespace sig


template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau);

}  // namespace batchlas

namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(orgqr)
BATCHLAS_ACCEPT_OWNING(orgqr_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(orgqr)
BATCHLAS_DISPATCH_ON_QUEUE(orgqr_buffer_size)

}  // namespace batchlas
