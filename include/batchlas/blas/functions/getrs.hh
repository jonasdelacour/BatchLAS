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
using getrs = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Transpose, Span<int64_t>, Span<std::byte>);

template <typename T>
using getrs_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Transpose);
}  // namespace sig


template <Backend Back, typename T>
Event getrs(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Transpose transA,
           Span<int64_t> pivots,
           Span<std::byte> work_space);

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Transpose transA);

}  // namespace batchlas

namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(getrs)
BATCHLAS_ACCEPT_OWNING(getrs_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(getrs)
BATCHLAS_DISPATCH_ON_QUEUE(getrs_buffer_size)

}  // namespace batchlas
