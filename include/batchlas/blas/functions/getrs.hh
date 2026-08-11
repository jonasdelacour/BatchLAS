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
inline Event getrs(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   Transpose transA,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space) {
        return getrs<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA, pivots, work_space);
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Transpose transA);

template <Backend Back, typename T>
inline size_t getrs_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 const Matrix<T, MatrixFormat::Dense>& Bmat,
                                                 Transpose transA) {
        return getrs_buffer_size<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA);
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getrs)
BATCHLAS_DISPATCH_ON_QUEUE(getrs_buffer_size)

}  // namespace batchlas
