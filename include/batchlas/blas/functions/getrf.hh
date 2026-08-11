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
using getrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>);

template <typename T>
using getrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event getrf(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), pivots, work_space);
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getrf_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getrf)
BATCHLAS_DISPATCH_ON_QUEUE(getrf_buffer_size)

}  // namespace batchlas
