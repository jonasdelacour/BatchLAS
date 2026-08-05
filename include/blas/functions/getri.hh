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
using getri = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Span<int64_t>, Span<std::byte>);

template <typename T>
using getri_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&);
}  // namespace sig


template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event getri(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        const Matrix<T, MatrixFormat::Dense>& Cmat,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getri<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), pivots, work_space);
}

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getri_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getri_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(getri)
BATCHLAS_DISPATCH_ON_QUEUE(getri_buffer_size)

}  // namespace batchlas
