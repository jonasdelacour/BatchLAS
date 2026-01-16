#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
inline Event orgqr(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> workspace) {
        return orgqr<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, workspace);
}

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t orgqr_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return orgqr_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

}  // namespace batchlas
