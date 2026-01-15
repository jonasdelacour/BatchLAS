#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

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
