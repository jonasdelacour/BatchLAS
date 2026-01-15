#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

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
