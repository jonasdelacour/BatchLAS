#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

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
