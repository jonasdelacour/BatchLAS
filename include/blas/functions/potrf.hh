#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    Uplo uplo);

template <Backend B, typename T>
Event potrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        Uplo uplo,
        Span<std::byte> workspace);

template <Backend B, typename T>
inline size_t potrf_buffer_size(Queue& ctx,
                                        const Matrix<T, MatrixFormat::Dense>& A,
                                        Uplo uplo) {
        return potrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), uplo);
}

template <Backend B, typename T>
inline Event potrf(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace) {
        return potrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), uplo, workspace);
}

}  // namespace batchlas
