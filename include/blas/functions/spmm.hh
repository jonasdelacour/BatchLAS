#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T, MatrixFormat F>
using spmm = Event(Queue&,
                   const MatrixView<T, F>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Transpose, Transpose, Span<std::byte>);

template <typename T, MatrixFormat F>
using spmm_buffer_size = size_t(Queue&,
                                const MatrixView<T, F>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                const MatrixView<T, MatrixFormat::Dense>&,
                                T, T, Transpose, Transpose);
}  // namespace sig


template <Backend B, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx,
    const MatrixView<T, MFormat>& A,
    const MatrixView<T, MatrixFormat::Dense>& descrB,
    const MatrixView<T, MatrixFormat::Dense>& descrC,
    T alpha,
    T beta,
    Transpose transA,
    Transpose transB,
    Span<std::byte> workspace);

template <Backend B, typename T, MatrixFormat MFormat>
inline Event spmm(Queue& ctx,
        const Matrix<T, MFormat>& A,
        const Matrix<T, MatrixFormat::Dense>& Bmat,
        const Matrix<T, MatrixFormat::Dense>& Cmat,
        T alpha,
        T beta,
        Transpose transA,
        Transpose transB,
        Span<std::byte> workspace) {
        return spmm<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB, workspace);
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB);

template <Backend B, typename T, MatrixFormat MFormat>
inline size_t spmm_buffer_size(Queue& ctx,
                                                const Matrix<T, MFormat>& A,
                                                const Matrix<T, MatrixFormat::Dense>& Bmat,
                                                const Matrix<T, MatrixFormat::Dense>& Cmat,
                                                T alpha,
                                                T beta,
                                                Transpose transA,
                                                Transpose transB) {
        return spmm_buffer_size<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB);
}

}  // namespace batchlas
