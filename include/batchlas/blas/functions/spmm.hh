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

// backend::spmm_vendor / _vendor_buffer_size share the public signatures.
template <typename T, MatrixFormat F>
using spmm_vendor = spmm<T, F>;
template <typename T, MatrixFormat F>
using spmm_vendor_buffer_size = spmm_buffer_size<T, F>;
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


namespace batchlas::backend {

// The vendor path for spmm -- declaration only; see the note on gemm_vendor in
// gemm.hh. Unlike the dense ops, spmm carries a MatrixFormat template
// parameter, so its instantiations are hand-written in each vendor TU.
template <Backend B, typename T, MatrixFormat MFormat>
Event spmm_vendor(Queue& ctx,
                  const MatrixView<T, MFormat>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B_mat,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha,
                  T beta,
                  Transpose transA,
                  Transpose transB,
                  Span<std::byte> workspace);

template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_vendor_buffer_size(Queue& ctx,
                               const MatrixView<T, MFormat>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B_mat,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               T beta,
                               Transpose transA,
                               Transpose transB);

}  // namespace batchlas::backend

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(spmm)
BATCHLAS_DISPATCH_ON_QUEUE(spmm_buffer_size)

}  // namespace batchlas
