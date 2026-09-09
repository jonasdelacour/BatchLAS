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
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB);

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

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(spmm)
BATCHLAS_ACCEPT_OWNING(spmm_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(spmm)
BATCHLAS_DISPATCH_ON_QUEUE(spmm_buffer_size)

}  // namespace batchlas
