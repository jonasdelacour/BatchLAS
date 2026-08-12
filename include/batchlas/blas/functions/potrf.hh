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
using potrf = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Uplo, Span<std::byte>);

template <typename T>
using potrf_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Uplo);
}  // namespace sig


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

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(potrf)
BATCHLAS_DISPATCH_ON_QUEUE(potrf_buffer_size)

}  // namespace batchlas
