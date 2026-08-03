#pragma once

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declaration below.
namespace sig {
template <typename T>
using gemm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Transpose, Transpose, ComputePrecision);
}  // namespace sig

template <Backend Back, typename T>
Event gemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           ComputePrecision precision = ComputePrecision::Default);

template <Backend Back, typename T>
inline Event gemm(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision = ComputePrecision::Default) {
        return gemm<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB, precision);
}

template <Backend Back, typename T>
inline Event gemm_heterogeneous(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                T alpha,
                                T beta,
                                Transpose transA,
                                Transpose transB,
                                ComputePrecision precision = ComputePrecision::Default) {
        return gemm<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
}

template <Backend Back, typename T>
inline Event gemm_heterogeneous(Queue& ctx,
                                const Matrix<T, MatrixFormat::Dense>& A,
                                const Matrix<T, MatrixFormat::Dense>& B,
                                const Matrix<T, MatrixFormat::Dense>& C,
                                T alpha,
                                T beta,
                                Transpose transA,
                                Transpose transB,
                                ComputePrecision precision = ComputePrecision::Default) {
        return gemm_heterogeneous<Back, T>(ctx,
                                           MatrixView<T, MatrixFormat::Dense>(A),
                                           MatrixView<T, MatrixFormat::Dense>(B),
                                           MatrixView<T, MatrixFormat::Dense>(C),
                                           alpha,
                                           beta,
                                           transA,
                                           transB,
                                           precision);
}

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(gemm)
BATCHLAS_DISPATCH_ON_QUEUE(gemm_heterogeneous)

}  // namespace batchlas
