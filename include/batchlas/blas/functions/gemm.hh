#pragma once

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

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

// There is no separate gemm_heterogeneous entry point. `gemm` handles a
// heterogeneous batch -- one where the items carry differing active_rows /
// active_cols -- natively on every backend: each of them tests
// gemm_has_heterogeneous_batch(A, B, C) and routes accordingly. The alias that
// used to live here forwarded to `gemm` with an unchanged argument list, so the
// only thing the second name added was the impression that plain `gemm` did not
// support heterogeneous batches.
//
// (The Python binding keeps a `gemm_heterogeneous` name, and that one is not
// redundant: it coerces a list of differently-shaped arrays, which `gemm` does
// not.)

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(gemm)

}  // namespace batchlas
