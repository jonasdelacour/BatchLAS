#pragma once

#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using syr2k = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    T, T, Uplo, Transpose);

// Spelled out, not aliased to sig::syr2k: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using syr2k_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          T,
                          Uplo,
                          Transpose);
}  // namespace sig


/// @brief Batched symmetric rank-2k update.
///
/// For every batch item computes, with `A` and `B` n x k (`NoTrans`) or k x n
/// (`Trans`) and `C` symmetric n x n,
/// \f[ C := \alpha A B^T + \alpha B A^T + \beta C \quad (\texttt{NoTrans}) \f]
/// \f[ C := \alpha A^T B + \alpha B^T A + \beta C \quad (\texttt{Trans}) \f]
/// Only the triangle of `C` named by `uplo` is written; the other is left as it
/// was (use `MatrixView::symmetrize` to mirror it).
///
/// Constrained to real `T`; the complex spelling is her2k. Also callable as
/// `syr2k(ctx, A, B, C, Syr2kOptions<T>{...})`, with owning `Matrix` arguments,
/// and without `Ba` (taken from `ctx.backend()`).
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `float` or `double`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of n x k (NoTrans) or k x n (Trans) matrices; not modified
/// @param B       batch with the same shape as `A`; not modified
/// @param C       batch of n x n matrices; the `uplo` triangle is updated in place
/// @param alpha   scale of both products
/// @param beta    scale of the input `C`
/// @param uplo    which triangle of `C` is written
/// @param transA  `Transpose::NoTrans` or `Transpose::Trans`, applied to `A` and `B`
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item; k > 0.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Ba`,
///         unless the call is `Backend::CUDA` on a GPU, `float`, not `ConjTrans`,
///         with homogeneous operands and batch <= 65535 (the `triangular` tiles;
///         @ref md_docs_2perf_2level3).
/// @see her2k, syrk, Syr2kOptions, @ref md_docs_2cpp-api
/// @ingroup blas3
template <Backend Ba, RealScalar T>
BATCHLAS_API Event syr2k(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         T alpha,
                         T beta,
                         Uplo uplo,
                         Transpose transA);

} // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// The public syr2k is defined in src/ops/syr2k/syr2k.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of syr2k (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::syr2k calls it. Same arguments and semantics.
/// @ingroup dispatch
template <Backend Back, RealScalar T>
BATCHLAS_API Event syr2k_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                T alpha,
                                T beta,
                                Uplo uplo,
                                Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(syr2k)

BATCHLAS_DISPATCH_ON_QUEUE(syr2k)

}  // namespace batchlas
