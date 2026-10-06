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
using syrk = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Uplo, Transpose);

// Spelled out, not aliased to sig::syrk: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using syrk_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          T,
                          Uplo,
                          Transpose);
}  // namespace sig


/// @brief Batched symmetric rank-k update.
///
/// For every batch item computes
/// \f[ C := \alpha A A^T + \beta C \quad (\texttt{NoTrans},\ A \text{ is } n \times k), \qquad
///     C := \alpha A^T A + \beta C \quad (\texttt{Trans},\ A \text{ is } k \times n) \f]
/// with `C` symmetric n x n. Only the triangle of `C` named by `uplo` is written;
/// the other triangle is left exactly as it was, uninitialised memory included
/// (use `MatrixView::symmetrize` to mirror it).
///
/// Constrained to real `T`; the complex spelling is herk. Also callable as
/// `syrk(ctx, A, C, SyrkOptions<T>{...})`, with owning `Matrix` arguments, and
/// without `Ba` (taken from `ctx.backend()`).
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `float` or `double`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of n x k (NoTrans) or k x n (Trans) matrices; not modified
/// @param C       batch of n x n matrices; the `uplo` triangle is updated in place
/// @param alpha   scale of the product
/// @param beta    scale of the input `C`
/// @param uplo    which triangle of `C` is written
/// @param transA  `Transpose::NoTrans` or `Transpose::Trans`
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre `A` and `C` have the same batch size and conforming shapes per item.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS
///         for `Ba`, unless the call is `Backend::CUDA`, `float` and inside the
///         native kernel's shape window (@ref md_docs_2perf_2level3).
/// @see herk, syr2k, SyrkOptions, @ref md_docs_2cpp-api
/// @ingroup blas3
template <Backend Ba, RealScalar T>
BATCHLAS_API Event syrk(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Uplo uplo,
                        Transpose transA);

} // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// The public syrk is defined in src/ops/level3/level3.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of syrk (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::syrk calls it. Same arguments and semantics.
/// @ingroup dispatch
template <Backend Back, RealScalar T>
BATCHLAS_API Event syrk_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               T beta,
                               Uplo uplo,
                               Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(syrk)

BATCHLAS_DISPATCH_ON_QUEUE(syrk)

}  // namespace batchlas
