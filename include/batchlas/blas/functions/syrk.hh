// SPDX-License-Identifier: MIT
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
/// with `C` symmetric \f$n \times n\f$. Only the triangle of `C` named by `uplo` is written;
/// the other triangle is left exactly as it was, uninitialised memory included
/// (use `MatrixView::symmetrize` to mirror it).
///
/// Real or complex-symmetric `T` (no conjugate); the Hermitian spelling is herk. Also callable as
/// `syrk(ctx, A, C, SyrkOptions<T>{...})`, with owning `Matrix` arguments, and
/// without `Ba` (taken from `ctx.backend()`).
///
/// Each call takes a kernel family from the tuned table for its shape (@ref selection_tables):
/// `gram` (one tile covers `C`, \f$n \le 128\f$), `triangular` (128-wide tiles of the
/// requested half) or `vendor` (the library's per-item `?syrk` loop). Pin one with
/// `BATCHLAS_SYRK_ROUTE` = `auto` | `native` | `vendor` | `gram` | `triangular`.
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `float`, `double`, or complex: complex-symmetric \f$AA^T\f$ (no conjugate; that is herk),
///             with `Transpose::NoTrans` or `Transpose::Trans` only
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of \f$n \times k\f$ (`NoTrans`) or \f$k \times n\f$ (`Trans`) matrices;
///                        not modified
/// @param[in,out] C       batch of symmetric \f$n \times n\f$ matrices; the `uplo` triangle is read
///                        scaled by \f$\beta\f$ and overwritten with the result, the other is untouched
/// @param[in]     alpha   scale \f$\alpha\f$ of the product
/// @param[in]     beta    scale \f$\beta\f$ of the input `C`
/// @param[in]     uplo    which triangle of `C` is read and written
/// @param[in]     transA  `Transpose::NoTrans` or `Transpose::Trans` (`ConjTrans` is accepted and means `Trans`)
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre `A` and `C` have the same batch size and conforming shapes per item.
/// @throws std::invalid_argument if the shapes do not conform, or if `BATCHLAS_SYRK_ROUTE` names
///         a family that is not compiled for `T` or cannot run this call (`native` and `vendor`
///         instead fall back to the tuned choice with a warning)
/// @throws std::runtime_error if no family can run the call in a build with the vendor BLAS
///         (a heterogeneous batch: every family takes one \f$(n, k)\f$ per launch)
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Ba`,
///         unless the call is on a GPU (`Backend::CUDA` or `Backend::SYCL`) with homogeneous
///         operands (the `triangular` tiles serve every such shape; @ref md_docs_2perf_2level3).
/// @see herk, syr2k, SyrkOptions, @ref md_docs_2cpp-api
/// @ingroup api_blas3
template <Backend Ba, FloatingOrComplexScalar T>
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
// The public syrk is defined in src/ops/syrk/syrk.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of syrk (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::syrk calls it. Same arguments and semantics.
/// @tparam Back  backend whose vendor library runs the call
/// @tparam T     `float`, `double`, or complex (complex-symmetric, no conjugate)
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of \f$n \times k\f$ (`NoTrans`) or \f$k \times n\f$ (`Trans`) matrices;
///                        not modified
/// @param[in,out] C       batch of symmetric \f$n \times n\f$ matrices; the `uplo` triangle is read
///                        scaled by \f$\beta\f$ and overwritten with the result
/// @param[in]     alpha   scale \f$\alpha\f$ of the product
/// @param[in]     beta    scale \f$\beta\f$ of the input `C`
/// @param[in]     uplo    which triangle of `C` is read and written
/// @param[in]     transA  `Transpose::NoTrans` or `Transpose::Trans` (`ConjTrans` is accepted and means `Trans`)
/// @return event of the last enqueued vendor call; `C` is valid once it completes
/// @ingroup api_dispatch
template <Backend Back, FloatingOrComplexScalar T>
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


BATCHLAS_DISPATCH_ON_QUEUE(syrk)

}  // namespace batchlas
