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
using symm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, T, Side, Uplo);

// Spelled out, not aliased to sig::symm: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using symm_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          T,
                          Side,
                          Uplo);
}  // namespace sig


/// @brief Batched symmetric matrix-matrix multiply.
///
/// For every batch item computes
/// \f[ C := \alpha A B + \beta C \quad (\texttt{Side::Left}), \qquad
///     C := \alpha B A + \beta C \quad (\texttt{Side::Right}) \f]
/// with `A` symmetric: only the triangle named by `uplo` is read and the other
/// is taken to be its transpose. `B` and `C` are \f$m \times n\f$; `A` is \f$m \times m\f$ (Left) or
/// \f$n \times n\f$ (Right).
///
/// Real or complex-symmetric `T` (no conjugate); the Hermitian spelling is hemm. Also callable as
/// `symm(ctx, A, B, C, SymmOptions<T>{...})`, with owning `Matrix` arguments,
/// and without `Ba` (taken from `ctx.backend()`).
///
/// Each call takes a kernel family from the tuned table for its shape (@ref selection_tables):
/// `expand` (mirror `A`'s triangle into scratch leased from the queue's arena, then one
/// strided-batched gemm) or `vendor` (the library's per-item `?symm` loop). Pin one with
/// `BATCHLAS_SYMM_ROUTE` = `auto` | `native` | `vendor` | `expand`.
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `float`, `double`, or complex: complex-symmetric A (no conjugate; that is hemm)
/// @param[in]     ctx    queue the work is enqueued on
/// @param[in]     A      batch of symmetric \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right)
///                       matrices; only the `uplo` triangle is read
/// @param[in]     B      batch of \f$m \times n\f$ matrices; not modified
/// @param[in,out] C      batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten
///                       with the result
/// @param[in]     alpha  scale \f$\alpha\f$ of the product
/// @param[in]     beta   scale \f$\beta\f$ of the input `C`
/// @param[in]     side   whether `A` multiplies from the left or the right
/// @param[in]     uplo   which triangle of `A` holds the data
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item.
/// @throws std::invalid_argument if the shapes do not conform, or if `BATCHLAS_SYMM_ROUTE` names
///         a family that is not compiled for `T` or cannot run this call (`native` and `vendor`
///         instead fall back to the tuned choice with a warning)
/// @throws std::runtime_error if no family can run the call in a build with the vendor BLAS
///         (a heterogeneous batch: every family takes one \f$(m, n)\f$ per launch)
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Ba`,
///         unless the call is `Backend::CUDA` or `Backend::ROCM` on a GPU, with homogeneous operands,
///         and one item's expansion scratch that fits (the `expand` choice chunks the batch;
///         @ref md_docs_2perf_2level3).
/// @note `Backend::ROCM` has no vendor symm: its `vendor` choice symmetrizes a copy of `A`
///       and calls gemm, and blocks the host until it finishes.
/// @see hemm, SymmOptions, @ref md_docs_2cpp-api
/// @ingroup api_blas3
template <Backend Ba, FloatingOrComplexScalar T>
BATCHLAS_API Event symm(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Side side,
                        Uplo uplo);

}  // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// The public symm is defined in src/ops/symm/symm.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of symm (cuBLAS, host BLAS).
///
/// Not an entry point: batchlas::symm calls it. Same arguments and semantics.
/// @tparam Back  backend whose vendor library runs the call
/// @tparam T     `float`, `double`, or complex (complex-symmetric `A`, no conjugate)
/// @param[in]     ctx    queue the work is enqueued on
/// @param[in]     A      batch of symmetric \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right)
///                       matrices; only the `uplo` triangle is read
/// @param[in]     B      batch of \f$m \times n\f$ matrices; not modified
/// @param[in,out] C      batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten
///                       with the result
/// @param[in]     alpha  scale \f$\alpha\f$ of the product
/// @param[in]     beta   scale \f$\beta\f$ of the input `C`
/// @param[in]     side   whether `A` multiplies from the left or the right
/// @param[in]     uplo   which triangle of `A` holds the data
/// @return event of the last enqueued vendor call; `C` is valid once it completes
/// @ingroup api_dispatch
template <Backend Back, FloatingOrComplexScalar T>
BATCHLAS_API Event symm_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               T beta,
                               Side side,
                               Uplo uplo);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(symm)

}  // namespace batchlas
