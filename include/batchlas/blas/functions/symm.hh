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
/// is taken to be its transpose. `B` and `C` are m x n; `A` is m x m (Left) or
/// n x n (Right).
///
/// Constrained to real `T`; the complex spelling is hemm. Also callable as
/// `symm(ctx, A, B, C, SymmOptions<T>{...})`, with owning `Matrix` arguments,
/// and without `Ba` (taken from `ctx.backend()`).
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `float` or `double`
/// @param ctx    queue the work is enqueued on
/// @param A      batch of symmetric matrices; only the `uplo` triangle is read
/// @param B      batch of m x n matrices; not modified
/// @param C      batch of m x n matrices; input scaled by `beta`, overwritten with the result
/// @param alpha  scale of the product
/// @param beta   scale of the input `C`
/// @param side   whether `A` multiplies from the left or the right
/// @param uplo   which triangle of `A` holds the data
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS
///         for `Ba`, unless the call is `Backend::CUDA`, `float` and inside the
///         native kernel's shape window (@ref md_docs_2perf_2level3).
/// @note Not instantiated for `Backend::ROCM`.
/// @see hemm, SymmOptions, @ref md_docs_2cpp-api
/// @ingroup blas3
template <Backend Ba, RealScalar T>
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
// The public symm is defined in src/ops/level3/level3.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of symm (cuBLAS, host BLAS).
///
/// Not an entry point: batchlas::symm calls it. Same arguments and semantics.
/// @ingroup dispatch
template <Backend Back, RealScalar T>
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

BATCHLAS_ACCEPT_OWNING(symm)

BATCHLAS_DISPATCH_ON_QUEUE(symm)

}  // namespace batchlas
