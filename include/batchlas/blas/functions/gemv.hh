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
using gemv = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const VectorView<T>&,
                   const VectorView<T>&,
                   T, T, Transpose);

// Spelled out, not aliased to sig::gemv: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using gemv_vendor = Event(Queue&,
                          const MatrixView<T,MatrixFormat::Dense>&,
                          const VectorView<T>&,
                          const VectorView<T>&,
                          T,
                          T,
                          Transpose);
}  // namespace sig


/// @brief Batched general matrix-vector multiply.
///
/// For every batch item \f$b\f$ computes
/// \f[ y_b := \alpha \, \mathrm{op}(A_b) \, x_b + \beta \, y_b \f]
/// where \f$\mathrm{op}(A)\f$ is \f$A\f$, \f$A^T\f$ or \f$A^H\f$ according to
/// `transA`. For an m x n `A`, `X` has length n and `Y` length m under
/// `NoTrans`, and the other way round under `Trans` / `ConjTrans`. The vectors'
/// `inc` (element spacing) and `stride` (batch spacing) are honoured.
///
/// Also callable as `gemv(ctx, A, x, y, GemvOptions<T>{...})`, with owning
/// `Matrix` / `Vector` arguments, and without `B` (taken from `ctx.backend()`).
///
/// @tparam B  backend the call is compiled for; must match `ctx`'s device
/// @tparam T  scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of m x n matrices
/// @param X       batch of input vectors
/// @param Y       batch of output vectors; input scaled by `beta`, overwritten with the result
/// @param alpha   scale of the product
/// @param beta    scale of the input `Y`
/// @param transA  op() applied to `A`
/// @return event of the last enqueued kernel; `Y` is valid once it completes
/// @pre `A`, `X` and `Y` have the same batch size and conforming lengths.
/// @throws batchlas::dispatch::NoRouteError in a build without the vendor BLAS
///         for `B` when the native kernels do not support the call (for example a
///         heterogeneous `A`).
/// @note No argument validation is done up front: a shape the native kernels
///       refuse goes to the vendor library, which reports it.
/// @see GemvOptions, @ref md_docs_2cpp-api
/// @ingroup blas2
template <Backend B, typename T>
BATCHLAS_API Event gemv(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const VectorView<T>& X,
                        const VectorView<T>& Y,
                        T alpha,
                        T beta,
                        Transpose transA);

}  // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of gemv (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::gemv calls it when it routes to the vendor.
/// Same arguments and semantics as batchlas::gemv.
/// @ingroup dispatch
template <Backend B, typename T>
BATCHLAS_API Event gemv_vendor(Queue& ctx,
                               const MatrixView<T,MatrixFormat::Dense>& A,
                               const VectorView<T>& X,
                               const VectorView<T>& Y,
                               T alpha,
                               T beta,
                               Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(gemv)

BATCHLAS_DISPATCH_ON_QUEUE(gemv)

}  // namespace batchlas
