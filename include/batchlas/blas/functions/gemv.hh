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
/// `transA`. For an \f$m \times n\f$ `A`, `X` has length \f$n\f$ and `Y` length \f$m\f$ under
/// `NoTrans`, and the other way round under `Trans` / `ConjTrans`. The vectors'
/// `inc` (element spacing) and `stride` (batch spacing) are honoured.
///
/// Also callable as `gemv(ctx, A, x, y, GemvOptions<T>{...})`, with owning
/// `Matrix` / `Vector` arguments, and without `B` (taken from `ctx.backend()`).
///
/// @tparam B  backend the call is compiled for; must match `ctx`'s device
/// @tparam T  scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of \f$m \times n\f$ matrices \f$A_b\f$; read only
/// @param[in]     X       batch of vectors \f$x_b\f$, length \f$n\f$ under `NoTrans`, else \f$m\f$; read only
/// @param[in,out] Y       batch of vectors \f$y_b\f$, length \f$m\f$ under `NoTrans`, else \f$n\f$;
///                        read and scaled by \f$\beta\f$ (not read when \f$\beta = 0\f$), then overwritten with the result
/// @param[in]     alpha   scalar \f$\alpha\f$ multiplying \f$\mathrm{op}(A_b) \, x_b\f$
/// @param[in]     beta    scalar \f$\beta\f$ multiplying the input \f$y_b\f$
/// @param[in]     transA  selects \f$\mathrm{op}(A)\f$: \f$A\f$, \f$A^T\f$ or \f$A^H\f$
/// @return event of the last enqueued kernel; `Y` is valid once it completes
/// @pre `A`, `X` and `Y` have the same batch size and conforming lengths.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `B`
///         when no native kernel can run the call (for example a heterogeneous `A`).
/// @note No argument validation is done up front: a shape the native kernels
///       refuse goes to the vendor library, which reports it.
/// @see GemvOptions, @ref md_docs_2cpp-api
/// @ingroup api_blas2
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

// DECLARATION ONLY: each vendor TU defines and instantiates it for its Backend;
// the public gemv is defined in src/ops/gemv/gemv.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of gemv (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::gemv calls it when it selects the `vendor`
/// kernel family. Same arguments and semantics as batchlas::gemv:
/// \f$y_b := \alpha \, \mathrm{op}(A_b) \, x_b + \beta \, y_b\f$.
/// @tparam B  vendor backend this definition is compiled for
/// @tparam T  scalar type
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of \f$m \times n\f$ matrices \f$A_b\f$; read only
/// @param[in]     X       batch of vectors \f$x_b\f$, length \f$n\f$ under `NoTrans`, else \f$m\f$; read only
/// @param[in,out] Y       batch of vectors \f$y_b\f$, length \f$m\f$ under `NoTrans`, else \f$n\f$;
///                        read and scaled by \f$\beta\f$ (not read when \f$\beta = 0\f$), then overwritten with the result
/// @param[in]     alpha   scalar \f$\alpha\f$ multiplying \f$\mathrm{op}(A_b) \, x_b\f$
/// @param[in]     beta    scalar \f$\beta\f$ multiplying the input \f$y_b\f$
/// @param[in]     transA  selects \f$\mathrm{op}(A)\f$: \f$A\f$, \f$A^T\f$ or \f$A^H\f$
/// @return event of the last enqueued vendor call; `Y` is valid once it completes
/// @ingroup api_dispatch
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


BATCHLAS_DISPATCH_ON_QUEUE(gemv)

}  // namespace batchlas
