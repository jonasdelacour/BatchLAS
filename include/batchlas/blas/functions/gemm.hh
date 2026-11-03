// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
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

// backend::gemm_vendor shares gemm's signature.
template <typename T>
using gemm_vendor = gemm<T>;
}  // namespace sig

/// @brief Batched general matrix-matrix multiply.
///
/// For every batch item \f$b\f$ computes
/// \f[ C_b := \alpha \, \mathrm{op}(A_b) \, \mathrm{op}(B_b) + \beta \, C_b \f]
/// where \f$\mathrm{op}(X)\f$ is \f$X\f$, \f$X^T\f$ or \f$X^H\f$ according to
/// `transA` / `transB`. With \f$\mathrm{op}(A)\f$ of size \f$m \times k\f$ and
/// \f$\mathrm{op}(B)\f$ of size \f$k \times n\f$, `C` is \f$m \times n\f$.
///
/// **Heterogeneous batch.** A batch whose items carry differing
/// `active_rows` / `active_cols` is handled by this same entry point on every
/// backend: an item with \f$m = 0\f$ or \f$n = 0\f$ is skipped, an item with \f$k = 0\f$
/// computes \f$C_b := \beta C_b\f$, and an all-skipped batch still returns a
/// valid Event. There is no separate `gemm_heterogeneous` in C++ (the Python
/// binding's one exists to coerce a list of differently-shaped arrays).
///
/// Also callable as `gemm(ctx, A, B, C, GemmOptions<T>{...})`, with owning
/// `Matrix` arguments, and without `Back` (taken from `ctx.backend()`).
///
/// @tparam Back  backend the call is compiled for; must match `ctx`'s device
/// @tparam T     scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param[in]     ctx        queue the work is enqueued on
/// @param[in]     A          batch of \f$m \times k\f$ (NoTrans) or \f$k \times m\f$ (Trans/ConjTrans) matrices
/// @param[in]     B          batch of \f$k \times n\f$ (NoTrans) or \f$n \times k\f$ (Trans/ConjTrans) matrices
/// @param[in,out] C          batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten with the result
/// @param[in]     alpha      \f$\alpha\f$, scale of the product
/// @param[in]     beta       \f$\beta\f$, scale of the input `C`
/// @param[in]     transA     \f$\mathrm{op}(\cdot)\f$ applied to `A`
/// @param[in]     transB     \f$\mathrm{op}(\cdot)\f$ applied to `B`
/// @param[in]     precision  compute precision; `ComputePrecision::Default` computes in `T`. The scalar's
///                           own precision (F32 for float, F64 for double) is always served; F16/BF16/TF32
///                           on float are permissions, taken by the vendor library and answered at full
///                           precision without one (see ComputePrecision); other pairs are refused.
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre `A`, `B` and `C` have the same batch size and conforming shapes per item.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Back`
///         when no native kernel can run the call (a refused `precision` pair).
///         An empty batch, \f$m = 0\f$ or \f$n = 0\f$ is a no-op, and \f$k = 0\f$ computes \f$C := \beta C\f$;
///         neither reaches selection, so `precision` is not checked there.
/// @see GemmOptions, @ref md_docs_2cpp-api
/// @ingroup api_blas3
template <Backend Back, typename T>
BATCHLAS_API Event gemm(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB,
                        ComputePrecision precision = ComputePrecision::Default);

}  // namespace batchlas

namespace batchlas::backend {

// DECLARATION ONLY: each vendor TU defines and instantiates it for its Backend;
// the public gemm is defined in src/ops/gemm/gemm.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of gemm (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::gemm calls it when it selects the `vendor`
/// kernel family. Same arguments and semantics as batchlas::gemm.
/// @tparam Back  vendor backend whose library serves the call
/// @tparam T     scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param[in]     ctx        queue the work is enqueued on
/// @param[in]     A          batch of \f$m \times k\f$ (NoTrans) or \f$k \times m\f$ (Trans/ConjTrans) matrices
/// @param[in]     B          batch of \f$k \times n\f$ (NoTrans) or \f$n \times k\f$ (Trans/ConjTrans) matrices
/// @param[in,out] C          batch of \f$m \times n\f$ matrices; input scaled by \f$\beta\f$, overwritten with the result
/// @param[in]     alpha      \f$\alpha\f$, scale of the product
/// @param[in]     beta       \f$\beta\f$, scale of the input `C`
/// @param[in]     transA     \f$\mathrm{op}(\cdot)\f$ applied to `A`
/// @param[in]     transB     \f$\mathrm{op}(\cdot)\f$ applied to `B`
/// @param[in]     precision  compute precision, as for batchlas::gemm
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @ingroup api_dispatch
template <Backend Back, typename T>
BATCHLAS_API Event gemm_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               T beta,
                               Transpose transA,
                               Transpose transB,
                               ComputePrecision precision);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(gemm)

}  // namespace batchlas
