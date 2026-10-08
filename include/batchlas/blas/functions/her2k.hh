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
using her2k = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    T, float_t<T>, Uplo, Transpose);

// Spelled out, not aliased to sig::her2k: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using her2k_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          float_t<T>,
                          Uplo,
                          Transpose);
}  // namespace sig


/// @brief Batched Hermitian rank-2k update.
///
/// For every batch item computes, with `A` and `B` n x k (`NoTrans`) or k x n
/// (`ConjTrans`) and `C` Hermitian n x n,
/// \f[ C := \alpha A B^H + \bar\alpha B A^H + \beta C \quad (\texttt{NoTrans}) \f]
/// \f[ C := \alpha A^H B + \bar\alpha B^H A + \beta C \quad (\texttt{ConjTrans}) \f]
/// Only the triangle of `C` named by `uplo` is written; the other is left as it
/// was (use `MatrixView::hermitize` to mirror it). The diagonal comes out real.
///
/// The second term is the conjugate transpose of the first, not a copy with the
/// operands swapped; that is what makes the sum Hermitian for any complex
/// `alpha`, and is the whole difference from syr2k. `beta` scales an
/// already-Hermitian `C` and so is real. Also callable as
/// `her2k(ctx, A, B, C, Her2kOptions<T>{...})`, with owning `Matrix` arguments,
/// and without `Ba` (taken from `ctx.backend()`).
///
/// Not table-selected. On `Backend::CUDA` the cuBLAS backend chooses by a fixed rule between one
/// strided-batched gemm into scratch that carries both terms, folded into the `uplo` triangle,
/// and a per-item `cublas?her2k` loop (the loop also serves a scratch that does not fit);
/// `BATCHLAS_EXPAND_ROUTE` = `expand` | `loop` pins it. The host backend runs the per-item
/// `cblas_?her2k` loop.
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `std::complex<float>` or `std::complex<double>`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of n x k (NoTrans) or k x n (ConjTrans) matrices; not modified
/// @param B       batch with the same shape as `A`; not modified
/// @param C       batch of n x n matrices; the `uplo` triangle is updated in place
/// @param alpha   complex scale of \f$A B^H\f$ (its conjugate scales the mirror term)
/// @param beta    real scale of the input `C`
/// @param uplo    which triangle of `C` is written
/// @param transA  `Transpose::NoTrans` or `Transpose::ConjTrans`
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item.
/// @throws std::invalid_argument if the shapes do not conform (`Backend::CUDA`; the host
///         backend checks inside its deferred host task and throws std::runtime_error there)
/// @throws batchlas::NoRouteError in a build without the vendor BLAS
///         for `Ba`: her2k has no native implementation.
/// @note Not instantiated for `Backend::ROCM`.
/// @see syr2k, herk, Her2kOptions, @ref md_docs_2cpp-api, @ref md_docs_2perf_2level3 (the rule's crossover)
/// @ingroup api_blas3
template <Backend Ba, ComplexScalar T>
BATCHLAS_API Event her2k(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         T alpha,
                         float_t<T> beta,
                         Uplo uplo,
                         Transpose transA);

} // namespace batchlas


namespace batchlas::backend {

// DECLARATION ONLY: each vendor TU defines and instantiates it for its Backend;
// the public her2k is defined in src/ops/level3/level3.cc.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of her2k (cuBLAS, host BLAS).
///
/// Not an entry point: batchlas::her2k calls it. Same arguments and semantics.
/// @ingroup api_dispatch
template <Backend Back, ComplexScalar T>
BATCHLAS_API Event her2k_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& B,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                T alpha,
                                float_t<T> beta,
                                Uplo uplo,
                                Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(her2k)

BATCHLAS_DISPATCH_ON_QUEUE(her2k)

}  // namespace batchlas
