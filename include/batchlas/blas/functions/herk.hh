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
using herk = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   float_t<T>, float_t<T>, Uplo, Transpose);

// Spelled out, not aliased to sig::herk: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using herk_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          float_t<T>,
                          float_t<T>,
                          Uplo,
                          Transpose);
}  // namespace sig


/// @brief Batched Hermitian rank-k update.
///
/// For every batch item computes
/// \f[ C := \alpha A A^H + \beta C \quad (\texttt{NoTrans},\ A \text{ is } n \times k), \qquad
///     C := \alpha A^H A + \beta C \quad (\texttt{ConjTrans},\ A \text{ is } k \times n) \f]
/// with `C` Hermitian n x n. Only the triangle of `C` named by `uplo` is written;
/// the other triangle is left exactly as it was (use `MatrixView::hermitize` to
/// mirror it). The diagonal comes out real: an imaginary part on `C`'s diagonal
/// is neither read nor produced.
///
/// `alpha` and `beta` are real (`float_t<T>`), as in BLAS `?herk`: a complex
/// alpha would make \f$\alpha A A^H\f$ non-Hermitian. Constrained to complex `T`;
/// the real spelling is syrk. Also callable as
/// `herk(ctx, A, C, HerkOptions<T>{...})`, with owning `Matrix` arguments, and
/// without `Ba` (taken from `ctx.backend()`).
///
/// Not table-selected. On `Backend::CUDA` the cuBLAS backend chooses by a fixed rule between one
/// strided-batched gemm into scratch folded into the `uplo` triangle and a per-item
/// `cublas?herk` loop (the loop also serves a scratch that does not fit);
/// `BATCHLAS_EXPAND_ROUTE` = `expand` | `loop` pins it. `BATCHLAS_SYRK_ROUTE=gram` opts a GPU
/// call into syrk's conjugating Gram tile kernel where it admits the shape (measured slower; kept
/// for testing). The host backend runs the per-item `cblas_?herk` loop.
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   `std::complex<float>` or `std::complex<double>`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of n x k (NoTrans) or k x n (ConjTrans) matrices; not modified
/// @param C       batch of n x n matrices; the `uplo` triangle is updated in place
/// @param alpha   real scale of the product
/// @param beta    real scale of the input `C`
/// @param uplo    which triangle of `C` is written
/// @param transA  `Transpose::NoTrans` or `Transpose::ConjTrans`
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre `A` and `C` have the same batch size and conforming shapes per item.
/// @throws std::invalid_argument if the shapes do not conform (`Backend::CUDA`; the host
///         backend checks inside its deferred host task and throws std::runtime_error there)
/// @throws batchlas::NoRouteError in a build without the vendor BLAS
///         for `Ba`: herk has no native implementation.
/// @note Not instantiated for `Backend::ROCM`.
/// @see syrk, her2k, HerkOptions, @ref md_docs_2cpp-api, @ref md_docs_2perf_2level3 (the rule's crossover)
/// @ingroup api_blas3
template <Backend Ba, ComplexScalar T>
BATCHLAS_API Event herk(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        float_t<T> alpha,
                        float_t<T> beta,
                        Uplo uplo,
                        Transpose transA);

} // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// The public herk is defined in src/ops/level3/level3.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of herk (cuBLAS, host BLAS).
///
/// Not an entry point: batchlas::herk calls it. Same arguments and semantics.
/// @ingroup api_dispatch
template <Backend Back, ComplexScalar T>
BATCHLAS_API Event herk_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               float_t<T> alpha,
                               float_t<T> beta,
                               Uplo uplo,
                               Transpose transA);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(herk)

BATCHLAS_DISPATCH_ON_QUEUE(herk)

}  // namespace batchlas
