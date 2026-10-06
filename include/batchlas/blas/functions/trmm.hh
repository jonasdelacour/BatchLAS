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
using trmm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, Side, Uplo, Transpose, Diag);

// Spelled out, not aliased to sig::trmm: a vendor parameter order may differ
// from the public one (trsm's alpha is last).
template <typename T>
using trmm_vendor = Event(Queue&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          const MatrixView<T, MatrixFormat::Dense>&,
                          T,
                          Side,
                          Uplo,
                          Transpose,
                          Diag);
}  // namespace sig


/// @brief Batched triangular matrix-matrix multiply, out of place.
///
/// For every batch item computes
/// \f[ C := \alpha \, \mathrm{op}(A) \, B \quad (\texttt{Side::Left}), \qquad
///     C := \alpha \, B \, \mathrm{op}(A) \quad (\texttt{Side::Right}) \f]
/// with `A` triangular (`uplo`), unit or non-unit diagonal (`diag`), and
/// \f$\mathrm{op}(A)\f$ one of \f$A, A^T, A^H\f$. `B` and `C` are m x n; `A` is
/// m x m (Left) or n x n (Right). Under `Diag::Unit` the stored diagonal of `A`
/// is not read.
///
/// Unlike reference BLAS `?trmm`, which overwrites `B`, this writes the product
/// into `C` and leaves `B` unmodified. trsm, by contrast, is in place.
/// Also callable as `trmm(ctx, A, B, C, TrmmOptions<T>{...})`, with owning
/// `Matrix` arguments, and without `Ba` (taken from `ctx.backend()`).
///
/// @tparam Ba  backend the call is compiled for; must match `ctx`'s device
/// @tparam T   scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param ctx     queue the work is enqueued on
/// @param A       batch of triangular matrices; only the `uplo` triangle is read
/// @param B       batch of m x n input matrices; not modified
/// @param C       batch of m x n output matrices; overwritten with the product
/// @param alpha   scale of the product
/// @param side    whether op(A) multiplies from the left or the right
/// @param uplo    which triangle of `A` holds the data
/// @param transA  op() applied to `A`
/// @param diag    whether `A` has an implicit unit diagonal
/// @return event of the last enqueued kernel; `C` is valid once it completes
/// @pre All operands have the same batch size and conforming shapes per item.
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Ba`,
///         unless the call is `Backend::CUDA` on a GPU, with homogeneous operands,
///         batch <= 65535 and a shape the `triangular` tiles or the `expand`
///         scratch admit (@ref md_docs_2perf_2level3).
/// @see trsm, TrmmOptions, @ref md_docs_2cpp-api
/// @ingroup blas3
template <Backend Ba, typename T>
BATCHLAS_API Event trmm(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             T alpha,
                             Side side,
                             Uplo uplo,
                             Transpose transA,
                             Diag diag);

}  // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor TU defines and instantiates it for its Backend.
// The public trmm is defined in src/ops/trmm/trmm.cc, outside every vendor TU.
// evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of trmm (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::trmm calls it. Same arguments and semantics.
/// @ingroup dispatch
template <Backend Back, typename T>
BATCHLAS_API Event trmm_vendor(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& A,
                               const MatrixView<T, MatrixFormat::Dense>& B,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               T alpha,
                               Side side,
                               Uplo uplo,
                               Transpose transA,
                               Diag diag);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(trmm)

BATCHLAS_DISPATCH_ON_QUEUE(trmm)

}  // namespace batchlas
