// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/export.hh>
#include <stdexcept>
#include <string>
#include <algorithm>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using trsm = Event(Queue&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   const MatrixView<T, MatrixFormat::Dense>&,
                   T, Side, Uplo, Transpose, Diag);

// NOT an alias for sig::trsm: the vendor signature puts alpha last.
template <typename T>
using trsm_vendor = Event(Queue&,
                          const MatrixView<T,MatrixFormat::Dense>&,
                          const MatrixView<T,MatrixFormat::Dense>&,
                          Side,
                          Uplo,
                          Transpose,
                          Diag,
                          T);
}  // namespace sig


/// @brief Host-side argument check that trsm runs on every backend before kernel selection.
///
/// With \f$m\f$ = `B.rows()` and \f$n\f$ = `B.cols()`, requires \f$m, n \ge 0\f$; valid enum values
/// for `side`, `uplo`, `transA` and `diag`; `A` square of order \f$m\f$ (Left) or \f$n\f$
/// (Right) with `A.ld()` \f$\ge \max(1, \text{order})\f$; `B.ld()` \f$\ge \max(1, m)\f$; and `A` and `B`
/// with the same batch size. Reads only the views' shape metadata, never their data.
/// @param[in] A       triangular operand
/// @param[in] B       right-hand sides
/// @param[in] side    Side::Left or Side::Right
/// @param[in] uplo    Uplo::Lower or Uplo::Upper
/// @param[in] transA  NoTrans, Trans or ConjTrans
/// @param[in] diag    Diag::NonUnit or Diag::Unit
/// @throws batchlas::invalid_argument naming the first violated requirement
/// @ingroup api_blas3_lowlevel
template <typename T>
inline void trsm_validate_params(
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        Side side,
                        Uplo uplo,
                        Transpose transA,
                        Diag diag) {
        int m = B.rows(), n = B.cols();
        int lda = A.ld(), ldb = B.ld();

        if (m < 0 || n < 0) {
                throw batchlas::invalid_argument("trsm: Matrix dimensions cannot be negative (m=" + std::to_string(m) +
                                          ", n=" + std::to_string(n) + ")");
        }

        if (transA != Transpose::NoTrans && transA != Transpose::Trans && transA != Transpose::ConjTrans) {
                throw batchlas::invalid_argument("trsm: Invalid transpose operation: " + std::to_string(static_cast<int>(transA)));
        }
        if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
                throw batchlas::invalid_argument("trsm: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
        }
        if (side != Side::Left && side != Side::Right) {
                throw batchlas::invalid_argument("trsm: Invalid side parameter: " + std::to_string(static_cast<int>(side)));
        }
        if (diag != Diag::NonUnit && diag != Diag::Unit) {
                throw batchlas::invalid_argument("trsm: Invalid diag parameter: " + std::to_string(static_cast<int>(diag)));
        }

        if (side == Side::Left) {
                if (A.rows() != m || A.cols() != m) {
                        throw batchlas::invalid_argument("trsm: For left side, A must be square matrix of size m x m. Got " +
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                                " instead of " + std::to_string(m) + "x" + std::to_string(m));
                }
                if (lda < std::max(1, m)) {
                        throw batchlas::invalid_argument("trsm: lda must be >= max(1, m). Got lda=" +
                                                std::to_string(lda) + ", m=" + std::to_string(m));
                }
        } else {
                if (A.rows() != n || A.cols() != n) {
                        throw batchlas::invalid_argument("trsm: For right side, A must be square matrix of size n x n. Got " +
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) +
                                                " instead of " + std::to_string(n) + "x" + std::to_string(n));
                }
                if (lda < std::max(1, n)) {
                        throw batchlas::invalid_argument("trsm: lda must be >= max(1, n). Got lda=" +
                                                std::to_string(lda) + ", n=" + std::to_string(n));
                }
        }

        if (ldb < std::max(1, m)) {
                throw batchlas::invalid_argument("trsm: ldb must be >= max(1, m). Got ldb=" +
                                        std::to_string(ldb) + ", m=" + std::to_string(m));
        }
        // Every kernel walks one batch count over both operands; a mismatch indexes past one.
        if (A.batch_size() != B.batch_size()) {
                throw batchlas::invalid_argument("trsm: A and B must have the same batch size. Got " +
                                        std::to_string(A.batch_size()) + " and " +
                                        std::to_string(B.batch_size()));
        }
}

// alpha sits right after the matrices, as in trmm; it used to come last.
// evidence: docs/cpp-api.md#trsm-alpha-moved-next-to-the-matrices
/// @brief Batched triangular solve with multiple right-hand sides, in place.
///
/// For every batch item solves
/// \f[ \mathrm{op}(A) \, X = \alpha B \quad (\texttt{Side::Left}), \qquad
///     X \, \mathrm{op}(A) = \alpha B \quad (\texttt{Side::Right}) \f]
/// and overwrites `B` with \f$X\f$. `A` is triangular (`uplo`), unit or
/// non-unit diagonal (`diag`), and \f$\mathrm{op}(A)\f$ is one of
/// \f$A, A^T, A^H\f$. `B` is \f$m \times n\f$; `A` is \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right).
/// No singularity check is made: a zero on a non-unit diagonal gives Inf/NaN.
///
/// Also callable as `trsm(ctx, A, B, TrsmOptions<T>{...})`, with owning
/// `Matrix` arguments, and without `Back` (taken from `ctx.backend()`).
///
/// @tparam Back  backend the call is compiled for; must match `ctx`'s device
/// @tparam T     scalar type: `float`, `double`, `std::complex<float>` or `std::complex<double>`
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of triangular matrices, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right);
///                        only the `uplo` triangle is read
/// @param[in,out] B       batch of \f$m \times n\f$ right-hand sides \f$B\f$; overwritten with the solution \f$X\f$
/// @param[in]     alpha   scale \f$\alpha\f$ of the right-hand side
/// @param[in]     side    whether \f$\mathrm{op}(A)\f$ is applied from the left or the right
/// @param[in]     uplo    which triangle of `A` holds the data
/// @param[in]     transA  \f$\mathrm{op}()\f$ applied to `A`
/// @param[in]     diag    whether `A` has an implicit unit diagonal
/// @return event of the last enqueued kernel; `B` holds \f$X\f$ once it completes
/// @throws batchlas::invalid_argument from trsm_validate_params (bad shape, `ld`,
///         enum value, or `A` and `B` batch sizes that differ), before anything
///         is enqueued
/// @throws batchlas::NoRouteError in a build without the vendor BLAS for `Back`
///         when no native kernel can run the shape
/// @note On `Backend::NETLIB`, \f$\alpha = 0\f$ still reads `B`, so a NaN in `B`
///       survives (known defect 5, @ref md_docs_2design_2known-defects).
/// @see trmm, TrsmOptions, @ref perf_trsm, @ref md_docs_2cpp-api
/// @ingroup api_blas3
template <Backend Back, typename T>
BATCHLAS_API Event trsm(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B,
                        T alpha,
                        Side side,
                        Uplo uplo,
                        Transpose transA,
                        Diag diag);

// Tombstones for the old order; both spellings need one.
// evidence: docs/cpp-api.md#trsm-alpha-moved-next-to-the-matrices
/// @brief Deleted: the old argument order with `alpha` last. Pass `alpha` right
/// after `B`.
/// @ingroup api_blas3
template <Backend Back, typename T>
Event trsm(Queue&,
           const MatrixView<T, MatrixFormat::Dense>&,
           const MatrixView<T, MatrixFormat::Dense>&,
           Side, Uplo, Transpose, Diag, T) = delete;

}  // namespace batchlas


namespace batchlas::backend {

// Declaration only: each vendor wrapper TU defines and instantiates it for its
// own Backend; the public trsm is defined in src/ops/trsm/trsm.cc, so dropping a
// vendor library drops only this path. evidence: docs/design/vendor-independence.md#the-entry-point-facade
/// @brief Vendor-library implementation of trsm (cuBLAS, rocBLAS, host BLAS).
///
/// Not an entry point: batchlas::trsm calls it after validation, when kernel
/// selection picks the Vendor family. Same semantics as batchlas::trsm, but
/// `alpha` is the last argument.
/// @param[in]     ctx     queue the work is enqueued on
/// @param[in]     A       batch of triangular matrices, \f$m \times m\f$ (Left) or \f$n \times n\f$ (Right);
///                        only the `uplo` triangle is read
/// @param[in,out] B       batch of \f$m \times n\f$ right-hand sides \f$B\f$; overwritten with the solution \f$X\f$
/// @param[in]     side    whether \f$\mathrm{op}(A)\f$ is applied from the left or the right
/// @param[in]     uplo    which triangle of `A` holds the data
/// @param[in]     transA  \f$\mathrm{op}()\f$ applied to `A`
/// @param[in]     diag    whether `A` has an implicit unit diagonal
/// @param[in]     alpha   scale \f$\alpha\f$ of the right-hand side
/// @return event of the enqueued work; `B` holds \f$X\f$ once it completes
/// @ingroup api_dispatch
template <Backend Back, typename T>
BATCHLAS_API Event trsm_vendor(Queue& ctx,
                               const MatrixView<T,MatrixFormat::Dense>& A,
                               const MatrixView<T,MatrixFormat::Dense>& B,
                               Side side,
                               Uplo uplo,
                               Transpose transA,
                               Diag diag,
                               T alpha);

}  // namespace batchlas::backend

namespace batchlas {

// Owning-argument (`f(ctx, Matrix, ...)`) and backend-deducing (`f(ctx, ...)`)
// overloads; see blas/queue-dispatch.hh.


BATCHLAS_DISPATCH_ON_QUEUE(trsm)

}  // namespace batchlas
