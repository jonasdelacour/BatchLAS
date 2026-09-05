#pragma once

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
}  // namespace sig


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
                throw std::invalid_argument("TRSM: Matrix dimensions cannot be negative (m=" + std::to_string(m) + 
                                          ", n=" + std::to_string(n) + ")");
        }

        if (transA != Transpose::NoTrans && transA != Transpose::Trans && transA != Transpose::ConjTrans) {
                throw std::invalid_argument("TRSM: Invalid transpose operation: " + std::to_string(static_cast<int>(transA)));
        }
        if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
                throw std::invalid_argument("TRSM: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
        }
        if (side != Side::Left && side != Side::Right) {
                throw std::invalid_argument("TRSM: Invalid side parameter: " + std::to_string(static_cast<int>(side)));
        }
        if (diag != Diag::NonUnit && diag != Diag::Unit) {
                throw std::invalid_argument("TRSM: Invalid diag parameter: " + std::to_string(static_cast<int>(diag)));
        }

        if (side == Side::Left) {
                if (A.rows() != m || A.cols() != m) {
                        throw std::invalid_argument("TRSM: For left side, A must be square matrix of size m x m. Got " + 
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + 
                                                " instead of " + std::to_string(m) + "x" + std::to_string(m));
                }
                if (lda < std::max(1, m)) {
                        throw std::invalid_argument("TRSM: lda must be >= max(1, m). Got lda=" + 
                                                std::to_string(lda) + ", m=" + std::to_string(m));
                }
        } else {
                if (A.rows() != n || A.cols() != n) {
                        throw std::invalid_argument("TRSM: For right side, A must be square matrix of size n x n. Got " + 
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + 
                                                " instead of " + std::to_string(n) + "x" + std::to_string(n));
                }
                if (lda < std::max(1, n)) {
                        throw std::invalid_argument("TRSM: lda must be >= max(1, n). Got lda=" + 
                                                std::to_string(lda) + ", n=" + std::to_string(n));
                }
        }

        if (ldb < std::max(1, m)) {
                throw std::invalid_argument("TRSM: ldb must be >= max(1, m). Got ldb=" + 
                                        std::to_string(ldb) + ", m=" + std::to_string(m));
        }
}

// alpha sits in position 4, immediately after the matrices, to match trmm (see
// functions/trmm.hh). It used to come last here, so the two triangular routines
// disagreed on where the scalar went and only one of them could be written from
// memory; the deleted overloads below turn the old spelling into a diagnostic
// rather than leaving it to be rediscovered.
template <Backend Back, typename T>
Event trsm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag);

template <Backend Back, typename T>
inline Event trsm(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   T alpha,
                   Side side,
                   Uplo uplo,
                   Transpose transA,
                   Diag diag) {
        return trsm<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), alpha, side, uplo, transA, diag);
}

// Tombstones for the pre-reorder argument order. Side/Uplo/Transpose/Diag are
// all enum class, so nothing implicitly converts to or from T and a stale call
// could never have silently compiled into a wrong answer -- but without these
// the error would be "no matching function", which does not say what changed.
// Both spellings need one: deleting only the MatrixView overload would leave a
// Matrix-argument call binding to the new order with alpha where side belongs.
template <Backend Back, typename T>
Event trsm(Queue&,
           const MatrixView<T, MatrixFormat::Dense>&,
           const MatrixView<T, MatrixFormat::Dense>&,
           Side, Uplo, Transpose, Diag, T) = delete;

template <Backend Back, typename T>
Event trsm(Queue&,
           const Matrix<T, MatrixFormat::Dense>&,
           const Matrix<T, MatrixFormat::Dense>&,
           Side, Uplo, Transpose, Diag, T) = delete;

}  // namespace batchlas

namespace batchlas {

// Backend-deducing overloads: `f(ctx, ...)` uses ctx.backend().
// See BATCHLAS_DISPATCH_ON_QUEUE in blas/queue-dispatch.hh.

BATCHLAS_DISPATCH_ON_QUEUE(trsm)

}  // namespace batchlas
