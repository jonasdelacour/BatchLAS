#pragma once

#include <stdexcept>
#include <string>
#include <algorithm>

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <blas/enums.hh>

namespace batchlas {

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
                throw std::runtime_error("TRSM: Matrix dimensions cannot be negative (m=" + std::to_string(m) + 
                                          ", n=" + std::to_string(n) + ")");
        }

        if (transA != Transpose::NoTrans && transA != Transpose::Trans && transA != Transpose::ConjTrans) {
                throw std::runtime_error("TRSM: Invalid transpose operation: " + std::to_string(static_cast<int>(transA)));
        }
        if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
                throw std::runtime_error("TRSM: Invalid uplo parameter: " + std::to_string(static_cast<int>(uplo)));
        }
        if (side != Side::Left && side != Side::Right) {
                throw std::runtime_error("TRSM: Invalid side parameter: " + std::to_string(static_cast<int>(side)));
        }
        if (diag != Diag::NonUnit && diag != Diag::Unit) {
                throw std::runtime_error("TRSM: Invalid diag parameter: " + std::to_string(static_cast<int>(diag)));
        }

        if (side == Side::Left) {
                if (A.rows() != m || A.cols() != m) {
                        throw std::runtime_error("TRSM: For left side, A must be square matrix of size m x m. Got " + 
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + 
                                                " instead of " + std::to_string(m) + "x" + std::to_string(m));
                }
                if (lda < std::max(1, m)) {
                        throw std::runtime_error("TRSM: lda must be >= max(1, m). Got lda=" + 
                                                std::to_string(lda) + ", m=" + std::to_string(m));
                }
        } else {
                if (A.rows() != n || A.cols() != n) {
                        throw std::runtime_error("TRSM: For right side, A must be square matrix of size n x n. Got " + 
                                                std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + 
                                                " instead of " + std::to_string(n) + "x" + std::to_string(n));
                }
                if (lda < std::max(1, n)) {
                        throw std::runtime_error("TRSM: lda must be >= max(1, n). Got lda=" + 
                                                std::to_string(lda) + ", n=" + std::to_string(n));
                }
        }

        if (ldb < std::max(1, m)) {
                throw std::runtime_error("TRSM: ldb must be >= max(1, m). Got ldb=" + 
                                        std::to_string(ldb) + ", m=" + std::to_string(m));
        }
}

template <Backend Back, typename T>
Event trsm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag,
           T alpha);

template <Backend Back, typename T>
inline Event trsm(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   Side side,
                   Uplo uplo,
                   Transpose transA,
                   Diag diag,
                   T alpha) {
        return trsm<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), side, uplo, transA, diag, alpha);
}

}  // namespace batchlas
