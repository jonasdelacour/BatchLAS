#ifndef BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
#define BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <util/sycl-span.hh>
#include <complex>
#include <sycl/sycl.hpp>

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx,
    const MatrixView<T, MFormat>& A,
    const MatrixView<T, MatrixFormat::Dense>& descrB,
    const MatrixView<T, MatrixFormat::Dense>& descrC,
    T alpha,
    T beta,
    Transpose transA,
    Transpose transB,
    Span<std::byte> workspace);


template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB);

template <Backend Back, typename T>
Event gemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           ComputePrecision precision = ComputePrecision::Default);

template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA);

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

        // Check for negative dimensions
        if (m < 0 || n < 0) {
                throw std::runtime_error("TRSM: Matrix dimensions cannot be negative (m=" + std::to_string(m) + 
                                          ", n=" + std::to_string(n) + ")");
        }

        // Validate enum parameters
        if (transA != Transpose::NoTrans && transA != Transpose::Trans) {
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

        // Check dimensions and strides
        if (side == Side::Left) {
                // A is m x m
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
                // A is n x n
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
Event getrs(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Transpose transA,
           Span<int64_t> pivots,
           Span<std::byte> work_space);

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Transpose transA);

template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C, //C is overwritten with inverse of A
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
Event geqrf(Queue& ctx,
            MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
            Span<T> tau,
            Span<std::byte> work_space);

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau);


template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    Uplo uplo);
template <Backend B, typename T>
Event potrf(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA,
        Uplo uplo,
        Span<std::byte> workspace);

        template <Backend B, typename T>
Event syev(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& descrA, //A is overwritten with eigenvectors
        Span<typename base_type<T>::type> eigenvalues,
        JobType jobtype,
        Uplo uplo,
        Span<std::byte> workspace);

template <Backend B, typename T>
size_t syev_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        Span<typename base_type<T>::type> eigenvalues,
        JobType jobtype,
        Uplo uplo);


} // namespace batchlas

#endif // BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
