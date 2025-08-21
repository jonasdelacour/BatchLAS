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

// Forwarding overload for owning matrices (A, B, C)
template <Backend B, typename T, MatrixFormat MFormat>
inline Event spmm(Queue& ctx,
        const Matrix<T, MFormat>& A,
        const Matrix<T, MatrixFormat::Dense>& Bmat,
        const Matrix<T, MatrixFormat::Dense>& Cmat,
        T alpha,
        T beta,
        Transpose transA,
        Transpose transB,
        Span<std::byte> workspace) {
        return spmm<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB, workspace);
}


template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB);

// Forwarding overload for buffer size (owning matrices)
template <Backend B, typename T, MatrixFormat MFormat>
inline size_t spmm_buffer_size(Queue& ctx,
                                                const Matrix<T, MFormat>& A,
                                                const Matrix<T, MatrixFormat::Dense>& Bmat,
                                                const Matrix<T, MatrixFormat::Dense>& Cmat,
                                                T alpha,
                                                T beta,
                                                Transpose transA,
                                                Transpose transB) {
        return spmm_buffer_size<B,T,MFormat>(ctx, MatrixView<T,MFormat>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB);
}

template <Backend Ba, typename T>
Event trmm(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                const MatrixView<T, MatrixFormat::Dense>& C,
                T alpha,
                Side side,
                Uplo uplo,
                Transpose transA,
                Diag diag);

template <Backend Ba, typename T>
inline Event trmm(Queue& ctx,
                                 const Matrix<T, MatrixFormat::Dense>& A,
                                 const Matrix<T, MatrixFormat::Dense>& Bmat,
                                 const Matrix<T, MatrixFormat::Dense>& Cmat,
                                 T alpha,
                                 Side side,
                                 Uplo uplo,
                                 Transpose transA,
                                 Diag diag) {
        return trmm<Ba,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, side, uplo, transA, diag);
}

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

template <Backend Back, typename T>
inline Event gemm(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   const Matrix<T, MatrixFormat::Dense>& Cmat,
                   T alpha,
                   T beta,
                   Transpose transA,
                   Transpose transB,
                   ComputePrecision precision = ComputePrecision::Default) {
        return gemm<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), MatrixView<T, MatrixFormat::Dense>(Cmat), alpha, beta, transA, transB, precision);
}

template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA);

// Forwarding overload to allow passing owning Vector<T> directly
template <Backend B, typename T>
inline Event gemv(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const Vector<T>& X,
            const Vector<T>& Y,
            T alpha,
            T beta,
            Transpose transA) {
    return gemv<B, T>(ctx, A,
                         static_cast<VectorView<T>>(X),
                         static_cast<VectorView<T>>(Y),
                         alpha, beta, transA);
}

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

template <Backend Back, typename T>
Event getrs(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Transpose transA,
           Span<int64_t> pivots,
           Span<std::byte> work_space);

template <Backend Back, typename T>
inline Event getrs(Queue& ctx,
                   const Matrix<T, MatrixFormat::Dense>& A,
                   const Matrix<T, MatrixFormat::Dense>& Bmat,
                   Transpose transA,
                   Span<int64_t> pivots,
                   Span<std::byte> work_space) {
        return getrs<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA, pivots, work_space);
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& B,
                         Transpose transA);

template <Backend Back, typename T>
inline size_t getrs_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 const Matrix<T, MatrixFormat::Dense>& Bmat,
                                                 Transpose transA) {
        return getrs_buffer_size<Back,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Bmat), transA);
}

template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event getrf(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), pivots, work_space);
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getrf_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C, //C is overwritten with inverse of A
            Span<int64_t> pivots,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event getri(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        const Matrix<T, MatrixFormat::Dense>& Cmat,
                        Span<int64_t> pivots,
                        Span<std::byte> work_space) {
        return getri<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), pivots, work_space);
}

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A);

template <Backend B, typename T>
inline size_t getri_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A) {
        return getri_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A));
}

template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
            Span<T> tau,
            Span<std::byte> work_space);

template <Backend B, typename T>
inline Event geqrf(Queue& ctx,
                        const Matrix<T,MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> work_space) {
        return geqrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, work_space);
}

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t geqrf_buffer_size(Queue& ctx,
                                                 const Matrix<T,MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return geqrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A, //A overwritten with Q
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
inline Event orgqr(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        Span<T> tau,
                        Span<std::byte> workspace) {
        return orgqr<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau, workspace);
}

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t orgqr_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 Span<T> tau) {
        return orgqr_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), tau);
}

template <Backend B, typename T>
Event ormqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Side side,
            Transpose trans,
            Span<T> tau,
            Span<std::byte> workspace);

template <Backend B, typename T>
inline Event ormqr(Queue& ctx,
                        const Matrix<T, MatrixFormat::Dense>& A,
                        const Matrix<T, MatrixFormat::Dense>& Cmat,
                        Side side,
                        Transpose trans,
                        Span<T> tau,
                        Span<std::byte> workspace) {
        return ormqr<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), side, trans, tau, workspace);
}

template <Backend B, typename T>
size_t ormqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau);

template <Backend B, typename T>
inline size_t ormqr_buffer_size(Queue& ctx,
                                                 const Matrix<T, MatrixFormat::Dense>& A,
                                                 const Matrix<T, MatrixFormat::Dense>& Cmat,
                                                 Side side,
                                                 Transpose trans,
                                                 Span<T> tau) {
        return ormqr_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), MatrixView<T, MatrixFormat::Dense>(Cmat), side, trans, tau);
}


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
inline size_t potrf_buffer_size(Queue& ctx,
                                        const Matrix<T, MatrixFormat::Dense>& A,
                                        Uplo uplo) {
        return potrf_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), uplo);
}
template <Backend B, typename T>
inline Event potrf(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace) {
        return potrf<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), uplo, workspace);
}

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

template <Backend B, typename T>
inline Event syev(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& descrA,
                Span<typename base_type<T>::type> eigenvalues,
                JobType jobtype,
                Uplo uplo,
                Span<std::byte> workspace) {
        return syev<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(descrA), eigenvalues, jobtype, uplo, workspace);
}
template <Backend B, typename T>
inline size_t syev_buffer_size(Queue& ctx,
                const Matrix<T, MatrixFormat::Dense>& A,
                Span<typename base_type<T>::type> eigenvalues,
                JobType jobtype,
                Uplo uplo) {
        return syev_buffer_size<B,T>(ctx, MatrixView<T, MatrixFormat::Dense>(A), eigenvalues, jobtype, uplo);
}


} // namespace batchlas

#endif // BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
