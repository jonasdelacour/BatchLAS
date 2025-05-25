#ifndef BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
#define BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

#include <util/sycl-device-queue.hh>
#include <blas/matrix_handle_new.hh>
#include <util/sycl-span.hh>
#include <complex>
#include <sycl/sycl.hpp>

namespace batchlas {

template <Backend B, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx,
    MatrixView<T, MFormat>& A,
    MatrixView<T, MatrixFormat::Dense>& descrB,
    MatrixView<T, MatrixFormat::Dense>& descrC,
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
           const MatrixView<T,MatrixFormat::Dense>& A,
           const MatrixView<T,MatrixFormat::Dense>& B,
           const MatrixView<T,MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           ComputePrecision precision = ComputePrecision::Default);

template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA);

template <Backend Back, typename T>
Event trsm(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const MatrixView<T,MatrixFormat::Dense>& B,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag,
           T alpha);

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
