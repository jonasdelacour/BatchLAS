#ifndef BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
#define BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

#include <util/sycl-device-queue.hh>
#include <blas/matrix_handle_new.hh>
#include <util/sycl-span.hh>
#include <complex>
#include <sycl/sycl.hpp>

namespace batchlas {

// Forward declarations for cuBLAS backend functions using MatrixView

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

} // namespace batchlas

#endif // BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
