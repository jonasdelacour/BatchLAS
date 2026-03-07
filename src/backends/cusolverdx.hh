#pragma once

#include <cstddef>

#include <blas/extensions.hh>
#include <blas/functions/syev.hh>

#if BATCHLAS_HAS_CUDA_BACKEND
    #include <cuda_runtime_api.h>
#endif

namespace batchlas::backend::cusolverdx {

template <typename T>
Event heev(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           Span<typename base_type<T>::type> eigenvalues,
           JobType jobz,
           Uplo uplo,
           Span<std::byte> workspace);

template <typename T>
size_t heev_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues,
                        JobType jobz,
                        Uplo uplo);

template <typename T>
Event htev(Queue& ctx,
           const VectorView<T>& d,
           const VectorView<T>& e,
           const VectorView<T>& eigenvalues,
           JobType jobz,
           const MatrixView<T, MatrixFormat::Dense>& eigvects,
           Span<std::byte> workspace,
           Uplo uplo = Uplo::Lower);

template <typename T>
size_t htev_buffer_size(Queue& ctx,
                        const VectorView<T>& d,
                        const VectorView<T>& e,
                        JobType jobz,
                        Uplo uplo = Uplo::Lower);

#if BATCHLAS_HAS_CUDA_BACKEND
namespace cuda_kernels {

bool available();
bool heev_supported_n(int n);
bool htev_supported_n(int n);

cudaError_t heev_launch_float(float* A,
                              int n,
                              int lda,
                              float* lambda,
                              int* info,
                              int batches,
                              bool compute_vectors,
                              bool lower,
                              cudaStream_t stream);

cudaError_t heev_launch_double(double* A,
                               int n,
                               int lda,
                               double* lambda,
                               int* info,
                               int batches,
                               bool compute_vectors,
                               bool lower,
                               cudaStream_t stream);

cudaError_t htev_launch_float(float* d,
                              int n,
                              float* e,
                              float* V,
                              int ldv,
                              int* info,
                              int batches,
                              bool compute_vectors,
                              cudaStream_t stream);

cudaError_t htev_launch_double(double* d,
                               int n,
                               double* e,
                               double* V,
                               int ldv,
                               int* info,
                               int batches,
                               bool compute_vectors,
                               cudaStream_t stream);

} // namespace cuda_kernels
#endif

} // namespace batchlas::backend::cusolverdx
