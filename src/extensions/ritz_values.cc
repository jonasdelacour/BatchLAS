#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

namespace batchlas {
    
    template <Backend B, typename T, MatrixFormat MFormat>
    struct RitzValuesKernel {};

    /**
     * @brief Computes the Ritz values given a matrix and trial vectors
     * 
     * Ritz values are approximations to eigenvalues computed from the Rayleigh quotient:
     * For each column v_j of V: ritz_value[j] = (v_j^T * A * v_j) / (v_j^T * v_j)
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix (can be sparse or dense)
     * @param V Trial vectors (dense matrix, columns are trial eigenvectors)
     * @param ritz_vals Output vector for Ritz values
     * @param workspace Pre-allocated workspace buffer
     * @return Event Event to track operation completion
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    Event ritz_values(Queue& ctx,
                      const MatrixView<T, MFormat>& A,
                      const MatrixView<T, MatrixFormat::Dense>& V,
                      const VectorView<typename base_type<T>::type>& ritz_vals,
                      Span<std::byte> workspace) {
        using real_t = typename base_type<T>::type;
        
        auto pool = BumpAllocator(workspace);
        auto batch_size = A.batch_size();
        auto n = A.rows();
        auto k = V.cols(); // Number of trial vectors
        
        assert(A.cols() == n && "Matrix A must be square");
        assert(V.rows() == n && "V must have same number of rows as A");
        assert(V.batch_size() == batch_size && "V must have same batch size as A");
        assert(ritz_vals.size() >= k && "ritz_vals must have at least k elements");
        assert(ritz_vals.batch_size() == batch_size && "ritz_vals must have same batch size as A");
        
        // Allocate workspace for A*V computation
        auto AV_mem = pool.allocate<T>(ctx, n * k * batch_size);
        auto AV = MatrixView<T, MatrixFormat::Dense>(AV_mem.data(), n, k, n, n * k, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        
        Event last_event;
        
        // Compute AV = A * V using spmm for sparse or gemm for dense
        if constexpr (MFormat == MatrixFormat::Dense) {
            last_event = gemm<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        } else {
            auto spmm_ws = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            last_event = spmm<B>(ctx, A, V, AV, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_ws);
        }
        
        // Compute Ritz values as (v_j^T * (A*v_j)) / (v_j^T * v_j) for each column j
        auto real_part = [](T value) { 
            if constexpr (sycl::detail::is_complex<T>::value) return value.real(); 
            else return value; 
        };
        
        last_event = ctx.submit([&](sycl::handler& h) {
            h.depends_on(last_event);
            auto V_span = V.data();
            auto AV_span = AV.data();
            auto ritz_span = ritz_vals.data();
            auto V_stride = V.stride();
            auto AV_stride = AV.stride();
            auto ritz_stride = ritz_vals.stride();
            auto V_ld = V.ld();
            auto AV_ld = AV.ld();
            
            h.parallel_for<RitzValuesKernel<B, T, MFormat>>(
                sycl::nd_range<2>(
                    sycl::range<2>(batch_size, k),
                    sycl::range<2>(1, 1)
                ),
                [=](sycl::nd_item<2> item) {
                    int b = item.get_global_id(0);
                    int j = item.get_global_id(1);
                    
                    // Compute v_j^T * (A*v_j)
                    T numerator = T(0);
                    for (int i = 0; i < n; ++i) {
                        T v_ij = V_span[b * V_stride + j * V_ld + i];
                        T av_ij = AV_span[b * AV_stride + j * AV_ld + i];
                        if constexpr (sycl::detail::is_complex<T>::value) {
                            numerator += sycl::conj(v_ij) * av_ij;
                        } else {
                            numerator += v_ij * av_ij;
                        }
                    }
                    
                    // Compute v_j^T * v_j
                    T denominator = T(0);
                    for (int i = 0; i < n; ++i) {
                        T v_ij = V_span[b * V_stride + j * V_ld + i];
                        if constexpr (sycl::detail::is_complex<T>::value) {
                            denominator += sycl::conj(v_ij) * v_ij;
                        } else {
                            denominator += v_ij * v_ij;
                        }
                    }
                    
                    // Store Ritz value
                    ritz_span[b * ritz_stride + j] = real_part(numerator / denominator);
                }
            );
        });
        
        return last_event;
    }

    /**
     * @brief Computes the required workspace size for ritz_values
     * 
     * @param ctx Execution context/device queue
     * @param A Matrix (can be sparse or dense)
     * @param V Trial vectors (dense matrix)
     * @param ritz_vals Output vector for Ritz values
     * @return size_t Required workspace size in bytes
     */
    template <Backend B, typename T, MatrixFormat MFormat>
    size_t ritz_values_workspace(Queue& ctx,
                                 const MatrixView<T, MFormat>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& V,
                                 const VectorView<typename base_type<T>::type>& ritz_vals) {
        auto batch_size = A.batch_size();
        auto n = A.rows();
        auto k = V.cols();
        
        size_t size = 0;
        
        // Space for A*V
        size += BumpAllocator::allocation_size<T>(ctx, n * k * batch_size);
        size += BumpAllocator::allocation_size<T*>(ctx, batch_size);
        
        // Additional space for spmm if A is sparse
        if constexpr (MFormat != MatrixFormat::Dense) {
            // Create temporary views for spmm_buffer_size calculation
            auto AV_temp = MatrixView<T, MatrixFormat::Dense>(nullptr, n, k, n, n * k, batch_size, nullptr);
            size += spmm_buffer_size<B>(ctx, A, V, AV_temp, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        }
        
        return size;
    }

    // Explicit template instantiations for common types
    #define RITZ_VALUES_INSTANTIATE(back, fp, fmt) \
        template Event ritz_values<back, fp, fmt>(Queue&, const MatrixView<fp, fmt>&, const MatrixView<fp, MatrixFormat::Dense>&, const VectorView<typename base_type<fp>::type>&, Span<std::byte>); \
        template size_t ritz_values_workspace<back, fp, fmt>(Queue&, const MatrixView<fp, fmt>&, const MatrixView<fp, MatrixFormat::Dense>&, const VectorView<typename base_type<fp>::type>&);

    #define RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
        RITZ_VALUES_INSTANTIATE(back, fp, MatrixFormat::Dense) \
        RITZ_VALUES_INSTANTIATE(back, fp, MatrixFormat::CSR)

    #define RITZ_VALUES_INSTANTIATE_FOR_BACKEND(back) \
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, float) \
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, double) \
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, std::complex<float>) \
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND_TYPE(back, std::complex<double>)

    #if BATCHLAS_HAS_HOST_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
    #endif

    #if BATCHLAS_HAS_GPU_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::SYCL)
    #endif

    #if BATCHLAS_HAS_CUDA_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
    #endif

    #if BATCHLAS_HAS_ROCM_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
    #endif

    #if BATCHLAS_HAS_MKL_BACKEND
        RITZ_VALUES_INSTANTIATE_FOR_BACKEND(Backend::MKL)
    #endif
}
