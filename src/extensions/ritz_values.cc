#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <util/kernel-heuristics.hh>

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
        // using work-group parallelism with SYCL reductions for better performance
        auto real_part = [](T value) { 
            if constexpr (sycl::detail::is_complex<T>::value) return value.real(); 
            else return value; 
        };
        
        // Determine work-group size for dot product reductions
        size_t wg_size = std::min(size_t(256), ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
        wg_size = std::min(wg_size, size_t(n)); // Don't use more threads than vector length
        
        auto [global_size, local_size, use_grid_stride] = compute_batched_nd_range_sizes(
            batch_size * k * n, ctx.device(), KernelType::ELEMENTWISE, batch_size, n);

        last_event = ctx -> submit([&](sycl::handler& h) {
            h.depends_on(*last_event);
            
            // Create VectorViews for accessing columns with proper stride/increment handling
            auto V_data = V.data();
            auto AV_data = AV.data();
            
            // Allocate local memory for partial sums
            auto dot_mem = sycl::local_accessor<T>(wg_size, h);
            auto AV_view = AV.kernel_view();
            auto V_view = V.kernel_view();
            
            h.parallel_for<RitzValuesKernel<B, T, MFormat>>(
                sycl::nd_range<1>(global_size, local_size),
                [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group(0); // Which batch*k combination
                    auto tid = item.get_local_id(0);    // Thread within work-group
                    auto wg = item.get_local_range(0);  // Work-group size
                    auto cta = item.get_group();        // Work-group for reductions
                    
                    //Grid stride loop:
                    for (size_t group_id = bid; group_id < batch_size * k; group_id += item.get_group_range(0)) {
                    // Decompose group_id into batch and trial vector indices
                    int b = group_id / k;
                    int j = group_id % k;
                    
                    // Compute v_j^T * (A*v_j) using work-group reduction
                    // Each thread computes partial sum for its assigned elements
                    T numerator_partial = T(0);
                    for (int i = tid; i < n; i += wg) {
                        T v_i = V_view(i, j, b);
                        T av_i = AV_view(i, j, b);
                        if constexpr (sycl::detail::is_complex<T>::value) {
                            numerator_partial += std::conj(v_i) * av_i;
                        } else {
                            numerator_partial += v_i * av_i;
                        }
                    }
                    dot_mem[tid] = numerator_partial;
                    sycl::group_barrier(cta);
                    
                    // Reduce across work-group
                    T numerator = sycl::joint_reduce(cta, dot_mem.begin(), dot_mem.begin() + wg, T(0), sycl::plus<T>());
                    
                    // Compute v_j^T * v_j using work-group reduction
                    T denominator_partial = T(0);
                    for (int i = tid; i < n; i += wg) {
                        T v_i = V_view(i, j, b);
                        if constexpr (sycl::detail::is_complex<T>::value) {
                            denominator_partial += std::conj(v_i) * v_i;
                        } else {
                            denominator_partial += v_i * v_i;
                        }
                    }
                    dot_mem[tid] = denominator_partial;
                    sycl::group_barrier(cta);
                    
                    // Reduce across work-group
                    T denominator = sycl::joint_reduce(cta, dot_mem.begin(), dot_mem.begin() + wg, T(0), sycl::plus<T>());
                    
                    // Only the first thread in the work-group writes the result
                    if (tid == 0) {
                        ritz_vals(j, b) = real_part(numerator / denominator);
                    }
                }
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
