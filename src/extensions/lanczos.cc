//Implementation file for Lanczos algorithm
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/dpl/random>
#include <oneapi/dpl/algorithm>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

namespace batchlas {
    template <Backend B, typename T, MatrixFormat MF>
    struct InitKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    struct LanczosKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    struct ApplyReflectionsKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    struct QTQKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    struct DiagonalizeKernel {};

    template <Backend B, typename T, MatrixFormat MF>
    Event lanczos(Queue& ctx,
                const MatrixView<T, MF>& A,
                Span<typename base_type<T>::type> W, //Output eigenvalues
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V, //Output eigenvectors for jobz == JobType::EigenVectors
                const LanczosParams<T>& params) {
        using real_t = typename base_type<T>::type;

        auto pool = BumpAllocator(workspace);

        auto batch_size = A.batch_size();
        auto n = A.rows();

        auto real_part = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return value.real(); else return value; };
        auto Vmem = pool.allocate<T>(ctx, (1+n)*n*batch_size);
        auto V_vectormem = pool.allocate<T>(ctx, n*2*batch_size);
        auto alphas = pool.allocate<T>(ctx, n*batch_size);
        auto betas = pool.allocate<T>(ctx, n*batch_size);
        
        //Batched SpMV is not supported so we have to represent our vector as a dense matrix padded with an extra column (to prevent fallback to SpMV)
        auto padded_output = MatrixView(V_vectormem.data(), n, 2, n, 2*n, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto padded_vector = MatrixView(Vmem.data(), n, 2, n, (n+1)*n, batch_size);
        
        Span<std::byte> spmm_buffer;
        if constexpr (!(MF == MatrixFormat::Dense)){
            spmm_buffer = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, padded_vector, padded_output, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans));
        }
        auto ortho_buffer = pool.allocate<std::byte>(ctx, ortho_buffer_size<B>(ctx, MatrixView(Vmem.data(), n, 2, n, (n+1)*n, batch_size), 
            MatrixView(Vmem.data(), n, n-2, n, (n+1)*n, batch_size), Transpose::NoTrans, Transpose::NoTrans, params.ortho_algorithm, params.ortho_iterations));

        auto basis_ptr_mem = pool.allocate<T*>(ctx, batch_size);
        auto vector_view_ptr_mem = pool.allocate<T*>(ctx, batch_size);

        
        auto wg_init = std::min(size_t(n), get_kernel_max_wg_size<InitKernel<B, T, MF>>(ctx));
        auto wg_lanczos = std::min(size_t(n), get_kernel_max_wg_size<LanczosKernel<B, T, MF>>(ctx));
        
        //Initial guess
        auto init = [&](){
            ctx -> submit([&](sycl::handler& h) {
                auto reduce_mem = sycl::local_accessor<T>(n, h);
                h.parallel_for<InitKernel<B, T, MF>>(sycl::nd_range<1>(batch_size*wg_init, wg_init), [=](sycl::nd_item<1> item) {
                    auto bid = item.get_group_linear_id();
                    auto tid = item.get_local_linear_id();
                    auto bdim = item.get_local_range()[0];
                    auto cta = item.get_group();
                    
                    oneapi::dpl::uniform_real_distribution<typename base_type<T>::type> distr(0.0, 1.0);            
                    oneapi::dpl::minstd_rand engine(42, tid);

                    auto localV = Vmem.subspan(bid*(n+1)*n, n);

                    for (int i = tid; i < n; i += bdim) {
                        localV[i] = distr(engine); //Random initialization of the first Ritz vector
                        reduce_mem[i] = localV[i] * localV[i];
                    }
                    auto norm = sycl::sqrt(real_part(sycl::joint_reduce(cta, reduce_mem.begin(), reduce_mem.end(), T(0), sycl::plus<T>())));
                    for (int i = tid; i < n; i += bdim) {
                        localV[i] /= norm; //Normalize the Ritz vector
                    }
                });
            });
        };

        auto lanczos_iteration = [&](const int iterations){
            for (int it = 0; it < iterations; it++) {
                if (it > 0) {
                    auto basis_view = MatrixView(Vmem.data(), n, (it-1), n, (n+1)*n, batch_size, basis_ptr_mem.data());
                    auto vector_view = MatrixView(Vmem.data() + (it-1)*n, n, 2, n, (n+1)*n, batch_size, vector_view_ptr_mem.data());
                    if((it % params.reorthogonalization_iterations == 0) || (it == iterations - 1)) {
                        ortho<B>(ctx, vector_view, basis_view, Transpose::NoTrans, Transpose::NoTrans, ortho_buffer, params.ortho_algorithm, params.ortho_iterations);
                    }
                }

                auto padded_vector = MatrixView(Vmem.data() + it*n, n, 2, n, (n+1)*n, batch_size);
                auto spmm_start = std::chrono::steady_clock::now();
                if constexpr (!(MF == MatrixFormat::Dense)) {
                    spmm<B>(ctx, A, padded_vector, padded_output, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans, spmm_buffer);
                } else {
                    gemm<B>(ctx, A, padded_vector, padded_output, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
                }
                ctx -> submit([&](sycl::handler& h) {
                    auto v_prev_ptr = Vmem.data() + (std::max(it-1,0))*n;
                    auto v_current_ptr = Vmem.data() + it*n;
                    auto v_next_ptr = padded_output.data_ptr();
                    auto dot_mem = sycl::local_accessor<T>(n, h);
                    h.parallel_for<LanczosKernel<B, T, MF>>(sycl::nd_range<1>(batch_size*wg_lanczos, wg_lanczos), [=](sycl::nd_item<1> item) {
                        auto bid = item.get_group_linear_id();
                        auto tid = item.get_local_linear_id();
                        auto wg = item.get_local_range()[0];
                        auto cta = item.get_group();

                        auto local_v_prev =     Span(v_prev_ptr +       bid*(n+1)*n,n);
                        auto local_v_current =  Span(v_current_ptr +    bid*(n+1)*n,n);
                        auto local_v_next =     Span(v_next_ptr +       bid*2*n,n);
                        auto local_output =     Span(Vmem.data() + bid*(n+1)*n + (it+1)*n, n);
                    
                        auto localAlphas = alphas.subspan(bid*n, n);
                        auto localBetas = betas.subspan(bid*n, n);

                        for (int i = tid; i < n; i += wg) {
                            dot_mem[i] = local_v_current[i] * local_v_next[i];
                        }
                        auto alpha = sycl::joint_reduce(cta, dot_mem.begin(), dot_mem.end(), T(0), sycl::plus<T>());
                        localAlphas[it] = alpha;
                        if (it > 0) {
                            for (int i = tid; i < n; i += wg) {
                                local_v_next[i] -= localBetas[it - 1] * local_v_prev[i] + alpha * local_v_current[i];
                            }
                        } else {
                            for (int i = tid; i < n; i += wg) {
                                local_v_next[i] -= alpha * local_v_current[i];
                            }
                        }
                        for (int i = tid; i < n; i += wg) {
                            dot_mem[i] = local_v_next[i] * local_v_next[i];
                        }
                        auto beta = sycl::sqrt(real_part(sycl::joint_reduce(cta, dot_mem.begin(), dot_mem.end(), T(0), sycl::plus<T>())));
                        localBetas[it] = beta;
                        
                        if( it < iterations - 1) {
                            for (size_t i = tid; i < n; i += wg) {
                                local_output[i] = local_v_next[i] / beta; //Normalize the Ritz vector
                            }
                        }
                    });
                });
            }
        };

        init();
        lanczos_iteration(n);

        //Allocate workspace and compute eigenvalues/eigenvectors of the tridiagonal matrix
        auto Q_eigenvectors = pool.allocate<T>(ctx, jobz == JobType::EigenVectors ? n * n * batch_size : 0);
        auto qr_workspace = pool.allocate<std::byte>(ctx, tridiagonal_solver_buffer_size<B,T>(ctx, n, batch_size, jobz));

        tridiagonal_solver<B>(ctx,
                alphas,
                betas,
                W,
                qr_workspace,
                jobz,
                MatrixView<T, MatrixFormat::Dense>(Q_eigenvectors.data(), n, n, n, n*n, batch_size),
                n,
                batch_size);

        if (jobz == JobType::EigenVectors) {
            gemm<B>(ctx,
                    MatrixView(Vmem.data(), n, n, n, (n+1)*n, batch_size),
                    MatrixView(Q_eigenvectors.data(), n, n, n, n*n, batch_size),
                    V,
                    T(1), T(0),
                    Transpose::NoTrans, Transpose::NoTrans);
        }

        //Sort the eigenvalues and eigenvectors using sycl::experimental::joint_sort
        if (params.sort_enabled){
        ctx->submit([&](sycl::handler& h) {
            // Calculate memory required for joint_sort operation
            // We need more memory for larger arrays
            size_t sort_mem_size = sycl::ext::oneapi::experimental::default_sorters::joint_sorter<std::less<>>::memory_required<T>(
                    sycl::memory_scope::work_group,  n);
            
            sycl::local_accessor<std::byte, 1> scratch(sycl::range<1>(sort_mem_size), h);
            sycl::local_accessor<T, 1> temp_eigenvalues(n, h);
            
            // Allocate global memory for indices array used during sorting
            auto indices_mem = pool.allocate<int>(ctx, n * batch_size);
            auto temp_vec_mem = pool.allocate<T>(ctx, n * batch_size);
            
            auto Vstride = V.stride();
            auto Vdata = V.data_ptr();
            auto Vld = V.ld();

            h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto tid = item.get_local_linear_id();
                auto cta = item.get_group();
                
                // Set up pointers to the batch-specific data
                typename base_type<T>::type* batch_W = W.data() + bid * n;
                int* batch_indices = indices_mem.data() + bid * n;
                T* batch_temp_vec = temp_vec_mem.data() + bid * n;
                
                // Create group helper with scratch memory
                auto group_helper = sycl::ext::oneapi::experimental::group_with_scratchpad(
                    cta, sycl::span<std::byte>(scratch.get_pointer(), scratch.size()));
                
                if (jobz == JobType::EigenVectors) {
                    // Initialize indices - each thread handles a portion
                    for (int i = tid; i < n; i += item.get_local_range()[0]) {
                        batch_indices[i] = i;
                    }
                    
                    sycl::group_barrier(cta);
                    
                    // Use joint_sort with a custom comparator that also swaps indices
                    // Only one thread needs to start the sort
                    
                    sycl::ext::oneapi::experimental::joint_sort(
                        group_helper, 
                        batch_indices, 
                        batch_indices + n,
                        [batch_W, batch_indices, params](auto idx_a, auto idx_b) {
                            // Sort by value in ascending order
                            if (params.sort_order == SortOrder::Descending) {
                                return batch_W[idx_a] > batch_W[idx_b];
                            } else {
                                return batch_W[idx_a] < batch_W[idx_b];
                            }
                        }
                    );

                    for (int i = tid; i < n; i += item.get_local_range()[0]) {
                        temp_eigenvalues[i] = batch_W[batch_indices[i]];
                    }
                    sycl::group_barrier(cta);
                    
                    for (int i = tid; i < n; i += item.get_local_range()[0]) {
                        batch_W[i] = temp_eigenvalues[i];
                    }
                    
                    
                    
                    // Reorder eigenvectors based on the sorted indices
                    T* batch_V_out = const_cast<T*>(Vdata) + bid * Vstride;
                    
                    // Reorder the eigenvectors column by column
                    for (int i = 0; i < n; i++) {
                        // Each thread handles some rows of the eigenvector
                        for (int j = tid; j < n; j += item.get_local_range()[0]) {
                            // Store in temp buffer
                            batch_temp_vec[j] = batch_V_out[batch_indices[i] * n + j];
                        }
                        
                        sycl::group_barrier(cta);
                        
                        // Copy from temp buffer to output
                        for (int j = tid; j < n; j += item.get_local_range()[0]) {
                            batch_V_out[i * Vld + j] = batch_temp_vec[j];
                        }
                        
                        sycl::group_barrier(cta);
                    }
                } else {
                    // If only eigenvalues are needed, just sort them directly
                    sycl::ext::oneapi::experimental::joint_sort(
                        group_helper, 
                        batch_W, 
                        batch_W + n,
                        [params](auto a, auto b) {
                            // Sort by value in ascending order
                            if (params.sort_order == SortOrder::Descending) {
                                return a > b;
                            } else {
                                return a < b;
                            }
                        }
                    );
                    
                }
            });
        });
        }

        return ctx.get_event();
    }

    template <Backend B, typename T, MatrixFormat MF>
    size_t lanczos_buffer_size(
        Queue& ctx,
        const MatrixView<T, MF>& A,
        Span<typename base_type<T>::type> W, //Output eigenvalues
        JobType jobz,
        const MatrixView<T, MatrixFormat::Dense>& V, //Output eigenvectors for jobz == JobType::EigenVectors
        const LanczosParams<T>& params
    ) {
        auto n = A.rows();
        auto batch_size = A.batch_size();
        auto padded_vector_view = MatrixView(V.data_ptr(), n, 2, n, (n+1)*n, batch_size);
        auto padded_output_vector_view = MatrixView(V.data_ptr(), n, 2, n, (2)*n, batch_size);
        auto single_vector_view = MatrixView(V.data_ptr(), n, 1, n, (n+1)*n, batch_size);
        auto subspace_view = MatrixView(V.data_ptr(), n, n - 1, n, (n+1)*n, batch_size);

        // Basic memory required for Lanczos iteration
        size_t basic_size = BumpAllocator::allocation_size<T>(ctx, (n+1)*n*batch_size) + 
                          BumpAllocator::allocation_size<T>(ctx, n*2*batch_size) +
                          BumpAllocator::allocation_size<T>(ctx, n*batch_size) +    // alphas
                          BumpAllocator::allocation_size<T>(ctx, n*batch_size) +    // betas
                          BumpAllocator::allocation_size<T*>(ctx, batch_size) * 3 +
                          BumpAllocator::allocation_size<std::byte>(ctx, ortho_buffer_size<B>(ctx, padded_vector_view, subspace_view, Transpose::NoTrans, Transpose::NoTrans, params.ortho_algorithm, params.ortho_iterations));
        if constexpr (!(MF == MatrixFormat::Dense)){
            basic_size += BumpAllocator::allocation_size<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, padded_vector_view, padded_output_vector_view, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans));
        }
        // Additional memory required for the tridiagonal solver
        basic_size += BumpAllocator::allocation_size<std::byte>(ctx,
                        tridiagonal_solver_buffer_size<B,T>(ctx, n, batch_size, jobz));

        // Eigenvector memory (only if needed)
        size_t eigenvectors_size = (jobz == JobType::EigenVectors) ? 
                                 BumpAllocator::allocation_size<T>(ctx, n*n*batch_size) : 
                                 BumpAllocator::allocation_size<T>(ctx, n*batch_size);
        
        size_t sort_size = BumpAllocator::allocation_size<int>(ctx, n*batch_size) + 
                           BumpAllocator::allocation_size<T>(ctx, n*batch_size);
        return basic_size  + eigenvectors_size + sort_size;
    }

    #define LANCZOS_INSTANTIATE(back, fp, mf) \
    template Event lanczos<back, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        Span<std::byte>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \
    template size_t lanczos_buffer_size<back, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \
    
    #define LANCZOS_INSANTIATE_FOR_BACKEND(back)\
        LANCZOS_INSTANTIATE(back, float, MatrixFormat::CSR)\
        LANCZOS_INSTANTIATE(back, float, MatrixFormat::Dense)\
        LANCZOS_INSTANTIATE(back, double, MatrixFormat::CSR)\
        LANCZOS_INSTANTIATE(back, double, MatrixFormat::Dense)

    #if BATCHLAS_HAS_CUDA_BACKEND
        LANCZOS_INSANTIATE_FOR_BACKEND(Backend::CUDA)
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        LANCZOS_INSANTIATE_FOR_BACKEND(Backend::ROCM)
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND
        LANCZOS_INSANTIATE_FOR_BACKEND(Backend::NETLIB)
    #endif

    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<float>)
    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef LANCZOS_INSTANTIATE
    #undef LANCZOS_INSTANTIATE_FOR_FP


}