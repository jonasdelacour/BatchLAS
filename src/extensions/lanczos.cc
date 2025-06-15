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

    template <typename T>
    std::array<T,2> eigvalsh2x2(const std::array<T,4> &A){
        auto [a,b,c,d] = A;
        T D = sycl::sqrt(4*b*c+(a-d)*(a-d));
        return {(a+d-D)/2, (a+d+D)/2};
    }

    template <typename T>
    void apply_all_reflections(const sycl::group<1> &cta, const Span<T> V, const int n, const int m, T* Q) {   
        auto tid = cta.get_local_linear_id();
        auto bdim = cta.get_local_range()[0];
        
        for(int k = 0; k < n; k++) {
            const T &v0 = V[2*k], &v1 = V[2*k+1];      
            // Apply Householder reflection
            for(int l = tid; l < m; l += bdim) {
                T &q0 = Q[k*m+l], &q1 = Q[(k+1)*m+l];
                T vTA = q0*v0 + q1*v1;
                q0 -= 2*v0*vTA;
                q1 -= 2*v1*vTA;
            }      
        }
    }

    template <typename T>
    void T_QTQ(sycl::group<1>& cta, const int n, const Span<T> D, const Span<T> L, const Span<T> U, const Span<T> Vout, T shift=0) {
        using real_t = typename base_type<T>::type;
        using coord3d = std::array<T, 3>;
    
        int tix = cta.get_local_linear_id();
        int bdim = cta.get_local_range()[0];
        
        // Find max norm for numerical stability
        T local_max = T(0.);
        for (int i = tix; i < n; i += bdim){
            local_max = std::max(local_max, std::abs(D[i]) + 2*std::abs(L[i]));
        }
        T max_norm = sycl::reduce_over_group(cta, local_max, sycl::maximum<T>());
        T numerical_zero = 10*std::numeric_limits<T>::epsilon();
        T d_n, l_n, l_nm1;
        
        d_n = D[n]; l_n = L[n]; l_nm1 = L[n-1];
        
        sycl::group_barrier(cta);
        
        real_t a[2], v[2];
        for(int k = tix; k < n + 1; k += bdim){
            D[k] -= shift;
            U[n+1 + k] = real_t(0.);
            if(k < n-1){
                U[k] = L[k];
                Vout[2*k] = real_t(0.); Vout[2*k+1] = real_t(0.);
            } else {
                L[k] = real_t(0.);
                U[k] = real_t(0.);
            }
        }

        sycl::group_barrier(cta);
        
        // This part must execute serially by a single thread
        if(tix == 0) {
            for(int k = 0; k < n-1; k++) {
                if (std::abs(L[k]) > numerical_zero) {
                    a[0] = D[k]; a[1] = L[k];       // a = T[k:k+2,k] is the vector of nonzeros in kth subdiagonal column.
                    
                    real_t anorm = sycl::sqrt(a[0]*a[0] + a[1]*a[1]); 

                    v[0] = D[k]; v[1] = L[k];
                    real_t alpha = -sycl::copysign(anorm,a[0]);
                    v[0] -= alpha;

                    real_t vnorm = sycl::sqrt(v[0]*v[0]+v[1]*v[1]);
                    real_t norm_inv = real_t(1.)/vnorm;
                    v[0] *= norm_inv;  v[1] *= norm_inv;

                    Vout[2*k] = v[0]; Vout[2*k+1] = v[1];
                    
                    coord3d vTA = { D[ k ]*v[0] + L[ k ]*v[1],
                                    U[ k ]*v[0] + D[k+1]*v[1],
                                    U[(n+1)+k]*v[0] + U[k+1]*v[1]};

                    D[k]     -= real_t(2.)*v[0]*vTA[0];
                    L[k]     -= real_t(2.)*v[1]*vTA[0];
                    U[k]     -= real_t(2.)*v[0]*vTA[1];
                    D[k+1]     -= real_t(2.)*v[1]*vTA[1];
                    U[(n+1)+k] -= real_t(2.)*v[0]*vTA[2];
                    U[k+1]     -= real_t(2.)*v[1]*vTA[2];
                }
            }
        }

        // Still single-threaded part
        if(tix == 0) {
            int k = 0;
            const real_t *v = &Vout[0];
            real_t vTA[2] = {D[k]*v[0] + U[k]*v[1],
                            0 + D[k+1]*v[1]};
            
            D[k]       -= real_t(2.)*v[0]*vTA[0];
            U[k]       -= real_t(2.)*v[1]*vTA[0];
            L[k]       -= real_t(2.)*v[0]*vTA[1];
            D[k+1]     -= real_t(2.)*v[1]*vTA[1];
        }
        
        sycl::group_barrier(cta);

        // Another single-threaded part
        if(tix == 0) {
            for(int k = 1; k < n-1; k++) {
                const real_t *v = &Vout[2*k];
                coord3d vTA = {U[k-1]*v[0] + U[(n+1)+k-1]*v[1],
                                D[k]*v[0] + U[k]*v[1],
                                L[k]*v[0] + D[k+1]*v[1]};

                U[k-1]     -= real_t(2.)*v[0]*vTA[0];
                U[(n+1)+(k-1)] -= real_t(2.)*v[1]*vTA[0];
                U[k]       -= real_t(2.)*v[1]*vTA[1];
                D[k]       -= real_t(2.)*v[0]*vTA[1];
                L[k]       -= real_t(2.)*v[0]*vTA[2];
                D[k+1]     -= real_t(2.)*v[1]*vTA[2];
            }
        }

        sycl::group_barrier(cta);
        
        // Copy working diagonals to output - parallel
        for (int k = tix; k < n; k += bdim) {
            D[k] += shift;
            if(k < n-1) {
                L[k] = U[k];
            }
        }
        
        sycl::group_barrier(cta);
        
        if (tix == 0) {
            D[n] = d_n;
            L[n-1] = l_nm1;
            L[n] = l_n;
        }
        sycl::group_barrier(cta);
    }

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

        auto T_ortho =      std::chrono::duration<double, std::micro>(0);
        auto T_spmm =       std::chrono::duration<double, std::micro>(0);
        auto T_lanczos =    std::chrono::duration<double, std::micro>(0);

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

        //Allocate memory for the QR factorization
        auto smem_required = sizeof(T) * (3 * n + 3 * (n + 1));
        bool smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= smem_required;

        auto V_reflectors = pool.allocate<T>(ctx, !smem_possible ? 2 * n * batch_size : 0);
        auto D_global = pool.allocate<T>(ctx, !smem_possible ? (n+1) * batch_size : 0);   // Diagonal of tridiag matrix (same as alphas)
        auto U_global = pool.allocate<T>(ctx, !smem_possible ? 2 * (n + 1) * batch_size : 0);   // Upper diagonal of tridiag matrix (same as betas)
        auto L_global = pool.allocate<T>(ctx, !smem_possible ? n * batch_size : 0);   // Lower diagonal of tridiag matrix (same as betas)
        auto Q_eigenvectors = pool.allocate<T>(ctx, jobz == JobType::EigenVectors ? n * n * batch_size : 0);

        // Run the QR algorithm to get eigenvalues (and eigenvectors if requested)
        ctx->submit([&](sycl::handler& h) {
            auto Vstride = V.stride();
            auto Vdata = V.data_ptr();
            auto Vld = V.ld();

            sycl::local_accessor<T, 1> D_smem(smem_possible ? n + 1 : 0, h);
            sycl::local_accessor<T, 1> L_smem(smem_possible ? n : 0, h);
            sycl::local_accessor<T, 1> U_smem(smem_possible ? 2 * (n + 1) : 0, h);
            sycl::local_accessor<T, 1> V_smem(smem_possible ? 2 * n : 0, h);

            h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto tid = item.get_local_linear_id();
                auto bdim = item.get_local_range()[0];
                auto cta = item.get_group();

                auto D = Span(smem_possible ? D_smem.begin() : D_global.data() + bid*(n+1), n+1);
                auto L = Span(smem_possible ? L_smem.begin() : L_global.data() + bid*n, n);
                auto U = Span(smem_possible ? U_smem.begin() : U_global.data() + bid*2*(n+1), 2 * (n + 1));
                auto V = Span(smem_possible ? V_smem.begin() : V_reflectors.data() + bid*2*n, 2 * n);
                
                // Set up the pointers to the batch-specific data
                auto batch_alphas = Span(alphas.data() + bid * n, n);
                auto batch_betas = Span(betas.data() + bid * n, n);
                T* batch_Q = Q_eigenvectors.data() + (jobz == JobType::EigenVectors ? bid * n * n : bid * n);
                typename base_type<T>::type* batch_W = W.data() + bid * n;

                //If jobz == EigenVectors, initialize the eigenvector matrix to identity
                if (jobz == JobType::EigenVectors) {
                    for (int i = tid; i < n*n; i += bdim) {
                        batch_Q[i] = T(0); // Initialize to zero matrix
                    }
                    sycl::group_barrier(cta);
                    for (int i = tid; i < n; i += bdim) {
                        batch_Q[i * (n + 1)] = T(1); // Set diagonal to 1
                    }
                }

                for( int i = tid; i < n; i += bdim) {
                    D[i] = batch_alphas[i];
                    L[i] = batch_betas[i];
                }
                
                sycl::group_barrier(cta);
                
                // Process each eigenvalue using the QR algorithm
                for (int k = n-1; k >= 0; k--) {
                    T d = D[k];
                    T shift = d;
                    
                    int i = 0;
                    real_t GR = (k > 0 ? std::abs(L[k-1]) : 0) + std::abs(L[k]);
                    int not_done = 1;
                    
                    while (not_done > 0) {
                        i++;
                        
                        // Apply the QR transformation to the tridiagonal matrix
                        T_QTQ(cta, k+1, D, L, U, V, shift);
                        
                        // Update eigenvectors if requested
                        if (jobz == JobType::EigenVectors) {
                            apply_all_reflections(cta, V, k, n, batch_Q);
                        }
                        
                        
                        GR = (k > 0 ? std::abs(L[k-1]) : 0) + (k+1 < n ? std::abs(L[k]) : 0);
                        
                        if (k > 0) {
                            std::array<T,4> args = {D[k-1], L[k-1], L[k-1], D[k]};
                            auto [l0, l1] = eigvalsh2x2(args);
                            shift = std::abs(l0-d) < std::abs(l1-d) ? l0 : l1;
                        } else {
                            shift = D[k];
                        }
                        
                        if (GR <= std::numeric_limits<real_t>::epsilon() * real_t(10.0)) {
                            not_done--;
                        }
                        
                        if (i > 5) {
                            // Convergence failed, use current best estimate
                            break;
                        }
                    }
                }
                
                // Copy eigenvalues to output array
                for (int i = tid; i < n; i += bdim) {
                    batch_W[i] = real_part(D[i]);
                }
            });
        });

        //We have computed the decomposition T = Q^H * diag(W) * Q
        //Furthermore we have A = V * T * V^H
        //Which means means A = V * Q^H * diag(W) * Q * V^H
        //So it follows that the eigenvectors of A are given by V * Q
        if (jobz == JobType::EigenVectors) {
            gemm<B>(ctx, MatrixView(Vmem.data(), n, n, n, (n+1)*n, batch_size), 
                        MatrixView(Q_eigenvectors.data(), n, n, n, n*n, batch_size), 
                        V, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
        }
        /* std::cout << "Lanczos iteration time: " << T_lanczos.count() / batch_size << " µs per matrix" << std::endl; */
        /* std::cout << "Orthogonalization time: " << T_ortho.count() / batch_size << " µs per matrix" << std::endl; */
        /* std::cout << "SpMV time: " << T_spmm.count() / batch_size << " µs per matrix" << std::endl; */
        /* std::cout << "QR iteration time: " << elapsed_qr / batch_size << " µs per matrix" << std::endl; */
        /* std::cout << "Eigenvector time: " << elapsed_eigenvectors / batch_size << " µs per matrix" << std::endl; */
        /* std::cout << "Total time: " << (T_lanczos.count() + T_ortho.count() + T_spmm.count() + elapsed_qr + elapsed_eigenvectors) / batch_size << " µs per matrix" << std::endl; */

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
        // Additional memory required for QR factorization if there is insufficient shared memory for work-group local storage
        bool smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * (3 * n + 3 * (n + 1));

            // Allocate memory for the Householder reflection vectors and the tridiagonal matrix
        basic_size += BumpAllocator::allocation_size<T>(ctx,    smem_possible ? (2*n*batch_size) : 0) + // Householder reflection vectors
                        BumpAllocator::allocation_size<T>(ctx,  smem_possible ? ((n+1)*batch_size) : 0) + // Diagonal of tridiagonal matrix
                        BumpAllocator::allocation_size<T>(ctx,  smem_possible ? (2*(n+1)*batch_size) : 0) + // Upper diagonal of tridiagonal matrix
                        BumpAllocator::allocation_size<T>(ctx,  smem_possible ? (n*batch_size) : 0); // Lower diagonal of tridiagonal matrix

        // Eigenvector memory (only if needed)
        size_t eigenvectors_size = (jobz == JobType::EigenVectors) ? 
                                 BumpAllocator::allocation_size<T>(ctx, n*n*batch_size) : 
                                 BumpAllocator::allocation_size<T>(ctx, n*batch_size);
        
        size_t sort_size = BumpAllocator::allocation_size<int>(ctx, n*batch_size) + 
                           BumpAllocator::allocation_size<T>(ctx, n*batch_size);
        return basic_size  + eigenvectors_size + sort_size;
    }

    #define LANCZOS_INSTANTIATE(fp, mf) \
    template Event lanczos<Backend::CUDA, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        Span<std::byte>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \
    template size_t lanczos_buffer_size<Backend::CUDA, fp, mf>( \
        Queue&, \
        const MatrixView<fp, mf>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const LanczosParams<fp>&); \
    
    #define LANCZOS_INSTANTIATE_FOR_FP(fp) \
        LANCZOS_INSTANTIATE(fp, MatrixFormat::CSR) \
        LANCZOS_INSTANTIATE(fp, MatrixFormat::Dense)

    LANCZOS_INSTANTIATE_FOR_FP(float)
    LANCZOS_INSTANTIATE_FOR_FP(double)
    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<float>)
    //LANCZOS_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef LANCZOS_INSTANTIATE
    #undef LANCZOS_INSTANTIATE_FOR_FP


}