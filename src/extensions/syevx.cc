#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/dpl/random>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <blas/extra.hh>
#include "../math-helpers.hh"

namespace batchlas {
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxResidualsKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W, //Output eigenvalues
                size_t neigs, //Number of eigenvalues to compute
                Span<std::byte> workspace,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V, //Output eigenvectors for jobz == JobType::EigenVectors
                const SyevxParams<T>& params 
        ) {
        using float_type = typename base_type<T>::type;
        // Implementation of the syevx function
        // This function computes the eigenvalues and eigenvectors of a symmetric matrix
        int64_t block_vectors = neigs + params.extra_directions;
        auto pool = BumpAllocator(workspace);
        auto n = A.rows_;
        auto batch_size = A.batch_size();
        auto Sdata =        pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto ASdata =       pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto S_newdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto Stempdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto StASdata =     pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);
        auto C_pdata =      pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * batch_size);
        auto lambdas =      pool.allocate<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);
        auto residuals =    pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        auto best_residuals = pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        
        auto S =    MatrixView(Sdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto X = S({0,n}, {0,block_vectors});                       //First block of S
        auto P = S({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of S
        auto R = S({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of S
        auto XP = S({0,n}, {0,2 * block_vectors});                  //First two blocks of S
        
        auto AS =   MatrixView(ASdata.data(), n, block_vectors*3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto AX =   AS({0,n}, {0,block_vectors});                       //First block of AS
        auto AP =   AS({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of AS
        auto AR =   AS({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of AS

        auto StAS_base = MatrixView(StASdata.data(), block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3 * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto XtAX = MatrixView(StASdata.data(), block_vectors, block_vectors, block_vectors, block_vectors * block_vectors, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto C_p =  MatrixView(C_pdata.data(), block_vectors * 3, block_vectors, block_vectors*3, block_vectors * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto S_new = MatrixView(S_newdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());

        auto X_new =  S_new({0,n}, {0,block_vectors});                       //First block of S_new
        auto P_new = S_new({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of S_new
        auto R_new = S_new({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of S_new
        auto XP_new = S_new({0,n}, {0,2 * block_vectors});                 //First two blocks of S_new

        auto AS_new = MatrixView(Stempdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto AX_new = AS_new({0,n}, {0,block_vectors});                       //First block of AS_new
        auto AP_new = AS_new({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of AS_new
        auto AR_new = AS_new({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of AS_new


        Span<std::byte> spmm_workspace;
        if constexpr (MFormat == MatrixFormat::CSR) {
            spmm_workspace = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, S, AS, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
        }

        auto syev_workspace = pool.allocate<std::byte>(ctx, syev_buffer_size<B>(ctx, StAS_base, W, JobType::EigenVectors, Uplo::Lower));
        auto ortho_workspace = pool.allocate<std::byte>(ctx, std::max(  ortho_buffer_size<B>(ctx, R, XP, Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                        ortho_buffer_size<B>(ctx, C_p, StAS_base, Transpose::NoTrans, Transpose::NoTrans, params.algorithm)));
        
        //Double buffering pointer swap approach as opposed to copying data unnecessarily                                                                        
        auto swap_subspace = [&](){
            std::swap(X, X_new);
            std::swap(P, P_new);
            std::swap(R, R_new);
            std::swap(XP, XP_new);
            std::swap(AX, AX_new);
            std::swap(AP, AP_new);
            std::swap(S, S_new);
            std::swap(AS, AS_new);
            std::swap(AR, AR_new);
        };

        auto trans = (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) ? Transpose::ConjTrans : Transpose::Trans;

        S.fill_random(ctx);

        //Orthonormalize initial vectors
        ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
        //Compute AX
        if constexpr (MFormat == MatrixFormat::Dense) {
            gemm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        } else {
            //For sparse matrices we use the spmm function
            spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
        }
        //Compute X^T AX
        gemm<B>(ctx, X, AX, XtAX, T(1.0), T(0.0), trans, Transpose::NoTrans);
        //Solve the eigenvalue problem
        syev<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
        //Update X and corresponding implicit update of AX
        gemm<B>(ctx, X, XtAX, X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        gemm<B>(ctx, AX, XtAX, AX_new , T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

        swap_subspace();
        bool restart = true;

        size_t residual_wg_size = std::min(get_kernel_max_wg_size<SyevxResidualsKernel<B,T,MFormat>>(ctx), size_t(n));

        //Compute R = AX - X * diag(lambdas)
        for(int it = 0; it < params.iterations; it++){
            int Nvecs = restart ? block_vectors * 2 : block_vectors * 3;
            //Compute R = AX - X * diag(lambdas)
            ctx -> submit([&](sycl::handler& h){
                auto Rdata = R.data_ptr();
                auto Xdata = X.data_ptr();
                auto AXdata = AX.data_ptr();
                auto smem = sycl::local_accessor<float_type>(n,h);
                h.parallel_for<SyevxResidualsKernel<B,T,MFormat>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*residual_wg_size)}, sycl::range{size_t(residual_wg_size)}), [=](sycl::nd_item<1> item){
                    auto num_eigvals = it < 2 ? (it+1) * block_vectors : 3*block_vectors;

                    auto tid = item.get_local_linear_id();
                    sycl::group<1> cta = item.get_group();
                    auto bid = item.get_group_linear_id();
                    auto blockR = Span(Rdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockX = Span(Xdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockAX = Span(AXdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockLambdas = lambdas.subspan(bid * (num_eigvals), num_eigvals);
                    auto blockresiduals = residuals.subspan(bid * (neigs), neigs);
                    auto blockbestresiduals = best_residuals.subspan(bid * (neigs), neigs);
                    auto blockW = W.subspan(bid * (neigs), neigs);
    
                    sycl::group_barrier(cta);
                    for (int i = tid; i < n*block_vectors; i+=cta.get_local_range(0)){
                        auto eigvect_id = i / n;
                        auto eigval = blockLambdas[params.find_largest ? (num_eigvals - 1 - eigvect_id) : eigvect_id];
                        blockR[i] = blockAX[i] - blockX[i] * eigval;
                    }
                    sycl::group_barrier(cta);
                    
                    for (int i = 0; i < neigs; i++){
                        smem[tid] = internal::norm_squared(blockR[i*n + tid]);
                        blockresiduals[i] = sycl::sqrt((sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())));
                        smem[tid] = internal::norm_squared(blockX[i*n + tid]);
                        blockresiduals[i] /= sycl::sqrt(sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())) * blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                    }
                    
                    sycl::group_barrier(cta);
                    if (tid < neigs){
                        auto bestresidual = blockbestresiduals[tid];
                        auto residual = blockresiduals[tid];
                        if (bestresidual > residual || it == 0){
                            blockbestresiduals[tid] = residual;
                            blockW[tid] = blockLambdas[params.find_largest ? (num_eigvals - 1 - tid) : tid];
                        }
                    }
                });
            });

            ortho<B>(ctx, R, restart ? X : XP, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);

            if (restart){
                ctx -> submit([&](sycl::handler& h){
                    auto Sdata = S.data_ptr();
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size*128)}, sycl::range{size_t(128)}), [=](sycl::nd_item<1> item){
                        auto tid = item.get_local_linear_id();
                        auto bid = item.get_group_linear_id();
                        auto cta = item.get_group();
                        auto block_src = Span(Sdata + (bid * 3 + 2) * n * block_vectors, n * block_vectors);
                        auto block_dst = Span(Sdata + (bid * 3 + 1) * n * block_vectors, n * block_vectors);
                        for(int i = tid; i < n*block_vectors; i+=cta.get_local_range(0)){
                            block_dst[i] = block_src[i];
                        }
                    });
                });
            }
            //Compute AR
            if constexpr (MFormat == MatrixFormat::Dense) {
                gemm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            } else {
                spmm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            }

            auto StAS = MatrixView(StAS_base, Nvecs, Nvecs, Nvecs, Nvecs * Nvecs);
            //Compute S^T A S
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), AS({0,n}, {0,Nvecs}), StAS, T(1.0), T(0.0), trans, Transpose::NoTrans);
            //Solve the eigenvalue problem
            syev<B>(ctx, StAS, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
            //If largest = true, then the order of the eigenvectors is reversed
            if(params.find_largest){
                ctx -> submit([&](sycl::handler& h){
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * 256)}, sycl::range{size_t(256)}), [=](sycl::nd_item<1> item){
                        auto tid = item.get_local_linear_id();
                        auto bid = item.get_group_linear_id();
                        auto cta = item.get_group();
                        auto dst_acc = StASdata.subspan(bid * Nvecs * Nvecs, Nvecs * Nvecs);
                        auto src_acc = StASdata.subspan(bid * Nvecs * Nvecs, Nvecs * Nvecs);
                        for(int i = tid; i < Nvecs*(Nvecs/2); i+=cta.get_local_range(0)){
                            auto src_ix = i % Nvecs + (Nvecs - 1 - i / Nvecs)*Nvecs ; 
                            std::swap(dst_acc[i], src_acc[src_ix]);
                        }
                    });
                });
            }
            //X(i+1) =  [X(i), R(i), P(i)] * [Zx, Zr, Zp]^T

            //Compute C_p = C_x - [I 
            //                     0
            //                     0]
            ctx -> submit([&](sycl::handler& h){
                auto Cstride = block_vectors * block_vectors * 3;
                auto Nactive = block_vectors; //When we start soft-locking the eigenvectors, we will need to update this
                h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size*Nactive)}, sycl::range{size_t(Nactive)}), [=](sycl::nd_item<1> item){
                    auto cta = item.get_group();
                    auto tid = item.get_local_linear_id();
                    auto bid = item.get_group_linear_id();
    
                    auto C_x = StASdata.subspan(bid * Nvecs * Nvecs, Nvecs * Nvecs);
                    auto C_p = C_pdata.subspan(bid * Cstride, Nvecs * Nactive);
    
                    for(int i = tid; i < Nactive * Nvecs; i+=cta.get_local_range(0)) {C_p[i] = C_x[i];}
                    
                    sycl::group_barrier(cta);
                    C_p[tid * Nvecs + tid] -= 1;
                });
            });
            


            //Compute new search directions
            //X = [X, P, R] * C_x
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), StAS({0,Nvecs}, {0,block_vectors}), X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AX: AX = [AX, AP, AR] * C_x
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), StAS({0,Nvecs}, {0,block_vectors}), AX_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Orthonormalize C_p against the best eigenvectors
            ortho<B>(ctx, MatrixView(C_p, Nvecs, block_vectors, Nvecs), StAS({0,Nvecs}, {0,block_vectors}), Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            //Compute P = [X, P, R] * C_p
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), MatrixView(C_p, Nvecs, block_vectors, Nvecs), P_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AP
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), MatrixView(C_p, Nvecs, block_vectors, Nvecs), AP_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

            swap_subspace(); //AX <=> AX_new, AP <=> AP_new, X <=> X_new, P <=> P_new ...
            restart = false;
        }
       return ctx.get_event();

    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params){
        auto block_vectors = neigs + params.extra_directions;
            auto batch_size = A.batch_size();
            auto n = A.rows();
            size_t work_size = 0;
            auto Xview = MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors, n, n * block_vectors, batch_size, nullptr);
            auto AXview = MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors, n, n * block_vectors, batch_size, nullptr);
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,syev_buffer_size<B>(ctx, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors*3,block_vectors*3,block_vectors*3, 3*3*block_vectors*block_vectors,batch_size, nullptr), W, JobType::EigenVectors, Uplo::Lower));
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,std::max(    ortho_buffer_size<B>(ctx, Xview, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors*2, n, n * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                                    ortho_buffer_size<B>(ctx, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors, block_vectors * 3), MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm)));
            if constexpr (MFormat == MatrixFormat::CSR) {
                work_size += BumpAllocator::allocation_size<std::byte>(ctx,spmm_buffer_size<B>(ctx, A, Xview, AXview, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            }
                        
            work_size += BumpAllocator::allocation_size<T*>(ctx, batch_size) * 7;
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * 3 * batch_size) * 4;                    //Sdata, ASdata, S_newdata, Stempdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);        //StASdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * batch_size);            //C_pdata
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);  //lambdas
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //residuals
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //best residuals
            return work_size;
    }

    #define SYEVX_INSTANTIATE(back, fp, fmt) \
    template Event syevx<back, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        Span<typename base_type<fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<fp, MatrixFormat::Dense>&,\
        const SyevxParams<fp>&);\
    template size_t syevx_buffer_size<back, fp, fmt>(\
        Queue&,\
        const MatrixView<fp, fmt>&,\
        Span<typename base_type<fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<fp, MatrixFormat::Dense>&,\
        const SyevxParams<fp>&);
    

    #define SYEVX_INSTANTIATE_FOR_BACKEND(back)\
        SYEVX_INSTANTIATE(back, float, MatrixFormat::Dense)\
        SYEVX_INSTANTIATE(back, double, MatrixFormat::Dense)\
        SYEVX_INSTANTIATE(back, std::complex<float>, MatrixFormat::Dense)\
        SYEVX_INSTANTIATE(back, std::complex<double>, MatrixFormat::Dense)\
        SYEVX_INSTANTIATE(back, float, MatrixFormat::CSR)\
        SYEVX_INSTANTIATE(back, double, MatrixFormat::CSR)\
        SYEVX_INSTANTIATE(back, std::complex<float>, MatrixFormat::CSR)\
        SYEVX_INSTANTIATE(back, std::complex<double>, MatrixFormat::CSR)

    #if BATCHLAS_HAS_CUDA_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
    #endif

    #undef SYEVX_INSTANTIATE
}