#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/dpl/random>

namespace batchlas {
    template <Backend B, typename T, Format F, BatchType BT>
    struct SyevxResidualsKernel;
    template <Backend B, typename T, Format F, BatchType BT>
    struct RandomInitializationKernel;


    template <Backend B, typename T, Format F, BatchType BT>
    Event syevx(Queue& ctx,
                SparseMatHandle<T, F, BT>& A,
                Span<typename base_type<T>::type> W, //Output eigenvalues
                size_t neigs, //Number of eigenvalues to compute
                Span<std::byte> workspace,
                JobType jobz,
                const DenseMatView<T, BT>& V, //Output eigenvectors for jobz == JobType::EigenVectors
                const SyevxParams<T>& params 
        ) {
        using float_type = typename base_type<T>::type;
        // Implementation of the syevx function
        // This function computes the eigenvalues and eigenvectors of a symmetric matrix
        auto block_vectors = neigs + params.extra_directions;
        auto pool = BumpAllocator(workspace);
        auto n = A.rows_;
        auto batch_size = get_batch_size(A);
        auto Sdata =        pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto ASdata =       pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto S_newdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto Stempdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto StASdata =     pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);
        auto C_pdata =      pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * batch_size);
        auto lambdas =      pool.allocate<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);
        auto residuals =    pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        auto best_residuals = pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        
        auto S =    create_view<T, BT>(Sdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto X =    create_view<T, BT>(Sdata.data(), n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto R =    create_view<T, BT>(Sdata.data() + 2 * n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto P =    create_view<T, BT>(Sdata.data() + n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto XP =   create_view<T, BT>(Sdata.data(), n, block_vectors * 2, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));

        auto AS =   create_view<T, BT>(ASdata.data(), n, block_vectors*3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AX =   create_view<T, BT>(ASdata.data(), n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AP =   create_view<T, BT>(ASdata.data() + n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AR =   create_view<T, BT>(ASdata.data() + 2 * n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto StAS = create_view<T, BT>(StASdata.data(), block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3 * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto XtAX = create_view<T, BT>(StASdata.data(), block_vectors, block_vectors, block_vectors, block_vectors * block_vectors, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto C_p =  create_view<T, BT>(C_pdata.data(), block_vectors * 3, block_vectors, block_vectors*3, block_vectors * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto S_new = create_view<T, BT>(S_newdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto X_new =  create_view<T, BT>(S_newdata.data(), n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto P_new =  create_view<T, BT>(S_newdata.data() + n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto XP_new = create_view<T, BT>(S_newdata.data(), n, block_vectors * 2, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto R_new =  create_view<T, BT>(S_newdata.data() + 2 * n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));

        auto AS_new = create_view<T, BT>(Stempdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AX_new = create_view<T, BT>(Stempdata.data(), n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AP_new = create_view<T, BT>(Stempdata.data() + n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        auto AR_new = create_view<T, BT>(Stempdata.data() + 2 * n * block_vectors, n, block_vectors, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size));
        //auto Xnew = create_view<T, BT>(pool.allocate<T>(ctx, n*block_vectors*batch_size).data(), n, block_vectors, n, n * block_vectors, batch_size, pool.allocate<T*>(ctx, batch_size));
        //auto AXnew = create_view<T, BT>(pool.allocate<T>(ctx, n*block_vectors*batch_size).data(), n, block_vectors, n, n * block_vectors, batch_size, pool.allocate<T*>(ctx, batch_size));
        
        auto spmm_workspace = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, S, AS, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
        auto syev_workspace = pool.allocate<std::byte>(ctx, syev_buffer_size<B>(ctx, StAS, W, JobType::EigenVectors, Uplo::Lower));
        auto ortho_workspace = pool.allocate<std::byte>(ctx, std::max(  ortho_buffer_size<B>(ctx, R, XP, Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                        ortho_buffer_size<B>(ctx, C_p, StAS, Transpose::NoTrans, Transpose::NoTrans, params.algorithm)));
        
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

        auto initialization_wg_size = std::min(get_kernel_max_wg_size<RandomInitializationKernel<B,T,F,BT>>(ctx), size_t(n));

        ctx -> submit([&](sycl::handler& h ){
            h.parallel_for<RandomInitializationKernel<B,T,F,BT>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*initialization_wg_size)}, sycl::range{size_t(initialization_wg_size)}), [=](sycl::nd_item<1> item){
                auto tid = item.get_local_linear_id();
                sycl::group<1> cta = item.get_group();
                auto bid = item.get_group_linear_id();
                oneapi::dpl::uniform_real_distribution<typename base_type<T>::type> distr(0.0, 1.0);            
                oneapi::dpl::minstd_rand engine(42, tid);
                auto blockX = Sdata.subspan(bid * block_vectors*3 * n, block_vectors*n);
    
                sycl::group_barrier(cta);
                for (int i = tid; i < n*block_vectors; i+=cta.get_local_range(0)){
                    blockX[i] = distr(engine);
                }
            });
        });

        //Orthonormalize initial vectors
        ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
        //Compute AX
        spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
        //Compute X^T AX
        gemm<B>(ctx, X, AX, XtAX, T(1.0), T(0.0), Transpose::Trans, Transpose::NoTrans);
        //Solve the eigenvalue problem
        syev<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
        //Update X and corresponding implicit update of AX
        gemm<B>(ctx, X, XtAX, X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        gemm<B>(ctx, AX, XtAX, AX_new , T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

        swap_subspace();
        bool restart = true;

        size_t residual_wg_size = std::min(get_kernel_max_wg_size<SyevxResidualsKernel<B,T,F,BT>>(ctx), size_t(n));


        //Compute R = AX - X * diag(lambdas)
        for(int it = 0; it < params.iterations; it++){
            int Nvecs = restart ? block_vectors * 2 : block_vectors * 3;
            //Compute R = AX - X * diag(lambdas)
            ctx -> submit([&](sycl::handler& h){
                auto Rdata = get_data(R);
                auto Xdata = get_data(X);
                auto AXdata = get_data(AX);
                auto smem = sycl::local_accessor<float_type>(n,h);
                h.parallel_for<SyevxResidualsKernel<B,T,F,BT>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*residual_wg_size)}, sycl::range{size_t(residual_wg_size)}), [=](sycl::nd_item<1> item){
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
                    
                    for (int i = 0; i < neigs; i++){
                        if constexpr (sycl::detail::is_complex<T>::value){
                            smem[tid] = blockR[i*n + tid].real() * blockR[i*n + tid].real();
                            blockresiduals[i] = sycl::sqrt((sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())));
                            smem[tid] = blockX[i*n + tid].real() * blockX[i*n + tid].real();
                            blockresiduals[i] /= sycl::sqrt(sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())) * blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                        } else {
                            smem[tid] = blockR[i*n + tid] * blockR[i*n + tid];
                            blockresiduals[i] = sycl::sqrt((sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())));
                            smem[tid] = blockX[i*n + tid] * blockX[i*n + tid];
                            blockresiduals[i] /= sycl::sqrt(sycl::joint_reduce(cta, smem.get_pointer(), smem.get_pointer() + n, sycl::plus<float_type>())) * blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                        }
                    }
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
                    auto Sdata = get_data(S);
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
            spmm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            //Compute S^T A S
            gemm<B>(ctx, subview(S, n, Nvecs), subview(AS, n, Nvecs), subview(StAS, Nvecs, Nvecs, Nvecs, Nvecs * Nvecs), T(1.0), T(0.0), Transpose::Trans, Transpose::NoTrans);
            //Solve the eigenvalue problem
            syev<B>(ctx, subview(StAS, Nvecs, Nvecs, Nvecs, Nvecs * Nvecs), lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
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
            //X = [X, P, R] * [Zx, Zp, Zr]^T
            gemm<B>(ctx, subview(S, n, Nvecs), subview(StAS, Nvecs, Nvecs, Nvecs, Nvecs * Nvecs), X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AX
            gemm<B>(ctx, subview(AS, n, Nvecs), subview(StAS, Nvecs, Nvecs, Nvecs, Nvecs * Nvecs), AX_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Orthonormalize C_p against the best eigenvectors
            ortho<B>(ctx, subview(C_p, Nvecs,block_vectors, Nvecs), subview(StAS, Nvecs, block_vectors, Nvecs, Nvecs * Nvecs), Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            //Compute P = [X, P, R] * C_p
            gemm<B>(ctx, subview(S, n, Nvecs), subview(C_p, Nvecs, block_vectors, Nvecs), P_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AP
            gemm<B>(ctx, subview(AS, n, Nvecs), subview(C_p, Nvecs, block_vectors, Nvecs), AP_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

            swap_subspace(); //AX <=> AX_new, AP <=> AP_new, X <=> X_new, P <=> P_new ...
            restart = false;
        }
       return ctx.get_event();

    }

    template <Backend B, typename T, Format F, BatchType BT>
    size_t syevx_buffer_size(Queue& ctx,
                SparseMatHandle<T, F, BT>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const DenseMatView<T, BT>& V,
                const SyevxParams<T>& params){
            auto block_vectors = neigs + params.extra_directions;
            auto batch_size = get_batch_size(A);
            auto n = A.rows_;
            size_t work_size = 0;
            auto Xview = create_view<T,BT>(get_data(A),n, block_vectors * 3, n, n * block_vectors * 3, batch_size, Span<T*>{});
            auto AXview = create_view<T,BT>(get_data(A),n, block_vectors * 3, n, n * block_vectors * 3, batch_size, Span<T*>{});
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,syev_buffer_size<B>(ctx, create_view<T,BT>(get_data(A),block_vectors*3,block_vectors*3,block_vectors*3, 3*3*block_vectors*block_vectors,batch_size, Span<T*>{}), W, JobType::EigenVectors, Uplo::Lower));
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,ortho_buffer_size<B>(ctx, Xview, create_view<T,BT>(get_data(A),n, block_vectors, n, n * block_vectors * 3, batch_size, Span<T*>{}), Transpose::NoTrans, Transpose::NoTrans, params.algorithm));
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,spmm_buffer_size<B>(ctx, A, Xview, AXview, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            
            if constexpr (BT == BatchType::Batched) {
                work_size += BumpAllocator::allocation_size<T*>(ctx, batch_size) * 20;
            }
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * 3 * batch_size) * 4;
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * batch_size);
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * batch_size);
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * batch_size);
            return work_size;
        }


    #define SYEVX_INSTANTIATE(fp, BT) \
    template Event syevx<Backend::CUDA, fp, Format::CSR, BT>(\
        Queue&,\
        SparseMatHandle<fp, Format::CSR, BT>&,\
        Span<typename base_type<fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const DenseMatView<fp, BT>&,\
        const SyevxParams<fp>&);\
    template size_t syevx_buffer_size<Backend::CUDA, fp, Format::CSR, BT>(\
        Queue&,\
        SparseMatHandle<fp, Format::CSR, BT>&,\
        Span<typename base_type<fp>::type>,\
        size_t,\
        JobType,\
        const DenseMatView<fp, BT>&,\
        const SyevxParams<fp>&);\

    SYEVX_INSTANTIATE(float, BatchType::Single)
    SYEVX_INSTANTIATE(float, BatchType::Batched)
    SYEVX_INSTANTIATE(double, BatchType::Single)
    SYEVX_INSTANTIATE(double, BatchType::Batched)
    SYEVX_INSTANTIATE(std::complex<float>, BatchType::Single)
    SYEVX_INSTANTIATE(std::complex<float>, BatchType::Batched)
    SYEVX_INSTANTIATE(std::complex<double>, BatchType::Single)
    SYEVX_INSTANTIATE(std::complex<double>, BatchType::Batched)
    #undef SYEVX_INSTANTIATE
}
