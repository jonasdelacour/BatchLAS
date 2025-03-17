#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>

// High-level orthogonalization functions built on top of primitive BLAS operations
namespace batchlas {

    template <Backend B, typename T, BatchType BT>
    SyclEvent ortho(SyclQueue& ctx,
                    DenseMatView<T,BT> A,
                    Transpose transA,
                    Span<std::byte> workspace,
                    OrthoAlgorithm algo) {
        //If transA == NoTrans we find the orthonormal basis of the column space
        //Else we find the orthonormal basis of the row space

        static LinalgHandle<B> handle;
        auto batch_size = get_batch_size(A);
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto [m, k] = get_effective_dims(A, transA);
        Transpose inv_trans = transA == Transpose::Trans ? Transpose::NoTrans : Transpose::Trans;
        assert(k <= m);
        //If k > m && transA == NoTrans the columns of A are linearly dependent
        //Else if k > m && transA == Trans the rows of A are linearly dependent
        auto ATA =          pool.allocate<T>(ctx, k*k*batch_size);
        auto matAmem =      pool.allocate<T*>(ctx, batch_size);
        auto matATAmem =    pool.allocate<T*>(ctx, batch_size);
        auto ATA_stride = k*k;
        auto C = create_view<T, BT>(ATA.data(), k, k, k, ATA_stride, batch_size, matATAmem);
        auto potrf_workspace = pool.allocate<std::byte>(ctx, potrf_buffer_size<B>(ctx, C, Uplo::Lower));
        
        auto chol_alg = [&](){
            constexpr T alpha = 1.0;
            constexpr T beta = 0.0;
            
            //Compute StS = S^T * S or StS = S * S^T (depending on transA)
            gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);
            //Compute the Cholesky Factorization of StS
            potrf<B>(ctx, C, Uplo::Lower, potrf_workspace);
            //Compute Q = S * StS^-1 (S is overwritten with Q)
            trsm<B>(ctx, C, A, Side::Right, Uplo::Lower, inv_trans, Diag::NonUnit, alpha);
            //Compute the QR factorization of Q
            gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);
        };

        if (algo == OrthoAlgorithm::Cholesky){
            chol_alg();
        } else if (algo == OrthoAlgorithm::Chol2){
            chol_alg();
            chol_alg();
        } else if (algo == OrthoAlgorithm::ShiftChol3) {
            gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);

            auto ATA_ptr = get_data(C);
            ctx -> submit([&](sycl::handler& h){
                h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * k)}, sycl::range{size_t(k)}), [=](sycl::nd_item<1> item){
                    auto tid = item.get_local_linear_id();
                    auto bid = item.get_group_linear_id();
                    auto cta = item.get_group();
                    auto ATA_acc = ATA_ptr + bid * ATA_stride;
                    T g_norm = 0.0;
                    if constexpr (sycl::detail::is_complex<T>::value){
                        g_norm = sycl::reduce_over_group(cta, std::sqrt(ATA_acc[tid * k + tid].real()), sycl::maximum<typename T::value_type>());
                    } else {
                        g_norm = sycl::reduce_over_group(cta, std::sqrt(ATA_acc[tid * k + tid]), sycl::maximum<T>());
                    }
                    auto eps = std::numeric_limits<T>::epsilon();
                    auto shift = T(11.0) * T( T(m * k) * T(eps) + T(k + 1) * T(k) * T(eps)) * g_norm;
                    ATA_acc[tid * k + tid] += shift;
                });
            });

            //Compute the Cholesky Factorization of StS
            potrf<B>(ctx, C, Uplo::Lower, potrf_workspace);
            trsm<B>(ctx, C, A, Side::Right, Uplo::Lower, inv_trans, Diag::NonUnit, T(1.0));
            chol_alg();
            chol_alg();
        }
        
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    size_t ortho_buffer_size(SyclQueue& ctx,
                             DenseMatView<T,BT> A,
                             Transpose transA,
                             OrthoAlgorithm algo) {
        size_t size = 0;
        auto [m, k] = get_effective_dims(A, transA);
        return  BumpAllocator::allocation_size<std::byte>(ctx, potrf_buffer_size<B>(ctx, create_view<T, BT>(A.data_, m, k, m, m*k, get_batch_size(A), Span<T*>{}), Uplo::Lower)) + 
        2*BumpAllocator::allocation_size<T*>(ctx, get_batch_size(A)) + 
        BumpAllocator::allocation_size<T>(ctx, k*k*get_batch_size(A));
    }

    template <Backend B, typename T, BatchType BT>
    SyclEvent ortho(SyclQueue& ctx,
                    DenseMatView<T,BT> A,
                    DenseMatView<T,BT> M,
                    Transpose transA,
                    Transpose transM,
                    Span<std::byte> workspace,
                    OrthoAlgorithm algo,
                    size_t iterations) {
        BumpAllocator pool(workspace);
        //When orthogonalizing against an external basis M,
        //M must be an orthonormal basis
        //Both A and M must be either tall-and-skinny or short-and-fat
        //Furthermore the number of vectors in A and M must sum to at most the dimension of these vectors 
        auto nM = transM == Transpose::NoTrans ? M.cols_ : M.rows_;
        auto nA = transA == Transpose::NoTrans ? A.cols_ : A.rows_;
        auto k = transA == Transpose::NoTrans ? A.rows_ : A.cols_;
        assert(nA + nM <= k);
        assert(k == (transM == Transpose::NoTrans ? M.rows_ : M.cols_));

        auto inv_transA = transA == Transpose::Trans ? Transpose::NoTrans : Transpose::Trans;
        auto inv_transM = transM == Transpose::Trans ? Transpose::NoTrans : Transpose::Trans;
        auto MAmem =          pool.allocate<T>(ctx, nM*nA * get_batch_size(A));
        auto orthoworkspace = pool.allocate<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo));
        auto descrMA = create_view<T, BT>(MAmem.data(), nM, nA, nM, nM*nA, get_batch_size(A), Span<T*>{});
        auto isAtrans = transA == Transpose::Trans;
        auto is_first_transposed = static_cast<Transpose>(((transA == Transpose::Trans) || (transM == Transpose::Trans)));
        auto is_second_transposed = static_cast<Transpose>(((transA == Transpose::Trans) && (transM == Transpose::NoTrans)));
        for (size_t i = 0; i < iterations; i++){
            gemm<B>(ctx, M, A, descrMA, T(1.0), T(0.0), inv_transM, transA);
            gemm<B>(ctx, isAtrans ? descrMA : M, isAtrans ? M : descrMA, A, T(-1.0), T(1.0), is_first_transposed, is_second_transposed);

            ortho<B>(ctx, A, transA, orthoworkspace, algo);
        }
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    size_t ortho_buffer_size(SyclQueue& ctx,
                             DenseMatView<T,BT> A,
                             DenseMatView<T,BT> M,
                             Transpose transA,
                             Transpose transM,
                             OrthoAlgorithm algo,
                             size_t iterations) {
        auto nM = transM == Transpose::NoTrans ? M.cols_ : M.rows_;
        auto nA = transA == Transpose::NoTrans ? A.cols_ : A.rows_;
        return  BumpAllocator::allocation_size<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo)) +
                BumpAllocator::allocation_size<T>(ctx, nM*nA * get_batch_size(A));
    }  

    #define ORTHO_INSTANTIATE(fp, BT) \
    template SyclEvent ortho<Backend::CUDA, fp, BT>( \
        SyclQueue&, \
        DenseMatView<fp, BT>, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm); \
    template SyclEvent ortho<Backend::CUDA, fp, BT>( \
        SyclQueue&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        Transpose, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm, \
        size_t); \
    template size_t ortho_buffer_size<Backend::CUDA, fp, BT>( \
        SyclQueue&, \
        DenseMatView<fp, BT>, \
        Transpose, \
        OrthoAlgorithm); \
    template size_t ortho_buffer_size<Backend::CUDA, fp, BT>( \
        SyclQueue&, \
        DenseMatView<fp, BT>, \
        DenseMatView<fp, BT>, \
        Transpose, \
        Transpose, \
        OrthoAlgorithm, \
        size_t);

    // Instantiate all ortho functions for different floating point types and batch types
    #define ORTHO_INSTANTIATE_FOR_FP(fp) \
        ORTHO_INSTANTIATE(fp, BatchType::Batched) \
        ORTHO_INSTANTIATE(fp, BatchType::Single)
    
    // Instantiate for the floating-point types of interest
    ORTHO_INSTANTIATE_FOR_FP(float)
    ORTHO_INSTANTIATE_FOR_FP(double)
    ORTHO_INSTANTIATE_FOR_FP(std::complex<float>)
    ORTHO_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef ORTHO_INSTANTIATE
    #undef ORTHO_INSTANTIATE_FOR_FP
}