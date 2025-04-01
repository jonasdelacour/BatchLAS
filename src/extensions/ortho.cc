#include "../../include/blas/linalg.hh"
#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <numeric>
#include <algorithm>

// High-level orthogonalization functions built on top of primitive BLAS operations
namespace batchlas {

    template <Backend B, typename T, BatchType BT>
    struct OrthoNormalizeVector {};

    template <Backend B, typename T, BatchType BT>
    Event ortho(Queue& ctx,
                    const DenseMatView<T,BT>& A,
                    Transpose transA,
                    Span<std::byte> workspace,
                    OrthoAlgorithm algo) {
        //If transA == NoTrans we find the orthonormal basis of the column space
        //Else we find the orthonormal basis of the row space
        using float_t = base_type<T>::type;
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
        } else if (algo == OrthoAlgorithm::Householder) {
            throw std::runtime_error("Householder is not implemented yet");
        } else if (algo == OrthoAlgorithm::GramSchmidt) {
            //Implemented as an iterative process:
            //1. Compute orthogonality of A[:,0 .. k-1] and A[:,k .. m-1]
            //2. Subtract the projection of A[:,k .. m-1] onto A[:,0 .. k-1]
            //3. Normalize A[:,0 .. k-1]
            //Repeat until all vectors are orthogonal
            auto Ymem = pool.allocate<T>(ctx, batch_size * m);
            auto normalize_wg_size = std::min(get_kernel_max_wg_size<OrthoNormalizeVector<B, T, BT>>(ctx), size_t(m));
            for (int i = 0; i < k; i++){
                //View of the first i vectors (either columns or rows of A depending on transA)
                auto A_i = transA == Transpose::NoTrans ? create_view<T, BT>(A.data_, m, i, m, get_stride(A), get_batch_size(A), Span<T*>()) : create_view<T, BT>(A.data_, i, m, m, get_stride(A), get_batch_size(A), Span<T*>());
                //View of the next vector (either column or row of A depending on transA)
                auto C = create_vec<T, BT>(Ymem.data(), i, 1, i, get_batch_size(A));
                auto A_next = create_vec<T, BT>(A.data_ + m * i, m, 1, get_stride(A), get_batch_size(A));
                //output vector
                if (i > 0){ //If it's the first vector we just need to normalize it
                    gemv<B>(ctx, A_i, A_next, C, T(1.0), T(0.0), inv_trans);
                    gemv<B>(ctx, A_i, C, A_next, T(-1.0), T(1.0), transA);
                }
                auto real_part = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return value.real(); else return value; };
                //Normalize A_i
                ctx -> submit([&](sycl::handler& h){
                    auto Anext_ptr = get_data(A_next);
                    auto Anext_squared = sycl::local_accessor<float_t, 1>(sycl::range<1>(m), h);
                    h.parallel_for<OrthoNormalizeVector<B, T, BT>>(sycl::nd_range<1>(sycl::range{size_t(batch_size * normalize_wg_size)}, sycl::range{size_t(normalize_wg_size)}), [=](sycl::nd_item<1> item){
                        auto tid = item.get_local_linear_id();
                        auto bid = item.get_group_linear_id();
                        auto cta = item.get_group();
                        auto A_local_vector = Span<T>(Anext_ptr + bid * m, m);
                        float_t norm = float_t(0.0);
                        for (int j = tid; j < m; j+= cta.get_local_linear_range()){
                            Anext_squared[j] = real_part(A_local_vector[j]) * real_part(A_local_vector[j]);
                        }
                        
                        norm = std::sqrt(sycl::joint_reduce(cta, Anext_squared.begin(), Anext_squared.end(), sycl::plus<float_t>()));

                        for (int j = tid; j < m; j+= cta.get_local_linear_range()){
                            A_local_vector[j] /= norm;
                        }
                    });
                });
            }
        } else {
            throw std::runtime_error("Unknown orthogonalization algorithm");
        }
        
        return ctx.get_event();
    }

    template <Backend B, typename T, BatchType BT>
    size_t ortho_buffer_size(Queue& ctx,
                             const DenseMatView<T,BT>& A,
                             Transpose transA,
                             OrthoAlgorithm algo) {
        size_t size = 0;
        auto [m, k] = get_effective_dims(A, transA);
        return  BumpAllocator::allocation_size<std::byte>(ctx, potrf_buffer_size<B>(ctx, create_view<T, BT>(A.data_, m, k, m, m*k, get_batch_size(A), Span<T*>{}), Uplo::Lower)) + 
        2*BumpAllocator::allocation_size<T*>(ctx, get_batch_size(A)) + 
        BumpAllocator::allocation_size<T>(ctx, k*k*get_batch_size(A)) +
        BumpAllocator::allocation_size<T>(ctx, m*get_batch_size(A));
    }

    template <Backend B, typename T, BatchType BT>
    Event ortho(Queue& ctx,
                    const DenseMatView<T,BT>& A,
                    const DenseMatView<T,BT>& M,
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
        if(nA + nM > k){
            throw std::runtime_error("The number of vectors in A (" + std::to_string(nA) + ") and M (" + std::to_string(nM) + ") must sum to at most the dimension of these vectors (" + std::to_string(k) + ")");
        }
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
    size_t ortho_buffer_size(Queue& ctx,
                             const DenseMatView<T,BT>& A,
                             const DenseMatView<T,BT>& M,
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
    template Event ortho<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm); \
    template Event ortho<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        const DenseMatView<fp, BT>&, \
        Transpose, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm, \
        size_t); \
    template size_t ortho_buffer_size<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        Transpose, \
        OrthoAlgorithm); \
    template size_t ortho_buffer_size<Backend::CUDA, fp, BT>( \
        Queue&, \
        const DenseMatView<fp, BT>&, \
        const DenseMatView<fp, BT>&, \
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