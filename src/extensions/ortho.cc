#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <numeric>
#include <algorithm>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>


// High-level orthogonalization functions built on top of primitive BLAS operations
// Implementation using the new MatrixView structure
namespace batchlas {

    template <Backend B, typename T>
    struct OrthoNormalizeVector {};
    
    template <Backend B, typename T>
    struct StridedCopyKernel {};

    template <Backend B, typename T>
    Event ortho(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Transpose transA,
                Span<std::byte> workspace,
                OrthoAlgorithm algo) {
        //If transA == NoTrans we find the orthonormal basis of the column space
        //Else we find the orthonormal basis of the row space
        using float_t = typename base_type<T>::type;
        constexpr auto fmt = MatrixFormat::Dense;
        static LinalgHandle<B> handle;
        auto batch_size = A.batch_size();
        handle.setStream(ctx);
        BumpAllocator pool(workspace);
        auto [m, k] = get_effective_dims(A, transA);
        bool is_A_trans = transA == Transpose::Trans || transA == Transpose::ConjTrans;
        Transpose inv_trans = is_A_trans ? Transpose::NoTrans : 
                            std::is_same_v<T, std::complex<float_t>> ? Transpose::ConjTrans : Transpose::Trans;
        assert(k <= m);
        //If k > m && transA == NoTrans the columns of A are linearly dependent
        //Else if k > m && transA == Trans the rows of A are linearly dependent
        auto ATA =          pool.allocate<T>(ctx, k*k*batch_size);
        auto matAmem =      pool.allocate<T*>(ctx, batch_size);
        auto matATAmem =    pool.allocate<T*>(ctx, batch_size);
        auto ATA_stride = k*k;
        auto is_cholesky = algo == OrthoAlgorithm::Cholesky || algo == OrthoAlgorithm::Chol2 || algo == OrthoAlgorithm::ShiftChol3;

        auto C = MatrixView<T, fmt>(ATA.data(), k, k, k, ATA_stride, batch_size, matATAmem.data());
        auto potrf_workspace = pool.allocate<std::byte>(ctx, is_cholesky ? potrf_buffer_size<B>(ctx, C, Uplo::Lower) : 0);
        
        
        auto real_part = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return value.real(); else return value; };
        auto square = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return (value * std::conj(value)).real(); else return value * value; };
        
        auto chol_alg = [&](){
            constexpr T alpha = 1.0;
            constexpr T beta = 0.0;
            //Compute StS = S^T * S or StS = S * S^T (depending on transA)
            gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);
            //Compute the Cholesky Factorization of StS
            potrf<B>(ctx, C, Uplo::Lower, potrf_workspace);
            //Solve X * Chol(StS) = S
            trsm<B>(ctx, C, A, is_A_trans ? Side::Left : Side::Right, Uplo::Lower, inv_trans, Diag::NonUnit, alpha);
        };

        auto cgs_alg = [&](){
            //Implemented as an iterative process:
            //1. Compute orthogonality of A[:,0 .. k-1] and A[:,k .. m-1]
            //2. Subtract the projection of A[:,k .. m-1] onto A[:,0 .. k-1]
            //3. Normalize A[:,0 .. k-1]
            //Repeat until all vectors are orthogonal
            auto Ymem = pool.allocate<T>(ctx, batch_size * m);
            auto normalize_wg_size = std::min(get_kernel_max_wg_size<OrthoNormalizeVector<B, T>>(ctx), size_t(m));
            for (int i = 0; i < k; i++){
                //View of the first i vectors (either columns or rows of A depending on transA)
                auto A_i = transA == Transpose::NoTrans ? 
                      MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size) 
                    : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
                //View of the next vector (either column or row of A depending on transA)
                auto C = VectorView(Ymem.data(), i, 1, i, batch_size);
                auto A_next = VectorView(A.data_ptr() + m * i, m, 1, A.stride(), batch_size);
                //output vector
                if (i > 0){ //If it's the first vector we just need to normalize it
                    for (int j = 0; j < 2; j++){
                        gemv<B>(ctx, A_i, A_next, C, T(1.0), T(0.0), inv_trans);
                        gemv<B>(ctx, A_i, C, A_next, T(-1.0), T(1.0), transA);
                    }
                }
                //Normalize A_i
                ctx -> submit([&](sycl::handler& h){
                    auto Anext_squared = sycl::local_accessor<float_t, 1>(m, h);
                    auto A_stride = A.stride();
                    auto Aspan = Span(A.data_ptr(), A_stride * batch_size);

                    h.parallel_for<OrthoNormalizeVector<B, T>>(
                        sycl::nd_range<1>(sycl::range{size_t(batch_size * normalize_wg_size)}, sycl::range{size_t(normalize_wg_size)}), 
                        [=](sycl::nd_item<1> item){
                            auto tid = item.get_local_linear_id();
                            auto bid = item.get_group_linear_id();
                            auto cta = item.get_group();
                            auto A_local_vector = Aspan.subspan(bid * A_stride + i*m, m);
                            auto Anext_squared_span = Span(static_cast<float_t*>(Anext_squared.get_pointer()), m);
                            
                            for (int j = tid; j < m; j+= cta.get_local_linear_range()){
                                Anext_squared_span[j] = square(A_local_vector[j]);
                            }
                            
                            sycl::group_barrier(cta);
                            auto squared_norm = sycl::joint_reduce(cta, Anext_squared.begin(), Anext_squared.end(), sycl::plus<float_t>());
                            auto norm = std::sqrt(squared_norm);

                            for (int j = tid; j < m; j+= cta.get_local_linear_range()){
                                A_local_vector[j] /= norm;
                            }
                        }); 
                });
            }
        };

        auto shift_chol_alg = [&](){
            gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);

            auto ATA_ptr = C.data_ptr();
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
                    auto shift = T(11.0) * T(T(m * k) * T(eps) + T(k + 1) * T(k) * T(eps)) * g_norm;
                    ATA_acc[tid * k + tid] += shift;
                });
            });
            //Compute the Cholesky Factorization of StS
            potrf<B>(ctx, C, Uplo::Lower, potrf_workspace);
            trsm<B>(ctx, C, A, is_A_trans ? Side::Left : Side::Right, Uplo::Lower, inv_trans, Diag::NonUnit, T(1.0));
            chol_alg();
            chol_alg();
        };

        switch (algo) {
            case OrthoAlgorithm::Cholesky:
                chol_alg();
                break;
            case OrthoAlgorithm::Chol2:
                chol_alg();
                chol_alg();
                break;
            case OrthoAlgorithm::ShiftChol3:
                shift_chol_alg();
                break;
            case OrthoAlgorithm::Householder: {
                auto tau = pool.allocate<T>(ctx, k);
                auto geqrf_workspace = pool.allocate<std::byte>(ctx, geqrf_buffer_size<B>(ctx, A, tau));
                auto orgqr_workspace = pool.allocate<std::byte>(ctx, orgqr_buffer_size<B>(ctx, A, tau));
                geqrf<B>(ctx, A, tau, geqrf_workspace);
                orgqr<B>(ctx, A, tau, orgqr_workspace);
                break;
            }
            case OrthoAlgorithm::CGS2:
                cgs_alg();
                break;
            case OrthoAlgorithm::SVQB: {
                auto diags = pool.allocate<float_t>(ctx, batch_size * k);
                auto lambdas = pool.allocate<float_t>(ctx, batch_size * k);
                auto syev_workspace = pool.allocate<std::byte>(ctx, syev_buffer_size<B>(ctx, C, lambdas, JobType::EigenVectors, Uplo::Lower));
                auto output_basis = pool.allocate<T>(ctx, batch_size * m * k);
                //Compute A^H * A
                gemm<B>(ctx, A, A, C, T(1.0), T(0.0), inv_trans, transA);
                //Compute D = diag(A^H * A) ^-1/2
                ctx -> submit([&](sycl::handler& h) {
                    auto ATA_ptr = C.data_ptr();
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * k)}, sycl::range{size_t(k)}), [=](sycl::nd_item<1> item){
                    auto tid = item.get_local_linear_id();
                    auto bid = item.get_group_linear_id();
                    auto ATA_acc = Span(ATA_ptr + bid * k * k, k * k);
                    auto diags_acc = diags.subspan(bid * k, k);
                    diags_acc[tid] = sycl::rsqrt(real_part(ATA_acc[tid * k + tid]));
                    });
                });
                ctx -> wait();
                std::cout << diags << std::endl;
                //Compute StS = D * StS * D
                ctx -> submit([&](sycl::handler& h){
                    auto D_local = sycl::local_accessor<float_t, 1>(k, h);
                    auto ATA_ptr = C.data_ptr();
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size*k)}, sycl::range{size_t(k)}), [=](sycl::nd_item<1> item){
                    auto tid = item.get_local_linear_id();
                    auto bid = item.get_group_linear_id();
                    auto cta = item.get_group();
                    auto D_acc = diags.subspan(bid * k, k);
                    D_local[tid] = D_acc[tid];
                    sycl::group_barrier(cta);
                    auto AtA_acc = Span(ATA_ptr + bid * k * k, k * k);
                    auto D_tid = D_local[tid];
                    for(int i = 0; i < k; i++){
                        auto D_i = D_local[i];
                        AtA_acc[i * k + tid] *= D_tid * D_i;
                    }
                    });
                });
                ctx -> wait();
                std::cout << C << std::endl;

                syev<B>(ctx, C, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
                ctx -> wait();
                std::cout << "Eigenvalues: " << lambdas << std::endl;
                //First Compute D * EigenVectors * Lambda^-1/2
                ctx -> submit([&](sycl::handler& h){
                    auto D_local = sycl::local_accessor<float_t, 1>(k, h);
                    auto C_ptr = C.data_ptr();
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size*k)}, sycl::range{size_t(k)}), [=](sycl::nd_item<1> item){
                    auto tid = item.get_local_linear_id();
                    auto bid = item.get_group_linear_id();
                    auto cta = item.get_group();
                    auto D_acc = diags.subspan(bid * k, k);
                    D_local[tid] = D_acc[tid];
                    sycl::group_barrier(cta);
                    auto C_acc = Span(C_ptr + bid * k * k, k * k);
                    auto D_tid = D_local[tid];
                    auto tau = std::numeric_limits<float_t>::epsilon() * std::abs(lambdas[bid * k + k - 1]);
                    for(int i = 0; i < k; i++){
                        auto lambda_i = lambdas[bid * k + i] < tau ? tau : lambdas[bid * k + i];
                        C_acc[i * k + tid] *= D_tid * sycl::rsqrt(std::abs(real_part(lambda_i)));
                    }
                    });
                });
                //Compute Q = S * D * EigenVectors * Lambda^-1/2
                auto output_view = MatrixView<T,fmt>(output_basis.data(), m, k, m, k*m, batch_size);
                gemm<B>(ctx, A, C, output_view, T(1.0), T(0.0), transA, Transpose::NoTrans);
                //Memcpy
                auto A_stride = A.stride();
                auto A_ld = A.ld();
                auto Adata = A.data();
                auto wgs = std::min(get_kernel_max_wg_size<StridedCopyKernel<B, T>>(ctx), size_t(m * k));
                ctx -> parallel_for<StridedCopyKernel<B,T>>(sycl::nd_range<1>(sycl::range{size_t(batch_size * wgs)}, sycl::range{size_t(wgs)}), [=](sycl::nd_item<1> item){
                    auto batch_idx = item.get_group().get_id(0);
                    for (int linear_ix = item.get_local_linear_id(); linear_ix < m * k; linear_ix += item.get_group().get_local_linear_range()) {
                    auto i = linear_ix % m;
                    auto j = linear_ix / m;
                    Adata[batch_idx * A_stride + j * A_ld + i] = output_basis[batch_idx * m * k + j*m + i];
                    }
                });
                break;
            }
            default:
                throw std::runtime_error("Unknown orthogonalization algorithm");
        }
        
        return ctx.get_event();
    }

    template <Backend B, typename T>
    size_t ortho_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Transpose transA,
                             OrthoAlgorithm algo) {
        size_t size = 0;
        auto [m, k] = get_effective_dims(A, transA);
        auto batch_size = A.batch_size();
        
        // Create temporary matrices for calculating buffer sizes
        auto temp_C = MatrixView<T, MatrixFormat::Dense>(nullptr, k, k, k, k * k, batch_size);
        auto temp_view = MatrixView<T, MatrixFormat::Dense>(nullptr, m, k, m, m * k, batch_size);
        auto is_cholesky = algo == OrthoAlgorithm::Cholesky || algo == OrthoAlgorithm::Chol2 || algo == OrthoAlgorithm::ShiftChol3;

        auto mem_for_svqb = algo == OrthoAlgorithm::SVQB ? 
            (BumpAllocator::allocation_size<std::byte>(ctx, syev_buffer_size<B>(ctx, temp_view, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower)) +
            BumpAllocator::allocation_size<T>(ctx, k * batch_size) +
            BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, k * batch_size) +
            BumpAllocator::allocation_size<T>(ctx, m * k * batch_size)) : 0;
        
        auto mem_for_cgs = algo == OrthoAlgorithm::CGS2 ? 
            (BumpAllocator::allocation_size<T>(ctx, m * batch_size)) : 0;

        auto mem_for_householder = algo == OrthoAlgorithm::Householder ? 
            (BumpAllocator::allocation_size<T>(ctx, k) + 
             BumpAllocator::allocation_size<std::byte>(ctx, geqrf_buffer_size<B>(ctx, temp_view, Span<T>(nullptr, k))) +
             BumpAllocator::allocation_size<std::byte>(ctx, orgqr_buffer_size<B>(ctx, temp_view, Span<T>(nullptr, k)))) : 0;
        
        return  BumpAllocator::allocation_size<std::byte>(ctx, is_cholesky ? potrf_buffer_size<B>(ctx, temp_C, Uplo::Lower) : 0) +
                BumpAllocator::allocation_size<T>(ctx, k*k*batch_size) +
                BumpAllocator::allocation_size<T>(ctx, m*batch_size)+ 
                BumpAllocator::allocation_size<T>(ctx, m*k*batch_size) +
                mem_for_svqb +
                mem_for_cgs +
                mem_for_householder;

    }

    template <Backend B, typename T>
    Event ortho(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& M,
                Transpose transA,
                Transpose transM,
                Span<std::byte> workspace,
                OrthoAlgorithm algo,
                size_t iterations) {
        BumpAllocator pool(workspace);
        constexpr auto fmt = MatrixFormat::Dense;
        //When orthogonalizing against an external basis M,
        //M must be an orthonormal basis
        //Both A and M must be either tall-and-skinny or short-and-fat
        //Furthermore the number of vectors in A and M must sum to at most the dimension of these vectors 
        auto nM = transM == Transpose::NoTrans ? M.cols_ : M.rows_;
        auto nA = transA == Transpose::NoTrans ? A.cols_ : A.rows_;
        auto k = transA == Transpose::NoTrans ? A.rows_ : A.cols_;
        
        // Initialize the matrices if not already done
        if(nA + nM > k){
            throw std::runtime_error("The number of vectors in A (" + std::to_string(nA) + ") and M (" + std::to_string(nM) + ") must sum to at most the dimension of these vectors (" + std::to_string(k) + ")");
        }
        assert(k == (transM == Transpose::NoTrans ? M.rows_ : M.cols_));

        auto inv_transA = transA == Transpose::Trans ? Transpose::NoTrans : Transpose::Trans;
        auto inv_transM = transM == Transpose::Trans ? Transpose::NoTrans : Transpose::Trans;
        auto batch_size = A.batch_size();
        auto MAmem = pool.allocate<T>(ctx, nM*nA * batch_size);
        auto orthoworkspace = pool.allocate<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo));
        auto descrMA = MatrixView<T, fmt>(MAmem.data(), nM, nA, nM, nM*nA, batch_size);
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

    template <Backend B, typename T>
    size_t ortho_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& M,
                             Transpose transA,
                             Transpose transM,
                             OrthoAlgorithm algo,
                             size_t iterations) {
        auto nM = transM == Transpose::NoTrans ? M.cols_ : M.rows_;
        auto nA = transA == Transpose::NoTrans ? A.cols_ : A.rows_;
        auto batch_size = A.batch_size();
        
        return  BumpAllocator::allocation_size<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo)) +
                BumpAllocator::allocation_size<T>(ctx, nM*nA * batch_size);
    }  

    #define ORTHO_INSTANTIATE(back, fp) \
    template Event ortho<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm); \
    template Event ortho<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm, \
        size_t); \
    template size_t ortho_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        OrthoAlgorithm); \
    template size_t ortho_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Transpose, \
        Transpose, \
        OrthoAlgorithm, \
        size_t);

    // Instantiate for the floating-point types of interest
    #define INSTANTIATE_ORTHO_FOR_BACKEND(back)\
        ORTHO_INSTANTIATE(back, float) \
        ORTHO_INSTANTIATE(back, double)\
        ORTHO_INSTANTIATE(back, std::complex<float>)\
        ORTHO_INSTANTIATE(back, std::complex<double>)

    #if BATCHLAS_HAS_CUDA_BACKEND
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::CUDA)
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND 
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::ROCM)
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND 
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::NETLIB)
    #endif

    #undef ORTHO_INSTANTIATE
}