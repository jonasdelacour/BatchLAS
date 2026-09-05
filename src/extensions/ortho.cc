#include "../linalg-impl.hh"
#include <batchlas/blas/dispatch/route_compiled.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <numeric>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>

#include "../util/template-instantiations.hh"


// High-level orthogonalization functions built on top of primitive BLAS operations
// Implementation using the new MatrixView structure
namespace batchlas {

    template <Backend B, typename T>
    struct OrthoNormalizeVector {};
    
    template <Backend B, typename T>
    struct StridedCopyKernel {};

    template <typename T>
    struct OrthoWorkspace {
        MatrixView<T, MatrixFormat::Dense> C;   // k x k Gram matrix A^H A
        Span<std::byte> potrf_ws;
        Span<typename base_type<T>::type> diags;
        Span<typename base_type<T>::type> lambdas;
        Span<std::byte> syev_ws;
        Span<T> output_basis;
        Span<T> Ymem;                           // CGS2 only
        Span<T> tau;                            // Householder only
        Span<std::byte> geqrf_ws;
        Span<std::byte> orgqr_ws;
    };

    // Single description of ortho's workspace; see workspace_bytes() in
    // util/mempool.hh.
    //
    // Every branch is allocated with a zero size when its algorithm was not
    // selected, which is how the mutually exclusive tails (CGS2's Ymem versus
    // Householder's tau/geqrf/orgqr) stay in one linear description.
    //
    // The nested size queries are asked about C, `lambdas` and `tau`, all of
    // which live in this workspace. That is sound only because none of those
    // queries dereferences the pointer it is given -- the previous sizing code
    // passed literal nullptrs to exactly these calls.
    template <Backend B, typename T>
    OrthoWorkspace<T> ortho_layout(Queue& ctx,
                                   BumpAllocator& pool,
                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                   int64_t m,
                                   int64_t k,
                                   OrthoAlgorithm algo) {
        using float_t = typename base_type<T>::type;
        constexpr auto fmt = MatrixFormat::Dense;
        const auto batch_size = A.batch_size();

        const bool is_cholesky = algo == OrthoAlgorithm::Cholesky || algo == OrthoAlgorithm::Chol2 ||
                                 algo == OrthoAlgorithm::ShiftChol3;
        const bool is_svqb = algo == OrthoAlgorithm::SVQB || algo == OrthoAlgorithm::SVQB2;
        const bool is_cgs = algo == OrthoAlgorithm::CGS2;
        const bool is_householder = algo == OrthoAlgorithm::Householder;

        auto ATA = pool.allocate<T>(ctx, k * k * batch_size);
        auto matAmem = pool.allocate<T*>(ctx, batch_size);
        auto matATAmem = pool.allocate<T*>(ctx, batch_size);
        static_cast<void>(matAmem);

        auto C = MatrixView<T, fmt>(ATA.data(), k, k, k, k * k, batch_size, matATAmem.data());
        auto potrf_ws = pool.allocate<std::byte>(ctx, is_cholesky ? potrf_buffer_size<B>(ctx, C, Uplo::Lower) : 0);

        auto diags = pool.allocate<float_t>(ctx, is_svqb ? batch_size * k : 0);
        auto lambdas = pool.allocate<float_t>(ctx, is_svqb ? batch_size * k : 0);
        auto syev_ws = pool.allocate<std::byte>(
            ctx, is_svqb ? syev_buffer_size<B>(ctx, C, lambdas, JobType::EigenVectors, Uplo::Lower) : 0);
        auto output_basis = pool.allocate<T>(ctx, is_svqb ? batch_size * m * k : 0);

        auto Ymem = pool.allocate<T>(ctx, is_cgs ? batch_size * m : 0);

        auto tau = pool.allocate<T>(ctx, is_householder ? k * batch_size : 0);
        auto geqrf_ws = pool.allocate<std::byte>(ctx, is_householder ? geqrf_buffer_size<B>(ctx, A, tau) : 0);
        auto orgqr_ws = pool.allocate<std::byte>(ctx, is_householder ? orgqr_buffer_size<B>(ctx, A, tau) : 0);

        return {C, potrf_ws, diags, lambdas, syev_ws, output_basis, Ymem, tau, geqrf_ws, orgqr_ws};
    }

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
        // `B == Backend::NETLIB` meant "there are no device kernels here", which
        // is a property of the DEVICE, not of the backend enum. Asked directly it
        // stays correct for a host queue reached through any backend. Today's
        // outcome is unchanged: NETLIB is the only backend that runs on a host
        // device in this tree.
        if (ctx.device().type != DeviceType::GPU) {
            algo = OrthoAlgorithm::Householder;
        }
        bool is_A_trans = transA == Transpose::Trans || transA == Transpose::ConjTrans;
        Transpose inv_trans = is_A_trans ? Transpose::NoTrans : 
                            std::is_same_v<T, std::complex<float_t>> ? Transpose::ConjTrans : Transpose::Trans;
        assert(k <= m);
        //If k > m && transA == NoTrans the columns of A are linearly dependent
        //Else if k > m && transA == Trans the rows of A are linearly dependent
        auto wsl = ortho_layout<B, T>(ctx, pool, A, m, k, algo);
        auto C = wsl.C;
        auto potrf_workspace = wsl.potrf_ws;
        auto ATA_stride = k * k;

        // C = A^H A is a Gram matrix, which is exactly what syrk spells, and it
        // does half the arithmetic a GEMM does. PR #61 measured this
        // substitution and rejected it -- correctly at the time, because syrk
        // reached no batched kernel at these shapes and fell to a host loop over
        // cublasXsyrk: 96x slower in float, and 115 ms against a 0.9 ms GEMM in
        // double. `syrk_gram_tiles` is that missing kernel.
        //
        // Two conditions, both measured rather than assumed (RTX 4090 / sm_89,
        // batches that saturate):
        //
        //   k          the single-tile Gram kernel lives at k <= 128, but the
        //              useful limit differs by precision and the end-to-end
        //              numbers say so (m = 1024, batch 512, Chol2):
        //
        //                k    float           double
        //                32   1.62x           1.02x
        //                64   1.12x           1.20x
        //                128  0.96x  <- loss  1.34x
        //
        //              Float at k = 128 is a wash because the SGEMM it replaces
        //              is already against both the compute and the bandwidth
        //              roof, so there is nothing for the halved arithmetic to
        //              buy. FP64 runs at 1/64 rate on this part, so double is
        //              squarely compute bound and the halving lands in full --
        //              and it grows with k, where float's shrinks. Hence 64 for
        //              float and 128 for double, not one number for both.
        //              Above those, double and complex are still on the host
        //              loop, which at k = 256 loses to the GEMM by 2x.
        //   real only  a complex multiply is four real ones, so herk is compute
        //              bound where syrk is bandwidth bound, and the existing
        //              GEMM-plus-Hermitian-fold beats the tile kernel at every
        //              Gram shape. Complex keeps the GEMM.
        //
        // Only the lower triangle is produced. Everything downstream of these
        // two call sites reads exactly that -- potrf and trsm both default to
        // Uplo::Lower, and shift_chol_alg's shift kernel touches only the
        // diagonal. svqb_alg is the exception and keeps its GEMM: it scales the
        // whole k x k before handing it to syev, so a half-written C would leave
        // it multiplying uninitialised workspace.
        // BATCHLAS_ORTHO_GRAM=gemm pins the old spelling, so the substitution
        // stays measurable from one binary rather than needing a build of the
        // parent commit to compare against.
        constexpr bool gram_is_real = !sycl::detail::is_complex<T>::value;
        const bool gram_pinned_to_gemm = [] {
            const char* raw = std::getenv("BATCHLAS_ORTHO_GRAM");
            return raw != nullptr && std::string(raw) == "gemm";
        }();
        constexpr int gram_max_k = std::is_same_v<T, float> ? 64 : 128;
        // "syrk reaches the gram tile kernel on this route", not "this is NVIDIA".
        const bool gram_via_syrk =
            dispatch::level3_tile_kernels_compiled<B> && gram_is_real &&
            k <= gram_max_k && !gram_pinned_to_gemm;
        auto gram_into_C = [&](const auto& in_mat) {
            if constexpr (dispatch::level3_tile_kernels_compiled<B> && gram_is_real) {
                if (gram_via_syrk) {
                    return syrk<B, T>(ctx, in_mat, C, T(1), T(0), Uplo::Lower, inv_trans);
                }
            }
            return gemm<B>(ctx, in_mat, in_mat, C, {.transA = inv_trans, .transB = transA});
        };


        auto real_part = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return value.real(); else return value; };
        auto square = [](T value) { if constexpr (sycl::detail::is_complex<T>::value) return (value * std::conj(value)).real(); else return value * value; };
        
        auto chol_alg = [&](){
            constexpr T alpha = 1.0;
            constexpr T beta = 0.0;
            //Compute StS = S^T * S or StS = S * S^T (depending on transA)
            gram_into_C(A);
            //Compute the Cholesky Factorization of StS
            potrf<B>(ctx, C, PotrfOptions{}, potrf_workspace);
            //Solve X * Chol(StS) = S
            trsm<B>(ctx,
                    C,
                    A,
                    {.alpha = alpha, .side = is_A_trans ? Side::Left : Side::Right, .trans = inv_trans});
        };

        auto cgs_alg = [&](){
            //Implemented as an iterative process:
            //1. Compute orthogonality of A[:,0 .. k-1] and A[:,k .. m-1]
            //2. Subtract the projection of A[:,k .. m-1] onto A[:,0 .. k-1]
            //3. Normalize A[:,0 .. k-1]
            //Repeat until all vectors are orthogonal
            auto Ymem = wsl.Ymem;
            auto normalize_wg_size = std::min(get_kernel_max_wg_size<OrthoNormalizeVector<B, T>>(ctx), size_t(m));
            for (int i = 0; i < k; i++){
                //View of the first i vectors (either columns or rows of A depending on transA)
                auto A_i = transA == Transpose::NoTrans ? 
                      MatrixView<T, fmt>(A.data_ptr(), m, i, m, A.stride(), batch_size) 
                    : MatrixView<T, fmt>(A.data_ptr(), i, m, m, A.stride(), batch_size);
                //View of the next vector (either column or row of A depending on transA)
                auto C = VectorView(Ymem.data(), i, batch_size);
                auto A_next = A(Slice(), i);  //VectorView(A.data_ptr() + m * i, m, batch_size, 1, A.stride());
                //output vector
                if (i > 0){ //If it's the first vector we just need to normalize it
                    for (int j = 0; j < 2; j++){
                        gemv<B>(ctx, A_i, A_next, C, {.transA = inv_trans});
                        gemv<B>(ctx,
                                A_i,
                                C,
                                A_next,
                                {.alpha = T(-1.0), .beta = T(1.0), .transA = transA});
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
                            auto Anext_squared_span = Span(static_cast<float_t*>(util::get_raw_ptr(Anext_squared)), m);
                            
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
            gram_into_C(A);

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
            potrf<B>(ctx, C, PotrfOptions{}, potrf_workspace);
            trsm<B>(ctx, C, A, {.side = is_A_trans ? Side::Left : Side::Right, .trans = inv_trans});
            chol_alg();
            chol_alg();
        };
        
        auto diags = wsl.diags;
        auto lambdas = wsl.lambdas;
        auto syev_workspace = wsl.syev_ws;
        auto output_basis = wsl.output_basis;

        auto svqb_alg = [&](auto in_mat, auto out_mat) {
            //Compute A^H * A
            gemm<B>(ctx, in_mat, in_mat, C, {.transA = inv_trans, .transB = transA});
            //Compute D = diag(A^H * A) ^-1/2
            ctx -> submit([&](sycl::handler& h) {
                auto ATA_ptr = C.data_ptr();
                h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * k)}, sycl::range{size_t(k)}), [=](sycl::nd_item<1> item){
                auto tid = item.get_local_linear_id();
                auto bid = item.get_group_linear_id();
                auto ATA_acc = Span(ATA_ptr + bid * k * k, k * k);
                auto diags_acc = diags.subspan(bid * k, k);

                // D = diag(A^H A)^(-1/2)
                // Guard against exact / flushed-to-zero diagonals to avoid Inf/NaN from rsqrt(0).
                // Keep the threshold extremely small so we don't distort legitimate (small) column norms.
                const float_t diag = sycl::fmax(float_t(0), real_part(ATA_acc[tid * k + tid]));
                const float_t tau = std::numeric_limits<float_t>::min();
                diags_acc[tid] = (diag <= tau) ? float_t(0) : sycl::rsqrt(diag);
                });
            });
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

            syev<B>(ctx, C, lambdas, SyevOptions{}, syev_workspace);

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
            gemm<B>(ctx, in_mat, C, out_mat, {.transA = transA});
            //Memcpy
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
                geqrf<B>(ctx, A, wsl.tau, wsl.geqrf_ws);
                orgqr<B>(ctx, A, wsl.tau, wsl.orgqr_ws);
                break;
            }
            case OrthoAlgorithm::CGS2:
                cgs_alg();
                break;
            case OrthoAlgorithm::SVQB: {
                auto output_view = MatrixView<T,fmt>(output_basis.data(), m, k, m, k*m, batch_size);
                svqb_alg(A, output_view);
                auto A_stride = A.stride();
                auto A_ld = A.ld();
                auto Adata = A.data();
                auto wgs = std::min(get_kernel_max_wg_size<StridedCopyKernel<B, T>>(ctx), size_t(m * k));
                ctx -> parallel_for<StridedCopyKernel<B,T>>(sycl::nd_range<1>(sycl::range{size_t(batch_size * wgs)}, sycl::range{size_t(wgs)}), [=](sycl::nd_item<1> item){
                    auto batch_idx = item.get_group().get_group_id()[0];
                    for (int linear_ix = item.get_local_linear_id(); linear_ix < m * k; linear_ix += item.get_group().get_local_linear_range()) {
                    auto i = linear_ix % m;
                    auto j = linear_ix / m;
                    Adata[batch_idx * A_stride + j * A_ld + i] = output_basis[batch_idx * m * k + j*m + i];
                    }
                });
                break;
            }
            case OrthoAlgorithm::SVQB2: {
                auto output_view = MatrixView<T,fmt>(output_basis.data(), m, k, m, k*m, batch_size);
                svqb_alg(A, output_view);
                svqb_alg(output_view, A);
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
        auto [m, k] = get_effective_dims(A, transA);
        if constexpr (B == Backend::NETLIB) {
            algo = OrthoAlgorithm::Householder;
        }
        return workspace_bytes([&, m = m, k = k](BumpAllocator& pool) {
            return ortho_layout<B, T>(ctx, pool, A, m, k, algo);
        });
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
        if constexpr (B == Backend::NETLIB) {
            algo = OrthoAlgorithm::Householder;
        }
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
        auto trans = sycl::detail::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;
        auto no_trans = Transpose::NoTrans;
        auto inv_transA = transA == trans ? no_trans : trans;
        auto inv_transM = transM == trans ? no_trans : trans;
        auto batch_size = A.batch_size();
        auto MAmem = pool.allocate<T>(ctx, nM*nA * batch_size);
        auto orthoworkspace = pool.allocate<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo));
        auto descrMA = MatrixView<T, fmt>(MAmem.data(), nM, nA, nM, nM*nA, batch_size);
        auto isAtrans = transA == trans;
        auto is_first_transposed = static_cast<Transpose>(((transA == trans) || (transM == trans)));
        auto is_second_transposed = static_cast<Transpose>(((transA == trans) && (transM == no_trans)));
        
        for (size_t i = 0; i < iterations; i++){
            gemm<B>(ctx, M, A, descrMA, {.transA = inv_transM, .transB = transA});
            gemm<B>(ctx,
                    isAtrans ? descrMA : M,
                    isAtrans ? M : descrMA,
                    A,
                    {.alpha = T(-1.0), .beta = T(1.0), .transA = is_first_transposed, .transB = is_second_transposed});

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
        if constexpr (B == Backend::NETLIB) {
            algo = OrthoAlgorithm::Householder;
        }
        
        return  BumpAllocator::allocation_size<std::byte>(ctx, ortho_buffer_size<B>(ctx, A, transA, algo)) +
                BumpAllocator::allocation_size<T>(ctx, nM*nA * batch_size);
    }  

    #define ORTHO_INSTANTIATE(back, fp) \
    template Event ortho<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm); \
    template Event ortho<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Transpose, \
        Transpose, \
        Span<std::byte>, \
        OrthoAlgorithm, \
        size_t); \
    template size_t ortho_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Transpose, \
        OrthoAlgorithm); \
    template size_t ortho_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        Transpose, \
        Transpose, \
        OrthoAlgorithm, \
        size_t);

    // Instantiate for the floating-point types of interest
    #define INSTANTIATE_ORTHO_FOR_BACKEND(back)\
        BATCHLAS_FOR_EACH_SCALAR_TYPE_1(ORTHO_INSTANTIATE, back)

    #if BATCHLAS_HAS_CUDA_BACKEND
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::CUDA)
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND 
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::ROCM)
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND 
        INSTANTIATE_ORTHO_FOR_BACKEND(Backend::NETLIB)
    #endif

    #undef INSTANTIATE_ORTHO_FOR_BACKEND
    #undef ORTHO_INSTANTIATE
}