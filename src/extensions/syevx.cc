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
#include <blas/functions/syev.hh>
#include "../math-helpers.hh"
#include <internal/sort.hh>

namespace batchlas {
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxResidualsKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxReverseEigenvectorsKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxReverseEigenvectorsStasKernel;

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

        const bool trace_enabled = []() {
            const char* v = std::getenv("BATCHLAS_SYEVX_TRACE");
            if (!v) return false;
            return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
                    std::string(v) == "on" || std::string(v) == "ON");
        }();
        auto trace = [&](const char* msg) {
            if (!trace_enabled) return;
            std::cout << msg << std::endl;
        };
        auto trace_wait = [&](const char* msg) {
            if (!trace_enabled) return;
            std::cout << msg << std::endl;
            ctx.wait_and_throw();
        };
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
        auto converged_flags = pool.allocate<int32_t>(ctx, batch_size);

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
        // XtAX is a per-batch (block_vectors x block_vectors) matrix. It lives in the
        // top-left corner of the backing StAS_base buffer for each batch.
        // IMPORTANT: keep StAS_base's stride so batches do not overlap.
        auto XtAX = StAS_base({0, block_vectors}, {0, block_vectors});
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

        // NOTE: SYEVX repeatedly solves *tiny* dense eigenproblems (XtAX, StAS).
        // For diagnosis and benchmarking, allow opting into/out of using the vendor
        // implementation for these projected solves.
        const bool prefer_vendor_projected_syev =
            (B != Backend::NETLIB) &&
            ([]() {
                if (const char* v = std::getenv("BATCHLAS_SYEVX_PROJECTED_VENDOR")) {
                    return (v[0] == '1') || (v[0] == 't') || (v[0] == 'T') || (v[0] == 'y') || (v[0] == 'Y');
                }
                return false;
            })();

        // NOTE: syevx relies on repeated small eigenproblems (XtAX, StAS).
        // The chosen SYEV provider can change with matrix size (e.g. CTA for n<=32
        // but blocked/vendor for larger n), so a single pre-sized workspace must cover
        // the maximum of the internal problems.
        const size_t ws_xtax = syev_buffer_size<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
        const size_t ws_stas = syev_buffer_size<B>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
        size_t ws_projected = std::max(ws_xtax, ws_stas);
        if (prefer_vendor_projected_syev) {
            const size_t ws_xtax_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
            const size_t ws_stas_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
            ws_projected = std::max(ws_projected, std::max(ws_xtax_vendor, ws_stas_vendor));
        }
        auto syev_workspace = pool.allocate<std::byte>(ctx, ws_projected);
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
        trace("syevx: ortho init");
        ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
        trace_wait("syevx: ortho init done");
        //Compute AX
        if constexpr (MFormat == MatrixFormat::Dense) {
            trace("syevx: gemm A*X");
            gemm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        } else {
            //For sparse matrices we use the spmm function
            trace("syevx: spmm A*X");
            spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
        }
        trace_wait("syevx: A*X done");
        //Compute X^T AX
        trace("syevx: gemm X^T*(A*X)");
        gemm<B>(ctx, X, AX, XtAX, T(1.0), T(0.0), trans, Transpose::NoTrans);
        trace_wait("syevx: XtAX gemm done");
        //Solve the eigenvalue problem
        trace("syevx: syev XtAX");
        if (prefer_vendor_projected_syev) {
            backend::syev_vendor<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
        } else {
            syev<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
        }
        trace_wait("syevx: syev XtAX done");

        // If we are looking for the largest eigenpairs, reorder the eigenvectors so that
        // the first columns correspond to the largest eigenvalues.
        // Matrix storage is column-major (BLAS), so reversing the eigenvector order means
        // swapping columns, not rows.
        if (params.find_largest) {
            const auto XtAX_ptr = XtAX.data_ptr();
            const int64_t XtAX_stride = XtAX.stride();
            const int64_t XtAX_ld = XtAX.ld();
            const int64_t k = block_vectors;
            constexpr size_t wg = 256;
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxReverseEigenvectorsKernel<B, T, MFormat>>(
                    sycl::nd_range<1>(sycl::range{size_t(batch_size * wg)}, sycl::range{wg}),
                    [=](sycl::nd_item<1> item) {
                        const auto tid = item.get_local_linear_id();
                        const auto bid = item.get_group_linear_id();
                        auto* mat = XtAX_ptr + bid * XtAX_stride;

                        const int64_t half_cols = k / 2;
                        const int64_t swap_count = k * half_cols;
                        for (int64_t linear = int64_t(tid); linear < swap_count; linear += int64_t(item.get_local_range(0))) {
                            const int64_t row = linear % k;
                            const int64_t col = linear / k;
                            const int64_t col2 = (k - 1) - col;
                            const int64_t idx1 = row + col * XtAX_ld;
                            const int64_t idx2 = row + col2 * XtAX_ld;
                            const auto tmp = mat[idx1];
                            mat[idx1] = mat[idx2];
                            mat[idx2] = tmp;
                        }
                    });
            });
            trace_wait("syevx: reverse XtAX eigenvectors done");
        }
        //Update X and corresponding implicit update of AX
        trace("syevx: gemm X*Z (update X)");
        gemm<B>(ctx, X, XtAX, X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        trace_wait("syevx: update X done");

        trace("syevx: gemm AX*Z (update AX)");
        gemm<B>(ctx, AX, XtAX, AX_new , T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
        trace_wait("syevx: update AX done");

        swap_subspace();
        bool restart = true;

        // Tracks how many eigenvalues are currently stored per batch in `lambdas`.
        // After the initial XtAX solve, this is `block_vectors`. After subsequent StAS
        // solves it becomes 2*block_vectors (restart step) or 3*block_vectors.
        int64_t current_num_eigvals = block_vectors;

        size_t residual_wg_size = std::min(get_kernel_max_wg_size<SyevxResidualsKernel<B,T,MFormat>>(ctx), size_t(n));

        //Compute R = AX - X * diag(lambdas)
        for(int it = 0; it < params.iterations; it++){
            int Nvecs = restart ? block_vectors * 2 : block_vectors * 3;
            //Compute R = AX - X * diag(lambdas)
            trace("syevx: residual kernel submit");
            const float_type abs_tol = static_cast<float_type>(std::abs(params.absolute_tolerance));
            const float_type rel_tol = static_cast<float_type>(std::abs(params.relative_tolerance));
            const float_type tol = std::max(abs_tol, rel_tol);
            ctx -> submit([&](sycl::handler& h){
                auto Rdata = R.data_ptr();
                auto Xdata = X.data_ptr();
                auto AXdata = AX.data_ptr();
                auto flags = converged_flags.data();
                h.parallel_for<SyevxResidualsKernel<B,T,MFormat>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*residual_wg_size)}, sycl::range{size_t(residual_wg_size)}), [=](sycl::nd_item<1> item){
                    auto num_eigvals = it < 2 ? (it+1) * block_vectors : 3*block_vectors;

                    auto tid = item.get_local_linear_id();
                    sycl::group<1> cta = item.get_group();
                    const auto local_size = item.get_local_range(0);
                    auto bid = item.get_group_linear_id();
                    auto blockR = Span(Rdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockX = Span(Xdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockAX = Span(AXdata + block_vectors*n*3*bid, block_vectors*n);
                    auto blockLambdas = lambdas.subspan(bid * (num_eigvals), num_eigvals);
                    auto blockresiduals = residuals.subspan(bid * (neigs), neigs);
                    auto blockbestresiduals = best_residuals.subspan(bid * (neigs), neigs);
                    auto blockW = W.subspan(bid * (neigs), neigs);
    
                    sycl::group_barrier(cta);
                    for (int i = tid; i < n*block_vectors; i+=local_size){
                        auto eigvect_id = i / n;
                        auto eigval = blockLambdas[params.find_largest ? (num_eigvals - 1 - eigvect_id) : eigvect_id];
                        blockR[i] = blockAX[i] - blockX[i] * eigval;
                    }
                    sycl::group_barrier(cta);
                    
                    for (size_t i = 0; i < neigs; i++){
                        float_type r_partial = 0;
                        float_type x_partial = 0;
                        for (int j = int(tid); j < n; j += int(local_size)){
                            r_partial += internal::norm_squared(blockR[int(i)*n + j]);
                            x_partial += internal::norm_squared(blockX[int(i)*n + j]);
                        }

                        const float_type r_sum = sycl::reduce_over_group(cta, r_partial, sycl::plus<float_type>());
                        const float_type x_sum = sycl::reduce_over_group(cta, x_partial, sycl::plus<float_type>());
                        if (tid == 0){
                            auto residual = sycl::sqrt(r_sum);
                            const auto x_norm = sycl::sqrt(x_sum);
                            const auto eigval = blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                            const auto denom = x_norm * sycl::fabs(eigval);
                            if (denom > float_type(0)){
                                residual /= denom;
                            }
                            blockresiduals[i] = residual;
                        }
                    }
                    
                    sycl::group_barrier(cta);
                    if (tid < neigs){
                        auto bestresidual = blockbestresiduals[tid];
                        auto residual = blockresiduals[tid];
                        const bool update = (bestresidual > residual) || (it == 0);
                        if (update){
                            blockbestresiduals[tid] = residual;
                            blockW[tid] = blockLambdas[params.find_largest ? (num_eigvals - 1 - tid) : tid];
                        }
                    }

                    sycl::group_barrier(cta);
                    if (tid == 0) {
                        int32_t ok = 1;
                        for (size_t i = 0; i < neigs; ++i) {
                            if (blockbestresiduals[i] > tol) {
                                ok = 0;
                                break;
                            }
                        }
                        flags[bid] = ok;
                    }
                });
            });

            trace_wait("syevx: residual kernel done");

            // Early exit once all batches have converged for the requested eigenpairs.
            // This is intentionally conservative: it checks the best residual so far.
            bool all_converged = true;
            for (int64_t b = 0; b < batch_size; ++b) {
                if (converged_flags[static_cast<std::size_t>(b)] == 0) {
                    all_converged = false;
                    break;
                }
            }
            if (all_converged) {
                break;
            }

            trace("syevx: ortho R vs (X or XP)");
            ortho<B>(ctx, R, restart ? X : XP, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            trace_wait("syevx: ortho R done");

            if (restart){
                trace("syevx: restart shift P<-R (device copy)");
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
                trace_wait("syevx: restart shift done");
            }
            //Compute AR
            if constexpr (MFormat == MatrixFormat::Dense) {
                trace("syevx: gemm A*(P or R)");
                gemm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            } else {
                trace("syevx: spmm A*(P or R)");
                spmm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            }
            trace_wait("syevx: A*(P or R) done");

            // StAS is stored in a backing buffer sized for (3*block_vectors)x(3*block_vectors).
            // When taking a logical Nvecs x Nvecs view we must preserve the backing ld/stride,
            // otherwise batched matrices overlap and cuSolver/BLAS will read/write out of bounds.
            auto StAS = MatrixView(StAS_base, Nvecs, Nvecs, StAS_base.ld(), StAS_base.stride());
            //Compute S^T A S
            trace("syevx: gemm S^T*(A*S) (StAS)");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), AS({0,n}, {0,Nvecs}), StAS, T(1.0), T(0.0), trans, Transpose::NoTrans);
            trace_wait("syevx: StAS gemm done");
            //Solve the eigenvalue problem
            trace("syevx: syev StAS");
            if (prefer_vendor_projected_syev) {
                backend::syev_vendor<B>(ctx, StAS, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
            } else {
                syev<B>(ctx, StAS, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
            }
            trace_wait("syevx: syev StAS done");
            current_num_eigvals = static_cast<int64_t>(Nvecs);

            trace("syevx: post syev StAS (host)");

            // syev returns eigenvalues in ascending order.
            // For find_largest=true, we take the last `block_vectors` eigenvectors, but we also
            // need column 0 to correspond to the *largest* eigenvalue to stay consistent with
            // the residual computation and W ordering (largest-first).
            const int64_t eig_col_start = params.find_largest ? (Nvecs - block_vectors) : 0;
            if (params.find_largest) {
                auto* StAS_ptr = StAS.data_ptr();
                const int64_t StAS_stride = StAS.stride();
                const int64_t StAS_ld = StAS.ld();
                trace("syevx: reverse selected StAS eigvec block submit");
                ctx->submit([&](sycl::handler& h) {
                    h.parallel_for<SyevxReverseEigenvectorsStasKernel<B, T, MFormat>>(
                        sycl::nd_range<1>(sycl::range{size_t(batch_size * 256)}, sycl::range{size_t(256)}),
                        [=](sycl::nd_item<1> item) {
                            const int64_t tid = int64_t(item.get_local_linear_id());
                            const int64_t bid = int64_t(item.get_group_linear_id());
                            const int64_t local_size = int64_t(item.get_local_range(0));
                            auto* mat = StAS_ptr + bid * StAS_stride;

                            const int64_t half_cols = block_vectors / 2;
                            const int64_t swap_count = Nvecs * half_cols;
                            for (int64_t linear = tid; linear < swap_count; linear += local_size) {
                                const int64_t row = linear % Nvecs;
                                const int64_t c = linear / Nvecs;
                                const int64_t col1 = eig_col_start + c;
                                const int64_t col2 = eig_col_start + (block_vectors - 1 - c);
                                const int64_t idx1 = row + col1 * StAS_ld;
                                const int64_t idx2 = row + col2 * StAS_ld;
                                const auto tmp = mat[idx1];
                                mat[idx1] = mat[idx2];
                                mat[idx2] = tmp;
                            }
                        });
                });
                trace_wait("syevx: reverse selected StAS eigvec block done");
            }
            auto Z = StAS({0, Nvecs}, {eig_col_start, eig_col_start + block_vectors});
            //X(i+1) =  [X(i), R(i), P(i)] * [Zx, Zr, Zp]^T

            //Compute C_p = C_x - [I 
            //                     0
            //                     0]
            trace("syevx: build C_p submit");
            ctx -> submit([&](sycl::handler& h){
                const int64_t Nactive = block_vectors; // When we start soft-locking, update this
                const auto* Z_ptr = Z.data_ptr();
                const int64_t Z_stride = Z.stride();
                const int64_t Z_ld = Z.ld();
                auto* Cp_ptr = C_p.data_ptr();
                const int64_t Cp_stride = C_p.stride();
                const int64_t Cp_ld = C_p.ld();

                h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * 256)}, sycl::range{size_t(256)}), [=](sycl::nd_item<1> item){
                    const int64_t tid = int64_t(item.get_local_linear_id());
                    const int64_t bid = int64_t(item.get_group_linear_id());
                    const int64_t local_size = int64_t(item.get_local_range(0));

                    const auto* Zb = Z_ptr + bid * Z_stride;
                    auto* Cpb = Cp_ptr + bid * Cp_stride;

                    // C_p = Z - I (on the first Nactive columns/rows)
                    const int64_t total = Nvecs * Nactive;
                    for (int64_t linear = tid; linear < total; linear += local_size) {
                        const int64_t row = linear % Nvecs;
                        const int64_t col = linear / Nvecs;
                        const int64_t idxZ = row + col * Z_ld;
                        const int64_t idxC = row + col * Cp_ld;
                        auto v = Zb[idxZ];
                        if (row == col) {
                            v -= T(1);
                        }
                        Cpb[idxC] = v;
                    }
                });
            });
            trace_wait("syevx: build C_p done");
            


            //Compute new search directions
            //X = [X, P, R] * C_x
            trace("syevx: update X/AX submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), Z, X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AX: AX = [AX, AP, AR] * C_x
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), Z, AX_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Orthonormalize C_p against the best eigenvectors
            trace("syevx: ortho C_p vs Z submit");
            ortho<B>(ctx, MatrixView(C_p, Nvecs, block_vectors, Nvecs), Z, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            //Compute P = [X, P, R] * C_p
            trace("syevx: update P/AP submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), MatrixView(C_p, Nvecs, block_vectors, Nvecs), P_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AP
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), MatrixView(C_p, Nvecs, block_vectors, Nvecs), AP_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

            swap_subspace(); //AX <=> AX_new, AP <=> AP_new, X <=> X_new, P <=> P_new ...
            restart = false;
        }

        // The residual kernel already populated W with the best eigenvalues seen during iterations.
        // For eigenvectors, copy the final X to V (they should be reasonably close to the true eigenvectors).
        if (jobz == JobType::EigenVectors){
            MatrixView<T, MatrixFormat::Dense>::copy(ctx, V({0,n}, {0,int64_t(neigs)}), X({0, n}, {0, int64_t(neigs)}));
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

            {
                // Match runtime: XtAX is a (block_vectors x block_vectors) view into the
                // top-left corner of a (3*block_vectors x 3*block_vectors) backing buffer.
                auto XtAX_dummy = MatrixView<T, MatrixFormat::Dense>(nullptr,
                    static_cast<int>(block_vectors), static_cast<int>(block_vectors),
                    static_cast<int>(block_vectors * 3),
                    static_cast<int>(3 * 3 * block_vectors * block_vectors),
                    static_cast<int>(batch_size), nullptr);
                auto StAS_base_dummy = MatrixView<T, MatrixFormat::Dense>(nullptr,
                    static_cast<int>(block_vectors * 3), static_cast<int>(block_vectors * 3),
                    static_cast<int>(block_vectors * 3), static_cast<int>(3 * 3 * block_vectors * block_vectors),
                    static_cast<int>(batch_size), nullptr);

                const size_t ws_xtax = syev_buffer_size<B>(ctx, XtAX_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                const size_t ws_stas = syev_buffer_size<B>(ctx, StAS_base_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                size_t ws_projected = std::max(ws_xtax, ws_stas);

                // Match the runtime behavior: projected problems prefer the vendor SYEV path on GPUs.
                if constexpr (B != Backend::NETLIB) {
                    const size_t ws_xtax_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, XtAX_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    const size_t ws_stas_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_base_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    ws_projected = std::max(ws_projected, std::max(ws_xtax_vendor, ws_stas_vendor));
                }

                work_size += BumpAllocator::allocation_size<std::byte>(ctx, ws_projected);
            }

            work_size += BumpAllocator::allocation_size<std::byte>(ctx,std::max(    ortho_buffer_size<B>(ctx, Xview, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors*2, n, n * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                                    ortho_buffer_size<B>(ctx, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors, block_vectors * 3), MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm)));
            if constexpr (MFormat == MatrixFormat::CSR) {
                work_size += BumpAllocator::allocation_size<std::byte>(ctx,spmm_buffer_size<B>(ctx, A, Xview, AXview, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            }
                        
            work_size += BumpAllocator::allocation_size<T*>(ctx, batch_size) * 7;
            work_size += BumpAllocator::allocation_size<int32_t>(ctx, batch_size); // converged_flags
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