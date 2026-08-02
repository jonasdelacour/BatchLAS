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
#include <blas/functions/iluk.hh>
#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"
#include <internal/sort.hh>

namespace batchlas {
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxResidualsKernel;

    // Only used by the params.iterations == 0 cold path; the per-iteration column
    // reversals it used to serve are folded into the X_best snapshot (SYEVX_PLAN.md 7.8).
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxReverseEigenvectorsKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxLobpcgInitKernel;

namespace {

// Block power-iteration steps applied to the random start (SYEVX_PLAN.md §7.9).
// Only meaningful when searching for the largest eigenpairs -- see the block that
// uses this for the measurements that set the default.
//
// 4 rather than a larger number: the gain keeps growing with the step count on the
// matrices measured, but every step compresses the block further toward the
// dominant directions, and an over-compressed block is rank-deficient in floating
// point, at which point the Cholesky-based ortho returns NaN rather than an error
// (the failure mode documented in syevx_filtered.cc). 4 buys most of the win with
// margin; BATCHLAS_SYEVX_INIT_POWER exists for A/B-ing that choice.
constexpr int kDefaultInitPowerIterations = 4;

inline int lobpcg_init_power_iterations(int from_params, bool find_largest) {
    int steps = from_params < 0 ? kDefaultInitPowerIterations : from_params;
    if (const char* v = std::getenv("BATCHLAS_SYEVX_INIT_POWER")) {
        const int parsed = std::atoi(v);
        if (parsed >= 0) steps = parsed;
    }
    // Powers of A amplify the *largest* eigendirections. With find_largest = false
    // that drives the start away from what is wanted, so the steps are dropped
    // rather than applied backwards.
    return find_largest ? steps : 0;
}

// Search-space width. `extra_directions == 0` means "choose one", matching the
// convention SyevxParams::filter_degree uses.
//
// Running LOBPCG with exactly `neigs` vectors and no guard block is a known way
// to converge slowly: the last wanted pair has nothing above it to separate
// against. A guard of ~25% of neigs is standard practice and usually cuts the
// iteration count for a cost only linear in the extra width. A caller that
// genuinely wants no guard can still say so by asking for a width explicitly.
//
// The search space is n x 3*block_vectors; letting that exceed n would make the
// block rank-deficient by construction and break the Cholesky-based
// orthogonalization, so the guard is dropped rather than allowed to push past it.
inline int64_t lobpcg_block_vectors(size_t neigs, size_t extra_directions, int64_t n) {
    const int64_t k = static_cast<int64_t>(neigs);
    if (extra_directions > 0) return k + static_cast<int64_t>(extra_directions);
    // Escape hatch for A/B-ing the guard itself; 0 reproduces the old behaviour.
    int64_t extra = std::max<int64_t>(2, k / 4);
    if (const char* v = std::getenv("BATCHLAS_SYEVX_EXTRA_DIRECTIONS")) {
        const int parsed = std::atoi(v);
        if (parsed >= 0) extra = parsed;
    }
    if (extra <= 0) return k;
    const int64_t guarded = k + extra;
    if (n > 0 && 3 * guarded > n) return std::max<int64_t>(k, std::min<int64_t>(guarded, n / 3));
    return guarded;
}

// How often the host reads back the convergence flags. Every iteration -- the
// old behaviour -- costs a full pipeline drain each time; see SYEVX_PLAN.md §7.1.
inline int64_t lobpcg_check_every() {
    if (const char* v = std::getenv("BATCHLAS_SYEVX_CHECK_EVERY")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) return parsed;
    }
    return 4;
}

} // namespace

    template <Backend B, typename T, MatrixFormat MFormat>
    Event syevx_lobpcg(Queue& ctx,
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
        if (params.preconditioner != nullptr && params.build_preconditioner) {
            throw std::invalid_argument(
                "syevx: SyevxParams::preconditioner and SyevxParams::build_preconditioner are "
                "mutually exclusive; supply a factor or ask syevx to build one, not both");
        }
        const bool use_preconditioner = params.preconditioner != nullptr || params.build_preconditioner;
        // An ILU(k) factorization approximates A^{-1}. Applying it to the LOBPCG
        // residual accelerates convergence toward the smallest eigenpairs, but for
        // the largest eigenpairs it suppresses the wanted directions and boosts the
        // unwanted ones -- measurably worse than running unpreconditioned. Reject the
        // combination instead of silently degrading.
        if (use_preconditioner && params.find_largest) {
            throw std::invalid_argument(
                "syevx: an ILU(k) preconditioner approximates A^{-1} and is only valid when "
                "searching for the smallest eigenpairs; set SyevxParams::find_largest = false "
                "or clear SyevxParams::preconditioner / build_preconditioner");
        }
        if constexpr (MFormat != MatrixFormat::CSR) {
            if (params.build_preconditioner) {
                throw std::invalid_argument(
                    "syevx: SyevxParams::build_preconditioner requires a CSR matrix; ILU(k) is "
                    "only defined for sparse input");
            }
        }

        // Implementation of the syevx function
        // This function computes the eigenvalues and eigenvectors of a symmetric matrix
        int64_t block_vectors = lobpcg_block_vectors(neigs, params.extra_directions, A.rows_);
        const int64_t convergence_check_every = lobpcg_check_every();
        auto pool = BumpAllocator(workspace);
        auto n = A.rows_;
        auto batch_size = A.batch_size();
        const bool want_eigenvectors = jobz == JobType::EigenVectors;
        auto Sdata =        pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto ASdata =       pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto S_newdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto Stempdata =    pool.allocate<T>(ctx, n * block_vectors * 3 * batch_size);
        auto StASdata =     pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);
        auto C_pdata =      pool.allocate<T>(ctx, block_vectors * block_vectors * 3 * batch_size);
        auto lambdas =      pool.allocate<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);
        auto residuals =    pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        auto best_residuals = pool.allocate<typename base_type<T>::type>(ctx, neigs * batch_size);
        auto best_quality = pool.allocate<typename base_type<T>::type>(ctx, batch_size);
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

        MatrixView<T, MatrixFormat::Dense> X_best;
        if (want_eigenvectors) {
            auto X_bestdata = pool.allocate<T>(ctx, n * neigs * batch_size);
            X_best = MatrixView(
                X_bestdata.data(),
                n,
                static_cast<int>(neigs),
                n,
                n * static_cast<int64_t>(neigs),
                batch_size,
                pool.allocate<T*>(ctx, batch_size).data());
        }

        // Built either from the caller's factor or, when asked, from one formed here
        // out of `workspace` -- no allocation of its own, so syevx stays pool-only.
        ILUKView<T> precond;
        if (params.preconditioner != nullptr) {
            precond = params.preconditioner->view();
        } else if (params.build_preconditioner) {
            if constexpr (MFormat == MatrixFormat::CSR) {
                // Hand ILU(k) the unclaimed tail of the pool and take back only what
                // it used. Asking iluk_buffer_size first would work, but it costs a
                // second symbolic factorization on the critical path for a number
                // the factorization is about to compute anyway.
                size_t iluk_bytes = 0;
                precond = iluk_factorize<B, T>(ctx, A, pool.remaining(), params.iluk_params, &iluk_bytes);
                pool.consume(iluk_bytes);
            }
        }

        // No staging buffer for the preconditioner input: iluk_apply indexes its
        // operands as b*stride_ + col*ld_ (src/extensions/iluk.cc), so it reads R's
        // n x 3k slice directly. Repacking R into a packed-batch copy first was a
        // full n x k x batch copy per iteration, plus the allocation, for nothing.

        auto R_preconditioned_data = pool.allocate<T>(ctx, n * block_vectors * batch_size);
        auto R_preconditioned = MatrixView(
            R_preconditioned_data.data(),
            n,
            block_vectors,
            n,
            n * block_vectors,
            batch_size,
            pool.allocate<T*>(ctx, batch_size).data());

        auto AS_new = MatrixView(Stempdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto AX_new = AS_new({0,n}, {0,block_vectors});                       //First block of AS_new
        auto AP_new = AS_new({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of AS_new
        auto AR_new = AS_new({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of AS_new

        Span<std::byte> spmm_workspace;
        if constexpr (MFormat == MatrixFormat::CSR) {
              spmm_workspace = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
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
        // Restart iterations solve a 2*block_vectors projected problem rather than the
        // full 3*block_vectors one (see Nvecs below). Because the provider is chosen
        // from the shape, that intermediate size can demand *more* workspace than
        // either the block_vectors or 3*block_vectors problem, so it has to be sized
        // explicitly instead of assumed to be bounded by them.
        auto StAS_restart = MatrixView(StAS_base, block_vectors * 2, block_vectors * 2,
                                       StAS_base.ld(), StAS_base.stride());
        const size_t ws_xtax = syev_buffer_size<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
        const size_t ws_stas_restart = syev_buffer_size<B>(ctx, StAS_restart, lambdas, JobType::EigenVectors, Uplo::Lower);
        const size_t ws_stas = syev_buffer_size<B>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
        size_t ws_projected = std::max(ws_xtax, std::max(ws_stas_restart, ws_stas));
        if (prefer_vendor_projected_syev) {
            const size_t ws_xtax_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
            const size_t ws_stas_restart_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_restart, lambdas, JobType::EigenVectors, Uplo::Lower);
            const size_t ws_stas_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
            ws_projected = std::max(ws_projected,
                                    std::max(ws_xtax_vendor, std::max(ws_stas_restart_vendor, ws_stas_vendor)));
        }
        auto syev_workspace = pool.allocate<std::byte>(ctx, ws_projected);
        // Three distinct ortho calls share this buffer: the single-matrix ortho(X)
        // below, and the two external-metric variants inside the loop. The
        // single-matrix one was missing from this max, so its workspace was
        // whatever the other two happened to need -- fine until a shape came along
        // where it needed more (n = 64, block_vectors = 20 was one).
        auto ortho_workspace = pool.allocate<std::byte>(ctx, std::max(
                          ortho_buffer_size<B>(ctx, X, Transpose::NoTrans, params.algorithm),
                          std::max(ortho_buffer_size<B>(ctx, R, XP, Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                          ortho_buffer_size<B>(ctx, C_p, StAS_base, Transpose::NoTrans, Transpose::NoTrans, params.algorithm))));
        
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

        // Only the X block of S is read before the rest is overwritten (P and R are
        // both recomputed from the first Rayleigh-Ritz onwards), so filling all of
        // S was 3x the random generation and 3x the write traffic for nothing.
        //
        // The linear index below reproduces MatrixView::fill_random's exactly --
        // fill_random ignores ld/stride and walks the buffer flat, so for the X
        // block of a contiguous S that index is b*(3k*n) + c*n + r. Keeping it
        // identical means the starting block is bit-for-bit what it used to be:
        // this change is pure waste elimination, not a behaviour change, and stays
        // reproducible across runs.
        {
            auto Xk = X.kernel_view();
            const int64_t nn = n;
            const int64_t kk = block_vectors;
            const unsigned int seed = 42; // fill_random's default
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxLobpcgInitKernel<B, T, MFormat>>(
                    sycl::range<1>(static_cast<size_t>(batch_size * nn * kk)), [=](sycl::id<1> tid) {
                        const int64_t local = static_cast<int64_t>(tid[0]);
                        const int b = static_cast<int>(local / (nn * kk));
                        const int64_t rem = local - static_cast<int64_t>(b) * nn * kk;
                        const int r = static_cast<int>(rem % nn);
                        const int c = static_cast<int>(rem / nn);
                        const size_t idx = static_cast<size_t>(b) * (3 * kk * nn) +
                                           static_cast<size_t>(c) * nn + static_cast<size_t>(r);
                        oneapi::dpl::uniform_real_distribution<float_type> dist(-1.0, 1.0);
                        oneapi::dpl::minstd_rand engine(seed, idx);
                        const auto r1 = dist(engine);
                        if constexpr (is_std_complex_v<T>) {
                            const auto r2 = dist(engine);
                            Xk(r, c, b) = T(r1, r2);
                        } else {
                            Xk(r, c, b) = T(r1);
                        }
                    });
            });
        }

        // Block power-iteration start (SYEVX_PLAN.md §7.9): X <- ortho(A X), a few
        // times. Powers of A amplify the largest eigendirections, so this is a valid
        // improvement *only* for find_largest; lobpcg_init_power_iterations returns 0
        // otherwise and the block below does not run.
        //
        // MEASURED (NETLIB/CPU, double, dense random Hermitian, batch 4, tol 1e-5,
        // BATCHLAS_SYEVX_CHECK_EVERY=1, mean iterations over 3 seeds), p = steps:
        //
        //   n    k   find_largest      p=0    p=1    p=2    p=4    p=8
        //   64   4   true            22.67  20.67  20.00  18.67  16.00
        //   64  16   true             8.33   7.00   6.00   4.67   3.00
        //  128  16   true            19.00  17.33  16.00  15.00  12.67
        //  256   8   true            36.00  35.33  34.33  31.00  27.33
        //  256  16   true            27.67  26.33  25.33  24.00  21.00
        //
        // i.e. 12-44% fewer iterations at p = 4, and wall time moved the same way
        // (n=64,k=16: 23.4 -> 11.3 ms; n=256,k=16: 146.6 -> 124.7 ms).
        //
        // The smallest end was measured too, using sigma*I - A with sigma a Gershgorin
        // upper bound (whose largest eigenpairs are A's smallest). It was flat --
        // 23.00 -> 21.67 at n=64,k=4 and unchanged at 8.00 / 19.00 / 26.33 elsewhere,
        // inside run-to-run noise -- because the amplification ratio
        // (sigma - l_1)/(sigma - l_{k+1}) is ~1 for a wide spectrum. That variant paid
        // matvecs for nothing, so it is not implemented; the restriction above stands.
        const int init_power_steps =
            lobpcg_init_power_iterations(params.init_power_iterations, params.find_largest);
        for (int step = 0; step < init_power_steps; ++step) {
            ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
            if constexpr (MFormat == MatrixFormat::Dense) {
                gemm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            } else {
                spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            }
            MatrixView<T, MatrixFormat::Dense>::copy(ctx, X, AX);
        }

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

        // NOTE (SYEVX_PLAN.md 7.8): the search block `X` is deliberately kept in the
        // ascending order `syev` produces, even when `find_largest`. Nothing inside the
        // iteration cares about the order of X's columns -- they are a basis -- so the
        // two batch-wide column-reversal kernels that used to run here and on the
        // selected StAS block every iteration were pure launch overhead. The
        // largest-first presentation is applied exactly once, where the wanted block is
        // snapshotted into `X_best` (a copy the residual kernel already performs), via
        // `reported_col()` below.
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
        int32_t completed_iterations = 0;

        size_t residual_wg_size = std::min(get_kernel_max_wg_size<SyevxResidualsKernel<B,T,MFormat>>(ctx), size_t(n));

        //Compute R = AX - X * diag(lambdas)
        for(int it = 0; it < params.iterations; it++){
            completed_iterations = static_cast<int32_t>(it + 1);
            int Nvecs = restart ? block_vectors * 2 : block_vectors * 3;
            //Compute R = AX - X * diag(lambdas)
            trace("syevx: residual kernel submit");
            const float_type abs_tol = static_cast<float_type>(std::abs(params.absolute_tolerance));
            const float_type rel_tol = static_cast<float_type>(std::abs(params.relative_tolerance));
            const float_type tol = std::max(abs_tol, rel_tol);
            auto residual_evt = ctx -> submit([&](sycl::handler& h){
                auto Rdata = R.data_ptr();
                auto Xdata = X.data_ptr();
                auto AXdata = AX.data_ptr();
                auto flags = converged_flags.data();
                auto best_quality_data = best_quality.data();
                auto Xbest_data = want_eigenvectors ? X_best.data_ptr() : nullptr;
                auto update_best = sycl::local_accessor<int32_t, 1>(1, h);
                // Per-column partials for ||r||, ||x|| and ||Ax||, accumulated in
                // one pass instead of 2*neigs sequential group reductions (each of
                // which is a full work-group barrier -- 128 of them at neigs = 64).
                auto lsums = sycl::local_accessor<float_type, 1>(3 * neigs, h);
                const bool find_largest = params.find_largest;
                h.parallel_for<SyevxResidualsKernel<B,T,MFormat>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*residual_wg_size)}, sycl::range{size_t(residual_wg_size)}), [=](sycl::nd_item<1> item){
                    auto num_eigvals = it < 2 ? (it+1) * block_vectors : 3*block_vectors;

                    // X's columns are in the ascending order syev returned them in, so
                    // column j of X pairs with lambda[eig_offset + j]. For find_largest
                    // the selected block is the *top* block_vectors of num_eigvals, hence
                    // the offset; the wanted neigs pairs are then the *last* neigs columns.
                    const int64_t eig_offset = find_largest ? (int64_t(num_eigvals) - block_vectors) : 0;
                    // Reported slot i (largest-first when find_largest) <- column of X.
                    const auto reported_col = [=](int64_t i) {
                        return find_largest ? (block_vectors - 1 - i) : i;
                    };
                    // Inverse of reported_col: column of X -> reported slot (may be >= neigs).
                    const auto col_to_slot = [=](int64_t c) {
                        return find_largest ? (block_vectors - 1 - c) : c;
                    };

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
                    if (tid == 0) {
                        update_best[0] = 0;
                    }
                    for (size_t s = tid; s < 3 * neigs; s += local_size) {
                        lsums[s] = float_type(0);
                    }
                    sycl::group_barrier(cta);

                    // Form R and accumulate the three per-column norms in the same
                    // sweep. The block is column-major with n contiguous entries per
                    // column, so a work-item stays inside one column for long runs;
                    // keeping a running partial and flushing only on a column change
                    // turns what would be one atomic per element into roughly one per
                    // thread per column.
                    {
                        int cur_col = -1;
                        float_type acc_r = 0, acc_x = 0, acc_ax = 0;
                        auto flush = [&]() {
                            if (cur_col < 0 || cur_col >= int(neigs)) return;
                            sycl::atomic_ref<float_type, sycl::memory_order::relaxed,
                                             sycl::memory_scope::work_group,
                                             sycl::access::address_space::local_space>
                                ar(lsums[cur_col]), ax(lsums[neigs + cur_col]),
                                aax(lsums[2 * neigs + cur_col]);
                            ar.fetch_add(acc_r);
                            ax.fetch_add(acc_x);
                            aax.fetch_add(acc_ax);
                        };
                        for (int i = tid; i < n*block_vectors; i+=local_size){
                            const int eigvect_id = i / n;
                            auto eigval = blockLambdas[eig_offset + eigvect_id];
                            const T rval = blockAX[i] - blockX[i] * eigval;
                            blockR[i] = rval;
                            const int slot = int(col_to_slot(eigvect_id));
                            if (slot >= int(neigs)) continue;
                            if (slot != cur_col) {
                                flush();
                                cur_col = slot;
                                acc_r = acc_x = acc_ax = float_type(0);
                            }
                            acc_r  += internal::norm_squared(rval);
                            acc_x  += internal::norm_squared(blockX[i]);
                            acc_ax += internal::norm_squared(blockAX[i]);
                        }
                        flush();
                    }
                    sycl::group_barrier(cta);

                    // Backward-stable convergence measure: ||r|| / (||Ax|| + |lambda|*||x||).
                    //
                    // The previous denominator was ||x||*|lambda| alone, which
                    // collapses as lambda -> 0: a perfectly good eigenpair with a
                    // near-zero eigenvalue has its residual divided by something
                    // approaching zero and can never register as converged. Adding
                    // ||Ax|| keeps the denominator bounded away from zero for any
                    // nonzero A (Duersch et al.).
                    for (size_t i = tid; i < neigs; i += local_size){
                        const float_type r_norm  = sycl::sqrt(lsums[i]);
                        const float_type x_norm  = sycl::sqrt(lsums[neigs + i]);
                        const float_type ax_norm = sycl::sqrt(lsums[2 * neigs + i]);
                        const auto eigval = blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                        const float_type denom = ax_norm + sycl::fabs(eigval) * x_norm;
                        blockresiduals[i] = (denom > float_type(0)) ? (r_norm / denom) : r_norm;
                    }

                    sycl::group_barrier(cta);
                    if (tid == 0){
                        float_type current_quality = blockresiduals[0];
                        for (size_t i = 1; i < neigs; ++i) {
                            if (blockresiduals[i] > current_quality) {
                                current_quality = blockresiduals[i];
                            }
                        }

                        const bool update = (it == 0) || (current_quality < best_quality_data[bid]);
                        if (update){
                            best_quality_data[bid] = current_quality;
                            for (size_t i = 0; i < neigs; ++i) {
                                blockbestresiduals[i] = blockresiduals[i];
                                blockW[i] = blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                            }
                            update_best[0] = 1;
                        }
                    }

                    sycl::group_barrier(cta);
                    if (want_eigenvectors && update_best[0] != 0) {
                        auto* blockXbest = Xbest_data + bid * (n * neigs);
                        // This copy is where the largest-first presentation happens: slot
                        // i of the snapshot takes column reported_col(i) of X. Folding the
                        // permutation into a copy that already touches every one of these
                        // elements is what makes the two reversal kernels unnecessary.
                        for (int i = int(tid); i < int(n * neigs); i += int(local_size)) {
                            const int64_t slot = i / int(n);
                            const int64_t row = i - slot * int(n);
                            blockXbest[i] = blockX[reported_col(slot) * int64_t(n) + row];
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
            // Only drain the pipeline when a host-side reader actually needs the
            // results this iteration: the convergence check (every check_every
            // iterations) or instrumentation (which still reads on the host --
            // SYEVX_PLAN.md §7.2). Previously this waited unconditionally, so a
            // 30-iteration solve paid 30 full round-trips; for small n and large
            // batch that dominated the run. Overshooting the stopping point by a
            // few iterations is far cheaper than the drains it replaces.
            const bool instrumentation_active =
                params.instrumentation && params.instrumentation->max_iterations > 0 &&
                params.instrumentation->store_every > 0;
            const bool last_iteration = (it + 1 >= static_cast<int64_t>(params.iterations));
            const bool check_convergence =
                (it % convergence_check_every == 0) || last_iteration;
            if (instrumentation_active || check_convergence) {
                residual_evt.wait_and_throw();
            }
            trace("syevx: residual kernel done");

            if (params.instrumentation && params.instrumentation->max_iterations > 0 && params.instrumentation->store_every > 0 &&
                (static_cast<size_t>(it) % params.instrumentation->store_every) == 0) {
                const auto& instr = *params.instrumentation;
                using real_t = typename base_type<T>::type;
                const size_t sample_id = static_cast<size_t>(it) / instr.store_every;
                if (sample_id < instr.max_iterations) {
                    const size_t batch_stride = instr.batch_stride == 0 ? neigs : instr.batch_stride;
                    const size_t iter_stride = instr.iteration_stride == 0 ? batch_size * batch_stride : instr.iteration_stride;
                    const size_t num_eigvals = static_cast<size_t>(it < 2 ? (it + 1) * block_vectors : 3 * block_vectors);

                    for (int64_t b = 0; b < batch_size; ++b) {
                        for (size_t i = 0; i < neigs; ++i) {
                            const size_t dst = sample_id * iter_stride + static_cast<size_t>(b) * batch_stride + i;
                            const auto cur = residuals[static_cast<size_t>(b) * neigs + i];
                            const auto best = best_residuals[static_cast<size_t>(b) * neigs + i];

                            if (instr.best_residual_history.size() > dst) {
                                instr.best_residual_history[dst] = best;
                            }
                            if (instr.store_current_residual && instr.current_residual_history.size() > dst) {
                                instr.current_residual_history[dst] = cur;
                            }
                            if (instr.store_ritz_values && instr.ritz_value_history.size() > dst) {
                                const size_t eig_idx = params.find_largest ? ((num_eigvals - 1) - i) : i;
                                const size_t src = static_cast<size_t>(b) * num_eigvals + eig_idx;
                                if (lambdas.size() > src) {
                                    instr.ritz_value_history[dst] = lambdas[src];
                                }
                            }
                            if (instr.store_convergence_rate && instr.convergence_rate_history.size() > dst) {
                                real_t rate = real_t(1);
                                if (sample_id > 0 && instr.best_residual_history.size() > (dst - iter_stride)) {
                                    const auto prev = instr.best_residual_history[dst - iter_stride];
                                    if (prev > real_t(0)) {
                                        rate = best / prev;
                                    }
                                }
                                instr.convergence_rate_history[dst] = rate;
                            }
                        }
                    }
                }
            }

            // Early exit once all batches have converged for the requested eigenpairs.
            // This is intentionally conservative: it checks the best residual so far.
            bool all_converged = false;
            if (check_convergence) {
                all_converged = true;
                for (int64_t b = 0; b < batch_size; ++b) {
                    if (converged_flags[static_cast<std::size_t>(b)] == 0) {
                        all_converged = false;
                        break;
                    }
                }
            }
            if (all_converged) {
                break;
            }

            if (use_preconditioner) {
                trace("syevx: ILU(k) apply on residuals");
                // R_preconditioned stays a distinct destination: the forward solve
                // writes into `out` as its temporary y while still reading `rhs`, so
                // aliasing them would corrupt the solve. The two wait_and_throw calls
                // that used to bracket this were full pipeline drains per iteration
                // on top of §7.1; the queue ordering already sequences these.
                iluk_apply<B, T>(ctx, precond, R, R_preconditioned);
                MatrixView<T, MatrixFormat::Dense>::copy(ctx, R, R_preconditioned);
                trace("syevx: ILU(k) apply done");
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

            // syev returns eigenvalues in ascending order. For find_largest=true we take
            // the last `block_vectors` Ritz vectors; they stay in ascending order (see the
            // note after the initial XtAX solve). The residual kernel knows that column j
            // of the resulting X pairs with lambda[Nvecs - block_vectors + j], and it is
            // the X_best snapshot -- not a separate kernel -- that flips the wanted block
            // to largest-first on the way out.
            const int64_t eig_col_start = params.find_largest ? (Nvecs - block_vectors) : 0;
            auto Z = StAS({0, Nvecs}, {eig_col_start, eig_col_start + block_vectors});
            // X(i+1) = S * C_x. For the next search block, keep only the non-X
            // coefficient rows of the selected Ritz vectors, which avoids the
            // fragile difference direction X(i+1)-X(i) while preserving the
            // locally-optimal combination of the P/R parts of the trial basis.
            auto C_p_active = MatrixView(C_p, Nvecs, block_vectors, C_p.ld(), C_p.stride());
            auto Z_search = Z({block_vectors, Nvecs}, {0, block_vectors});

            trace("syevx: build search-direction coefficients");
            C_p_active.fill_zeros(ctx);
            MatrixView<T, MatrixFormat::Dense>::copy(
                ctx,
                C_p_active({block_vectors, Nvecs}, {0, block_vectors}),
                Z_search);
            trace_wait("syevx: build search-direction coefficients done");


            //Compute new search directions
            //X = [X, P, R] * C_x
            trace("syevx: update X/AX submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), Z, X_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AX: AX = [AX, AP, AR] * C_x
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), Z, AX_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Orthonormalize C_p against the best eigenvectors
            trace("syevx: ortho C_p vs Z submit");
            ortho<B>(ctx, C_p_active, Z, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            //Compute P = [X, P, R] * C_p
            trace("syevx: update P/AP submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), C_p_active, P_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);
            //Make an implicit update of AP
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), C_p_active, AP_new, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans);

            swap_subspace(); //AX <=> AX_new, AP <=> AP_new, X <=> X_new, P <=> P_new ...
            restart = false;
        }

        if (params.instrumentation && params.instrumentation->iterations_done != nullptr) {
            for (int64_t b = 0; b < batch_size; ++b) {
                params.instrumentation->iterations_done[b] = completed_iterations;
            }
        }

        // The residual kernel snapshots the best Ritz block seen during the iteration.
        if (want_eigenvectors){
            if (completed_iterations > 0) {
                // X_best is already largest-first (the residual kernel's snapshot applies
                // the permutation).
                MatrixView<T, MatrixFormat::Dense>::copy(ctx, V({0,n}, {0,int64_t(neigs)}), X_best);
            } else if (!params.find_largest) {
                MatrixView<T, MatrixFormat::Dense>::copy(
                    ctx, V({0,n}, {0,int64_t(neigs)}), X({0, n}, {0, static_cast<int64_t>(neigs)}));
            } else {
                // params.iterations == 0: the loop never ran, so no snapshot exists and X
                // is still in ascending order. The wanted block is its *last* neigs
                // columns, reversed. One cold-path launch, outside any loop.
                auto Vslice = V({0,n}, {0,int64_t(neigs)});
                auto* Vptr = Vslice.data_ptr();
                const int64_t V_ld = Vslice.ld();
                const int64_t V_stride = Vslice.stride();
                auto* Xptr = X.data_ptr();
                const int64_t X_stride = X.stride();
                const int64_t X_ld = X.ld();
                const int64_t k = block_vectors;
                const int64_t nn = n;
                const int64_t ncols = static_cast<int64_t>(neigs);
                ctx->submit([&](sycl::handler& h) {
                    h.parallel_for<SyevxReverseEigenvectorsKernel<B, T, MFormat>>(
                        sycl::nd_range<1>(sycl::range{size_t(batch_size * 256)}, sycl::range{size_t(256)}),
                        [=](sycl::nd_item<1> item) {
                            const int64_t tid = int64_t(item.get_local_linear_id());
                            const int64_t bid = int64_t(item.get_group_linear_id());
                            const int64_t local_size = int64_t(item.get_local_range(0));
                            auto* src = Xptr + bid * X_stride;
                            auto* dst = Vptr + bid * V_stride;
                            for (int64_t i = tid; i < nn * ncols; i += local_size) {
                                const int64_t col = i / nn;
                                const int64_t row = i - col * nn;
                                dst[row + col * V_ld] = src[row + (k - 1 - col) * X_ld];
                            }
                        });
                });
            }
        }

        return ctx.get_event();
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t syevx_lobpcg_buffer_size(Queue& ctx,
                const MatrixView<T, MFormat>& A,
                Span<typename base_type<T>::type> W,
                size_t neigs,
                JobType jobz,
                const MatrixView<T, MatrixFormat::Dense>& V,
                const SyevxParams<T>& params){
        auto block_vectors = lobpcg_block_vectors(neigs, params.extra_directions, A.rows_);
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
                // The projected problem is Nvecs x Nvecs with Nvecs = 2*block_vectors on
                // restart iterations and 3*block_vectors otherwise, always viewed with the
                // backing buffer's ld. Both must be sized: syev picks its provider from the
                // matrix shape, so workspace demand is not monotone in Nvecs and the
                // 3*block_vectors figure does not bound the 2*block_vectors one. Omitting
                // the restart shape made syevx throw "insufficient workspace for chosen
                // provider" at, for instance, block_vectors = 12 and 16.
                auto projected_dummy = [&](int64_t nvecs) {
                    return MatrixView<T, MatrixFormat::Dense>(nullptr,
                        static_cast<int>(nvecs), static_cast<int>(nvecs),
                        static_cast<int>(block_vectors * 3),
                        static_cast<int>(3 * 3 * block_vectors * block_vectors),
                        static_cast<int>(batch_size), nullptr);
                };
                auto StAS_restart_dummy = projected_dummy(block_vectors * 2);
                auto StAS_base_dummy = projected_dummy(block_vectors * 3);

                const size_t ws_xtax = syev_buffer_size<B>(ctx, XtAX_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                const size_t ws_stas_restart = syev_buffer_size<B>(ctx, StAS_restart_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                const size_t ws_stas = syev_buffer_size<B>(ctx, StAS_base_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                size_t ws_projected = std::max(ws_xtax, std::max(ws_stas_restart, ws_stas));

                // Match the runtime behavior: projected problems prefer the vendor SYEV path on GPUs.
                if constexpr (B != Backend::NETLIB) {
                    const size_t ws_xtax_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, XtAX_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    const size_t ws_stas_restart_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_restart_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    const size_t ws_stas_vendor = backend::syev_vendor_buffer_size<B, T>(ctx, StAS_base_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    ws_projected = std::max(ws_projected,
                                            std::max(ws_xtax_vendor, std::max(ws_stas_restart_vendor, ws_stas_vendor)));
                }

                work_size += BumpAllocator::allocation_size<std::byte>(ctx, ws_projected);
            }

            // Must mirror the runtime max exactly, including the single-matrix
            // ortho(X) term -- see the comment at the runtime allocation.
            //
            // The C_p stand-in also has to carry the real batch size and stride.
            // Built with only (data, rows, cols, ld) it defaulted to batch_size = 1,
            // so this term came back sized for one item while the runtime call is
            // batched -- an under-allocation that grew with the batch.
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,std::max(
                                                                                    ortho_buffer_size<B>(ctx, Xview, Transpose::NoTrans, params.algorithm),
                                                                                    std::max(ortho_buffer_size<B>(ctx, Xview, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors*2, n, n * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                                    ortho_buffer_size<B>(ctx, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm))));
            if constexpr (MFormat == MatrixFormat::CSR) {
                work_size += BumpAllocator::allocation_size<std::byte>(ctx,spmm_buffer_size<B>(ctx, A, Xview, AXview, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            }
                        
            // R_contiguous is gone: iluk_apply reads R's strided slice directly.
            if constexpr (MFormat == MatrixFormat::CSR) {
                // This runs ILU(k)'s symbolic phase to get an upper bound on the fill.
                // syevx itself does not repeat it -- it sub-allocates from the pool tail
                // and reports back what it took -- so the cost lands here, on the sizing
                // call a caller makes once and amortizes over every solve.
                if (params.build_preconditioner) {
                    work_size += BumpAllocator::allocation_size<std::byte>(ctx, iluk_buffer_size<B, T>(ctx, A, params.iluk_params));
                }
            }
            work_size += BumpAllocator::allocation_size<T*>(ctx, batch_size) * 7;
            if (jobz == JobType::EigenVectors) {
                work_size += BumpAllocator::allocation_size<T*>(ctx, batch_size);
                work_size += BumpAllocator::allocation_size<T>(ctx, n * neigs * batch_size);
            }
            work_size += BumpAllocator::allocation_size<int32_t>(ctx, batch_size); // converged_flags
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * 3 * batch_size) * 4;                    //Sdata, ASdata, S_newdata, Stempdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);        //StASdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * batch_size);            //C_pdata
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * batch_size);                            //R_preconditioned_data
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);  //lambdas
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //residuals
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //best residuals
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, batch_size);                      //best block quality

            return work_size;
    }

    #define SYEVX_INSTANTIATE(back, fp, fmt) \
    template Event syevx_lobpcg<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_lobpcg_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);
    

    #define SYEVX_INSTANTIATE_FOR_BACKEND(back)\
        BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_INSTANTIATE_FOR_BACKEND_TYPE, back)

    #define SYEVX_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
        BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_INSTANTIATE, back, fp)

    #if BATCHLAS_HAS_CUDA_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND
        SYEVX_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
    #endif

    #undef SYEVX_INSTANTIATE_FOR_BACKEND_TYPE
    #undef SYEVX_INSTANTIATE_FOR_BACKEND
    #undef SYEVX_INSTANTIATE
}
