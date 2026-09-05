#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <stdexcept>
#include <oneapi/dpl/random>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/iluk.hh>
#include "../math-helpers.hh"
#include "../util/template-instantiations.hh"
#include "../sort.hh"

namespace batchlas {
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxResidualsKernel;

    // Only used by the params.iterations == 0 cold path (SYEVX_PLAN.md 7.8).
    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxReverseEigenvectorsKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxLobpcgInitKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxJacobiDiagonalKernel;

    template <Backend B, typename T, MatrixFormat MFormat>
    struct SyevxJacobiApplyKernel;

namespace {

// Block power-iteration steps applied to the random start; meaningful only when
// searching for the largest eigenpairs. Kept small deliberately: an over-compressed
// block is rank-deficient in floating point, at which point the Cholesky-based ortho
// returns NaN rather than an error. Evidence: SYEVX_PLAN.md §7.9.
constexpr int kDefaultInitPowerIterations = 4;

inline int lobpcg_init_power_iterations(int from_params, bool find_largest) {
    int steps = from_params < 0 ? kDefaultInitPowerIterations : from_params;
    if (const char* v = std::getenv("BATCHLAS_SYEVX_INIT_POWER")) {
        const int parsed = std::atoi(v);
        if (parsed >= 0) steps = parsed;
    }
    // Powers of A amplify the *largest* eigendirections, so the steps are dropped
    // rather than applied backwards when the smallest are wanted.
    return find_largest ? steps : 0;
}

// Search-space width. `extra_directions == 0` means "choose one" (a ~25% guard block),
// matching the convention SyevxParams::filter_degree uses. The search space is
// n x 3*block_vectors; letting that exceed n makes the block rank-deficient by
// construction and breaks the Cholesky-based ortho, so the guard is dropped instead.
inline int64_t lobpcg_block_vectors(size_t neigs, size_t extra_directions, int64_t n) {
    const int64_t k = static_cast<int64_t>(neigs);
    if (extra_directions > 0) return k + static_cast<int64_t>(extra_directions);
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

// How often the host reads back the convergence flags; each check is a full pipeline
// drain (SYEVX_PLAN.md §7.1).
inline int64_t lobpcg_check_every() {
    if (const char* v = std::getenv("BATCHLAS_SYEVX_CHECK_EVERY")) {
        const int parsed = std::atoi(v);
        if (parsed > 0) return parsed;
    }
    return 4;
}

// Instrumentation staging plan (SYEVX_PLAN.md §7.2). The caller-supplied
// SyevxInstrumentation spans are not guaranteed device-accessible -- the Python binding
// hands us a plain std::vector -- so NO kernel may write them. The residual kernel stores
// here instead, and one host pass after the loop scatters into the caller's spans.
struct LobpcgInstrumentationPlan {
    bool active = false;         // instrumentation requested at all
    size_t samples = 0;          // upper bound on stored samples
    size_t slot = 0;             // values per sample (batch_size * neigs)
    bool stage_current = false;  // current_residual_history wanted
    bool stage_ritz = false;     // ritz_value_history wanted

    // Must be identical in syevx_lobpcg and syevx_lobpcg_buffer_size.
    template <typename Real>
    size_t staging_bytes(Queue& ctx) const {
        if (!active) return 0;
        const size_t one = BumpAllocator::allocation_size<Real>(ctx, samples * slot);
        return one * (1 + (stage_current ? 1 : 0) + (stage_ritz ? 1 : 0));
    }
};

// A/B escape hatch: the device-staged path must produce exactly the host path's values,
// and tests/syevx_tests.cc checks that by running both.
inline bool lobpcg_instrumentation_force_host() {
    const char* v = std::getenv("BATCHLAS_SYEVX_INSTR_HOST");
    return v && (v[0] == '1' || v[0] == 't' || v[0] == 'T' || v[0] == 'y' || v[0] == 'Y');
}

template <typename T>
LobpcgInstrumentationPlan lobpcg_instrumentation_plan(const SyevxParams<T>& params,
                                                      size_t neigs,
                                                      int64_t batch_size) {
    LobpcgInstrumentationPlan plan;
    const auto* instr = params.instrumentation;
    if (instr == nullptr || instr->max_iterations == 0 || instr->store_every == 0) return plan;
    if (neigs == 0 || batch_size <= 0 || params.iterations == 0) return plan;
    const size_t reachable = (params.iterations + instr->store_every - 1) / instr->store_every;
    plan.samples = std::min(instr->max_iterations, reachable);
    if (plan.samples == 0) return plan;
    plan.active = true;
    plan.slot = neigs * static_cast<size_t>(batch_size);
    plan.stage_current = instr->store_current_residual && instr->current_residual_history.size() > 0;
    plan.stage_ritz = instr->store_ritz_values && instr->ritz_value_history.size() > 0;
    return plan;
}

// Soft locking, variant (a): column masking. OFF by default and deliberately so -- the
// mechanism is implemented and correct, but it saves no flops (the block shapes are
// fixed) and measured no benefit. Evidence: SYEVX_PLAN.md §7.5.
inline bool lobpcg_soft_locking() {
    if (const char* v = std::getenv("BATCHLAS_SYEVX_SOFT_LOCK")) {
        return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
    }
    return false;
}

// Safety factor on the locking threshold: a column is masked only once its residual is
// this multiple of the requested tolerance. Locking exactly at `tol` oscillates -- a
// masked column is not frozen, its Ritz vector is still recombined every iteration, so a
// column at the boundary loses its correction direction and drifts back above `tol`.
inline double lobpcg_lock_factor() {
    if (const char* v = std::getenv("BATCHLAS_SYEVX_LOCK_FACTOR")) {
        const double parsed = std::atof(v);
        if (parsed > 0.0) return parsed;
    }
    return 0.1;
}

// Relative floor below which a Jacobi shift is treated as singular and the column entry
// is left unpreconditioned. `d_ii - lambda` genuinely reaches zero: for a nearly diagonal
// A the wanted Ritz value converges *to* some d_ii. Dividing by it costs the whole block,
// since the Cholesky-based ortho returns NaN on a rank-deficient input rather than
// failing loudly.
template <typename R>
inline constexpr R jacobi_singular_tolerance() {
    return std::sqrt(std::numeric_limits<R>::epsilon());
}

// The unshifted diag(A)^{-1} is a valid LOBPCG preconditioner only where it is SPD, i.e.
// where every diagonal entry is strictly positive; applying it anyway drives the solve to
// the iteration cap. Positivity is the real condition -- this floor merely keeps the
// amplification finite, and is deliberately loose because tightening it also disables the
// preconditioner on graded matrices whose diagonal is perfectly positive.
template <typename R>
inline constexpr R jacobi_definiteness_floor() {
    return R(1e-6);
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

        // LOBPCG converges to whichever EXTREME the trial block is biased toward; it has
        // no il/iu to honour. `syevx` never routes a non-extremal request here, but this
        // is a public entry point too, and silently returning extremal eigenpairs to an
        // interior request is the one failure mode no downstream check can catch.
        // See SYEVX_RANGE_PLAN.md §2.5.
        if (params.select != SyevxSelect::Extremal) {
            throw std::invalid_argument(
                "syevx_lobpcg: only SyevxSelect::Extremal is supported; LOBPCG converges to an "
                "extreme of the spectrum by construction. Use syevx_direct or "
                "syevx_direct_subset for an index or value range");
        }

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
        const bool iluk_configured = params.preconditioner != nullptr || params.build_preconditioner;
        // Resolved identically in syevx_lobpcg_buffer_size -- the Jacobi path adds a
        // pool allocation, so the two must not be able to disagree.
        const auto precond_kind =
            syevx_select_preconditioner(params.preconditioner_type, iluk_configured, params.find_largest);
        // The `&& iluk_configured` is defensive rather than redundant: the dispatcher
        // rejects ILUK-without-a-factor, but syevx_lobpcg is a public entry point too.
        const bool use_preconditioner =
            (precond_kind == SyevxPreconditioner::ILUK) && iluk_configured;
        const bool jacobi_shifted = (precond_kind == SyevxPreconditioner::JacobiShifted);
        const bool use_jacobi = (precond_kind == SyevxPreconditioner::Jacobi) || jacobi_shifted;
        // An ILU(k) factorization approximates A^{-1}: for the largest eigenpairs it
        // suppresses the wanted directions. Reject rather than silently degrade.
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

        int64_t block_vectors = lobpcg_block_vectors(neigs, params.extra_directions, A.rows_);
        const int64_t convergence_check_every = lobpcg_check_every();
        const bool soft_locking = lobpcg_soft_locking();
        const double soft_lock_factor = lobpcg_lock_factor();
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
        // Per-column convergence state for soft locking; 1 == column i of batch bid met
        // the tolerance on the *current* iterate. Consumed on-device only.
        auto col_converged = pool.allocate<int32_t>(ctx, neigs * batch_size);

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
        // XtAX is the top-left (block_vectors x block_vectors) corner of StAS_base's
        // backing buffer. IMPORTANT: keep StAS_base's ld/stride so batches do not overlap.
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
                // Hand ILU(k) the unclaimed tail of the pool and take back only what it
                // used; asking iluk_buffer_size first costs a second symbolic pass.
                size_t iluk_bytes = 0;
                precond = iluk_factorize<B, T>(ctx, A, pool.remaining(), params.iluk_params, &iluk_bytes);
                pool.consume(iluk_bytes);
            }
        }

        // No staging buffer for the preconditioner input: iluk_apply indexes its operands
        // as b*stride_ + col*ld_, so it reads R's strided n x 3k slice directly.

        auto R_preconditioned_data = pool.allocate<T>(ctx, n * block_vectors * batch_size);
        auto R_preconditioned = MatrixView(
            R_preconditioned_data.data(),
            n,
            block_vectors,
            n,
            n * block_vectors,
            batch_size,
            pool.allocate<T*>(ctx, batch_size).data());

        // diag(A), extracted once (a per-row search for CSR). Kept as T rather than
        // float_type so the Hermitian complex case needs no special storage; the shift
        // arithmetic below is done in T anyway.
        Span<T> jacobi_diag;
        Span<int32_t> jacobi_usable;
        if (use_jacobi) {
            jacobi_diag = pool.allocate<T>(ctx, n * batch_size);
            jacobi_usable = pool.allocate<int32_t>(ctx, batch_size);
        }

        auto AS_new = MatrixView(Stempdata.data(), n, block_vectors * 3, n, n * block_vectors * 3, batch_size, pool.allocate<T*>(ctx, batch_size).data());
        auto AX_new = AS_new({0,n}, {0,block_vectors});                       //First block of AS_new
        auto AP_new = AS_new({0,n}, {block_vectors, 2 * block_vectors});      //Middle block of AS_new
        auto AR_new = AS_new({0,n}, {2 * block_vectors, 3 * block_vectors});  //Last block of AS_new

        Span<std::byte> spmm_workspace;
        if constexpr (MFormat == MatrixFormat::CSR) {
              spmm_workspace = pool.allocate<std::byte>(ctx, spmm_buffer_size<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
        }

        // Diagnostic escape hatch for the tiny projected solves (XtAX, StAS).
        const bool prefer_vendor_projected_syev =
            (B != Backend::NETLIB) &&
            ([]() {
                if (const char* v = std::getenv("BATCHLAS_SYEVX_PROJECTED_VENDOR")) {
                    return (v[0] == '1') || (v[0] == 't') || (v[0] == 'T') || (v[0] == 'y') || (v[0] == 'Y');
                }
                return false;
            })();

        // syev picks its provider from the matrix shape, so workspace demand is NOT
        // monotone in the problem size: the 2*block_vectors restart problem can demand
        // more than either the block_vectors or the 3*block_vectors one. Size all three.
        auto StAS_restart = MatrixView(StAS_base, block_vectors * 2, block_vectors * 2,
                                       StAS_base.ld(), StAS_base.stride());
        const size_t ws_xtax = syev_buffer_size<B>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
        const size_t ws_stas_restart = syev_buffer_size<B>(ctx, StAS_restart, lambdas, JobType::EigenVectors, Uplo::Lower);
        const size_t ws_stas = syev_buffer_size<B>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
        size_t ws_projected = std::max(ws_xtax, std::max(ws_stas_restart, ws_stas));
        if (prefer_vendor_projected_syev) {
            const size_t ws_xtax_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower);
            const size_t ws_stas_restart_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, StAS_restart, lambdas, JobType::EigenVectors, Uplo::Lower);
            const size_t ws_stas_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, StAS_base, lambdas, JobType::EigenVectors, Uplo::Lower);
            ws_projected = std::max(ws_projected,
                                    std::max(ws_xtax_vendor, std::max(ws_stas_restart_vendor, ws_stas_vendor)));
        }
        auto syev_workspace = pool.allocate<std::byte>(ctx, ws_projected);
        // Three distinct ortho calls share this buffer: the single-matrix ortho(X) below
        // and the two external-metric variants inside the loop. All three must be in this
        // max, or a shape needing more than the others get is silently under-sized.
        auto ortho_workspace = pool.allocate<std::byte>(ctx, std::max(
                          ortho_buffer_size<B>(ctx, X, Transpose::NoTrans, params.algorithm),
                          std::max(ortho_buffer_size<B>(ctx, R, XP, Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                          ortho_buffer_size<B>(ctx, C_p, StAS_base, Transpose::NoTrans, Transpose::NoTrans, params.algorithm))));

        // Allocated last so the "does it fit?" test sees everything else already claimed.
        // A caller that filled in params.instrumentation only *after* calling
        // syevx_lobpcg_buffer_size then falls back to the host-read path (correct, but a
        // pipeline drain per iteration) instead of failing to allocate.
        const auto instr_plan = lobpcg_instrumentation_plan(params, neigs, batch_size);
        Span<float_type> stage_best, stage_current, stage_ritz;
        bool stage_device_side = false;
        if (instr_plan.active && !lobpcg_instrumentation_force_host() &&
            pool.remaining().size() >= instr_plan.template staging_bytes<float_type>(ctx)) {
            stage_best = pool.allocate<float_type>(ctx, instr_plan.samples * instr_plan.slot);
            if (instr_plan.stage_current) stage_current = pool.allocate<float_type>(ctx, instr_plan.samples * instr_plan.slot);
            if (instr_plan.stage_ritz) stage_ritz = pool.allocate<float_type>(ctx, instr_plan.samples * instr_plan.slot);
            stage_device_side = true;
        }
        size_t staged_samples = 0;


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

        if (use_jacobi) {
            trace("syevx: extract diag(A) for Jacobi");
            auto Akv = A.kernel_view();
            auto diag_ptr = jacobi_diag.data();
            auto usable_ptr = jacobi_usable.data();
            const float_type def_floor = jacobi_definiteness_floor<float_type>();
            constexpr size_t wg = 256;
            ctx->submit([&](sycl::handler& h) {
                h.parallel_for<SyevxJacobiDiagonalKernel<B, T, MFormat>>(
                    sycl::nd_range<1>(sycl::range{size_t(batch_size * wg)}, sycl::range{wg}),
                    [=](sycl::nd_item<1> item) {
                        const int bid = int(item.get_group_linear_id());
                        const int64_t tid = int64_t(item.get_local_linear_id());
                        const int64_t local_size = int64_t(item.get_local_range(0));
                        sycl::group<1> cta = item.get_group();
                        const auto Ab = Akv.batch_item(bid);
                        T* dst = diag_ptr + int64_t(bid) * n;
                        float_type local_max = float_type(0);
                        float_type local_min = std::numeric_limits<float_type>::max();
                        for (int64_t i = tid; i < n; i += local_size) {
                            T d;
                            if constexpr (MFormat == MatrixFormat::Dense) {
                                d = Ab(int(i), int(i));
                            } else {
                                // An absent entry reads back as 0, which both the
                                // definiteness test and the singularity guard then
                                // treat as unusable.
                                d = Ab.get(int(i), int(i));
                            }
                            dst[i] = d;
                            // Real part only: a Hermitian A has a real diagonal, and its
                            // sign decides whether diag(A)^{-1} is positive definite.
                            float_type dr;
                            if constexpr (internal::is_complex<T>::value) dr = d.real();
                            else                                          dr = d;
                            local_max = sycl::max(local_max, sycl::fabs(dr));
                            local_min = sycl::min(local_min, dr);
                        }
                        const float_type dmax = sycl::reduce_over_group(cta, local_max, sycl::maximum<float_type>());
                        const float_type dmin = sycl::reduce_over_group(cta, local_min, sycl::minimum<float_type>());
                        if (tid == 0) {
                            // Decided once per batch item, not per entry: a mixed-sign
                            // diagonal makes the whole operator indefinite, which no
                            // per-entry guard repairs.
                            usable_ptr[bid] = (dmax > float_type(0) && dmin > def_floor * dmax) ? 1 : 0;
                        }
                    });
            });
            trace_wait("syevx: diag(A) extracted");
        }

        // Only the X block of S is filled: P and R are recomputed from the first
        // Rayleigh-Ritz onwards. The linear index below reproduces fill_random's exactly
        // -- fill_random ignores ld/stride and walks the buffer flat -- so the starting
        // block stays bit-for-bit reproducible.
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

        // Block power-iteration start: X <- ortho(A X), a few times. Powers of A amplify
        // the largest eigendirections, so this is valid *only* for find_largest;
        // lobpcg_init_power_iterations returns 0 otherwise and this loop does not run.
        // Evidence: SYEVX_PLAN.md §7.9.
        const int init_power_steps =
            lobpcg_init_power_iterations(params.init_power_iterations, params.find_largest);
        for (int step = 0; step < init_power_steps; ++step) {
            ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
            if constexpr (MFormat == MatrixFormat::Dense) {
                gemm<B>(ctx, A, X, AX, GemmOptions<T>{});
            } else {
                spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            }
            MatrixView<T, MatrixFormat::Dense>::copy(ctx, X, AX);
        }

        trace("syevx: ortho init");
        ortho<B>(ctx, X, Transpose::NoTrans, ortho_workspace, params.algorithm);
        trace_wait("syevx: ortho init done");
        if constexpr (MFormat == MatrixFormat::Dense) {
            trace("syevx: gemm A*X");
            gemm<B>(ctx, A, X, AX, GemmOptions<T>{});
        } else {
            trace("syevx: spmm A*X");
            spmm<B>(ctx, A, X, AX, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
        }
        trace_wait("syevx: A*X done");
        trace("syevx: gemm X^T*(A*X)");
        gemm<B>(ctx, X, AX, XtAX, {.transA = trans});
        trace_wait("syevx: XtAX gemm done");
        trace("syevx: syev XtAX");
        if (prefer_vendor_projected_syev) {
            blas::dispatch::detail::syev_vendor_or_throw<B, T>(ctx, XtAX, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
        } else {
            syev<B>(ctx, XtAX, lambdas, SyevOptions{}, syev_workspace);
        }
        trace_wait("syevx: syev XtAX done");

        // The search block `X` is deliberately left in the ascending order `syev`
        // produces, even when `find_largest`: nothing inside the iteration cares about
        // the order of X's columns, they are a basis. The largest-first presentation is
        // applied exactly once, in the `X_best` snapshot, via `reported_col()` below.
        trace("syevx: gemm X*Z (update X)");
        gemm<B>(ctx, X, XtAX, X_new, GemmOptions<T>{});
        trace_wait("syevx: update X done");

        trace("syevx: gemm AX*Z (update AX)");
        gemm<B>(ctx, AX, XtAX, AX_new, GemmOptions<T>{});
        trace_wait("syevx: update AX done");

        swap_subspace();
        bool restart = true;

        // Eigenvalues currently stored per batch in `lambdas`: block_vectors after the
        // initial XtAX solve, then 2*block_vectors (restart) or 3*block_vectors.
        int64_t current_num_eigvals = block_vectors;
        int32_t completed_iterations = 0;

        size_t residual_wg_size = std::min(get_kernel_max_wg_size<SyevxResidualsKernel<B,T,MFormat>>(ctx), size_t(n));

        for(int it = 0; it < params.iterations; it++){
            completed_iterations = static_cast<int32_t>(it + 1);
            int Nvecs = restart ? block_vectors * 2 : block_vectors * 3;
            //Compute R = AX - X * diag(lambdas)
            trace("syevx: residual kernel submit");
            const float_type abs_tol = static_cast<float_type>(std::abs(params.absolute_tolerance));
            const float_type rel_tol = static_cast<float_type>(std::abs(params.relative_tolerance));
            const float_type tol = std::max(abs_tol, rel_tol);

            // Decided on the host and handed to the kernel as pre-offset pointers; null
            // means "do not store".
            const bool store_this_iteration =
                stage_device_side &&
                (static_cast<size_t>(it) % params.instrumentation->store_every) == 0 &&
                staged_samples < instr_plan.samples;
            float_type* stage_best_ptr = nullptr;
            float_type* stage_current_ptr = nullptr;
            float_type* stage_ritz_ptr = nullptr;
            if (store_this_iteration) {
                const size_t off = staged_samples * instr_plan.slot;
                stage_best_ptr = stage_best.data() + off;
                if (!stage_current.empty()) stage_current_ptr = stage_current.data() + off;
                if (!stage_ritz.empty()) stage_ritz_ptr = stage_ritz.data() + off;
            }

            const float_type lock_tol = static_cast<float_type>(tol * static_cast<float_type>(soft_lock_factor));
            auto residual_evt = ctx -> submit([&](sycl::handler& h){
                auto Rdata = R.data_ptr();
                auto Xdata = X.data_ptr();
                auto AXdata = AX.data_ptr();
                auto flags = converged_flags.data();
                auto col_conv = col_converged.data();
                auto best_quality_data = best_quality.data();
                auto Xbest_data = want_eigenvectors ? X_best.data_ptr() : nullptr;
                auto update_best = sycl::local_accessor<int32_t, 1>(1, h);
                // Per-column partials for ||r||, ||x|| and ||Ax||, accumulated in one
                // pass instead of 2*neigs sequential (barriering) group reductions.
                auto lsums = sycl::local_accessor<float_type, 1>(3 * neigs, h);
                const bool find_largest = params.find_largest;
                h.parallel_for<SyevxResidualsKernel<B,T,MFormat>>(sycl::nd_range<1>(sycl::range{size_t(batch_size*residual_wg_size)}, sycl::range{size_t(residual_wg_size)}), [=](sycl::nd_item<1> item){
                    auto num_eigvals = it < 2 ? (it+1) * block_vectors : 3*block_vectors;

                    // X's columns are in the ascending order syev returned them in, so
                    // column j of X pairs with lambda[eig_offset + j]. For find_largest the
                    // selected block is the *top* block_vectors, hence the offset.
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

                    // Form R and accumulate the three per-column norms in one sweep. The
                    // block is column-major, so flushing a running partial only on a
                    // column change turns one atomic per element into one per column.
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
                    // ||x||*|lambda| alone collapses as lambda -> 0, so a perfectly good
                    // eigenpair with a near-zero eigenvalue could never register as
                    // converged; ||Ax|| keeps the denominator bounded away from zero for
                    // any nonzero A (Duersch et al.).
                    for (size_t i = tid; i < neigs; i += local_size){
                        const float_type r_norm  = sycl::sqrt(lsums[i]);
                        const float_type x_norm  = sycl::sqrt(lsums[neigs + i]);
                        const float_type ax_norm = sycl::sqrt(lsums[2 * neigs + i]);
                        const auto eigval = blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                        const float_type denom = ax_norm + sycl::fabs(eigval) * x_norm;
                        const float_type rel = (denom > float_type(0)) ? (r_norm / denom) : r_norm;
                        blockresiduals[i] = rel;
                        // Deliberately the *current* residual, not the running best: it
                        // labels the residual block just written into R.
                        col_conv[bid * neigs + i] = (rel <= lock_tol) ? 1 : 0;
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
                    // Instrumentation history, staged device-side (SYEVX_PLAN.md §7.2).
                    if (stage_best_ptr != nullptr) {
                        for (size_t i = tid; i < neigs; i += local_size) {
                            const size_t dst = static_cast<size_t>(bid) * neigs + i;
                            stage_best_ptr[dst] = blockbestresiduals[i];
                            if (stage_current_ptr != nullptr) stage_current_ptr[dst] = blockresiduals[i];
                            if (stage_ritz_ptr != nullptr) {
                                stage_ritz_ptr[dst] = blockLambdas[params.find_largest ? (num_eigvals - 1 - i) : i];
                            }
                        }
                    }
                    if (want_eigenvectors && update_best[0] != 0) {
                        auto* blockXbest = Xbest_data + bid * (n * neigs);
                        // Where the largest-first presentation happens: slot i of the
                        // snapshot takes column reported_col(i) of X, folded into a copy
                        // that already touches every one of these elements.
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
            if (store_this_iteration) {
                ++staged_samples;
            }

            // Drain only when a host-side reader needs the results this iteration: the
            // convergence check, or instrumentation that could not be staged on the
            // device. Overshooting the stopping point by a few iterations is far cheaper
            // than a drain per iteration (SYEVX_PLAN.md §7.1).
            const bool instrumentation_host_readback = instr_plan.active && !stage_device_side;
            const bool last_iteration = (it + 1 >= static_cast<int64_t>(params.iterations));
            const bool check_convergence =
                (it % convergence_check_every == 0) || last_iteration;
            if (instrumentation_host_readback || check_convergence) {
                residual_evt.wait_and_throw();
            }
            trace("syevx: residual kernel done");

            // Fallback path only: the device staging buffer did not fit, so the histories
            // are filled by a host read of unified memory every stored iteration.
            if (instrumentation_host_readback &&
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

            // Intentionally conservative: this checks the best residual seen so far.
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
                // R_preconditioned must stay a distinct destination: the forward solve
                // writes into `out` as its temporary y while still reading `rhs`, so
                // aliasing them corrupts the solve. Queue ordering sequences the copy.
                iluk_apply<B, T>(ctx, precond, R, R_preconditioned);
                MatrixView<T, MatrixFormat::Dense>::copy(ctx, R, R_preconditioned);
                trace("syevx: ILU(k) apply done");
            }

            // ---- Soft locking, variant (a): column masking (SYEVX_PLAN.md §7.5)
            //
            // The masking happens in two places, and the reason is the one non-obvious
            // thing about the feature: `ortho` MIXES COLUMNS. Householder QR and the
            // Chol/Chol2 triangular solve both replace R by R * (upper triangular)^-1, so
            // column j of the orthonormalised block spans r_0..r_j. Masking only *after*
            // ortho therefore deletes span content the unconverged columns were
            // orthogonalised against and convergence freezes; masking only *before* it
            // leaves zero columns, which make R^T R singular and make the Cholesky-based
            // algorithms (Chol2 is the default) silently produce NaN from potrf.
            //
            // So: before ortho, *replace* each converged column with a pseudo-random
            // vector scaled to the largest surviving residual column, keeping the block
            // full rank; after ortho, zero those same columns so the injected directions
            // never reach AR/StAS. Skipped on the restart iteration, where P is a copy of
            // R taken just below and would keep the zero columns permanently.
            const bool mask_this_iteration = soft_locking && !restart;
            if (mask_this_iteration) {
                trace("syevx: soft-lock fill converged residual columns");
                ctx->submit([&](sycl::handler& h) {
                    auto Rdata = R.data_ptr();
                    const int64_t R_ld = R.ld();
                    const int64_t R_stride = R.stride();
                    auto col_conv = col_converged.data();
                    const size_t nlock = neigs;
                    const int64_t nblk = block_vectors;
                    const uint32_t seed = static_cast<uint32_t>(it) * 2654435761u + 1u;
                    auto colnorm = sycl::local_accessor<float_type, 1>(size_t(block_vectors), h);
                    auto scale_acc = sycl::local_accessor<float_type, 1>(1, h);
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * 128)}, sycl::range{size_t(128)}),
                        [=](sycl::nd_item<1> item) {
                            const auto tid = item.get_local_linear_id();
                            const auto bid = item.get_group_linear_id();
                            const auto local_size = item.get_local_range(0);
                            sycl::group<1> cta = item.get_group();
                            auto* block = Rdata + int64_t(bid) * R_stride;

                            for (int64_t j = int64_t(tid); j < nblk; j += int64_t(local_size))
                                colnorm[j] = float_type(0);
                            if (tid == 0) scale_acc[0] = float_type(0);
                            sycl::group_barrier(cta);

                            // Norm of every column that is *not* being replaced, so the
                            // filler matches the live block's scale: a unit-norm filler
                            // beside 1e-6 residual columns hands Chol2 a Gram matrix
                            // with condition ~1e12.
                            for (int64_t j = 0; j < nblk; ++j) {
                                const bool locked = (j < int64_t(nlock)) && (col_conv[bid * nlock + size_t(j)] != 0);
                                if (locked) continue;
                                const auto* col = block + j * R_ld;
                                float_type acc = 0;
                                for (int64_t i = int64_t(tid); i < n; i += int64_t(local_size))
                                    acc += internal::norm_squared(col[i]);
                                const float_type s = sycl::reduce_over_group(cta, acc, sycl::plus<float_type>());
                                if (tid == 0) colnorm[j] = sycl::sqrt(s);
                            }
                            sycl::group_barrier(cta);
                            if (tid == 0) {
                                float_type mx = 0;
                                for (int64_t j = 0; j < nblk; ++j) mx = sycl::fmax(mx, colnorm[j]);
                                // Every column locked (possible when the caller asks for
                                // no guard block): fall back to a unit scale so the
                                // filler still produces a full-rank block.
                                scale_acc[0] = (mx > float_type(0)) ? mx : float_type(1);
                            }
                            sycl::group_barrier(cta);

                            // Deterministic per (iteration, batch, column, row) hash
                            // rather than a stateful RNG: reproducible, no allocation.
                            const float_type amp =
                                scale_acc[0] * sycl::sqrt(float_type(3) / float_type(n));
                            for (size_t j = 0; j < nlock; ++j) {
                                if (col_conv[bid * nlock + j] == 0) continue;
                                auto* col = block + int64_t(j) * R_ld;
                                for (int64_t i = int64_t(tid); i < n; i += int64_t(local_size)) {
                                    uint32_t hsh = seed ^ (uint32_t(bid) * 0x9E3779B9u) ^
                                                   (uint32_t(j) * 0x85EBCA6Bu) ^ (uint32_t(i) * 0xC2B2AE35u);
                                    hsh ^= hsh >> 16; hsh *= 0x7FEB352Du;
                                    hsh ^= hsh >> 15; hsh *= 0x846CA68Bu;
                                    hsh ^= hsh >> 16;
                                    const float_type u =
                                        float_type(hsh & 0xFFFFFFu) / float_type(0x800000u) - float_type(1);
                                    col[i] = T(amp * u);
                                }
                            }
                        });
                });
                trace_wait("syevx: soft-lock fill done");
            }

            if (use_jacobi) {
                trace("syevx: Jacobi apply on residuals");
                // In-place on R: the operator is diagonal, so unlike the ILU(k) solve
                // there is no read-after-write hazard. For JacobiShifted the shift is the
                // Ritz value of the column being preconditioned; that index mirrors the
                // residual kernel's rather than sharing it, because the two kernels run at
                // different points and `lambdas` holds fewer values early on.
                const int64_t num_eigvals = it < 2 ? int64_t(it + 1) * block_vectors : 3 * block_vectors;
                const bool find_largest = params.find_largest;
                const bool shifted = jacobi_shifted;
                const float_type sing_tol = jacobi_singular_tolerance<float_type>();
                ctx->submit([&](sycl::handler& h) {
                    auto Rdata = R.data_ptr();
                    auto diag_ptr = jacobi_diag.data();
                    auto usable_ptr = jacobi_usable.data();
                    auto lambdas_span = lambdas;
                    h.parallel_for<SyevxJacobiApplyKernel<B, T, MFormat>>(
                        sycl::nd_range<1>(sycl::range{size_t(batch_size * residual_wg_size)},
                                          sycl::range{size_t(residual_wg_size)}),
                        [=](sycl::nd_item<1> item) {
                            const int64_t tid = int64_t(item.get_local_linear_id());
                            const int64_t bid = int64_t(item.get_group_linear_id());
                            const int64_t local_size = int64_t(item.get_local_range(0));
                            // R.data_ptr() already carries R's offset within S, so only
                            // the batch stride is applied here. Unshifted diag(A)^{-1}
                            // needs a positive definite diagonal; where it is not, leave
                            // the residual alone rather than apply an indefinite operator.
                            if (!shifted && usable_ptr[bid] == 0) return;
                            T* blockR = Rdata + bid * block_vectors * n * 3;
                            const T* blockD = diag_ptr + bid * n;
                            auto blockLambdas = lambdas_span.subspan(bid * num_eigvals, num_eigvals);

                            for (int64_t idx = tid; idx < n * block_vectors; idx += local_size) {
                                const int64_t col = idx / n;
                                const int64_t row = idx - col * n;
                                const float_type lam =
                                    shifted ? blockLambdas[find_largest ? (num_eigvals - 1 - col) : col]
                                            : float_type(0);
                                const T d = blockD[row];
                                const T denom = d - T(lam);
                                const float_type denom2 = internal::norm_squared(denom);
                                const float_type scale = sycl::max(sycl::fabs(lam), internal::abs(d));
                                const float_type floor_sq = (scale * sing_tol) * (scale * sing_tol);
                                if (denom2 <= floor_sq || !(denom2 > float_type(0))) continue;
                                if constexpr (internal::is_complex<T>::value) {
                                    const T inv = T(denom.real(), -denom.imag()) *
                                                  (float_type(1) / denom2);
                                    blockR[idx] = blockR[idx] * inv;
                                } else {
                                    blockR[idx] = blockR[idx] * (float_type(1) / denom);
                                }
                            }
                        });
                });
                trace_wait("syevx: Jacobi apply done");
            }

            trace("syevx: ortho R vs (X or XP)");
            ortho<B>(ctx, R, restart ? X : XP, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            trace_wait("syevx: ortho R done");

            // Second half of the masking: drop the rank-filler directions before they can
            // reach AR and StAS. See the comment above the fill kernel.
            if (mask_this_iteration) {
                trace("syevx: soft-lock zero converged residual columns");
                ctx->submit([&](sycl::handler& h) {
                    auto Rdata = R.data_ptr();
                    const int64_t R_ld = R.ld();
                    const int64_t R_stride = R.stride();
                    auto col_conv = col_converged.data();
                    const size_t nlock = neigs;
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * 128)}, sycl::range{size_t(128)}),
                        [=](sycl::nd_item<1> item) {
                            const auto tid = item.get_local_linear_id();
                            const auto bid = item.get_group_linear_id();
                            const auto local_size = item.get_local_range(0);
                            auto* block = Rdata + int64_t(bid) * R_stride;
                            for (size_t j = 0; j < nlock; ++j) {
                                if (col_conv[bid * nlock + j] == 0) continue;
                                auto* col = block + int64_t(j) * R_ld;
                                for (int64_t i = int64_t(tid); i < n; i += int64_t(local_size))
                                    col[i] = T(0);
                            }
                        });
                });
                trace_wait("syevx: soft-lock zero done");
            }


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
            if constexpr (MFormat == MatrixFormat::Dense) {
                trace("syevx: gemm A*(P or R)");
                gemm<B>(ctx, A, restart ? P : R, restart ? AP : AR, GemmOptions<T>{});
            } else {
                trace("syevx: spmm A*(P or R)");
                spmm<B>(ctx, A, restart ? P : R, restart ? AP : AR, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans, spmm_workspace);
            }
            trace_wait("syevx: A*(P or R) done");

            // A logical Nvecs x Nvecs view of a (3*block_vectors)^2 backing buffer MUST
            // keep the backing ld/stride, or batched matrices overlap and the BLAS reads
            // and writes out of bounds.
            auto StAS = MatrixView(StAS_base, Nvecs, Nvecs, StAS_base.ld(), StAS_base.stride());
            trace("syevx: gemm S^T*(A*S) (StAS)");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), AS({0,n}, {0,Nvecs}), StAS, {.transA = trans});
            trace_wait("syevx: StAS gemm done");

            // A masked residual column zeroes the corresponding row and column of StAS,
            // so the projected problem gains a spurious eigenvalue 0 that can be selected
            // as a Ritz pair -- zero columns in X_new, a hard breakdown. Plant a sentinel
            // on its diagonal instead: ||StAS||_F + 1 is strictly outside the real
            // spectrum, and the masked rows contribute nothing to that norm.
            if (mask_this_iteration) {
                trace("syevx: soft-lock StAS deflation submit");
                ctx->submit([&](sycl::handler& h) {
                    auto StAS_ptr = StAS.data_ptr();
                    const int64_t StAS_ld = StAS.ld();
                    const int64_t StAS_stride = StAS.stride();
                    auto col_conv = col_converged.data();
                    const size_t nlock = neigs;
                    const int64_t r_offset = block_vectors * 2;
                    const int64_t nv = Nvecs;
                    const bool largest = params.find_largest;
                    auto partials = sycl::local_accessor<float_type, 1>(128, h);
                    h.parallel_for(sycl::nd_range<1>(sycl::range{size_t(batch_size * 128)}, sycl::range{size_t(128)}),
                        [=](sycl::nd_item<1> item) {
                            const auto tid = item.get_local_linear_id();
                            const auto bid = item.get_group_linear_id();
                            const auto local_size = item.get_local_range(0);
                            sycl::group<1> cta = item.get_group();
                            auto* mat = StAS_ptr + int64_t(bid) * StAS_stride;
                            float_type acc = 0;
                            for (int64_t linear = int64_t(tid); linear < nv * nv; linear += int64_t(local_size)) {
                                const int64_t row = linear % nv;
                                const int64_t col = linear / nv;
                                acc += internal::norm_squared(mat[row + col * StAS_ld]);
                            }
                            partials[tid] = acc;
                            sycl::group_barrier(cta);
                            const float_type total =
                                sycl::joint_reduce(cta, partials.begin(), partials.end(), sycl::plus<float_type>());
                            const float_type sentinel = sycl::sqrt(total) + float_type(1);
                            for (size_t j = tid; j < nlock; j += local_size) {
                                if (col_conv[bid * nlock + j] == 0) continue;
                                const int64_t d = r_offset + int64_t(j);
                                mat[d + d * StAS_ld] = T(largest ? -sentinel : sentinel);
                            }
                        });
                });
                trace_wait("syevx: soft-lock StAS deflation done");
            }
            trace("syevx: syev StAS");
            if (prefer_vendor_projected_syev) {
                blas::dispatch::detail::syev_vendor_or_throw<B, T>(ctx, StAS, lambdas, JobType::EigenVectors, Uplo::Lower, syev_workspace);
            } else {
                syev<B>(ctx, StAS, lambdas, SyevOptions{}, syev_workspace);
            }
            trace_wait("syevx: syev StAS done");
            current_num_eigvals = static_cast<int64_t>(Nvecs);

            trace("syevx: post syev StAS (host)");

            // syev returns eigenvalues ascending. For find_largest we take the last
            // `block_vectors` Ritz vectors and leave them ascending; the residual kernel
            // pairs column j with lambda[Nvecs - block_vectors + j], and the X_best
            // snapshot is what flips the wanted block to largest-first on the way out.
            const int64_t eig_col_start = params.find_largest ? (Nvecs - block_vectors) : 0;
            auto Z = StAS({0, Nvecs}, {eig_col_start, eig_col_start + block_vectors});
            // X(i+1) = S * C_x. For the next search block keep only the non-X coefficient
            // rows of the selected Ritz vectors: that avoids the fragile difference
            // direction X(i+1)-X(i) while preserving the locally-optimal combination of
            // the P/R parts of the trial basis.
            auto C_p_active = MatrixView(C_p, Nvecs, block_vectors, C_p.ld(), C_p.stride());
            auto Z_search = Z({block_vectors, Nvecs}, {0, block_vectors});

            trace("syevx: build search-direction coefficients");
            C_p_active.fill_zeros(ctx);
            MatrixView<T, MatrixFormat::Dense>::copy(
                ctx,
                C_p_active({block_vectors, Nvecs}, {0, block_vectors}),
                Z_search);
            trace_wait("syevx: build search-direction coefficients done");


            // X = [X, P, R] * C_x, and AX = [AX, AP, AR] * C_x implicitly.
            trace("syevx: update X/AX submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), Z, X_new, GemmOptions<T>{});
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), Z, AX_new, GemmOptions<T>{});
            trace("syevx: ortho C_p vs Z submit");
            ortho<B>(ctx, C_p_active, Z, Transpose::NoTrans, Transpose::NoTrans, ortho_workspace, params.algorithm, params.ortho_iterations);
            // P = [X, P, R] * C_p, and AP likewise.
            trace("syevx: update P/AP submit");
            gemm<B>(ctx, S({0,n}, {0,Nvecs}), C_p_active, P_new, GemmOptions<T>{});
            gemm<B>(ctx, AS({0,n}, {0,Nvecs}), C_p_active, AP_new, GemmOptions<T>{});

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
                // X_best is already largest-first; the snapshot applied the permutation.
                MatrixView<T, MatrixFormat::Dense>::copy(ctx, V({0,n}, {0,int64_t(neigs)}), X_best);
            } else if (!params.find_largest) {
                MatrixView<T, MatrixFormat::Dense>::copy(
                    ctx, V({0,n}, {0,int64_t(neigs)}), X({0, n}, {0, static_cast<int64_t>(neigs)}));
            } else {
                // params.iterations == 0: the loop never ran, so no snapshot exists and X
                // is still ascending. The wanted block is its *last* neigs columns,
                // reversed.
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

        // One host round-trip for the whole run: drain once, then scatter the staged
        // history into the caller's spans. The caller's memory is only ever written from
        // the host, so it may be a plain std::vector.
        if (stage_device_side && staged_samples > 0) {
            ctx.wait_and_throw();
            const auto& instr = *params.instrumentation;
            const size_t batch_stride = instr.batch_stride == 0 ? neigs : instr.batch_stride;
            const size_t iter_stride = instr.iteration_stride == 0 ? batch_size * batch_stride : instr.iteration_stride;
            for (size_t sample_id = 0; sample_id < staged_samples; ++sample_id) {
                const size_t src_base = sample_id * instr_plan.slot;
                for (int64_t b = 0; b < batch_size; ++b) {
                    for (size_t i = 0; i < neigs; ++i) {
                        const size_t src = src_base + static_cast<size_t>(b) * neigs + i;
                        const size_t dst = sample_id * iter_stride + static_cast<size_t>(b) * batch_stride + i;
                        const auto best = stage_best[src];

                        if (instr.best_residual_history.size() > dst) {
                            instr.best_residual_history[dst] = best;
                        }
                        if (!stage_current.empty() && instr.current_residual_history.size() > dst) {
                            instr.current_residual_history[dst] = stage_current[src];
                        }
                        if (!stage_ritz.empty() && instr.ritz_value_history.size() > dst) {
                            instr.ritz_value_history[dst] = stage_ritz[src];
                        }
                        // Cross-iteration term: this sample's best over the previous
                        // sample's best, read from the staged series rather than from
                        // the caller's span.
                        if (instr.store_convergence_rate && instr.convergence_rate_history.size() > dst) {
                            float_type rate = float_type(1);
                            if (sample_id > 0 && instr.best_residual_history.size() > (dst - iter_stride)) {
                                const auto prev = stage_best[src - instr_plan.slot];
                                if (prev > float_type(0)) {
                                    rate = best / prev;
                                }
                            }
                            instr.convergence_rate_history[dst] = rate;
                        }
                    }
                }
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
        // Must reject exactly what the solver rejects: a sizing call that returns a number
        // for a request the solve will refuse is a caller-visible inconsistency.
        if (params.select != SyevxSelect::Extremal) {
            throw std::invalid_argument(
                "syevx_lobpcg_buffer_size: only SyevxSelect::Extremal is supported; see "
                "syevx_lobpcg");
        }
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
                // Nvecs = 2*block_vectors on restart iterations and 3*block_vectors
                // otherwise, always viewed with the backing buffer's ld. Both must be
                // sized: syev picks its provider from the shape, so workspace demand is
                // not monotone in Nvecs and 3*block_vectors does not bound the
                // 2*block_vectors case.
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
                    const size_t ws_xtax_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, XtAX_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    const size_t ws_stas_restart_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, StAS_restart_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    const size_t ws_stas_vendor = blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, T>(ctx, StAS_base_dummy, Span<typename base_type<T>::type>(), JobType::EigenVectors, Uplo::Lower);
                    ws_projected = std::max(ws_projected,
                                            std::max(ws_xtax_vendor, std::max(ws_stas_restart_vendor, ws_stas_vendor)));
                }

                work_size += BumpAllocator::allocation_size<std::byte>(ctx, ws_projected);
            }

            // Must mirror the runtime max exactly, including the single-matrix ortho(X)
            // term. Every stand-in must carry the real batch size and stride: built with
            // only (data, rows, cols, ld) a MatrixView defaults to batch_size = 1, which
            // under-allocates by a factor that grows with the batch.
            work_size += BumpAllocator::allocation_size<std::byte>(ctx,std::max(
                                                                                    ortho_buffer_size<B>(ctx, Xview, Transpose::NoTrans, params.algorithm),
                                                                                    std::max(ortho_buffer_size<B>(ctx, Xview, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),n, block_vectors*2, n, n * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm),
                                                                                    ortho_buffer_size<B>(ctx, MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), MatrixView<T,MatrixFormat::Dense>(A.data_ptr(),block_vectors * 3, block_vectors * 3, block_vectors * 3, block_vectors * block_vectors * 3, batch_size, nullptr), Transpose::NoTrans, Transpose::NoTrans, params.algorithm))));
            if constexpr (MFormat == MatrixFormat::CSR) {
                work_size += BumpAllocator::allocation_size<std::byte>(ctx,spmm_buffer_size<B>(ctx, A, Xview, AXview, T(1.0), T(0.0), Transpose::NoTrans, Transpose::NoTrans));
            }
                        
            if constexpr (MFormat == MatrixFormat::CSR) {
                // Runs ILU(k)'s symbolic phase for an upper bound on the fill; the solve
                // does not repeat it, it sub-allocates from the pool tail.
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
            work_size += BumpAllocator::allocation_size<int32_t>(ctx, neigs * batch_size); // col_converged (soft locking)
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * 3 * batch_size) * 4;                    //Sdata, ASdata, S_newdata, Stempdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * 3 * batch_size);        //StASdata
            work_size += BumpAllocator::allocation_size<T>(ctx, block_vectors * block_vectors * 3 * batch_size);            //C_pdata
            work_size += BumpAllocator::allocation_size<T>(ctx, n * block_vectors * batch_size);                            //R_preconditioned_data
            // Mirrors the runtime `if (use_jacobi)` allocations, resolved through the same
            // function on the same inputs so the two cannot drift apart.
            {
                const auto precond_kind = syevx_select_preconditioner(
                    params.preconditioner_type,
                    params.preconditioner != nullptr || params.build_preconditioner,
                    params.find_largest);
                if (precond_kind == SyevxPreconditioner::Jacobi ||
                    precond_kind == SyevxPreconditioner::JacobiShifted) {
                    work_size += BumpAllocator::allocation_size<T>(ctx, n * batch_size);           //jacobi_diag
                    work_size += BumpAllocator::allocation_size<int32_t>(ctx, batch_size);         //jacobi_usable
                }
            }
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, (block_vectors)*3 * batch_size);  //lambdas
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //residuals
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, neigs * batch_size);              //best residuals
            work_size += BumpAllocator::allocation_size<typename base_type<T>::type>(ctx, batch_size);                      //best block quality
            // Mirrors the conditional staging allocation in syevx_lobpcg exactly -- same
            // plan, same per-channel predicates. Zero when no instrumentation is attached.
            work_size += lobpcg_instrumentation_plan(params, neigs, batch_size)
                             .template staging_bytes<typename base_type<T>::type>(ctx);

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
