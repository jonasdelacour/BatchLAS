// syevx: dispatch over the partial-eigensolve algorithm families.
//
// `syevx` is not one algorithm. The right method depends on the matrix format and
// on how much of the spectrum is wanted; see SYEVX_PLAN.md §2 for the cost model
// that produces the thresholds below.
//
//   dense, n <= SMALL_N                       -> Direct
//   dense, eigenvalues only                   -> Direct
//   dense, vectors, n <  SUBSET_N             -> Direct
//   dense, vectors, n >= SUBSET_N, small batch-> Direct
//   dense, vectors, n >= SUBSET_N, big batch  -> DirectSubset
//   sparse                                    -> LOBPCG
//
// DirectSubset requires a real scalar type and dense input; where it is not
// available the choice degrades to Direct (or LOBPCG below the iterative
// threshold, where a full decomposition is clearly wrong).
//
// The thresholds below are MEASURED on an RTX 4090 via `BM_SYEVX_Crossover` in
// `benchmarks/syevx_benchmark.cc` plus an eigenvector-mode sweep; they are no
// longer the flop-count estimates this file originally shipped with, and the two
// disagree sharply. See the note above kSyevxSubsetMinN.

#include "../linalg-impl.hh"
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <complex>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"

namespace batchlas {

namespace {

// MEASURED thresholds (RTX 4090, CUDA backend, float). These replace the
// flop-count estimates that stood here until the sweep in
// benchmarks/syevx_benchmark.cc was finally run on GPU; see SYEVX_PLAN.md §13.
//
// A correction to what this comment used to say. It attributed DirectSubset's
// loss to cuSOLVER being "better optimized than our two-stage + subset chain".
// That comparison never happened: `Direct` calls `syev`, and `syev`'s Auto order
// listed BatchLAS_Blocked ahead of Vendor, so on a GPU with n > 32 the baseline
// was always our own *blocked* solver, never cuSOLVER. The blocked reduction is
// parallel over the batch and starves at small batch -- at n=1024, batch=1 its
// panel kernel is 88% of the solve -- so the old baseline was slow for exactly
// the same reason DirectSubset is slow there, and the two comparing "evenly" at
// batch 1 was two starved kernels, not a fair fight.
//
// With that fixed (see syev_prefer_vendor in include/blas/functions/syev.hh),
// Direct got up to 15.4x faster and the thresholds below had to be re-measured
// against it. What survives:
//
//   * eigenvalues-only: Direct still wins everywhere -- the subset path pays the
//     full reduction with no back-transform to narrow, so it has nothing to win
//     with. Unchanged conclusion, sounder baseline.
//   * with eigenvectors: DirectSubset wins only at large n AND large batch, by
//     up to 2.4x; at small batch it now loses by up to 16x. The old gate was n
//     alone, which sent batch-1 calls into that loss.
constexpr int64_t kSyevxSmallN = 64;

// With eigenvectors, DirectSubset only starts paying at this dimension...
constexpr int64_t kSyevxSubsetMinN = 1024;

// ...and enough total work to fill the device. See the table at the use site:
// n=1024 needs batch >= 128 and n=2048 needs batch >= 64, and both are this
// product. Below it DirectSubset loses, by up to 16x at batch 1.
constexpr int64_t kSyevxSubsetMinWork = 128 * 1024;

SyevxAlgorithm parse_syevx_algorithm(const char* v) {
    if (!v || !*v) return SyevxAlgorithm::Auto;
    std::string s(v);
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (s == "auto") return SyevxAlgorithm::Auto;
    if (s == "direct") return SyevxAlgorithm::Direct;
    if (s == "direct_subset" || s == "direct-subset") return SyevxAlgorithm::DirectSubset;
    if (s == "filtered") return SyevxAlgorithm::Filtered;
    if (s == "lobpcg") return SyevxAlgorithm::LOBPCG;
    // Unknown value: stay conservative.
    return SyevxAlgorithm::Auto;
}

// Preconditioner arguments describe the *problem*, not the algorithm, so they have
// to be validated before dispatch. They used to be checked inside syevx_lobpcg,
// which was equivalent only while every path led there; once dense input started
// routing to Direct/DirectSubset, an illegal combination on a dense matrix
// silently reached a solver that ignores it.
template <typename T, MatrixFormat MFormat>
void validate_syevx_preconditioner_params(const SyevxParams<T>& params) {
    if (params.preconditioner != nullptr && params.build_preconditioner) {
        throw std::invalid_argument(
            "syevx: SyevxParams::preconditioner and SyevxParams::build_preconditioner are "
            "mutually exclusive; supply a factor or ask syevx to build one, not both");
    }
    const bool iluk_configured = params.preconditioner != nullptr || params.build_preconditioner;
    // An ILU(k) factorization approximates A^{-1}, so it only accelerates the
    // smallest eigenpairs; for the largest it damps exactly what is being sought.
    //
    // Whether the same restriction applies to Jacobi depends on which Jacobi.
    //
    // `Jacobi` = diag(A)^{-1} is an approximate A^{-1} just as ILU(k) is, differing
    // only in how crude it is, so it inherits the restriction verbatim. That is not
    // a theoretical concern: forcing it on with find_largest turned 21-47 iterations
    // into 127-300 (i.e. non-convergence at the cap) across the sweep in
    // tests/syevx_tests.cc, in the same direction and for the same reason as ILU(k).
    //
    // `JacobiShifted` = (diag(A) - lambda I)^{-1} is a different operator: its shift
    // comes from the *current Ritz value*, so it is a diagonal approximation to
    // (A - lambda I)^{-1} and amplifies whatever is near lambda -- the wanted end by
    // construction, at either end of the spectrum. Allowing find_largest with it is
    // a deliberate decision backed by the same sweep (0.85-1.2x on random symmetric
    // input either way), not an oversight.
    if (iluk_configured && params.find_largest) {
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
    // An explicit preconditioner_type has to be consistent with the ILU(k) fields.
    // Anything else silently drops one of the two requests: either a factor the
    // caller built at real cost is never applied, or a family is asked for that has
    // nothing behind it.
    if (params.preconditioner_type == SyevxPreconditioner::ILUK && !iluk_configured) {
        throw std::invalid_argument(
            "syevx: SyevxPreconditioner::ILUK requires SyevxParams::preconditioner or "
            "SyevxParams::build_preconditioner to be set");
    }
    if (iluk_configured && params.preconditioner_type != SyevxPreconditioner::Auto &&
        params.preconditioner_type != SyevxPreconditioner::ILUK) {
        throw std::invalid_argument(
            "syevx: an ILU(k) factor was supplied or requested but "
            "SyevxParams::preconditioner_type asks for a different family; clear one of them");
    }
    if (params.preconditioner_type == SyevxPreconditioner::Jacobi && params.find_largest) {
        throw std::invalid_argument(
            "syevx: SyevxPreconditioner::Jacobi is diag(A)^{-1}, an approximate A^{-1}, and is "
            "only valid when searching for the smallest eigenpairs; set "
            "SyevxParams::find_largest = false or use SyevxPreconditioner::JacobiShifted, "
            "whose shift makes it valid at either end");
    }
}

SyevxPreconditioner parse_syevx_preconditioner(const char* v) {
    if (!v || !*v) return SyevxPreconditioner::Auto;
    std::string s(v);
    for (char& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    if (s == "auto") return SyevxPreconditioner::Auto;
    if (s == "none" || s == "off") return SyevxPreconditioner::None;
    if (s == "jacobi" || s == "diagonal" || s == "diag") return SyevxPreconditioner::Jacobi;
    if (s == "jacobi_shifted" || s == "jacobi-shifted") return SyevxPreconditioner::JacobiShifted;
    if (s == "iluk" || s == "ilu") return SyevxPreconditioner::ILUK;
    return SyevxPreconditioner::Auto;
}

SyevxAlgorithm algorithm_from_env(SyevxAlgorithm fallback) {
    const char* v = std::getenv("BATCHLAS_SYEVX_ALGORITHM");
    if (!v || !*v) return fallback;
    return parse_syevx_algorithm(v);
}

} // namespace

SyevxAlgorithm syevx_select_algorithm(MatrixFormat format,
                                      int64_t n,
                                      size_t neigs,
                                      SyevxAlgorithm requested,
                                      bool subset_supported,
                                      JobType jobz,
                                      int64_t batch_size) {
    const SyevxAlgorithm want = algorithm_from_env(requested);
    const bool dense = (format == MatrixFormat::Dense);

    // Sparse input has no dense fallback: LOBPCG is the only implemented option.
    if (!dense) return SyevxAlgorithm::LOBPCG;

    if (want != SyevxAlgorithm::Auto) {
        switch (want) {
            case SyevxAlgorithm::Direct:       return SyevxAlgorithm::Direct;
            case SyevxAlgorithm::LOBPCG:       return SyevxAlgorithm::LOBPCG;
            case SyevxAlgorithm::DirectSubset:
                return subset_supported ? SyevxAlgorithm::DirectSubset : SyevxAlgorithm::Direct;
            case SyevxAlgorithm::Filtered:     return SyevxAlgorithm::Filtered;
            default:                           break;
        }
    }

    if (n <= kSyevxSmallN || n <= 0) return SyevxAlgorithm::Direct;
    (void)neigs;

    // Eigenvalues-only: Direct won at every measured shape, by 3-5x. The subset
    // path pays the full reduction and has no back-transform to save on, so there
    // is nothing for it to win with.
    if (jobz != JobType::EigenVectors) return SyevxAlgorithm::Direct;

    // DirectSubset's reduction is parallel over the batch, exactly like the
    // blocked syev it used to be compared against, so it starves at small batch
    // for the same reason. The previous gate was n alone, which sent batch-1
    // calls -- its worst case -- straight into it.
    //
    // MEASURED (RTX 4090, float, eigenvectors, BM_SYEVX_CrossoverVectors),
    // Direct/DirectSubset, so > 1 means DirectSubset wins:
    //
    //   n=1024, k=8:    b=1 0.09   b=4 0.34   b=16 0.36   b=64 1.00   b=256 2.40
    //   n=2048, k=8:    b=1 0.06   b=4 0.28   b=16 0.43   b=64 1.12   b=256 1.98
    //   n=1024, b=128:  k=8 1.47   k=25 1.57  k=51 1.38   k=102 1.51
    //   n=1024, b=256:  k=8 2.12   k=25 1.93  k=51 1.96   k=102 1.83
    //                   k=256 1.43  k=512 1.00
    //
    // Two anchors bound the win region: n=1024 needs batch >= 128, n=2048 needs
    // batch >= 64. Both are `n * batch >= 128 * 1024`, which is the form used
    // here. Above n=2048 that extrapolates rather than interpolates, but it
    // extrapolates in the direction the two anchors already move.
    //
    // k is deliberately absent: the ratio is flat in k from 0.8% to 25% of the
    // spectrum and only decays to a tie at 50%, so it does not discriminate.
    if (subset_supported && n >= kSyevxSubsetMinN &&
        n * batch_size >= kSyevxSubsetMinWork) {
        return SyevxAlgorithm::DirectSubset;
    }

    // Filtered wins a genuine but narrow niche -- n >= 1024 at k/n around 1%, and
    // only at small batch (at batch 64 Direct won there too). It is left opt-in
    // rather than routed to by Auto: the margin is under 2x, it is the only path
    // with a convergence failure mode, and the niche is too batch-dependent to
    // encode from three data points.
    return SyevxAlgorithm::Direct;
}

SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested,
                                                bool iluk_configured,
                                                bool find_largest) {
    if (requested != SyevxPreconditioner::Auto) return requested;
    // A configured ILU(k) factor is the strongest signal of intent there is, and it
    // was paid for before the call, so it wins over any environment default.
    if (iluk_configured) return SyevxPreconditioner::ILUK;
    const SyevxPreconditioner from_env =
        parse_syevx_preconditioner(std::getenv("BATCHLAS_SYEVX_PRECONDITIONER"));
    // ILUK from the environment is not actionable: there is no factor and syevx
    // will not silently build one behind the caller's back (that needs CSR input and
    // find_largest = false, neither of which the environment can know).
    //
    // An environment default degrades where an explicit request would throw. The
    // point of the variable is "run this whole application/suite with X" for
    // diagnosis; making it abort on the first call that happens to want the largest
    // eigenpairs would make that sweep impossible rather than informative.
    if (from_env == SyevxPreconditioner::Jacobi && !find_largest) return SyevxPreconditioner::Jacobi;
    if (from_env == SyevxPreconditioner::JacobiShifted) return SyevxPreconditioner::JacobiShifted;
    return SyevxPreconditioner::None;
}

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx(Queue& ctx,
            const MatrixView<T, MFormat>& A,
            Span<typename base_type<T>::type> W,
            size_t neigs,
            Span<std::byte> workspace,
            JobType jobz,
            const MatrixView<T, MatrixFormat::Dense>& V,
            const SyevxParams<T>& params) {
    validate_syevx_preconditioner_params<T, MFormat>(params);
    const auto chosen = syevx_select_algorithm(MFormat, A.rows(), neigs, params.method,
                                              syevx_direct_subset_supported<T, MFormat>(), jobz,
                                              A.batch_size());
    if (chosen == SyevxAlgorithm::Direct) {
        return syevx_direct<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
    }
    if (chosen == SyevxAlgorithm::DirectSubset) {
        return syevx_direct_subset<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
    }
    if (chosen == SyevxAlgorithm::Filtered) {
        return syevx_filtered<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
    }
    return syevx_lobpcg<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t syevx_buffer_size(Queue& ctx,
                         const MatrixView<T, MFormat>& A,
                         Span<typename base_type<T>::type> W,
                         size_t neigs,
                         JobType jobz,
                         const MatrixView<T, MatrixFormat::Dense>& V,
                         const SyevxParams<T>& params) {
    validate_syevx_preconditioner_params<T, MFormat>(params);
    const auto chosen = syevx_select_algorithm(MFormat, A.rows(), neigs, params.method,
                                              syevx_direct_subset_supported<T, MFormat>(), jobz,
                                              A.batch_size());
    if (chosen == SyevxAlgorithm::Direct) {
        return syevx_direct_buffer_size<B, T, MFormat>(ctx, A, W, neigs, jobz, V, params);
    }
    if (chosen == SyevxAlgorithm::DirectSubset) {
        return syevx_direct_subset_buffer_size<B, T, MFormat>(ctx, A, W, neigs, jobz, V, params);
    }
    if (chosen == SyevxAlgorithm::Filtered) {
        return syevx_filtered_buffer_size<B, T, MFormat>(ctx, A, W, neigs, jobz, V, params);
    }
    return syevx_lobpcg_buffer_size<B, T, MFormat>(ctx, A, W, neigs, jobz, V, params);
}

#define SYEVX_INSTANTIATE(back, fp, fmt) \
    template Event syevx<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template size_t syevx_buffer_size<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        size_t,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&);

#define SYEVX_INSTANTIATE_FOR_BACKEND_TYPE(back, fp) \
    BATCHLAS_FOR_EACH_MATRIX_FORMAT_2(SYEVX_INSTANTIATE, back, fp)

#define SYEVX_INSTANTIATE_FOR_BACKEND(back)\
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEVX_INSTANTIATE_FOR_BACKEND_TYPE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEVX_INSTANTIATE_FOR_BACKEND(Backend::CUDA);
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    SYEVX_INSTANTIATE_FOR_BACKEND(Backend::ROCM);
#endif
#if BATCHLAS_HAS_HOST_BACKEND
    SYEVX_INSTANTIATE_FOR_BACKEND(Backend::NETLIB);
#endif

#undef SYEVX_INSTANTIATE_FOR_BACKEND
#undef SYEVX_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_INSTANTIATE

} // namespace batchlas
