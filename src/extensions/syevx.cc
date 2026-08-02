// syevx: dispatch over the partial-eigensolve algorithm families.
//
// `syevx` is not one algorithm. The right method depends on the matrix format and
// on how much of the spectrum is wanted; see SYEVX_PLAN.md §2 for the cost model
// that produces the thresholds below.
//
//   dense, n <= SMALL_N            -> Direct
//   dense, eigenvalues only        -> Direct   (subset lost 3-5x at every shape)
//   dense, vectors, n <  SUBSET_N  -> Direct
//   dense, vectors, n >= SUBSET_N  -> DirectSubset
//   sparse                         -> LOBPCG
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
// The headline: the flop model said DirectSubset should beat Direct by ~3x for
// k/n below 25%. It does not. cuSOLVER's full eigensolve is enough better
// optimized than our two-stage + subset chain that a 3x flop advantage does not
// survive contact with it. Measured, DirectSubset is
//
//   * SLOWER than Direct at every shape measured in eigenvalues-only mode
//     (3-5x slower -- the reduction is pure cost there, with no back-transform
//     to narrow), and
//   * slower for n <= 512 even with eigenvectors,
//   * faster only at n >= 1024 with eigenvectors, and by 1.16-1.46x, not 3x.
//
// So Auto now sends dense input to Direct unless it is in the one regime the
// subset solver actually wins.
constexpr int64_t kSyevxSmallN = 64;

// With eigenvectors, DirectSubset only starts paying at this dimension. At n=512
// it still lost by 1.1-1.3x; at n=1024 it won by 1.16x (batch 1) to 1.46x
// (batch 64). Raise or lower this from measurement, not from the cost model.
constexpr int64_t kSyevxSubsetMinN = 1024;

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
                                      JobType jobz) {
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

    if (subset_supported && n >= kSyevxSubsetMinN) return SyevxAlgorithm::DirectSubset;

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
                                              syevx_direct_subset_supported<T, MFormat>(), jobz);
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
                                              syevx_direct_subset_supported<T, MFormat>(), jobz);
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
