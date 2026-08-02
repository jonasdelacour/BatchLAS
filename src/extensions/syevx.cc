// syevx: dispatch over the partial-eigensolve algorithm families.
//
// `syevx` is not one algorithm. The right method depends on the matrix format and
// on how much of the spectrum is wanted; see SYEVX_PLAN.md §2 for the cost model
// that produces the thresholds below.
//
//   dense, n <= SMALL_N            -> Direct   (a subset solver cannot beat syev_cta)
//   dense, neigs/n >  DENSE_DIRECT -> Direct   (iterative methods cannot amortize)
//   dense, neigs/n >  ITERATIVE    -> DirectSubset
//   dense, neigs/n <= ITERATIVE    -> DirectSubset (Filtered, Tier 3, would go here)
//   sparse                         -> LOBPCG
//
// DirectSubset requires a real scalar type and dense input; where it is not
// available the choice degrades to Direct (or LOBPCG below the iterative
// threshold, where a full decomposition is clearly wrong).
//
// The thresholds are derived from flop counts, not measurement. Producing measured
// ones is the deliverable of `benchmarks/syevx_benchmark.cc`.

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

// Matrices at or below this dimension always use the full solver: the work a
// subset solver removes is already hidden behind tridiagonalization there.
constexpr int64_t kSyevxSmallN = 64;

// Above this fraction of the spectrum, a full decomposition wins outright.
constexpr double kSyevxDenseDirectFraction = 0.25;

// Below this fraction, iterative/filtered methods can amortize their matvecs.
constexpr double kSyevxIterativeFraction = 0.02;

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
    const bool use_preconditioner = params.preconditioner != nullptr || params.build_preconditioner;
    // An ILU(k) factorization approximates A^{-1}, so it only accelerates the
    // smallest eigenpairs; for the largest it damps exactly what is being sought.
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
                                      bool subset_supported) {
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

    if (n <= kSyevxSmallN) return SyevxAlgorithm::Direct;
    if (n <= 0) return SyevxAlgorithm::Direct;

    const double fraction = static_cast<double>(neigs) / static_cast<double>(n);
    if (fraction > kSyevxDenseDirectFraction) return SyevxAlgorithm::Direct;
    if (fraction > kSyevxIterativeFraction) {
        return subset_supported ? SyevxAlgorithm::DirectSubset : SyevxAlgorithm::Direct;
    }
    // Below the iterative threshold, Filtered (Tier 3) is the algorithm the cost
    // model favours -- but Auto still picks the subset solver where it is
    // available, because that one is direct: no convergence risk, no degree to
    // tune, and its cost is exactly what the model says. Filtered has a
    // convergence failure mode the direct path does not, and the crossover has
    // not been measured on real hardware (SYEVX_PLAN.md §2.4). Promote it to the
    // Auto default for this band once it has been.
    return subset_supported ? SyevxAlgorithm::DirectSubset : SyevxAlgorithm::LOBPCG;
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
                                              syevx_direct_subset_supported<T, MFormat>());
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
                                              syevx_direct_subset_supported<T, MFormat>());
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
