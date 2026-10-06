// syevx: dispatch over the partial-eigensolve algorithm families. The routing
// rule (dense: Direct, or DirectSubset for real T with eigenvectors at large
// n * batch; sparse: LOBPCG, or Filtered on request) is measured, not flop-derived.
// evidence: docs/perf/syevx.md#syevx-routing-thresholds-as-they-stand

#include "../linalg-impl.hh"
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <stdexcept>
#include <batchlas/blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"
#include <batchlas/settings.hh>

namespace batchlas {

// Fills the caller's `m` with a statically known count, for the two solvers that
// do not produce one themselves. See the dispatch in `syevx`.
template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxFillCountsKernel;

namespace {

// Measured against a Direct that really is the vendor solver. Trap: the first
// grid compared against our batch-starved blocked syev; do not reuse it.
// evidence: docs/perf/syevx.md#syevx-the-direct-baseline-was-never-cusolver
constexpr int64_t kSyevxSmallN = 64;

// DirectSubset (eigenvectors) needs both n and n * batch large; evidence at the use site.
constexpr int64_t kSyevxSubsetMinN = 1024;
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

// Preconditioner and range arguments describe the *problem*, so they are validated
// before dispatch: a solver that ignores them must not see an illegal request.
// evidence: docs/design/syevx.md#syevx-dispatcher-problem-arguments-are-validated-before-dispatch
template <typename T, MatrixFormat MFormat>
void validate_syevx_preconditioner_params(const SyevxParams<T>& params) {
    if (params.preconditioner != nullptr && params.build_preconditioner) {
        throw batchlas::invalid_argument(
            "syevx: SyevxParams::preconditioner and SyevxParams::build_preconditioner are "
            "mutually exclusive; supply a factor or ask syevx to build one, not both");
    }
    const bool iluk_configured = params.preconditioner != nullptr || params.build_preconditioner;
    // ILU(k) and Jacobi approximate A^{-1}: smallest eigenpairs only. JacobiShifted
    // is allowed with find_largest deliberately, since its shift tracks the Ritz value.
    // evidence: docs/design/syevx.md#syevx-dispatcher-which-preconditioner-is-legal-at-which-end
    if (iluk_configured && params.find_largest) {
        throw batchlas::invalid_argument(
            "syevx: an ILU(k) preconditioner approximates A^{-1} and is only valid when "
            "searching for the smallest eigenpairs; set SyevxParams::find_largest = false "
            "or clear SyevxParams::preconditioner / build_preconditioner");
    }
    if constexpr (MFormat != MatrixFormat::CSR) {
        if (params.build_preconditioner) {
            throw batchlas::invalid_argument(
                "syevx: SyevxParams::build_preconditioner requires a CSR matrix; ILU(k) is "
                "only defined for sparse input");
        }
    }
    // preconditioner_type must agree with the ILU(k) fields, or one request is dropped.
    if (params.preconditioner_type == SyevxPreconditioner::ILUK && !iluk_configured) {
        throw batchlas::invalid_argument(
            "syevx: SyevxPreconditioner::ILUK requires SyevxParams::preconditioner or "
            "SyevxParams::build_preconditioner to be set");
    }
    if (iluk_configured && params.preconditioner_type != SyevxPreconditioner::Auto &&
        params.preconditioner_type != SyevxPreconditioner::ILUK) {
        throw batchlas::invalid_argument(
            "syevx: an ILU(k) factor was supplied or requested but "
            "SyevxParams::preconditioner_type asks for a different family; clear one of them");
    }
    if (params.preconditioner_type == SyevxPreconditioner::Jacobi && params.find_largest) {
        throw batchlas::invalid_argument(
            "syevx: SyevxPreconditioner::Jacobi is diag(A)^{-1}, an approximate A^{-1}, and is "
            "only valid when searching for the smallest eigenpairs; set "
            "SyevxParams::find_largest = false or use SyevxPreconditioner::JacobiShifted, "
            "whose shift makes it valid at either end");
    }
}

// "Extremal + contradicting order throws" is deliberately absent, and the
// "non-extremal needs a dense direct path" rule lives in syevx_select_algorithm.
// evidence: docs/design/syevx.md#syevx-dispatcher-problem-arguments-are-validated-before-dispatch
template <typename T, MatrixFormat MFormat>
void validate_syevx_range_params(const SyevxParams<T>& params,
                                 int64_t n,
                                 size_t neigs,
                                 // False only where no `m` can report a Value count;
                                 // true for sizing, which must accept a Value range.
                                 bool value_range_reportable) {
    if (params.select == SyevxSelect::Index) {
        const int64_t iu = (params.iu < 0) ? (n - 1) : params.iu;
        if (params.il < 0 || iu >= n || params.il > iu) {
            throw batchlas::invalid_argument(
                "syevx: SyevxSelect::Index requires 0 <= il <= iu < n (iu < 0 means n-1); "
                "an empty block is expressed with neigs == 0, not with il > iu");
        }
        if (static_cast<int64_t>(neigs) != iu - params.il + 1) {
            throw batchlas::invalid_argument(
                "syevx: SyevxSelect::Index requires neigs == iu - il + 1; neigs is validated "
                "against the range rather than derived from it so that a mismatched pair is a "
                "loud error instead of a silently under- or over-filled output buffer");
        }
    }
    if (params.select == SyevxSelect::Value) {
        if (!(params.vl < params.vu)) {
            throw batchlas::invalid_argument(
                "syevx: SyevxSelect::Value requires vl < vu for the half-open interval "
                "(vl, vu]; an empty or inverted interval is almost always swapped arguments, "
                "and the cost of being wrong is a full O(n^3) reduction that returns nothing");
        }
        if (!value_range_reportable) {
            throw batchlas::invalid_argument(
                "syevx: SyevxSelect::Value needs the overload that takes an `m` output span -- "
                "the number of eigenvalues in an interval is data-dependent and differs per "
                "batch item, so it cannot be inferred from neigs (which is only a capacity)");
        }
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

// `from_env` is load-bearing: an environment default degrades where an explicit
// request throws. Trap: a set-but-unrecognized value still wins, as Auto (deliberate).
// evidence: docs/design/syevx.md#syevx-dispatcher-environment-override-versus-explicit-request
SyevxAlgorithm algorithm_from_env(SyevxAlgorithm fallback, bool& from_env) {
    const char* v = batchlas::settings().selection.syevx_algorithm.get();
    from_env = (v != nullptr && *v != '\0');
    if (!from_env) return fallback;
    return parse_syevx_algorithm(v);
}

const char* syevx_select_name(SyevxSelect select) {
    switch (select) {
        case SyevxSelect::Index: return "SyevxSelect::Index";
        case SyevxSelect::Value: return "SyevxSelect::Value";
        default:                 return "SyevxSelect::Extremal";
    }
}

} // namespace

SyevxResolvedRange syevx_resolve_range(int64_t n,
                                       size_t neigs,
                                       SyevxSelect select,
                                       bool find_largest,
                                       int64_t il,
                                       int64_t iu,
                                       SortOrder order) {
    SyevxResolvedRange rr{};
    const int64_t nn = std::max<int64_t>(n, 0);
    // Clamps only the WORK COUNT; the caller's `neigs` stays the output stride
    // everywhere, which keeps item b's results out of item b+1's slots.
    const int64_t capacity = std::min<int64_t>(static_cast<int64_t>(neigs), nn);

    switch (select) {
        case SyevxSelect::Value:
            rr.value_range = true;
            rr.il = 0;
            rr.iu = -1;                 // unused; a Value range has no static block
            rr.max_count = capacity;
            rr.reverse = (order == SortOrder::Descending);
            break;

        case SyevxSelect::Index: {
            rr.value_range = false;
            // Clamped into [0, n-1], not merely translated: syevx_direct indexes
            // lam[il .. il+max_count-1] with no bound of its own.
            // evidence: docs/design/syevx-range-selection.md#syevx-range-normalization-and-validation
            const int64_t lo = std::max<int64_t>(il, 0);
            const int64_t hi = std::min<int64_t>((iu < 0) ? (nn - 1) : iu, nn - 1);
            if (hi < lo) {
                // Canonical empty block: iu - il + 1 == max_count for every range.
                rr.il = 0;
                rr.iu = -1;
                rr.max_count = 0;
            } else {
                rr.il = lo;
                rr.iu = hi;
                rr.max_count = hi - lo + 1;
            }
            rr.reverse = (order == SortOrder::Descending);
            break;
        }

        case SyevxSelect::Extremal:
        default:
            rr.value_range = false;
            rr.il = find_largest ? (nn - capacity) : 0;
            rr.iu = find_largest ? (nn - 1) : (capacity - 1);
            rr.max_count = capacity;
            // NOT from `order`: find_largest implying descending is the historical contract.
            rr.reverse = find_largest;
            break;
    }
    return rr;
}

SyevxAlgorithm syevx_select_algorithm(MatrixFormat format,
                                      int64_t n,
                                      size_t neigs,
                                      SyevxAlgorithm requested,
                                      bool subset_supported,
                                      JobType jobz,
                                      int64_t batch_size,
                                      SyevxSelect select) {
    bool from_env = false;
    const SyevxAlgorithm want = algorithm_from_env(requested, from_env);
    const bool dense = (format == MatrixFormat::Dense);
    const bool extremal = (select == SyevxSelect::Extremal);

    // Only Direct and DirectSubset answer Index/Value ranges; LOBPCG and Filtered
    // would silently return an extreme instead, so this throws rather than degrades.
    // evidence: docs/design/syevx-range-selection.md#syevx-range-throw-do-not-degrade
    if (!extremal) {
        if (!dense) {
            throw batchlas::invalid_argument(
                std::string("syevx: ") + syevx_select_name(select) +
                " is not supported for sparse input; LOBPCG is the only sparse path and it "
                "can only converge to an extreme of the spectrum. Convert to dense, or use "
                "SyevxSelect::Extremal");
        }
        if (want == SyevxAlgorithm::LOBPCG || want == SyevxAlgorithm::Filtered) {
            const char* name = (want == SyevxAlgorithm::LOBPCG) ? "LOBPCG" : "Filtered";
            if (!from_env) {
                // Unlike the degrade below: a substituted spectrum changes the answer.
                throw batchlas::invalid_argument(
                    std::string("syevx: SyevxAlgorithm::") + name + " cannot honour " +
                    syevx_select_name(select) +
                    " -- it computes an extreme of the spectrum by construction, so it would "
                    "silently return different eigenpairs than were asked for. Use "
                    "SyevxAlgorithm::Auto, Direct or DirectSubset for a non-extremal range");
            }
            // Environment override: degrade (to Direct, the one universal
            // substitute) so a forced-algorithm diagnostic sweep can finish.
            static std::once_flag warned;
            std::call_once(warned, [name]() {
                std::fprintf(stderr,
                             "batchlas: BATCHLAS_SYEVX_ALGORITHM=%s cannot answer a non-extremal "
                             "range (SyevxSelect::Index / ::Value); degrading to Direct for those "
                             "calls. This warning is printed once per process.\n",
                             name);
            });
            return SyevxAlgorithm::Direct;
        }
    }

    // Sparse: dense-only requests degrade to LOBPCG, but an explicit Filtered is
    // honoured (it has real spmm branches). Trap: must stay below the range check.
    // evidence: docs/design/syevx.md#syevx-dispatcher-sparse-input-and-an-explicit-filtered-request
    if (!dense) {
        return (want == SyevxAlgorithm::Filtered) ? SyevxAlgorithm::Filtered
                                                  : SyevxAlgorithm::LOBPCG;
    }

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
    // k enters no threshold below; callers still pass the resolved max_count.
    (void)neigs;

    // Eigenvalues-only: Direct won every measured shape (3-5x).
    if (jobz != JobType::EigenVectors) return SyevxAlgorithm::Direct;

    // Batch-parallel reduction starves at small batch, hence the n * batch term.
    // k and spectrum position are deliberately absent.
    // evidence: docs/perf/syevx.md#syevx-directsubset-batch-crossover-with-eigenvectors
    // evidence: docs/design/syevx-range-selection.md#syevx-range-why-the-thresholds-carry-over-unchanged
    if (subset_supported && n >= kSyevxSubsetMinN &&
        n * batch_size >= kSyevxSubsetMinWork) {
        return SyevxAlgorithm::DirectSubset;
    }

    // Filtered's small-batch dense niche is deliberately left opt-in.
    // evidence: docs/perf/syevx.md#syevx-the-first-gpu-crossover-measurement-superseded-baseline
    return SyevxAlgorithm::Direct;
}

SyevxPreconditioner syevx_select_preconditioner(SyevxPreconditioner requested,
                                                bool iluk_configured,
                                                bool find_largest) {
    if (requested != SyevxPreconditioner::Auto) return requested;
    // A configured ILU(k) factor wins over any environment default.
    if (iluk_configured) return SyevxPreconditioner::ILUK;
    const SyevxPreconditioner from_env =
        parse_syevx_preconditioner(batchlas::settings().selection.syevx_preconditioner.get());
    // ILUK from the environment is ignored (no factor), and Jacobi degrades for
    // find_largest where an explicit request would throw.
    // evidence: docs/design/syevx.md#syevx-dispatcher-environment-override-versus-explicit-request
    if (from_env == SyevxPreconditioner::Jacobi && !find_largest) return SyevxPreconditioner::Jacobi;
    if (from_env == SyevxPreconditioner::JacobiShifted) return SyevxPreconditioner::JacobiShifted;
    return SyevxPreconditioner::None;
}

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx(Queue& ctx,
            const MatrixView<T, MFormat>& A,
            Span<typename base_type<T>::type> W,
            Span<int32_t> m,
            size_t neigs,
            Span<std::byte> workspace,
            JobType jobz,
            const MatrixView<T, MatrixFormat::Dense>& V,
            const SyevxParams<T>& params,
            Span<int32_t> info) {
    validate_syevx_preconditioner_params<T, MFormat>(params);
    // This overload can report a data-dependent count, so a Value range is legal.
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/true);
    // A short `m` is a silent out-of-bounds device write in release builds.
    if (params.select == SyevxSelect::Value || !m.empty()) {
        if (static_cast<int64_t>(m.size()) < A.batch_size()) {
            throw batchlas::invalid_argument("syevx: m must cover every batch item");
        }
    }
    const auto rr = syevx_resolve_range(A.rows(), neigs, params);
    const auto chosen = syevx_select_algorithm(MFormat, A.rows(),
                                              static_cast<size_t>(std::max<int64_t>(rr.max_count, 0)),
                                              params.method,
                                              syevx_direct_subset_supported<T, MFormat>(), jobz,
                                              A.batch_size(), params.select);
    // `info` (converged?) is not `m` (how many); each arm clears and fills info.
    if (chosen == SyevxAlgorithm::Direct) {
        return syevx_direct<B, T, MFormat>(ctx, A, W, m, neigs, workspace, jobz, V, params, info);
    }
    if (chosen == SyevxAlgorithm::DirectSubset) {
        return syevx_direct_subset<B, T, MFormat>(ctx, A, W, m, neigs, workspace,
                                                  jobz, V, params, info);
    }
    // LOBPCG/Filtered only see Extremal ranges, so m is static; filled here, and
    // BEFORE the solve so the returned Event covers it on the in-order queue.
    if (!m.empty()) {
        const int64_t batch_size = A.batch_size();
        const int32_t count = static_cast<int32_t>(std::max<int64_t>(rr.max_count, 0));
        int32_t* m_ptr = m.data();
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for<SyevxFillCountsKernel<B, T, MFormat>>(
                sycl::range<1>(static_cast<size_t>(batch_size)),
                [=](sycl::id<1> idx) { m_ptr[idx[0]] = count; });
        });
    }
    if (chosen == SyevxAlgorithm::Filtered) {
        return syevx_filtered<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params, info);
    }
    return syevx_lobpcg<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params, info);
}

template <Backend B, typename T, MatrixFormat MFormat>
Event syevx(Queue& ctx,
            const MatrixView<T, MFormat>& A,
            Span<typename base_type<T>::type> W,
            size_t neigs,
            Span<std::byte> workspace,
            JobType jobz,
            const MatrixView<T, MatrixFormat::Dense>& V,
            const SyevxParams<T>& params,
            Span<int32_t> info) {
    // No `m` here, so a Value range is rejected before any device work.
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/false);
    return syevx<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace, jobz, V, params, info);
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
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/true);
    // Selector arguments must be identical to the solve's, or the workspace is
    // sized for a different algorithm than the one that runs.
    const auto rr = syevx_resolve_range(A.rows(), neigs, params);
    const auto chosen = syevx_select_algorithm(MFormat, A.rows(),
                                              static_cast<size_t>(std::max<int64_t>(rr.max_count, 0)),
                                              params.method,
                                              syevx_direct_subset_supported<T, MFormat>(), jobz,
                                              A.batch_size(), params.select);
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
        const SyevxParams<BATCHLAS_UNPAREN fp>&,\
        Span<int32_t>);\
    template Event syevx<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        Span<int32_t>,\
        size_t,\
        Span<std::byte>,\
        JobType,\
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,\
        const SyevxParams<BATCHLAS_UNPAREN fp>&,\
        Span<int32_t>);\
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

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEVX_INSTANTIATE_FOR_BACKEND_TYPE)

#undef SYEVX_INSTANTIATE_FOR_BACKEND_TYPE
#undef SYEVX_INSTANTIATE

} // namespace batchlas
