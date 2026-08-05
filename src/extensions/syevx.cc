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
#include <algorithm>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <stdexcept>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "../util/template-instantiations.hh"

namespace batchlas {

// Fills the caller's `m` with a statically known count, for the two solvers that
// do not produce one themselves. See the dispatch in `syevx`.
template <Backend B, typename T, MatrixFormat MFormat>
struct SyevxFillCountsKernel;

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

// Range arguments, like the preconditioner arguments above, describe the *problem*
// and so must be rejected before dispatch -- otherwise an illegal request reaches a
// solver that would answer a different question instead of failing.
//
// Deliberately NOT implemented, from the plan's rule list: "select == Extremal and
// `order` contradicts `find_largest`". SortOrder has only Ascending and Descending
// and SyevxParams::order defaults to Ascending, so there is no way to tell an
// explicit Ascending from an unset one; the rule as written would reject the
// library's own defaults (Extremal + find_largest = true + Ascending), i.e. nearly
// every existing call. `order` is documented as ignored for Extremal instead.
//
// The remaining rule -- "`select != Extremal` may not resolve to LOBPCG or
// Filtered, and sparse input may not ask for a non-extremal range" -- lives in
// `syevx_select_algorithm` rather than here. It needs to distinguish an explicit
// SyevxParams::method from a BATCHLAS_SYEVX_ALGORITHM override (the first throws,
// the second degrades), and the selector is the only place that sees both.
template <typename T, MatrixFormat MFormat>
void validate_syevx_range_params(const SyevxParams<T>& params,
                                 int64_t n,
                                 size_t neigs,
                                 // False for the solve entry points, which have no
                                 // `m` argument to report a data-dependent count
                                 // through. True for the sizing entry points, which
                                 // write no counts at all and must therefore still
                                 // accept a Value range or sizing one is impossible.
                                 bool value_range_reportable) {
    if (params.select == SyevxSelect::Index) {
        const int64_t iu = (params.iu < 0) ? (n - 1) : params.iu;
        if (params.il < 0 || iu >= n || params.il > iu) {
            throw std::invalid_argument(
                "syevx: SyevxSelect::Index requires 0 <= il <= iu < n (iu < 0 means n-1); "
                "an empty block is expressed with neigs == 0, not with il > iu");
        }
        if (static_cast<int64_t>(neigs) != iu - params.il + 1) {
            throw std::invalid_argument(
                "syevx: SyevxSelect::Index requires neigs == iu - il + 1; neigs is validated "
                "against the range rather than derived from it so that a mismatched pair is a "
                "loud error instead of a silently under- or over-filled output buffer");
        }
    }
    if (params.select == SyevxSelect::Value) {
        if (!(params.vl < params.vu)) {
            throw std::invalid_argument(
                "syevx: SyevxSelect::Value requires vl < vu for the half-open interval "
                "(vl, vu]; an empty or inverted interval is almost always swapped arguments, "
                "and the cost of being wrong is a full O(n^3) reduction that returns nothing");
        }
        if (!value_range_reportable) {
            throw std::invalid_argument(
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

// Resolves BATCHLAS_SYEVX_ALGORITHM against SyevxParams::method.
//
// `from_env` reports WHICH of the two won, and that distinction is load-bearing:
// an environment default degrades where an explicit request throws (see the
// range rules in syevx_select_algorithm, and syevx_select_preconditioner for the
// same asymmetry). Note the environment wins whenever the variable is set at
// all, including when its value is unrecognized -- that parses to `Auto`, i.e.
// "ignore params.method and use the heuristics". That is pre-existing behaviour
// and is preserved deliberately.
SyevxAlgorithm algorithm_from_env(SyevxAlgorithm fallback, bool& from_env) {
    const char* v = std::getenv("BATCHLAS_SYEVX_ALGORITHM");
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
    // Clamp rather than reject: `neigs` is a capacity now, and a capacity above n
    // is harmless -- it just means the tail of W and V goes unwritten. Note this
    // clamps only the WORK COUNT; the caller's `neigs` remains the output stride
    // everywhere, which is the distinction that keeps batch item b's results out of
    // item b+1's slots.
    const int64_t capacity = std::min<int64_t>(static_cast<int64_t>(neigs), nn);

    switch (select) {
        case SyevxSelect::Value:
            rr.value_range = true;
            rr.il = 0;
            rr.iu = -1;                 // unused; a Value range has no static block
            rr.max_count = capacity;
            rr.reverse = (order == SortOrder::Descending);
            break;

        case SyevxSelect::Index:
            rr.value_range = false;
            rr.il = il;
            rr.iu = (iu < 0) ? (nn - 1) : iu;
            rr.max_count = rr.iu - rr.il + 1;
            rr.reverse = (order == SortOrder::Descending);
            break;

        case SyevxSelect::Extremal:
        default:
            rr.value_range = false;
            rr.il = find_largest ? (nn - capacity) : 0;
            rr.iu = find_largest ? (nn - 1) : (capacity - 1);
            rr.max_count = capacity;
            // NOT from `order`: find_largest implying descending is the historical
            // contract, and preserving it is the whole reason Extremal exists as a
            // separate selector rather than being spelled as an index block.
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

    // ---- Range feasibility ------------------------------------------------
    //
    // Only Direct and DirectSubset implement Index and Value ranges. LOBPCG
    // converges to whichever *extreme* its trial block is biased toward, and
    // syevx_filtered's Chebyshev filter is a high-pass, built by mapping the
    // unwanted interval into [-1,1] and letting the wanted END fall outside --
    // an interior interval has unwanted spectrum on both sides, which that
    // construction cannot express. Neither would fail on an interior request;
    // both would quietly answer a different question, which is why this is a
    // throw and not a degrade. See SYEVX_RANGE_PLAN.md §2.5 and §12.
    if (!extremal) {
        // Sparse: LOBPCG is the only implemented path, so there is nothing to
        // fall back to. Returning the extremal eigenpairs instead would be the
        // worst available outcome.
        if (!dense) {
            throw std::invalid_argument(
                std::string("syevx: ") + syevx_select_name(select) +
                " is not supported for sparse input; LOBPCG is the only sparse path and it "
                "can only converge to an extreme of the spectrum. Convert to dense, or use "
                "SyevxSelect::Extremal");
        }
        if (want == SyevxAlgorithm::LOBPCG || want == SyevxAlgorithm::Filtered) {
            const char* name = (want == SyevxAlgorithm::LOBPCG) ? "LOBPCG" : "Filtered";
            if (!from_env) {
                // Note the precedent immediately below, which DEGRADES an
                // unavailable algorithm to its nearest implemented neighbour.
                // That precedent deliberately does not apply here: substituting
                // an algorithm changes only the performance characteristics the
                // caller asked for, while substituting the requested part of
                // the spectrum changes the answer.
                throw std::invalid_argument(
                    std::string("syevx: SyevxAlgorithm::") + name + " cannot honour " +
                    syevx_select_name(select) +
                    " -- it computes an extreme of the spectrum by construction, so it would "
                    "silently return different eigenpairs than were asked for. Use "
                    "SyevxAlgorithm::Auto, Direct or DirectSubset for a non-extremal range");
            }
            // Environment override: degrade rather than throw. The variable
            // exists so that a whole application or test suite can be forced
            // onto one algorithm for diagnosis; aborting on the first interior
            // call would make that sweep impossible rather than informative.
            // Exactly the reasoning syevx_select_preconditioner applies to
            // BATCHLAS_SYEVX_PRECONDITIONER.
            static std::once_flag warned;
            std::call_once(warned, [name]() {
                std::fprintf(stderr,
                             "batchlas: BATCHLAS_SYEVX_ALGORITHM=%s cannot answer a non-extremal "
                             "range (SyevxSelect::Index / ::Value); degrading to Direct for those "
                             "calls. This warning is printed once per process.\n",
                             name);
            });
            // Direct, not a fall-through to the heuristics below: it is the
            // universal fallback (every scalar type, every range, every jobz),
            // and a diagnostic sweep wants one substitute, not a shape-dependent
            // one.
            return SyevxAlgorithm::Direct;
        }
    }

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
    // k does not enter any threshold below -- see the note on that at the
    // DirectSubset gate -- so `neigs` is unused past this point. Callers still
    // pass the resolved max_count rather than a raw capacity, so that this stays
    // true by construction if a k-dependent term is ever added.
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
    //
    // WHERE in the spectrum the k eigenpairs sit is absent for a stronger
    // reason: it cannot enter the cost of either path. Direct always runs a full
    // syev and then copies a block. DirectSubset's band width kd is a function of
    // n alone, its bisection does the same number of steps for every index, and
    // both back-transforms act on the same fixed n x k slice wherever the block
    // sits. So the crossovers measured for extremal ranges carry over to Index
    // and Value ranges unchanged, and no re-measurement was needed to extend
    // this routing to them. (SYEVX_RANGE_PLAN.md §8.5, §9.2.)
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
            Span<int32_t> m,
            size_t neigs,
            Span<std::byte> workspace,
            JobType jobz,
            const MatrixView<T, MatrixFormat::Dense>& V,
            const SyevxParams<T>& params) {
    validate_syevx_preconditioner_params<T, MFormat>(params);
    // This overload can report a data-dependent count, so a Value range is legal.
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/true);
    // A short `m` is an out-of-bounds device write with no host-side diagnostic
    // (Span::operator[]'s assert is compiled out in release), so it is checked
    // here rather than left to the solver. Same wording as stebz's own check.
    if (params.select == SyevxSelect::Value || !m.empty()) {
        if (static_cast<int64_t>(m.size()) < A.batch_size()) {
            throw std::invalid_argument("syevx: m must cover every batch item");
        }
    }
    // The resolved range decides the routing question ("can this algorithm answer
    // it at all?") and supplies the k the thresholds are keyed on. max_count is
    // the capacity for a Value range and the block size otherwise -- what both
    // dense paths actually do work proportional to.
    const auto rr = syevx_resolve_range(A.rows(), neigs, params);
    const auto chosen = syevx_select_algorithm(MFormat, A.rows(),
                                              static_cast<size_t>(std::max<int64_t>(rr.max_count, 0)),
                                              params.method,
                                              syevx_direct_subset_supported<T, MFormat>(), jobz,
                                              A.batch_size(), params.select);
    if (chosen == SyevxAlgorithm::Direct) {
        return syevx_direct<B, T, MFormat>(ctx, A, W, m, neigs, workspace, jobz, V, params);
    }
    if (chosen == SyevxAlgorithm::DirectSubset) {
        return syevx_direct_subset<B, T, MFormat>(ctx, A, W, m, neigs, workspace,
                                                  jobz, V, params);
    }
    // LOBPCG and Filtered only ever see an Extremal range -- syevx_select_algorithm
    // throws (or degrades to Direct) otherwise -- so the count is static and equal
    // to the resolved block size. Neither solver takes an `m` argument; filling it
    // here keeps the output contract uniform across all four algorithms.
    //
    // Submitted BEFORE the solve so that the solve's Event, which is what the
    // caller waits on, covers it on the in-order queue this library assumes
    // throughout. It aliases nothing the solvers touch.
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
        return syevx_filtered<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
    }
    return syevx_lobpcg<B, T, MFormat>(ctx, A, W, neigs, workspace, jobz, V, params);
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
    // This overload has nowhere to report a data-dependent count, so a Value
    // range is rejected here -- before any device work, and before the m-taking
    // overload below gets a chance to complain that `m` is empty.
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/false);
    // Extremal and Index both have m[b] == neigs by construction, which the
    // caller already knows, so an empty span is exactly right.
    return syevx<B, T, MFormat>(ctx, A, W, Span<int32_t>(), neigs, workspace, jobz, V, params);
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
    // Sizing writes no counts, so the "Value needs an `m` span" rule does not apply:
    // if it did, sizing the workspace for a value-range solve would be impossible.
    validate_syevx_range_params<T, MFormat>(params, A.rows(), neigs,
                                            /*value_range_reportable=*/true);
    // Resolve the range here too, and feed the selector the identical arguments
    // the solve will: routing has to make the same decision on both sides or the
    // workspace is sized for a different algorithm than the one that runs. Since
    // Phase 4 the size itself is range-dependent as well (a Value range needs
    // room for up to n eigenvalues per item in DirectSubset's internal stebz
    // output, regardless of the caller's capacity), which the sizing functions
    // derive from their own syevx_resolve_range call on the same params.
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
        const SyevxParams<BATCHLAS_UNPAREN fp>&);\
    template Event syevx<back, BATCHLAS_UNPAREN fp, fmt>(\
        Queue&,\
        const MatrixView<BATCHLAS_UNPAREN fp, fmt>&,\
        Span<typename base_type<BATCHLAS_UNPAREN fp>::type>,\
        Span<int32_t>,\
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
