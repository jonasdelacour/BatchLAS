#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <tuple>

using namespace batchlas;

// SYEVX benchmark operating on dense symmetric matrices

template <typename T, Backend B>
static void BM_SYEVX(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);

    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 10;
    params.extra_directions = 0;
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;

    size_t ws_size = syevx_buffer_size(*q, A.view(), W.to_span(), neigs,
                                          JobType::NoEigenVectors,
                                          MatrixView<T, MatrixFormat::Dense>(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    neigs,
                    std::move(workspace),
                    JobType::NoEigenVectors,
                    MatrixView<T, MatrixFormat::Dense>(),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syevx(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}


// Crossover sweep: the same problem solved by each implemented algorithm, at a
// fixed accuracy target rather than a fixed iteration budget. Comparing at equal
// tolerance is the point — a fixed small iteration count makes LOBPCG look fast
// while returning a much less converged answer.
//
// state.range(3) is a batchlas::SyevxAlgorithm value (see SyevxCrossoverSizes).
// This is the sweep that replaces the flop-count thresholds in SYEVX_PLAN.md §2
// with measured ones.
template <typename T, Backend B>
static void BM_SYEVX_Crossover(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);
    const auto method = static_cast<SyevxAlgorithm>(state.range(3));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);

    SyevxParams<T> params;
    params.method = method;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 200;
    // A guard block is standard practice for LOBPCG and materially affects the
    // comparison; without it the iterative side is handicapped unrealistically.
    params.extra_directions = std::max<size_t>(1, neigs / 4);
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;

    size_t ws_size = syevx_buffer_size(*q, A.view(), W.to_span(), neigs,
                                          JobType::NoEigenVectors,
                                          MatrixView<T, MatrixFormat::Dense>(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    neigs,
                    std::move(workspace),
                    JobType::NoEigenVectors,
                    MatrixView<T, MatrixFormat::Dense>(),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syevx(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// The same sweep in eigenvector mode.
//
// This is not a variant for completeness: the two modes are different problems.
// Eigenvalues-only has no back-transform, which is the term the subset solver
// exists to narrow, so DirectSubset can only lose there. Every routing decision
// in syevx.cc that mentions `jobz` was made from an eigenvector-mode sweep, and
// until now that sweep lived outside the repo. Keeping it here is what makes the
// thresholds reproducible.
template <typename T, Backend B>
static void BM_SYEVX_CrossoverVectors(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);
    const auto method = static_cast<SyevxAlgorithm>(state.range(3));

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);
    auto V = Matrix<T>(n, neigs, batch);

    SyevxParams<T> params;
    params.method = method;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 200;
    params.extra_directions = std::max<size_t>(1, neigs / 4);
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;

    size_t ws_size = syevx_buffer_size(*q, A.view(), W.to_span(), neigs,
                                          JobType::EigenVectors, V.view(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    neigs,
                    std::move(workspace),
                    JobType::EigenVectors,
                    // Ownership moves into the harness: a `V.view()` here would
                    // dangle, since V is a local destroyed before the kernel runs.
                    std::move(V),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syevx(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// ---------------------------------------------------------------------------
// Sparse (CSR) crossover: Chebyshev-filtered subspace iteration vs LOBPCG.
// ---------------------------------------------------------------------------
//
// The sweeps above are all `MatrixFormat::Dense`, and on dense input the answer
// is already known and already encoded in syevx.cc: Filtered wins a niche so
// narrow (n >= 1024, k/n ~ 1%, small batch, under 2x) that `Auto` deliberately
// does not route to it. This is the question those sweeps structurally cannot
// reach.
//
// On CSR it is not the same comparison. `syevx_select_algorithm` short-circuits
// sparse input to LOBPCG -- "Sparse input has no dense fallback" -- before any of
// the (n, batch) heuristics run, so Filtered's rival here is not a tuned vendor
// eigensolver but another iterative method, one that may not converge at all.
// SYEVX_PLAN.md section 6.4 records the documented failure mode: UNPRECONDITIONED
// LOBPCG stagnates, its residuals oscillating around a floor rather than
// descending. BatchLAS's only preconditioner is ILU(k), which requires
// find_largest = false, so at the largest end LOBPCG is unpreconditioned BY
// CONSTRUCTION and section 6.4's failure mode is the default configuration there,
// not a hypothetical.
//
// TIME ALONE CANNOT ANSWER THIS, and that is the structural difference from the
// dense sweeps. A direct solver's time is its whole story; an iterative solver
// that exhausts its iteration budget and returns a badly converged answer reports
// a perfectly respectable number of microseconds, and a stagnating run reports
// the SAME number as a converging one -- both stop at `iterations`. So every case
// here also reports the accuracy it actually reached, and the two columns are
// meaningless apart: a fast time beside a large error is a failure, not a win.
//
// state.range(3) selects one of five (algorithm, spectrum end, preconditioner)
// combinations; see SparseConfig. Both ends are swept because they are genuinely
// different problems: ILU(k) exists only at the smallest end, and that is the
// only configuration in which LOBPCG gets the preconditioner its convergence
// theory assumes.
//
// state.range(4) is 10x the generator's `diagonal_boost`, i.e. how strictly
// diagonally dominant A is, which is this generator's spectral-separation knob.
// It is a first-class axis rather than a fixed constant because the two
// algorithms degrade for DIFFERENT reasons as the gap closes -- LOBPCG stalls,
// Chebyshev needs a higher degree per outer iteration -- and a single
// well-separated operating point would hide exactly that divergence.

namespace {

enum SparseConfig : int {
    kFilteredLargest    = 0,
    kLobpcgLargest      = 1,
    kFilteredSmallest   = 2,
    kLobpcgSmallest     = 3,
    kLobpcgSmallestIluk = 4,
};

// Nonzeros per row, held CONSTANT as n grows. A fixed `density` instead would
// make nnz grow as n^2 and quietly turn a sparse sweep into a dense one at the
// large end, which is the regime the sweep exists to probe.
constexpr int kSparseNnzPerRow = 16;

inline float sparse_density(int n) {
    return static_cast<float>(kSparseNnzPerRow) / static_cast<float>(n);
}

template <typename T>
SyevxParams<T> sparse_params(int config, size_t neigs) {
    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 200;
    // Same guard-block convention as the dense crossover. SYEVX_PLAN.md section
    // 7.6 measured 34-50% fewer iterations with one, so omitting it would
    // handicap LOBPCG for a reason that has nothing to do with filtering, and
    // would flatter the conclusion this sweep exists to test.
    params.extra_directions = std::max<size_t>(1, neigs / 4);
    params.absolute_tolerance = T(1e-6);
    params.relative_tolerance = T(1e-6);

    switch (config) {
        case kFilteredLargest:
            params.method = SyevxAlgorithm::Filtered;
            params.find_largest = true;
            break;
        case kLobpcgLargest:
            params.method = SyevxAlgorithm::LOBPCG;
            params.find_largest = true;
            break;
        case kFilteredSmallest:
            params.method = SyevxAlgorithm::Filtered;
            params.find_largest = false;
            break;
        case kLobpcgSmallest:
            params.method = SyevxAlgorithm::LOBPCG;
            params.find_largest = false;
            break;
        case kLobpcgSmallestIluk:
            params.method = SyevxAlgorithm::LOBPCG;
            params.find_largest = false;
            // Built INSIDE syevx out of the caller's workspace, so the timing
            // covers formation as well as application. That is the honest
            // comparison: Filtered pays no setup cost at all, and charging
            // ILU(k) only for its application would hide the difference.
            params.build_preconditioner = true;
            break;
        default:
            break;
    }
    return params;
}

// The accuracy half of the measurement, without which the timings do not mean
// anything (see the header comment).
//
// Solves one batch-1 problem drawn from the same distribution as the timed batch
// and compares the eigenvalues it returns against a full dense `syev` of that
// same matrix. EIGENVALUES rather than residuals, deliberately: a self-consistent
// (lambda, v) pair proves nothing about whether it is one of the WANTED pairs,
// which is precisely how both a stagnating LOBPCG and a mis-filtered Chebyshev
// iteration fail. tests/syevx_tests.cc makes that argument for the dense filtered
// tests; this is its sparse counterpart.
//
// A separate batch-1 matrix rather than item 0 of the timed batch: `convert_to`
// is an owning-Matrix operation, so densifying one item of a large batch would
// mean densifying all of it (4 GB at n = 4096, batch = 64). Same n, same density,
// same boost, different seed -- which is all the question needs, since it asks
// whether the algorithm converges on this CLASS of operator.
//
// Memoized on (n, neigs, config, boost): it is independent of batch size and
// would otherwise be repeated once per batch point for no new information.
template <typename T, Backend B>
double sparse_eig_error(Queue& q, int n, size_t neigs, int config, float boost) {
    using Real = typename base_type<T>::type;
    static std::map<std::tuple<int, size_t, int, int>, double> cache;
    const auto key = std::make_tuple(n, neigs, config, static_cast<int>(boost * 10.0f));
    if (auto it = cache.find(key); it != cache.end()) return it->second;

    const auto params = sparse_params<T>(config, neigs);
    double err = std::numeric_limits<double>::quiet_NaN();

    try {
        auto A = Matrix<T, MatrixFormat::CSR>::RandomSparseHermitian(
            n, sparse_density(n), 1, /*seed=*/1234u, static_cast<Real>(boost));

        UnifiedVector<Real> W(neigs);
        auto V = Matrix<T>(n, static_cast<int>(neigs), 1);

        UnifiedVector<std::byte> ws(syevx_buffer_size(
            q, A.view(), W.to_span(), neigs, JobType::EigenVectors, V.view(), params));
        syevx(q, A.view(), W.to_span(), neigs, ws.to_span(),
              JobType::EigenVectors, V.view(), params);
        q.wait_and_throw();

        auto A_dense = A.template convert_to<MatrixFormat::Dense>();
        UnifiedVector<Real> W_ref(n);
        UnifiedVector<std::byte> syev_ws(syev_buffer_size(
            q, A_dense.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower));
        syev(q, A_dense.view(), W_ref.to_span(),
             {.jobz = JobType::NoEigenVectors}, syev_ws);
        q.wait_and_throw();

        err = 0.0;
        for (size_t i = 0; i < neigs; ++i) {
            // syevx returns the wanted end first; syev returns the whole
            // spectrum ascending.
            const size_t ref_idx = params.find_largest
                                 ? static_cast<size_t>(n) - 1 - i
                                 : i;
            const double got = static_cast<double>(W[i]);
            const double ref = static_cast<double>(W_ref[ref_idx]);
            err = std::max(err, std::abs(got - ref) / std::max(std::abs(ref), 1e-30));
        }
    } catch (const std::exception&) {
        // Left as NaN on purpose. A configuration that could not run is not a
        // configuration that ran accurately, and reporting 0 would invert the
        // meaning of the column.
    }

    cache[key] = err;
    return err;
}

}  // namespace

template <typename T, Backend B>
static void BM_SYEVX_SparseImpl(minibench::State& state) {
    const int n = state.range(0);
    const size_t batch = static_cast<size_t>(state.range(1));
    const size_t neigs = static_cast<size_t>(state.range(2));
    const int config = state.range(3);
    const float boost = static_cast<float>(state.range(4)) / 10.0f;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T, MatrixFormat::CSR>::RandomSparseHermitian(
        n, sparse_density(n), static_cast<int>(batch), /*seed=*/42u,
        static_cast<typename base_type<T>::type>(boost));

    UnifiedVector<typename base_type<T>::type> W(neigs * batch);
    auto V = Matrix<T>(n, static_cast<int>(neigs), static_cast<int>(batch));

    const auto params = sparse_params<T>(config, neigs);

    // Reported before the kernel is registered, so that a configuration which
    // throws during sizing still leaves its accuracy column behind.
    state.SetMetric("Max rel. eig error",
                    sparse_eig_error<T, B>(*q, n, neigs, config, boost));

    // Eigenvector mode. Not a variant chosen for completeness: both algorithms
    // here are subspace methods that carry the vectors along whether or not the
    // caller wants them, so NoEigenVectors would measure nearly the same work
    // while leaving the accuracy probe with nothing to check -- and vectors are
    // what a sparse partial eigensolve is normally called for.
    size_t ws_size = syevx_buffer_size(*q, A.view(), W.to_span(), neigs,
                                       JobType::EigenVectors, V.view(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    // No `bench::pristine` here, unlike the dense sweeps: those
                    // need it because the Direct path runs syev, which destroys
                    // its input. Both sparse paths only ever multiply by A, so
                    // there is nothing to restore between runs -- and pristine()
                    // static_asserts on Dense anyway.
                    std::move(A),
                    std::move(W),
                    neigs,
                    std::move(workspace),
                    JobType::EigenVectors,
                    // Ownership moves into the harness; a V.view() would dangle.
                    std::move(V),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syevx(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// Shape sweep: does Filtered beat LOBPCG on CSR, and how does that move with n
// and with batch? Diagonal boost is pinned at 10 -- the value
// benchmarks/syevx_acc.cc already uses for sparse eigensolver accuracy work --
// so this sweep varies shape alone.
template <typename T, Backend B>
static void BM_SYEVX_SparseShape(minibench::State& state) {
    BM_SYEVX_SparseImpl<T, B>(state);
}

// Hardness sweep: one shape, spectral separation varied. This is the half that
// probes section 6.4's stagnation claim directly -- if unpreconditioned LOBPCG
// has a floor, closing the gap is what should expose it, and it shows up in the
// accuracy column rather than the timing one.
template <typename T, Backend B>
static void BM_SYEVX_SparseGap(minibench::State& state) {
    BM_SYEVX_SparseImpl<T, B>(state);
}

// Does WHERE in the spectrum the wanted block sits change what it costs?
//
// syevx.cc's routing says no, and says it structurally rather than empirically:
// `Direct` always runs a full syev and copies a block out of it; `DirectSubset`
// picks its band width kd from n alone, bisects the same number of steps for any
// index, and back-transforms the same fixed n x k slice wherever the block sits.
// So the crossover thresholds measured for extremal ranges were carried over to
// Index and Value ranges without re-measurement (SYEVX_RANGE_PLAN.md sections
// 8.5 and 9.2). This benchmark is the one point that CHECKS that argument
// instead of trusting it.
//
// state.range(3) is the position: 0 = bottom of the spectrum, 1 = middle,
// 2 = top. The block WIDTH is identical in all three, so any difference in the
// measured time is position and nothing else.
//
// state.range(4) is the algorithm, and `Direct` is here as a CONTROL, not as a
// second subject: it provably cannot depend on position, so whatever spread it
// shows is this machine's noise floor. A DirectSubset spread inside that band
// means flat; outside it means section 9.2 is wrong and syevx.cc's threshold
// needs a range-aware term.
//
// Eigenvector mode only. Eigenvalues-only routes to Direct at every shape, so
// there would be nothing to compare -- and the back-transform is the term whose
// position-independence is the non-obvious half of the claim.
template <typename T, Backend B>
static void BM_SYEVX_RangePosition(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);
    const int position = static_cast<int>(state.range(3));
    const auto method = static_cast<SyevxAlgorithm>(state.range(4));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);
    auto V = Matrix<T>(n, neigs, batch);

    const int64_t il = position == 0 ? int64_t(0)
                     : position == 1 ? int64_t((n - neigs) / 2)
                                     : int64_t(n - neigs);

    SyevxParams<T> params;
    params.method = method;
    // An explicit index block. find_largest is ignored under Index -- the order
    // comes from params.order -- so it is left alone deliberately.
    params.select = SyevxSelect::Index;
    params.il = il;
    params.iu = il + int64_t(neigs) - 1;
    params.order = SortOrder::Ascending;

    size_t ws_size = syevx_buffer_size<B>(*q, A.view(), W.to_span(), neigs,
                                          JobType::EigenVectors, V.view(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    neigs,
                    std::move(workspace),
                    JobType::EigenVectors,
                    // Ownership moves into the harness; a V.view() would dangle.
                    std::move(V),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syevx<B>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

namespace {

// One (n, batch) AT SATURATION and nothing else. n = 1024, batch = 128 is
// exactly syevx.cc's kSyevxSubsetMinN / kSyevxSubsetMinWork corner, i.e. the
// smallest shape at which DirectSubset is the routed choice at all -- comparing
// anywhere below it would be comparing two starved kernels, which this repo's
// measurement-hygiene rule says produces overhead ratios, not algorithm ratios.
//
// Deliberately NOT a sweep: this exists to answer one yes/no question, and the
// existing crossover map (benchmarks/syevx_crossover_rtx4090.csv) stays valid.
template <typename Benchmark>
inline void SyevxRangePositionSizes(Benchmark* b) {
    for (int position : {0, 1, 2}) {
        for (int algo : {1, 2}) {  // 1 = Direct (control), 2 = DirectSubset
            b->Args({1024, 128, 64, position, algo});
        }
    }
}

// The CPU backend has no saturation point in the same sense and DirectSubset is
// not the routed choice there; this exists so the NETLIB registration macro has
// a symbol, and gives the same yes/no answer at a size a CPU can finish.
template <typename Benchmark>
inline void SyevxRangePositionSizesNetlib(Benchmark* b) {
    for (int position : {0, 1, 2}) {
        for (int algo : {1, 2}) {
            b->Args({256, 8, 16, position, algo});
        }
    }
}

// Sparse shape sweep. neigs is ~1% of n throughout: that is the fraction at
// which a partial solver is worth reaching for at all, and holding it fixed
// keeps n and batch as the only moving parts. All five configs run at every
// shape so the two ends of the spectrum can be read side by side.
//
// n starts at 1024. Below that the dense route is available and better by a wide
// margin, so a sparse-only comparison there would be answering a question nobody
// asks; and this repo's measurement-hygiene rule is that unsaturated shapes
// produce overhead ratios rather than algorithm ratios.
template <typename Benchmark>
inline void SyevxSparseShapeSizes(Benchmark* b) {
    for (int n : {1024, 2048, 4096}) {
        for (int bs : {1, 8, 64}) {
            const int ne = std::max(4, n / 100);
            for (int cfg = 0; cfg <= 4; ++cfg) {
                b->Args({n, bs, ne, cfg, 100});   // boost 10.0
            }
        }
    }
}

template <typename Benchmark>
inline void SyevxSparseShapeSizesNetlib(Benchmark* b) {
    for (int n : {256, 512}) {
        for (int bs : {1, 8}) {
            const int ne = std::max(4, n / 100);
            for (int cfg = 0; cfg <= 4; ++cfg) {
                b->Args({n, bs, ne, cfg, 100});
            }
        }
    }
}

// Sparse hardness sweep. One shape, boost from 1.0 (a plain random sparse
// symmetric operator, no imposed diagonal dominance) up to 30.0 (strongly
// dominant, wide gaps). Batch 8 and n = 2048 sit mid-range in the shape sweep
// above, so any movement here is separation and not shape.
template <typename Benchmark>
inline void SyevxSparseGapSizes(Benchmark* b) {
    for (int boost10 : {10, 20, 50, 100, 300}) {
        for (int cfg = 0; cfg <= 4; ++cfg) {
            b->Args({2048, 8, 20, cfg, boost10});
        }
    }
}

template <typename Benchmark>
inline void SyevxSparseGapSizesNetlib(Benchmark* b) {
    for (int boost10 : {10, 50, 100}) {
        for (int cfg = 0; cfg <= 4; ++cfg) {
            b->Args({256, 4, 4, cfg, boost10});
        }
    }
}

} // namespace

BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX, SyevxBenchSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_Crossover, SyevxCrossoverSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_CrossoverVectors, SyevxCrossoverSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_RangePosition, SyevxRangePositionSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_SparseShape, SyevxSparseShapeSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_SparseGap, SyevxSparseGapSizes);

MINI_BENCHMARK_MAIN();
