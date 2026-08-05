#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"
using namespace batchlas;

// SYEVX benchmark operating on dense symmetric matrices

template <typename T, Backend B>
static void BM_SYEVX(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);

    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 10;
    params.extra_directions = 0;
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;

    size_t ws_size = syevx_buffer_size<B>(*q, A.view(), W.to_span(), neigs,
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
                        syevx<B>(q, std::forward<decltype(xs)>(xs)...);
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

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
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

    size_t ws_size = syevx_buffer_size<B>(*q, A.view(), W.to_span(), neigs,
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
                        syevx<B>(q, std::forward<decltype(xs)>(xs)...);
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

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
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

    size_t ws_size = syevx_buffer_size<B>(*q, A.view(), W.to_span(), neigs,
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
                        syevx<B>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
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

} // namespace

BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX, SyevxBenchSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_Crossover, SyevxCrossoverSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_CrossoverVectors, SyevxCrossoverSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_RangePosition, SyevxRangePositionSizes);

MINI_BENCHMARK_MAIN();
