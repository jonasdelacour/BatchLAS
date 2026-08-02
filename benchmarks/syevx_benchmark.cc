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

BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX, SyevxBenchSizes);
BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX_Crossover, SyevxCrossoverSizes);

MINI_BENCHMARK_MAIN();
