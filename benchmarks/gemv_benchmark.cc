#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single GEMV benchmark
template <typename T, Backend B>
static void BM_GEMV_Single(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);

    auto A = Matrix<T>::Random(m, n, false);
    UnifiedVector<T> x(n);
    UnifiedVector<T> y(m);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto y_copy = y;
        state.ResumeTiming();
        gemv<B>(queue, A.view(), VectorView<T>(x.data(), n, 1),
                 VectorView<T>(y_copy.data(), m, 1), T(1), T(0),
                 Transpose::NoTrans);
        queue.wait();
        benchmark::DoNotOptimize(y_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(2.0 * m * n,
                                                   benchmark::Counter::kIsRate);
}

// Batched GEMV benchmark
template <typename T, Backend B>
static void BM_GEMV_Batched(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int batch = state.range(2);

    auto A = Matrix<T>::Random(m, n, false, batch);
    UnifiedVector<T> x(n * batch);
    UnifiedVector<T> y(m * batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto y_copy = y;
        state.ResumeTiming();
        gemv<B>(queue, A.view(), VectorView<T>(x.data(), n, 1, n, batch),
                 VectorView<T>(y_copy.data(), m, 1, m, batch), T(1), T(0),
                 Transpose::NoTrans);
        queue.wait();
        benchmark::DoNotOptimize(y_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(
        static_cast<double>(batch) * 2.0 * m * n, benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_GEMV_Single, float, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Single, double, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Single, float, Backend::CUDA)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Single, double, Backend::CUDA)->Apply(bench_utils::SquareSizes);

BENCHMARK_TEMPLATE(BM_GEMV_Batched, float, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Batched, double, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Batched, float, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMV_Batched, double, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);

BENCHMARK_MAIN();
