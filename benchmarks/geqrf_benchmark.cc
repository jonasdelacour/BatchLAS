#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <vector>
#include <random>
#include <memory>
#include <string_view>

// Helper function to generate random matrix data
using namespace batchlas;

// Single matrix GEQRF benchmark
template<typename T, Backend B>
static void BM_GEQRF_Single(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    
    auto matrix = Matrix<T>::Random(m, n, false);  // Generate random matrix
    UnifiedVector<T> tau(std::min(m, n));

    // Create queue based on backend parameter
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    // Get buffer size and allocate workspace
    size_t buffer_size = geqrf_buffer_size<B>(queue, matrix.view(), tau.to_span());
    UnifiedVector<std::byte> workspace(buffer_size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto matrix_copy = matrix;  // Reset matrix for each iteration
        state.ResumeTiming();
        geqrf<B>(queue, matrix_copy.view(), tau.to_span(), workspace.to_span());
        queue.wait();
        benchmark::DoNotOptimize(matrix_copy.data());
        benchmark::DoNotOptimize(tau.data());
    }
    
    state.SetComplexityN(m * n * std::min(m, n));
    state.counters["FLOPS"] = benchmark::Counter(
        (2 * m * n * n + (2.0 / 3.0) * n * n * n),
        benchmark::Counter::kIsRate
    );
}

// Batched matrix GEQRF benchmark
template<typename T, Backend B>
static void BM_GEQRF_Batched(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int batch_size = state.range(2);
    
    auto matrices = Matrix<T>::Random(m, n, false, batch_size);
    UnifiedVector<T> taus(batch_size * std::min(m, n));
    
    // Create queue based on backend parameter
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    // Get buffer size and allocate workspace
    size_t buffer_size = geqrf_buffer_size<B>(queue, matrices.view(), taus.to_span());
    UnifiedVector<std::byte> workspace(buffer_size);

    for (auto _ : state) {
        state.PauseTiming();
        auto matrices_copy = matrices;  // Reset matrices for each iteration
        state.ResumeTiming();
        geqrf<B>(queue, matrices_copy.view(), taus.to_span(), workspace.to_span());
        queue.wait();
        benchmark::DoNotOptimize(matrices_copy.data());
        benchmark::DoNotOptimize(taus.data());
    }
    
    state.SetComplexityN(batch_size * m * n * std::min(m, n));
    state.counters["FLOPS"] = benchmark::Counter(
        batch_size * (2 * m * n * n + (2.0 / 3.0) * n * n * n),
        benchmark::Counter::kIsRate
    );
    state.counters["BatchSize"] = batch_size;
}

// Register single matrix benchmarks
BENCHMARK_TEMPLATE(BM_GEQRF_Single, float, batchlas::Backend::NETLIB)
    ->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_GEQRF_Single, double, batchlas::Backend::NETLIB)
    ->Apply(bench_utils::SquareSizes);

BENCHMARK_TEMPLATE(BM_GEQRF_Single, float, batchlas::Backend::CUDA)
    ->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_GEQRF_Single, double, batchlas::Backend::CUDA)
    ->Apply(bench_utils::SquareSizes);

// Register batched matrix benchmarks
BENCHMARK_TEMPLATE(BM_GEQRF_Batched, float, batchlas::Backend::NETLIB)
    ->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_GEQRF_Batched, double, batchlas::Backend::NETLIB)
    ->Apply(bench_utils::SquareBatchSizes);

BENCHMARK_TEMPLATE(BM_GEQRF_Batched, float, batchlas::Backend::CUDA)
    ->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_GEQRF_Batched, double, batchlas::Backend::CUDA)
    ->Apply(bench_utils::SquareBatchSizes);

BENCHMARK_MAIN();
