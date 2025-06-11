#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM_Single(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);

    auto A = Matrix<T>::Random(m, k, false);
    auto Bm = Matrix<T>::Random(k, n, false);
    auto C = Matrix<T>::Random(m, n, false);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto C_copy = C;
        state.ResumeTiming();
        gemm<B>(queue, A.view(), Bm.view(), C_copy.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans);
        queue.wait();
        benchmark::DoNotOptimize(C_copy.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(1e-9 * (2.0 * m * n * k),
                                                   benchmark::Counter::kIsRate);
}

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM_Batched(benchmark::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);
    const int k = state.range(2);
    const int batch = state.range(3);

    auto A = Matrix<T>::Random(m, k, false, batch);
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto C_copy = C;
        state.ResumeTiming();
        gemm<B>(queue, A.view(), Bm.view(), C_copy.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans);
        queue.wait();
        benchmark::DoNotOptimize(C_copy.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(batch) * 1e-9 * (2.0 * m * n * k), benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_GEMM_Single, float, Backend::NETLIB)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Single, double, Backend::NETLIB)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Single, float, Backend::CUDA)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Single, double, Backend::CUDA)->Apply(bench_utils::CubeSizes);

BENCHMARK_TEMPLATE(BM_GEMM_Batched, float, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Batched, double, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Batched, float, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM_Batched, double, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);

BENCHMARK_MAIN();
