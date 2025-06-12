#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM(benchmark::State& state) {
    state.PauseTiming();
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    auto A = Matrix<T>::Random(m, k, false, batch);
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    state.ResumeTiming();
    for (auto _ : state) {
        gemm<B>(queue, A.view(), Bm.view(), C.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
        benchmark::DoNotOptimize(C.data());
    }
    queue.wait();
    state.counters["GFLOPS"] = benchmark::Counter(
        static_cast<double>(batch) * 1e-9 * (2.0 * m * n * k), benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_GEMM, float, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM, double, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM, float, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_GEMM, double, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);

BENCHMARK_MAIN();
