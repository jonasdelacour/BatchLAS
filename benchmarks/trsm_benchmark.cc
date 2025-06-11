#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM_Single(benchmark::State& state) {
    const int n = state.range(0);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5));
    auto Bm = Matrix<T>::Random(n, n, false);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto B_copy = Bm;
        state.ResumeTiming();
        trsm<B>(queue, A.view(), B_copy.view(), Side::Left, Uplo::Lower,
                Transpose::NoTrans, Diag::NonUnit, T(1));
        queue.wait();
        benchmark::DoNotOptimize(B_copy.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(1e-9 * 1.0 * n * n,
                                                   benchmark::Counter::kIsRate);
}

// Batched TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM_Batched(benchmark::State& state) {
    const int n = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        state.PauseTiming();
        auto B_copy = Bm;
        state.ResumeTiming();
        trsm<B>(queue, A.view(), B_copy.view(), Side::Left, Uplo::Lower,
                Transpose::NoTrans, Diag::NonUnit, T(1));
        queue.wait();
        benchmark::DoNotOptimize(B_copy.data());
    }

    state.counters["GFLOPS"] = benchmark::Counter(1e-9 *
        static_cast<double>(batch) * n * n, benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_TRSM_Single, float, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Single, double, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Single, float, Backend::CUDA)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Single, double, Backend::CUDA)->Apply(bench_utils::SquareSizes);

BENCHMARK_TEMPLATE(BM_TRSM_Batched, float, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Batched, double, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Batched, float, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_TRSM_Batched, double, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);

BENCHMARK_MAIN();
