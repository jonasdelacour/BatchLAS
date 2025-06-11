#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single SPMM benchmark (CSR * Dense)
template <typename T, Backend B>
static void BM_SPMM_Single(benchmark::State& state) {
    const int m = state.range(0);
    const int k = state.range(1);
    const int n = state.range(2);

    // Create random dense matrix and convert to CSR for sparsity
    auto A_dense = Matrix<T>::Random(m, k, false);
    auto A = A_dense.template convert_to<MatrixFormat::CSR>();
    auto Bm = Matrix<T>::Random(k, n, false);
    auto C = Matrix<T>::Random(m, n, false);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t ws_size = spmm_buffer_size<B>(queue, A.view(), Bm.view(), C.view(),
                                        T(1), T(0), Transpose::NoTrans,
                                        Transpose::NoTrans);
    UnifiedVector<std::byte> workspace(ws_size);

    for (auto _ : state) {
        state.PauseTiming();
        auto C_copy = C;
        state.ResumeTiming();
        spmm<B>(queue, A.view(), Bm.view(), C_copy.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans, workspace.to_span());
        queue.wait();
        benchmark::DoNotOptimize(C_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(
        2.0 * A.nnz() * n, benchmark::Counter::kIsRate);
}

// Batched SPMM benchmark
template <typename T, Backend B>
static void BM_SPMM_Batched(benchmark::State& state) {
    const int m = state.range(0);
    const int k = state.range(1);
    const int n = state.range(2);
    const int batch = state.range(3);

    auto A_dense = Matrix<T>::Random(m, k, false, batch);
    auto A = A_dense.template convert_to<MatrixFormat::CSR>();
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t ws_size = spmm_buffer_size<B>(queue, A.view(), Bm.view(), C.view(),
                                        T(1), T(0), Transpose::NoTrans,
                                        Transpose::NoTrans);
    UnifiedVector<std::byte> workspace(ws_size);

    for (auto _ : state) {
        state.PauseTiming();
        auto C_copy = C;
        state.ResumeTiming();
        spmm<B>(queue, A.view(), Bm.view(), C_copy.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans, workspace.to_span());
        queue.wait();
        benchmark::DoNotOptimize(C_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(
        static_cast<double>(batch) * 2.0 * A.nnz() * n,
        benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_SPMM_Single, float, Backend::NETLIB)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Single, double, Backend::NETLIB)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Single, float, Backend::CUDA)->Apply(bench_utils::CubeSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Single, double, Backend::CUDA)->Apply(bench_utils::CubeSizes);

BENCHMARK_TEMPLATE(BM_SPMM_Batched, float, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Batched, double, Backend::NETLIB)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Batched, float, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);
BENCHMARK_TEMPLATE(BM_SPMM_Batched, double, Backend::CUDA)->Apply(bench_utils::CubeBatchSizes);

BENCHMARK_MAIN();
