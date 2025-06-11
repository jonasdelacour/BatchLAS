#include <benchmark/benchmark.h>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORMQR benchmark
template <typename T, Backend B>
static void BM_ORMQR_Single(benchmark::State& state) {
    const int m = state.range(0);

    auto A = Matrix<T>::Random(m, m, false);
    UnifiedVector<T> tau(m);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t geqrf_ws = geqrf_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);
    geqrf<B>(queue, A.view(), tau.to_span(), ws_geqrf.to_span());
    queue.wait();

    auto Q = Matrix<T>::Identity(m);
    size_t orm_ws = ormqr_buffer_size<B>(queue, A.view(), Q.view(), Side::Left,
                                         Transpose::NoTrans, tau.to_span());
    UnifiedVector<std::byte> ws(orm_ws);

    for (auto _ : state) {
        state.PauseTiming();
        auto Q_copy = Q;
        state.ResumeTiming();
        ormqr<B>(queue, A.view(), Q_copy.view(), Side::Left, Transpose::NoTrans,
                 tau.to_span(), ws.to_span());
        queue.wait();
        benchmark::DoNotOptimize(Q_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(
        4.0 * m * m * m, benchmark::Counter::kIsRate);
}

// Batched ORMQR benchmark
template <typename T, Backend B>
static void BM_ORMQR_Batched(benchmark::State& state) {
    const int m = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Random(m, m, false, batch);
    UnifiedVector<T> tau(m * batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t geqrf_ws = geqrf_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);
    geqrf<B>(queue, A.view(), tau.to_span(), ws_geqrf.to_span());
    queue.wait();

    auto Q = Matrix<T>::Identity(m, batch);
    size_t orm_ws = ormqr_buffer_size<B>(queue, A.view(), Q.view(), Side::Left,
                                         Transpose::NoTrans, tau.to_span());
    UnifiedVector<std::byte> ws(orm_ws);

    for (auto _ : state) {
        state.PauseTiming();
        auto Q_copy = Q;
        state.ResumeTiming();
        ormqr<B>(queue, A.view(), Q_copy.view(), Side::Left, Transpose::NoTrans,
                 tau.to_span(), ws.to_span());
        queue.wait();
        benchmark::DoNotOptimize(Q_copy.data());
    }

    state.counters["FLOPS"] = benchmark::Counter(
        static_cast<double>(batch) * 4.0 * m * m * m,
        benchmark::Counter::kIsRate);
    state.counters["BatchSize"] = batch;
}

BENCHMARK_TEMPLATE(BM_ORMQR_Single, float, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Single, double, Backend::NETLIB)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Single, float, Backend::CUDA)->Apply(bench_utils::SquareSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Single, double, Backend::CUDA)->Apply(bench_utils::SquareSizes);

BENCHMARK_TEMPLATE(BM_ORMQR_Batched, float, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Batched, double, Backend::NETLIB)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Batched, float, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);
BENCHMARK_TEMPLATE(BM_ORMQR_Batched, double, Backend::CUDA)->Apply(bench_utils::SquareBatchSizes);

BENCHMARK_MAIN();
