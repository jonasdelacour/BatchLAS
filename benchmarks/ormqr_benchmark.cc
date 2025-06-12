#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORMQR benchmark
template <typename T, Backend B>
static void BM_ORMQR(minibench::State& state) {
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
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        ormqr<B>(queue, A.view(), Q.view(), Side::Left, Transpose::NoTrans,
                 tau.to_span(), ws.to_span());
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 4.0 * m * m * m), true);
}


MINI_BENCHMARK_REGISTER_SIZES(ormqr_float_cpu, (BM_ORMQR<float, Backend::NETLIB>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(ormqr_double_cpu, (BM_ORMQR<double, Backend::NETLIB>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(ormqr_float_gpu, (BM_ORMQR<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(ormqr_double_gpu, (BM_ORMQR<double, Backend::CUDA>), SquareBatchSizes);

MINI_BENCHMARK_MAIN();
