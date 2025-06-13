#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORGQR benchmark
template <typename T, Backend B>
static void BM_ORGQR(minibench::State& state) {
    const int m = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Random(m, m, false, batch);
    UnifiedVector<T> tau(m * batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t geqrf_ws = geqrf_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);
    geqrf<B>(queue, A.view(), tau.to_span(), ws_geqrf.to_span());
    queue.wait();

    size_t org_ws = orgqr_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws(org_ws);
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        orgqr<B>(queue, A.view(), tau.to_span(), ws.to_span());
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * m * m), true);
}

MINI_BENCHMARK_REGISTER_SIZES((BM_ORGQR<float, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORGQR<double, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORGQR<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORGQR<double, Backend::CUDA>), SquareBatchSizes);

MINI_BENCHMARK_MAIN();

