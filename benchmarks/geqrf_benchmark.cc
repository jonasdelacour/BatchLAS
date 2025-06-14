#include <util/minibench.hh>
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
static void BM_GEQRF(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch_size = state.range(2);

    auto matrices = Matrix<T>::Random(m, n, false, batch_size);
    UnifiedVector<T> tau(batch_size * std::min(m, n));

    // Create queue based on backend parameter
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    // Get buffer size and allocate workspace
    size_t buffer_size = geqrf_buffer_size<B>(queue, matrices.view(), tau.to_span());
    UnifiedVector<std::byte> workspace(buffer_size);
    state.ResetTiming(); state.ResumeTiming();    
    for (auto _ : state) {
        geqrf<B>(queue, matrices.view(), tau.to_span(), workspace.to_span());
    }
    queue.wait();
    auto time = state.StopTiming();
    state.SetMetric("GFLOPS", batch_size * (1e-9 * (2 * m * n * n + (2.0 / 3.0) * n * n * n)), true);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch_size) * time * 1e3, false);

}


MINI_BENCHMARK_REGISTER_SIZES((BM_GEQRF<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_GEQRF<double, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_GEQRF<float, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_GEQRF<double, Backend::NETLIB>), SquareBatchSizesNetlib);

MINI_BENCHMARK_MAIN();
