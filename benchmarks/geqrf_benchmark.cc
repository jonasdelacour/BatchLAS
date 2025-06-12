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
    const int m = state.range(0);
    const int n = state.range(1);
    const int batch_size = state.range(2);

    auto matrices = Matrix<T>::Random(m, n, false, batch_size);
    UnifiedVector<T> tau(batch_size * std::min(m, n));

    // Create queue based on backend parameter
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    // Get buffer size and allocate workspace
    size_t buffer_size = geqrf_buffer_size<B>(queue, matrices.view(), tau.to_span());
    UnifiedVector<std::byte> workspace(buffer_size);
    
    for (auto _ : state) {
        state.PauseTiming();
        auto matrices_copy = matrices;  // Reset matrices for each iteration
        state.ResumeTiming();
        geqrf<B>(queue, matrices_copy.view(), tau.to_span(), workspace.to_span());
        queue.wait();
    }
    state.SetComplexityN(batch_size * m * n * std::min(m, n));
    state.SetMetric("GFLOPS", batch_size * (1e-9 * (2 * m * n * n + (2.0 / 3.0) * n * n * n)), true);
    state.SetMetric("BatchSize", static_cast<double>(batch_size));
}

static auto* bench_geqrf_f_cpu =
    minibench::RegisterBenchmark("geqrf_float_cpu", BM_GEQRF<float, Backend::NETLIB>);
static auto* bench_geqrf_d_cpu =
    minibench::RegisterBenchmark("geqrf_double_cpu", BM_GEQRF<double, Backend::NETLIB>);
static auto* bench_geqrf_f_gpu =
    minibench::RegisterBenchmark("geqrf_float_gpu", BM_GEQRF<float, Backend::CUDA>);
static auto* bench_geqrf_d_gpu =
    minibench::RegisterBenchmark("geqrf_double_gpu", BM_GEQRF<double, Backend::CUDA>);

bench_utils::SquareBatchSizes(bench_geqrf_f_cpu);
bench_utils::SquareBatchSizes(bench_geqrf_d_cpu);
bench_utils::SquareBatchSizes(bench_geqrf_f_gpu);
bench_utils::SquareBatchSizes(bench_geqrf_d_gpu);

MINI_BENCHMARK_MAIN();
