#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_Ortho(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);
    const OrthoAlgorithm algo = static_cast<OrthoAlgorithm>(state.range(3));

    auto A = Matrix<T, MatrixFormat::Dense>::Random(m, n, false, batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    auto workspace = UnifiedVector<std::byte>(ortho_buffer_size<B>(queue, A.view(), Transpose::NoTrans, algo));

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        ortho<B>(queue, A.view(), Transpose::NoTrans, workspace.to_span(), algo);
    }
    queue.wait();
    auto time = state.StopTiming();

    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * time * 1e3, false);
}

BATCHLAS_REGISTER_BENCHMARK(BM_Ortho, OrthoBenchSizes);

MINI_BENCHMARK_MAIN();