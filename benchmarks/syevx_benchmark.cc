#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <batchlas/backend_config.h>

using namespace batchlas;

// SYEVX benchmark operating on dense symmetric matrices

template <typename T, Backend B>
static void BM_SYEVX(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t neigs = state.range(2);

    auto A = Matrix<T>::Random(n, n, true, batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<typename base_type<T>::type> W(neigs * batch);

    SyevxParams<T> params;
    params.algorithm = OrthoAlgorithm::Chol2;
    params.iterations = 10;
    params.extra_directions = 0;
    params.find_largest = true;
    params.absolute_tolerance = 1e-6;
    params.relative_tolerance = 1e-6;

    size_t ws_size = syevx_buffer_size<B>(queue, A.view(), W.to_span(), neigs,
                                         JobType::NoEigenVectors,
                                         MatrixView<T, MatrixFormat::Dense>(), params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        syevx<B>(queue, A.view(), W.to_span(), neigs, workspace.to_span(),
                 JobType::NoEigenVectors, MatrixView<T, MatrixFormat::Dense>(), params);
    }
    queue.wait();
    auto time = state.StopTiming();
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * time * 1e3, false);
}



BATCHLAS_REGISTER_BENCHMARK(BM_SYEVX, SyevxBenchSizes);

MINI_BENCHMARK_MAIN();
