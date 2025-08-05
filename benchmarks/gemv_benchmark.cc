#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Single GEMV benchmark
template <typename T, Backend B>
static void BM_GEMV(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Random(m, n, false, batch);
    UnifiedVector<T> x(n * batch);
    UnifiedVector<T> y(m * batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        gemv<B>(queue, A.view(), VectorView<T>(x.data(), n, 1, n, batch),
                 VectorView<T>(y.data(), m, 1, m, batch), T(1), T(0),
                 Transpose::NoTrans);
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n),
                    minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6,
                    minibench::Reciprocal);
}



BATCHLAS_REGISTER_BENCHMARK(BM_GEMV, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
