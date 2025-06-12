#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single GEMV benchmark
template <typename T, Backend B>
static void BM_GEMV(minibench::State& state) {
    const int m = state.range(0);
    const int n = state.range(1);

    const int batch = state.range(2);

    auto A = Matrix<T>::Random(m, n, false, batch);
    UnifiedVector<T> x(n * batch);
    UnifiedVector<T> y(m * batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        gemv<B>(queue, A.view(), VectorView<T>(x.data(), n, 1, n, batch),
                 VectorView<T>(y.data(), m, 1, m, batch), T(1), T(0),
                 Transpose::NoTrans);
    }
    queue.wait();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n),
                    true);
    state.SetMetric("BatchSize", static_cast<double>(batch));
}


MINI_BENCHMARK_REGISTER_SIZES(gemv_float_cpu, (BM_GEMV<float, Backend::NETLIB>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(gemv_double_cpu, (BM_GEMV<double, Backend::NETLIB>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(gemv_float_gpu, (BM_GEMV<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(gemv_double_gpu, (BM_GEMV<double, Backend::CUDA>), SquareBatchSizes);

MINI_BENCHMARK_MAIN();
