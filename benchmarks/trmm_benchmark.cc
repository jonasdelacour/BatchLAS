#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/extensions.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_TRMM(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu", false);
    auto C = Matrix<T>::Random(m, n, false, batch);
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto A = Matrix<T>::RandomTriangular(n, Uplo::Lower, Diag::NonUnit, batch);

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        trmm<B>(queue, A.view(), Bm.view(), C.view(), T(1), Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit);
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK(BM_TRMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
