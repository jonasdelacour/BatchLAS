#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>

using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        trsm<B>(queue, A.view(), Bm.view(), Side::Left, Uplo::Lower,
                Transpose::NoTrans, Diag::NonUnit, T(1));
    }
    queue.wait();
    auto time = state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n), true);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * time * 1e3, false);
}


#ifdef BATCHLAS_HAS_CUDA_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<double, Backend::CUDA>), SquareBatchSizes);
#endif
#ifdef BATCHLAS_HAS_ROCM_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<float, Backend::ROCM>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<double, Backend::ROCM>), SquareBatchSizes);
#endif
#ifdef BATCHLAS_HAS_MKL_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<float, Backend::MKL>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<double, Backend::MKL>), SquareBatchSizes);
#endif
#ifdef BATCHLAS_HAS_HOST_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<float, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_TRSM<double, Backend::NETLIB>), SquareBatchSizesNetlib);
#endif

MINI_BENCHMARK_MAIN();
