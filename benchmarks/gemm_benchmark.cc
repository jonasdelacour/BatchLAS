#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    auto A = Matrix<T>::Random(m, k, false, batch);
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        gemm<B>(queue, A.view(), Bm.view(), C.view(), T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), true);
}



// Register size/batch combinations at static‑init time using macro
//MINI_BENCHMARK_REGISTER_SIZES(BM_GEMM<float, Backend::NETLIB>, CubeBatchSizesNetlib);
//MINI_BENCHMARK_REGISTER_SIZES(BM_GEMM<double, Backend::NETLIB>, CubeBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES(BM_GEMM<float, Backend::CUDA>, CubeBatchSizes);
//MINI_BENCHMARK_REGISTER_SIZES(BM_GEMM<double, Backend::CUDA>, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
