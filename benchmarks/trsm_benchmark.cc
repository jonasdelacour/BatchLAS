#include <util/minibench.hh>
#include <blas/linalg.hh>

using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    const int n = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        trsm<B>(queue, A.view(), Bm.view(), Side::Left, Uplo::Lower,
                Transpose::NoTrans, Diag::NonUnit, T(1));
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n), true);
}


MINI_BENCHMARK_REGISTER_SIZES(BM_TRSM<float, Backend::NETLIB>, SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES(BM_TRSM<double, Backend::NETLIB>, SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES(BM_TRSM<float, Backend::CUDA>, SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES(BM_TRSM<double, Backend::CUDA>, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
