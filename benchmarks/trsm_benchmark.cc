#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    const int n = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");

    for (auto _ : state) {
        trsm<B>(queue, A.view(), Bm.view(), Side::Left, Uplo::Lower,
                Transpose::NoTrans, Diag::NonUnit, T(1));
    }
    queue.wait();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n), true);
    state.SetMetric("BatchSize", static_cast<double>(batch));
}

static auto* bench_trsm_f_cpu =
    minibench::RegisterBenchmark("trsm_float_cpu", BM_TRSM<float, Backend::NETLIB>);
static auto* bench_trsm_d_cpu =
    minibench::RegisterBenchmark("trsm_double_cpu", BM_TRSM<double, Backend::NETLIB>);
static auto* bench_trsm_f_gpu =
    minibench::RegisterBenchmark("trsm_float_gpu", BM_TRSM<float, Backend::CUDA>);
static auto* bench_trsm_d_gpu =
    minibench::RegisterBenchmark("trsm_double_gpu", BM_TRSM<double, Backend::CUDA>);

bench_utils::SquareBatchSizes(bench_trsm_f_cpu);
bench_utils::SquareBatchSizes(bench_trsm_d_cpu);
bench_utils::SquareBatchSizes(bench_trsm_f_gpu);
bench_utils::SquareBatchSizes(bench_trsm_d_gpu);

MINI_BENCHMARK_MAIN();
