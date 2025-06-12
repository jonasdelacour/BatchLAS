#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single SPMM benchmark (CSR * Dense)
template <typename T, Backend B>
static void BM_SPMM(minibench::State& state) {
    const int m = state.range(0);
    const int k = state.range(1);
    const int n = state.range(2);
    const int batch = state.range(3);

    auto A_dense = Matrix<T>::Random(m, k, false, batch);
    auto A = A_dense.template convert_to<MatrixFormat::CSR>();
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);

    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t ws_size = spmm_buffer_size<B>(queue, A.view(), Bm.view(), C.view(),
                                        T(1), T(0), Transpose::NoTrans,
                                        Transpose::NoTrans);
    UnifiedVector<std::byte> workspace(ws_size);

    for (auto _ : state) {
        state.PauseTiming();
        auto C_copy = C;
        state.ResumeTiming();
        spmm<B>(queue, A.view(), Bm.view(), C_copy.view(), T(1), T(0),
                Transpose::NoTrans, Transpose::NoTrans, workspace.to_span());
        queue.wait();
    }
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * A.nnz() * n), true);
    state.SetMetric("BatchSize", static_cast<double>(batch));
}

static auto* bench_spmm_f_cpu =
    minibench::RegisterBenchmark("spmm_float_cpu", BM_SPMM<float, Backend::NETLIB>);
static auto* bench_spmm_d_cpu =
    minibench::RegisterBenchmark("spmm_double_cpu", BM_SPMM<double, Backend::NETLIB>);
static auto* bench_spmm_f_gpu =
    minibench::RegisterBenchmark("spmm_float_gpu", BM_SPMM<float, Backend::CUDA>);
static auto* bench_spmm_d_gpu =
    minibench::RegisterBenchmark("spmm_double_gpu", BM_SPMM<double, Backend::CUDA>);

bench_utils::CubeBatchSizes(bench_spmm_f_cpu);
bench_utils::CubeBatchSizes(bench_spmm_d_cpu);
bench_utils::CubeBatchSizes(bench_spmm_f_gpu);
bench_utils::CubeBatchSizes(bench_spmm_d_gpu);

MINI_BENCHMARK_MAIN();
