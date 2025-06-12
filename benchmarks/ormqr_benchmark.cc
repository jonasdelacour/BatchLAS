#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORMQR benchmark
template <typename T, Backend B>
static void BM_ORMQR(minibench::State& state) {
    const int m = state.range(0);
    const int batch = state.range(1);

    auto A = Matrix<T>::Random(m, m, false, batch);
    UnifiedVector<T> tau(m * batch);
    Queue queue(B == Backend::NETLIB ? "cpu" : "gpu");
    size_t geqrf_ws = geqrf_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws_geqrf(geqrf_ws);
    geqrf<B>(queue, A.view(), tau.to_span(), ws_geqrf.to_span());
    queue.wait();

    auto Q = Matrix<T>::Identity(m, batch);
    size_t orm_ws = ormqr_buffer_size<B>(queue, A.view(), Q.view(), Side::Left,
                                         Transpose::NoTrans, tau.to_span());
    UnifiedVector<std::byte> ws(orm_ws);

    for (auto _ : state) {
        ormqr<B>(queue, A.view(), Q.view(), Side::Left, Transpose::NoTrans,
                 tau.to_span(), ws.to_span());
    }
    queue.wait();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 4.0 * m * m * m), true);
    state.SetMetric("BatchSize", static_cast<double>(batch));
}

static auto* bench_ormqr_f_cpu =
    minibench::RegisterBenchmark("ormqr_float_cpu", BM_ORMQR<float, Backend::NETLIB>);
static auto* bench_ormqr_d_cpu =
    minibench::RegisterBenchmark("ormqr_double_cpu", BM_ORMQR<double, Backend::NETLIB>);
static auto* bench_ormqr_f_gpu =
    minibench::RegisterBenchmark("ormqr_float_gpu", BM_ORMQR<float, Backend::CUDA>);
static auto* bench_ormqr_d_gpu =
    minibench::RegisterBenchmark("ormqr_double_gpu", BM_ORMQR<double, Backend::CUDA>);

bench_utils::SquareBatchSizes(bench_ormqr_f_cpu);
bench_utils::SquareBatchSizes(bench_ormqr_d_cpu);
bench_utils::SquareBatchSizes(bench_ormqr_f_gpu);
bench_utils::SquareBatchSizes(bench_ormqr_d_gpu);

MINI_BENCHMARK_MAIN();
