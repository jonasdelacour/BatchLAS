#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORMQR benchmark
template <typename T, Backend B>
static void BM_ORMQR(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Random(m, n, false, batch);
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
    state.ResetTiming(); state.ResumeTiming();
    for (auto _ : state) {
        ormqr<B>(queue, A.view(), Q.view(), Side::Left, Transpose::NoTrans,
                 tau.to_span(), ws.to_span());
    }
    queue.wait();
    state.StopTiming();
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * (4 * m * n * n - 2 * n * n * n + 3 * n * n)), true);
    //Appendix C https://www.netlib.org/lapack/lawnspdf/lawn18.pdf
}


MINI_BENCHMARK_REGISTER_SIZES((BM_ORMQR<float, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORMQR<double, Backend::NETLIB>), SquareBatchSizesNetlib);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORMQR<float, Backend::CUDA>), SquareBatchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_ORMQR<double, Backend::CUDA>), SquareBatchSizes);

MINI_BENCHMARK_MAIN();
