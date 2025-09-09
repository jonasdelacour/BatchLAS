#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

using namespace batchlas;

// Single ORGQR benchmark
template <typename T, Backend B>
static void BM_ORGQR(minibench::State& state) {
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

    size_t org_ws = orgqr_buffer_size<B>(queue, A.view(), tau.to_span());
    UnifiedVector<std::byte> ws(org_ws);
    state.SetKernel([&]{
        orgqr<B>(queue, A.view(), tau.to_span(), ws.to_span());
    });
    state.SetBatchEnd([&]{ queue.wait(); });
    //FLOP calculation for ORGQR derived from: https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/lse/functn_orgqr.htm
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * (2 * m * n * n - 2.0 / 3.0 * n * n * n)), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);

}


BATCHLAS_REGISTER_BENCHMARK(BM_ORGQR, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
