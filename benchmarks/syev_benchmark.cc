#include <util/minibench.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"
using namespace batchlas;

// Symmetric eigenvalue decomposition benchmark
template <typename T, Backend B>
static void BM_SYEV(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Random(n, n, true, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    size_t ws_size = syev_buffer_size<B>(*q, A.view(), W.to_span(),
                                         JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel([=]() {
        syev<B>(*q, A, W.to_span(),
                JobType::EigenVectors, Uplo::Lower, workspace.to_span());
    });
    state.SetBatchEndWait(q);
    double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}


BATCHLAS_REGISTER_BENCHMARK(BM_SYEV, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
