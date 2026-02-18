#include <util/minibench.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"
using namespace batchlas;

// Symmetric eigenvalue decomposition benchmark
template <typename T, Backend B>
static void BM_SYEV(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    size_t ws_size = syev_buffer_size<B>(*q, A.view(), W.to_span(),
                                         JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::EigenVectors,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}


BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYEV, SteqrBenchSizes);

MINI_BENCHMARK_MAIN();
