#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

template <typename T, Backend B>
static void BM_SYMM(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, false, batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);
    auto C = Matrix<T>::Random(n, n, false, batch);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    T(1),
                    Side::Left,
                    Uplo::Lower,
                    [](Queue& q, auto&&... xs) {
                        symm<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_SYMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();