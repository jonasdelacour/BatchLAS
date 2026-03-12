#include <util/minibench.hh>
#include <blas/linalg.hh>

#include "bench_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_SYRK(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t k = state.range(1);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, k, false, batch);
    auto C = Matrix<T>::Random(n, n, false, batch);

    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(C),
                    T(1),
                    T(1),
                    Uplo::Lower,
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        syrk<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS",
                    static_cast<double>(batch) * (1e-9 * static_cast<double>(n) * static_cast<double>(n + 1) * static_cast<double>(k)),
                    minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK(BM_SYRK, CubeBatchSizes);

MINI_BENCHMARK_MAIN();