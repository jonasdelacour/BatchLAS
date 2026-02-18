#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Single GEMV benchmark
template <typename T, Backend B>
static void BM_GEMV(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t batch = state.range(2);

    auto A = Matrix<T>::Random(m, n, false, batch);
    auto x = Vector<T>::random(n, batch);
    auto y = Vector<T>::zeros(m, batch);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetKernel(q,
                    std::move(A),
                    std::move(x),
                    std::move(y),
                    T(1),
                    T(0),
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        gemv<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n),
                    minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6,
                    minibench::Reciprocal);
}



BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMV, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
