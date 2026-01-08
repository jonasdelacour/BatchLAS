#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// Batched GEMM benchmark
template <typename T, Backend B>
static void BM_GEMM(minibench::State& state) {
    const size_t m = state.range(0);
    const size_t n = state.range(1);
    const size_t k = state.range(2);
    const size_t batch = state.range(3);

    auto A = Matrix<T>::Random(m, k, false, batch);
    auto Bm = Matrix<T>::Random(k, n, false, batch);
    auto C = Matrix<T>::Random(m, n, false, batch);
    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    T(1),
                    Transpose::NoTrans,
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        gemm<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * m * n * k), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}



// Register size/batch combinations at static‑init time using macro

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_GEMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
