#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <batchlas/backend_config.h>
#include "bench_utils.hh"
using namespace batchlas;

// Single TRSM benchmark
template <typename T, Backend B>
static void BM_TRSM(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto A = Matrix<T>::Triangular(n, Uplo::Lower, T(1), T(0.5), batch);
    auto Bm = Matrix<T>::Random(n, n, false, batch);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(Bm),
                    Side::Left,
                    Uplo::Lower,
                    Transpose::NoTrans,
                    Diag::NonUnit,
                    T(1),
                    [](Queue& q, auto&&... xs) {
                        trsm<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_TRSM, SquareBatchSizes);

MINI_BENCHMARK_MAIN();
