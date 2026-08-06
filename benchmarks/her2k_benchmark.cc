#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// As herk_benchmark: no batched vendor HER2K exists, so the per-batch loop over
// cublasCher2k/cublasZher2k is reached from this same binary with
// BATCHLAS_EXPAND_ROUTE=loop, and gemm_benchmark gives the equal-work floor.
//
// HER2K's two terms are conjugate transposes of one another, so the GEMM route
// computes only one of them and folds it in twice. That halves the arithmetic
// against a literal reading of the formula, and is why this route can beat the
// vendor by more than herk's does.
template <typename T, Backend B>
static void BM_HER2K(minibench::State& state) {
    using real_t = typename base_type<T>::type;
    const size_t n = state.range(0);
    const size_t k = state.range(1);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, k, false, batch);
    auto Bm = Matrix<T>::Random(n, k, false, batch);
    auto C = Matrix<T>::Random(n, n, false, batch);

    state.SetKernel(q,
                    std::move(A),
                    std::move(Bm),
                    bench::pristine(C),
                    T(1),
                    real_t(1),
                    Uplo::Lower,
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        her2k(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS",
                    static_cast<double>(batch) * (2e-9 * static_cast<double>(n) * static_cast<double>(n + 1) * static_cast<double>(k)),
                    minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_COMPLEX_TYPES(BM_HER2K, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
