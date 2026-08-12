#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// There is no batched vendor HERK to compare against -- cuBLAS ships only the
// single-matrix cublasCherk/cublasZherk. The per-batch loop over those is the
// route this same binary takes with BATCHLAS_EXPAND_ROUTE=loop, so running it
// twice measures both implementations without a second benchmark, and
// gemm_benchmark at the same complex type and shape gives the equal-work floor.
//
// The GFLOP count is the rank-k update's own, n * (n + 1) * k, matching
// syrk_benchmark so the two families read on the same scale. The GEMM route
// does twice that arithmetic -- it computes both triangles of the product and
// keeps one -- which is exactly what the rate is meant to expose.
template <typename T, Backend B>
static void BM_HERK(minibench::State& state) {
    using real_t = typename base_type<T>::type;
    const size_t n = state.range(0);
    const size_t k = state.range(1);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, k, false, batch);
    auto C = Matrix<T>::Random(n, n, false, batch);

    state.SetKernel(q,
                    std::move(A),
                    bench::pristine(C),
                    real_t(1),
                    real_t(1),
                    Uplo::Lower,
                    Transpose::NoTrans,
                    [](Queue& q, auto&&... xs) {
                        herk(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS",
                    static_cast<double>(batch) * (1e-9 * static_cast<double>(n) * static_cast<double>(n + 1) * static_cast<double>(k)),
                    minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_COMPLEX_TYPES(BM_HERK, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
