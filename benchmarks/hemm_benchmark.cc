#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

// There is no batched vendor HEMM to compare against -- cuBLAS ships only the
// single-matrix cublasChemm/cublasZhemm. The per-batch loop over those is the
// route this same binary takes with BATCHLAS_EXPAND_MAX_BYTES=0, so running it
// twice measures both implementations without a second benchmark, and
// gemm_benchmark at the same complex type and shape gives the equal-work floor.
template <typename T, Backend B>
static void BM_HEMM(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(3);

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, true, batch);
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
                        hemm(q, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * 2.0 * n * n * n), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_COMPLEX_TYPES(BM_HEMM, CubeBatchSizes);

MINI_BENCHMARK_MAIN();
