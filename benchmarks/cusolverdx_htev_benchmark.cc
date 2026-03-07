#include <util/minibench.hh>
#include <blas/linalg.hh>
#include "bench_utils.hh"

#include "../src/backends/cusolverdx.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_CUSOLVERDX_HTEV(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto q = std::make_shared<Queue>("gpu");
    auto diags = Vector<T>::random(static_cast<int>(n), static_cast<int>(batch));
    auto off_diags = Vector<T>::random(static_cast<int>(n - 1), static_cast<int>(batch));
    auto eigvals = Vector<T>::zeros(static_cast<int>(n), static_cast<int>(batch));
    auto eigvects = Matrix<T>::Identity(static_cast<int>(n), static_cast<int>(batch));

    const size_t ws_size = backend::cusolverdx::htev_buffer_size<T>(
        *q,
        static_cast<VectorView<T>>(diags),
        static_cast<VectorView<T>>(off_diags),
        JobType::EigenVectors,
        Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(diags),
                    bench::pristine(off_diags),
                    std::move(eigvals),
                    JobType::EigenVectors,
                    bench::pristine(eigvects),
                    std::move(workspace),
                    Uplo::Lower,
                    [](Queue& q_local, auto&&... xs) {
                        backend::cusolverdx::htev<T>(q_local, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
#else
    static_cast<void>(state);
#endif
#if !BATCHLAS_HAS_CUDA_BACKEND
    static_cast<void>(B);
#endif
}

#if BATCHLAS_HAS_CUDA_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HTEV<float, batchlas::Backend::CUDA>), SteqrBenchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HTEV<double, batchlas::Backend::CUDA>), SteqrBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
