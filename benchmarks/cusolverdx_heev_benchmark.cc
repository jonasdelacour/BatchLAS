#include <util/minibench.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"

#include "../src/backends/cusolverdx.hh"

using namespace batchlas;

template <typename T, Backend B>
static void BM_CUSOLVERDX_HEEV(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto q = std::make_shared<Queue>("gpu");
    auto A = Matrix<T>::Random(static_cast<int>(n), static_cast<int>(n), true, static_cast<int>(batch));
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = backend::cusolverdx::heev_buffer_size<T>(
        *q,
        A.view(),
        W.to_span(),
        JobType::EigenVectors,
        Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::EigenVectors,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q_local, auto&&... xs) {
                        backend::cusolverdx::heev<T>(q_local, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
#else
    static_cast<void>(state);
#endif
#if !BATCHLAS_HAS_CUDA_BACKEND
    static_cast<void>(B);
#endif
}

#if BATCHLAS_HAS_CUDA_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HEEV<float, batchlas::Backend::CUDA>), SteqrBenchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HEEV<double, batchlas::Backend::CUDA>), SteqrBenchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HEEV<std::complex<float>, batchlas::Backend::CUDA>), SteqrBenchSizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_CUSOLVERDX_HEEV<std::complex<double>, batchlas::Backend::CUDA>), SteqrBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
