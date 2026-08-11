#include <batchlas/util/minibench.hh>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>

#include "bench_utils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

// Head-to-head grid for the Jacobi CTA eigensolver against the
// tridiagonalization-based syev_cta on identical inputs.
template <typename Benchmark>
inline void SyevJacobiCtaBenchSizes(Benchmark* b) {
    for (int n : {4, 8, 16, 32}) {
        for (int bs : {64, 256, 1024, 4096}) {
            for (int jobz : {0, 1}) {
                for (int wg_mult : {1, 2, 4}) {
                    b->Args({n, bs, jobz, wg_mult});
                }
            }
        }
    }
}

inline JobType parse_jobz(int v) {
    return (v == 0) ? JobType::NoEigenVectors : JobType::EigenVectors;
}

} // namespace

template <typename T, Backend B>
static void BM_SYEV_JACOBI_CTA(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const JobType jobz = parse_jobz(static_cast<int>(state.range(2)));
    const size_t wg_mult = state.range(3) > 0 ? state.range(3) : 1;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    JacobiParams<T> params;
    params.cta_wg_size_multiplier = wg_mult;

    UnifiedVector<std::byte> workspace(0);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    params,
                    [](Queue& q, auto&&... xs) {
                        syev_jacobi_cta(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// Same grid, tridiagonalization-based path, for a direct comparison.
template <typename T, Backend B>
static void BM_SYEV_CTA_TRIDIAG_REF(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const JobType jobz = parse_jobz(static_cast<int>(state.range(2)));
    const size_t wg_mult = state.range(3) > 0 ? state.range(3) : 1;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    SteqrParams<T> params;
    params.cta_wg_size_multiplier = wg_mult;

    const size_t ws_size = syev_cta_buffer_size(*q, A.view(), jobz, params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    params,
                    wg_mult,
                    [](Queue& q, auto&&... xs) {
                        syev_cta(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_JACOBI_CTA, SyevJacobiCtaBenchSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_CTA_TRIDIAG_REF, SyevJacobiCtaBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_JACOBI_CTA, SyevJacobiCtaBenchSizes);
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_CTA_TRIDIAG_REF, SyevJacobiCtaBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
