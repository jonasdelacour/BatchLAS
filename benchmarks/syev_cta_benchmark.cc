#include <util/minibench.hh>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>

#include "bench_utils.hh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SyevCtaBenchSizes(Benchmark* b) {
    for (int n : {8, 16, 32}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
            for (int jobz : {0, 1}) {
                for (int uplo : {0, 1}) {
                    for (int wg_mult : {1, 2, 4, 8}) {
                        b->Args({n, bs, jobz, uplo, wg_mult});
                    }
                }
            }
        }
    }
}

// Keep the same argument grid for NETLIB so outputs are directly comparable.
template <typename Benchmark>
inline void SyevCtaBenchSizesNetlib(Benchmark* b) {
    SyevCtaBenchSizes(b);
}

inline JobType parse_jobz(int v) {
    return (v == 0) ? JobType::NoEigenVectors : JobType::EigenVectors;
}

inline Uplo parse_uplo(int v) {
    return (v == 0) ? Uplo::Lower : Uplo::Upper;
}

} // namespace

// CTA SYEV-like benchmark (CUDA/ROCm/MKL GPU backends)
template <typename T, Backend B>
static void BM_SYEV_CTA(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int jobz_i = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));
    const size_t wg_mult = state.range(4) > 0 ? state.range(4) : 1;

    const JobType jobz = parse_jobz(jobz_i);
    const Uplo uplo = parse_uplo(uplo_i);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    SteqrParams<T> params;
    params.cta_wg_size_multiplier = wg_mult;

    const size_t ws_size = syev_cta_buffer_size<B, T>(*q, A.view(), jobz, params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    uplo,
                    std::move(workspace),
                    params,
                    wg_mult,
                    [](Queue& q, auto&&... xs) {
                        syev_cta<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// Reference benchmark using LAPACKE SYEV (NETLIB backend)
template <typename T, Backend B>
static void BM_SYEV_NETLIB_REF(minibench::State& state) {
    static_assert(B == Backend::NETLIB, "BM_SYEV_NETLIB_REF is only intended for NETLIB backend.");

    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int jobz_i = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));

    const JobType jobz = parse_jobz(jobz_i);
    const Uplo uplo = parse_uplo(uplo_i);

    auto q = std::make_shared<Queue>("cpu");
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = syev_buffer_size<B>(*q, A.view(), W.to_span(), jobz, uplo);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    uplo,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_CTA, SyevCtaBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_CTA, SyevCtaBenchSizes);
#endif

#if BATCHLAS_HAS_MKL_BACKEND
BATCHLAS_BENCH_MKL_ALL_TYPES(BM_SYEV_CTA, SyevCtaBenchSizes);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BATCHLAS_BENCH_NETLIB_ALL_TYPES(BM_SYEV_NETLIB_REF, SyevCtaBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
