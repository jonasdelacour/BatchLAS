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
inline void SyevBlockedBenchSizes(Benchmark* b) {
    // Args: n, batch, jobz
    for (int n : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128}) {
            for (int jobz : {0, 1}) {
                b->Args({n, bs, jobz});
            }
        }
    }
}

template <typename Benchmark>
inline void SyevBlockedBenchSizesNetlib(Benchmark* b) {
    // Reduced grid for CPU runs.
    for (int n : {32, 64, 128, 256}) {
        for (int bs : {1, 10, 100}) {
            for (int jobz : {0, 1}) {
                b->Args({n, bs, jobz});
            }
        }
    }
}

inline JobType parse_jobz(int v) {
    return (v == 0) ? JobType::NoEigenVectors : JobType::EigenVectors;
}

} // namespace

// Blocked SYEV-like benchmark (sytrd_blocked + stedc + ormqr_blocked).
template <typename T, Backend B>
static void BM_SYEV_BLOCKED(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int jobz_i = static_cast<int>(state.range(2));

    const JobType jobz = parse_jobz(jobz_i);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = syev_blocked_buffer_size<B, T>(*q,
                                                         A.view(),
                                                         jobz,
                                                         Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev_blocked<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * double(n) * double(n);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_BLOCKED, SyevBlockedBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_BLOCKED, SyevBlockedBenchSizes);
#endif

#if BATCHLAS_HAS_MKL_BACKEND
BATCHLAS_BENCH_MKL_ALL_TYPES(BM_SYEV_BLOCKED, SyevBlockedBenchSizes);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BATCHLAS_BENCH_NETLIB_ALL_TYPES(BM_SYEV_BLOCKED, SyevBlockedBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
