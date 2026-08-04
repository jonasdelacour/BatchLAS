#include <util/minibench.hh>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>

#include "bench_utils.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

using namespace batchlas;

namespace {

// Head-to-head grid: the fused single-kernel SYEV against the three-kernel
// sytrd_cta -> steqr_cta -> ormqx_cta pipeline on identical inputs.
//
// The interesting axis is n: fusion buys the elimination of the intermediate
// global round trips and of two launch latencies, and costs local memory (the
// reduced tile and the rotation accumulator are live at once), so the win should
// be largest at small n and shrink as n approaches the partition width of 32.
template <typename Benchmark>
inline void SyevCtaFusedBenchSizes(Benchmark* b) {
    for (int n : {4, 8, 16, 32}) {
        for (int bs : {64, 256, 1024, 4096, 16384}) {
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
static void BM_SYEV_CTA_FUSED(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const JobType jobz = parse_jobz(static_cast<int>(state.range(2)));
    const size_t wg_mult = state.range(3) > 0 ? state.range(3) : 1;

    auto q = std::make_shared<Queue>(Device(B == Backend::NETLIB ? "cpu" : "gpu"), B);
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    SteqrParams<T> params;
    params.cta_wg_size_multiplier = wg_mult;

    // Partition-resident: no global workspace at all.
    UnifiedVector<std::byte> workspace(0);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    params,
                    wg_mult,
                    [](Queue& q, auto&&... xs) {
                        syev_cta_fused(q, std::forward<decltype(xs)>(xs)...);
                    });

    const double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}

// Same grid, three-kernel pipeline, for a direct comparison.
template <typename T, Backend B>
static void BM_SYEV_CTA_PIPELINED(minibench::State& state) {
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
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_CTA_FUSED, SyevCtaFusedBenchSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_CTA_PIPELINED, SyevCtaFusedBenchSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_CTA_FUSED, SyevCtaFusedBenchSizes);
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_CTA_PIPELINED, SyevCtaFusedBenchSizes);
#endif

MINI_BENCHMARK_MAIN();
