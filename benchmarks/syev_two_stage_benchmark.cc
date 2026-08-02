// Two-stage (sy2sb -> sb2st_hh -> stedc -> Q2 -> Q1) versus the blocked
// single-stage baseline (sytrd_blocked -> stedc -> ormqr_blocked), with
// eigenvectors. The band width kd is swept as range(2) so the two-stage cost
// model (stage-1 and Q2 favour large kd, stage-2 favours small) can be measured
// directly rather than inferred.

#include <util/minibench.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include "bench_utils.hh"

#include <cstdlib>
#include <string>

using namespace batchlas;

namespace {

inline void Sizes(minibench::Benchmark* b) {
    for (int kd : {16, 32, 48, 64, 96}) {
        b->Args({128, 2048, kd});
        b->Args({256, 1024, kd});
        b->Args({512, 512, kd});
        b->Args({1024, 128, kd});
        b->Args({2048, 32, kd});
    }
}

// The baseline ignores kd; one entry per size keeps the output readable.
inline void BaselineSizes(minibench::Benchmark* b) {
    b->Args({64, 4096, 0});
    b->Args({128, 2048, 0});
    b->Args({256, 1024, 0});
    b->Args({512, 512, 0});
    b->Args({1024, 128, 0});
    b->Args({2048, 32, 0});
}

} // namespace

template <typename T, Backend B>
static void BM_SYEV_TWO_STAGE(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int kd = static_cast<int>(state.range(2));

    ::setenv("BATCHLAS_SYEV_TWO_STAGE_KD", std::to_string(kd).c_str(), 1);

    auto q = std::make_shared<Queue>("gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = syev_two_stage_buffer_size<B>(
        *q, A.view(), JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q, bench::pristine(A), std::move(W), JobType::EigenVectors,
                    Uplo::Lower, std::move(workspace),
                    [](Queue& qq, auto&&... xs) {
                        syev_two_stage<B, T>(qq, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (ms)", 1.0, minibench::Reciprocal);
}

template <typename T, Backend B>
static void BM_SYEV_BLOCKED_BASELINE(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);

    auto q = std::make_shared<Queue>("gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size = syev_blocked_buffer_size<B>(
        *q, A.view(), JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q, bench::pristine(A), std::move(W), JobType::EigenVectors,
                    Uplo::Lower, std::move(workspace),
                    [](Queue& qq, auto&&... xs) {
                        syev_blocked<B, T>(qq, std::forward<decltype(xs)>(xs)...);
                    });
    state.SetMetric("Time (ms)", 1.0, minibench::Reciprocal);
}

#if BATCHLAS_HAS_CUDA_BACKEND
MINI_BENCHMARK_REGISTER_SIZES((BM_SYEV_TWO_STAGE<float, batchlas::Backend::CUDA>), Sizes);
MINI_BENCHMARK_REGISTER_SIZES((BM_SYEV_BLOCKED_BASELINE<float, batchlas::Backend::CUDA>), BaselineSizes);
#endif

MINI_BENCHMARK_MAIN();
