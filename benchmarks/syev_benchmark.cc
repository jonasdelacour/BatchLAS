#include <util/minibench.hh>
#include <blas/functions.hh>
#include "bench_utils.hh"

#include <cstdlib>
#include <string>
using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SyevBenchSizes(Benchmark* b) {
    auto add_cases = [&](int n, int batch, std::initializer_list<int> nbs) {
        for (int nb : nbs) {
            for (int fuse : {0, 1}) {
                // arg4 = jobz, 1 = EigenVectors. Passed explicitly so the registered sizes
                // keep the behaviour they had before jobz became an argument.
                b->Args({n, batch, nb, fuse, 1});
            }
        }
    };
    add_cases(64, 4096, {8, 12, 16, 24});
    add_cases(128, 2048, {8, 12, 16, 24, 32});
    add_cases(256, 1024, {8, 12, 16, 24, 32, 48, 64});
    add_cases(512, 512, {8, 12, 16, 24, 32, 48, 64});
}

template <typename Benchmark>
inline void SyevBenchSizesNetlib(Benchmark* b) {
    for (int n : {64, 128, 256}) {
        for (int batch : {1, 10}) {
            b->Args({n, batch, 16, 0, 1});
        }
    }
}

// arg4: 0 = NoEigenVectors, 1 = EigenVectors. Same convention as syev_cta_benchmark and
// syev_blocked_benchmark. state.range() returns 0 for an absent argument, so a caller passing
// only four args gets eigenvalues-only; every registered size passes jobz explicitly.
inline JobType parse_jobz(int v) {
    return (v == 0) ? JobType::NoEigenVectors : JobType::EigenVectors;
}

} // namespace

// Symmetric eigenvalue decomposition benchmark
template <typename T, Backend B>
static void BM_SYEV(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int sytrd_block_size = static_cast<int>(state.range(2));
    const int fuse_panel_update = static_cast<int>(state.range(3));
    const JobType jobz = parse_jobz(static_cast<int>(state.range(4)));

    ::setenv("BATCHLAS_SYTRD_BLOCK_SIZE", std::to_string(sytrd_block_size).c_str(), 1);
    ::setenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE", fuse_panel_update ? "1" : "0", 1);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    size_t ws_size = syev_buffer_size<B>(*q, A.view(), W.to_span(),
                                         jobz, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    jobz,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });
    double flops = 4.0 / 3.0 * static_cast<double>(n) * n * n;
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / batch) * 1e6, minibench::Reciprocal);
}


BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYEV, SyevBenchSizes);

MINI_BENCHMARK_MAIN();
