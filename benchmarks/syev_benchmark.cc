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
                b->Args({n, batch, nb, fuse});
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
            b->Args({n, batch, 16, 0});
        }
    }
}

} // namespace

// Symmetric eigenvalue decomposition benchmark
template <typename T, Backend B>
static void BM_SYEV(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int sytrd_block_size = static_cast<int>(state.range(2));
    const int fuse_panel_update = static_cast<int>(state.range(3));

    ::setenv("BATCHLAS_SYTRD_BLOCK_SIZE", std::to_string(sytrd_block_size).c_str(), 1);
    ::setenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE", fuse_panel_update ? "1" : "0", 1);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    size_t ws_size = syev_buffer_size<B>(*q, A.view(), W.to_span(),
                                         JobType::EigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::EigenVectors,
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
