#include <batchlas/util/minibench.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SytrdBlockedBenchSizes(Benchmark* b) {
    auto ns = std::array{64, 128, 256, 512};
    auto bs = std::array{8192, 4096, 2048, 1024};
    auto nb = std::array{8,12,16,24,32};
    for (size_t i = 0; i < ns.size(); ++i) {
        for (int j = 0; j < nb.size(); ++j) {
            b->Args({ns[i], bs[i], nb[j], 0}); // lower
        }
    }
}

} // namespace

// Batched SYTRD-blocked benchmark
template <typename T, Backend B>
static void BM_SYTRD_BLOCKED(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int nb = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));
    const Uplo uplo = (uplo_i == 0) ? Uplo::Lower : Uplo::Upper;

    // Rough flop model for blocked SYTRD is still O(n^3).
    const double total_flops = (4.0 / 3.0) * double(n) * double(n) * double(n) * double(batch);

    auto A0 = Matrix<T>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2026);
    auto A = Matrix<T>::Zeros(n, n, batch);
    auto d = Vector<T>::zeros(n, batch);
    auto e = Vector<T>::zeros(n - 1, batch);
    auto tau = Vector<T>::zeros(n - 1, batch);

    auto q = std::make_shared<Queue>(Device("gpu"), B, /*in_order=*/true);

    const size_t ws_bytes = sytrd_blocked_buffer_size(*q,
                                                           A.view(),
                                                           VectorView<T>(d),
                                                           VectorView<T>(e),
                                                           VectorView<T>(tau),
                                                           uplo,
                                                           nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(A0), //sytrd_blocked mutates A so if it is not kept pristine between runs the speed results will change between runs.
        d,e,tau,uplo,ws,nb,
        [](Queue& q, auto&&... xs) {
            sytrd_blocked(q, std::forward<decltype(xs)>(xs)...);
        });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

// Register at static-init time
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYTRD_BLOCKED, SytrdBlockedBenchSizes);

MINI_BENCHMARK_MAIN();
