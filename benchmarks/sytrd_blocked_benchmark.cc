#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SytrdBlockedBenchSizes(Benchmark* b) {
    for (int n : {64, 128, 256, 512, 1024}) {
        for (int bs : {1, 8, 32, 64}) {
            for (int nb : {8, 16, 32, 64}) {
                // uplo: 0=Lower, 1=Upper (blocked currently implements Lower only)
                b->Args({n, bs, nb, 0});
            }
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

    auto q = std::make_shared<Queue>("gpu", /*in_order=*/true);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, T>(*q,
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
            sytrd_blocked<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

// Register at static-init time
BATCHLAS_BENCH_CUDA(BM_SYTRD_BLOCKED, SytrdBlockedBenchSizes);

MINI_BENCHMARK_MAIN();
