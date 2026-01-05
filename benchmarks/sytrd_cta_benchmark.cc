#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/extensions.hh>
#include "bench_utils.hh"
#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SytrdCtaBenchSizes(Benchmark* b) {
    for (int n : {8, 16, 32}) {
        for (int bs : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
            for (int wg_mult : {1, 2, 4, 8}) {
                for (int uplo : {0, 1}) {
                    b->Args({n, bs, wg_mult, uplo});
                }
            }
        }
    }
}

} // namespace

// Batched SYTRD-CTA benchmark
template <typename T, Backend B>
static void BM_SYTRD_CTA(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const size_t wg_mult = state.range(2) > 0 ? state.range(2) : 1;
    const int uplo_i = state.range(3);
    const Uplo uplo = (uplo_i == 0) ? Uplo::Lower : Uplo::Upper;

    // Approx flop model: symmetric tridiagonal reduction (unblocked) is O(n^3).
    // We use ~4/3 n^3 as a rough proxy.
    const double total_flops = (4.0 / 3.0) * double(n) * double(n) * double(n) * double(batch);

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2025);
    auto d = Vector<T>::zeros(n, batch);
    auto e = Vector<T>::zeros(n - 1, batch);
    auto tau = Vector<T>::zeros(n - 1, batch);

    auto q = std::make_shared<Queue>("gpu");
    UnifiedVector<std::byte> ws_dummy(1, std::byte{0});

    state.SetKernel([=]() {
        sytrd_cta<B>(*q,
                     A.view(),
                     VectorView<T>(d),
                     VectorView<T>(e),
                     VectorView<T>(tau),
                     uplo,
                     ws_dummy.to_span(),
                     wg_mult);
    });
    state.SetBatchEndWait(q);
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

// Register size/batch combinations at static-init time
BATCHLAS_BENCH_CUDA(BM_SYTRD_CTA, SytrdCtaBenchSizes);

MINI_BENCHMARK_MAIN();
