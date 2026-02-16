#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <algorithm>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void SytrdSy2sbBenchSizes(Benchmark* b) {
    for (int n : {128, 256, 512, 1024, 2048}) {
        for (int bs : {1, 8, 32, 64}) {
            for (int kd : {8, 16, 32, 64}) {
                if (kd >= n) continue;
                // uplo: 0=Lower, 1=Upper (sy2sb currently implements Lower only)
                b->Args({n, bs, kd, 0});
            }
        }
    }
}

template <typename Benchmark>
inline void SytrdSy2sbBenchSizesNetlib(Benchmark* b) {
    // Smaller sizes for CPU testing to keep runtimes reasonable
    for (int n : {128, 256, 512}) {
        for (int bs : {1, 8, 16}) {
            for (int kd : {8, 16, 32}) {
                if (kd >= n) continue;
                b->Args({n, bs, kd, 0});
            }
        }
    }
}

} // namespace

template <typename T, Backend B>
static void BM_SYTRD_SY2SB(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int kd = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));
    const Uplo uplo = (uplo_i == 0) ? Uplo::Lower : Uplo::Upper;

    auto A0 = Matrix<T>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2026);
    auto A = Matrix<T>::Zeros(n, n, batch);
    auto AB = Matrix<T, MatrixFormat::Dense>::Zeros(kd + 1, n, batch);
    auto tau = Vector<T>::zeros(std::max<int>(0, static_cast<int>(n) - kd), batch);

    const std::string device_str = (B == Backend::NETLIB) ? "cpu" : "gpu";
    const bool in_order = true;
    auto q = std::make_shared<Queue>(device_str, /*in_order=*/in_order);

    const size_t ws_bytes = sytrd_sy2sb_buffer_size<B, T>(*q,
                                                         A.view(),
                                                         AB.view(),
                                                         VectorView<T>(tau),
                                                         uplo,
                                                         kd);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(A0),
        AB,
        tau,
        uplo,
        kd,
        ws,
        [](Queue& q, auto&&... xs) {
            sytrd_sy2sb<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });

    // Estimate FLOPs: dense-to-band reduction involves QR + orthogonal transforms
    // Approximate: 4/3 * n^2 * kd for the upper-left panel work + trailing matrix updates
    const double flops = (4.0 / 3.0) * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(kd);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / Batch", (1.0 / static_cast<double>(batch)) * 1e6, minibench::Reciprocal);
}

BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_SYTRD_SY2SB, SytrdSy2sbBenchSizes);

MINI_BENCHMARK_MAIN();
