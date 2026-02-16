#include <util/minibench.hh>
#include <blas/linalg.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

using namespace batchlas;

namespace {

template <typename Benchmark>
inline void BandReductionBenchSizes(Benchmark* b) {
    // n, kd, batch, nb_target
    for (int n : {64, 128, 256, 512}) {
        for (int kd : {8, 16, 32}) {
            if (kd >= n) continue; // bandwidth must be less than matrix size
            for (int batch : {1, 8, 32, 64}) {
                for (int nb : {8, 16, 32}) {
                    if (nb > kd) continue; // block size shouldn't exceed bandwidth
                    b->Args({n, kd, batch, nb});
                }
            }
        }
    }
}

template <typename Benchmark>
inline void BandReductionBenchSizesNetlib(Benchmark* b) {
    // Smaller sizes for CPU backend
    for (int n : {64, 128, 256}) {
        for (int kd : {8, 16}) {
            if (kd >= n) continue;
            for (int batch : {1, 4, 8}) {
                for (int nb : {8, 16}) {
                    if (nb > kd) continue;
                    b->Args({n, kd, batch, nb});
                }
            }
        }
    }
}

template <typename Benchmark>
inline void BandReductionSingleStepBenchSizes(Benchmark* b) {
    // n, kd, batch, nb_target, max_steps
    for (int n : {64, 128, 256, 512}) {
        for (int kd : {8, 16, 32}) {
            if (kd >= n) continue;
            for (int batch : {1, 8, 32, 64}) {
                for (int nb : {8, 16, 32}) {
                    if (nb > kd) continue;
                    for (int max_steps : {1, 5, 10}) {
                        b->Args({n, kd, batch, nb, max_steps});
                    }
                }
            }
        }
    }
}

template <typename Benchmark>
inline void BandReductionSingleStepBenchSizesNetlib(Benchmark* b) {
    // Smaller sizes for CPU backend
    for (int n : {64, 128, 256}) {
        for (int kd : {8, 16}) {
            if (kd >= n) continue;
            for (int batch : {1, 4, 8}) {
                for (int nb : {8, 16}) {
                    if (nb > kd) continue;
                    for (int max_steps : {1, 5}) {
                        b->Args({n, kd, batch, nb, max_steps});
                    }
                }
            }
        }
    }
}

} // namespace

// Full band reduction benchmark (reduces to tridiagonal form)
template <typename T, Backend B>
static void BM_BAND_REDUCTION(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t kd = state.range(1);
    const size_t batch = state.range(2);
    const int nb_target = static_cast<int>(state.range(3));
    const Uplo uplo = Uplo::Lower;

    // Rough flop model: band reduction is approximately O(n * kd^2) per sweep,
    // with multiple sweeps to reduce from kd to 1.
    // Using kd sweeps as approximation.
    const double total_flops = double(n) * double(kd) * double(kd) * double(kd) * double(batch);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu", /*in_order=*/true);

    // Create input band matrix (kd+1 rows, n columns, in lower-band format)
    auto ab_in = Matrix<T>::Random(kd + 1, n, /*hermitian=*/false, batch, /*seed=*/2027);
    
    using Real = typename base_type<T>::type;
    auto d = Vector<Real>::zeros(n, batch);
    auto e = Vector<Real>::zeros(n - 1, batch);
    auto tau = Vector<T>::zeros(n - 1, batch);

    SytrdBandReductionParams params;
    params.block_size = nb_target;
    params.kd_work = std::min(static_cast<int>(kd * 3 / 2), static_cast<int>(n - 1));
    params.max_sweeps = -1; // run until completion
    params.d = 1; // reduce by 1 per sweep

    const size_t ws_bytes = sytrd_band_reduction_buffer_size<B, T>(*q,
                                                                   ab_in.view(),
                                                                   VectorView<Real>(d),
                                                                   VectorView<Real>(e),
                                                                   VectorView<T>(tau),
                                                                   uplo,
                                                                   kd,
                                                                   params);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(ab_in),
        d, e, tau, uplo, static_cast<int32_t>(kd), ws, params,
        [](Queue& q, auto&&... xs) {
            sytrd_band_reduction<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

// Single-step band reduction benchmark (partial reduction with step limit)
template <typename T, Backend B>
static void BM_BAND_REDUCTION_SINGLE_STEP(minibench::State& state) {
    const size_t n = state.range(0);
    const size_t kd = state.range(1);
    const size_t batch = state.range(2);
    const int nb_target = static_cast<int>(state.range(3));
    const int max_steps = static_cast<int>(state.range(4));
    const Uplo uplo = Uplo::Lower;

    // Flop model: proportional to number of steps
    const double total_flops = double(n) * double(kd) * double(kd) * double(max_steps) * double(batch);

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu", /*in_order=*/true);

    // Create input band matrix
    auto ab_in = Matrix<T>::Random(kd + 1, n, /*hermitian=*/false, batch, /*seed=*/2028);
    
    // Working bandwidth (allows for fill-in during reduction)
    const int kd_work = std::min(static_cast<int>(kd * 3 / 2), static_cast<int>(n - 1));
    auto abw_out = Matrix<T>::Zeros(kd_work + 1, n, batch);

    SytrdBandReductionParams params;
    params.block_size = nb_target;
    params.kd_work = kd_work;
    params.max_steps = max_steps;
    params.d = 1;

    const size_t ws_bytes = sytrd_band_reduction_single_step_buffer_size<B, T>(*q,
                                                                               ab_in.view(),
                                                                               abw_out.view(),
                                                                               uplo,
                                                                               kd,
                                                                               params);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(ab_in),
        abw_out, uplo, static_cast<int32_t>(kd), ws, params,
        [](Queue& q, auto&&... xs) {
            sytrd_band_reduction_single_step<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });
    state.SetMetric("GFLOPS", total_flops * 1e-9, minibench::Rate);
    state.SetMetric("T(µs)/Batch", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
}

// Register benchmarks
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_BAND_REDUCTION, BandReductionBenchSizes);
BATCHLAS_REGISTER_BENCHMARK_ALL_TYPES(BM_BAND_REDUCTION_SINGLE_STEP, BandReductionSingleStepBenchSizes);

MINI_BENCHMARK_MAIN();
