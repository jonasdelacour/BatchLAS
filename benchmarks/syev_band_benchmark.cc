// Head-to-head benchmark for the band-reduction eigensolver.
//
// Both pipelines are registered in the same binary so a single run compares
// them under identical machine conditions:
//   BM_SYEV_REF_BLOCKED  -- syev_blocked   (sytrd_blocked + stedc)      [baseline]
//   BM_SYEV_BAND      -- syev_band      (sy2sb + BANDR1 + stedc)
//   BM_SYEV_KDSWEEP   -- syev_band with an explicit kd, for tuning sweeps
//
// All are eigenvalues-only, which is the only mode syev_band supports.

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

// Shared grid so BAND and REF rows line up one-for-one.
template <typename Benchmark>
inline void SyevBandSizes(Benchmark* b) {
    // Args: n, batch
    for (int n : {128, 256, 512, 1024}) {
        for (int bs : {1, 32}) {
            b->Args({n, bs});
        }
    }
}

template <typename Benchmark>
inline void SyevBandSizesNetlib(Benchmark* b) {
    for (int n : {64, 128, 256}) {
        b->Args({n, 1});
    }
}

// Stage-2 schedule sweep: n, batch, kd, d.
template <typename Benchmark>
inline void SyevSchedSizes(Benchmark* b) {
    for (int n : {256, 512}) {
        for (int kd : {16, 32}) {
            for (int d : {1, 2, 4, 8, 16, 31}) {
                if (d > kd - 1) continue;
                b->Args({n, 1, kd, d});
            }
        }
    }
}

template <typename Benchmark>
inline void SyevSchedSizesNetlib(Benchmark* b) {
    for (int d : {1, 4, 15}) b->Args({128, 1, 16, d});
}

// kd sweep at a couple of representative shapes.
template <typename Benchmark>
inline void SyevBandKdSizes(Benchmark* b) {
    // Args: n, batch, kd
    for (int n : {256, 512, 1024}) {
        for (int bs : {1, 32}) {
            for (int kd : {8, 16, 24, 32, 48, 64, 96, 128}) {
                if (kd >= n) continue;
                b->Args({n, bs, kd});
            }
        }
    }
}

template <typename Benchmark>
inline void SyevBandKdSizesNetlib(Benchmark* b) {
    for (int n : {128, 256}) {
        for (int kd : {8, 16, 32, 64}) {
            b->Args({n, 1, kd});
        }
    }
}

inline void set_common_metrics(minibench::State& state, size_t n, size_t batch) {
    const double flops = 4.0 / 3.0 * static_cast<double>(n) * double(n) * double(n);
    state.SetMetric("GFLOPS", static_cast<double>(batch) * (1e-9 * flops), minibench::Rate);
    state.SetMetric("Time (µs) / matrix", (1.0 / static_cast<double>(batch)) * 1e6,
                    minibench::Reciprocal);
}

} // namespace

// Baseline: the blocked pipeline syev_band has to beat.
template <typename T, Backend B>
static void BM_SYEV_REF_BLOCKED(minibench::State& state) {
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size =
        syev_blocked_buffer_size<B, T>(*q, A.view(), JobType::NoEigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::NoEigenVectors,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev_blocked<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// The band-reduction pipeline with its auto-tuned kd.
template <typename T, Backend B>
static void BM_SYEV_BAND(minibench::State& state) {
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    const size_t ws_size =
        syev_band_buffer_size<B, T>(*q, A.view(), JobType::NoEigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::NoEigenVectors,
                    Uplo::Lower,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        syev_band<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// Same pipeline, explicit kd, for tuning the default.
template <typename T, Backend B>
static void BM_SYEV_KDSWEEP(minibench::State& state) {
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));
    const int32_t kd = static_cast<int32_t>(state.range(2));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    UnifiedVector<typename base_type<T>::type> W(n * batch);

    SyevBandParams params;
    params.kd = kd;

    StedcParams<typename base_type<T>::type> stedc_params;

    const size_t ws_size = syev_band_buffer_size<B, T>(*q, A.view(), JobType::NoEigenVectors,
                                                       Uplo::Lower, stedc_params, params);
    UnifiedVector<std::byte> workspace(ws_size);

    state.SetKernel(q,
                    bench::pristine(A),
                    std::move(W),
                    JobType::NoEigenVectors,
                    Uplo::Lower,
                    std::move(workspace),
                    stedc_params,
                    params,
                    [](Queue& q, auto&&... xs) {
                        syev_band<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// ---- per-stage attribution -------------------------------------------------
// Same shapes as BM_SYEV_BAND, but each runs one stage of the pipeline so the
// three costs can be read off directly instead of inferred.

// Stage 1 only: dense -> band.
template <typename T, Backend B>
static void BM_SYEV_STAGE1_SY2SB(minibench::State& state) {
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));
    const int32_t kd = (n <= 256) ? 16 : 32;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    Matrix<T> AB(kd + 1, n, batch);
    Vector<T> tau(std::max<int32_t>(0, static_cast<int32_t>(n) - kd), batch);

    UnifiedVector<std::byte> workspace(
        sytrd_sy2sb_buffer_size<B, T>(*q, A.view(), AB.view(), tau, Uplo::Lower, kd));

    state.SetKernel(q, bench::pristine(A), AB.view(), tau, Uplo::Lower, kd,
                    std::move(workspace),
                    [](Queue& q, auto&&... xs) {
                        sytrd_sy2sb<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// Stage 2 only: band -> tridiagonal via BANDR1.
template <typename T, Backend B>
static void BM_SYEV_STAGE2_BANDR1(minibench::State& state) {
    using Real = typename base_type<T>::type;
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));
    const int32_t kd = (n <= 256) ? 16 : 32;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    // Build a genuine band matrix by running stage 1 once, outside the timer.
    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    Matrix<T> AB(kd + 1, n, batch);
    Vector<T> tau1(std::max<int32_t>(0, static_cast<int32_t>(n) - kd), batch);
    {
        UnifiedVector<std::byte> ws1(
            sytrd_sy2sb_buffer_size<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd));
        sytrd_sy2sb<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd, ws1.to_span()).wait();
    }

    const int32_t em = std::max<int32_t>(0, static_cast<int32_t>(n) - 1);
    Vector<Real> d(n, batch), e(em, batch);
    Vector<T> tau2(em, batch);
    const int32_t nb = std::max<int32_t>(1, std::min<int32_t>(kd, 32));

    UnifiedVector<std::byte> workspace(sytrd_band_reduction_buffer_size<B, T>(
        *q, AB.view(), d, e, tau2, Uplo::Lower, kd, nb));

    // BANDR1 copies AB into its own workspace, so AB is not mutated.
    state.SetKernel(q, AB.view(), d, e, tau2, Uplo::Lower, kd, std::move(workspace), nb,
                    [](Queue& q, auto&&... xs) {
                        sytrd_band_reduction<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// Stage 2 with an explicit reduction-per-sweep d.
//
// The chase cost is ~n^2/(2*nb*b) steps per sweep, and the schedule constrains
// nb <= b - d. So d trades sweeps against panel width: d=1 means kd-1 sweeps of
// wide panels, d=kd-1 means a single sweep of nb=1. Args: n, batch, kd, d.
template <typename T, Backend B>
static void BM_SYEV_SCHED(minibench::State& state) {
    using Real = typename base_type<T>::type;
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));
    const int32_t kd = static_cast<int32_t>(state.range(2));
    const int32_t d = static_cast<int32_t>(state.range(3));

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    Matrix<T> AB(kd + 1, n, batch);
    Vector<T> tau1(std::max<int32_t>(0, static_cast<int32_t>(n) - kd), batch);
    {
        UnifiedVector<std::byte> ws1(
            sytrd_sy2sb_buffer_size<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd));
        sytrd_sy2sb<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd, ws1.to_span()).wait();
    }

    const int32_t em = std::max<int32_t>(0, static_cast<int32_t>(n) - 1);
    Vector<Real> dv(n, batch), ev(em, batch);
    Vector<T> tau2(em, batch);

    SytrdBandReductionParams p;
    p.d_seq = {d};
    // nb_target = kd lets the implementation take the widest panel the schedule
    // allows (nb = b - d), so d alone determines the schedule.
    p.block_size_seq = {kd};
    p.max_sweeps = -1;
    p.kd_work = 0;

    UnifiedVector<std::byte> workspace(sytrd_band_reduction_buffer_size<B, T>(
        *q, AB.view(), dv, ev, tau2, Uplo::Lower, kd, p));

    state.SetKernel(q, AB.view(), dv, ev, tau2, Uplo::Lower, kd, std::move(workspace), p,
                    [](Queue& q, auto&&... xs) {
                        sytrd_band_reduction<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

// Stage 2 alternative: the existing sytrd_sb2st, for reference. Same input
// band, same output tridiagonal -- this is the bar BANDR1 has to reach.
template <typename T, Backend B>
static void BM_SYEV_STAGE2_SB2ST(minibench::State& state) {
    using Real = typename base_type<T>::type;
    const size_t n = state.range(0);
    const int batch = static_cast<int>(state.range(1));
    const int32_t kd = (n <= 256) ? 16 : 32;

    auto q = std::make_shared<Queue>(B == Backend::NETLIB ? "cpu" : "gpu");

    auto A = Matrix<T>::Random(n, n, /*hermitian=*/true, batch);
    Matrix<T> AB(kd + 1, n, batch);
    Vector<T> tau1(std::max<int32_t>(0, static_cast<int32_t>(n) - kd), batch);
    {
        UnifiedVector<std::byte> ws1(
            sytrd_sy2sb_buffer_size<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd));
        sytrd_sy2sb<B, T>(*q, A.view(), AB.view(), tau1, Uplo::Lower, kd, ws1.to_span()).wait();
    }

    const int32_t em = std::max<int32_t>(0, static_cast<int32_t>(n) - 1);
    Vector<Real> dv(n, batch), ev(em, batch);
    Vector<T> tau2(em, batch);
    const int32_t block_size = 32;

    UnifiedVector<std::byte> workspace(sytrd_sb2st_buffer_size<B, T>(
        *q, AB.view(), dv, ev, tau2, Uplo::Lower, kd, block_size));

    state.SetKernel(q, bench::pristine(AB), dv, ev, tau2, Uplo::Lower, kd, std::move(workspace),
                    block_size,
                    [](Queue& q, auto&&... xs) {
                        sytrd_sb2st<B, T>(q, std::forward<decltype(xs)>(xs)...);
                    });

    set_common_metrics(state, n, batch);
}

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_STAGE1_SY2SB, SyevBandSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_STAGE2_BANDR1, SyevBandSizes);
BATCHLAS_BENCH_CUDA(BM_SYEV_SCHED, SyevSchedSizes);
BATCHLAS_BENCH_CUDA(BM_SYEV_STAGE2_SB2ST, SyevBandSizes);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_REF_BLOCKED, SyevBandSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_BAND, SyevBandSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYEV_KDSWEEP, SyevBandKdSizes);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_REF_BLOCKED, SyevBandSizes);
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_BAND, SyevBandSizes);
BATCHLAS_BENCH_ROCM_ALL_TYPES(BM_SYEV_KDSWEEP, SyevBandKdSizes);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
BATCHLAS_BENCH_NETLIB_ALL_TYPES(BM_SYEV_REF_BLOCKED, SyevBandSizes);
BATCHLAS_BENCH_NETLIB_ALL_TYPES(BM_SYEV_BAND, SyevBandSizes);
BATCHLAS_BENCH_NETLIB_ALL_TYPES(BM_SYEV_KDSWEEP, SyevBandKdSizes);
#endif

MINI_BENCHMARK_MAIN();
