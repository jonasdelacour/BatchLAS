#include <util/minibench.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include <algorithm>
#include <complex>

using namespace batchlas;

namespace {

template <typename T>
inline T conj_if_needed(const T& x) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T>
inline void fill_lower_band_from_dense(const MatrixView<T, MatrixFormat::Dense>& A,
                                      MatrixView<T, MatrixFormat::Dense> AB,
                                      int n,
                                      int kd) {
    // AB is (kd+1) x n, lower band: AB(r,j) = A(j+r, j)
    for (int j = 0; j < n; ++j) {
        const int rmax = std::min(kd, n - 1 - j);
        for (int r = 0; r <= rmax; ++r) {
            AB(r, j, 0) = A(j + r, j, 0);
        }
        for (int r = rmax + 1; r <= kd; ++r) {
            AB(r, j, 0) = T(0);
        }
    }
}

template <typename Benchmark>
inline void SytrdSb2stBenchSizes(Benchmark* b) {
    for (int n : {128, 256, 512}) {
        for (int bs : {1, 8, 32, 64}) {
            for (int kd : {4, 8, 16}) {
                if (kd >= n) continue;
                // uplo: 0=Lower, 1=Upper (sb2st currently implements Lower only)
                b->Args({n, bs, kd, 0});
            }
        }
    }
}

} // namespace

template <typename T, Backend B>
static void BM_SYTRD_SB2ST(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int kd = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));
    const Uplo uplo = (uplo_i == 0) ? Uplo::Lower : Uplo::Upper;

    // NOTE: `sytrd_sb2st` currently ignores `block_size`.
    constexpr int ib = 32;

    // Build a Hermitian/symmetric band input once.
    auto A0 = Matrix<T>::Random(n, n, /*hermitian=*/true, /*batch=*/1, /*seed=*/2027);

    auto AB = Matrix<T, MatrixFormat::Dense>::Zeros(kd + 1, n, batch);
    {
        // Fill batch 0 from A0, and replicate across batch for deterministic inputs.
        Matrix<T, MatrixFormat::Dense> AB0(kd + 1, n, /*batch=*/1);
        fill_lower_band_from_dense<T>(A0.view(), AB0.view(), static_cast<int>(n), kd);
        for (size_t b = 0; b < batch; ++b) {
            for (int j = 0; j < static_cast<int>(n); ++j) {
                for (int r = 0; r <= kd; ++r) {
                    AB(r, j, static_cast<int>(b)) = AB0(r, j, 0);
                }
            }
        }
    }

    using Real = typename base_type<T>::type;
    auto d = Vector<Real>::zeros(n, batch);
    auto e = Vector<Real>::zeros(n > 0 ? n - 1 : 0, batch);
    auto tau = Vector<T>::zeros(n > 0 ? n - 1 : 0, batch);

    auto q = std::make_shared<Queue>("gpu", /*in_order=*/true);

    const size_t ws_bytes = sytrd_sb2st_buffer_size<B, T>(*q,
                                                         AB.view(),
                                                         VectorView<Real>(d),
                                                         VectorView<Real>(e),
                                                         VectorView<T>(tau),
                                                         uplo,
                                                         kd,
                                                         ib);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(AB),
        d,
        e,
        bench::pristine(tau),
        uplo,
        kd,
        ws,
        ib,
        [](Queue& q, auto&&... xs) {
            sytrd_sb2st<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });

    state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

template <typename T, Backend B>
static void BM_SYTRD_BAND_REDUCTION(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
    const size_t n = state.range(0);
    const size_t batch = state.range(1);
    const int kd = static_cast<int>(state.range(2));
    const int uplo_i = static_cast<int>(state.range(3));
    const Uplo uplo = (uplo_i == 0) ? Uplo::Lower : Uplo::Upper;

    constexpr int ib = 32;

    // Build a Hermitian/symmetric band input once.
    auto A0 = Matrix<T>::Random(n, n, /*hermitian=*/true, /*batch=*/1, /*seed=*/2028);

    auto AB = Matrix<T, MatrixFormat::Dense>::Zeros(kd + 1, n, batch);
    {
        Matrix<T, MatrixFormat::Dense> AB0(kd + 1, n, /*batch=*/1);
        fill_lower_band_from_dense<T>(A0.view(), AB0.view(), static_cast<int>(n), kd);
        for (size_t b = 0; b < batch; ++b) {
            for (int j = 0; j < static_cast<int>(n); ++j) {
                for (int r = 0; r <= kd; ++r) {
                    AB(r, j, static_cast<int>(b)) = AB0(r, j, 0);
                }
            }
        }
    }

    using Real = typename base_type<T>::type;
    auto d = Vector<Real>::zeros(n, batch);
    auto e = Vector<Real>::zeros(n > 0 ? n - 1 : 0, batch);
    auto tau = Vector<T>::zeros(n > 0 ? n - 1 : 0, batch);

    auto q = std::make_shared<Queue>("gpu", /*in_order=*/true);

    const size_t ws_bytes = sytrd_band_reduction_buffer_size<B, T>(*q,
                                                                   AB.view(),
                                                                   VectorView<Real>(d),
                                                                   VectorView<Real>(e),
                                                                   VectorView<T>(tau),
                                                                   uplo,
                                                                   kd,
                                                                   ib);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    state.SetKernel(
        q,
        bench::pristine(AB),
        d,
        e,
        bench::pristine(tau),
        uplo,
        kd,
        ws,
        ib,
        [](Queue& q, auto&&... xs) {
            sytrd_band_reduction<B, T>(q, std::forward<decltype(xs)>(xs)...);
        });

    state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
    (void)state;
#endif
}

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYTRD_SB2ST, SytrdSb2stBenchSizes);
BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_SYTRD_BAND_REDUCTION, SytrdSb2stBenchSizes);

MINI_BENCHMARK_MAIN();
