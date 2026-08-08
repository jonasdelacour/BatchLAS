// End-to-end checks for syev_two_stage in eigenvector mode.
//
// Until sytrd_sb2st_hh existed, this path forced kd=1 (syev_two_stage.cc), so
// stage 1 degenerated to an unblocked BLAS-2 reduction and stage 2 was a no-op.
// Now kd > 1 is used and the eigenvectors come back through two back-transforms,
// Z := Q1 (Q2 Z). These tests pin the properties that ordering has to satisfy:
// the residual A Z = Z diag(w), orthonormality of Z, and agreement of the
// spectrum with the blocked path.

#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename U>
inline U conj_if(const U& x) {
    if constexpr (std::is_same_v<U, std::complex<float>> ||
                  std::is_same_v<U, std::complex<double>>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T, Backend Back>
struct TwoStageConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = Back;
};

template <typename Config>
class SyevTwoStageTest : public test_utils::BatchLASTest<Config> {};

#if BATCHLAS_HAS_CUDA_BACKEND
using TwoStageTypes = ::testing::Types<
    TwoStageConfig<float, Backend::CUDA>,
    TwoStageConfig<double, Backend::CUDA>,
    TwoStageConfig<std::complex<float>, Backend::CUDA>,
    TwoStageConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using TwoStageTypes = ::testing::Types<
    TwoStageConfig<float, Backend::ROCM>,
    TwoStageConfig<double, Backend::ROCM>>;
#else
using TwoStageTypes = ::testing::Types<TwoStageConfig<float, Backend::NETLIB>>;
#endif

TYPED_TEST_SUITE(SyevTwoStageTest, TwoStageTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SyevTwoStageTest, EigenvectorResidualAndOrthogonality) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);

    for (int n : {32, 64, 129}) {
        const int batch = 3;

        Matrix<T, MatrixFormat::Dense> A0 =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/17);

        // Keep a host copy: syev_two_stage overwrites A with the eigenvectors.
        std::vector<T> Aref(static_cast<size_t>(batch) * n * n);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Aref[(static_cast<size_t>(b) * n + i) * n + j] = A0(i, j, b);

        UnifiedVector<Real> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(syev_two_stage_buffer_size(
            ctx, A0.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));

        syev_two_stage(ctx, A0.view(), w.to_span(), JobType::EigenVectors,
                             Uplo::Lower, ws.to_span(), StedcParams<Real>{})
            .wait();

        for (int b = 0; b < batch; ++b) {
            Real anorm = Real(0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const Real m = std::abs(Aref[(static_cast<size_t>(b) * n + i) * n + j]);
                    anorm += m * m;
                }
            anorm = std::max(std::sqrt(anorm), Real(1));

            // ||A Z - Z diag(w)||_F / ||A||_F
            Real resid = Real(0);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    T acc = T(0);
                    for (int l = 0; l < n; ++l)
                        acc += Aref[(static_cast<size_t>(b) * n + i) * n + l] * A0(l, j, b);
                    const T diff = acc - A0(i, j, b) * T(w[static_cast<size_t>(b) * n + j]);
                    resid += std::abs(diff) * std::abs(diff);
                }
            }
            EXPECT_LT(std::sqrt(resid) / anorm, tol)
                << "residual n=" << n << " b=" << b;

            // ||Z^H Z - I||_F
            Real orth = Real(0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    T acc = T(0);
                    for (int l = 0; l < n; ++l) acc += conj_if(A0(l, i, b)) * A0(l, j, b);
                    if (i == j) acc -= T(1);
                    orth += std::abs(acc) * std::abs(acc);
                }
            EXPECT_LT(std::sqrt(orth), tol * Real(n)) << "orthogonality n=" << n << " b=" << b;
        }
    }
}

// None of 100/130/150/200/250 is a multiple of the nominal kd=32, so each one
// leaves a short final panel in sy2sb. That used to skip part of the two-sided
// update and return garbage with a residual of 0.1-0.8 -- silently, which is
// what made it survive so long. Keep these sizes covered: they are the cheapest
// guard against the trailing-panel handling regressing.
TYPED_TEST(SyevTwoStageTest, AwkwardSizesStayAccurate) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);

    for (int n : {100, 130, 150, 200, 250}) {
        const int batch = 2;

        Matrix<T, MatrixFormat::Dense> A0 =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/29);

        std::vector<T> Aref(static_cast<size_t>(batch) * n * n);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Aref[(static_cast<size_t>(b) * n + i) * n + j] = A0(i, j, b);

        UnifiedVector<Real> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(syev_two_stage_buffer_size(
            ctx, A0.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));

        syev_two_stage(ctx, A0.view(), w.to_span(), JobType::EigenVectors,
                             Uplo::Lower, ws.to_span(), StedcParams<Real>{})
            .wait();

        for (int b = 0; b < batch; ++b) {
            Real anorm = Real(0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const Real m = std::abs(Aref[(static_cast<size_t>(b) * n + i) * n + j]);
                    anorm += m * m;
                }
            anorm = std::max(std::sqrt(anorm), Real(1));

            Real resid = Real(0);
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    T acc = T(0);
                    for (int l = 0; l < n; ++l)
                        acc += Aref[(static_cast<size_t>(b) * n + i) * n + l] * A0(l, j, b);
                    const T diff = acc - A0(i, j, b) * T(w[static_cast<size_t>(b) * n + j]);
                    resid += std::abs(diff) * std::abs(diff);
                }
            }
            EXPECT_LT(std::sqrt(resid) / anorm, tol)
                << "residual n=" << n << " b=" << b;
        }
    }
}

// Baseline: the identical residual check run through syev_blocked. If both this
// and the two-stage test fail at a given size/type, the fault is shared
// machinery (ormqr_blocked, stedc), not the two-stage path.
TYPED_TEST(SyevTwoStageTest, BlockedBaselineResidual) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);

    for (int n : {32, 64, 129}) {
        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A0 =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/17);
        std::vector<T> Aref(static_cast<size_t>(batch) * n * n);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    Aref[(static_cast<size_t>(b) * n + i) * n + j] = A0(i, j, b);

        UnifiedVector<Real> w(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> ws(syev_blocked_buffer_size(
            ctx, A0.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_blocked(ctx, A0.view(), w.to_span(), JobType::EigenVectors,
                           Uplo::Lower, ws.to_span(), StedcParams<Real>{})
            .wait();

        for (int b = 0; b < batch; ++b) {
            Real anorm = Real(0);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) {
                    const Real m = std::abs(Aref[(static_cast<size_t>(b) * n + i) * n + j]);
                    anorm += m * m;
                }
            anorm = std::max(std::sqrt(anorm), Real(1));
            Real resid = Real(0);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    T acc = T(0);
                    for (int l = 0; l < n; ++l)
                        acc += Aref[(static_cast<size_t>(b) * n + i) * n + l] * A0(l, j, b);
                    const T diff = acc - A0(i, j, b) * T(w[static_cast<size_t>(b) * n + j]);
                    resid += std::abs(diff) * std::abs(diff);
                }
            EXPECT_LT(std::sqrt(resid) / anorm, tol)
                << "blocked residual n=" << n << " b=" << b;
        }
    }
}

// Eigenvalues-only exercises sy2sb with kd>1 plus the *Givens* stage 2.
//
// DISABLED, and the recorded cause is now STALE. It read: syev_two_stage passes
// NoEigenVectors to stedc, stedc_impl honours jobz only at the leaf, so above the
// leaves the merge reads child eigenvector rows that steqr_cta never formed and
// the returned values are those of the *split* matrix. Neither side does that any
// more -- two_stage routes values mode to stebz, and so does syev_blocked as of
// the WP1 change. Flip DISABLED_ off and re-measure; it sorts both sides before
// comparing, so it asserts spectra, not ordering.
TYPED_TEST(SyevTwoStageTest, DISABLED_EigenvaluesOnlyMatchesBlocked) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);

    for (int n : {32, 64, 129}) {
        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A1 =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/23);
        Matrix<T, MatrixFormat::Dense> A2(n, n, batch);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) A2(i, j, b) = A1(i, j, b);

        UnifiedVector<Real> w1(static_cast<size_t>(n) * batch);
        UnifiedVector<Real> w2(static_cast<size_t>(n) * batch);

        UnifiedVector<std::byte> ws1(syev_two_stage_buffer_size(
            ctx, A1.view(), JobType::NoEigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_two_stage(ctx, A1.view(), w1.to_span(), JobType::NoEigenVectors,
                             Uplo::Lower, ws1.to_span(), StedcParams<Real>{})
            .wait();

        UnifiedVector<std::byte> ws2(syev_blocked_buffer_size(
            ctx, A2.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_blocked(ctx, A2.view(), w2.to_span(), JobType::EigenVectors,
                           Uplo::Lower, ws2.to_span(), StedcParams<Real>{})
            .wait();

        for (int b = 0; b < batch; ++b) {
            std::vector<Real> a(w1.begin() + static_cast<ptrdiff_t>(b) * n,
                                w1.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::vector<Real> c(w2.begin() + static_cast<ptrdiff_t>(b) * n,
                                w2.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::sort(a.begin(), a.end());
            std::sort(c.begin(), c.end());
            Real scale = Real(1);
            for (Real v : c) scale = std::max(scale, std::abs(v));
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(a[i], c[i], tol * scale)
                    << "evals-only eigenvalue " << i << " n=" << n << " b=" << b;
        }
    }
}

TYPED_TEST(SyevTwoStageTest, SpectrumMatchesBlocked) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);

    for (int n : {32, 64, 129}) {
        const int batch = 3;

        Matrix<T, MatrixFormat::Dense> A1 =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/23);
        Matrix<T, MatrixFormat::Dense> A2(n, n, batch);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) A2(i, j, b) = A1(i, j, b);

        UnifiedVector<Real> w1(static_cast<size_t>(n) * batch);
        UnifiedVector<Real> w2(static_cast<size_t>(n) * batch);

        UnifiedVector<std::byte> ws1(syev_two_stage_buffer_size(
            ctx, A1.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_two_stage(ctx, A1.view(), w1.to_span(), JobType::EigenVectors,
                             Uplo::Lower, ws1.to_span(), StedcParams<Real>{})
            .wait();

        UnifiedVector<std::byte> ws2(syev_blocked_buffer_size(
            ctx, A2.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_blocked(ctx, A2.view(), w2.to_span(), JobType::EigenVectors,
                           Uplo::Lower, ws2.to_span(), StedcParams<Real>{})
            .wait();

        for (int b = 0; b < batch; ++b) {
            std::vector<Real> a(w1.begin() + static_cast<ptrdiff_t>(b) * n,
                                w1.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::vector<Real> c(w2.begin() + static_cast<ptrdiff_t>(b) * n,
                                w2.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::sort(a.begin(), a.end());
            std::sort(c.begin(), c.end());
            Real scale = Real(1);
            for (Real v : c) scale = std::max(scale, std::abs(v));
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(a[i], c[i], tol * scale)
                    << "eigenvalue " << i << " n=" << n << " b=" << b;
        }
    }
}
#endif

// Isolates stage 1: sy2sb must be a similarity, so the band it produces has to
// have the same spectrum as the input. sytrd_sy2sb_tests covers only float and
// double, and eigenvector mode is the first caller to use kd>1, so this is also
// the first complex coverage sy2sb has had.
//
// The (n, kd) pairs are chosen around the trailing-panel bug this test found:
// sy2sb skipped Q^H on columns [i+pk, i+kd) of the final panel whenever
// pk = n % kd was < kd, corrupting the band for every n % kd >= 2. The first
// four pairs all have n % kd >= 2 and failed at O(0.1); {129,16} has n % kd == 1
// and failed for complex only, because a 1x1 complex larfg still returns a
// nonzero tau (it rotates the diagonal real) where the real one returns 0; the
// last two divide evenly and always passed.
TYPED_TEST(SyevTwoStageTest, Sy2sbBandPreservesSpectrum) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    const Real tol = std::is_same_v<Real, float> ? Real(2e-3) : Real(1e-9);
    const std::vector<std::pair<int, int>> cases = {
        {64, 48}, {96, 64}, {128, 48}, {100, 32}, {129, 16}, {128, 16}, {96, 24}};

    for (const auto& nk : cases) {
        const int n = nk.first;
        const int kd = nk.second;
        {
        const int batch = 2;

        Matrix<T, MatrixFormat::Dense> A =
            Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/31);
        Matrix<T, MatrixFormat::Dense> Acopy(n, n, batch);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) Acopy(i, j, b) = A(i, j, b);

        Matrix<T, MatrixFormat::Dense> ab(kd + 1, n, batch);
        Vector<T> tau1(std::max(1, n - kd), batch);
        UnifiedVector<std::byte> ws1(sytrd_sy2sb_buffer_size(
            ctx, A.view(), ab.view(), tau1.view(), Uplo::Lower, kd));
        sytrd_sy2sb(ctx, A.view(), ab.view(), tau1.view(), Uplo::Lower, kd, ws1.to_span())
            .wait();

        // Expand the band back to a dense Hermitian matrix.
        Matrix<T, MatrixFormat::Dense> Bdense(n, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j) Bdense(i, j, b) = T(0);
            for (int j = 0; j < n; ++j)
                for (int r = 0; r <= kd; ++r) {
                    const int i = j + r;
                    if (i >= n) continue;
                    const T v = ab(r, j, b);
                    if (r == 0) {
                        // Hermitian diagonal must be real; writing v then
                        // conj(v) would flip the sign of any imaginary residue.
                        if constexpr (std::is_same_v<T, std::complex<float>> ||
                                      std::is_same_v<T, std::complex<double>>) {
                            Bdense(j, j, b) = T(v.real(), typename base_type<T>::type(0));
                        } else {
                            Bdense(j, j, b) = v;
                        }
                    } else {
                        Bdense(i, j, b) = v;
                        Bdense(j, i, b) = conj_if(v);
                    }
                }
        }

        UnifiedVector<Real> wA(static_cast<size_t>(n) * batch);
        UnifiedVector<Real> wB(static_cast<size_t>(n) * batch);
        UnifiedVector<std::byte> wsA(syev_blocked_buffer_size(
            ctx, Acopy.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_blocked(ctx, Acopy.view(), wA.to_span(), JobType::EigenVectors,
                           Uplo::Lower, wsA.to_span(), StedcParams<Real>{}).wait();
        UnifiedVector<std::byte> wsB(syev_blocked_buffer_size(
            ctx, Bdense.view(), JobType::EigenVectors, Uplo::Lower, StedcParams<Real>{}));
        syev_blocked(ctx, Bdense.view(), wB.to_span(), JobType::EigenVectors,
                           Uplo::Lower, wsB.to_span(), StedcParams<Real>{}).wait();

        for (int b = 0; b < batch; ++b) {
            std::vector<Real> x(wA.begin() + static_cast<ptrdiff_t>(b) * n,
                                wA.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::vector<Real> y(wB.begin() + static_cast<ptrdiff_t>(b) * n,
                                wB.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            std::sort(x.begin(), x.end());
            std::sort(y.begin(), y.end());
            Real scale = Real(1);
            for (Real v : x) scale = std::max(scale, std::abs(v));
            for (int i = 0; i < n; ++i)
                EXPECT_NEAR(x[i], y[i], tol * scale)
                    << "sy2sb band eigenvalue " << i << " n=" << n << " kd=" << kd << " b=" << b;
        }
      }
    }
}

} // namespace
