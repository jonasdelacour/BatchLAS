#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

template <typename Scalar>
using RealOf = typename base_type<Scalar>::type;

template <typename Scalar>
static RealOf<Scalar> abs_val(const Scalar& x) {
    return static_cast<RealOf<Scalar>>(std::abs(x));
}

template <typename Scalar>
static RealOf<Scalar> norm2_val(const Scalar& x) {
    using Real = RealOf<Scalar>;
    if constexpr (std::is_same_v<Scalar, Real>) {
        return x * x;
    } else {
        return static_cast<Real>(std::norm(x));
    }
}

template <typename Scalar>
static Scalar conj_val(const Scalar& x) {
    if constexpr (std::is_same_v<Scalar, RealOf<Scalar>>) {
        return x;
    } else {
        return std::conj(x);
    }
}

template <typename Scalar>
static void check_orthonormal_columns(const MatrixView<Scalar, MatrixFormat::Dense>& V,
                                      int n, int b, RealOf<Scalar> tol) {
    using Real = RealOf<Scalar>;
    Real max_err = Real(0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            Scalar dot = Scalar(0);
            for (int r = 0; r < n; ++r) {
                dot += conj_val(V(r, i, b)) * V(r, j, b);
            }
            const Real target = (i == j) ? Real(1) : Real(0);
            max_err = std::max(max_err, abs_val(dot - Scalar(target)));
        }
    }
    EXPECT_LE(max_err, tol) << "max |V^H V - I| = " << max_err << " (batch " << b << ")";
}

template <typename Scalar>
static void check_eigen_residual(const MatrixView<Scalar, MatrixFormat::Dense>& A0,
                                 const MatrixView<Scalar, MatrixFormat::Dense>& V,
                                 const UnifiedVector<RealOf<Scalar>>& W,
                                 int n, int b, RealOf<Scalar> tol) {
    using Real = RealOf<Scalar>;

    Real a_norm2 = Real(0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a_norm2 += norm2_val(A0(i, j, b));
        }
    }
    const Real a_norm = std::sqrt(a_norm2);

    Real r_norm2 = Real(0);
    for (int j = 0; j < n; ++j) {
        const Real wj = W[static_cast<std::size_t>(b) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
        for (int i = 0; i < n; ++i) {
            Scalar sum = Scalar(0);
            for (int k = 0; k < n; ++k) {
                sum += A0(i, k, b) * V(k, j, b);
            }
            sum -= Scalar(wj) * V(i, j, b);
            r_norm2 += norm2_val(sum);
        }
    }

    const Real r_norm = std::sqrt(r_norm2);
    const Real denom = (a_norm > Real(0)) ? (a_norm * Real(n)) : Real(1);
    const Real rel = r_norm / denom;
    EXPECT_LE(rel, tol) << "relative residual ||AV - VW||/(||A||*n) = " << rel << " (batch " << b << ")";
}

// Accumulated rounding grows with the number of rotations, which is O(n^2) per
// sweep, so the fixed absolute tolerance from test_utils is scaled by sqrt(n).
// Identical rationale (and constants) to syev_jacobi_cta_tests.
template <typename Scalar>
static RealOf<Scalar> vec_tol(int n, RealOf<Scalar> extra = RealOf<Scalar>(1)) {
    using Real = RealOf<Scalar>;
    return test_utils::tolerance<Scalar>() * std::sqrt(Real(n)) * extra;
}

template <typename Scalar>
static RealOf<Scalar> eig_compare_tol(const UnifiedVector<RealOf<Scalar>>& w_ref, int n, int batch) {
    using Real = RealOf<Scalar>;
    Real lambda_max = Real(1);
    for (int i = 0; i < n * batch; ++i) {
        lambda_max = std::max(lambda_max, std::abs(w_ref[static_cast<std::size_t>(i)]));
    }
    return test_utils::tolerance<Scalar>() * lambda_max * std::sqrt(Real(n));
}

// Runs the solver and returns the eigenvalues, taking care of the workspace.
template <Backend B, typename Scalar>
static void run_blocked(Queue& ctx,
                        MatrixView<Scalar, MatrixFormat::Dense> a,
                        Span<RealOf<Scalar>> w,
                        JobType jobz,
                        Uplo uplo,
                        JacobiParams<Scalar> params = JacobiParams<Scalar>()) {
    const std::size_t ws_bytes = syev_jacobi_blocked_buffer_size<B, Scalar>(ctx, a, jobz, params);
    auto ws = UnifiedVector<std::byte>(ws_bytes);
    syev_jacobi_blocked<B, Scalar>(ctx, a, w, jobz, uplo, ws.to_span(), params).wait();
}

// ---------------------------------------------------------------------------
// Independent CPU reference: cyclic-by-rows two-sided Jacobi in double.
//
// A double LAPACK syev cannot serve as the truth for the graded test: it
// tridiagonalizes, so its own error on the smallest eigenvalue of a strongly
// graded matrix is ~eps_double*||A||, far larger than the eigenvalue itself.
// ---------------------------------------------------------------------------
static std::vector<double> reference_jacobi_eigenvalues(const std::vector<double>& A_in, int n) {
    std::vector<double> A = A_in; // column-major n x n
    const double tol = double(n) * std::numeric_limits<double>::epsilon();

    for (int sweep = 0; sweep < 100; ++sweep) {
        int rotations = 0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[p + q * n];
                const double app = A[p + p * n];
                const double aqq = A[q + q * n];
                if (std::abs(apq) <= tol * std::sqrt(std::abs(app) * std::abs(aqq))) continue;
                if (apq == 0.0) continue;

                const double tau = (aqq - app) / (2.0 * apq);
                const double t = std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;

                for (int r = 0; r < n; ++r) {
                    const double arp = A[r + p * n];
                    const double arq = A[r + q * n];
                    A[r + p * n] = c * arp - s * arq;
                    A[r + q * n] = s * arp + c * arq;
                }
                for (int cc = 0; cc < n; ++cc) {
                    const double apc = A[p + cc * n];
                    const double aqc = A[q + cc * n];
                    A[p + cc * n] = c * apc - s * aqc;
                    A[q + cc * n] = s * apc + c * aqc;
                }
                A[p + q * n] = 0.0;
                A[q + p * n] = 0.0;
                ++rotations;
            }
        }
        if (rotations == 0) break;
    }

    std::vector<double> w(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = A[i + i * n];
    std::sort(w.begin(), w.end());
    return w;
}

// Graded SPD matrix A = D * M * D, D_ii = 2^{-e_i}, with the exponents spread
// linearly over [0, span] rather than stepped by a fixed amount per index.
//
// The fixed-step form used at n <= 32 does not survive to n = 128: a step of 2
// puts the smallest entry at 2^{-4*127}, which is not merely inaccurate in
// float, it is *zero*, so the test would be measuring underflow rather than the
// solver. Spreading a fixed total span keeps the smallest eigenvalue a decade
// or two above FLT_MIN at every n while kappa(A) stays ~2^(2*span).
//
// Every entry is a dyadic rational, so float and double see exactly the same
// matrix and any difference in the answer is precision, not input rounding. The
// column-equilibrated condition number stays O(kappa(M)) -- the regime where
// the relative-accuracy bound bites.
static void make_graded_spd(int n, int span, unsigned seed,
                            std::vector<double>& out_double,
                            std::vector<float>& out_float) {
    std::minstd_rand rng(seed);
    std::uniform_int_distribution<int> dist(-16, 16);

    std::vector<double> M(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            if (i == j) {
                M[i + j * n] = 1.0;
            } else {
                const double v = double(dist(rng)) / 256.0; // dyadic, |v| <= 1/16
                M[i + j * n] = v;
                M[j + i * n] = v;
            }
        }
    }

    std::vector<double> D(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int e = (n > 1) ? (span * i) / (n - 1) : 0;
        D[static_cast<std::size_t>(i)] = std::ldexp(1.0, -e);
    }

    out_double.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
    out_float.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const double v = D[static_cast<std::size_t>(i)] * M[i + j * n] * D[static_cast<std::size_t>(j)];
            out_double[static_cast<std::size_t>(i + j * n)] = v;
            out_float[static_cast<std::size_t>(i + j * n)] = static_cast<float>(v);
        }
    }
}

template <typename T, Backend B>
struct SyevJacobiBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevJacobiBlockedTestTypes = ::testing::Types<
    SyevJacobiBlockedConfig<float, Backend::CUDA>,
    SyevJacobiBlockedConfig<double, Backend::CUDA>,
    SyevJacobiBlockedConfig<std::complex<float>, Backend::CUDA>,
    SyevJacobiBlockedConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SyevJacobiBlockedTestTypes = ::testing::Types<
    SyevJacobiBlockedConfig<float, Backend::ROCM>,
    SyevJacobiBlockedConfig<double, Backend::ROCM>,
    SyevJacobiBlockedConfig<std::complex<float>, Backend::ROCM>,
    SyevJacobiBlockedConfig<std::complex<double>, Backend::ROCM>>;
#else
using SyevJacobiBlockedTestTypes = ::testing::Types<SyevJacobiBlockedConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SyevJacobiBlockedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevJacobiBlockedTest, SyevJacobiBlockedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND

// n = 64 is the l == 2 case: the pivot block is the whole matrix, so the panel
// update and the mirror never run. n = 128 exercises the general path with four
// block columns and six pivot pairs per sweep.
TYPED_TEST(SyevJacobiBlockedTest, EigenvaluesMatchNetlib) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 3;

    for (int n : {64, 128}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n
                            << " uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper"));

            Matrix<Scalar, MatrixFormat::Dense> A0 =
                Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/321);
            Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;
            Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

            auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
            auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

#if BATCHLAS_HAS_HOST_BACKEND
            {
                auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(
                    *this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, uplo));
                syev(*this->ctx,
                     A_ref.view(),
                     W_ref.to_span(),
                     {.jobz = JobType::NoEigenVectors, .uplo = uplo},
                     ws_ref.to_span()).wait();
            }
#endif

            run_blocked<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(),
                                   JobType::NoEigenVectors, uplo);

#if BATCHLAS_HAS_HOST_BACKEND
            const Real tol = eig_compare_tol<Scalar>(W_ref, n, batch);
            for (int b = 0; b < batch; ++b) {
                for (int i = 0; i < n; ++i) {
                    const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
                                          + static_cast<std::size_t>(i);
                    ASSERT_NEAR(W_jac[idx], W_ref[idx], tol)
                        << "eigenvalue mismatch i=" << i << " batch=" << b << " n=" << n;
                }
            }
#endif
        }
    }
}

TYPED_TEST(SyevJacobiBlockedTest, EigenvectorsResidualAndOrtho) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 2;

    for (int n : {64, 128}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n
                            << " uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper"));

            Matrix<Scalar, MatrixFormat::Dense> A0 =
                Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/654);
            Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;

            auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
            run_blocked<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(),
                                   JobType::EigenVectors, uplo);

            for (int b = 0; b < batch; ++b) {
                check_orthonormal_columns(A_jac.view(), n, b, vec_tol<Scalar>(n));
                check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, b, vec_tol<Scalar>(n));
            }
        }
    }
}

// Sizes that are not multiples of the auto-selected block width, plus odd n,
// which pads the round-robin index space and forces the phantom-pivot skip.
TYPED_TEST(SyevJacobiBlockedTest, RaggedAndOddSizes) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 2;

    for (int n : {33, 47, 65, 97, 129}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);

        Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(
            n, n, /*hermitian=*/true, batch, /*seed=*/static_cast<unsigned>(2000 + n));
        Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;

        auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
        run_blocked<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(),
                               JobType::EigenVectors, Uplo::Lower);

        for (int b = 0; b < batch; ++b) {
            check_orthonormal_columns(A_jac.view(), n, b, vec_tol<Scalar>(n));
            check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, b, vec_tol<Scalar>(n));
        }
    }
}

// A narrow forced block width turns one pivot pair per sweep into many, so this
// is the test that actually exercises the panel update, the mirror and the
// cyclic block ordering rather than the fully resident degenerate case.
TYPED_TEST(SyevJacobiBlockedTest, ForcedNarrowBlockWidth) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 64;
    const int batch = 2;

    for (std::size_t nb : {std::size_t(4), std::size_t(8), std::size_t(11)}) {
        SCOPED_TRACE(::testing::Message() << "nb=" << nb);

        Matrix<Scalar, MatrixFormat::Dense> A0 =
            Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/777);
        Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;

        JacobiParams<Scalar> params;
        params.block_size = nb;

        auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
        run_blocked<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(),
                               JobType::EigenVectors, Uplo::Lower, params);

        for (int b = 0; b < batch; ++b) {
            check_orthonormal_columns(A_jac.view(), n, b, vec_tol<Scalar>(n));
            check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, b, vec_tol<Scalar>(n));
        }
    }
}

// inner_sweeps = 1 is the inexact/block-oriented variant; a value at or above
// max_sweeps diagonalizes each pivot block exactly. Both have to reach the same
// answer; only the sweep count (and hence the cost) differs.
TYPED_TEST(SyevJacobiBlockedTest, InnerSweepVariantsAgree) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 96;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A0 =
        Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/8181);

    auto W_inexact = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
    auto W_exact = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

    {
        Matrix<Scalar, MatrixFormat::Dense> A = A0;
        JacobiParams<Scalar> params;
        params.inner_sweeps = 1;
        params.block_size = 8;
        run_blocked<B, Scalar>(*this->ctx, A.view(), W_inexact.to_span(),
                               JobType::NoEigenVectors, Uplo::Lower, params);
    }
    {
        Matrix<Scalar, MatrixFormat::Dense> A = A0;
        JacobiParams<Scalar> params;
        params.inner_sweeps = params.max_sweeps; // diagonalize each pivot block exactly
        params.block_size = 8;
        run_blocked<B, Scalar>(*this->ctx, A.view(), W_exact.to_span(),
                               JobType::NoEigenVectors, Uplo::Lower, params);
    }

    const Real tol = eig_compare_tol<Scalar>(W_exact, n, batch);
    for (int i = 0; i < n * batch; ++i) {
        ASSERT_NEAR(W_inexact[static_cast<std::size_t>(i)], W_exact[static_cast<std::size_t>(i)], tol)
            << "inexact and exact inner solves disagree at i=" << i;
    }
}

TYPED_TEST(SyevJacobiBlockedTest, DescendingSortOrder) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 64;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A0 =
        Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/91);

    auto W_asc = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
    auto W_desc = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

    {
        Matrix<Scalar, MatrixFormat::Dense> A = A0;
        run_blocked<B, Scalar>(*this->ctx, A.view(), W_asc.to_span(), JobType::NoEigenVectors, Uplo::Lower);
    }
    {
        Matrix<Scalar, MatrixFormat::Dense> A = A0;
        JacobiParams<Scalar> params;
        params.sort_order = SortOrder::Descending;
        run_blocked<B, Scalar>(*this->ctx, A.view(), W_desc.to_span(),
                               JobType::NoEigenVectors, Uplo::Lower, params);
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            const std::size_t asc = static_cast<std::size_t>(b) * n + static_cast<std::size_t>(i);
            const std::size_t desc = static_cast<std::size_t>(b) * n + static_cast<std::size_t>(n - 1 - i);
            ASSERT_EQ(W_asc[asc], W_desc[desc]) << "descending order is not the reverse of ascending, i=" << i;
        }
        for (int i = 1; i < n; ++i) {
            const std::size_t base = static_cast<std::size_t>(b) * n;
            ASSERT_LE(W_asc[base + i - 1], W_asc[base + i]);
        }
    }
}

// ---------------------------------------------------------------------------
// The payoff. Graded SPD input where kappa(A) ~ 1e36 but the column-equilibrated
// condition number stays O(1). syev_jacobi_cta proves this at n <= 32; this test
// is the statement that blocking does not destroy it at n = 64 and n = 128.
//
// The reference is an independent double-precision CPU Jacobi. The vendor/
// blocked syev column is printed for context but deliberately not asserted --
// it is expected to be catastrophically wrong here, and pinning someone else's
// error is not this test's job.
// ---------------------------------------------------------------------------
TEST(SyevJacobiBlockedRelativeAccuracy, GradedSpd) {
#if BATCHLAS_HAS_CUDA_BACKEND
    constexpr Backend B = Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
    constexpr Backend B = Backend::ROCM;
#else
    GTEST_SKIP() << "No GPU backend built.";
#endif
#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
    auto ctx = std::make_shared<Queue>(Device("gpu"), B);

    for (int n : {64, 128}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);

        // kappa(A) ~ 2^(2*span) ~ 1e35, with the smallest eigenvalue near
        // 2^-116 ~ 1e-35 -- three orders above FLT_MIN, so it is a number
        // float can represent and therefore a number the solver is obliged to
        // resolve. A tridiagonalizing solver's relative error here is
        // ~ eps_float * kappa(A) ~ 1e28.
        const int span = 58;

        std::vector<double> Ad;
        std::vector<float> Af;
        make_graded_spd(n, span, /*seed=*/13, Ad, Af);

        const std::vector<double> w_ref = reference_jacobi_eigenvalues(Ad, n);

        Matrix<float, MatrixFormat::Dense> A(n, n, 1);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A.view()(i, j, 0) = Af[static_cast<std::size_t>(i + j * n)];
            }
        }

        auto W = UnifiedVector<float>(static_cast<std::size_t>(n));
        JacobiParams<float> params;
        const std::size_t ws_bytes =
            syev_jacobi_blocked_buffer_size<B, float>(*ctx, A.view(), JobType::NoEigenVectors, params);
        auto ws = UnifiedVector<std::byte>(ws_bytes);
        syev_jacobi_blocked<B, float>(*ctx, A.view(), W.to_span(), JobType::NoEigenVectors,
                                      Uplo::Lower, ws.to_span(), params).wait();

        auto max_relative_error = [&](const UnifiedVector<float>& w) {
            double worst = 0.0;
            for (int i = 0; i < n; ++i) {
                const double ref = w_ref[static_cast<std::size_t>(i)];
                if (ref == 0.0) continue;
                worst = std::max(worst,
                                 std::abs(double(w[static_cast<std::size_t>(i)]) - ref) / std::abs(ref));
            }
            return worst;
        };

        const double max_rel = max_relative_error(W);

        // The routed syev on identical input, for contrast. It tridiagonalizes,
        // so its relative error on the small end is ~ eps * kappa(A); the figure
        // is printed rather than asserted, because pinning another solver's
        // error is not this test's job.
        Matrix<float, MatrixFormat::Dense> A_ref(n, n, 1);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A_ref.view()(i, j, 0) = Af[static_cast<std::size_t>(i + j * n)];
            }
        }
        auto W_ref_dev = UnifiedVector<float>(static_cast<std::size_t>(n));
        {
            auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(
                *ctx, A_ref.view(), W_ref_dev.to_span(), JobType::NoEigenVectors, Uplo::Lower));
            syev(*ctx, A_ref.view(), W_ref_dev.to_span(),
                 {.jobz = JobType::NoEigenVectors, .uplo = Uplo::Lower}, ws_ref.to_span()).wait();
        }

        std::cout << "  n=" << n << " span=" << span
                  << " spectrum [" << w_ref.front() << ", " << w_ref.back() << "]"
                  << "  max relative eigenvalue error: jacobi_blocked = " << max_rel
                  << ", syev = " << max_relative_error(W_ref_dev) << std::endl;

        // A tridiagonalizing solver has relative error ~ eps*kappa(A) here,
        // i.e. astronomically large. Jacobi's bound is ~ eps * kappa of the
        // equilibrated matrix, a small multiple of float eps. The constant is
        // generous relative to the measured value so the test pins the property
        // rather than the exact rounding.
        EXPECT_LT(max_rel, 1e-4) << "relative accuracy on graded SPD input was lost at n=" << n;
    }
#endif
}

#endif // GPU backend
