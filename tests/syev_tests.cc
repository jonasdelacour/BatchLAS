#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
// syev_cta, and SteqrParams with it: the `info` cases below reach the CTA tier
// directly because no cap on the public entry point can force a non-convergence.
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/sycl-device-queue.hh>
#include <cstdint>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>

using namespace batchlas;

template <typename T, Backend B>
struct SyevConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using SyevTestTypes = typename test_utils::backend_types<SyevConfig>::type;

template <typename Config>
class SyevTest : public test_utils::BatchLASTest<Config> {
protected:
};

TYPED_TEST_SUITE(SyevTest, SyevTestTypes);

TYPED_TEST(SyevTest, DiagTest) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    auto diag = UnifiedVector<T>(n);
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename base_type<T>::type> dis(-10.0, 10.0);

    // Populate diagonal with random values
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            diag[i] = {dis(gen), 0.0};
        } else if constexpr (std::is_floating_point_v<T>) {
            diag[i] = dis(gen);
        }
    }
    
    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Diagonal(diag.to_span());
    auto A_view = A.view();

    auto W = UnifiedVector<typename base_type<T>::type>(n);
    auto workspace = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx, A_view, W.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    std::sort(diag.begin(), diag.end(), [](const T& a, const T& b) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            return std::real(a) < std::real(b);
        } else if constexpr (std::is_floating_point_v<T>) {
            return a < b;
        }
    });
    syev(*this->ctx, A_view, W.to_span(), {.jobz = JobType::NoEigenVectors}, workspace.to_span());
    (*this->ctx).wait();
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            EXPECT_NEAR(std::real(diag[i]), W[i], 1e-5);
        } else if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(diag[i], W[i], 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// Per-item convergence status (`info`).
//
// syev is where a dropped status hurts most: it is the entry point everything
// else calls, it has SIX implementation tiers below it (cta, cta_fused,
// jacobi_cta, blocked, two_stage, vendor), and every one of them ends in an
// iteration that can run out of budget. Before this work package none of them
// said so -- cuSOLVER's syevj/syevd already RETURN an info array and it was
// allocated, passed and dropped; LAPACKE_?syev's return value IS the LAPACK info
// and the netlib arm called the no-handle wrapper that discards it.
//
// The batch below is mixed on purpose: even items are diagonal (converged before
// the first sweep), odd items are a tridiagonal Toeplitz (not). Both have a
// closed-form spectrum, so "reported converged" can be checked against "actually
// right" per item rather than in aggregate.
// ---------------------------------------------------------------------------

namespace {

// Even items: diag(1, 2, ..., n). Odd items: the symmetric tridiagonal
// Toeplitz(0.5, 1, 0.5). Written through the host-side view -- a Matrix is USM,
// and every other test in this suite reads it the same way.
//
// The view is taken BY VALUE, not by const reference: MatrixView::operator() has
// a const overload returning `const T&`, so a const-reference parameter would
// make every assignment below a compile error. A MatrixView is a non-owning
// descriptor, so copying it costs nothing.
template <typename Scalar>
void fill_mixed_convergence_matrices(MatrixView<Scalar, MatrixFormat::Dense> A) {
    const int n = A.rows();
    const int batch = A.batch_size();
    for (int b = 0; b < batch; ++b) {
        const bool easy = (b % 2) == 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) A(i, j, b) = Scalar(0);
        }
        for (int i = 0; i < n; ++i) A(i, i, b) = easy ? Scalar(i + 1) : Scalar(1);
        if (!easy) {
            for (int i = 0; i + 1 < n; ++i) {
                A(i + 1, i, b) = Scalar(0.5);
                A(i, i + 1, b) = Scalar(0.5);
            }
        }
    }
}

// Ascending eigenvalue `i` of item `b`, in closed form.
template <typename Real>
Real mixed_convergence_eigenvalue(int b, int i, int n) {
    if ((b % 2) == 0) return Real(i + 1);
    return Real(1) - Real(std::cos(M_PI * double(i + 1) / double(n + 1)));
}

}  // namespace

TYPED_TEST(SyevTest, InfoIsZeroOnAConvergingBatch) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    fill_mixed_convergence_matrices<T>(A.view());
    UnifiedVector<Real> W(static_cast<size_t>(n) * static_cast<size_t>(batch));

    // -1, NOT 0: a span left at zero cannot tell "the solver wrote 0" from
    // "nothing wrote it at all", and the second is the failure this whole
    // mechanism exists to rule out. The entry point clears the span once (the
    // accumulator rule in src/extensions/info_span.hh), so a surviving -1 says
    // the clear never ran and the zeros would have been an accident.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        syev_buffer_size<B, T>(*this->ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(*this->ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower,
               ws.to_span(), info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: nothing wrote "
                                  "the span, so a zero here would have proved nothing";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a batch whose "
                                 "spectrum is available in closed form";
    }

    const double tol = std::is_same_v<Real, float> ? 1e-3 : 1e-8;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(static_cast<double>(W[static_cast<size_t>(b) * n + i]),
                        static_cast<double>(mixed_convergence_eigenvalue<Real>(b, i, n)), tol)
                << "batch " << b << " eigenvalue " << i;
        }
    }
}

// An EMPTY span is "not requested": it must change neither the answer nor the
// workspace. syev_buffer_size takes no `info` argument, so the size cannot
// depend on it by construction -- what is checked is that asking for status does
// not perturb the solve, which is the part a caller could actually observe.
TYPED_TEST(SyevTest, EmptyInfoSpanChangesNeitherAnswerNorWorkspace) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 4;

    Matrix<T, MatrixFormat::Dense> A0(n, n, batch), A1(n, n, batch);
    fill_mixed_convergence_matrices<T>(A0.view());
    fill_mixed_convergence_matrices<T>(A1.view());
    UnifiedVector<Real> W0(static_cast<size_t>(n) * batch), W1(static_cast<size_t>(n) * batch);
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    const size_t bytes_a = syev_buffer_size<B, T>(*this->ctx, A0.view(), W0.to_span(),
                                                  JobType::NoEigenVectors, Uplo::Lower);
    const size_t bytes_b = syev_buffer_size<B, T>(*this->ctx, A1.view(), W1.to_span(),
                                                  JobType::NoEigenVectors, Uplo::Lower);
    EXPECT_EQ(bytes_a, bytes_b);

    UnifiedVector<std::byte> ws0(bytes_a), ws1(bytes_a);
    syev<B, T>(*this->ctx, A0.view(), W0.to_span(), JobType::NoEigenVectors, Uplo::Lower,
               ws0.to_span(), info.to_span());
    syev<B, T>(*this->ctx, A1.view(), W1.to_span(), JobType::NoEigenVectors, Uplo::Lower,
               ws1.to_span(), Span<int32_t>{});
    this->ctx->wait();

    for (size_t i = 0; i < W0.size(); ++i) {
        EXPECT_EQ(W0[i], W1[i]) << "requesting status changed the answer at index " << i;
    }
}

// THE FORCED DIRECTION, and it has to go through a TIER rather than through
// `syev` itself.
//
// There is no knob that forces non-convergence on the public entry point:
// syev_dispatch pins detail::syev_cta_steqr_params (max_sweeps = 400) for the
// CTA route and default-constructed StedcParams for Blocked and TwoStage
// (blas/functions/syev.hh:127, :466, :474), and SyevOptions carries only jobz
// and uplo. So the honest forcing point is syev_cta, which IS what `syev` routes
// n <= 32 to, called with its own SteqrParams.
//
// max_sweeps = 1, not a "reduced" 50: syev_cta.cc:177-180 silently REWRITES
// max_sweeps to 400 whenever a caller leaves it at the default 50 with the
// default shift strategy, so a test that lowered the cap to 50 would be answered
// by 400 and would pass while testing nothing.
#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SyevTest, InfoReportsItemsThatExhaustTheSweepBudget) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    // syev_cta needs a sub-group width of 32; the host device generally has less
    // and the tier throws batchlas::unsupported there rather than running slowly.
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "syev_cta requires sub-group 32; not available on the host backend";
    } else {
        const int n = 32;
        const int batch = 8;

        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        fill_mixed_convergence_matrices<T>(A.view());
        UnifiedVector<Real> W(static_cast<size_t>(n) * static_cast<size_t>(batch));
        UnifiedVector<int32_t> info(batch, int32_t(-1));

        SteqrParams<T> params;
        params.max_sweeps = 1;

        auto ws = UnifiedVector<std::byte>(
            syev_cta_buffer_size<B, T>(*this->ctx, A.view(), JobType::NoEigenVectors, params));
        syev_cta<B, T>(*this->ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower,
                       ws.to_span(), params, /*cta_wg_size_multiplier=*/1, info.to_span());
        this->ctx->wait();

        int reported = 0;
        for (int b = 0; b < batch; ++b) {
            ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value";
            ASSERT_GE(info[b], 0) << "info is LAPACK-like: 0 or a positive count, never negative";
            if (info[b] != 0) ++reported;
        }
        EXPECT_GT(reported, 0)
            << "a one-sweep budget on a 32x32 Toeplitz batch reported universal convergence; "
               "either the status is not written, or it is written unconditionally zero";

        // The half that stops a report-failure-everywhere implementation from
        // passing: an item syev_cta says converged must still be correct.
        const double tol = std::is_same_v<Real, float> ? 1e-3 : 1e-8;
        for (int b = 0; b < batch; ++b) {
            if (info[b] != 0) continue;
            for (int i = 0; i < n; ++i) {
                EXPECT_NEAR(static_cast<double>(W[static_cast<size_t>(b) * n + i]),
                            static_cast<double>(mixed_convergence_eigenvalue<Real>(b, i, n)), tol)
                    << "item " << b << " reported info == 0 but eigenvalue " << i << " is wrong";
            }
        }
    }
}
#endif  // BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
