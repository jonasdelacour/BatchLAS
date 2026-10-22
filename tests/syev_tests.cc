#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
// syev_cta, and SteqrParams with it: the `info` cases below reach the CTA tier
// directly because no cap on the public entry point can force a non-convergence.
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/sycl-device-queue.hh>
#include <cstdint>
#include <cstring>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/env.hh>
#include <src/select/vendor.hh>
#include <algorithm>
#include <complex>
#include <string>
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

#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>
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
    // Fixed seed: a random_device seed made every run, and every compiler, a different matrix.
    std::mt19937 gen(4747u);
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
    (void)syev(*this->ctx, A_view, W.to_span(), {.jobz = JobType::NoEigenVectors}, workspace.to_span());
    (*this->ctx).wait();
    std::vector<std::vector<double>> ref(1);
    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        ref[0].push_back(static_cast<double>(std::real(diag[i])));
        scale = std::max(scale, std::fabs(ref[0].back()));
    }
    EXPECT_VERIFY(T, verify::Check::values, n,
                  MatrixView<T>::values_error(VectorView<typename base_type<T>::type>(W, n), ref, scale, std::vector<int>{0}));
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

// The items whose info is 0 against the closed-form spectrum, each relative to its own ||A||_2,
// at Check::values; the worst item is judged.
template <typename T>
void expect_converged_items_exact(const UnifiedVector<typename base_type<T>::type>& W, const UnifiedVector<int32_t>& info, int n, int batch) {
    using Real = typename base_type<T>::type;
    const VectorView<Real> w(const_cast<Real*>(W.data()), n, batch);
    std::vector<std::vector<double>> ref(static_cast<size_t>(batch));
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        double scale = 0;
        for (int i = 0; i < n; ++i) {
            ref[static_cast<size_t>(b)].push_back(double(mixed_convergence_eigenvalue<Real>(b, i, n)));
            scale = verify::nanmax(scale, std::fabs(ref[static_cast<size_t>(b)].back()));
        }
        const int item[] = {b};
        worst = verify::nanmax(worst, MatrixView<T, MatrixFormat::Dense>::values_error(w, ref, scale, item));
    }
    EXPECT_VERIFY(T, verify::Check::values, n, worst);
}

}  // namespace

TYPED_TEST(SyevTest, InfoIsZeroOnAConvergingBatch) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    fill_mixed_convergence_matrices<T>(A);
    UnifiedVector<Real> W(static_cast<size_t>(n) * static_cast<size_t>(batch));

    // -1, NOT 0: a span left at zero cannot tell "the solver wrote 0" from
    // "nothing wrote it at all", and the second is the failure this whole
    // mechanism exists to rule out. The entry point clears the span once (the
    // accumulator rule in src/kernels/common/info_span.hh), so a surviving -1 says
    // the clear never ran and the zeros would have been an accident.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        syev_buffer_size<B, T>(*this->ctx, A, W.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    (void)syev<B, T>(*this->ctx, A, W.to_span(), JobType::NoEigenVectors, Uplo::Lower,
               ws.to_span(), info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: nothing wrote "
                                  "the span, so a zero here would have proved nothing";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a batch whose "
                                 "spectrum is available in closed form";
    }

    expect_converged_items_exact<T>(W, info, n, batch);
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
    fill_mixed_convergence_matrices<T>(A0);
    fill_mixed_convergence_matrices<T>(A1);
    UnifiedVector<Real> W0(static_cast<size_t>(n) * batch), W1(static_cast<size_t>(n) * batch);
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    const size_t bytes_a = syev_buffer_size<B, T>(*this->ctx, A0, W0.to_span(),
                                                  JobType::NoEigenVectors, Uplo::Lower);
    const size_t bytes_b = syev_buffer_size<B, T>(*this->ctx, A1, W1.to_span(),
                                                  JobType::NoEigenVectors, Uplo::Lower);
    EXPECT_EQ(bytes_a, bytes_b);

    UnifiedVector<std::byte> ws0(bytes_a), ws1(bytes_a);
    (void)syev<B, T>(*this->ctx, A0, W0.to_span(), JobType::NoEigenVectors, Uplo::Lower,
               ws0.to_span(), info.to_span());
    (void)syev<B, T>(*this->ctx, A1, W1.to_span(), JobType::NoEigenVectors, Uplo::Lower,
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
// syev pins its CTA STEQR parameters (max_sweeps = 400) for the small-n
// kernels and default-constructed StedcParams for Blocked and TwoStage
// (src/ops/syev/syev.cc cta_steqr_params, launch), and SyevOptions carries only jobz
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
        fill_mixed_convergence_matrices<T>(A);
        UnifiedVector<Real> W(static_cast<size_t>(n) * static_cast<size_t>(batch));
        UnifiedVector<int32_t> info(batch, int32_t(-1));

        SteqrParams<T> params;
        params.max_sweeps = 1;

        auto ws = UnifiedVector<std::byte>(
            syev_cta_buffer_size<B, T>(*this->ctx, A, JobType::NoEigenVectors, params));
        (void)syev_cta<B, T>(*this->ctx, A, W.to_span(), JobType::NoEigenVectors, Uplo::Lower,
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
        expect_converged_items_exact<T>(W, info, n, batch);
    }
}
#endif  // BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND

// ---------------------------------------------------------------------------
// The vendor route on non-natural accessors (ld > n, sub-views, padded batch
// stride), the layouts LOBPCG hands it; the wrapper was suspected of an ld bug and cleared.
// evidence: docs/design/known-defects.md#lobpcg-picked-a-spurious-zero-ritz-value-from-a-null-p-column
// ---------------------------------------------------------------------------

namespace {

struct VendorSyevShape {
    int n, ld, stride, batch;
};

// Graded, sign-mixed spectrum: |lambda| from 1 to 1e3.
double vendor_syev_lambda(int i, int n) {
    const double mag = std::pow(10.0, 3.0 * double(i) / double(std::max(1, n - 1)));
    return (i % 2) ? -mag : mag;
}

// Item b is Q_b diag(lambda) Q_b^H with a random Householder Q_b (complex for complex T).
template <typename T>
std::vector<std::complex<double>> vendor_syev_item(int n, int b) {
    std::mt19937 gen(1234u + 7u * unsigned(b));
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    std::vector<std::complex<double>> u(n);
    double unorm2 = 0.0;
    for (auto& x : u) {
        x = {dis(gen), is_std_complex_v<T> ? dis(gen) : 0.0};
        unorm2 += std::norm(x);
    }
    std::vector<std::complex<double>> Q(size_t(n) * n), A(size_t(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Q[i + size_t(j) * n] = (i == j ? 1.0 : 0.0) - 2.0 * u[i] * std::conj(u[j]) / unorm2;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            std::complex<double> acc = 0.0;
            for (int k = 0; k < n; ++k)
                acc += Q[i + size_t(k) * n] * vendor_syev_lambda(k, n) * std::conj(Q[j + size_t(k) * n]);
            A[i + size_t(j) * n] = acc;
        }
    return A;
}

template <typename T>
T vendor_syev_cast(std::complex<double> z) {
    if constexpr (is_std_complex_v<T>) return T(z.real(), z.imag());
    else return T(z.real());
}

}  // namespace

TYPED_TEST(SyevTest, VendorRouteHonoursLdSubViewsAndBatchStride) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    if constexpr (!batchlas::select::solver_vendor_available<TestFixture::BackendType>) {
        GTEST_SKIP() << "no vendor solver in this build";
    } else {
        const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", "vendor");
        using MV = MatrixView<T, MatrixFormat::Dense>;
        const T poison = vendor_syev_cast<T>({777.0, is_std_complex_v<T> ? 333.0 : 0.0});
        // natural packed; packed with ld > n; LOBPCG's 12x12 corner of an 18x18 slab;
        // ld > n with a padded batch stride; batch 1 with ld > n.
        const VendorSyevShape shapes[] = {
            {18, 18, 18 * 18, 2}, {12, 15, 15 * 12, 3}, {12, 18, 18 * 18, 2}, {9, 12, 12 * 9 + 7, 3}, {7, 10, 70, 1}};
        for (const auto& s : shapes) {
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                const std::string where = "n=" + std::to_string(s.n) + " ld=" + std::to_string(s.ld) +
                                          " stride=" + std::to_string(s.stride) + " batch=" +
                                          std::to_string(s.batch) + (jobz == JobType::EigenVectors ? " V" : " N");
                UnifiedVector<T> buf(size_t(s.stride) * s.batch, poison);
                Matrix<T, MatrixFormat::Dense> A0(s.n, s.n, s.batch);
                for (int b = 0; b < s.batch; ++b) {
                    const auto ref = vendor_syev_item<T>(s.n, b);
                    for (int j = 0; j < s.n; ++j)
                        for (int i = 0; i < s.n; ++i) {
                            const T v = vendor_syev_cast<T>(ref[i + size_t(j) * s.n]);
                            buf[size_t(b) * s.stride + i + size_t(j) * s.ld] = v;
                            A0.view()(i, j, b) = v;
                        }
                }
                MatrixView<T, MatrixFormat::Dense> A(buf.data(), s.n, s.n, s.ld, s.stride, s.batch);
                UnifiedVector<Real> W(size_t(s.n) * s.batch, Real(-1));
                UnifiedVector<std::byte> ws(syev_buffer_size(*this->ctx, A, W.to_span(), jobz, Uplo::Lower));
                (void)syev(*this->ctx, A, W.to_span(), SyevOptions{.jobz = jobz}, ws.to_span());
                this->ctx->wait();

                std::vector<double> want(s.n);
                for (int i = 0; i < s.n; ++i) want[i] = vendor_syev_lambda(i, s.n);
                std::sort(want.begin(), want.end());
                for (int b = 0; b < s.batch; ++b) {
                    const T* item = buf.data() + size_t(b) * s.stride;
                    for (int o = 0; o < s.stride; ++o) {
                        if (o % s.ld < s.n && o / s.ld < s.n) continue;
                        ASSERT_EQ(item[o], poison) << where << ": item " << b << " padding offset " << o << " written";
                    }
                }
                // ||A||_2 = max |lambda| = 1e3.
                const VectorView<Real> w(W.data(), s.n, s.batch);
                EXPECT_VERIFY(T, verify::Check::values, s.n,
                              MV::values_error(w, std::vector<std::vector<double>>(s.batch, want), 1e3, A0.all_items()))
                    << where;
                if (jobz != JobType::EigenVectors) continue;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, s.n,
                              MV::eigen_residual(*this->ctx, A0, A, w, A0.all_items()))
                    << where;
                EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, s.n,
                              MV::orthogonality(*this->ctx, A, A0.all_items()))
                    << where;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Defect 14 (the entry mirrors the named triangle) and repeatability under a pinned route.
// evidence: docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle
// ---------------------------------------------------------------------------

namespace {

// Item b is H diag(d) H^H with H = I - 2 u u^H, |u| = 1 (complex u for complex T) and a graded,
// sign-mixed d = +-10^[-2, 2]: entry (i, j) = d_i [i = j] + 2 (2 s - d_i - d_j) u_i conj(u_j), with
// s = sum d_k |u_k|^2, so the batch costs O(n^2). `buf` (ld = n + 3, stride = ld n + 5) gets the
// `uplo` triangle and `poison` in the other one; `full` gets the whole matrix. Returns sorted d.
template <typename T>
std::vector<std::vector<double>> fill_graded_hermitian(T* buf, int ld, int stride, Uplo uplo, T poison,
                                                       MatrixView<T, MatrixFormat::Dense> full) {
    const int n = full.rows();
    std::vector<std::vector<double>> lambda(full.batch_size());
    for (int b = 0; b < full.batch_size(); ++b) {
        std::mt19937 gen(91u + 13u * unsigned(b) + unsigned(n));
        std::uniform_real_distribution<double> dis(-1.0, 1.0);
        std::vector<std::complex<double>> u(n);
        double nrm = 0.0;
        for (auto& x : u) {
            x = {dis(gen), is_std_complex_v<T> ? dis(gen) : 0.0};
            nrm += std::norm(x);
        }
        std::vector<double> d(n);
        double s = 0.0;
        for (int i = 0; i < n; ++i) {
            u[i] /= std::sqrt(nrm);
            d[i] = (i % 2 ? -1.0 : 1.0) * std::pow(10.0, n == 1 ? 0.0 : -2.0 + 4.0 * i / (n - 1)) * (1.0 + 0.01 * b);
            s += d[i] * std::norm(u[i]);
        }
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                std::complex<double> a = 2.0 * (2.0 * s - (d[i] + d[j])) * u[i] * std::conj(u[j]);
                if (i == j) a = {a.real() + d[i], 0.0};
                const T v = vendor_syev_cast<T>(a);
                full(i, j, b) = v;
                const bool named = uplo == Uplo::Lower ? i >= j : i <= j;
                buf[size_t(b) * stride + i + size_t(j) * ld] = named ? v : poison;
            }
        std::sort(d.begin(), d.end());
        lambda[b] = d;
    }
    return lambda;
}

template <typename T>
T unreferenced_poison() {
    return vendor_syev_cast<T>({-9.0e3, is_std_complex_v<T> ? 7.0e3 : 0.0});
}

}  // namespace

// Large finite poison (90 ||A||_2) in the triangle the caller did not name, under Auto, on a strided
// batch. Auto reaches the small solvers at n <= 32 and blocked / two_stage above (by jobz and dtype);
// before the entry mirror several of them read the other triangle.
TYPED_TEST(SyevTest, AutoIgnoresUnreferencedTriangle) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    const int batch = 2;
    for (const int n : {5, 32, 40, 400, 1024})
        for (const Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                const std::string where = "n=" + std::to_string(n) + (uplo == Uplo::Lower ? " L" : " U") +
                                          (jobz == JobType::EigenVectors ? " V" : " N");
                const int ld = n + 3, stride = ld * n + 5;
                UnifiedVector<T> buf(size_t(stride) * batch, T(0));
                Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
                const auto lambda = fill_graded_hermitian<T>(buf.data(), ld, stride, uplo, unreferenced_poison<T>(), A0);
                MV A(buf.data(), n, n, ld, stride, batch);
                UnifiedVector<Real> W(size_t(n) * batch, Real(0));
                UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(*this->ctx, A, W.to_span(), jobz, uplo));
                (void)syev<B, T>(*this->ctx, A, W.to_span(), jobz, uplo, ws.to_span(), Span<int32_t>{});
                this->ctx->wait();

                double scale = 0.0;
                for (const auto& l : lambda) scale = std::max({scale, std::fabs(l.front()), std::fabs(l.back())});
                const VectorView<Real> w(W.data(), n, batch);
                EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(w, lambda, scale, A0.all_items())) << where;
                if (jobz != JobType::EigenVectors) continue;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, A0, A, w, A0.all_items()))
                    << where;
                EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV::orthogonality(*this->ctx, A, A0.all_items()))
                    << where;
            }
}

// Same device, same route, same build: two solves of copies of one batch agree bit for bit (W and
// the whole buffer: eigenvectors, the other triangle and the padding). cta's batch of 1024 fills
// the device with sub-group partitions.
TYPED_TEST(SyevTest, PinnedRouteIsBitwiseRepeatable) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "the pinned native families are GPU kernels";
    } else {
        struct Case {
            const char* route;
            int n, batch;
        };
        const Case cases[] = {{"cta", 17, 64},      {"cta", 32, 1024},     {"blocked", 32, 64},
                              {"blocked", 400, 64}, {"two_stage", 40, 64}, {"two_stage", 400, 64}};
        for (const Case& c : cases) {
            if (B == Backend::ROCM && std::string(c.route) == "cta") continue;
            const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", c.route);
            for (const Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                    const std::string where = std::string(c.route) + " n=" + std::to_string(c.n) +
                                              (uplo == Uplo::Lower ? " L" : " U") +
                                              (jobz == JobType::EigenVectors ? " V" : " N");
                    const int ld = c.n + 3, stride = ld * c.n + 5;
                    Matrix<T, MatrixFormat::Dense> A0(c.n, c.n, c.batch);
                    UnifiedVector<T> buf[2] = {UnifiedVector<T>(size_t(stride) * c.batch, T(0)),
                                               UnifiedVector<T>(size_t(stride) * c.batch, T(0))};
                    UnifiedVector<Real> W[2] = {UnifiedVector<Real>(size_t(c.n) * c.batch, Real(-1)),
                                                UnifiedVector<Real>(size_t(c.n) * c.batch, Real(-2))};
                    for (int r = 0; r < 2; ++r) {
                        (void)fill_graded_hermitian<T>(buf[r].data(), ld, stride, uplo, unreferenced_poison<T>(), A0);
                        MatrixView<T, MatrixFormat::Dense> A(buf[r].data(), c.n, c.n, ld, stride, c.batch);
                        UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(*this->ctx, A, W[r].to_span(), jobz, uplo));
                        (void)syev<B, T>(*this->ctx, A, W[r].to_span(), jobz, uplo, ws.to_span(), Span<int32_t>{});
                        this->ctx->wait();
                    }
                    EXPECT_EQ(0, std::memcmp(W[0].data(), W[1].data(), W[0].size() * sizeof(Real)))
                        << where << ": eigenvalues differ between two runs";
                    EXPECT_EQ(0, std::memcmp(buf[0].data(), buf[1].data(), buf[0].size() * sizeof(T)))
                        << where << ": eigenvectors or A differ between two runs";
                }
        }
    }
}
