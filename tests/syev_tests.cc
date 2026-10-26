// SPDX-License-Identifier: MIT
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
#include <batchlas/tuning_params.hh>
#include <src/select/vendor.hh>
#include <src/ops/syev/parts.hh>
#include <batchlas/no_route.hh>
#include <algorithm>
#include <cstring>
#include <complex>
#include <initializer_list>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <initializer_list>
#include <sstream>
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
    // A vendor-free build has no syev on the CPU: every native family is a GPU kernel, so a NETLIB
    // queue throws NoRouteError by design (docs/perf/syev.md#syev-vendor-free-double-band).
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        const std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        if (Config::BackendVal == Backend::NETLIB && !batchlas::select::kHasNetlib && name != "VendorFreeCpuHasNoRoute")
            GTEST_SKIP() << "vendor-free build: no syev route on the CPU";
    }
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
// max_sweeps = 1, not a "reduced" 50: syev_cta.cc:178-181 silently REWRITES
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

// ---- L2 batch chunking (docs/perf/syev.md#syev-l2-batch-chunking) ------------------------
//
// cta at n = 12 and blocked at n = 40, each pinned, with the chunk forced to 7 over 23 items
// (7 + 7 + 9: the last part takes the remainder), a padded ld and an odd stride. Item b is
// (1 + b/8) times a graded Householder similarity, so every item has its own spectrum and an
// output sub-range at the wrong offset moves only the items from the second part on.

namespace {

struct ChunkRoute {
    const char* route;
    int n;
};
constexpr ChunkRoute kChunkRoutes[] = {{"cta", 12}, {"blocked", 40}};

double chunk_item_scale(int b) { return 1.0 + 0.125 * b; }

// Writes item b of the chunk batch into buf (ld, stride) and into the natural-layout A0.
template <typename T>
void chunk_fill(UnifiedVector<T>& buf, Matrix<T, MatrixFormat::Dense>& A0, int n, int ld, int stride, int b, int src) {
    const auto ref = vendor_syev_item<T>(n, src);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const T v = vendor_syev_cast<T>(ref[i + size_t(j) * n] * chunk_item_scale(src));
            buf[size_t(b) * stride + i + size_t(j) * ld] = v;
            A0.view()(i, j, b) = v;
        }
}

}  // namespace

TYPED_TEST(SyevTest, ChunksKeepEigenpairsPerItem) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "cta and blocked are GPU drivers";
    } else {
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "7");
        const int batch = 23;
        ASSERT_EQ(tuning::blocked_chunk_items(batch, 1, 1, 1), 7) << "the override did not reach the accessor";
        const T poison = vendor_syev_cast<T>({777.0, is_std_complex_v<T> ? 333.0 : 0.0});
        for (const auto& r : kChunkRoutes) {
            const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", r.route);
            const int n = r.n, ld = n + 5, stride = ld * n + 11;
            for (const Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                    const std::string where = std::string(r.route) + (uplo == Uplo::Lower ? " L" : " U") +
                                              (jobz == JobType::EigenVectors ? " V" : " N");
                    UnifiedVector<T> buf(size_t(stride) * batch, poison);
                    Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
                    for (int b = 0; b < batch; ++b) chunk_fill<T>(buf, A0, n, ld, stride, b, b);
                    MV A(buf.data(), n, n, ld, stride, batch);
                    UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
                    UnifiedVector<int32_t> info(batch, int32_t(-1));
                    UnifiedVector<std::byte> ws(syev_buffer_size<TestFixture::BackendType, T>(*this->ctx, A, W.to_span(), jobz, uplo));
                    (void)syev<TestFixture::BackendType, T>(*this->ctx, A, W.to_span(), jobz, uplo, ws.to_span(),
                                                            info.to_span());
                    this->ctx->wait();

                    const VectorView<Real> w(W.data(), n, batch);
                    double worst = 0;
                    for (int b = 0; b < batch; ++b) {
                        EXPECT_EQ(info[b], 0) << where << ": item " << b << " (part " << std::min(b / 7, 2) << ")";
                        std::vector<std::vector<double>> want(batch);
                        for (int i = 0; i < n; ++i) want[b].push_back(vendor_syev_lambda(i, n) * chunk_item_scale(b));
                        std::sort(want[b].begin(), want[b].end());
                        const int item[] = {b};
                        worst = verify::nanmax(worst, MV::values_error(w, want, 1e3 * chunk_item_scale(b), item));
                    }
                    EXPECT_VERIFY(T, verify::Check::values, n, worst) << where;
                    if (jobz != JobType::EigenVectors) continue;
                    EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, A0, A, w, A0.all_items()))
                        << where;
                    EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV::orthogonality(*this->ctx, A, A0.all_items()))
                        << where;
                    for (int b = 0; b < batch; ++b)
                        for (int o = 0; o < stride; ++o)
                            if (!(o % ld < n && o / ld < n))
                                ASSERT_EQ(buf[size_t(b) * stride + o], poison) << where << ": item " << b << " padding " << o;
                }
            }
        }
    }
}

// info through the parts: syev_cta with a one-sweep budget (see InfoReportsItemsThatExhaustTheSweepBudget)
// on the mixed batch, chunked (7) against one pass. Per item, info, W and the vectors must be bit-identical
// to the one-pass run; info both requested and not.
TYPED_TEST(SyevTest, ChunksKeepCtaInfoPerItem) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "syev_cta requires sub-group 32";
    } else {
        const int n = 32, batch = 23, ld = n + 3, stride = ld * n + 7;
        SteqrParams<T> params;
        params.max_sweeps = 1;
        auto run = [&](const char* chunk_items, bool pass_info, std::vector<int32_t>& info_out, std::vector<Real>& w_out,
                       std::vector<T>& a_out) {
            const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", chunk_items);
            Matrix<T, MatrixFormat::Dense> M(n, n, batch);
            fill_mixed_convergence_matrices<T>(M);
            UnifiedVector<T> buf(size_t(stride) * batch, T(-3));
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) buf[size_t(b) * stride + i + size_t(j) * ld] = M.view()(i, j, b);
            MatrixView<T, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
            UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
            UnifiedVector<int32_t> info(batch, int32_t(-1));
            UnifiedVector<std::byte> ws(syev_cta_buffer_size<B, T>(*this->ctx, A, JobType::EigenVectors, params));
            (void)syev_cta<B, T>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), params, 1,
                                 pass_info ? info.to_span() : Span<int32_t>());
            this->ctx->wait();
            info_out.assign(info.begin(), info.end());
            w_out.assign(W.begin(), W.end());
            a_out.assign(buf.begin(), buf.end());
        };
        std::vector<int32_t> info1, infoc;
        std::vector<Real> w1, wc;
        std::vector<T> a1, ac;
        run("100000000", true, info1, w1, a1);
        int reported = 0;
        for (int b = 0; b < batch; ++b) reported += info1[b] != 0;
        ASSERT_GT(reported, 0) << "the one-sweep budget reported universal convergence: nothing to carry";
        for (bool pass_info : {true, false}) {
            run("7", pass_info, infoc, wc, ac);
            for (int b = 0; b < batch; ++b) {
                const std::string where = "item " + std::to_string(b) + " (part " + std::to_string(std::min(b / 7, 2)) +
                                          (pass_info ? ")" : ", info not requested)");
                if (pass_info) EXPECT_EQ(infoc[b], info1[b]) << where;
                else EXPECT_EQ(infoc[b], -1) << where << ": an unrequested info was written";
                EXPECT_EQ(std::memcmp(wc.data() + size_t(b) * n, w1.data() + size_t(b) * n, sizeof(Real) * n), 0) << where;
                EXPECT_EQ(std::memcmp(ac.data() + size_t(b) * stride, a1.data() + size_t(b) * stride, sizeof(T) * stride), 0)
                    << where;
            }
        }
    }
}

// SATURATING: 1024 copies of one matrix in four chunks of 256, per route; every item's eigenvalues
// and vectors must be bit-identical to item 0's, so parts that overlap or skip items show.
TYPED_TEST(SyevTest, ChunksAtSaturatingBatchAreBitIdentical) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "cta and blocked are GPU drivers";
    } else {
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "256");
        const int batch = 1024;
        for (const auto& r : kChunkRoutes) {
            const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", r.route);
            const int n = r.n, ld = n + 3, stride = ld * n + 5;
            UnifiedVector<T> buf(size_t(stride) * batch, T(-11));
            Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
            for (int b = 0; b < batch; ++b) chunk_fill<T>(buf, A0, n, ld, stride, b, 5);
            MV A(buf.data(), n, n, ld, stride, batch);
            UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
            UnifiedVector<std::byte> ws(syev_buffer_size(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower));
            (void)syev(*this->ctx, A, W.to_span(), SyevOptions{}, ws.to_span());
            this->ctx->wait();
            int wrong = 0, first = -1;
            for (int b = 1; b < batch; ++b) {
                bool same = std::memcmp(W.data() + size_t(b) * n, W.data(), sizeof(Real) * n) == 0;
                for (int j = 0; j < n && same; ++j)
                    same = std::memcmp(buf.data() + size_t(b) * stride + size_t(j) * ld, buf.data() + size_t(j) * ld,
                                       sizeof(T) * n) == 0;
                if (!same && wrong++ == 0) first = b;
            }
            EXPECT_EQ(wrong, 0) << r.route << ": items differing from item 0; first at " << first;
            std::vector<std::vector<double>> want(1);
            for (int i = 0; i < n; ++i) want[0].push_back(vendor_syev_lambda(i, n) * chunk_item_scale(5));
            std::sort(want[0].begin(), want[0].end());
            const int item0[] = {0};
            EXPECT_VERIFY(T, verify::Check::values, n,
                          MV::values_error(VectorView<Real>(W.data(), n, batch), want, 1e3 * chunk_item_scale(5), item0))
                << r.route;
            EXPECT_VERIFY(T, verify::Check::eigen_residual, n,
                          MV::eigen_residual(*this->ctx, A0, A, VectorView<Real>(W.data(), n, batch), item0))
                << r.route;
        }
    }
}

// The public entry point on an out-of-order Queue: every part shares the one part-sized workspace,
// so a part that starts before the previous one finishes overwrites its scratch. Eight reps per route.
TYPED_TEST(SyevTest, ChunksOnAnOutOfOrderQueue) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "cta and blocked are GPU drivers";
    } else {
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "7");
        const int batch = 23;
        Queue ooo(*this->ctx, /*in_order=*/false);
        ASSERT_FALSE(ooo.in_order());
        for (const auto& r : kChunkRoutes) {
            const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", r.route);
            const int n = r.n, ld = n + 2, stride = ld * n + 3;
            for (int rep = 0; rep < 8; ++rep) {
                UnifiedVector<T> buf(size_t(stride) * batch, T(-7));
                Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
                for (int b = 0; b < batch; ++b) chunk_fill<T>(buf, A0, n, ld, stride, b, b);
                MV A(buf.data(), n, n, ld, stride, batch);
                UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
                UnifiedVector<std::byte> ws(syev_buffer_size(ooo, A, W.to_span(), JobType::EigenVectors, Uplo::Lower));
                (void)syev(ooo, A, W.to_span(), SyevOptions{}, ws.to_span());
                ooo.wait();
                const VectorView<Real> w(W.data(), n, batch);
                double worst = 0;
                for (int b = 0; b < batch; ++b) {
                    std::vector<std::vector<double>> want(batch);
                    for (int i = 0; i < n; ++i) want[b].push_back(vendor_syev_lambda(i, n) * chunk_item_scale(b));
                    std::sort(want[b].begin(), want[b].end());
                    const int item[] = {b};
                    worst = verify::nanmax(worst, MV::values_error(w, want, 1e3 * chunk_item_scale(b), item));
                }
                const std::string where = std::string(r.route) + " rep " + std::to_string(rep);
                EXPECT_VERIFY(T, verify::Check::values, n, worst) << where;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, A0, A, w, A0.all_items()))
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

// A graded Hermitian batch from the library fill (lambda_i = 10^(-6 i / (n - 1)), a different Q per
// item, deterministic in the seed) in A's strided storage, copied whole into `full`. Then a large
// finite poison (9e3 against ||A||_2 = 1) overwrites the triangle `uplo` does not name.
template <typename T>
std::vector<std::vector<double>> fill_poisoned(Queue& ctx, MatrixView<T, MatrixFormat::Dense> A, Uplo uplo,
                                               const MatrixView<T, MatrixFormat::Dense>& full) {
    const int n = A.rows();
    A.fill_graded_hermitian(ctx, 6.0, 91u + unsigned(n)).wait();
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, full, A).wait();
    const T poison = vendor_syev_cast<T>({-9.0e3, is_std_complex_v<T> ? 7.0e3 : 0.0});
    for (int b = 0; b < A.batch_size(); ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                if (uplo == Uplo::Lower ? i < j : i > j) A(i, j, b) = poison;
    std::vector<double> lam(n);
    for (int i = 0; i < n; ++i) lam[i] = std::pow(10.0, n == 1 ? 0.0 : -6.0 * i / (n - 1));
    std::sort(lam.begin(), lam.end());
    return std::vector<std::vector<double>>(A.batch_size(), lam);
}

}  // namespace

namespace {

// Poison in the triangle the caller did not name, under Auto, on a strided batch (ld = n + 3,
// stride = ld n + 5), for every n in `ns`, both uplo and both jobz.
template <typename T, Backend B>
void expect_auto_ignores_other_triangle(Queue& ctx, std::initializer_list<int> ns) {
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const int batch = 2;
    for (const int n : ns)
        for (const Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                const std::string where = "n=" + std::to_string(n) + (uplo == Uplo::Lower ? " L" : " U") +
                                          (jobz == JobType::EigenVectors ? " V" : " N");
                const int ld = n + 3, stride = ld * n + 5;
                UnifiedVector<T> buf(size_t(stride) * batch, T(0));
                UnifiedVector<T*> ptrs(batch, nullptr);
                MV A(buf.data(), n, n, ld, stride, batch, ptrs.data());
                Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
                const auto lambda = fill_poisoned<T>(ctx, A, uplo, A0);
                UnifiedVector<Real> W(size_t(n) * batch, Real(0));
                UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(ctx, A, W.to_span(), jobz, uplo));
                (void)syev<B, T>(ctx, A, W.to_span(), jobz, uplo, ws.to_span(), Span<int32_t>{});
                ctx.wait();

                const VectorView<Real> w(W.data(), n, batch);
                EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(w, lambda, 1.0, A0.all_items())) << where;
                if (jobz != JobType::EigenVectors) continue;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(ctx, A0, A, w, A0.all_items()))
                    << where;
                EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, A, A0.all_items()))
                    << where;
            }
}

struct RepeatCase {
    const char* route;
    int n, batch;
};

// Same device, same route, same build: two solves of identical copies of one batch agree bit for bit
// (W and the whole buffer: eigenvectors, the other triangle and the padding).
template <typename T, Backend B>
void expect_pinned_route_repeatable(Queue& ctx, std::initializer_list<RepeatCase> cases) {
    using Real = typename base_type<T>::type;
    for (const RepeatCase& c : cases) {
        if (B == Backend::ROCM && std::string(c.route) == "cta") continue;
        const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", c.route);
        for (const Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                const std::string where = std::string(c.route) + " n=" + std::to_string(c.n) +
                                          " batch=" + std::to_string(c.batch) + (uplo == Uplo::Lower ? " L" : " U") +
                                          (jobz == JobType::EigenVectors ? " V" : " N");
                const int ld = c.n + 3, stride = ld * c.n + 5;
                Matrix<T, MatrixFormat::Dense> A0(c.n, c.n, c.batch);
                UnifiedVector<T> buf[2] = {UnifiedVector<T>(size_t(stride) * c.batch, T(0)),
                                           UnifiedVector<T>(size_t(stride) * c.batch, T(0))};
                UnifiedVector<Real> W[2] = {UnifiedVector<Real>(size_t(c.n) * c.batch, Real(-1)),
                                            UnifiedVector<Real>(size_t(c.n) * c.batch, Real(-2))};
                UnifiedVector<T*> ptrs[2] = {UnifiedVector<T*>(c.batch, nullptr), UnifiedVector<T*>(c.batch, nullptr)};
                for (int r = 0; r < 2; ++r) {
                    MatrixView<T, MatrixFormat::Dense> A(buf[r].data(), c.n, c.n, ld, stride, c.batch, ptrs[r].data());
                    (void)fill_poisoned<T>(ctx, A, uplo, A0);
                    if (r == 1)
                        ASSERT_EQ(0, std::memcmp(buf[0].data(), buf[1].data(), buf[0].size() * sizeof(T)))
                            << where << ": the inputs differ";
                }
                for (int r = 0; r < 2; ++r) {
                    MatrixView<T, MatrixFormat::Dense> A(buf[r].data(), c.n, c.n, ld, stride, c.batch, ptrs[r].data());
                    UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(ctx, A, W[r].to_span(), jobz, uplo));
                    (void)syev<B, T>(ctx, A, W[r].to_span(), jobz, uplo, ws.to_span(), Span<int32_t>{});
                    ctx.wait();
                }
                EXPECT_EQ(0, std::memcmp(W[0].data(), W[1].data(), W[0].size() * sizeof(Real)))
                    << where << ": eigenvalues differ between two runs";
                EXPECT_EQ(0, std::memcmp(buf[0].data(), buf[1].data(), buf[0].size() * sizeof(T)))
                    << where << ": eigenvectors or A differ between two runs";
            }
    }
}

}  // namespace

// Without the entry mirror this catches only the complex types (Auto sends them to cta, which reads
// the lower triangle for Upper). Real types reach no reading route at n <= 40; their guards are the
// heavy AutoReadsOnlyTheNamedTriangleAtLargeN and SyevCandidates.OtherTriangleIsNeverRead.
TYPED_TEST(SyevTest, AutoIgnoresUnreferencedTriangle) {
    if (TestFixture::BackendType == Backend::NETLIB && !BATCHLAS_HAS_LAPACKE)
        GTEST_SKIP() << "no LAPACKE: NETLIB syev has no route in this build";
    expect_auto_ignores_other_triangle<typename TestFixture::ScalarType, TestFixture::BackendType>(*this->ctx,
                                                                                                   {5, 32, 40});
}

// Heavy tier (*AutoReads*): two_stage is Auto's choice for jobz=N here and for float V at n = 1024.
// NETLIB is skipped: its route is reference LAPACK, which reads only `uplo` and is never mirrored,
// and its n = 1024 solves cost ~100 s per dtype.
TYPED_TEST(SyevTest, AutoReadsOnlyTheNamedTriangleAtLargeN) {
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "reference LAPACK reads only uplo; covered at small n by AutoIgnoresUnreferencedTriangle";
    } else {
        expect_auto_ignores_other_triangle<typename TestFixture::ScalarType, TestFixture::BackendType>(*this->ctx,
                                                                                                       {400, 1024});
    }
}

TYPED_TEST(SyevTest, PinnedRouteIsBitwiseRepeatable) {
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "the pinned native families are GPU kernels";
    } else {
        expect_pinned_route_repeatable<typename TestFixture::ScalarType, TestFixture::BackendType>(
            *this->ctx, {{"cta", 17, 64}, {"blocked", 32, 64}, {"two_stage", 40, 64}});
    }
}

// Heavy tier (*Saturating*): cta's batch of 1024 fills the device with sub-group partitions, and
// the n = 400 solves are the slow half of the cases.
TYPED_TEST(SyevTest, PinnedRouteIsBitwiseRepeatableAtSaturatingBatch) {
    if constexpr (TestFixture::BackendType == Backend::NETLIB) {
        GTEST_SKIP() << "the pinned native families are GPU kernels";
    } else {
        expect_pinned_route_repeatable<typename TestFixture::ScalarType, TestFixture::BackendType>(
            *this->ctx, {{"cta", 32, 1024}, {"blocked", 400, 64}, {"two_stage", 400, 64}});
    }
}

// ---------------------------------------------------------------------------
// P2.8: double and complex<double> for 449 <= n <= 1600 (the fullerene band: C120 n = 480 at
// batch ~1819, C300 n = 1200 at batch ~76). A vendor-free build must serve it natively.
// evidence: docs/perf/syev.md#syev-vendor-free-double-band
// ---------------------------------------------------------------------------

namespace {

template <typename T>
constexpr bool kDoubleScalar = std::is_same_v<typename base_type<T>::type, double>;

// The choice the outermost syev select-trace line names, plus " (last resort)" when no row served it.
template <typename F>
std::string syev_route_of(F&& run) {
    const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
    ::testing::internal::CaptureStderr();
    try {
        run();
    } catch (...) {
        (void)::testing::internal::GetCapturedStderr();
        throw;
    }
    std::istringstream in(::testing::internal::GetCapturedStderr());
    for (std::string line; std::getline(in, line);) {
        const auto arrow = line.find(" -> ");
        if (line.rfind("syev ", 0) != 0 || arrow == std::string::npos) continue;
        const std::string tail = line.substr(arrow + 4);
        return tail.substr(0, tail.find(' ')) + (line.find("last resort") != std::string::npos ? " (last resort)" : "");
    }
    return "<no syev trace line>";
}

template <Backend B, typename T, typename Real = typename base_type<T>::type>
std::string traced_syev(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, UnifiedVector<Real>& W, JobType jobz,
                        Uplo uplo) {
    return syev_route_of([&] {
        UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(ctx, A, W.to_span(), jobz, uplo));
        (void)syev<B, T>(ctx, A, W.to_span(), jobz, uplo, ws.to_span(), Span<int32_t>{});
        ctx.wait();
    });
}

// lambda_i = 10^(-6 i / (n - 1)), the spectrum fill_graded_hermitian(6.0) builds, sorted.
std::vector<double> graded_lambda(int n) {
    std::vector<double> lam(n);
    for (int i = 0; i < n; ++i) lam[i] = std::pow(10.0, -6.0 * i / (n - 1));
    std::sort(lam.begin(), lam.end());
    return lam;
}

// One solve of a poisoned graded batch (fill_poisoned: condition 1e6, ld = n + 3, stride = ld n + 5),
// checked through verify. Returns the route the trace names.
template <Backend B, typename T>
std::string solve_graded_band(Queue& ctx, int n, int batch, Uplo uplo, JobType jobz, const std::string& where) {
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const int ld = n + 3, stride = ld * n + 5;
    UnifiedVector<T> buf(size_t(stride) * batch, T(0));
    UnifiedVector<T*> ptrs(batch, nullptr);
    MV A(buf.data(), n, n, ld, stride, batch, ptrs.data());
    Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
    const auto lambda = fill_poisoned<T>(ctx, A, uplo, A0);
    UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
    const std::string route = traced_syev<B, T>(ctx, A, W, jobz, uplo);
    const VectorView<Real> w(W.data(), n, batch);
    const std::string tag = where + " " + route;
    EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(w, lambda, 1.0, A0.all_items())) << tag;
    if (jobz == JobType::EigenVectors) {
        EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(ctx, A0, A, w, A0.all_items())) << tag;
        EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, A, A0.all_items())) << tag;
    }
    return route;
}

std::string band_where(int n, Uplo uplo, JobType jobz) {
    return " n=" + std::to_string(n) + (uplo == Uplo::Lower ? " L" : " U") + (jobz == JobType::EigenVectors ? " V" : " N");
}

struct BandCase {
    int n;
    Uplo uplo;
    JobType jobz;
};

std::vector<BandCase> band_grid(std::initializer_list<int> orders) {
    std::vector<BandCase> cases;
    for (const int n : orders)
        for (const Uplo uplo : {Uplo::Lower, Uplo::Upper})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) cases.push_back({n, uplo, jobz});
    return cases;
}

// Auto: vendor-free, a GPU row must name a runnable native family (never the last resort); with the
// library the GPU row is served as transcribed and the CPU takes LAPACKE.
template <Backend B, typename T>
void expect_band_auto(Queue& ctx, const std::vector<BandCase>& cases) {
    constexpr bool kVendor = batchlas::select::solver_vendor_available<B>;
    for (const BandCase& c : cases) {
        const std::string where = band_where(c.n, c.uplo, c.jobz);
        const std::string route = solve_graded_band<B, T>(ctx, c.n, 2, c.uplo, c.jobz, where);
        if (B == Backend::NETLIB) {
            EXPECT_EQ(route.rfind("vendor", 0), 0u) << where << ": " << route;
        } else {
            EXPECT_EQ(route.find("last resort"), std::string::npos) << where << ": " << route;
            if (!kVendor) EXPECT_TRUE(route == "blocked" || route == "two_stage") << where << ": " << route;
        }
    }
}

// Each native family pinned, so both are exercised whatever the table ranks first.
template <Backend B, typename T>
void expect_band_pinned(Queue& ctx, const std::vector<BandCase>& cases) {
    for (const char* family : {"blocked", "two_stage"}) {
        const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", family);
        for (const BandCase& c : cases) {
            const std::string where = family + band_where(c.n, c.uplo, c.jobz);
            EXPECT_EQ((solve_graded_band<B, T>(ctx, c.n, 2, c.uplo, c.jobz, where)), family) << where;
        }
    }
}

}  // namespace

// Quick tier: three Auto solves across the band (the CPU: n = 449 only, its fill and LAPACKE are
// slow there) and every pinned (family, uplo, jobz) at n = 449.
TYPED_TEST(SyevTest, DoubleBandAutoIsServedByARow) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!kDoubleScalar<T>) {
        GTEST_SKIP() << "the band is a double / complex<double> requirement";
    } else if constexpr (B == Backend::NETLIB) {
        expect_band_auto<B, T>(*this->ctx, {{449, Uplo::Lower, JobType::EigenVectors}});
    } else {
        expect_band_auto<B, T>(*this->ctx, {{449, Uplo::Lower, JobType::EigenVectors},
                                            {480, Uplo::Upper, JobType::EigenVectors},
                                            {1200, Uplo::Upper, JobType::NoEigenVectors}});
    }
}

TYPED_TEST(SyevTest, DoubleBandPinnedNativeFamilies) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!kDoubleScalar<T> || B == Backend::NETLIB) {
        GTEST_SKIP() << "the native families are GPU kernels; the band is double / complex<double>";
    } else {
        expect_band_pinned<B, T>(*this->ctx, band_grid({449}));
    }
}

// Heavy tier (*DoubleBandEvery*): Auto at every band order with both uplo and jobz, and both
// families pinned at n = 1200.
TYPED_TEST(SyevTest, DoubleBandEveryOrderUploAndJobz) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!kDoubleScalar<T> || B == Backend::NETLIB) {
        GTEST_SKIP() << "the band is a double / complex<double> GPU requirement (LAPACKE: the quick case)";
    } else {
        expect_band_auto<B, T>(*this->ctx, band_grid({449, 480, 1024, 1200, 1600}));
        expect_band_pinned<B, T>(*this->ctx, band_grid({1200}));
    }
}

// Saturating batch (rule 5): 128 copies of one graded n = 480 item through each pinned native family;
// every item's W and V is bit-identical to item 0's, and item 0 verifies.
TYPED_TEST(SyevTest, DoubleBandSaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!kDoubleScalar<T> || B == Backend::NETLIB) {
        GTEST_SKIP() << "the native families are GPU kernels; the band is double / complex<double>";
    } else {
        const int n = 480, batch = 128, ld = n + 3, stride = ld * n + 5;
        const std::vector<std::vector<double>> lam{graded_lambda(n)};
        for (const char* family : {"blocked", "two_stage"})
            for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
                const std::string where = family + band_where(n, Uplo::Upper, jobz);
                const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", family);
                UnifiedVector<T> buf(size_t(stride) * batch, T(0));
                UnifiedVector<T*> ptrs(batch, nullptr);  // the vendor children (full build) read data_ptrs
                MV A(buf.data(), n, n, ld, stride, batch, ptrs.data());
                Matrix<T, MatrixFormat::Dense> A0(n, n, 1);
                A.batch_range(0, 1).fill_graded_hermitian(*this->ctx, 6.0, 4801u).wait();
                for (int b = 1; b < batch; ++b) (void)MV::copy(*this->ctx, A.batch_range(b, 1), A.batch_range(0, 1));
                (void)MV::copy(*this->ctx, A0, A.batch_range(0, 1));
                this->ctx->wait();
                UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
                ASSERT_EQ((traced_syev<B, T>(*this->ctx, A, W, jobz, Uplo::Upper)), family) << where;
                const VectorView<Real> w0(W.data(), n, 1);
                EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(w0, lam, 1.0, A0.all_items())) << where;
                if (jobz == JobType::EigenVectors)
                    EXPECT_VERIFY(T, verify::Check::eigen_residual, n,
                                  MV::eigen_residual(*this->ctx, A0, A.batch_range(0, 1), w0, A0.all_items()))
                        << where;
                int differ = 0;
                for (int b = 1; b < batch; ++b) {
                    const bool w_same = std::memcmp(&W[size_t(b) * n], &W[0], size_t(n) * sizeof(Real)) == 0;
                    bool v_same = true;
                    if (jobz == JobType::EigenVectors)
                        for (int j = 0; j < n && v_same; ++j)
                            v_same = std::memcmp(&buf[size_t(b) * stride + size_t(j) * ld], &buf[size_t(j) * ld],
                                                 size_t(n) * sizeof(T)) == 0;
                    differ += !(w_same && v_same);
                }
                EXPECT_EQ(differ, 0) << where << ": items differ from item 0";
            }
    }
}

// The two fullerene workloads at their own batch under Auto (double only: the Hamiltonians are real),
// jobz = V, Lower: the route sizes its workspace for the whole call and solves. First and last items verify.
TYPED_TEST(SyevTest, DoubleBandSaturatingFullereneShapes) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!std::is_same_v<T, double> || B == Backend::NETLIB) {
        GTEST_SKIP() << "the fullerene workloads are real double on the GPU";
    } else {
        constexpr bool kVendor = batchlas::select::solver_vendor_available<B>;
        struct Shape {
            int n, batch;
        };
        for (const Shape s : {Shape{480, 1819}, Shape{1200, 76}}) {
            const int n = s.n, batch = s.batch;
            const std::string where = "n=" + std::to_string(n) + " batch=" + std::to_string(batch);
            const std::vector<std::vector<double>> lam{graded_lambda(n)};
            Matrix<T, MatrixFormat::Dense> A(n, n, batch);
            A.view().fill_graded_hermitian(*this->ctx, 6.0, 1200u + unsigned(n)).wait();
            Matrix<T, MatrixFormat::Dense> first(n, n, 1), last(n, n, 1);
            (void)MV::copy(*this->ctx, first, A.view().batch_range(0, 1));
            (void)MV::copy(*this->ctx, last, A.view().batch_range(batch - 1, 1));
            this->ctx->wait();
            UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
            const std::size_t ws = syev_buffer_size<B, T>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower);
            ::testing::Test::RecordProperty("workspace_MiB_n" + std::to_string(n), std::to_string(ws >> 20));
            const std::string route = traced_syev<B, T>(*this->ctx, A.view(), W, JobType::EigenVectors, Uplo::Lower);
            EXPECT_EQ(route.find("last resort"), std::string::npos) << where << ": " << route;
            if (!kVendor) EXPECT_TRUE(route == "blocked" || route == "two_stage") << where << ": " << route;
            for (const int b : {0, batch - 1}) {
                const VectorView<Real> wb(&W[size_t(b) * n], n, 1);
                const auto& A0 = b == 0 ? first : last;
                const auto V = A.view().batch_range(b, 1);
                const std::string tag = where + " item " + std::to_string(b) + " " + route;
                EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(wb, lam, 1.0, A0.all_items())) << tag;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, n,
                              MV::eigen_residual(*this->ctx, A0, V, wb, A0.all_items()))
                    << tag;
                EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n,
                              MV::orthogonality(*this->ctx, V, A0.all_items()))
                    << tag;
            }
        }
    }
}

// The part loop on a batch small enough for the quick tier: a budget of 4.5 items' workspace (below the
// whole batch's) cuts batch 7 into parts of 2, 2 and 3 where the size is linear, a middle part included;
// each pinned family, poisoned strided storage, a per-item info span, and every item verified.
TYPED_TEST(SyevTest, BatchPartsUnderAScratchBudget) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "the part loop serves the native families, GPU kernels";
    } else {
        const int n = 64, batch = 7, ld = n + 3, stride = ld * n + 5;
        for (const char* family : {"blocked", "two_stage"})
            for (const BandCase& c : band_grid({n})) {
                const std::string where = family + band_where(n, c.uplo, c.jobz);
                const ScopedEnvVar pin("BATCHLAS_SYEV_ROUTE", family);
                UnifiedVector<T> buf(size_t(stride) * batch, T(0));
                UnifiedVector<T*> ptrs(batch, nullptr);
                MV A(buf.data(), n, n, ld, stride, batch, ptrs.data());
                Matrix<T, MatrixFormat::Dense> A0(n, n, batch);
                const auto lambda = fill_poisoned<T>(*this->ctx, A, c.uplo, A0);
                UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
                const auto size_of = [&](int items) {
                    return syev_buffer_size<B, T>(*this->ctx, A.batch_range(0, items), W.to_span(), c.jobz, c.uplo);
                };
                const std::size_t one = size_of(1), per = size_of(2) - one, three = size_of(3), whole = size_of(batch);
                ASSERT_GT(per, 0u) << where;
                const std::size_t budget_bytes = std::min(one + 3 * per + per / 2, whole - 1);
                const bool linear = three == one + 2 * per && whole == one + 6 * per;
                const ops::syev::detail::ScopedPartBudget budget(budget_bytes);
                const std::size_t parted = size_of(batch);
                // The parts' own pointer array (64-byte aligned, 3 items) follows the family's workspace.
                const std::size_t ptr_bytes = 64 + 3 * sizeof(T*);
                if (linear) EXPECT_EQ(parted, three + ptr_bytes) << where << ": the largest part is 3 items";
                EXPECT_LE(parted, budget_bytes + 64 + batch * sizeof(T*))
                    << where << " one=" << one << " per=" << per << " whole=" << whole;
                EXPECT_LT(parted, whole) << where;
                UnifiedVector<std::byte> ws(parted);
                UnifiedVector<int32_t> info(batch, int32_t(-1));
                (void)syev<B, T>(*this->ctx, A, W.to_span(), c.jobz, c.uplo, ws.to_span(), info.to_span());
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) EXPECT_EQ(info[b], 0) << where << " item " << b;
                const VectorView<Real> w(W.data(), n, batch);
                EXPECT_VERIFY(T, verify::Check::values, n, MV::values_error(w, lambda, 1.0, A0.all_items())) << where;
                if (c.jobz != JobType::EigenVectors) continue;
                EXPECT_VERIFY(T, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, A0, A, w, A0.all_items()))
                    << where;
                EXPECT_VERIFY(T, verify::Check::orthogonality_rotations, n, MV::orthogonality(*this->ctx, A, A0.all_items()))
                    << where;
            }
    }
}

// Vendor-free, the CPU has no syev family: the call and its size query throw NoRouteError, never a
// silent answer (docs/perf/syev.md#syev-vendor-free-double-band). With netlib, LAPACKE serves it.
TYPED_TEST(SyevTest, VendorFreeCpuHasNoRoute) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B != Backend::NETLIB) {
        GTEST_SKIP() << "a CPU-queue requirement";
    } else {
        if (batchlas::select::kHasNetlib) GTEST_SKIP() << "netlib present: LAPACKE serves the CPU";
        const int n = 64, batch = 2;
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        A.view().fill_graded_hermitian(*this->ctx, 6.0, 64u).wait();
        UnifiedVector<Real> W(size_t(n) * batch, Real(-1));
        for (const JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
            EXPECT_THROW(((void)syev_buffer_size<B, T>(*this->ctx, A.view(), W.to_span(), jobz, Uplo::Lower)),
                         batchlas::NoRouteError);
            UnifiedVector<std::byte> ws(size_t(1) << 20);
            EXPECT_THROW(((void)syev<B, T>(*this->ctx, A.view(), W.to_span(), jobz, Uplo::Lower, ws.to_span(),
                                           Span<int32_t>{})),
                         batchlas::NoRouteError);
        }
    }
}
