#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/linalg.hh>
// gesvdj_cta and GesvdjParams: the `info` cases below reach the Jacobi tier
// directly, because ops::gesvd::launch hands it a default-constructed GesvdjParams
// and so no sweep cap is reachable from the public entry point.
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/verify/reference.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

#include "../src/ops/gesvd/choice.hh"

using namespace batchlas;

template <typename T, Backend B>
struct GesvdConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct backend_real_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
        std::tuple<Config<float, Backend::NETLIB>,
                   Config<double, Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<float, Backend::ROCM>,
                   Config<double, Backend::ROCM>>{},
#endif
        std::tuple<>{}));

    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

template <template <typename, Backend> class Config>
struct backend_complex_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
        std::tuple<Config<std::complex<float>, Backend::NETLIB>,
                   Config<std::complex<double>, Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<std::complex<float>, Backend::CUDA>,
                   Config<std::complex<double>, Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
        std::tuple<Config<std::complex<float>, Backend::ROCM>,
                   Config<std::complex<double>, Backend::ROCM>>{},
#endif
        std::tuple<>{}));

    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using GesvdTestTypes = typename backend_real_types<GesvdConfig>::type;
using GesvdHermitianComplexTestTypes = typename backend_complex_types<GesvdConfig>::type;

template <typename Config>
class GesvdTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;

};

template <typename Config>
class GesvdHermitianComplexTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;
};

// Complex GENERAL input, as distinct from the Hermitian-complex fixture above.
// The suite had no such case at all: blocked runs complex only Hermitian (can_run),
// so before gesvdj_cta covered the 33..64 band these shapes fell through to
// Vendor and threw.
template <typename Config>
class GesvdGeneralComplexTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;

    // Mirrors sycl_gesvd::gesvd_jacobi_max_dim: complex<double> with vectors does not fit
    // local memory at the C=64 rung on this device.
    static constexpr int max_dim_with_vectors() {
        return std::is_same_v<Scalar, std::complex<double>> ? 32 : 64;
    }
};

TYPED_TEST_SUITE(GesvdTest, GesvdTestTypes);
TYPED_TEST_SUITE(GesvdHermitianComplexTest, GesvdHermitianComplexTestTypes);
TYPED_TEST_SUITE(GesvdGeneralComplexTest, GesvdHermitianComplexTestTypes);

template <typename T>
inline typename base_type<T>::type abs_squared_value(const T& value) {
    using Real = typename base_type<T>::type;
    if constexpr (test_utils::is_complex<T>::value) {
        return static_cast<Real>(std::norm(value));
    } else {
        return value * value;
    }
}

namespace {
// Defined with the other expect_* helpers below.
template <typename Scalar>
void expect_singular_values_match_lapacke(const Matrix<Scalar, MatrixFormat::Dense>& A_ref,
                                          const UnifiedVector<typename base_type<Scalar>::type>& s);
}  // namespace


TYPED_TEST(GesvdTest, ValuesOnlyMatchesLapacke) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    GTEST_SKIP() << "Reference LAPACKE backend unavailable.";
#else
    const int n = 8;
    const int batch = 3;

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 1337);
    Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U_dummy(n, n, batch);
    Matrix<Scalar, MatrixFormat::Dense> Vh_dummy(n, n, batch);

    const size_t ws_bytes = gesvd_buffer_size(*this->ctx,
                                                  A.view(),
                                                  s.to_span(),
                                                  U_dummy.view(),
                                                  Vh_dummy.view(),
                                                  SvdVectors::None,
                                                  SvdVectors::None);
    UnifiedVector<std::byte> ws(ws_bytes);

    auto evt = gesvd(*this->ctx,
                        A.view(),
                        s.to_span(),
                        U_dummy.view(),
                        Vh_dummy.view(),
                        SvdVectors::None,
                        SvdVectors::None,
                        ws.to_span());
    evt.wait();

    expect_singular_values_match_lapacke(A_ref, s);
#endif
}

namespace {

// BATCHLAS_GESVD_BIDIAG=normal selects the old normal-equations bidiagonal path, kept only so the
// three solvers can be A/B'd: it squares the condition number (~4e-1 relative error at kappa = 1e4),
// so the checks below that it reaches carry their own bound or slack.
inline bool gesvd_bidiag_is_normal_equations() {
    const char* v = std::getenv("BATCHLAS_GESVD_BIDIAG");
    return v != nullptr && std::string(v) == "normal";
}

// The reconstruction ||A - U S V^H||_F / ||A||_F has no library kind (verification.md, "Kept outside
// the library"), so it keeps this file's measured per-type bound. The float constant used to be 3e-1,
// which permitted a 30% error against a measured ~1.3e-6; BATCHLAS_GESVD_BIDIAG=normal squares the
// condition number and keeps the old bound.
template <typename Real>
inline Real gesvd_recon_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return gesvd_bidiag_is_normal_equations() ? Real(3e-1f) : Real(1e-4f);
    } else {
        return Real(1e-8);
    }
}

// The cta provider always takes the normal-equations bidiagonal path, which squares the condition
// number. Measured c needed for orthogonality: ~52 float, ~113 double. The factors keep the bounds
// accepted on 2026-10-09 (64 n eps float, 128 n eps double), tighter than orthogonality_rotations'
// c = 256. Pass it to the orthogonality checks of results produced by a "cta" pin on a real-typed
// input (the Hermitian complex cta case measures c ~ 2.8 and needs none).
template <typename Real>
inline batchlas::verify::Slack gesvd_cta_slack() {
    return {std::is_same_v<Real, float> ? 0.25 : 0.5,
            "gesvd cta provider always takes the normal-equations bidiagonal path (squares the condition number); measured c 52 float / 113 double, accepted bounds 64 / 128 n eps"};
}

template <typename Scalar, Backend B>
std::string run_gesvd_with_provider(Queue& ctx,
                                    Matrix<Scalar, MatrixFormat::Dense>& A,
                                    UnifiedVector<typename base_type<Scalar>::type>& s,
                                    Matrix<Scalar, MatrixFormat::Dense>& U,
                                    Matrix<Scalar, MatrixFormat::Dense>& Vh,
                                    SvdVectors jobu,
                                    SvdVectors jobvh,
                                    const char* provider,
                                    std::optional<Uplo> hermitian_uplo = std::nullopt) {
    // A pin is a ScopedPin: a refused one throws (R6) and lands in the returned message.
    std::optional<select::ScopedPin<ops::gesvd::GesvdChoice>> pin;
    if (provider != nullptr) {
        pin.emplace("gesvd", std::string_view(provider));
    }

    try {
        const size_t ws_bytes = hermitian_uplo.has_value()
            ? gesvd_buffer_size(ctx,
                                   A.view(),
                                   s.to_span(),
                                   U.view(),
                                   Vh.view(),
                                   jobu,
                                   jobvh,
                                   *hermitian_uplo)
            : gesvd_buffer_size(ctx,
                                   A.view(),
                                   s.to_span(),
                                   U.view(),
                                   Vh.view(),
                                   jobu,
                                   jobvh);
        UnifiedVector<std::byte> ws(ws_bytes);
        auto evt = hermitian_uplo.has_value()
            ? gesvd(ctx,
                       A.view(),
                       s.to_span(),
                       U.view(),
                       Vh.view(),
                       jobu,
                       jobvh,
                       *hermitian_uplo,
                       ws.to_span())
            : gesvd(ctx,
                       A.view(),
                       s.to_span(),
                       U.view(),
                       Vh.view(),
                       jobu,
                       jobvh,
                       ws.to_span());
        evt.wait();
    } catch (const std::exception& ex) {
        return ex.what();
    }

    return {};
}

template <typename Scalar>
void expect_singular_values_match_lapacke(const Matrix<Scalar, MatrixFormat::Dense>& A_ref,
                                          const UnifiedVector<typename base_type<Scalar>::type>& s) {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
    using Real = typename base_type<Scalar>::type;
    const int k = std::min(A_ref.rows(), A_ref.cols());
    const int batch = A_ref.batch_size();

    // The reference runs in double on the exact input, so it is not limited by this precision.
    std::vector<std::vector<double>> ref(static_cast<size_t>(batch));
    double sigma_max = 0.0;
    for (int b = 0; b < batch; ++b) {
        auto a = batchlas::verify::copy_item(A_ref.view(), b);
        ASSERT_TRUE(batchlas::verify::singular_values(A_ref.rows(), A_ref.cols(), a, ref[static_cast<size_t>(b)]))
            << "LAPACKE reference failed, batch=" << b;
        if (k > 0) sigma_max = batchlas::verify::nanmax(sigma_max, ref[static_cast<size_t>(b)].front());
    }
    const VectorView<Real> w(const_cast<Real*>(s.data()), k, batch);
    const double err = batchlas::verify::values_error(w, ref, sigma_max);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, k, err);
#else
    static_cast<void>(A_ref);
    static_cast<void>(s);
#endif
}

// s (k values per item, packed) against ref[item] over @p items at Check::values, relative to the
// largest reference value; no items, nothing to check.
template <typename Real>
void expect_singular_values_near(const UnifiedVector<Real>& s, const std::vector<std::vector<double>>& ref, int k, int batch,
                                 const std::vector<int>& items) {
    if (items.empty()) return;
    double scale = 0.0;
    for (int b : items)
        for (double r : ref[static_cast<size_t>(b)]) scale = batchlas::verify::nanmax(scale, std::fabs(r));
    const VectorView<Real> w(const_cast<Real*>(s.data()), k, batch);
    EXPECT_VERIFY(Real, batchlas::verify::Check::values, k, batchlas::verify::values_error(w, ref, scale, items));
}

// s against another run's values (thin vs full, blocked vs unblocked) at Check::values, every item.
template <typename Real>
void expect_singular_values_agree(const UnifiedVector<Real>& s, const UnifiedVector<Real>& s_ref, int k, int batch) {
    std::vector<std::vector<double>> ref(static_cast<size_t>(batch));
    for (int b = 0; b < batch; ++b)
        ref[static_cast<size_t>(b)].assign(s_ref.begin() + static_cast<std::ptrdiff_t>(b) * k, s_ref.begin() + static_cast<std::ptrdiff_t>(b + 1) * k);
    expect_singular_values_near(s, ref, k, batch, batchlas::verify::all_items(batch));
}

template <typename Real>
void expect_sorted_singular_values(const UnifiedVector<Real>& s,
                                   int n,
                                   int batch,
                                   const std::vector<Real>& expected_desc) {
    ASSERT_EQ(static_cast<int>(expected_desc.size()), n);
    const std::vector<std::vector<double>> ref(static_cast<size_t>(batch), std::vector<double>(expected_desc.begin(), expected_desc.end()));
    expect_singular_values_near(s, ref, n, batch, batchlas::verify::all_items(batch));
}

// Orthogonality n is the length of the vectors checked: M.rows() here, and M.cols() for
// expect_orthonormal_rows (the rows of M^H).
template <typename Scalar>
void expect_orthonormal_columns(const Matrix<Scalar, MatrixFormat::Dense>& M,
                                batchlas::verify::Slack slack = {1.0, "-"}) {
    using Real = typename base_type<Scalar>::type;
    const double err = batchlas::verify::orthogonality(M.view(), batchlas::verify::all_items(M.batch_size()));
    if (std::is_same_v<Real, float> && gesvd_bidiag_is_normal_equations()) {
        const batchlas::verify::Slack normal{0.5, "BATCHLAS_GESVD_BIDIAG=normal squares the condition number: measured once, ThinTallUnderNormalEquationsBidiag 128 x 48 (n = 128 rows), c 2.69 (2.05e-5); a whole-binary sweep under BIDIAG=normal is not measured. Accepted bound 128 n eps (0.5x of c=256)"};
        // A tighter slack the caller passed wins, with its own reason.
        EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::orthogonality_rotations, M.rows(), err, slack.factor < normal.factor ? slack : normal);
    } else if (slack.factor != 1.0) {
        EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::orthogonality_rotations, M.rows(), err, slack);
    } else {
        EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, M.rows(), err);
    }
}

// Rows are orthonormal when the columns of M^H are.
template <typename Scalar>
void expect_orthonormal_rows(const Matrix<Scalar, MatrixFormat::Dense>& M,
                             batchlas::verify::Slack slack = {1.0, "-"}) {
    Matrix<Scalar, MatrixFormat::Dense> Mh(M.cols(), M.rows(), M.batch_size());
    for (int b = 0; b < M.batch_size(); ++b) {
        auto Mb = M.view().batch_item(b);
        auto Hb = Mh.view().batch_item(b);
        for (int j = 0; j < M.cols(); ++j) {
            for (int i = 0; i < M.rows(); ++i) {
                const auto v = batchlas::verify::conj(batchlas::verify::up(Mb(i, j, 0)));
                if constexpr (test_utils::is_complex<Scalar>::value) {
                    Hb(j, i, 0) = Scalar(static_cast<typename base_type<Scalar>::type>(v.real()),
                                         static_cast<typename base_type<Scalar>::type>(v.imag()));
                } else {
                    Hb(j, i, 0) = static_cast<Scalar>(v);
                }
            }
        }
    }
    expect_orthonormal_columns(Mh, slack);
}

template <typename Scalar>
void expect_reconstruction(const Matrix<Scalar, MatrixFormat::Dense>& A_ref,
                           const UnifiedVector<typename base_type<Scalar>::type>& s,
                           const Matrix<Scalar, MatrixFormat::Dense>& U,
                           const Matrix<Scalar, MatrixFormat::Dense>& Vh,
                           typename base_type<Scalar>::type tol = gesvd_recon_tol<typename base_type<Scalar>::type>()) {
    using Real = typename base_type<Scalar>::type;
    const int m = A_ref.rows();
    const int n = A_ref.cols();
    const int k = std::min(m, n);
    const int batch = A_ref.batch_size();

    for (int b = 0; b < batch; ++b) {
        SCOPED_TRACE("batch=" + std::to_string(b));
        auto Ab = A_ref.view().batch_item(b);
        auto Ub = U.view().batch_item(b);
        auto Vhb = Vh.view().batch_item(b);
        const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(k);

        Real err2 = Real(0);
        Real ref2 = Real(0);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Scalar recon = Scalar(0);
                for (int kk = 0; kk < k; ++kk) {
                    recon += Ub(i, kk, 0) * Scalar(sb[kk]) * Vhb(kk, j, 0);
                }
                const Scalar ref = Ab(i, j, 0);
                const Scalar diff = recon - ref;
                err2 += abs_squared_value(diff);
                ref2 += abs_squared_value(ref);
            }
        }

        const Real rel_err = std::sqrt(err2 / std::max(ref2, Real(1e-20)));
        EXPECT_LE(rel_err, tol);
    }
}

TYPED_TEST(GesvdHermitianComplexTest, ValuesOnlyMatchesLapacke) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    GTEST_SKIP() << "Reference LAPACKE backend unavailable.";
#else
    const int n = 12;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 4242);
    Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U_dummy(1, 1, batch);
    Matrix<Scalar, MatrixFormat::Dense> Vh_dummy(1, 1, batch);

    const size_t ws_bytes = gesvd_buffer_size(*this->ctx,
                                                 A.view(),
                                                 s.to_span(),
                                                 U_dummy.view(),
                                                 Vh_dummy.view(),
                                                 SvdVectors::None,
                                                 SvdVectors::None,
                                                 Uplo::Lower);
    UnifiedVector<std::byte> ws(ws_bytes);

    auto evt = gesvd(*this->ctx,
                        A.view(),
                        s.to_span(),
                        U_dummy.view(),
                        Vh_dummy.view(),
                        SvdVectors::None,
                        SvdVectors::None,
                        Uplo::Lower,
                        ws.to_span());
    evt.wait();

    expect_singular_values_match_lapacke(A_ref, s);
#endif
}

TYPED_TEST(GesvdHermitianComplexTest, BlockedProviderFullVectorsMatchHermitianSvd) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Blocked native provider is only dispatched on GPU backends.";
    } else {
        const int n = 48;
        const int batch = 2;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 5151);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "blocked",
                                                                   Uplo::Lower);
        ASSERT_TRUE(err.empty()) << err;

        expect_singular_values_match_lapacke(A_ref, s);
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

TYPED_TEST(GesvdHermitianComplexTest, CtaProviderFullVectorsMatchHermitianSvd) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA) {
        GTEST_SKIP() << "CTA native provider is only covered on CUDA in this test pass.";
    } else {
        const int n = 16;
        const int batch = 2;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 6161);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "cta",
                                                                   Uplo::Lower);
        ASSERT_TRUE(err.empty()) << err;

        expect_singular_values_match_lapacke(A_ref, s);
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

template <typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> make_repeated_tiny_spectrum_matrix(int n, int batch) {
    using Real = typename base_type<Scalar>::type;
    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Zeros(n, n, batch);

    std::vector<Real> diag(static_cast<size_t>(n), Real(0));
    if (n > 0) diag[0] = Real(10);
    if (n > 1) diag[1] = Real(10);
    for (int i = 2; i < std::max(2, n - 2); ++i) {
        diag[static_cast<size_t>(i)] = std::max<Real>(Real(0.5), Real(9.0) - Real(0.1 * (i - 2)));
    }
    if (n > 2) diag[static_cast<size_t>(n - 2)] = Real(1e-4);
    if (n > 3) diag[static_cast<size_t>(n - 1)] = Real(1e-7);

    for (int b = 0; b < batch; ++b) {
        auto Ab = A.view().batch_item(b);
        const int shift = (3 * b) % std::max(1, n);
        for (int i = 0; i < n; ++i) {
            const int src = (i + shift) % std::max(1, n);
            Ab(i, i, 0) = static_cast<Scalar>(diag[static_cast<size_t>(src)]);
        }
    }

    return A;
}

struct GesvdJobCase {
    SvdVectors jobu;
    SvdVectors jobvh;
    const char* name;
};

constexpr std::array<GesvdJobCase, 4> kGesvdJobCases{{
    {SvdVectors::None, SvdVectors::None, "NN"},
    {SvdVectors::All, SvdVectors::None, "AN"},
    {SvdVectors::None, SvdVectors::All, "NA"},
    {SvdVectors::All, SvdVectors::All, "AA"},
}};

} // namespace

TYPED_TEST(GesvdTest, BlockedProviderCoversAllJobCombinations) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Blocked native provider is only dispatched on GPU backends.";
    } else {
        const int n = 64;
        const int batch = 2;

        for (size_t case_idx = 0; case_idx < kGesvdJobCases.size(); ++case_idx) {
            const auto& job = kGesvdJobCases[case_idx];
            SCOPED_TRACE(std::string("provider=blocked case=") + job.name);

            auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 4000u + static_cast<unsigned>(case_idx));
            Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
            MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

            UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
            Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
            Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

            const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                       A,
                                                                       s,
                                                                       U,
                                                                       Vh,
                                                                       job.jobu,
                                                                       job.jobvh,
                                                                       "blocked");
            ASSERT_TRUE(err.empty()) << err;

            expect_singular_values_match_lapacke(A_ref, s);
            if (job.jobu == SvdVectors::All) {
                expect_orthonormal_columns(U);
            }
            if (job.jobvh == SvdVectors::All) {
                expect_orthonormal_rows(Vh);
            }
            if (job.jobu == SvdVectors::All && job.jobvh == SvdVectors::All) {
                expect_reconstruction(A_ref, s, U, Vh);
            }
        }
    }
}

TYPED_TEST(GesvdTest, CtaProviderCoversAllJobCombinations) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA) {
        GTEST_SKIP() << "CTA native provider is only covered on CUDA in this test pass.";
    } else {
        const int n = 16;
        const int batch = 2;

        for (size_t case_idx = 0; case_idx < kGesvdJobCases.size(); ++case_idx) {
            const auto& job = kGesvdJobCases[case_idx];
            SCOPED_TRACE(std::string("provider=cta case=") + job.name);

            auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 5000u + static_cast<unsigned>(case_idx));
            Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
            MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

            UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
            Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
            Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

            const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                       A,
                                                                       s,
                                                                       U,
                                                                       Vh,
                                                                       job.jobu,
                                                                       job.jobvh,
                                                                       "cta");
            ASSERT_TRUE(err.empty()) << err;

            expect_singular_values_match_lapacke(A_ref, s);
            if (job.jobu == SvdVectors::All) {
                expect_orthonormal_columns(U, gesvd_cta_slack<typename TestFixture::Real>());
            }
            if (job.jobvh == SvdVectors::All) {
                expect_orthonormal_rows(Vh, gesvd_cta_slack<typename TestFixture::Real>());
            }
            if (job.jobu == SvdVectors::All && job.jobvh == SvdVectors::All) {
                expect_reconstruction(A_ref, s, U, Vh);
            }
        }
    }
}

TYPED_TEST(GesvdTest, BlockedProviderHandlesRepeatedAndTinySingularValues) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Blocked native provider is only dispatched on GPU backends.";
    } else {
        const int n = 64;
        const int batch = 2;

        auto A = make_repeated_tiny_spectrum_matrix<Scalar>(n, batch);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "blocked");
        ASSERT_TRUE(err.empty()) << err;

        std::vector<Real> expected(static_cast<size_t>(n), Real(0));
        if (n > 0) expected[0] = Real(10);
        if (n > 1) expected[1] = Real(10);
        for (int i = 2; i < std::max(2, n - 2); ++i) {
            expected[static_cast<size_t>(i)] = std::max<Real>(Real(0.5), Real(9.0) - Real(0.1 * (i - 2)));
        }
        if (n > 2) expected[static_cast<size_t>(n - 2)] = Real(1e-4);
        if (n > 3) expected[static_cast<size_t>(n - 1)] = Real(1e-7);
        std::sort(expected.begin(), expected.end(), std::greater<Real>());

        expect_sorted_singular_values(s, n, batch, expected);
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

TYPED_TEST(GesvdTest, CtaProviderHandlesRepeatedAndTinySingularValues) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA) {
        GTEST_SKIP() << "CTA native provider is only covered on CUDA in this test pass.";
    } else {
        const int n = 16;
        const int batch = 2;

        auto A = make_repeated_tiny_spectrum_matrix<Scalar>(n, batch);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "cta");
        ASSERT_TRUE(err.empty()) << err;

        std::vector<Real> expected(static_cast<size_t>(n), Real(0));
        if (n > 0) expected[0] = Real(10);
        if (n > 1) expected[1] = Real(10);
        for (int i = 2; i < std::max(2, n - 2); ++i) {
            expected[static_cast<size_t>(i)] = std::max<Real>(Real(0.5), Real(9.0) - Real(0.1 * (i - 2)));
        }
        if (n > 2) expected[static_cast<size_t>(n - 2)] = Real(1e-4);
        if (n > 3) expected[static_cast<size_t>(n - 1)] = Real(1e-7);
        std::sort(expected.begin(), expected.end(), std::greater<Real>());

        expect_sorted_singular_values(s, n, batch, expected);
        expect_orthonormal_columns(U, gesvd_cta_slack<typename TestFixture::Real>());
        expect_orthonormal_rows(Vh, gesvd_cta_slack<typename TestFixture::Real>());
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

TYPED_TEST(GesvdTest, BlockedProviderTallRectangularFullVectors) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Blocked native provider is only dispatched on GPU backends.";
    } else {
        const int m = 24;
        const int n = 16;
        const int batch = 2;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 7001);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(m, m, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "blocked");
        ASSERT_TRUE(err.empty()) << err;

        expect_singular_values_match_lapacke(A_ref, s);
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

TYPED_TEST(GesvdTest, CtaProviderWideRectangularFullVectors) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA) {
        GTEST_SKIP() << "CTA native provider is only covered on CUDA in this test pass.";
    } else {
        const int m = 12;
        const int n = 16;
        const int batch = 2;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 7002);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(m) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(m, m, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "cta");
        ASSERT_TRUE(err.empty()) << err;

        expect_singular_values_match_lapacke(A_ref, s);
        expect_orthonormal_columns(U, gesvd_cta_slack<typename TestFixture::Real>());
        expect_orthonormal_rows(Vh, gesvd_cta_slack<typename TestFixture::Real>());
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

TYPED_TEST(GesvdTest, BlockedProviderLargeTallRectangularFullVectors) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Blocked native provider is only dispatched on GPU backends.";
    } else {
        const int m = 192;
        const int n = 128;
        const int batch = 1;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 7003);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<typename base_type<Scalar>::type> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(m, m, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   "blocked");
        ASSERT_TRUE(err.empty()) << err;

        const Real recon_tol = std::is_same_v<Real, float> ? gesvd_recon_tol<Real>() : Real(2e-8);

        expect_singular_values_match_lapacke(A_ref, s);
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
        expect_reconstruction(A_ref, s, U, Vh, recon_tol);
    }
}

// ---------------------------------------------------------------------------
// Default-provider routing for n <= 32.
//
// gesvdj_cta used to sit behind BatchLAS_CTA in the shared provider order, so
// Auto never reached it for real input. The CTA path forms the normal
// equations; measured at n=32/float/256 samples, its singular-value relative
// error runs 1.4e-6 -> 3.1e-3 -> 0.235 -> 1.857 across log10(kappa) 1..6 while
// gesvdj_cta holds 4.8e-6 -> 1.2e-5 -> 7.1e-5 -> 5.6e-3. Jacobi leads every
// n <= 32 row of tuned/gesvd.*.txt.
//
// The dispatch decision itself is pinned by gesvd_candidates_tests
// (AutoReadsTheTranscribedTable); the test below pins the numerical consequence
// on the default path, so neither a table nor a can_run change can undo it.
// ---------------------------------------------------------------------------

namespace {

// A = H(u) * diag(sigma) * H(v), with H(x) = I - 2 x x^T the Householder
// reflector of a unit vector x. Both factors are orthogonal, so the singular
// values of A are exactly sigma -- no reference solve is needed.
//
// It has to be DENSE to discriminate here. make_repeated_tiny_spectrum_matrix
// above builds a diagonal matrix, whose columns are already orthogonal: Jacobi
// converges in zero sweeps and A^T A is diagonal, so the normal-equation path
// is exact too and the two are indistinguishable however ill-conditioned the
// spectrum is.
template <typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> make_graded_dense_matrix(int n,
                                                            int batch,
                                                            double log10cond) {
    using Real = typename base_type<Scalar>::type;

    std::vector<double> sigma(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double t = (n > 1) ? static_cast<double>(i) / static_cast<double>(n - 1) : 0.0;
        sigma[static_cast<size_t>(i)] = std::pow(10.0, -log10cond * t);
    }

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Zeros(n, n, batch);

    for (int b = 0; b < batch; ++b) {
        // Deterministic per-batch-item reflectors; a fixed LCG keeps this
        // reproducible without pulling in a generator that is itself suspect.
        std::vector<double> u(static_cast<size_t>(n)), v(static_cast<size_t>(n));
        uint64_t state = 0x9E3779B97F4A7C15ull + static_cast<uint64_t>(b) * 0x1000193ull;
        auto next = [&state]() {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            return static_cast<double>((state >> 11) & ((1ull << 53) - 1)) / static_cast<double>(1ull << 53) - 0.5;
        };
        double nu = 0.0, nv = 0.0;
        for (int i = 0; i < n; ++i) {
            u[static_cast<size_t>(i)] = next();
            v[static_cast<size_t>(i)] = next();
            nu += u[static_cast<size_t>(i)] * u[static_cast<size_t>(i)];
            nv += v[static_cast<size_t>(i)] * v[static_cast<size_t>(i)];
        }
        nu = std::sqrt(nu);
        nv = std::sqrt(nv);
        for (int i = 0; i < n; ++i) {
            u[static_cast<size_t>(i)] /= nu;
            v[static_cast<size_t>(i)] /= nv;
        }

        auto Ab = A.view().batch_item(b);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                for (int k = 0; k < n; ++k) {
                    const double h1 = (i == k ? 1.0 : 0.0) - 2.0 * u[static_cast<size_t>(i)] * u[static_cast<size_t>(k)];
                    const double h2 = (k == j ? 1.0 : 0.0) - 2.0 * v[static_cast<size_t>(k)] * v[static_cast<size_t>(j)];
                    acc += h1 * sigma[static_cast<size_t>(k)] * h2;
                }
                Ab(i, j, 0) = static_cast<Scalar>(static_cast<Real>(acc));
            }
        }
    }

    return A;
}

}  // namespace

TYPED_TEST(GesvdTest, DefaultProviderKeepsSingularValuesAtHighCondition) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native gesvd providers are only dispatched on GPU backends.";
    } else {
        const int n = 32;
        const int batch = 4;
        const double log10cond = 5.0;

        auto A = make_graded_dense_matrix<Scalar>(n, batch, log10cond);

        UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

        // nullptr => no pin, i.e. the Auto order.
        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx,
                                                                   A,
                                                                   s,
                                                                   U,
                                                                   Vh,
                                                                   SvdVectors::All,
                                                                   SvdVectors::All,
                                                                   nullptr);
        ASSERT_TRUE(err.empty()) << err;

        std::vector<Real> expected(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(n - 1);
            expected[static_cast<size_t>(i)] = static_cast<Real>(std::pow(10.0, -log10cond * t));
        }

        // RELATIVE error, per singular value -- the quantity the normal-equation
        // path destroys and an absolute check cannot see. At kappa = 1e5 the CTA
        // path measures ~1.0 here and gesvdj_cta ~6e-4, so this threshold
        // separates them by two orders of magnitude in each direction.
        const Real sv_rel_tol = std::is_same_v<Real, float> ? Real(1e-2f) : Real(1e-8);

        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                const Real got = s[static_cast<size_t>(b) * static_cast<size_t>(n) + static_cast<size_t>(i)];
                const Real want = expected[static_cast<size_t>(i)];
                EXPECT_LE(std::abs(got - want) / want, sv_rel_tol)
                    << "batch " << b << " sigma[" << i << "] = " << got << ", expected " << want;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Thin (economy) singular vectors on the blocked path.
//
// Note there is deliberately no GTEST_SKIP on the CUDA backend in the first
// test: dispatch pins NETLIB to Vendor, so running it on GesvdTest/0 and /1 is
// what exercises the new LAPACKE jobu='S' mapping, which is in turn the
// reference the GPU results are checked against.
// ---------------------------------------------------------------------------

TYPED_TEST(GesvdTest, ThinTallRectangular) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    const int m = 192;
    const int n = 64;
    const int k = std::min(m, n);
    const int batch = 2;

    auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 8101);
    Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

    UnifiedVector<Real> s(static_cast<size_t>(k) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U(m, k, batch);      // thin: m x k, not m x m
    Matrix<Scalar, MatrixFormat::Dense> Vh(k, n, batch);     // k == n here, so this is full

    const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx, A, s, U, Vh,
                                                               SvdVectors::Thin,
                                                               SvdVectors::Thin,
                                                               nullptr);
    ASSERT_TRUE(err.empty()) << err;

    expect_singular_values_match_lapacke(A_ref, s);
    expect_orthonormal_columns(U);
    expect_orthonormal_rows(Vh);
    expect_reconstruction(A_ref, s, U, Vh);
}

TYPED_TEST(GesvdTest, ThinWideRectangular) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    // m < n takes the transpose branch, where the thin factor is V^H and the
    // workspace view it is produced through (ut_view) had to become rectangular.
    const int m = 64;
    const int n = 192;
    const int k = std::min(m, n);
    const int batch = 2;

    auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 8102);
    Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

    UnifiedVector<Real> s(static_cast<size_t>(k) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U(m, k, batch);      // k == m here, so this is full
    Matrix<Scalar, MatrixFormat::Dense> Vh(k, n, batch);     // thin: k x n, not n x n

    const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx, A, s, U, Vh,
                                                               SvdVectors::Thin,
                                                               SvdVectors::Thin,
                                                               nullptr);
    ASSERT_TRUE(err.empty()) << err;

    expect_singular_values_match_lapacke(A_ref, s);
    expect_orthonormal_columns(U);
    expect_orthonormal_rows(Vh);
    expect_reconstruction(A_ref, s, U, Vh);
}

// The point of the whole item: a thin request must not pay the full U cost,
// in U *or* in the workspace. Without the direct-bidiag forcing rule the
// m x m allocation simply migrates into the scratch buffer and the caller
// still cannot run the shape.
TYPED_TEST(GesvdTest, ThinWorkspaceIsSmallerThanFull) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native blocked provider is only dispatched on GPU backends.";
    } else {
        const int m = 512;
        const int n = 32;
        const int k = std::min(m, n);
        const int batch = 2;

        Matrix<Scalar, MatrixFormat::Dense> A(m, n, batch);
        UnifiedVector<Real> s(static_cast<size_t>(k) * static_cast<size_t>(batch));

        Matrix<Scalar, MatrixFormat::Dense> U_full(m, m, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh_full(n, n, batch);
        const size_t ws_full = gesvd_buffer_size<B, Scalar>(
            *this->ctx, A.view(), s.to_span(), U_full.view(), Vh_full.view(),
            SvdVectors::All, SvdVectors::All);

        Matrix<Scalar, MatrixFormat::Dense> U_thin(m, k, batch);
        Matrix<Scalar, MatrixFormat::Dense> Vh_thin(k, n, batch);
        const size_t ws_thin = gesvd_buffer_size<B, Scalar>(
            *this->ctx, A.view(), s.to_span(), U_thin.view(), Vh_thin.view(),
            SvdVectors::Thin, SvdVectors::Thin);

        EXPECT_LT(ws_thin, ws_full)
            << "thin workspace " << ws_thin << " is not smaller than full " << ws_full;
    }
}

// Thin must be an economy mode, not a differently-computed answer.
TYPED_TEST(GesvdTest, ThinMatchesFullLeadingColumns) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    const int m = 96;
    const int n = 48;
    const int k = std::min(m, n);
    const int batch = 2;

    auto A_full = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 8103);
    Matrix<Scalar, MatrixFormat::Dense> A_thin(m, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_thin.view(), A_full.view()).wait();

    UnifiedVector<Real> s_full(static_cast<size_t>(k) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U_full(m, m, batch), Vh_full(n, n, batch);
    const std::string err_full = run_gesvd_with_provider<Scalar, B>(
        *this->ctx, A_full, s_full, U_full, Vh_full, SvdVectors::All, SvdVectors::All, nullptr);
    ASSERT_TRUE(err_full.empty()) << err_full;

    UnifiedVector<Real> s_thin(static_cast<size_t>(k) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U_thin(m, k, batch), Vh_thin(k, n, batch);
    const std::string err_thin = run_gesvd_with_provider<Scalar, B>(
        *this->ctx, A_thin, s_thin, U_thin, Vh_thin, SvdVectors::Thin, SvdVectors::Thin, nullptr);
    ASSERT_TRUE(err_thin.empty()) << err_thin;

    expect_singular_values_agree(s_thin, s_full, k, batch);
    for (int b = 0; b < batch; ++b) {
        auto Uf = U_full.view().batch_item(b);
        auto Ut = U_thin.view().batch_item(b);
        for (int c = 0; c < k; ++c) {
            // |<u_thin, u_full>| == 1: a singular vector's sign/phase is not
            // determined, so an entrywise comparison would be wrong.
            batchlas::verify::promoted_t<Scalar> acc = 0;
            for (int i = 0; i < m; ++i)
                acc += batchlas::verify::conj(batchlas::verify::up(Ut(i, c, 0))) * batchlas::verify::up(Uf(i, c, 0));
            EXPECT_NEAR(batchlas::verify::abs(acc), 1.0, 1e-2)
                << "U column " << c << " differs at b=" << b;
        }
    }
}

// gesvd_cta cannot produce a genuinely thin factor: mode CTA always takes the
// normal-equations branch, whose patch_zero_left_vectors writes m columns of U
// unconditionally. Two different things are asserted here, and they differ:
//
//  * A DIRECT gesvd_cta call must throw. Silently writing m columns into a U
//    that has k is an overrun.
//  * A `cta` pin through gesvd() throws (R6: a pin whose can_run is false is an
//    error, where the old router silently fell back to Auto), and Auto still
//    returns the right answer on a family that can serve it.
TYPED_TEST(GesvdTest, CtaRejectsGenuinelyThinAndItsPinThrows) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native CTA provider is only dispatched on GPU backends.";
    } else {
        const int m = 32, n = 8, k = 8, batch = 2;
        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 8104);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<Real> s(static_cast<size_t>(k) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(m, k, batch), Vh(k, n, batch);

        EXPECT_THROW(
            (gesvd_cta_buffer_size<B, Scalar>(*this->ctx, A.view(), s.to_span(),
                                              U.view(), Vh.view(),
                                              SvdVectors::Thin, SvdVectors::Thin)),
            std::invalid_argument);

        const std::string pinned = run_gesvd_with_provider<Scalar, B>(
            *this->ctx, A, s, U, Vh, SvdVectors::Thin, SvdVectors::Thin, "cta");
        EXPECT_NE(pinned.find("cannot run this shape"), std::string::npos) << pinned;
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A.view(), A_ref.view()).wait();
        const std::string err = run_gesvd_with_provider<Scalar, B>(
            *this->ctx, A, s, U, Vh, SvdVectors::Thin, SvdVectors::Thin, nullptr);
        ASSERT_TRUE(err.empty()) << err;
        expect_orthonormal_columns(U);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

// The only test of the direct-bidiag forcing rule. Without it, a thin tall U
// under BATCHLAS_GESVD_BIDIAG=normal reaches patch_zero_left_vectors, which
// writes m columns into a U that has only k.
TYPED_TEST(GesvdTest, ThinTallUnderNormalEquationsBidiag) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native blocked provider is only dispatched on GPU backends.";
    } else {
        ScopedEnvVar bidiag("BATCHLAS_GESVD_BIDIAG", "normal");

        const int m = 128, n = 48;
        const int k = std::min(m, n);
        const int batch = 2;

        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 8105);
        Matrix<Scalar, MatrixFormat::Dense> A_ref(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

        UnifiedVector<Real> s(static_cast<size_t>(k) * static_cast<size_t>(batch));
        Matrix<Scalar, MatrixFormat::Dense> U(m, k, batch), Vh(k, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(*this->ctx, A, s, U, Vh,
                                                                   SvdVectors::Thin,
                                                                   SvdVectors::Thin,
                                                                   nullptr);
        ASSERT_TRUE(err.empty()) << err;

        expect_orthonormal_columns(U);
        expect_reconstruction(A_ref, s, U, Vh);
    }
}

// ---------------------------------------------------------------------------
// Tall input with min(m, n) <= 32.
//
// This band had no coverage at all, which is why it stayed 64-179x slow. It
// needs min(m,n) <= 32 to reach the blocked provider's small-n branch and
// m > 32 for the level-2 bidiagonalisation to hurt, and until the gesvd
// benchmarks took m and n separately, every one of them built a square
// Random(n, n). The CTA and Jacobi predicates both require max(m,n) <= 32, so
// these shapes reach neither -- they are blocked-provider-only by construction.
// ---------------------------------------------------------------------------
TYPED_TEST(GesvdTest, TallNarrowBelowCtaCap) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native blocked provider is only dispatched on GPU backends.";
    } else {
        struct Shape { int m; int n; };
        const Shape shapes[] = {{256, 8}, {256, 16}, {512, 32}};

        for (const auto& sh : shapes) {
            SCOPED_TRACE("m=" + std::to_string(sh.m) + " n=" + std::to_string(sh.n));
            const int k = std::min(sh.m, sh.n);
            const int batch = 3;

            auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(sh.m, sh.n, false, batch, 9101);
            Matrix<Scalar, MatrixFormat::Dense> A_ref(sh.m, sh.n, batch);
            MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

            UnifiedVector<Real> s(static_cast<size_t>(k) * batch);
            Matrix<Scalar, MatrixFormat::Dense> U(sh.m, k, batch);
            Matrix<Scalar, MatrixFormat::Dense> Vh(k, sh.n, batch);

            // nullptr => the Auto order, which must land on Blocked here.
            const std::string err = run_gesvd_with_provider<Scalar, B>(
                *this->ctx, A, s, U, Vh, SvdVectors::Thin, SvdVectors::Thin, nullptr);
            ASSERT_TRUE(err.empty()) << err;

            expect_singular_values_match_lapacke(A_ref, s);
            expect_orthonormal_columns(U);
            expect_orthonormal_rows(Vh);
            expect_reconstruction(A_ref, s, U, Vh);
        }
    }
}

// The blocked bidiagonalisation must agree with the unblocked one it now
// replaces at small n. BATCHLAS_GESVD_BLOCKED_GEBRD_MIN restores the old path,
// so this is differential rather than absolute.
TYPED_TEST(GesvdTest, BlockedGebrdMatchesUnblockedAtSmallN) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native blocked provider is only dispatched on GPU backends.";
    } else {
        const int m = 192, n = 16, k = 16, batch = 3;

        auto A_ref = Matrix<Scalar, MatrixFormat::Dense>::Random(m, n, false, batch, 9102);
        Matrix<Scalar, MatrixFormat::Dense> A_blk(m, n, batch), A_unb(m, n, batch);
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_blk.view(), A_ref.view()).wait();
        MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_unb.view(), A_ref.view()).wait();

        UnifiedVector<Real> s_blk(static_cast<size_t>(k) * batch), s_unb(static_cast<size_t>(k) * batch);
        Matrix<Scalar, MatrixFormat::Dense> U_blk(m, k, batch), Vh_blk(k, n, batch);
        Matrix<Scalar, MatrixFormat::Dense> U_unb(m, k, batch), Vh_unb(k, n, batch);

        {
            const std::string err = run_gesvd_with_provider<Scalar, B>(
                *this->ctx, A_blk, s_blk, U_blk, Vh_blk, SvdVectors::Thin, SvdVectors::Thin, "blocked");
            ASSERT_TRUE(err.empty()) << "blocked gebrd: " << err;
        }
        {
            ScopedEnvVar old_path("BATCHLAS_GESVD_BLOCKED_GEBRD_MIN", "9999");
            const std::string err = run_gesvd_with_provider<Scalar, B>(
                *this->ctx, A_unb, s_unb, U_unb, Vh_unb, SvdVectors::Thin, SvdVectors::Thin, "blocked");
            ASSERT_TRUE(err.empty()) << "unblocked gebrd: " << err;
        }

        expect_singular_values_agree(s_blk, s_unb, k, batch);
        expect_orthonormal_columns(U_blk);
        expect_reconstruction(A_ref, s_blk, U_blk, Vh_blk);
    }
}

// ---------------------------------------------------------------------------
// Complex GENERAL SVD above n = 32 (follow-up item 5).
//
// This used to throw. The old router's blocked predicate returned false for complex
// and its cta predicate did too outside the Hermitian branch, so complex general
// input fell through to Vendor, whose only binding is gesvdjBatched at
// max(m,n) <= 32. Widening gesvdj_cta to 64 is what closes the band.
//
// Note the follow-up list attributes "18 gesvd_tests cases skip for this" to
// this gap. That is not so: all 18 skips are NETLIB-backend skips of GPU-only
// provider tests, none of them would newly pass, and there was no complex
// GENERAL case in the suite to un-skip. These are new.
// ---------------------------------------------------------------------------

TYPED_TEST(GesvdGeneralComplexTest, GeneralComplexAboveThirtyTwo) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native gesvd providers are only dispatched on GPU backends.";
    } else {
        if (TestFixture::max_dim_with_vectors() < 64) {
            GTEST_SKIP() << "complex<double> with vectors is capped at 32 (local memory)";
        }

        struct Shape { int m; int n; };
        const Shape shapes[] = {{48, 48}, {64, 40}, {40, 64}};

        for (const auto& sh : shapes) {
            SCOPED_TRACE("m=" + std::to_string(sh.m) + " n=" + std::to_string(sh.n));
            const int k = std::min(sh.m, sh.n);
            const int batch = 2;

            // hermitian=false: this is the GENERAL path, not the Hermitian one.
            auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(sh.m, sh.n, false, batch, 6161);
            Matrix<Scalar, MatrixFormat::Dense> A_ref(sh.m, sh.n, batch);
            MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();

            UnifiedVector<Real> s(static_cast<size_t>(k) * batch);
            Matrix<Scalar, MatrixFormat::Dense> U(sh.m, sh.m, batch);
            Matrix<Scalar, MatrixFormat::Dense> Vh(sh.n, sh.n, batch);

            // nullptr => the Auto order. Reaching a result at all is the point.
            const std::string err = run_gesvd_with_provider<Scalar, B>(
                *this->ctx, A, s, U, Vh, SvdVectors::All, SvdVectors::All, nullptr);
            ASSERT_TRUE(err.empty()) << err;

            for (int b = 0; b < batch; ++b) {
                for (int i = 1; i < k; ++i) {
                    const size_t idx = static_cast<size_t>(b) * k;
                    EXPECT_LE(s[idx + i], s[idx + i - 1] * Real(1 + 1e-4))
                        << "sigma not descending at b=" << b << " i=" << i;
                }
            }
            expect_orthonormal_columns(U);
            expect_orthonormal_rows(Vh);
            expect_reconstruction(A_ref, s, U, Vh);
        }
    }
}

// Above the Jacobi cap there is still no complex general route, and it must
// fail loudly rather than return something. This is also what proves the test
// above is exercising the new band rather than some pre-existing path.
TYPED_TEST(GesvdGeneralComplexTest, GeneralComplexAboveCapStillRefused) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native gesvd providers are only dispatched on GPU backends.";
    } else {
        const int n = 96, batch = 2;
        auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 6162);
        UnifiedVector<Real> s(static_cast<size_t>(n) * batch);
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch), Vh(n, n, batch);

        const std::string err = run_gesvd_with_provider<Scalar, B>(
            *this->ctx, A, s, U, Vh, SvdVectors::All, SvdVectors::All, nullptr);
        EXPECT_FALSE(err.empty())
            << "complex general at n=96 silently produced a result; there is no route for it";
    }
}


// ---------------------------------------------------------------------------
// Per-item convergence status (`info`).
//
// The SVD is the other half of the "LAPACK returns info > 0" family, and its
// status was dropped in three different ways before this work package: bdsqr
// computed a per-item fail_flags array, then collapsed it into one bool and
// threw for the WHOLE batch; the netlib arm captured LAPACKE_?gesvd's info and
// destroyed it in the same throw; gesvdj_cta's sweep loop had no flag at all,
// only an optional sweep COUNT that the public gesvd cannot even reach because
// ops::gesvd::launch passes a default-constructed GesvdjParams.
//
// A batch-wide throw is not a status: it says some item failed, never which, and
// it takes the good items' answers down with it.
// ---------------------------------------------------------------------------

namespace {

// Diagonal with distinct descending entries: singular values are exactly
// n, n-1, ..., 1, so "converged" can be checked against a closed form rather
// than against another run of the same code.
//
// Both fills take the view BY VALUE: MatrixView::operator() has a const overload
// returning `const T&`, so a const-reference parameter would make every
// assignment below a compile error. The view is a non-owning descriptor.
template <typename Scalar>
void fill_diagonal_svd_matrix(MatrixView<Scalar, MatrixFormat::Dense> A) {
    const int n = A.rows();
    const int batch = A.batch_size();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) A(i, j, b) = Scalar(0);
        }
        for (int i = 0; i < n; ++i) A(i, i, b) = Scalar(n - i);
    }
}

// Even items diagonal (a one-sided Jacobi sweep finds nothing to rotate, so they
// converge at the smallest budget that can prove convergence at all); odd items
// dense and generic (they do not). The mix is what lets the forced case below
// assert both directions from one run.
template <typename Scalar>
void fill_mixed_svd_matrices(MatrixView<Scalar, MatrixFormat::Dense> A, unsigned seed) {
    const int n = A.rows();
    const int batch = A.batch_size();
    unsigned state = seed;
    auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return double(state >> 8) / double(1u << 24);
    };
    for (int b = 0; b < batch; ++b) {
        const bool easy = (b % 2) == 0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A(i, j, b) = easy ? Scalar(0) : Scalar(next() * 2.0 - 1.0);
            }
        }
        if (easy) {
            for (int i = 0; i < n; ++i) A(i, i, b) = Scalar(n - i);
        }
    }
}

}  // namespace

TYPED_TEST(GesvdTest, InfoIsZeroOnAConvergingBatch) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    const int n = 32;
    const int batch = 8;

    Matrix<Scalar, MatrixFormat::Dense> A(n, n, batch);
    fill_diagonal_svd_matrix<Scalar>(A.view());
    Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch), Vh(n, n, batch);
    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));

    // -1, NOT 0: a zero-filled span cannot distinguish "the solver wrote 0" from
    // "nothing wrote it", and the entry point is required to clear the span
    // itself, so a surviving -1 is a defect and not a missing write.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    const size_t bytes = gesvd_buffer_size<B, Scalar>(*this->ctx, A.view(), s.to_span(), U.view(),
                                                      Vh.view(), SvdVectors::All, SvdVectors::All);
    UnifiedVector<std::byte> ws(bytes);
    (void)gesvd<B, Scalar>(*this->ctx, A.view(), s.to_span(), U.view(), Vh.view(), SvdVectors::All,
                     SvdVectors::All, ws.to_span(), info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: nothing wrote "
                                  "the span, so a zero here would have proved nothing";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a diagonal matrix";
    }
    std::vector<std::vector<double>> exact(static_cast<size_t>(batch));
    std::vector<int> converged;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        converged.push_back(b);
        for (int i = 0; i < n; ++i) exact[static_cast<size_t>(b)].push_back(double(n - i));
    }
    expect_singular_values_near(s, exact, n, batch, converged);
}

// An empty span is "not requested": neither the workspace query nor the answer
// may move. gesvd_buffer_size takes no `info` argument at all, so the first half
// holds by construction; the second is what a caller could observe.
TYPED_TEST(GesvdTest, EmptyInfoSpanChangesNeitherAnswerNorWorkspace) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    const int n = 32;
    const int batch = 4;

    Matrix<Scalar, MatrixFormat::Dense> A0(n, n, batch), A1(n, n, batch);
    fill_diagonal_svd_matrix<Scalar>(A0.view());
    fill_diagonal_svd_matrix<Scalar>(A1.view());
    Matrix<Scalar, MatrixFormat::Dense> U0(n, n, batch), Vh0(n, n, batch);
    Matrix<Scalar, MatrixFormat::Dense> U1(n, n, batch), Vh1(n, n, batch);
    UnifiedVector<Real> s0(static_cast<size_t>(n) * batch), s1(static_cast<size_t>(n) * batch);
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    const size_t bytes_a = gesvd_buffer_size<B, Scalar>(
        *this->ctx, A0.view(), s0.to_span(), U0.view(), Vh0.view(), SvdVectors::All, SvdVectors::All);
    const size_t bytes_b = gesvd_buffer_size<B, Scalar>(
        *this->ctx, A1.view(), s1.to_span(), U1.view(), Vh1.view(), SvdVectors::All, SvdVectors::All);
    EXPECT_EQ(bytes_a, bytes_b);

    UnifiedVector<std::byte> ws0(bytes_a), ws1(bytes_a);
    (void)gesvd<B, Scalar>(*this->ctx, A0.view(), s0.to_span(), U0.view(), Vh0.view(), SvdVectors::All,
                     SvdVectors::All, ws0.to_span(), info.to_span());
    (void)gesvd<B, Scalar>(*this->ctx, A1.view(), s1.to_span(), U1.view(), Vh1.view(), SvdVectors::All,
                     SvdVectors::All, ws1.to_span(), Span<int32_t>{});
    this->ctx->wait();

    for (size_t i = 0; i < s0.size(); ++i) {
        EXPECT_EQ(s0[i], s1[i]) << "requesting status changed the answer at index " << i;
    }
}

// THE FORCED DIRECTION, through the tier rather than the facade.
//
// ops::gesvd::launch hands gesvdj_cta a DEFAULT GesvdjParams (src/ops/gesvd/gesvd.cc),
// so no sweep cap is reachable from the public entry point -- not even the
// sweep_counts channel that has existed all along. gesvdj_cta itself takes its
// params, so that is where the cap goes.
//
// max_sweeps = 2, not 1, and the reason is the termination rule: gesvdj_cta
// requires TWO CONSECUTIVE zero-rotation sweeps before it will call an item
// converged (gesvdj_cta.cc:733-737, whose comment explains that one is a silent
// wrong answer). At a cap of 1 NOTHING can converge, so "the items reporting
// info == 0 are still correct" would be vacuously true. At 2 the diagonal items
// converge and the dense ones do not, which is the mixed outcome this case needs
// in order to assert both directions.
#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(GesvdTest, InfoReportsItemsThatExhaustTheSweepBudget) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "gesvdj_cta requires sub-group 32; not available on the host backend";
    } else {
        const int n = 32;
        const int batch = 8;

        // Full budget first: this is the reference the capped run's converged
        // items are held to, and it is the SAME tier on the SAME input, so a
        // difference can only come from the cap.
        Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
        fill_mixed_svd_matrices<Scalar>(A_ref.view(), 20260909u);
        Matrix<Scalar, MatrixFormat::Dense> U_ref(n, n, batch), Vh_ref(n, n, batch);
        UnifiedVector<Real> s_ref(static_cast<size_t>(n) * batch);
        UnifiedVector<int32_t> info_ref(batch, int32_t(-1));

        GesvdjParams<Scalar> full;
        (void)gesvdj_cta<B, Scalar>(*this->ctx, A_ref.view(), s_ref.to_span(), U_ref.view(),
                              Vh_ref.view(), SvdVectors::All, SvdVectors::All, Span<std::byte>{},
                              full, info_ref.to_span());
        this->ctx->wait();
        for (int b = 0; b < batch; ++b) {
            ASSERT_NE(info_ref[b], -1) << "info_ref[" << b << "] still holds the poison value";
            EXPECT_EQ(info_ref[b], 0)
                << "item " << b << " did not converge at the DEFAULT sweep cap; the forced case "
                   "below can then prove nothing about the cap";
        }

        Matrix<Scalar, MatrixFormat::Dense> A(n, n, batch);
        fill_mixed_svd_matrices<Scalar>(A.view(), 20260909u);
        Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch), Vh(n, n, batch);
        UnifiedVector<Real> s(static_cast<size_t>(n) * batch);
        UnifiedVector<int32_t> info(batch, int32_t(-1));

        GesvdjParams<Scalar> capped;
        capped.max_sweeps = 2;
        (void)gesvdj_cta<B, Scalar>(*this->ctx, A.view(), s.to_span(), U.view(), Vh.view(),
                              SvdVectors::All, SvdVectors::All, Span<std::byte>{}, capped,
                              info.to_span());
        this->ctx->wait();

        int reported = 0;
        for (int b = 0; b < batch; ++b) {
            ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value";
            ASSERT_GE(info[b], 0) << "info is LAPACK-like: 0 or a positive count, never negative";
            if (info[b] != 0) ++reported;
        }
        EXPECT_GT(reported, 0)
            << "a two-sweep budget on dense 32x32 items reported universal convergence; "
               "either the status is not written, or it is written unconditionally zero";

        // An item reporting info == 0 must match the full-budget run.
        std::vector<std::vector<double>> full_run(static_cast<size_t>(batch));
        std::vector<int> converged;
        for (int b = 0; b < batch; ++b) {
            if (info[b] != 0) continue;
            converged.push_back(b);
            full_run[static_cast<size_t>(b)].assign(s_ref.begin() + static_cast<std::ptrdiff_t>(b) * n, s_ref.begin() + static_cast<std::ptrdiff_t>(b + 1) * n);
        }
        expect_singular_values_near(s, full_run, n, batch, converged);
    }
}
#endif  // BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
