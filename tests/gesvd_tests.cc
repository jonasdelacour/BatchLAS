#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>

#if BATCHLAS_HAS_HOST_BACKEND
#include <lapacke.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

struct ScopedEnvVar {
    std::string key;
    bool had_old = false;
    std::string old;

    ScopedEnvVar(const char* k, const char* v) : key(k) {
        if (const char* prev = std::getenv(k)) {
            had_old = true;
            old = prev;
        }
        ::setenv(k, v, 1);
    }

    ~ScopedEnvVar() {
        if (had_old) {
            ::setenv(key.c_str(), old.c_str(), 1);
        } else {
            ::unsetenv(key.c_str());
        }
    }
};

} // namespace

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

    // Guards the values-only result against LAPACKE at n=8, absolute. Was
    // 5e-2, which at sigma_max ~ 3 is a 1.6% relative check; tightening the
    // three constants above without this one would just leave the loosest
    // guard here.
    static constexpr Real tol() {
        return std::is_same_v<Real, float> ? Real(2e-3f) : Real(1e-10);
    }
};

template <typename Config>
class GesvdHermitianComplexTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;

    static constexpr Real tol() {
        return std::is_same_v<Real, float> ? Real(5e-3f) : Real(1e-10);
    }
};

TYPED_TEST_SUITE(GesvdTest, GesvdTestTypes);
TYPED_TEST_SUITE(GesvdHermitianComplexTest, GesvdHermitianComplexTestTypes);

template <typename T>
inline T conj_value(const T& value) {
    if constexpr (test_utils::is_complex<T>::value) {
        return std::conj(value);
    } else {
        return value;
    }
}

template <typename T>
inline typename base_type<T>::type abs_squared_value(const T& value) {
    using Real = typename base_type<T>::type;
    if constexpr (test_utils::is_complex<T>::value) {
        return static_cast<Real>(std::norm(value));
    } else {
        return value * value;
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
template <typename Scalar>
int lapacke_gesvd_values_only_any(int m,
                                  int n,
                                  Scalar* a_col_major,
                                  typename base_type<Scalar>::type* s_out,
                                  typename base_type<Scalar>::type* superb) {
    using Real = typename base_type<Scalar>::type;
    if constexpr (std::is_same_v<Scalar, float>) {
        return LAPACKE_sgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              m,
                              n,
                              a_col_major,
                              m,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else if constexpr (std::is_same_v<Scalar, double>) {
        return LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              m,
                              n,
                              a_col_major,
                              m,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else if constexpr (std::is_same_v<Scalar, std::complex<float>>) {
        return LAPACKE_cgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              m,
                              n,
                              reinterpret_cast<lapack_complex_float*>(a_col_major),
                              m,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else {
        static_assert(std::is_same_v<Scalar, std::complex<double>>);
        return LAPACKE_zgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              m,
                              n,
                              reinterpret_cast<lapack_complex_double*>(a_col_major),
                              m,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    }
}
#endif // BATCHLAS_HAS_HOST_BACKEND

TYPED_TEST(GesvdTest, ValuesOnlyMatchesLapacke) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

#if !BATCHLAS_HAS_HOST_BACKEND
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

    std::vector<Real> s_ref(static_cast<size_t>(n));
    std::vector<Real> superb(static_cast<size_t>(n - 1));
    std::vector<Scalar> a_host(static_cast<size_t>(n) * static_cast<size_t>(n));

    for (int b = 0; b < batch; ++b) {
        auto Ab = A_ref.view().batch_item(b);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                a_host[static_cast<size_t>(j) * static_cast<size_t>(n) + static_cast<size_t>(i)] = Ab(i, j, 0);
            }
        }

        int info = 0;
        if constexpr (std::is_same_v<Real, float>) {
            info = LAPACKE_sgesvd(LAPACK_COL_MAJOR,
                                  'N',
                                  'N',
                                  n,
                                  n,
                                  reinterpret_cast<float*>(a_host.data()),
                                  n,
                                  reinterpret_cast<float*>(s_ref.data()),
                                  nullptr,
                                  1,
                                  nullptr,
                                  1,
                                  reinterpret_cast<float*>(superb.data()));
        } else {
            info = LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                                  'N',
                                  'N',
                                  n,
                                  n,
                                  reinterpret_cast<double*>(a_host.data()),
                                  n,
                                  reinterpret_cast<double*>(s_ref.data()),
                                  nullptr,
                                  1,
                                  nullptr,
                                  1,
                                  reinterpret_cast<double*>(superb.data()));
        }
        ASSERT_EQ(info, 0);

        Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(sb[i], s_ref[static_cast<size_t>(i)], TestFixture::tol());
        }
    }
#endif
}

namespace {

// The float constants used to be 5e-2 / 2e-1 / 3e-1. Those predate any path
// accurate enough to justify tightening them, and they had stopped guarding
// anything: 3e-1 permits a 30% relative reconstruction error against a measured
// ~1.3e-6 on these shapes.
//
// Chosen with margin over BOTH error sources, not only ours. The reference is
// LAPACKE_sgesvd in the SAME precision, whose own error is about
// eps_f32 * sigma_max ~= 1.1e-6 absolute at n=64; the test matrices are
// Random(-1,1), so sigma_max ~= 2*sqrt(n)/sqrt(3) ~= 9.2 there, and the
// singular-value check is ABSOLUTE.
//
// These were fitted by measurement across every provider and all three
// BATCHLAS_GESVD_BIDIAG settings, which is the sweep that matters: a value
// tuned only against the bdsdc default will fail the =normal path, whose
// relative error reaches 4e-1 at kappa=1e4.
// BATCHLAS_GESVD_BIDIAG=normal selects the OLD normal-equations bidiagonal
// path, which is retained purely so the three solvers can be A/B'd. It forms
// the tridiagonal of B^T B, so it squares the condition number and reaches ~4e-1
// relative error at kappa=1e4 -- it cannot meet the tolerances the default path
// meets, and it is not supposed to.
//
// So the tolerances are solver-aware rather than pinned to the worst path. The
// alternative was to keep 3e-1 forever, which is what made these guards
// vacuous in the first place; the alternative after that was to let the =normal
// A/B arm fail, which would quietly train people to ignore a red suite.
inline bool gesvd_bidiag_is_normal_equations() {
    const char* v = std::getenv("BATCHLAS_GESVD_BIDIAG");
    return v != nullptr && std::string(v) == "normal";
}

// The float constants used to be 5e-2 / 2e-1 / 3e-1. Those predate any path
// accurate enough to justify tightening them, and they had stopped guarding
// anything: 3e-1 permits a 30% relative reconstruction error against a measured
// ~1.3e-6 on these shapes.
//
// Chosen with margin over BOTH error sources, not only ours. The reference is
// LAPACKE_sgesvd in the SAME precision, whose own error is about
// eps_f32 * sigma_max ~= 1.1e-6 absolute at n=64; the test matrices are
// Random(-1,1), so sigma_max ~= 2*sqrt(n)/sqrt(3) ~= 9.2 there, and the
// singular-value check is ABSOLUTE. Verified against every provider and all
// three BATCHLAS_GESVD_BIDIAG settings.
template <typename Real>
inline Real gesvd_sv_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return gesvd_bidiag_is_normal_equations() ? Real(5e-2f) : Real(2e-3f);
    } else {
        return Real(1e-10);
    }
}

template <typename Real>
inline Real gesvd_ortho_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return gesvd_bidiag_is_normal_equations() ? Real(2e-1f) : Real(1e-3f);
    } else {
        return Real(5e-8);
    }
}

template <typename Real>
inline Real gesvd_recon_tol() {
    if constexpr (std::is_same_v<Real, float>) {
        return gesvd_bidiag_is_normal_equations() ? Real(3e-1f) : Real(1e-4f);
    } else {
        return Real(1e-8);
    }
}

#if BATCHLAS_HAS_HOST_BACKEND
template <typename Real>
int lapacke_gesvd_values_only(int n,
                              Real* a_col_major,
                              Real* s_out,
                              Real* superb) {
    if constexpr (std::is_same_v<Real, float>) {
        return LAPACKE_sgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              a_col_major,
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    } else {
        return LAPACKE_dgesvd(LAPACK_COL_MAJOR,
                              'N',
                              'N',
                              n,
                              n,
                              a_col_major,
                              n,
                              s_out,
                              nullptr,
                              1,
                              nullptr,
                              1,
                              superb);
    }
}
#endif

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
    std::unique_ptr<ScopedEnvVar> env;
    if (provider != nullptr) {
        env = std::make_unique<ScopedEnvVar>("BATCHLAS_GESVD_PROVIDER", provider);
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
                                          const UnifiedVector<typename base_type<Scalar>::type>& s,
                                          typename base_type<Scalar>::type tol = gesvd_sv_tol<typename base_type<Scalar>::type>()) {
#if BATCHLAS_HAS_HOST_BACKEND
    using Real = typename base_type<Scalar>::type;
    const int m = A_ref.rows();
    const int n = A_ref.cols();
    const int k = std::min(m, n);
    const int batch = A_ref.batch_size();

    std::vector<Real> s_ref(static_cast<size_t>(k));
    std::vector<Real> superb(static_cast<size_t>(std::max(0, k - 1)));
    std::vector<Scalar> a_host(static_cast<size_t>(m) * static_cast<size_t>(n));

    for (int b = 0; b < batch; ++b) {
        SCOPED_TRACE("batch=" + std::to_string(b));
        auto Ab = A_ref.view().batch_item(b);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                a_host[static_cast<size_t>(j) * static_cast<size_t>(m) + static_cast<size_t>(i)] = Ab(i, j, 0);
            }
        }

        const int info = lapacke_gesvd_values_only_any<Scalar>(m, n, a_host.data(), s_ref.data(), superb.data());
        EXPECT_EQ(info, 0);
        if (info != 0) {
            continue;
        }

        const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(k);
        for (int i = 0; i < k; ++i) {
            EXPECT_NEAR(sb[i], s_ref[static_cast<size_t>(i)], tol);
        }
    }
#else
    static_cast<void>(A_ref);
    static_cast<void>(s);
#endif
}

template <typename Real>
void expect_sorted_singular_values(const UnifiedVector<Real>& s,
                                   int n,
                                   int batch,
                                   const std::vector<Real>& expected_desc) {
    ASSERT_EQ(static_cast<int>(expected_desc.size()), n);
    for (int b = 0; b < batch; ++b) {
        SCOPED_TRACE("batch=" + std::to_string(b));
        const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(sb[i], expected_desc[static_cast<size_t>(i)], gesvd_sv_tol<Real>());
        }
    }
}

template <typename Scalar>
void expect_orthonormal_columns(const Matrix<Scalar, MatrixFormat::Dense>& M) {
    using Real = typename base_type<Scalar>::type;
    const int rows = M.rows();
    const int cols = M.cols();
    const int batch = M.batch_size();

    for (int b = 0; b < batch; ++b) {
        SCOPED_TRACE("batch=" + std::to_string(b));
        auto Mb = M.view().batch_item(b);
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < cols; ++j) {
                Scalar dot = Scalar(0);
                for (int row = 0; row < rows; ++row) {
                    dot += conj_value(Mb(row, i, 0)) * Mb(row, j, 0);
                }
                const Scalar target = (i == j) ? Scalar(1) : Scalar(0);
                test_utils::expect_near(dot, target, gesvd_ortho_tol<Real>());
            }
        }
    }
}

template <typename Scalar>
void expect_orthonormal_rows(const Matrix<Scalar, MatrixFormat::Dense>& M) {
    using Real = typename base_type<Scalar>::type;
    const int rows = M.rows();
    const int cols = M.cols();
    const int batch = M.batch_size();

    for (int b = 0; b < batch; ++b) {
        SCOPED_TRACE("batch=" + std::to_string(b));
        auto Mb = M.view().batch_item(b);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                Scalar dot = Scalar(0);
                for (int col = 0; col < cols; ++col) {
                    dot += Mb(i, col, 0) * conj_value(Mb(j, col, 0));
                }
                const Scalar target = (i == j) ? Scalar(1) : Scalar(0);
                test_utils::expect_near(dot, target, gesvd_ortho_tol<Real>());
            }
        }
    }
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

#if !BATCHLAS_HAS_HOST_BACKEND
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
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
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
        expect_orthonormal_columns(U);
        expect_orthonormal_rows(Vh);
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

        const Real sv_tol = std::is_same_v<Real, float> ? gesvd_sv_tol<Real>() : Real(2e-10);
        const Real recon_tol = std::is_same_v<Real, float> ? gesvd_recon_tol<Real>() : Real(2e-8);

        expect_singular_values_match_lapacke(A_ref, s, sv_tol);
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
// gesvdj_cta holds 4.8e-6 -> 1.2e-5 -> 7.1e-5 -> 5.6e-3. The order is now
// per-op (blas/dispatch/env.hh) and Jacobi leads for gesvd.
//
// These two tests guard that from opposite sides: the first pins the dispatch
// decision itself, the second pins the numerical consequence on the default
// path, so neither a reordering nor a predicate change can quietly undo it.
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

TYPED_TEST(GesvdTest, DefaultProviderRoutesSmallGeneralToJacobi) {
    using Scalar = typename TestFixture::Scalar;
    constexpr Backend B = TestFixture::B;
    namespace disp = batchlas::blas::dispatch;

    if constexpr (B != Backend::CUDA && B != Backend::ROCM) {
        GTEST_SKIP() << "Native gesvd providers are only dispatched on GPU backends.";
    } else {
        const disp::DeviceCaps caps = disp::query_caps(*this->ctx);
        const disp::DispatchPolicy policy = disp::policy_from_env("GESVD");

        // A stray BATCHLAS_GESVD_PROVIDER in the environment would make every
        // expectation below pass or fail for the wrong reason.
        ASSERT_EQ(policy.forced, disp::Provider::Auto)
            << "BATCHLAS_GESVD_PROVIDER is set; this test asserts the Auto order";

        Matrix<Scalar, MatrixFormat::Dense> A(32, 32, 2);

        // Every job combination at n <= 32, including values-only: the CTA path
        // is ~2.2x faster values-only at n=32 but has no correct digits past
        // kappa = 1e3, so it is not the default for any of them.
        for (SvdVectors jobu : {SvdVectors::None, SvdVectors::All}) {
            for (SvdVectors jobvh : {SvdVectors::None, SvdVectors::All}) {
                EXPECT_EQ(disp::detail::choose_gesvd_provider(policy, caps, A.view(), jobu, jobvh),
                          disp::Provider::BatchLAS_Jacobi)
                    << "jobu=" << static_cast<int>(jobu)
                    << " jobvh=" << static_cast<int>(jobvh);
            }
        }

        // Hermitian input is untouched: gesvd_supports_jacobi declines it, so
        // these still land on the CTA path.
        EXPECT_EQ(disp::detail::choose_gesvd_provider(policy, caps, A.view(),
                                                      SvdVectors::All, SvdVectors::All, Uplo::Lower),
                  disp::Provider::BatchLAS_CTA);

        // And n > 32 still reaches the blocked path rather than being captured
        // by the promoted Jacobi entry.
        Matrix<Scalar, MatrixFormat::Dense> Big(64, 64, 2);
        EXPECT_EQ(disp::detail::choose_gesvd_provider(policy, caps, Big.view(),
                                                      SvdVectors::All, SvdVectors::All),
                  disp::Provider::BatchLAS_Blocked);
    }
}

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

        // nullptr => no BATCHLAS_GESVD_PROVIDER override, i.e. the Auto order.
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

    expect_singular_values_match_lapacke(A_ref, s, gesvd_sv_tol<Real>());
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

    expect_singular_values_match_lapacke(A_ref, s, gesvd_sv_tol<Real>());
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

    const Real sv_tol = gesvd_sv_tol<Real>();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < k; ++i) {
            EXPECT_NEAR(static_cast<double>(s_thin[static_cast<size_t>(b) * k + i]),
                        static_cast<double>(s_full[static_cast<size_t>(b) * k + i]),
                        static_cast<double>(sv_tol))
                << "sigma mismatch b=" << b << " i=" << i;
        }
        auto Uf = U_full.view().batch_item(b);
        auto Ut = U_thin.view().batch_item(b);
        for (int c = 0; c < k; ++c) {
            // |<u_thin, u_full>| == 1: a singular vector's sign/phase is not
            // determined, so an entrywise comparison would be wrong.
            Scalar acc = Scalar(0);
            for (int i = 0; i < m; ++i) acc += conj_value(Ut(i, c, 0)) * Uf(i, c, 0);
            EXPECT_NEAR(static_cast<double>(std::abs(acc)), 1.0, 1e-2)
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
//  * Going through gesvd() with BATCHLAS_GESVD_PROVIDER=cta must still return
//    the right answer. Dispatch resets an unsupported forced provider to Auto
//    (gesvd.hh), so the request lands on a route that can serve it. That
//    degrade is pre-existing behaviour shared by every provider, not something
//    specific to Thin -- which is exactly why "it ran" is never by itself
//    evidence that a forced provider was used.
TYPED_TEST(GesvdTest, CtaRejectsGenuinelyThinButDispatchStillSucceeds) {
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

        const std::string err = run_gesvd_with_provider<Scalar, B>(
            *this->ctx, A, s, U, Vh, SvdVectors::Thin, SvdVectors::Thin, "cta");
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
