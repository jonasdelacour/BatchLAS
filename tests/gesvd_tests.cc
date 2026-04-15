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

    static constexpr Real tol() {
        return std::is_same_v<Real, float> ? Real(5e-2f) : Real(1e-10);
    }
};

template <typename Config>
class GesvdHermitianComplexTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    using Real = typename base_type<Scalar>::type;
    static constexpr Backend B = Config::BackendVal;

    static constexpr Real tol() {
        return std::is_same_v<Real, float> ? Real(8e-2f) : Real(1e-10);
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

    const size_t ws_bytes = gesvd_buffer_size<B>(*this->ctx,
                                                  A.view(),
                                                  s.to_span(),
                                                  U_dummy.view(),
                                                  Vh_dummy.view(),
                                                  SvdVectors::None,
                                                  SvdVectors::None);
    UnifiedVector<std::byte> ws(ws_bytes);

    auto evt = gesvd<B>(*this->ctx,
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

template <typename Real>
constexpr Real gesvd_sv_tol() {
    return std::is_same_v<Real, float> ? Real(5e-2f) : Real(1e-10);
}

template <typename Real>
constexpr Real gesvd_ortho_tol() {
    return std::is_same_v<Real, float> ? Real(2e-1f) : Real(5e-8);
}

template <typename Real>
constexpr Real gesvd_recon_tol() {
    return std::is_same_v<Real, float> ? Real(3e-1f) : Real(1e-8);
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
            ? gesvd_buffer_size<B>(ctx,
                                   A.view(),
                                   s.to_span(),
                                   U.view(),
                                   Vh.view(),
                                   jobu,
                                   jobvh,
                                   *hermitian_uplo)
            : gesvd_buffer_size<B>(ctx,
                                   A.view(),
                                   s.to_span(),
                                   U.view(),
                                   Vh.view(),
                                   jobu,
                                   jobvh);
        UnifiedVector<std::byte> ws(ws_bytes);
        auto evt = hermitian_uplo.has_value()
            ? gesvd<B>(ctx,
                       A.view(),
                       s.to_span(),
                       U.view(),
                       Vh.view(),
                       jobu,
                       jobvh,
                       *hermitian_uplo,
                       ws.to_span())
            : gesvd<B>(ctx,
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

    const size_t ws_bytes = gesvd_buffer_size<B>(*this->ctx,
                                                 A.view(),
                                                 s.to_span(),
                                                 U_dummy.view(),
                                                 Vh_dummy.view(),
                                                 SvdVectors::None,
                                                 SvdVectors::None,
                                                 Uplo::Lower);
    UnifiedVector<std::byte> ws(ws_bytes);

    auto evt = gesvd<B>(*this->ctx,
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
