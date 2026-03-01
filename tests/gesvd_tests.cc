#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>

#if BATCHLAS_HAS_HOST_BACKEND
#include <lapacke.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
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

using GesvdTestTypes = typename backend_real_types<GesvdConfig>::type;

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

TYPED_TEST_SUITE(GesvdTest, GesvdTestTypes);

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

    gesvd<B>(*this->ctx,
             A.view(),
             s.to_span(),
             U_dummy.view(),
             Vh_dummy.view(),
             SvdVectors::None,
             SvdVectors::None,
             ws.to_span());
    this->ctx->wait();

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

TYPED_TEST(GesvdTest, UOnlyBlockedPathProducesOrthonormalU) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    const int n = 8;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 4242);
    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
    Matrix<Scalar, MatrixFormat::Dense> Vh_dummy(n, n, batch);

    const size_t ws_bytes = gesvd_buffer_size<B>(*this->ctx,
                                                  A.view(),
                                                  s.to_span(),
                                                  U.view(),
                                                  Vh_dummy.view(),
                                                  SvdVectors::All,
                                                  SvdVectors::None);
    UnifiedVector<std::byte> ws(ws_bytes);

    ScopedEnvVar force_blocked("BATCHLAS_GESVD_PROVIDER", "blocked");

    EXPECT_NO_THROW({
        gesvd<B>(*this->ctx,
                 A.view(),
                 s.to_span(),
                 U.view(),
                 Vh_dummy.view(),
                 SvdVectors::All,
                 SvdVectors::None,
                 ws.to_span());
        this->ctx->wait();
    });

    const Real ortho_tol = std::is_same_v<Real, float> ? Real(2e-1f) : Real(1e-10);
    for (int b = 0; b < batch; ++b) {
        auto Ub = U.view().batch_item(b);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real dot = Real(0);
                for (int k = 0; k < n; ++k) {
                    dot += Ub(k, i, 0) * Ub(k, j, 0);
                }
                const Real target = (i == j) ? Real(1) : Real(0);
                EXPECT_NEAR(dot, target, ortho_tol);
            }
        }
    }
}

TYPED_TEST(GesvdTest, VhOnlyBlockedPathProducesOrthonormalVh) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    const int n = 8;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 31415);
    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U_dummy(n, n, batch);
    Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

    const size_t ws_bytes = gesvd_buffer_size<B>(*this->ctx,
                                                  A.view(),
                                                  s.to_span(),
                                                  U_dummy.view(),
                                                  Vh.view(),
                                                  SvdVectors::None,
                                                  SvdVectors::All);
    UnifiedVector<std::byte> ws(ws_bytes);

    ScopedEnvVar force_blocked("BATCHLAS_GESVD_PROVIDER", "blocked");

    EXPECT_NO_THROW({
        gesvd<B>(*this->ctx,
                 A.view(),
                 s.to_span(),
                 U_dummy.view(),
                 Vh.view(),
                 SvdVectors::None,
                 SvdVectors::All,
                 ws.to_span());
        this->ctx->wait();
    });

    const Real ortho_tol = std::is_same_v<Real, float> ? Real(2e-1f) : Real(1e-10);
    for (int b = 0; b < batch; ++b) {
        auto Vhb = Vh.view().batch_item(b);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real dot = Real(0);
                for (int k = 0; k < n; ++k) {
                    dot += Vhb(i, k, 0) * Vhb(j, k, 0);
                }
                const Real target = (i == j) ? Real(1) : Real(0);
                EXPECT_NEAR(dot, target, ortho_tol);
            }
        }
    }
}

TYPED_TEST(GesvdTest, FullVectorsBlockedPathProducesOrthonormalUVh) {
    using Scalar = typename TestFixture::Scalar;
    using Real = typename TestFixture::Real;
    constexpr Backend B = TestFixture::B;

    const int n = 8;
    const int batch = 2;

    Matrix<Scalar, MatrixFormat::Dense> A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, false, batch, 27182);
    Matrix<Scalar, MatrixFormat::Dense> A_ref(n, n, batch);
    MatrixView<Scalar, MatrixFormat::Dense>::copy(*this->ctx, A_ref.view(), A.view()).wait();
    UnifiedVector<Real> s(static_cast<size_t>(n) * static_cast<size_t>(batch));
    Matrix<Scalar, MatrixFormat::Dense> U(n, n, batch);
    Matrix<Scalar, MatrixFormat::Dense> Vh(n, n, batch);

    const size_t ws_bytes = gesvd_buffer_size<B>(*this->ctx,
                                                  A.view(),
                                                  s.to_span(),
                                                  U.view(),
                                                  Vh.view(),
                                                  SvdVectors::All,
                                                  SvdVectors::All);
    UnifiedVector<std::byte> ws(ws_bytes);

    ScopedEnvVar force_blocked("BATCHLAS_GESVD_PROVIDER", "blocked");

    EXPECT_NO_THROW({
        gesvd<B>(*this->ctx,
                 A.view(),
                 s.to_span(),
                 U.view(),
                 Vh.view(),
                 SvdVectors::All,
                 SvdVectors::All,
                 ws.to_span());
        this->ctx->wait();
    });

    const Real ortho_tol = std::is_same_v<Real, float> ? Real(2e-1f) : Real(1e-10);
    const Real recon_tol = std::is_same_v<Real, float> ? Real(3e-1f) : Real(1e-10);
    for (int b = 0; b < batch; ++b) {
        auto Ub = U.view().batch_item(b);
        auto Vhb = Vh.view().batch_item(b);
        auto Ab_ref = A_ref.view().batch_item(b);

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real dot_u = Real(0);
                Real dot_vh = Real(0);
                for (int k = 0; k < n; ++k) {
                    dot_u += Ub(k, i, 0) * Ub(k, j, 0);
                    dot_vh += Vhb(i, k, 0) * Vhb(j, k, 0);
                }
                const Real target = (i == j) ? Real(1) : Real(0);
                EXPECT_NEAR(dot_u, target, ortho_tol);
                EXPECT_NEAR(dot_vh, target, ortho_tol);
            }
        }

        Real err2 = Real(0);
        Real ref2 = Real(0);
        const Real* sb = s.data() + static_cast<size_t>(b) * static_cast<size_t>(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Real recon = Real(0);
                for (int k = 0; k < n; ++k) {
                    recon += Ub(i, k, 0) * sb[k] * Vhb(k, j, 0);
                }
                const Real ref = Ab_ref(i, j, 0);
                const Real diff = recon - ref;
                err2 += diff * diff;
                ref2 += ref * ref;
            }
        }
        const Real rel_err = sycl::sqrt(err2 / std::max(ref2, Real(1e-20)));
        if constexpr (B == Backend::NETLIB) {
            EXPECT_LE(rel_err, recon_tol);
        }
    }
}
