#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename Real>
Real tol_for() {
    if constexpr (std::is_same_v<Real, float>) return Real(2) * Real(test_utils::tolerance<float>());
    return Real(test_utils::tolerance<double>());
}

// Build Q from Householder vectors stored in A/tau for SYTD2-style Lower storage.
// This is generic (works for both CTA and blocked SYTRD paths), for batch=1.
template <Backend B, typename Real>
Matrix<Real, MatrixFormat::Dense> build_q_from_sytrd_lower(Queue& ctx,
                                                           const MatrixView<Real, MatrixFormat::Dense>& A_out,
                                                           const VectorView<Real>& tau,
                                                           int n) {
    if (n <= 1) return Matrix<Real, MatrixFormat::Dense>::Identity(n, /*batch_size=*/1);

    const int p = n - 1;
    auto Av = A_out;

    Matrix<Real, MatrixFormat::Dense> Aq(p, p, /*batch=*/1);
    Matrix<Real, MatrixFormat::Dense> Qsub = Matrix<Real, MatrixFormat::Dense>::Identity(p, /*batch_size=*/1);
    Vector<Real> tau_qr(p, /*batch=*/1);
    auto aq = Aq.view();

    for (int j = 0; j < p; ++j) {
        for (int i = 0; i < p; ++i) {
            aq(i, j, 0) = Real(0);
        }
    }

    for (int i = 0; i < p; ++i) {
        aq(i, i, 0) = Real(1);
        tau_qr(i, 0) = tau(i, 0);
        for (int r = i + 1; r < p; ++r) {
            // sub row r corresponds to global row (r+1)
            aq(r, i, 0) = Av(r + 1, i, 0);
        }
    }

    UnifiedVector<std::byte> ws_ormqr(
        ormqr_buffer_size<B>(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data()));

    ormqr<B>(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data(), ws_ormqr.to_span()).wait();

    Matrix<Real, MatrixFormat::Dense> Q = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, /*batch_size=*/1);
    auto Qv = Q.view();
    auto Qsv = Qsub.view();

    Qv(0, 0, 0) = Real(1);
    for (int r = 0; r < p; ++r) {
        for (int c = 0; c < p; ++c) {
            Qv(r + 1, c + 1, 0) = Qsv(r, c, 0);
        }
    }

    return Q;
}

template <typename Real>
void assert_tridiagonal_matches(const MatrixView<Real, MatrixFormat::Dense>& T,
                                int n,
                                const VectorView<Real>& d,
                                const VectorView<Real>& e,
                                Real tol) {
    for (int i = 0; i < n; ++i) {
        ASSERT_TRUE(std::isfinite(static_cast<double>(d(i, 0))));
        ASSERT_TRUE(std::isfinite(static_cast<double>(T(i, i, 0))));
        EXPECT_NEAR(T(i, i, 0), d(i, 0), tol) << "diag mismatch at i=" << i;
    }

    for (int i = 0; i < n - 1; ++i) {
        ASSERT_TRUE(std::isfinite(static_cast<double>(e(i, 0))));
        EXPECT_NEAR(T(i + 1, i, 0), e(i, 0), tol) << "offdiag mismatch at i=" << i;
        EXPECT_NEAR(T(i, i + 1, 0), e(i, 0), tol) << "offdiag mismatch at i=" << i;
    }

    const Real ztol = tol * Real(50);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (std::abs(i - j) <= 1) continue;
            EXPECT_NEAR(T(i, j, 0), Real(0), ztol) << "non-tridiagonal at (" << i << "," << j << ")";
        }
    }
}

template <typename T, Backend B>
struct SytrdBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdBlockedTestTypes = ::testing::Types<SytrdBlockedConfig<float, Backend::CUDA>, SytrdBlockedConfig<double, Backend::CUDA>>;
#else
using SytrdBlockedTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SytrdBlockedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdBlockedTest, SytrdBlockedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 128;
    const int batch = 128;
    const int nb = 32;
    const Real tol = tol_for<Real>();

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/789);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    // Validate a few representative batch items fully. (Validating all 128 would be expensive.)
    std::vector<int> batch_items;
    batch_items.push_back(0);
    if (batch > 1) batch_items.push_back(batch / 2);
    if (batch > 2) batch_items.push_back(batch - 1);

    auto dv = static_cast<VectorView<Real>>(d);
    auto ev = static_cast<VectorView<Real>>(e);
    auto tauv = static_cast<VectorView<Real>>(tau);
    auto A0v = A0.view();
    auto Av = A.view();

    for (int b : batch_items) {
        auto A0b = A0v.batch_item(b);
        auto Ab = Av.batch_item(b);
        auto db = dv.batch_item(b);
        auto eb = ev.batch_item(b);
        auto taub = tauv.batch_item(b);

        const auto Q = build_q_from_sytrd_lower<B>(*this->ctx, Ab, taub, n);

        Matrix<Real, MatrixFormat::Dense> AQ(n, n, /*batch=*/1);
        Matrix<Real, MatrixFormat::Dense> Tmat(n, n, /*batch=*/1);
        gemm<B>(*this->ctx, A0b, Q.view(), AQ.view(), Real(1), Real(0), Transpose::NoTrans, Transpose::NoTrans).wait();
        gemm<B>(*this->ctx, Q.view(), AQ.view(), Tmat.view(), Real(1), Real(0), Transpose::Trans, Transpose::NoTrans).wait();

        assert_tridiagonal_matches(Tmat.view(), n, db, eb, tol);
    }
}

TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower33) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 33;
    const int batch = 64;
    const int nb = 8;
    const Real tol = tol_for<Real>();

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/1337);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    std::vector<int> batch_items;
    batch_items.push_back(0);
    if (batch > 1) batch_items.push_back(batch / 2);
    if (batch > 2) batch_items.push_back(batch - 1);

    auto dv = static_cast<VectorView<Real>>(d);
    auto ev = static_cast<VectorView<Real>>(e);
    auto tauv = static_cast<VectorView<Real>>(tau);
    auto A0v = A0.view();
    auto Av = A.view();

    for (int b : batch_items) {
        auto A0b = A0v.batch_item(b);
        auto Ab = Av.batch_item(b);
        auto db = dv.batch_item(b);
        auto eb = ev.batch_item(b);
        auto taub = tauv.batch_item(b);

        const auto Q = build_q_from_sytrd_lower<B>(*this->ctx, Ab, taub, n);

        Matrix<Real, MatrixFormat::Dense> AQ(n, n, /*batch=*/1);
        Matrix<Real, MatrixFormat::Dense> Tmat(n, n, /*batch=*/1);
        gemm<B>(*this->ctx, A0b, Q.view(), AQ.view(), Real(1), Real(0), Transpose::NoTrans, Transpose::NoTrans).wait();
        gemm<B>(*this->ctx, Q.view(), AQ.view(), Tmat.view(), Real(1), Real(0), Transpose::Trans, Transpose::NoTrans).wait();

        assert_tridiagonal_matches(Tmat.view(), n, db, eb, tol);
    }
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
