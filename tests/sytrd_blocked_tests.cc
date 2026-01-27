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
UnifiedVector<double> netlib_ref_eigs_dense(const MatrixView<Real, MatrixFormat::Dense>& A) {
    const int n = A.rows();
    const int batch = A.batch_size();

    Queue ctx_cpu("cpu");
    auto A_d = A.template astype<double>();

    UnifiedVector<double> ref_eigs(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    const size_t ws_bytes = backend::syev_vendor_buffer_size<Backend::NETLIB, double>(
        ctx_cpu, A_d.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});
    backend::syev_vendor<Backend::NETLIB, double>(
        ctx_cpu, A_d.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span()).wait();
    ctx_cpu.wait();

    return ref_eigs;
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
    const double eig_tol = (std::is_same_v<Real, float> ? 10000.0 : 100.0) * test_utils::tolerance<double>();

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

    Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
    Tmat.view().fill_tridiag(*this->ctx, e, d, e).wait();

    const auto eig_ref = netlib_ref_eigs_dense(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense(Tmat.view());

    for (int b : batch_items) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = eig_ref[base + static_cast<std::size_t>(i)];
            double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            if constexpr (std::is_same_v<Real, float>) {
                err_tol = std::max(err_tol, 2e-6);
            }
            EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
}

TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower33) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 33;
    const int batch = 64;
    const int nb = 8;
    const double eig_tol = (std::is_same_v<Real, float> ? 10000.0 : 100.0) * test_utils::tolerance<double>();

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

    Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
    Tmat.view().fill_tridiag(*this->ctx, e, d, e).wait();

    const auto eig_ref = netlib_ref_eigs_dense(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense(Tmat.view());

    for (int b : batch_items) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = eig_ref[base + static_cast<std::size_t>(i)];
            double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            if constexpr (std::is_same_v<Real, float>) {
                err_tol = std::max(err_tol, 2e-6);
            }
            EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
