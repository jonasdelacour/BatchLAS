#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename Real>
struct SteinTol {
    // Residual ||T v - lambda v|| / ||T||, and orthogonality ||V^T V - I||.
    static Real residual() { return std::is_same_v<Real, float> ? Real(2e-4) : Real(1e-10); }
    static Real ortho() { return std::is_same_v<Real, float> ? Real(2e-4) : Real(1e-10); }
};

template <typename Real>
class SteinTest : public ::testing::Test {
protected:
    void SetUp() override { ctx = std::make_shared<Queue>(Device::default_device()); }

    // Runs stebz + stein over an index range and checks the two properties that
    // matter for SYEVX: small residual per pair, and orthonormal columns.
    void CheckTridiag(const std::vector<Real>& d_host,
                      const std::vector<Real>& e_host,
                      int n, int batch, int il, int iu,
                      Real residual_tol, Real ortho_tol) {
        const int k = iu - il + 1;

        UnifiedVector<Real> d(n * batch), e(std::max(0, n - 1) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) d[b * n + i] = d_host[i];
            for (int i = 0; i < n - 1; ++i) e[b * (n - 1) + i] = e_host[i];
        }
        auto d_view = VectorView<Real>(d.data(), n, batch, 1, n);
        auto e_view = VectorView<Real>(e.data(), std::max(0, n - 1), batch, 1, std::max(0, n - 1));

        UnifiedVector<Real> w(k * batch);
        UnifiedVector<int32_t> m(batch);
        StebzParams<Real> bp;
        bp.range = EigenRangeType::Index;
        bp.il = il;
        bp.iu = iu;
        auto bws = UnifiedVector<std::byte>(
            stebz_buffer_size<test_utils::gpu_backend, Real>(*ctx, n, batch, bp));
        stebz(*ctx, d_view, e_view,
                                       VectorView<Real>(w.data(), k, batch, 1, k),
                                       m.to_span(), bws, bp);
        ctx->wait();

        Matrix<Real, MatrixFormat::Dense> Z(n, k, batch);
        SteinParams<Real> sp;
        auto sws = UnifiedVector<std::byte>(
            stein_buffer_size<test_utils::gpu_backend, Real>(*ctx, n, k, batch, sp));
        stein(*ctx, d_view, e_view,
                                       VectorView<Real>(w.data(), k, batch, 1, k),
                                       k, Z.view(), sws, sp);
        ctx->wait();

        // ||T||_inf for scaling the residual.
        Real tnorm = 0;
        for (int i = 0; i < n; ++i) {
            const Real left = (i > 0) ? std::abs(e_host[i - 1]) : Real(0);
            const Real right = (i < n - 1) ? std::abs(e_host[i]) : Real(0);
            tnorm = std::max(tnorm, std::abs(d_host[i]) + left + right);
        }

        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < k; ++j) {
                const Real lambda = w[b * k + j];
                Real res2 = 0;
                for (int i = 0; i < n; ++i) {
                    Real tv = d_host[i] * Z.view()(i, j, b);
                    if (i > 0) tv += e_host[i - 1] * Z.view()(i - 1, j, b);
                    if (i < n - 1) tv += e_host[i] * Z.view()(i + 1, j, b);
                    const Real r = tv - lambda * Z.view()(i, j, b);
                    res2 += r * r;
                }
                EXPECT_LE(std::sqrt(res2) / tnorm, residual_tol)
                    << "residual too large, batch " << b << " vector " << j
                    << " (lambda=" << lambda << ")";
            }

            // Orthonormality of the returned block.
            for (int i = 0; i < k; ++i) {
                for (int j = i; j < k; ++j) {
                    Real dot = 0;
                    for (int r = 0; r < n; ++r) dot += Z.view()(r, i, b) * Z.view()(r, j, b);
                    const Real want = (i == j) ? Real(1) : Real(0);
                    EXPECT_LE(std::abs(dot - want), ortho_tol)
                        << "orthogonality failure, batch " << b
                        << " columns (" << i << "," << j << "): dot=" << dot;
                }
            }
        }
    }

    std::shared_ptr<Queue> ctx;
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(SteinTest, RealTypes);

} // namespace

// Well-separated spectrum: inverse iteration alone should suffice.
TYPED_TEST(SteinTest, WellSeparatedSpectrum) {
    using Real = TypeParam;
    constexpr int n = 64;
    std::vector<Real> d(n), e(n - 1);
    for (int i = 0; i < n; ++i) d[i] = Real(2);
    for (int i = 0; i < n - 1; ++i) e[i] = Real(-1);

    this->CheckTridiag(d, e, n, /*batch=*/3, /*il=*/0, /*iu=*/7,
                       SteinTol<Real>::residual(), SteinTol<Real>::ortho());
}

// The other end of the spectrum, which exercises the descending-index path.
TYPED_TEST(SteinTest, TopOfSpectrum) {
    using Real = TypeParam;
    constexpr int n = 64;
    std::vector<Real> d(n), e(n - 1);
    for (int i = 0; i < n; ++i) d[i] = Real(2);
    for (int i = 0; i < n - 1; ++i) e[i] = Real(-1);

    this->CheckTridiag(d, e, n, /*batch=*/2, /*il=*/n - 8, /*iu=*/n - 1,
                       SteinTol<Real>::residual(), SteinTol<Real>::ortho());
}

// Clustered spectrum: the Wilkinson matrix W+_(2m+1) is the classic hard case for
// inverse iteration. Its largest eigenvalues come in pairs that agree to nearly
// working precision, and unlike a block-diagonal matrix the paired eigenvectors
// share support -- so they are not orthogonal for free. Without the within-cluster
// reorthogonalization of stein's phase 2 the returned vectors collapse onto each
// other. Verified: with ortho_threshold forced to 0 this test fails.
TYPED_TEST(SteinTest, WilkinsonClusterStaysOrthogonal) {
    using Real = TypeParam;
    constexpr int m = 10;
    constexpr int n = 2 * m + 1;
    std::vector<Real> d(n), e(n - 1);
    for (int i = 0; i < n; ++i) d[i] = static_cast<Real>(std::abs(i - m));
    for (int i = 0; i < n - 1; ++i) e[i] = Real(1);

    // The top 6 eigenvalues contain three near-degenerate pairs.
    this->CheckTridiag(d, e, n, /*batch=*/2, /*il=*/n - 6, /*iu=*/n - 1,
                       SteinTol<Real>::residual(), SteinTol<Real>::ortho());
}

// A graded matrix, where the entries vary over several orders of magnitude.
TYPED_TEST(SteinTest, GradedMatrix) {
    using Real = TypeParam;
    constexpr int n = 48;
    std::vector<Real> d(n), e(n - 1);
    for (int i = 0; i < n; ++i) d[i] = static_cast<Real>(std::pow(1.2, i % 20));
    for (int i = 0; i < n - 1; ++i) e[i] = static_cast<Real>(0.5 * std::pow(1.1, i % 15));

    this->CheckTridiag(d, e, n, /*batch=*/2, /*il=*/10, /*iu=*/17,
                       SteinTol<Real>::residual(), SteinTol<Real>::ortho());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
