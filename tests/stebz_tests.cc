// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <tests/test_utils.hh>
#include <tests/eigen_verify.hh>

using namespace batchlas;

// Runs on test_utils::gpu_backend, which exists only with a GPU backend.
#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND

namespace {

// Symmetric tridiagonal Toeplitz: d_i = a, e_i = b. Eigenvalues are known in
// closed form, which gives an exact reference independent of any other solver.
//   lambda_k = a + 2*b*cos(k*pi/(n+1)),  k = 1..n
// Evaluated in double; a caller that stores it as Real adds one rounding (eps |lambda|), far
// below Check::values.
inline std::vector<double> toeplitz_reference(int n, double a, double b) {
    std::vector<double> ref(n);
    for (int k = 1; k <= n; ++k) {
        ref[k - 1] = a + 2.0 * b * std::cos(M_PI * k / (n + 1));
    }
    std::sort(ref.begin(), ref.end());
    return ref;
}

// Builds a batch of tridiagonal matrices; batch item b is scaled by (1 + b/10) so
// that the batch is not trivially uniform.
template <typename Real>
struct TriDiagBatch {
    int n;
    int batch;
    UnifiedVector<Real> d;
    UnifiedVector<Real> e;

    TriDiagBatch(int n_, int batch_, Real a, Real b)
        : n(n_), batch(batch_), d(n_ * batch_), e(std::max(0, n_ - 1) * batch_) {
        for (int bi = 0; bi < batch; ++bi) {
            const Real scale = Real(1) + Real(bi) / Real(10);
            for (int i = 0; i < n; ++i) d[bi * n + i] = a * scale;
            for (int i = 0; i < n - 1; ++i) e[bi * (n - 1) + i] = b * scale;
        }
    }

    VectorView<Real> d_view() { return VectorView<Real>(d.data(), n, batch, 1, n); }
    VectorView<Real> e_view() { return VectorView<Real>(e.data(), std::max(0, n - 1), batch, 1, std::max(0, n - 1)); }
};

template <typename Real>
class StebzTest : public ::testing::Test {
protected:
    void SetUp() override { ctx = std::make_shared<Queue>(Device::default_device()); }
    std::shared_ptr<Queue> ctx;
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(StebzTest, RealTypes);

} // namespace

// Full spectrum must match the closed form exactly, for every batch item.
TYPED_TEST(StebzTest, AllRangeMatchesClosedForm) {
    using Real = TypeParam;
    constexpr int n = 97;
    constexpr int batch = 4;
    const Real a = Real(2), b = Real(-1);

    TriDiagBatch<Real> tb(n, batch, a, b);
    UnifiedVector<Real> w(n * batch);
    UnifiedVector<int32_t> m(batch);

    StebzParams<Real> params;
    params.range = EigenRangeType::All;

    auto ws = UnifiedVector<std::byte>(
        stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));
    (void)stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    UnifiedVector<Real> w_ref(n * batch);
    for (int bi = 0; bi < batch; ++bi) {
        const double scale = 1.0 + bi / 10.0;
        const auto ref = toeplitz_reference(n, a * scale, b * scale);
        for (int i = 0; i < n; ++i) w_ref[bi * n + i] = static_cast<Real>(ref[i]);
    }
    test_utils::expect_eigenvalues_agree<Real>(w, w_ref, n, batch);
    for (int bi = 0; bi < batch; ++bi) {
        EXPECT_EQ(m[bi], n) << "wrong count for batch " << bi;
        // Ascending by construction.
        for (int i = 1; i < n; ++i) {
            EXPECT_LE(w[bi * n + i - 1], w[bi * n + i]) << "not ascending at " << i;
        }
    }
}

// An index subset must return exactly the same values the full solve does --
// this is the property SYEVX depends on.
TYPED_TEST(StebzTest, IndexRangeMatchesFullSpectrum) {
    using Real = TypeParam;
    constexpr int n = 64;
    constexpr int batch = 3;
    const Real a = Real(4), b = Real(1.5);

    TriDiagBatch<Real> tb(n, batch, a, b);

    struct Case { int il; int iu; };
    const Case cases[] = {{0, 4}, {n - 5, n - 1}, {20, 27}, {31, 31}};

    for (const auto& c : cases) {
        const int k = c.iu - c.il + 1;
        UnifiedVector<Real> w(k * batch);
        UnifiedVector<int32_t> m(batch);

        StebzParams<Real> params;
        params.range = EigenRangeType::Index;
        params.il = c.il;
        params.iu = c.iu;

        auto ws = UnifiedVector<std::byte>(
            stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));
        (void)stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                       VectorView<Real>(w.data(), k, batch, 1, k),
                                       m.to_span(), ws, params);
        this->ctx->wait();

        SCOPED_TRACE("range [" + std::to_string(c.il) + "," + std::to_string(c.iu) + "]");
        UnifiedVector<Real> w_ref(k * batch);
        for (int bi = 0; bi < batch; ++bi) {
            const double scale = 1.0 + bi / 10.0;
            const auto ref = toeplitz_reference(n, a * scale, b * scale);
            EXPECT_EQ(m[bi], k);
            for (int i = 0; i < k; ++i) w_ref[bi * k + i] = static_cast<Real>(ref[c.il + i]);
        }
        test_utils::expect_eigenvalues_agree<Real>(w, w_ref, k, batch);
    }
}

// Descending order must reverse the block, not change which values are returned.
TYPED_TEST(StebzTest, DescendingOrderReversesTheBlock) {
    using Real = TypeParam;
    constexpr int n = 48;
    constexpr int batch = 2;
    constexpr int k = 6;
    const Real a = Real(1), b = Real(2);

    TriDiagBatch<Real> tb(n, batch, a, b);
    UnifiedVector<Real> w_asc(k * batch), w_desc(k * batch);
    UnifiedVector<int32_t> m(batch);

    StebzParams<Real> params;
    params.range = EigenRangeType::Index;
    params.il = n - k;
    params.iu = n - 1;

    auto ws = UnifiedVector<std::byte>(
        stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));

    (void)stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w_asc.data(), k, batch, 1, k),
                                   m.to_span(), ws, params);
    params.order = SortOrder::Descending;
    (void)stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w_desc.data(), k, batch, 1, k),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    for (int bi = 0; bi < batch; ++bi) {
        for (int i = 0; i < k; ++i) {
            EXPECT_EQ(w_asc[bi * k + i], w_desc[bi * k + (k - 1 - i)])
                << "batch " << bi << " index " << i;
        }
    }
}

// A value range must find exactly the eigenvalues inside it.
TYPED_TEST(StebzTest, ValueRangeSelectsInterval) {
    using Real = TypeParam;
    constexpr int n = 50;
    constexpr int batch = 1;
    const Real a = Real(0), b = Real(1);

    TriDiagBatch<Real> tb(n, batch, a, b);
    const auto ref = toeplitz_reference(n, a, b);

    // An interval strictly between two reference eigenvalues, so the expected
    // count is unambiguous.
    const Real vl = static_cast<Real>((ref[9] + ref[10]) / 2);
    const Real vu = static_cast<Real>((ref[29] + ref[30]) / 2);
    const int expected_count = 20; // indices 10..29

    UnifiedVector<Real> w(n * batch);
    UnifiedVector<int32_t> m(batch);

    StebzParams<Real> params;
    params.range = EigenRangeType::Value;
    params.vl = vl;
    params.vu = vu;

    auto ws = UnifiedVector<std::byte>(
        stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));
    (void)stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    ASSERT_EQ(m[0], expected_count);
    UnifiedVector<Real> w_ref(expected_count);
    for (int i = 0; i < expected_count; ++i) w_ref[i] = static_cast<Real>(ref[10 + i]);
    test_utils::expect_eigenvalues_agree<Real>(w, w_ref, expected_count, 1);
}

// Clustered and degenerate spectra are where bisection's guard against a zero
// pivot matters: without it the Sturm count stops being monotone in x.
TYPED_TEST(StebzTest, HandlesRepeatedAndZeroOffDiagonals) {
    using Real = TypeParam;
    constexpr int n = 32;
    constexpr int batch = 1;

    UnifiedVector<Real> d(n * batch), e(std::max(0, n - 1) * batch);
    // Two decoupled identical blocks: every eigenvalue has multiplicity two, and
    // the coupling element is exactly zero.
    for (int i = 0; i < n; ++i) d[i] = Real(2);
    for (int i = 0; i < n - 1; ++i) e[i] = (i == n / 2 - 1) ? Real(0) : Real(-1);

    UnifiedVector<Real> w(n * batch);
    UnifiedVector<int32_t> m(batch);

    StebzParams<Real> params;
    params.range = EigenRangeType::All;

    auto ws = UnifiedVector<std::byte>(
        stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));
    (void)stebz(*this->ctx,
                                   VectorView<Real>(d.data(), n, batch, 1, n),
                                   VectorView<Real>(e.data(), n - 1, batch, 1, n - 1),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    // Each half is a 16x16 Toeplitz(2, -1); the spectrum is that, doubled.
    const auto half = toeplitz_reference(n / 2, 2.0, -1.0);
    UnifiedVector<Real> w_ref(n);
    for (int i = 0; i < n; ++i) w_ref[i] = static_cast<Real>(half[i / 2]);

    ASSERT_EQ(m[0], n);
    test_utils::expect_eigenvalues_agree<Real>(w, w_ref, n, 1);
}

#endif  // BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
