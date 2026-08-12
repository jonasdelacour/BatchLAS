#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

// Symmetric tridiagonal Toeplitz: d_i = a, e_i = b. Eigenvalues are known in
// closed form, which gives an exact reference independent of any other solver.
//   lambda_k = a + 2*b*cos(k*pi/(n+1)),  k = 1..n
template <typename Real>
std::vector<Real> toeplitz_reference(int n, Real a, Real b) {
    std::vector<Real> ref(n);
    for (int k = 1; k <= n; ++k) {
        ref[k - 1] = a + Real(2) * b * static_cast<Real>(std::cos(M_PI * k / (n + 1)));
    }
    std::sort(ref.begin(), ref.end());
    return ref;
}

template <typename Real>
struct StebzFixtureTraits {
    static constexpr Real tight() { return std::is_same_v<Real, float> ? Real(2e-4) : Real(1e-11); }
};

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
    stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    for (int bi = 0; bi < batch; ++bi) {
        const Real scale = Real(1) + Real(bi) / Real(10);
        const auto ref = toeplitz_reference<Real>(n, a * scale, b * scale);
        EXPECT_EQ(m[bi], n) << "wrong count for batch " << bi;
        for (int i = 0; i < n; ++i) {
            const Real got = w[bi * n + i];
            EXPECT_NEAR(std::abs(got - ref[i]) / std::max<Real>(std::abs(ref[i]), Real(1)),
                        Real(0), StebzFixtureTraits<Real>::tight())
                << "batch " << bi << " index " << i << ": got " << got << " want " << ref[i];
        }
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
        stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                       VectorView<Real>(w.data(), k, batch, 1, k),
                                       m.to_span(), ws, params);
        this->ctx->wait();

        for (int bi = 0; bi < batch; ++bi) {
            const Real scale = Real(1) + Real(bi) / Real(10);
            const auto ref = toeplitz_reference<Real>(n, a * scale, b * scale);
            EXPECT_EQ(m[bi], k);
            for (int i = 0; i < k; ++i) {
                const Real got = w[bi * k + i];
                const Real want = ref[c.il + i];
                EXPECT_NEAR(std::abs(got - want) / std::max<Real>(std::abs(want), Real(1)),
                            Real(0), StebzFixtureTraits<Real>::tight())
                    << "range [" << c.il << "," << c.iu << "] batch " << bi
                    << " index " << i << ": got " << got << " want " << want;
            }
        }
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

    stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w_asc.data(), k, batch, 1, k),
                                   m.to_span(), ws, params);
    params.order = SortOrder::Descending;
    stebz(*this->ctx, tb.d_view(), tb.e_view(),
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
    const auto ref = toeplitz_reference<Real>(n, a, b);

    // An interval strictly between two reference eigenvalues, so the expected
    // count is unambiguous.
    const Real vl = (ref[9] + ref[10]) / Real(2);
    const Real vu = (ref[29] + ref[30]) / Real(2);
    const int expected_count = 20; // indices 10..29

    UnifiedVector<Real> w(n * batch);
    UnifiedVector<int32_t> m(batch);

    StebzParams<Real> params;
    params.range = EigenRangeType::Value;
    params.vl = vl;
    params.vu = vu;

    auto ws = UnifiedVector<std::byte>(
        stebz_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, batch, params));
    stebz(*this->ctx, tb.d_view(), tb.e_view(),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    ASSERT_EQ(m[0], expected_count);
    for (int i = 0; i < expected_count; ++i) {
        const Real want = ref[10 + i];
        EXPECT_NEAR(std::abs(w[i] - want) / std::max<Real>(std::abs(want), Real(1)),
                    Real(0), StebzFixtureTraits<Real>::tight())
            << "index " << i;
    }
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
    stebz(*this->ctx,
                                   VectorView<Real>(d.data(), n, batch, 1, n),
                                   VectorView<Real>(e.data(), n - 1, batch, 1, n - 1),
                                   VectorView<Real>(w.data(), n, batch, 1, n),
                                   m.to_span(), ws, params);
    this->ctx->wait();

    // Each half is a 16x16 Toeplitz(2, -1); the spectrum is that, doubled.
    auto half = toeplitz_reference<Real>(n / 2, Real(2), Real(-1));
    std::vector<Real> ref;
    for (Real v : half) { ref.push_back(v); ref.push_back(v); }
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(m[0], n);
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(std::abs(w[i] - ref[i]) / std::max<Real>(std::abs(ref[i]), Real(1)),
                    Real(0), StebzFixtureTraits<Real>::tight())
            << "index " << i << ": got " << w[i] << " want " << ref[i];
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
