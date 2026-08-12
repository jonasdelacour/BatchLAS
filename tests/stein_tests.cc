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

namespace {

// Shared fixture for the per-item-count overload. Builds the 1D-Laplacian
// tridiagonal (d = 2, e = -1), whose spectrum 2 - 2cos(j*pi/(n+1)) is simple and
// whose smallest gaps near the bottom are still far wider than stein's default
// clustering threshold of 1e-3 * ||T||.
template <typename Real>
struct CountsFixture {
    static constexpr int n = 64;
    static constexpr int k = 8;      // capacity: columns of Z, entries of w per item
    static constexpr int batch = 2;

    std::vector<Real> d, e;
    UnifiedVector<Real> d_dev, e_dev, w;
    UnifiedVector<int32_t> counts;
    Real tnorm = 0;

    CountsFixture()
        : d(n, Real(2)), e(n - 1, Real(-1)),
          d_dev(n * batch), e_dev((n - 1) * batch), w(k * batch), counts(batch) {
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) d_dev[b * n + i] = d[i];
            for (int i = 0; i < n - 1; ++i) e_dev[b * (n - 1) + i] = e[i];
        }
        for (int i = 0; i < n; ++i) {
            const Real left = (i > 0) ? std::abs(e[i - 1]) : Real(0);
            const Real right = (i < n - 1) ? std::abs(e[i]) : Real(0);
            tnorm = std::max(tnorm, std::abs(d[i]) + left + right);
        }
    }

    VectorView<Real> dv() { return VectorView<Real>(d_dev.data(), n, batch, 1, n); }
    VectorView<Real> ev() { return VectorView<Real>(e_dev.data(), n - 1, batch, 1, n - 1); }
    VectorView<Real> wv() { return VectorView<Real>(w.data(), k, batch, 1, k); }

    // Fills w with the k lowest eigenvalues (identical for every item) via stebz.
    void FillEigenvalues(Queue& ctx) {
        UnifiedVector<int32_t> m(batch);
        StebzParams<Real> bp;
        bp.range = EigenRangeType::Index;
        bp.il = 0;
        bp.iu = k - 1;
        auto bws = UnifiedVector<std::byte>(
            stebz_buffer_size<test_utils::gpu_backend, Real>(ctx, n, batch, bp));
        stebz<test_utils::gpu_backend>(ctx, dv(), ev(), wv(), m.to_span(), bws, bp);
        ctx.wait();
    }

    Real residual(const MatrixView<Real, MatrixFormat::Dense>& Z,
                  int b, int j, Real lambda) const {
        Real res2 = 0;
        for (int i = 0; i < n; ++i) {
            Real tv = d[i] * Z(i, j, b);
            if (i > 0) tv += e[i - 1] * Z(i - 1, j, b);
            if (i < n - 1) tv += e[i] * Z(i + 1, j, b);
            const Real r = tv - lambda * Z(i, j, b);
            res2 += r * r;
        }
        return std::sqrt(res2) / tnorm;
    }
};

} // namespace

// Per-item counts: item 0 wants all 8 vectors, item 1 wants only 3.
//
// Item 1's slots 3..7 are deliberately poisoned with values a hair above its last
// real eigenvalue -- close enough that the phase-2 gap test would fold them into
// one bogus cluster anchored on that real eigenvalue, which is exactly the shape a
// stebz value range produces when one batch item finds fewer eigenvalues than the
// capacity and the tail of the workspace is stale.
//
// Two independent properties are asserted:
//   (a) item 1's three real eigenpairs are orthonormal and have a small residual;
//   (b) item 1's unused columns 3..7 are written as EXACTLY zero (they are
//       pre-poisoned with a sentinel, so this proves stein wrote them).
//
// (b) is the discriminating assertion, and (a) is not. SYEVX_RANGE_PLAN.md 7.3
// asks for (a) alone on the theory that the phase-2 cluster walk corrupts valid
// eigenvectors; it does not. Measured on this test: with the counts bound removed
// from BOTH phases, (a) reports zero failures and (b) reports 639. The phase-2
// MGS writes only column j while reading columns i < j and cluster_start comes
// only from w(0..j), so garbage past the prefix cannot reach a valid column.
// Keep (b); it is what makes this test fail when the bound regresses.
TYPED_TEST(SteinTest, PerItemCountsIgnorePoisonedTail) {
    using Real = TypeParam;
    CountsFixture<Real> f;
    constexpr int n = CountsFixture<Real>::n;
    constexpr int k = CountsFixture<Real>::k;

    f.FillEigenvalues(*this->ctx);

    // Item 0 keeps all k; item 1 keeps 3 and gets a bogus cluster in the tail.
    const int kb1 = 3;
    f.counts[0] = k;
    f.counts[1] = kb1;
    const Real anchor = f.w[1 * k + (kb1 - 1)];
    for (int j = kb1; j < k; ++j) {
        // Well inside gap_tol = 1e-3 * ||T|| = 4e-3, and inside the phase-1
        // degeneracy-separation window too, so both phases would treat these as
        // continuations of the last real eigenvalue.
        f.w[1 * k + j] = anchor + Real(1e-7) * Real(j - kb1 + 1);
    }

    Matrix<Real, MatrixFormat::Dense> Z(n, k, CountsFixture<Real>::batch);
    const Real sentinel = Real(-12345);
    for (int b = 0; b < CountsFixture<Real>::batch; ++b)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < n; ++i) Z.view()(i, j, b) = sentinel;

    SteinParams<Real> sp;
    auto sws = UnifiedVector<std::byte>(
        stein_buffer_size<test_utils::gpu_backend, Real>(
            *this->ctx, n, k, CountsFixture<Real>::batch, sp));
    stein<test_utils::gpu_backend>(*this->ctx, f.dv(), f.ev(), f.wv(), k,
                                   Span<const int32_t>(f.counts.data(), f.counts.size()),
                                   Z.view(), sws, sp);
    this->ctx->wait();

    const Real res_tol = SteinTol<Real>::residual();
    const Real ortho_tol = SteinTol<Real>::ortho();

    for (int b = 0; b < CountsFixture<Real>::batch; ++b) {
        const int kb = f.counts[b];

        // (a) the declared prefix is a genuine orthonormal invariant-subspace basis.
        for (int j = 0; j < kb; ++j) {
            const Real lambda = f.w[b * k + j];
            EXPECT_LE(f.residual(Z.view(), b, j, lambda), res_tol)
                << "residual too large, batch " << b << " vector " << j
                << " (lambda=" << lambda << ")";
        }
        for (int i = 0; i < kb; ++i) {
            for (int j = i; j < kb; ++j) {
                Real dot = 0;
                for (int r = 0; r < n; ++r) dot += Z.view()(r, i, b) * Z.view()(r, j, b);
                const Real want = (i == j) ? Real(1) : Real(0);
                EXPECT_LE(std::abs(dot - want), ortho_tol)
                    << "orthogonality failure, batch " << b
                    << " columns (" << i << "," << j << "): dot=" << dot;
            }
        }

        // (b) everything past the prefix is exactly zero -- not the sentinel, not
        // an inverse-iteration result on a garbage shift, and above all not NaN.
        for (int j = kb; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                const Real z = Z.view()(i, j, b);
                EXPECT_TRUE(std::isfinite(z))
                    << "non-finite in unused column, batch " << b << " column " << j
                    << " row " << i;
                EXPECT_EQ(z, Real(0))
                    << "unused column not zeroed, batch " << b << " column " << j
                    << " row " << i << " (value " << z << ")";
            }
        }
    }
}

// The counts overload with counts[b] == k for every item must reproduce the
// counts-less overload bit for bit: same kernels, same inputs, same launch
// geometry, and the deliberately fixed LCG seed makes inverse iteration
// reproducible. A difference here is a bounding bug, not a rounding difference.
//
// Deliberately NOT asserted here: that `stein_all_counts` (an empty span) matches
// the counts-less overload. It cannot fail -- the counts-less overload is DEFINED
// as a forwarder that passes an empty span (stein.cc), so the two are literally
// the same call, and comparing them would measure run-to-run determinism while
// reading as though it validated a bound. The second half of this test is a case
// where the counts genuinely bind differently, which is what makes the first half
// worth stating.
TYPED_TEST(SteinTest, FullCountsMatchesUniformOverload) {
    using Real = TypeParam;
    CountsFixture<Real> f;
    constexpr int n = CountsFixture<Real>::n;
    constexpr int k = CountsFixture<Real>::k;
    constexpr int batch = CountsFixture<Real>::batch;
    static_assert(batch >= 2, "the short-item half of this test needs a second item");

    // The reason the empty-span comparison is not here, as an assertion rather
    // than as a claim in a comment: `stein_all_counts` IS the empty span the
    // counts-less overload forwards, so the two calls are indistinguishable.
    ASSERT_TRUE(stein_all_counts.empty());

    f.FillEigenvalues(*this->ctx);
    for (int b = 0; b < batch; ++b) f.counts[b] = k;

    SteinParams<Real> sp;
    auto sws = UnifiedVector<std::byte>(
        stein_buffer_size<test_utils::gpu_backend, Real>(*this->ctx, n, k, batch, sp));

    Matrix<Real, MatrixFormat::Dense> Z_uniform(n, k, batch);
    stein<test_utils::gpu_backend>(*this->ctx, f.dv(), f.ev(), f.wv(), k,
                                   Z_uniform.view(), sws, sp);
    this->ctx->wait();

    Matrix<Real, MatrixFormat::Dense> Z_counts(n, k, batch);
    stein<test_utils::gpu_backend>(*this->ctx, f.dv(), f.ev(), f.wv(), k,
                                   Span<const int32_t>(f.counts.data(), f.counts.size()),
                                   Z_counts.view(), sws, sp);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                EXPECT_EQ(Z_counts.view()(i, j, b), Z_uniform.view()(i, j, b))
                    << "counts==k diverged from the uniform overload at ("
                    << i << "," << j << "," << b << ")";
            }
        }
    }

    // Now the discriminating half: one item drops a single column. The bound must
    // bind EXACTLY there and nowhere else -- item 0 is untouched, item 1's columns
    // 0..k-2 are untouched, and only its column k-1 changes, to exact zero. A
    // bound that was off by one, applied to the wrong item, or applied to the
    // whole batch shows up as a difference in one of the first two.
    for (int b = 0; b < batch; ++b) f.counts[b] = k;
    f.counts[1] = k - 1;

    Matrix<Real, MatrixFormat::Dense> Z_short(n, k, batch);
    stein<test_utils::gpu_backend>(*this->ctx, f.dv(), f.ev(), f.wv(), k,
                                   Span<const int32_t>(f.counts.data(), f.counts.size()),
                                   Z_short.view(), sws, sp);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
            const bool dropped = (b == 1 && j == k - 1);
            for (int i = 0; i < n; ++i) {
                if (dropped) {
                    EXPECT_EQ(Z_short.view()(i, j, b), Real(0))
                        << "the dropped column was not zeroed at (" << i << "," << j << "," << b << ")";
                } else {
                    EXPECT_EQ(Z_short.view()(i, j, b), Z_uniform.view()(i, j, b))
                        << "a kept column moved when a different column was dropped at ("
                        << i << "," << j << "," << b << ")";
                }
            }
        }
    }
    // Guard against the whole comparison being vacuous: the dropped column has to
    // have been NON-zero in the uniform run, or "it is zero now" proves nothing.
    Real dropped_norm2 = 0;
    for (int i = 0; i < n; ++i) {
        const Real v = Z_uniform.view()(i, k - 1, 1);
        dropped_norm2 += v * v;
    }
    EXPECT_GT(dropped_norm2, Real(0.5)) << "the uniform run left the dropped column empty too";
}

// counts[b] == 0: the item wants nothing (an empty value interval). Every column
// must come back zero and nothing may be run on its garbage shifts.
TYPED_TEST(SteinTest, ZeroCountYieldsZeroColumns) {
    using Real = TypeParam;
    CountsFixture<Real> f;
    constexpr int n = CountsFixture<Real>::n;
    constexpr int k = CountsFixture<Real>::k;

    f.FillEigenvalues(*this->ctx);
    f.counts[0] = k;
    f.counts[1] = 0;
    // Item 1's whole w is now meaningless; make that explicit.
    for (int j = 0; j < k; ++j) f.w[1 * k + j] = Real(0);

    Matrix<Real, MatrixFormat::Dense> Z(n, k, CountsFixture<Real>::batch);
    for (int b = 0; b < CountsFixture<Real>::batch; ++b)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < n; ++i) Z.view()(i, j, b) = Real(-12345);

    SteinParams<Real> sp;
    auto sws = UnifiedVector<std::byte>(
        stein_buffer_size<test_utils::gpu_backend, Real>(
            *this->ctx, n, k, CountsFixture<Real>::batch, sp));
    stein<test_utils::gpu_backend>(*this->ctx, f.dv(), f.ev(), f.wv(), k,
                                   Span<const int32_t>(f.counts.data(), f.counts.size()),
                                   Z.view(), sws, sp);
    this->ctx->wait();

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(Z.view()(i, j, 1), Real(0))
                << "zero-count item wrote a non-zero at (" << i << "," << j << ")";
        }
    }
    // Item 0 is unaffected by its neighbour's empty request.
    for (int j = 0; j < k; ++j) {
        EXPECT_LE(f.residual(Z.view(), 0, j, f.w[j]), SteinTol<Real>::residual())
            << "batch 0 vector " << j << " damaged by batch 1's zero count";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
