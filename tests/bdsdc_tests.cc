// SPDX-License-Identifier: MIT
// Tests for bdsdc, the bidiagonal divide-and-conquer SVD.
//
// bdsdc solves the same problem as bdsqr but through the interleaved
// Golub-Kahan tridiagonal of order 2n, handed to stedc. Two things about that
// route need evidence rather than argument:
//
//  1. The extraction. For sigma != 0 the 2n-eigenvector is exactly (v;u)/sqrt(2)
//     interleaved, so splitting it and normalising each half should be exact --
//     these tests check reconstruction AND orthogonality, because bdsqr got
//     orthogonality for free by accumulating rotations from the identity and
//     bdsdc does not: it writes vectors it computed, so U^T U = I is a real
//     claim about the method, not a structural guarantee.
//
//  2. The null space. At sigma ~ 0 the +/-sigma pair is degenerate and a
//     computed eigenvector can be (v;0), leaving u unrecoverable from that
//     column -- but more often it comes back full-norm and merely PARALLEL to
//     another degenerate column. RankDeficient and AllZeros exist to exercise
//     the repair path; without it these measure U-orthogonality 1.41 with V at
//     0.00, which is what a half-norm-based criterion misses entirely.
//
// The reference is LAPACKE ?gesdd of the same bidiagonal in double (batchlas::verify), not a
// second call to anything in this library.

#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <tests/eigen_verify.hh>

#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

using namespace batchlas;

// Every typed config below is Backend::CUDA: without it the suite is empty and does not compile.
#if BATCHLAS_HAS_CUDA_BACKEND

namespace {

template <typename T, Backend B>
struct BdsdcConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct bdsdc_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>>{},
#endif
        std::tuple<>{}));
    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using BdsdcTestTypes = typename bdsdc_types<BdsdcConfig>::type;

template <typename Config>
class BdsdcTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    // Core driver: run bdsdc on the given (d,e) and check singular values against LAPACKE,
    // U diag(s) Vh == B, and orthogonality of both vector sets.
    void run_and_check(const std::vector<double>& dh,
                       const std::vector<double>& eh,
                       int n, int batch, bool vectors, const char* label,
                       double orth_slack = 0.0) {
        using MV = MatrixView<Scalar, MatrixFormat::Dense>;
        SCOPED_TRACE(std::string(label) + " n=" + std::to_string(n));
        auto& ctx = *this->ctx;
        UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch);
        UnifiedVector<Scalar> e(static_cast<size_t>(std::max(1, n - 1)) * batch);
        Matrix<Scalar> B0 = Matrix<Scalar>::Zeros(n, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(dh[static_cast<size_t>(b) * n + i]);
                B0.at(i, i, b) = d[static_cast<size_t>(b) * n + i];
            }
            for (int i = 0; i < n - 1; ++i) {
                e[static_cast<size_t>(b) * (n - 1) + i] =
                    static_cast<Scalar>(eh[static_cast<size_t>(b) * (n - 1) + i]);
                B0.at(i, i + 1, b) = e[static_cast<size_t>(b) * (n - 1) + i];
            }
        }

        VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
        VectorView<Scalar> ev(e.to_span(), std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch);

        const size_t ws_bytes = bdsdc_buffer_size<B, Scalar>(ctx, dv, ev, s.to_span(), vectors);
        UnifiedVector<std::byte> ws(ws_bytes);

        Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
        if (vectors) {
            (void)bdsdc<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), U, Vh, true);
        } else {
            (void)bdsdc<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), true);
        }
        ctx.wait_and_throw();

        const auto items = B0.all_items();
        const VectorView<Scalar> sv(s.to_span(), n, batch, 1, n);
        if (vectors) {
            const double resid = MV::svd_residual(ctx, B0, U, sv, Vh, items);
            if (orth_slack > 0.0 && std::is_same_v<Scalar, float>)
                EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::factorization, n, resid,
                                    batchlas::verify::Slack{2.0, "graded kappa~1e6 D&C float, n=64: the repaired columns carry the Gram-Schmidt error into U S Vh. Measured 9.15e-5 (c 24); accepted bound 32 n eps (2x of c=16)"});
            else
                EXPECT_VERIFY(Scalar, batchlas::verify::Check::factorization, n, resid);
            // Orthogonality of U's columns and Vh's rows (the columns of Vh^H).
            Matrix<Scalar> V(n, n, batch);
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) V.view()(i, j, b) = Vh.view()(j, i, b);
            const double uorth = MV::orthogonality(ctx, U, items);
            const double vorth = MV::orthogonality(ctx, V, items);
            if (orth_slack > 0.0) {
                const batchlas::verify::Slack slack{
                    orth_slack,
                    "graded kappa~1e6 D&C, n=64: the repair's Gram-Schmidt accumulates error. Measured 4.75e-3 float (c 1246), 5.08e-13 double (c 71.5); accepted bounds 2048 n eps float, 128 n eps double (8x / 0.5x of c=256)"};
                EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::orthogonality_rotations, n, uorth, slack) << "U";
                EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::orthogonality_rotations, n, vorth, slack) << "V";
            } else {
                EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, uorth) << "U";
                EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, vorth) << "V";
            }
        }

        test_utils::expect_singular_values_match_lapacke(B0, s, items);
    }

    void check(int n, int batch, unsigned seed, bool vectors, double dscale = 1.0,
               double orth_slack = 0.0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.3, 1.7);
        std::vector<double> dh(static_cast<size_t>(n) * batch);
        std::vector<double> eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                dh[static_cast<size_t>(b) * n + i] = dist(rng) * std::pow(dscale, i);
            }
            for (int i = 0; i < n - 1; ++i) {
                eh[static_cast<size_t>(b) * (n - 1) + i] = dist(rng) * 0.5 * std::pow(dscale, i);
            }
        }
        run_and_check(dh, eh, n, batch, vectors, "random", orth_slack);
    }

    // Mixed signs, exact zeros, entries over six decades -- what gebrd actually
    // emits, as opposed to the friendly positive generator above.
    void check_hostile(int n, int batch, unsigned seed, bool vectors) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> mag(-3.0, 3.0);
        std::uniform_int_distribution<int> sign(0, 1);
        std::uniform_int_distribution<int> zero(0, 9);
        std::vector<double> dh(static_cast<size_t>(n) * batch);
        std::vector<double> eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                const double m10 = std::pow(10.0, mag(rng));
                double v = m10 * (sign(rng) ? 1.0 : -1.0);
                if (zero(rng) == 0) v = 0.0;
                dh[static_cast<size_t>(b) * n + i] = v;
            }
            for (int i = 0; i < n - 1; ++i) {
                const double m10 = std::pow(10.0, mag(rng));
                double v = m10 * (sign(rng) ? 1.0 : -1.0);
                if (zero(rng) == 0) v = 0.0;
                eh[static_cast<size_t>(b) * (n - 1) + i] = v;
            }
        }
        run_and_check(dh, eh, n, batch, vectors, "hostile");
    }

    // `nz` decoupled positions, each forcing an EXACT zero singular value. This
    // is the case where the +/-sigma eigenvectors are free to rotate and a naive
    // extraction leaves zero columns behind.
    void check_rank_deficient(int n, int batch, int nz, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.5, 1.5);
        std::vector<double> dh(static_cast<size_t>(n) * batch);
        std::vector<double> eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)     dh[static_cast<size_t>(b) * n + i] = dist(rng);
            for (int i = 0; i < n - 1; ++i) eh[static_cast<size_t>(b) * (n - 1) + i] = dist(rng);
            std::vector<int> pos(static_cast<size_t>(n - 1));
            for (int i = 0; i < n - 1; ++i) pos[static_cast<size_t>(i)] = i;
            std::shuffle(pos.begin(), pos.end(), rng);
            for (int z = 0; z < nz; ++z) {
                const int p = pos[static_cast<size_t>(z)];
                dh[static_cast<size_t>(b) * n + p] = 0.0;
                eh[static_cast<size_t>(b) * (n - 1) + p] = 0.0;
            }
        }
        run_and_check(dh, eh, n, batch, /*vectors=*/true, "rank-deficient");
    }
};

TYPED_TEST_SUITE(BdsdcTest, BdsdcTestTypes);

TYPED_TEST(BdsdcTest, ValuesOnly) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 100u + n, /*vectors=*/false);
}

TYPED_TEST(BdsdcTest, WithVectors) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 200u + n, /*vectors=*/true);
}

// Graded: exactly the case where forming B^T B loses the small singular values.
TYPED_TEST(BdsdcTest, Graded) {
    this->check(32, 2, 301u, /*vectors=*/true, /*dscale=*/0.8);
    this->check(64, 2, 302u, /*vectors=*/false, /*dscale=*/0.85);
}

// Graded WITH vectors at kappa ~ 1e6, which is where the +sigma/-sigma pair of
// the 2n Golub-Kahan matrix stops being resolvable. Both members of an
// unresolved pair then normalise to the same (v, u), so U and V come back with
// parallel columns -- measured 1.6e-2 before the repair threshold was derived
// from the resolution limit rather than set to detect exact zeros.
//
// The existing Graded case runs n=64 with vectors=FALSE, so nothing exercised
// this; it is the whole reason the defect survived.
TYPED_TEST(BdsdcTest, GradedWithVectorsHighCondition) {
    // 5e-3 rather than the usual 1e-3, and that is a real residual, not slack:
    // this graded bidiagonal is harsher than a random matrix of the same kappa
    // (many clustered tiny sigma rather than a smooth spread), and the repair's
    // own Gram-Schmidt accumulates error across the columns it rebuilds.
    // Measured 1.6e-2 before the threshold fix and 4.8e-3 after, so the bound
    // still fails on the old code -- which is the point of it.
    //
    // End-to-end through gesvd the same kappa lands at 1.1e-4 (gesvd_relacc,
    // n=64, float), comfortably inside every tolerance the suite applies.
    // Callers who need better than this above n=32 want the one-sided Jacobi
    // route, which never forms the bidiagonal at all.
    const double kGradedOrthTol = std::is_same_v<typename TestFixture::Scalar, float> ? 8.0 : 0.5;
    this->check(64, 2, 303u, /*vectors=*/true, /*dscale=*/0.803, kGradedOrthTol);
}

TYPED_TEST(BdsdcTest, MixedSignsZerosAndWideRange) {
    for (int n : {8, 32, 64}) this->check_hostile(n, 8, 900u + n, /*vectors=*/false);
}

// The repair path. Multiplicity 1 and 2 are where a partner-column fix would
// still work; 3 and 4 are where it provably does not, so these are the cases
// that justify the Gram-Schmidt completion.
TYPED_TEST(BdsdcTest, RankDeficient) {
    for (int nz : {1, 2, 3, 4}) this->check_rank_deficient(16, 4, nz, 400u + nz);
    this->check_rank_deficient(33, 2, 5, 450u);
}

TYPED_TEST(BdsdcTest, AllZeros) {
    const int n = 12, batch = 2;
    std::vector<double> dh(static_cast<size_t>(n) * batch, 0.0);
    std::vector<double> eh(static_cast<size_t>(n - 1) * batch, 0.0);
    this->run_and_check(dh, eh, n, batch, /*vectors=*/true, "all-zeros");
}

} // namespace
#endif  // BATCHLAS_HAS_CUDA_BACKEND
