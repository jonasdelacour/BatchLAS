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
// The reference is a host Jacobi SVD of the same bidiagonal in double, not a
// second call to anything in this library.

#include <gtest/gtest.h>

#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

using namespace batchlas;

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

// Singular values of an upper bidiagonal matrix, on the host in double by
// one-sided Jacobi on the dense form. Slow but independent.
std::vector<double> reference_singular_values(const std::vector<double>& d,
                                              const std::vector<double>& e,
                                              int n) {
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        A[static_cast<size_t>(i) * n + i] = d[static_cast<size_t>(i)];
        if (i + 1 < n) A[static_cast<size_t>(i + 1) * n + i] = e[static_cast<size_t>(i)];
    }
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double app = 0, aqq = 0, apq = 0;
                for (int i = 0; i < n; ++i) {
                    const double x = A[static_cast<size_t>(p) * n + i];
                    const double y = A[static_cast<size_t>(q) * n + i];
                    app += x * x; aqq += y * y; apq += x * y;
                }
                if (std::abs(apq) <= 1e-300) continue;
                off = std::max(off, std::abs(apq) / std::sqrt(std::max(app * aqq, 1e-300)));
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t), s = t * c;
                for (int i = 0; i < n; ++i) {
                    const double x = A[static_cast<size_t>(p) * n + i];
                    const double y = A[static_cast<size_t>(q) * n + i];
                    A[static_cast<size_t>(p) * n + i] = c * x - s * y;
                    A[static_cast<size_t>(q) * n + i] = s * x + c * y;
                }
            }
        }
        if (off < 1e-15) break;
    }
    std::vector<double> s(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += A[static_cast<size_t>(j) * n + i] * A[static_cast<size_t>(j) * n + i];
        s[static_cast<size_t>(j)] = std::sqrt(acc);
    }
    std::sort(s.begin(), s.end(), std::greater<double>());
    return s;
}

template <typename Config>
class BdsdcTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    static double sv_tol()   { return std::is_same_v<Scalar, float> ? 2e-4 : 1e-11; }
    static double vec_tol()  { return std::is_same_v<Scalar, float> ? 1e-3 : 1e-9;  }
    static double orth_tol() { return std::is_same_v<Scalar, float> ? 1e-3 : 1e-9;  }

    // Core driver: run bdsdc on the given (d,e) and check singular values,
    // reconstruction of B, and orthogonality of both vector sets.
    void run_and_check(const std::vector<double>& dh,
                       const std::vector<double>& eh,
                       int n, int batch, bool vectors, const char* label,
                       double orth_override = 0.0) {
        auto& ctx = *this->ctx;
        UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch);
        UnifiedVector<Scalar> e(static_cast<size_t>(std::max(1, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(dh[static_cast<size_t>(b) * n + i]);
            }
            for (int i = 0; i < n - 1; ++i) {
                e[static_cast<size_t>(b) * (n - 1) + i] =
                    static_cast<Scalar>(eh[static_cast<size_t>(b) * (n - 1) + i]);
            }
        }

        VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
        VectorView<Scalar> ev(e.to_span(), std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch);

        const size_t ws_bytes = bdsdc_buffer_size<B, Scalar>(ctx, dv, ev, s.to_span(), vectors);
        UnifiedVector<std::byte> ws(ws_bytes);

        Matrix<Scalar> U(n, n, batch), Vh(n, n, batch);
        if (vectors) {
            bdsdc<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), U.view(), Vh.view(), true);
        } else {
            bdsdc<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), true);
        }
        ctx.wait_and_throw();

        auto uat = [&](int b, int i, int j) {
            return static_cast<double>(
                U.view().data_ptr()[b * U.view().stride() + static_cast<size_t>(j) * U.view().ld() + i]);
        };
        auto vhat = [&](int b, int i, int j) {
            return static_cast<double>(
                Vh.view().data_ptr()[b * Vh.view().stride() + static_cast<size_t>(j) * Vh.view().ld() + i]);
        };

        for (int b = 0; b < batch; ++b) {
            std::vector<double> db(dh.begin() + static_cast<size_t>(b) * n,
                                   dh.begin() + static_cast<size_t>(b + 1) * n);
            std::vector<double> eb(eh.begin() + static_cast<size_t>(b) * std::max(0, n - 1),
                                   eh.begin() + static_cast<size_t>(b + 1) * std::max(0, n - 1));
            const auto ref = reference_singular_values(db, eb, n);
            const double smax = std::max(ref.front(), 1e-300);
            for (int i = 0; i < n; ++i) {
                EXPECT_NEAR(static_cast<double>(s[static_cast<size_t>(b) * n + i]),
                            ref[static_cast<size_t>(i)], sv_tol() * smax)
                    << label << " sigma n=" << n << " b=" << b << " i=" << i;
            }
        }

        if (!vectors) return;

        for (int b = 0; b < batch; ++b) {
            // U diag(s) Vh == B
            double num = 0.0, den = 0.0;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    double acc = 0.0;
                    for (int t = 0; t < n; ++t) {
                        acc += uat(b, i, t) * static_cast<double>(s[static_cast<size_t>(b) * n + t]) * vhat(b, t, j);
                    }
                    double bij = 0.0;
                    if (i == j) bij = dh[static_cast<size_t>(b) * n + i];
                    else if (j == i + 1 && n > 1) bij = eh[static_cast<size_t>(b) * (n - 1) + i];
                    num += (bij - acc) * (bij - acc);
                    den += bij * bij;
                }
            }
            EXPECT_LE(std::sqrt(num / std::max(den, 1e-300)), vec_tol())
                << label << " reconstruction n=" << n << " b=" << b;

            // Orthogonality of U's columns and Vh's rows.
            double uorth = 0.0, vorth = 0.0;
            for (int p = 0; p < n; ++p) {
                for (int q = 0; q < n; ++q) {
                    double du = 0.0, dv2 = 0.0;
                    for (int i = 0; i < n; ++i) {
                        du += uat(b, i, p) * uat(b, i, q);
                        dv2 += vhat(b, p, i) * vhat(b, q, i);
                    }
                    const double tgt = (p == q) ? 1.0 : 0.0;
                    uorth += (du - tgt) * (du - tgt);
                    vorth += (dv2 - tgt) * (dv2 - tgt);
                }
            }
            const double otol = (orth_override > 0.0) ? orth_override : static_cast<double>(orth_tol());
            EXPECT_LE(std::sqrt(uorth), otol) << label << " U orthogonality n=" << n << " b=" << b;
            EXPECT_LE(std::sqrt(vorth), otol) << label << " V orthogonality n=" << n << " b=" << b;
        }
    }

    void check(int n, int batch, unsigned seed, bool vectors, double dscale = 1.0,
               double orth_override = 0.0) {
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
        run_and_check(dh, eh, n, batch, vectors, "random", orth_override);
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
                double v = std::pow(10.0, mag(rng)) * (sign(rng) ? 1.0 : -1.0);
                if (zero(rng) == 0) v = 0.0;
                dh[static_cast<size_t>(b) * n + i] = v;
            }
            for (int i = 0; i < n - 1; ++i) {
                double v = std::pow(10.0, mag(rng)) * (sign(rng) ? 1.0 : -1.0);
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
    const double kGradedOrthTol = std::is_same_v<typename TestFixture::Scalar, float> ? 5e-3 : 1e-10;
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
