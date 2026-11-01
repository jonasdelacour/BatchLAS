// SPDX-License-Identifier: MIT
// Host-only tests of the tridiagonal scalar layer: each helper against a copy of the inline
// code it replaces (bitwise) and against long-double mathematics.
#include <gtest/gtest.h>

#include <src/ops/syev/stedc/loewner.hh>
#include <src/ops/syev/stedc/rank_one.hh>
#include <src/ops/syev/stedc/secular_equation.hh>
#include <src/ops/syev/steqr/ql_chase.hh>
#include <src/ops/syev/tridiag/tridiag_math.hh>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using namespace batchlas;
using namespace batchlas::tridiag;

namespace {

template <typename T>
auto bits(T x) {
    if constexpr (sizeof(T) == 4) {
        return std::bit_cast<uint32_t>(x);
    } else {
        return std::bit_cast<uint64_t>(x);
    }
}
#define EXPECT_BITEQ(a, b) EXPECT_EQ(bits(a), bits(b)) << #a " = " << (a) << ", " #b " = " << (b)

template <typename T>
constexpr T eps() { return std::numeric_limits<T>::epsilon(); }

// Uniform in [-1, 1], or sign * 10^U(-g, g) when graded.
template <typename T>
struct Sampler {
    std::mt19937_64 rng{20261101};
    T uniform() { return T(std::uniform_real_distribution<double>(-1, 1)(rng)); }
    T graded(double g) {
        const double e = std::uniform_real_distribution<double>(-g, g)(rng);
        return T((uniform() < 0 ? -1.0 : 1.0) * std::pow(10.0, e));
    }
    T any(int i) { return (i % 3 == 0) ? uniform() : graded((i % 3 == 1) ? (sizeof(T) == 4 ? 15 : 30) : 3); }
};

// Frozen copy of cta_lartg (steqr_cta_device.hh:33-69 at 9268775f), which annihilate_fast replaced.
template <typename T>
struct CtaRotation {
    T c, s, r;
};
template <typename T>
CtaRotation<T> cta_lartg(const T f, const T g) {
    if constexpr (std::is_same_v<T, float>) {
        const bool gz = (g == T(0));
        const T f_abs = sycl::fabs(f);
        const T g_abs = sycl::fabs(g);
        const T rtmin = sycl::sqrt(detail::safmin<T>());
        const T rtmax = sycl::sqrt(detail::safmax<T>() / T(2));
        T c = T(1), s = T(0), r = f;
        if (f_abs > rtmin && f_abs < rtmax && (gz || (g_abs > rtmin && g_abs < rtmax))) {
            const T t = f * f + g * g;
            T inv = sycl::rsqrt(t);
            inv = sycl::fma(inv * T(0.5), sycl::fma(-(t * inv), inv, T(1)), inv);
            const T d = sycl::sqrt(t);
            const T signed_inv = sycl::copysign(inv, f);
            c = f_abs * inv;
            s = g * signed_inv;
            r = sycl::copysign(d, f);
        } else if (!gz) {
            const auto res = detail::lartg(f, g);
            c = res.c;
            s = res.s;
            r = res.r;
        }
        return {gz ? T(1) : c, gz ? T(0) : s, gz ? f : r};
    } else {
        const auto res = detail::lartg(f, g);
        return {res.c, res.s, res.r};
    }
}

// Copy of steqr_cta_device.hh:395-410 at 9268775f (EXP chase step), returning every value it produces.
template <typename T>
std::array<T, 8> exp_inline(T x, T y, T di, T ei, T dj, T ej) {
    const auto [c1, s1, r1] = cta_lartg(x, y);
    const T sigma = -s1;
    const T e_hi_new = x * c1 - y * sigma;
    const T di_new = c1 * (c1 * di - ei * sigma) - sigma * (ei * c1 - sigma * dj);
    const T dj_new = c1 * (c1 * dj + ei * sigma) + sigma * (ei * c1 + sigma * di);
    const T ei_new = c1 * (c1 * ei + sigma * di) - sigma * (c1 * dj + sigma * ei);
    const T ej_new = c1 * ej;
    const T bulge_new = -ej * sigma;
    return {c1, sigma, di_new, dj_new, ei_new, ej_new, e_hi_new, bulge_new};
}

// Copy of steqr_cta_device.hh:499-520 at 9268775f (PG chase step) on the carried (g, c, s, p).
template <typename T>
std::array<T, 4> pg_inline(T& g, T& c, T& s, T& p, T ei, T di, T dip1) {
    const T f = s * ei;
    const auto [c1, s1, r1] = cta_lartg(g, f);
    const T g2 = dip1 - p;
    const T r2 = (di - g2) * s1 + T(2) * c1 * (c * ei);
    p = s1 * r2;
    const T d_ip1_new = g2 + p;
    g = c1 * r2 - (c * ei);
    c = c1;
    s = s1;
    return {c1, s1, d_ip1_new, r1};
}

// One QL sweep on the whole block [0, n-1] as the CTA solver runs it (lanes become array
// slots); Helper selects the new scalar layer or the inline copies.
template <typename T, bool Helper>
void exp_sweep(std::vector<T>& d, std::vector<T>& e, T mu) {
    const int32_t n = int32_t(d.size()), l = 0, m = n - 1;
    const std::vector<T> d_snap = d, e_snap = e;
    T di = d[m], ei = e[m - 1], e_own = T(0);
    T eprev = di - mu, bulge = ei;
    for (int32_t v = 0; v < m - l; ++v) {
        const int32_t hi = m - v, lo = m - v - 1;
        const T dj = d_snap[lo];
        const T ej = (lo >= 1) ? e_snap[lo - 1] : T(0);
        if constexpr (Helper) {
            const auto r = exp_rotation(ChaseState<T>{eprev, bulge}, di, ei, dj, ej);
            d[hi] = r.d_hi;
            if (hi < m) e[hi] = r.e_hi;
            eprev = r.next().pivot;
            bulge = r.next().bulge;
            di = r.d_lo;
            ei = r.e_below;
            e_own = r.e_mid;
        } else {
            const auto u = exp_inline(eprev, bulge, di, ei, dj, ej);
            d[hi] = u[2];
            if (hi < m) e[hi] = u[6];
            eprev = u[4];
            bulge = u[7];
            di = u[3];
            ei = u[5];
            e_own = u[4];
        }
    }
    d[l] = di;
    e[l] = e_own;
}

template <typename T, bool Helper>
void pg_sweep(std::vector<T>& d, std::vector<T>& e, T mu) {
    const int32_t n = int32_t(d.size()), l = 0, m = n - 1;
    const T p0 = d[l];
    T g = d[m] - mu, c = T(1), s = T(1), p = T(0);
    PgCarry<T> carry{{T(1), T(1)}, d[m] - mu, T(0)};
    for (int32_t i = m; i-- > l;) {
        const bool do_e_upd = (i != (m - 1)) && ((i + 1) < (n - 1));
        if constexpr (Helper) {
            const auto r = pg_rotation(carry, d[i], e[i], d[i + 1]);
            d[i + 1] = r.d_hi;
            if (do_e_upd) e[i + 1] = r.e_hi;
            carry = r.next;
        } else {
            const auto u = pg_inline(g, c, s, p, e[i], d[i], d[i + 1]);
            d[i + 1] = u[2];
            if (do_e_upd) e[i + 1] = u[3];
        }
    }
    d[l] = p0 - (Helper ? carry.p : p);
    e[l] = Helper ? carry.g : g;
}

// T' = Q^T T Q from the explicit QL factorisation T - mu I = Q L, dense, in long double.
void explicit_ql(std::vector<long double>& d, std::vector<long double>& e, long double mu) {
    const int n = int(d.size());
    std::vector<long double> A(n * n, 0.0L), M(n * n, 0.0L);
    for (int i = 0; i < n; ++i) {
        M[i * n + i] = d[i];
        if (i + 1 < n) M[i * n + i + 1] = M[(i + 1) * n + i] = e[i];
    }
    for (int i = 0; i < n * n; ++i) A[i] = M[i];
    for (int i = 0; i < n; ++i) A[i * n + i] -= mu;
    const auto rot_rows = [n](std::vector<long double>& X, int j, int k, long double c, long double s) {
        for (int col = 0; col < n; ++col) {
            const long double xk = X[k * n + col], xj = X[j * n + col];
            X[k * n + col] = c * xk + s * xj;
            X[j * n + col] = -s * xk + c * xj;
        }
    };
    for (int k = n - 1; k >= 1; --k) {
        const long double f = A[k * n + k], g = A[(k - 1) * n + k], r = std::hypot(f, g);
        const long double c = f / r, s = g / r;
        rot_rows(A, k - 1, k, c, s);
        rot_rows(M, k - 1, k, c, s);
        for (int row = 0; row < n; ++row) {
            const long double xk = M[row * n + k], xj = M[row * n + k - 1];
            M[row * n + k] = c * xk + s * xj;
            M[row * n + k - 1] = -s * xk + c * xj;
        }
    }
    for (int i = 0; i < n; ++i) {
        d[i] = M[i * n + i];
        if (i + 1 < n) e[i] = M[(i + 1) * n + i];
    }
}

// ||(T - mu I)^{-1}||_F by Gauss-Jordan in long double: the forward condition of the QL step.
long double shifted_inverse_norm(const std::vector<long double>& d, const std::vector<long double>& e, long double mu) {
    const int n = int(d.size());
    std::vector<long double> A(n * n, 0.0L), X(n * n, 0.0L);
    for (int i = 0; i < n; ++i) {
        A[i * n + i] = d[i] - mu;
        X[i * n + i] = 1.0L;
        if (i + 1 < n) A[i * n + i + 1] = A[(i + 1) * n + i] = e[i];
    }
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i) if (std::fabs(A[i * n + k]) > std::fabs(A[piv * n + k])) piv = i;
        for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[piv * n + j]), std::swap(X[k * n + j], X[piv * n + j]);
        const long double inv = 1.0L / A[k * n + k];
        for (int j = 0; j < n; ++j) A[k * n + j] *= inv, X[k * n + j] *= inv;
        for (int i = 0; i < n; ++i) {
            if (i == k) continue;
            const long double f = A[i * n + k];
            for (int j = 0; j < n; ++j) A[i * n + j] -= f * A[k * n + j], X[i * n + j] -= f * X[k * n + j];
        }
    }
    long double s = 0;
    for (long double x : X) s += x * x;
    return std::sqrt(s);
}

template <typename T>
class TridiagMath : public ::testing::Test {};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TridiagMath, RealTypes);

TYPED_TEST(TridiagMath, ExpRotationIsBitwiseTheInlineStep) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int i = 0; i < 200000; ++i) {
        const T x = rng.any(i), y = (i % 17 == 0) ? T(0) : rng.any(i + 1), di = rng.any(i + 2);
        const T ei = rng.any(i), dj = rng.any(i + 1), ej = rng.any(i + 2);
        const auto r = exp_rotation(ChaseState<T>{x, y}, di, ei, dj, ej);
        const auto u = exp_inline(x, y, di, ei, dj, ej);
        EXPECT_BITEQ(r.R.c, u[0]);
        EXPECT_BITEQ(-r.R.s, u[1]);
        EXPECT_BITEQ(r.d_hi, u[2]);
        EXPECT_BITEQ(r.d_lo, u[3]);
        EXPECT_BITEQ(r.e_mid, u[4]);
        EXPECT_BITEQ(r.e_below, u[5]);
        EXPECT_BITEQ(r.e_hi, u[6]);
        EXPECT_BITEQ(r.next().pivot, u[4]);
        EXPECT_BITEQ(r.next().bulge, u[7]);
        if (::testing::Test::HasFailure()) return;
    }
}

TYPED_TEST(TridiagMath, PgRotationIsBitwiseTheInlineStep) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int i = 0; i < 200000; ++i) {
        const auto prev = (i % 5 == 0) ? Givens<T>{T(1), T(1)} : annihilate(rng.uniform(), rng.uniform()).R;
        T g = rng.any(i), c = prev.c, s = prev.s, p = rng.any(i + 1);
        const PgCarry<T> carry{prev, g, p};
        const T ei = (i % 13 == 0) ? T(0) : rng.any(i + 2), di = rng.any(i), dip1 = rng.any(i + 1);
        const auto r = pg_rotation(carry, di, ei, dip1);
        const auto u = pg_inline(g, c, s, p, ei, di, dip1);
        EXPECT_BITEQ(r.R.c, u[0]);
        EXPECT_BITEQ(r.R.s, u[1]);
        EXPECT_BITEQ(r.d_hi, u[2]);
        EXPECT_BITEQ(r.e_hi, u[3]);
        EXPECT_BITEQ(r.next.prev.c, c);
        EXPECT_BITEQ(r.next.prev.s, s);
        EXPECT_BITEQ(r.next.g, g);
        EXPECT_BITEQ(r.next.p, p);
        if (::testing::Test::HasFailure()) return;
    }
}

TYPED_TEST(TridiagMath, WholeSweepsAreBitwiseTheInlineSweeps) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int n : {3, 4, 5, 7, 8, 13, 24, 32}) {
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<T> d(n), e(n - 1);
            for (int i = 0; i < n; ++i) d[i] = rng.any(trial + i);
            for (int i = 0; i + 1 < n; ++i) e[i] = rng.any(trial + i + 1);
            for (auto strategy : {SteqrShiftStrategy::Lapack, SteqrShiftStrategy::Wilkinson}) {
                const T mu = wilkinson_shift(strategy, d[0], e[0], d[1]);
                auto d1 = d, e1 = e, d2 = d, e2 = e, d3 = d, e3 = e, d4 = d, e4 = e;
                exp_sweep<T, true>(d1, e1, mu);
                exp_sweep<T, false>(d2, e2, mu);
                pg_sweep<T, true>(d3, e3, mu);
                pg_sweep<T, false>(d4, e4, mu);
                for (int i = 0; i < n; ++i) EXPECT_BITEQ(d1[i], d2[i]);
                for (int i = 0; i < n; ++i) EXPECT_BITEQ(d3[i], d4[i]);
                for (int i = 0; i + 1 < n; ++i) EXPECT_BITEQ(e1[i], e2[i]);
                for (int i = 0; i + 1 < n; ++i) EXPECT_BITEQ(e3[i], e4[i]);
                ASSERT_FALSE(::testing::Test::HasFailure()) << "n = " << n << ", trial " << trial;
            }
        }
    }
}

// Both chases are the implicit form of T - mu I = QL, T' = LQ + mu I (the sign of e' is free).
// The step's forward condition is kappa = ||T|| ||(T - mu I)^{-1}||_F: a shift near an eigenvalue
// (the point of the shift) makes Q ill-determined. Measured max err / (eps kappa): 5.8 (FP64).
TYPED_TEST(TridiagMath, SweepsMatchTheExplicitQlStepInLongDouble) {
    using T = TypeParam;
    const long double tol = std::is_same_v<T, double> ? 1e-14L : 5e-6L;  // for kappa <= 10
    Sampler<T> rng;
    int well_conditioned = 0;
    for (int n : {3, 5, 8, 16}) {
        for (int trial = 0; trial < 500; ++trial) {
            std::vector<T> d(n), e(n - 1);
            long double norm = 0;
            for (int i = 0; i < n; ++i) d[i] = (trial % 2) ? rng.graded(3) : rng.uniform();
            for (int i = 0; i + 1 < n; ++i) e[i] = (trial % 2) ? rng.graded(3) : rng.uniform();
            for (T x : d) norm = std::max(norm, std::fabs((long double)x));
            for (T x : e) norm = std::max(norm, std::fabs((long double)x));
            const T mu = wilkinson_shift(SteqrShiftStrategy::Lapack, d[0], e[0], d[1]);
            std::vector<long double> dl(d.begin(), d.end()), el(e.begin(), e.end());
            const long double kappa = norm * shifted_inverse_norm(dl, el, mu);
            explicit_ql(dl, el, mu);
            auto de = d, ee = e, dp = d, ep = e;
            exp_sweep<T, true>(de, ee, mu);
            pg_sweep<T, true>(dp, ep, mu);
            long double err_exp = 0, err_pg = 0;
            for (int i = 0; i < n; ++i) {
                err_exp = std::max(err_exp, std::fabs(de[i] - dl[i]) / norm);
                err_pg = std::max(err_pg, std::fabs(dp[i] - dl[i]) / norm);
            }
            for (int i = 0; i + 1 < n; ++i) {
                err_exp = std::max(err_exp, std::fabs(std::fabs((long double)ee[i]) - std::fabs(el[i])) / norm);
                err_pg = std::max(err_pg, std::fabs(std::fabs((long double)ep[i]) - std::fabs(el[i])) / norm);
            }
            EXPECT_LE(err_exp, 16 * eps<T>() * kappa) << "n = " << n << ", trial " << trial;
            EXPECT_LE(err_pg, 16 * eps<T>() * kappa) << "n = " << n << ", trial " << trial;
            if (kappa <= 10) {
                ++well_conditioned;
                EXPECT_LE(err_exp, tol) << "n = " << n << ", trial " << trial;
                EXPECT_LE(err_pg, tol) << "n = " << n << ", trial " << trial;
            }
            ASSERT_FALSE(::testing::Test::HasFailure());
        }
    }
    EXPECT_GE(well_conditioned, 200);
}

TYPED_TEST(TridiagMath, TransferIsTheTransposedCongruence) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int i = 0; i < 100000; ++i) {
        const Sym2x2<T> A{rng.any(i), rng.any(i + 1), rng.any(i + 2)};
        const auto R = annihilate(rng.uniform(), rng.uniform()).R;
        const auto t = A.transfer(R);
        const auto C = A.congruence(R.transpose());
        const T scale = 16 * eps<T>() * (std::fabs(A.a) + std::fabs(A.b) + std::fabs(A.c));
        EXPECT_NEAR(A.a - t.delta, C.a, scale);
        EXPECT_NEAR(t.b_next, C.b, scale);
        EXPECT_NEAR(A.c + t.delta, C.c, scale);
        if (::testing::Test::HasFailure()) return;
    }
}

TYPED_TEST(TridiagMath, AnnihilateEdgeCases) {
    using T = TypeParam;
    const T big = std::numeric_limits<T>::max() / T(4), tiny = std::numeric_limits<T>::min() * T(4);
    const T sub = std::numeric_limits<T>::denorm_min() * T(8);
    for (T f : {T(3), T(-3), big, -sub}) {
        const auto a = annihilate(f, T(0)), b = annihilate_fast(f, T(0));
        EXPECT_BITEQ(a.R.c, T(1));
        EXPECT_BITEQ(a.R.s, T(0));
        EXPECT_BITEQ(a.r, f);
        EXPECT_BITEQ(b.R.c, T(1));
        EXPECT_BITEQ(b.R.s, T(0));
        EXPECT_BITEQ(b.r, f);
    }
    for (T g : {T(3), T(-3), big, -sub}) {
        for (const auto a : {annihilate(T(0), g), annihilate_fast(T(0), g)}) {
            EXPECT_EQ(a.R.c, T(0));
            EXPECT_EQ(a.R.s, std::copysign(T(1), g));
            EXPECT_EQ(a.r, std::fabs(g));
        }
    }
    // Huge and tiny: R (f, g) = (r, 0) and r = sign(f) hypot(f, g), with no overflow or underflow.
    const std::array<std::array<T, 2>, 6> cases{{{big, big}, {-big, T(1)}, {tiny, -tiny}, {tiny, big}, {T(1), tiny}, {big, -tiny}}};
    for (const auto& [f, g] : cases) {
        for (const auto a : {annihilate(f, g), annihilate_fast(f, g)}) {
            const long double fl = f, gl = g, r = std::copysign(std::hypot(fl, gl), fl);
            EXPECT_NEAR((long double)a.r / r, 1.0L, 4 * eps<T>()) << f << ", " << g;
            EXPECT_NEAR(((long double)a.R.c * fl + (long double)a.R.s * gl) / r, 1.0L, 4 * eps<T>());
            EXPECT_LE(std::fabs(-(long double)a.R.s * fl + (long double)a.R.c * gl) / std::fabs(r), 4 * eps<T>());
        }
    }
    // Out of range annihilate_fast falls back to annihilate exactly; in range both are the frozen cta_lartg's.
    Sampler<T> rng;
    for (int i = 0; i < 100000; ++i) {
        const T f = rng.graded(sizeof(T) == 4 ? 40 : 320), g = rng.graded(sizeof(T) == 4 ? 40 : 320);
        const auto a = annihilate_fast(f, g);
        const auto ref = cta_lartg(f, g);
        EXPECT_BITEQ(a.R.c, ref.c);
        EXPECT_BITEQ(a.R.s, ref.s);
        EXPECT_BITEQ(a.r, ref.r);
        const T rtmin = std::sqrt(detail::safmin<T>()), rtmax = std::sqrt(detail::safmax<T>() / 2);
        const auto in = [&](T x) { return std::fabs(x) > rtmin && std::fabs(x) < rtmax; };
        if (!(in(f) && in(g))) {
            const auto b = annihilate(f, g);
            EXPECT_BITEQ(a.R.c, b.R.c);
            EXPECT_BITEQ(a.R.s, b.R.s);
            EXPECT_BITEQ(a.r, b.r);
        }
        if (::testing::Test::HasFailure()) return;
    }
}

// The split test straddled by one ulp of e, on the eps^2 term and on the safmin floor.
TYPED_TEST(TridiagMath, NegligibleAtThresholdsStraddledByOneUlp) {
    using T = TypeParam;
    const int p = std::numeric_limits<T>::digits;
    const T inf = std::numeric_limits<T>::infinity();
    for (int k : {0, 5, -7}) {
        const T d0 = std::ldexp(T(1), k), d1 = std::ldexp(T(1), -k);
        const T e = std::ldexp(T(1), -p);  // e^2 = eps^2 |d0 d1| exactly
        for (T s : {T(1), T(-1)}) {
            EXPECT_TRUE(negligible(s * e, s * d0, d1));
            EXPECT_FALSE(negligible(s * std::nextafter(e, inf), d0, -s * d1));
        }
    }
    const T floor = std::sqrt(detail::safmin<T>());  // d0 = 0: e^2 <= safmin
    EXPECT_TRUE(negligible(floor, T(0), T(1)));
    EXPECT_FALSE(negligible(std::nextafter(floor, inf), T(0), T(1)));
    EXPECT_FALSE(negligible(T(1), std::numeric_limits<T>::quiet_NaN(), T(1)));
}

TYPED_TEST(TridiagMath, WilkinsonShiftIsTheNearestEigenvalue) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int i = 0; i < 100000; ++i) {
        const T a = rng.uniform(), b = (i % 2) ? rng.uniform() : rng.graded(3), c = rng.uniform();
        const long double al = a, bl = b, cl = c;
        const long double mean = (al + cl) / 2, rad = std::hypot((al - cl) / 2, bl);
        const long double l1 = mean + rad, l2 = mean - rad;
        const long double nearest = std::fabs(l1 - al) < std::fabs(l2 - al) ? l1 : l2;
        const long double tol = 8 * eps<T>() * (std::fabs(al) + std::fabs(bl) + std::fabs(cl));
        EXPECT_NEAR(wilkinson_shift(SteqrShiftStrategy::Lapack, a, b, c), nearest, tol) << a << " " << b << " " << c;
        EXPECT_NEAR(wilkinson_shift(SteqrShiftStrategy::Wilkinson, a, b, c), nearest, tol) << a << " " << b << " " << c;
        if (::testing::Test::HasFailure()) return;
    }
}

TYPED_TEST(TridiagMath, BlockScaleBringsTheNormIntoRange) {
    using T = TypeParam;
    for (T anorm : {T(0), T(1), detail::ssfmin<T>(), detail::ssfmax<T>()}) {
        EXPECT_FALSE(BlockScale<T>::for_norm(anorm).active()) << anorm;
    }
    for (T anorm : {std::numeric_limits<T>::max(), std::numeric_limits<T>::min(), T(1e-30) * detail::ssfmin<T>()}) {
        const auto s = BlockScale<T>::for_norm(anorm);
        EXPECT_TRUE(s.active());
        EXPECT_BITEQ(s.inverse, T(1) / s.factor);
        const T safe = s.to_safe(anorm);
        EXPECT_NEAR(safe / (anorm > 1 ? detail::ssfmax<T>() : detail::ssfmin<T>()), T(1), 2 * eps<T>());
        EXPECT_BITEQ(s.back(T(1)), s.inverse);
    }
}

TYPED_TEST(TridiagMath, EigIsBitwiseLaev2) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int i = 0; i < 100000; ++i) {
        const Sym2x2<T> A{rng.any(i), (i % 11 == 0) ? T(0) : rng.any(i + 1), (i % 7 == 0) ? -rng.any(i) : rng.any(i + 2)};
        const auto [rt1, rt2, cs, sn] = detail::laev2(A.a, A.b, A.c);
        const auto E = A.eig();
        EXPECT_BITEQ(E.rt1, rt1);
        EXPECT_BITEQ(E.rt2, rt2);
        EXPECT_BITEQ(E.R.c, cs);
        EXPECT_BITEQ(E.R.s, sn);
        const auto D = A.congruence(E.R);  // R A R^T = diag(rt1, rt2)
        const T scale = 16 * eps<T>() * (std::fabs(A.a) + std::fabs(A.b) + std::fabs(A.c));
        EXPECT_NEAR(D.a, rt1, scale);
        EXPECT_NEAR(D.b, T(0), scale);
        if (::testing::Test::HasFailure()) return;
    }
}

struct SerialReduce {
    int32_t lane_id() const { return 0; }
    int32_t width() const { return 1; }
    template <typename T>
    T reduce_sum(T x) const { return x; }
};

// A K-pole secular equation with well separated poles and ||z|| = 1.
template <typename T>
struct SecularProblem {
    std::vector<T> d, z;
    T rho;
    long double f(int32_t origin, long double tau) const {
        long double s = 1.0L / rho;
        for (size_t j = 0; j < d.size(); ++j) s += (long double)z[j] * z[j] / (((long double)d[j] - d[origin]) - tau);
        return s;
    }
    explicit SecularProblem(Sampler<T>& rng, int K) : d(K), z(K), rho(T(0.25) + std::fabs(rng.uniform())) {
        T acc = rng.uniform(), zz = 0;
        for (int j = 0; j < K; ++j) {
            acc += T(0.1) + std::fabs(rng.uniform());
            d[j] = acc;
            z[j] = (rng.uniform() < 0 ? -1 : 1) * (T(0.2) + std::fabs(rng.uniform()));
            zz += z[j] * z[j];
        }
        for (T& x : z) x /= std::sqrt(zz);
    }
};

TYPED_TEST(TridiagMath, SecularSplitMatchesLongDoubleSums) {
    using T = TypeParam;
    Sampler<T> rng;
    const SerialReduce red;
    for (int trial = 0; trial < 200; ++trial) {
        SecularProblem<T> P(rng, 12);
        const SecularEquation<T, SerialReduce> eq{red, VectorView<T>(P.d.data(), 12), VectorView<T>(P.z.data(), 12), 12, T(1) / P.rho};
        const int32_t k = trial % 11;
        const T tau = T(0.3) * (P.d[k + 1] - P.d[k]);
        for (auto [ex, lo_end, hi_start] : {std::tuple{Exclude::None, k + 1, k + 1}, std::tuple{Exclude::Pole, k, k + 1},
                                            std::tuple{Exclude::PoleAndNext, k, k + 2}}) {
            long double psi = 0, phi = 0, dpsi = 0, dphi = 0, mag = 1.0L / P.rho;
            for (int j = 0; j < 12; ++j) {
                const long double r = (long double)P.z[j] / (((long double)P.d[j] - P.d[k]) - tau);
                if (j < lo_end) psi += P.z[j] * r, dpsi += r * r;
                if (j >= hi_start) phi += P.z[j] * r, dphi += r * r;
                if (j < lo_end || j >= hi_start) mag += std::fabs(P.z[j] * r);
            }
            const auto s = eq.split(k, ex, P.d[k], tau);
            const long double tol = 32 * eps<T>() * mag;
            EXPECT_NEAR(s.psi, psi, tol);
            EXPECT_NEAR(s.phi, phi, tol);
            EXPECT_NEAR(s.w, 1.0L / P.rho + psi + phi, tol);
            EXPECT_NEAR(s.dpsi, dpsi, 32 * eps<T>() * dpsi);
            EXPECT_NEAR(s.dphi, dphi, 32 * eps<T>() * dphi);
            EXPECT_NEAR(s.dw, dpsi + dphi, 32 * eps<T>() * (dpsi + dphi));
        }
    }
}

// rational_step inside the Bracket converges to the long-double root of every interval, and
// the bracket keeps the root at every step.
TYPED_TEST(TridiagMath, RationalStepAndBracketConvergeToTheLongDoubleRoot) {
    using T = TypeParam;
    Sampler<T> rng;
    const SerialReduce red;
    for (int trial = 0; trial < 200; ++trial) {
        const int K = 10;
        SecularProblem<T> P(rng, K);
        const SecularEquation<T, SerialReduce> eq{red, VectorView<T>(P.d.data(), K), VectorView<T>(P.z.data(), K), K, T(1) / P.rho};
        for (int32_t k = 0; k + 1 < K; ++k) {
            const T gap = P.d[k + 1] - P.d[k];
            const bool origin_lower = P.f(k, gap / 2) > 0;
            const int32_t wi = origin_lower ? k : k + 1;
            long double lo_l = origin_lower ? 0.0L : -(long double)gap, hi_l = origin_lower ? (long double)gap : 0.0L;
            for (int it = 0; it < 200; ++it) {
                const long double mid = (lo_l + hi_l) / 2;
                (P.f(wi, mid) > 0 ? hi_l : lo_l) = mid;
            }
            const long double root = (lo_l + hi_l) / 2;
            for (auto model : {RationalModel::FixedWeight, RationalModel::MiddleWay}) {
                Bracket<T> br = origin_lower ? Bracket<T>{T(0), gap / 2} : Bracket<T>{-gap / 2, T(0)};
                T tau = (br.lo + br.hi) / 2;
                const T origin = P.d[wi];
                for (int it = 0; it < 40; ++it) {
                    auto s = eq.split(wi, Exclude::Pole, origin, tau);
                    const T ratio = P.z[wi] / (P.d[wi] - origin - tau);
                    s.dw += ratio * ratio;
                    s.w += P.z[wi] * ratio;
                    if (s.w == T(0)) break;
                    br = br.tighten(tau, s.w);
                    ASSERT_LE(br.lo, root + 4 * eps<T>() * gap);
                    ASSERT_GE(br.hi, root - 4 * eps<T>() * gap);
                    const T eta = rational_step(model, s, P.d[k] - origin - tau, P.d[k + 1] - origin - tau, P.z[k] * P.z[k],
                                                P.z[k + 1] * P.z[k + 1], P.d[k], P.d[k + 1], origin_lower, ratio * ratio);
                    const T step = br.step_inside(tau, eta, s.w);
                    if (tau + step == tau) break;
                    tau += step;
                }
                EXPECT_NEAR(tau, root, 16 * eps<T>() * std::fabs(root) + 4 * std::numeric_limits<T>::min())
                    << "trial " << trial << " k " << k << " model " << int(model);
            }
        }
        if (::testing::Test::HasFailure()) return;
    }
}

// 300 factors whose running product leaves double range: mant * 2^exp tracks it, and an
// in-range product is bitwise the naive one.
TYPED_TEST(TridiagMath, ScaledProductMatchesLongDouble) {
    using T = TypeParam;
    Sampler<T> rng;
    for (int trial = 0; trial < 200; ++trial) {
        ScaledProduct<T> p;
        long double ref = 1.0L;
        const int n = 300;
        for (int i = 0; i < n; ++i) {
            const T u = T(1) + T(0.5) * rng.uniform();
            const T f = std::ldexp(u, (i < n / 2) ? -60 : 60);  // the running product leaves range
            p = p.times(f);
            ref *= f;
        }
        const long double value = std::ldexp((long double)p.mant, p.exp);
        EXPECT_NEAR(value / ref, 1.0L, n * eps<T>());
        EXPECT_NEAR(p.sqrt_value() / std::sqrt(std::fabs(ref)), 1.0L, n * eps<T>());

        ScaledProduct<T> q;
        T naive = T(1);
        for (int i = 0; i < 20; ++i) {
            const T f = T(1.5) * rng.uniform();
            q = q.times(f);
            naive *= f;
        }
        EXPECT_BITEQ(std::ldexp(q.mant, q.exp), naive);
    }
}

TYPED_TEST(TridiagMath, RankOneTearAndDeflationTests) {
    using T = TypeParam;
    EXPECT_EQ(Tear<T>{T(-0.75)}.rho(), T(1.5));
    EXPECT_EQ(Tear<T>{T(-0.75)}.sign(), T(-1));
    EXPECT_EQ(Tear<T>{T(-0.0)}.sign(), T(1));
    const auto tol = DeflationTolerance<T>::from(T(1), T(0.5));
    EXPECT_BITEQ(tol.tol, T(8) * eps<T>());
    const T inf = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(coincident_poles(T(1), T(1) + tol.tol, tol));
    EXPECT_FALSE(coincident_poles(T(1), std::nextafter(T(1) + tol.tol, inf), tol));
    EXPECT_TRUE(negligible_weight(T(-1), tol.tol, tol));
    EXPECT_FALSE(negligible_weight(T(-1), std::nextafter(tol.tol, inf), tol));
    EXPECT_TRUE(negligible_weight(T(1), std::numeric_limits<T>::quiet_NaN(), tol));
}

} // namespace
