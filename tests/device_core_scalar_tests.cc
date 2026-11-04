// SPDX-License-Identifier: MIT
// device_core_scalar_tests: the v2.2 core.hh and scalar.hh additions (design A.2) against host
// references: static_for / static_extent_v / MaybeZero / None, op::FMax / FMin with NaN,
// lamch, qnan, phase, sign_nonzero_real, eig2x2_nearest and SafeRecip.
// Guards (agent-guide section 8): SafeRecip straddles both of its thresholds (d where 1/d
// overflows: the divide arm; d = 0: the caller's value; ordinary d: the multiply arm), checked
// by the arm's own rounding, not only by closeness; phase at 0 and at -0 and on complex data with
// nonzero imaginary parts; FMax / FMin with a NaN on either side; eig2x2_nearest on graded inputs
// (b tiny against a, c) and on the tie a == c; device and host results compared, never device to
// itself.
// Red set (each break was made once, 2026-11-04, and this suite failed): SafeRecip with use_mul
// for an infinite reciprocal.
#include <tests/device_core_common.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreScalarTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreScalarTest, AllTypes);

// ---- compile-time vocabulary ---------------------------------------------------------------
static_assert(dv::static_extent_v<dv::Ext<7>> == 7 && dv::static_extent_v<dv::Dyn> == -1);
static_assert(dv::is_scalar_tag_v<dv::MaybeZero<float>>);
static_assert(std::is_same_v<dv::tag_t<dv::MaybeZero<double>>, dv::MaybeZero<double>>);
static_assert(dv::is_none_v<dv::None> && !dv::is_none_v<int>);
static_assert(dv::lamch<'E', double>() == std::numeric_limits<double>::epsilon() / 2);
static_assert(dv::lamch<'P', float>() == std::numeric_limits<float>::epsilon());
static_assert(dv::lamch<'S', double>() == dv::safmin<double>());
static_assert(dv::lamch<'O', float>() == std::numeric_limits<float>::max());

TEST(DeviceCoreScalarHost, StaticForVisitsEveryIndexInOrder) {
    int seen[6] = {};
    int order = 0;
    dv::static_for<6>([&](auto I) {
        static_assert(decltype(I)::value >= 0 && decltype(I)::value < 6);
        seen[I] = ++order;
    });
    for (int i = 0; i < 6; ++i) EXPECT_EQ(seen[i], i + 1);
    dv::NoOp{}(1, 2.0, "x");  // compiles to nothing for any arguments
}

TYPED_TEST(DeviceCoreScalarTest, PhaseSignQnanAndNanLosingMinMaxOnDevice) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    std::vector<T> x_h = vals<T>(64, 61);
    x_h[0] = T(0);
    x_h[1] = T(R(-0.0));
    x_h[2] = dv::lift<T>(R(-3));
    T* x = d.up(x_h);
    T* ph = d.template alloc<T>(64, T(0));
    R* sg = d.template alloc<R>(64, R(0));
    R* mm = d.template alloc<R>(8, R(0));
    T* qn = d.template alloc<T>(1, T(0));
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        for (int i = 0; i < 64; ++i) {
            ph[i] = dv::phase(x[i]);
            sg[i] = dv::sign_nonzero_real(x[i]);
        }
        const R one = R(1);
        const R n = dv::qnan<R>();
        mm[0] = dv::op::FMax{}(one, n);
        mm[1] = dv::op::FMax{}(n, one);
        mm[2] = dv::op::FMin{}(one, n);
        mm[3] = dv::op::FMin{}(n, one);
        mm[4] = dv::op::FMax{}(R(-2), R(5));
        mm[5] = dv::op::FMin{}(R(-2), R(5));
        qn[0] = dv::qnan<T>();
    });
    const auto P = d.down(ph, 64);
    const auto S = d.down(sg, 64);
    const auto M = d.down(mm, 8);
    const auto Q = d.down(qn, 1);
    EXPECT_TRUE(same_bits(P[0], T(1)));
    EXPECT_TRUE(same_bits(P[1], T(1)));
    EXPECT_EQ(std::real(P[2]), R(-1));
    EXPECT_EQ(S[0], R(1));
    EXPECT_EQ(S[1], R(1)) << "-0 has sign +1";
    EXPECT_EQ(S[2], R(-1));
    for (int i = 3; i < 64; ++i) {
        // |phase| = 1 and phase * |x| = x.
        EXPECT_LE(std::abs(std::abs(wide(P[i])) - 1.0L), tol<T>()) << i;
        EXPECT_LE(err(x_h[i], wide(P[i]) * std::abs(wide(x_h[i]))), tol<T>()) << i;
        EXPECT_EQ(S[i], std::real(x_h[i]) < 0 ? R(-1) : R(1)) << i;
    }
    EXPECT_EQ(M[0], R(1));
    EXPECT_EQ(M[1], R(1));
    EXPECT_EQ(M[2], R(1));
    EXPECT_EQ(M[3], R(1));
    EXPECT_EQ(M[4], R(5));
    EXPECT_EQ(M[5], R(-2));
    EXPECT_TRUE(std::isnan(std::real(Q[0])));
    if constexpr (is_cplx<T>) EXPECT_TRUE(std::isnan(std::imag(Q[0])));
}

TYPED_TEST(DeviceCoreScalarTest, Eig2x2NearestPicksTheRootNearC) {
    using T = TypeParam;
    using R = real_of<T>;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "a real symmetric 2x2";
    } else {
        auto& d = this->dev;
        std::vector<R> abc;
        for (int i = 0; i < 40; ++i) abc.insert(abc.end(), {uval<R>(3 * i) * 4, uval<R>(3 * i + 1), uval<R>(3 * i + 2) * 4});
        abc.insert(abc.end(), {R(1), R(1e-6), R(3)});  // graded: b tiny
        abc.insert(abc.end(), {R(2), R(0.5), R(2)});   // a == c: the roots straddle c equally
        const int n = static_cast<int>(abc.size() / 3);
        R* p = d.up(abc);
        R* out = d.template alloc<R>(n, R(0));
        d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
            for (int i = 0; i < n; ++i) out[i] = dv::eig2x2_nearest(p[3 * i], p[3 * i + 1], p[3 * i + 2]);
        });
        const auto got = d.down(out, n);
        for (int i = 0; i < n; ++i) {
            const long double a = abc[3 * i], b = abc[3 * i + 1], c = abc[3 * i + 2];
            const long double m = (a + c) / 2, r = std::sqrt((a - c) * (a - c) / 4 + b * b);
            const long double e1 = m + r, e2 = m - r;
            const long double near = std::abs(e1 - c) < std::abs(e2 - c) ? e1 : e2;
            if (std::abs(std::abs(e1 - c) - std::abs(e2 - c)) < 1e-3L) continue;  // the tie: either root
            EXPECT_LE(std::abs(got[i] - near), tol<R>(4) * std::max<long double>(1, std::abs(near))) << i;
        }
    }
}

// SafeRecip's three arms, each identified by its rounding: multiply by the Fma reciprocal, the
// division, and the caller's value at an exact zero.
TYPED_TEST(DeviceCoreScalarTest, SafeRecipTakesEachArmAtItsThreshold) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    const R tiny = std::numeric_limits<R>::denorm_min() * R(4);  // 1 / tiny overflows
    const std::vector<T> dd{T(R(3)), T(tiny), T(0), dv::lift<T>(dv::safmin<R>())};
    const std::vector<T> yy{val<T>(71), T(tiny), val<T>(73), val<T>(74)};
    T* dp = d.up(dd);
    T* yp = d.up(yy);
    T* out = d.template alloc<T>(4, T(0));
    int* flags = d.template alloc<int>(8, -1);
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        for (int i = 0; i < 4; ++i) {
            const auto s = dv::SafeRecip<T>::make(dp[i]);
            out[i] = s.apply(yp[i], dv::lift<T>(R(-7)));
            flags[2 * i] = s.use_mul;
            flags[2 * i + 1] = s.zero;
        }
    });
    const auto got = d.down(out, 4);
    const auto F = d.down(flags, 8);
    EXPECT_EQ(F[0], 1);
    EXPECT_EQ(F[1], 0);
    EXPECT_LE(err(got[0], wide(yy[0]) / LD<T>(3)), tol<T>());
    EXPECT_EQ(F[2], 0) << "1 / tiny overflows: the divide arm";
    EXPECT_EQ(F[3], 0);
    EXPECT_LE(err(got[1], LD<T>(1)), tol<T>()) << "tiny / tiny = 1 only through the division";
    EXPECT_EQ(F[4], 0);
    EXPECT_EQ(F[5], 1);
    EXPECT_TRUE(same_bits(got[2], dv::lift<T>(R(-7)))) << "exact zero: the caller's value";
    EXPECT_EQ(F[6], 1) << "1 / safmin is finite: multiply";
    EXPECT_LE(err(got[3], wide(yy[3]) / wide(dd[3])), tol<T>() * static_cast<double>(1 / dv::safmin<R>()));
}

}  // namespace
