// SPDX-License-Identifier: MIT
// device_core_arith_tests: the v2.2 arith.hh additions (design A.2) against scalar host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts; NaN-poisoned y for
// every "never reads y" claim (Zero beta, a set MaybeZero flag) and the one shape that reads it
// anyway (LinearNegZero); signed-zero cases that tell MacChain (0 + acc) from Sum (acc); the
// FmaSplit rounding on inputs where one fma and two roundings differ (gemv and native spmm call
// store<MacChain, Fma> directly); batch 1024 items, every item compared.
// Red set (each break was made once, 2026-11-04, and this suite failed): MacChain started from acc
// instead of 0. During bring-up the MaybeZero case also caught a real defect (tag_mac read the
// flag through a const type and ignored it).
#include <tests/device_core_common.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreArithTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreArithTest, AllTypes);

constexpr int kBatch = 1024;

template <class T> T nan_of() { return dv::qnan<T>(); }

// One y per item and case; out[c * kBatch + b] = y after case c.
enum Case : int { kSumZero, kSumMaybeZeroSet, kSumMaybeZeroClear, kSumOne, kSumValue, kMacChain, kMacChainMaybeZero, kMulMac,
                  kLinear, kLinearBetaZero, kLinearNegZeroBetaZero, kNumCases };

TYPED_TEST(DeviceCoreArithTest, StoreShapesMatchTheirSourceExpressions) {
    using T = TypeParam;
    auto& d = this->dev;
    const auto acc_h = vals<T>(kBatch, 1), y_h = vals<T>(kBatch, 2), al_h = vals<T>(kBatch, 3), be_h = vals<T>(kBatch, 4);
    // Every 7th y is NaN: a shape that must not read y keeps the item finite.
    std::vector<T> y_poison = y_h;
    for (int b = 0; b < kBatch; b += 7) y_poison[b] = nan_of<T>();
    T* acc = d.up(acc_h);
    T* y0 = d.up(y_poison);
    T* al = d.up(al_h);
    T* be = d.up(be_h);
    T* out = d.template alloc<T>(std::size_t(kNumCases) * kBatch, T(0));
    d.template run<1>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const T a = acc[b], yv = y0[b], alpha = al[b], beta = be[b];
        auto put = [&](int c, auto&& f) {
            T y = yv;
            f(y);
            out[c * kBatch + b] = y;
        };
        put(kSumZero, [&](T& y) { dv::store<dv::epi::Sum>(alpha, a, dv::Zero{}, y); });
        put(kSumMaybeZeroSet, [&](T& y) { dv::store<dv::epi::Sum>(alpha, a, dv::MaybeZero<T>{beta, true}, y); });
        put(kSumMaybeZeroClear, [&](T& y) { dv::store<dv::epi::Sum>(alpha, a, dv::MaybeZero<T>{beta, false}, y); });
        put(kSumOne, [&](T& y) { dv::store<dv::epi::Sum>(dv::One{}, a, dv::One{}, y); });
        put(kSumValue, [&](T& y) { dv::store<dv::epi::Sum>(alpha, a, beta, y); });
        put(kMacChain, [&](T& y) { dv::store<dv::epi::MacChain, dv::arith::Fma>(alpha, a, beta, y); });
        put(kMacChainMaybeZero, [&](T& y) { dv::store<dv::epi::MacChain, dv::arith::Fma>(alpha, a, dv::MaybeZero<T>{beta, true}, y); });
        put(kMulMac, [&](T& y) { dv::store<dv::epi::MulMac, dv::arith::Fma>(alpha, a, dv::MaybeZero<T>{beta, false}, y); });
        put(kLinear, [&](T& y) { dv::store<dv::epi::Linear>(alpha, a, beta, y); });
        put(kLinearBetaZero, [&](T& y) { dv::store<dv::epi::Linear>(alpha, a, T(0), y); });
        put(kLinearNegZeroBetaZero, [&](T& y) { dv::store<dv::epi::LinearNegZero>(alpha, a, T(0), y); });
    });
    const auto got = d.down(out, std::size_t(kNumCases) * kBatch);
    for (int b = 0; b < kBatch; ++b) {
        const bool poisoned = b % 7 == 0;
        const LD<T> A = wide(acc_h[b]), Y = wide(y_h[b]), AL = wide(al_h[b]), BE = wide(be_h[b]);
        auto at = [&](int c) { return got[c * kBatch + b]; };
        // Zero / set flag: y is never read, poisoned or not.
        EXPECT_LE(err(at(kSumZero), AL * A), tol<T>()) << b;
        EXPECT_LE(err(at(kSumMaybeZeroSet), AL * A), tol<T>()) << b;
        EXPECT_LE(err(at(kMacChainMaybeZero), AL * A), tol<T>()) << b;
        EXPECT_LE(err(at(kLinearBetaZero), AL * A), tol<T>()) << b;
        if (poisoned) {
            // The shapes that read y carry the NaN; LinearNegZero reads y even at beta == 0.
            for (int c : {kSumMaybeZeroClear, kSumOne, kSumValue, kMacChain, kMulMac, kLinear, kLinearNegZeroBetaZero})
                EXPECT_TRUE(std::isnan(std::real(wide(at(c))))) << "case " << c << " item " << b;
            continue;
        }
        const LD<T> ref = AL * A + BE * Y;
        for (int c : {kSumMaybeZeroClear, kSumValue, kMacChain, kMulMac, kLinear}) EXPECT_LE(err(at(c), ref), tol<T>(2)) << c << " " << b;
        EXPECT_LE(err(at(kSumOne), A + Y), tol<T>()) << b;
        EXPECT_LE(err(at(kLinearNegZeroBetaZero), AL * A), tol<T>()) << b;
    }
}

// MacChain starts from 0 (0 + acc), Sum and MulMac from acc: the sign of a zero result tells them
// apart, which is why both shapes exist (a port must name its source's).
TYPED_TEST(DeviceCoreArithTest, MacChainAndSumDifferOnlyInTheSignOfZero) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    T* out = d.template alloc<T>(4, T(1));
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        const T negzero = dv::lift<T>(R(-0.0));
        T y0 = T(0), y1 = T(0), y2 = T(0), y3 = T(0);
        dv::store<dv::epi::Sum>(dv::One{}, negzero, dv::Zero{}, y0);
        dv::store<dv::epi::MacChain>(dv::One{}, negzero, dv::Zero{}, y1);
        dv::store<dv::epi::MulMac>(dv::One{}, negzero, dv::Zero{}, y2);
        dv::store<dv::epi::LinearNegZero>(T(1), dv::lift<T>(R(0)), T(0), y3);  // 1 * 0 + (-0) * 0 = +0
        out[0] = y0;
        out[1] = y1;
        out[2] = y2;
        out[3] = y3;
    });
    const auto got = d.down(out, 4);
    EXPECT_TRUE(std::signbit(std::real(got[0]))) << "Sum: One * (-0) is -0";
    EXPECT_FALSE(std::signbit(std::real(got[1]))) << "MacChain: 0 + (-0) is +0";
    EXPECT_TRUE(std::signbit(std::real(got[2]))) << "MulMac: 1 * (-0) is -0";
    EXPECT_FALSE(std::signbit(std::real(got[3])));
}

// The scalar helpers, element by element over complex data with nonzero imaginary parts.
TYPED_TEST(DeviceCoreArithTest, ScalarHelpersMatchHost) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 257;
    const auto x_h = vals<T>(n, 5), z_h = vals<T>(n, 6);
    const auto s_h = vals<R>(n, 7);
    T* x = d.up(x_h);
    T* z = d.up(z_h);
    R* s = d.up(s_h);
    constexpr int kOut = 8;
    T* out = d.template alloc<T>(std::size_t(n) * kOut, T(0));
    R* rout = d.template alloc<R>(std::size_t(n) * 3, R(0));
    d.template run<1>(n, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int i = static_cast<int>(it.get_group_linear_id());
        const T a = x[i], c = z[i];
        const R r = s[i];
        T* o = out + i * kOut;
        o[0] = dv::force_real(a);
        o[1] = dv::conj_if<true>(a);
        o[2] = dv::conj_if<false>(a);
        o[3] = dv::conj_if(i % 2 == 0, a);
        o[4] = dv::scale_real(r, a);
        o[5] = dv::div_real(a, r);
        o[6] = dv::sub_scale_real(c, r, a);
        T acc = c;
        dv::tag_mac(acc, dv::MaybeZero<T>{a, i % 3 == 0}, a);
        o[7] = acc;
        rout[i * 3 + 0] = dv::abs_inf(a);
        const auto p = dv::by_parts(a);
        rout[i * 3 + 1] = p.re;
        rout[i * 3 + 2] = p.im;
    });
    const auto got = d.down(out, std::size_t(n) * kOut);
    const auto rg = d.down(rout, std::size_t(n) * 3);
    for (int i = 0; i < n; ++i) {
        const T a = x_h[i], c = z_h[i];
        const R r = s_h[i];
        const T* o = &got[std::size_t(i) * kOut];
        EXPECT_EQ(std::real(o[0]), std::real(a));
        EXPECT_EQ(std::imag(o[0]), R(0));
        T ca = a;
        if constexpr (is_cplx<T>) ca = std::conj(a);
        EXPECT_TRUE(same_bits(o[1], ca)) << i;
        EXPECT_TRUE(same_bits(o[2], a)) << i;
        EXPECT_TRUE(same_bits(o[3], i % 2 == 0 ? ca : a)) << i;
        EXPECT_LE(err(o[4], LD<T>(r) * wide(a)), tol<T>()) << i;
        EXPECT_LE(err(o[5], wide(a) / LD<T>(r)), tol<T>(1.0 / std::abs(double(r)))) << i;
        EXPECT_LE(err(o[6], wide(c) - LD<T>(r) * wide(a)), tol<T>(2)) << i;
        EXPECT_LE(err(o[7], i % 3 == 0 ? wide(c) : wide(c) + wide(a) * wide(a)), tol<T>(2)) << i;
        EXPECT_EQ(rg[i * 3 + 0], std::max(std::abs(std::real(a)), std::abs(std::imag(a))));
        EXPECT_EQ(rg[i * 3 + 1], std::real(a));
        EXPECT_EQ(rg[i * 3 + 2], std::imag(a));
    }
}

// arith::FmaSplit's msub rounds twice (sub(c, mul(a, b))), arith::Fma's once: on inputs where
// the two differ, each matches its own host reference bit for bit (real types).
TYPED_TEST(DeviceCoreArithTest, FmaSplitRoundsTwiceFmaOnce) {
    using T = TypeParam;
    if constexpr (is_cplx<T>) {
        GTEST_SKIP() << "the rounding claim is checked on the real types";
    } else {
        auto& d = this->dev;
        // a b = 1 + 2e + e^2 exactly; c = round(a b) = 1 + 2e: two roundings give 0, one gives -e^2.
        const T e = std::numeric_limits<T>::epsilon();
        const std::vector<T> in{T(1) + e, T(1) + e, T(1) + T(2) * e};
        T* p = d.up(in);
        T* out = d.template alloc<T>(2, T(0));
        d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
            T c1 = p[2], c2 = p[2];
            dv::ops_arith_t<dv::arith::FmaSplit, T>::msub(c1, p[0], p[1]);
            dv::ops_arith_t<dv::arith::Fma, T>::msub(c2, p[0], p[1]);
            out[0] = c1;
            out[1] = c2;
        });
        const auto got = d.down(out, 2);
        volatile T prod = in[0] * in[1];  // one rounding
        const T split = in[2] - prod;     // the second
        const T fused = std::fma(-in[0], in[1], in[2]);
        ASSERT_NE(split, fused) << "inputs do not separate the two forms";
        EXPECT_EQ(got[0], split) << "FmaSplit: c - round(a b)";
        EXPECT_EQ(got[1], fused) << "Fma: one rounding";
    }
}

}  // namespace
