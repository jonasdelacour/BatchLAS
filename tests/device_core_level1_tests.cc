// SPDX-License-Identifier: MIT
// device_core_level1_tests: level1.hh v2.2 additions (design A.2) against host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts; non-unit strides
// (inc 3) on memory vectors; sizes that are not a multiple of the executor (n = 37 on 32 items),
// odd and even reverse lengths; rscl's both arms (a finite reciprocal, and d below 1 / max where
// the reciprocal overflows); a product of 3000 factors that overflows and underflows a plain
// product; argmax ties (Lowest / Highest), a NaN (First / Skip), the skip index holding the
// maximum, a range that excludes it, on a WG = 1024 work-group through the Scratch exchange;
// lockstep team loops whose trip count must equal on every team. Red set (each break made once,
// 2026-11-04, and this suite failed): for_each_team_owned's lockstep trip count nout / nteams
// (no ceiling: the last outputs lost).
#include <tests/device_core_common.hh>
#include <batchlas/device/level1.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> T conj_t(const T& v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> class DeviceCoreLevel1Test : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreLevel1Test, AllTypes);

TYPED_TEST(DeviceCoreLevel1Test, FillMapManyInterleaveReverse) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int n = 37, inc = 3;
    const auto a_h = vals<T>(n, 1), b_h = vals<T>(n - 1, 2), r_h = vals<T>(n * inc, 3), q_h = vals<T>(n - 1, 4);
    T* a = d.up(a_h);
    T* b = d.up(b_h);
    T* r = d.up(r_h);  // reversed in place, stride inc (odd length)
    T* q = d.up(q_h);  // reversed in place on Thread (even length)
    T* y = d.template alloc<T>(2 * n, T(0));
    T* f = d.template alloc<T>(n * inc, T(9));
    T* m1 = d.template alloc<T>(n, T(0));
    T* m2 = d.template alloc<T>(n, T(0));
    const T alpha = val<T>(99);
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::SubGroup<32> ex{it.get_sub_group()};
        dv::interleave(ex, dv::vec(a, n), dv::vec(b, n - 1), dv::vec(y, 2 * n - 1));
        dv::fill(ex, alpha, dv::vec(f, n, inc));
        dv::map_many(ex, n, [&](int i) { m1[i] = a[i]; }, [&](int i) { m2[i] = dv::Arith<T>::add(a[i], m1[i]); });
        dv::reverse(ex, dv::vec(r, n, inc));
        if (ex.leader()) dv::reverse(dv::Thread{}, dv::vec(q, n - 1));
    });
    const auto yh = d.down(y, 2 * n), fh = d.down(f, n * inc), rh = d.down(r, n * inc), qh = d.down(q, n - 1);
    const auto m1h = d.down(m1, n), m2h = d.down(m2, n);
    for (int k = 0; k < n; ++k) EXPECT_TRUE(same_bits(yh[2 * k], a_h[k])) << k;
    for (int k = 0; k < n - 1; ++k) EXPECT_TRUE(same_bits(yh[2 * k + 1], b_h[k])) << k;
    EXPECT_TRUE(same_bits(yh[2 * n - 1], T(0)));  // not written
    for (int i = 0; i < n * inc; ++i) EXPECT_TRUE(same_bits(fh[i], i % inc == 0 ? alpha : T(9))) << i;
    for (int i = 0; i < n; ++i) {
        EXPECT_TRUE(same_bits(m1h[i], a_h[i]));
        EXPECT_TRUE(same_bits(m2h[i], a_h[i] + a_h[i]));  // the second map sees the first's write
    }
    for (int i = 0; i < n * inc; ++i)
        EXPECT_TRUE(same_bits(rh[i], i % inc == 0 ? r_h[(n - 1 - i / inc) * inc] : r_h[i])) << "reverse " << i;
    for (int i = 0; i < n - 1; ++i) EXPECT_TRUE(same_bits(qh[i], q_h[n - 2 - i])) << "reverse Thread " << i;
}

TYPED_TEST(DeviceCoreLevel1Test, RsclTakesTheReciprocalOnlyWhenItIsFinite) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 37;
    const auto x_h = vals<T>(n, 5);
    const T d1 = val<T>(17) + T(R(2));
    const T d2 = T(std::numeric_limits<R>::min() * R(0.25));  // 1 / d2 overflows: divide arm
    T* x1 = d.up(x_h);
    T* x2 = d.up(x_h);
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::SubGroup<32> ex{it.get_sub_group()};
        dv::rscl(ex, d1, dv::vec(x1, n));
        dv::rscl(ex, d2, dv::vec(x2, n));
    });
    const auto g1 = d.down(x1, n), g2 = d.down(x2, n);
    for (int i = 0; i < n; ++i) {
        const LD<T> ref1 = wide(x_h[i]) / wide(d1), ref2 = wide(x_h[i]) / wide(d2);
        EXPECT_LE(err(g1[i], ref1), tol<T>() * static_cast<double>(std::abs(ref1))) << i;
        if constexpr (!is_cplx<T>) {  // complex: the divide arm is Arith div, whose range is |d|^2's
            EXPECT_TRUE(std::isfinite(static_cast<double>(std::abs(g2[i])))) << "divide arm " << i;
            EXPECT_LE(static_cast<double>(std::abs(wide(g2[i]) - ref2) / std::abs(ref2)), tol<T>()) << i;
        }
    }
}

TYPED_TEST(DeviceCoreLevel1Test, DotPartialDotSlabUpperBound) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int n = 45;
    const auto x_h = vals<T>(n, 6), y_h = vals<T>(n, 7);
    T* x = d.up(x_h);
    T* y = d.up(y_h);
    T* out = d.template alloc<T>(3, T(0));
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::SubGroup<32> ex{it.get_sub_group()};
        const T p = dv::dot_partial<true>(ex, dv::vec(x, n), dv::vec(y, n));
        const T s = ex.team_reduce(p);
        if (ex.leader()) {
            out[0] = s;
            T acc = dv::Arith<T>::zero();
            dv::dot_slab<16>(acc, dv::vec(x + 8, n - 8), dv::vec(y + 8, n - 8), 8, 20);  // t < 12
            out[1] = acc;
        }
    });
    const auto o = d.down(out, 3);
    LD<T> ref = 0, slab = 0;
    for (int i = 0; i < n; ++i) ref += wide(conj_t(x_h[i])) * wide(y_h[i]);
    for (int t = 0; t < 12; ++t) slab += wide(x_h[8 + t]) * wide(y_h[8 + t]);
    EXPECT_LE(err(o[0], ref), tol<T>(4));
    EXPECT_LE(err(o[1], slab), tol<T>(4));
    using R = real_of<T>;
    const std::vector<R> asc{R(-2), R(-1), R(0.5), R(0.5), R(3), R(7)};
    const auto av = dv::vec(const_cast<R*>(asc.data()), int(asc.size()));
    EXPECT_EQ(dv::upper_bound(dv::Thread{}, av, R(0.5)), 4);
    EXPECT_EQ(dv::upper_bound(dv::Thread{}, av, R(-3)), 0);
    EXPECT_EQ(dv::upper_bound(dv::Thread{}, av, R(9)), 6);
}

TYPED_TEST(DeviceCoreLevel1Test, ProdScaledSurvivesWhatAPlainProductCannot) {
    using T = TypeParam;
    using R = real_of<T>;
    if constexpr (is_cplx<T>) GTEST_SKIP() << "prod_scaled is real";
    else {
        auto& d = this->dev;
        constexpr int n = 3000, WG = 256;
        std::vector<R> f_h(n);
        for (int i = 0; i < n; ++i) f_h[i] = (i < n / 2 ? R(37.5) : R(1) / R(41)) * (R(1) + uval<R>(i) * R(0.25));
        R* f = d.up(f_h);
        R* out = d.template alloc<R>(2, R(0));
        int* ex_out = d.template alloc<int>(1, 0);
        d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
            const dv::WorkGroup<WG, 32> ex{it};
            const auto q = dv::prod_scaled(ex, n, [&](int i) { return f[i]; });
            if (ex.leader()) {
                out[0] = q.mant;
                ex_out[0] = q.expo;
                out[1] = dv::sqrt_scaled(q);
            }
        });
        const auto o = d.down(out, 2);
        const int e = d.down(ex_out, 1)[0];
        long double lg = 0;
        for (R v : f_h) lg += std::log2(static_cast<long double>(v));
        const long double got = std::log2(static_cast<long double>(o[0])) + e;
        EXPECT_GE(o[0], R(0.5));
        EXPECT_LT(o[0], R(1));
        EXPECT_NEAR(static_cast<double>(got), static_cast<double>(lg), 1e-3 * std::abs(static_cast<double>(lg)) + 1e-2);
        const long double sq = std::log2(static_cast<long double>(o[1]));
        EXPECT_NEAR(static_cast<double>(sq), static_cast<double>(lg / 2), 1e-3 * std::abs(static_cast<double>(lg)) + 1e-2);
    }
}

TYPED_TEST(DeviceCoreLevel1Test, ArgmaxThroughScratchHonoursRangeSkipTieAndNaN) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int WG = 1024, n = 3000;
    std::vector<T> x_h = vals<T>(n, 8);
    x_h[2001] = T(R(50));   // the maximum, skipped
    x_h[500] = T(R(-40));   // second by |.|: tie with 2500 (Lowest: 500, Highest: 2500)
    x_h[2500] = T(R(40));
    x_h[2999] = T(R(60));   // outside the range
    std::vector<T> nan_h = x_h;
    nan_h[1500] = T(std::numeric_limits<R>::quiet_NaN());
    T* x = d.up(x_h);
    T* xn = d.up(nan_h);
    int* idx = d.template alloc<int>(4, -7);
    using AM = dv::ArgMax<R>;
    d.template run<WG>(1, 32 * sizeof(AM), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<WG, 32> ex{it};
        const dv::Scratch<AM, 32> scr{reinterpret_cast<AM*>(lm)};
        const dv::Range r{1, 2999};
        const auto a = dv::argmax<dv::tie::Lowest>(ex, scr, dv::vec(x, n), r, 2001);
        const auto b = dv::argmax<dv::tie::Highest>(ex, scr, dv::vec(x, n), r, 2001);
        const auto c = dv::argmax<dv::tie::Lowest, dv::nan::First>(ex, scr, dv::vec(xn, n), r, 2001);
        const auto e = dv::argmax<dv::tie::Lowest, dv::nan::Skip>(ex, scr, dv::vec(xn, n), r, 2001);
        if (ex.leader()) {
            idx[0] = a.index;
            idx[1] = b.index;
            idx[2] = c.index;
            idx[3] = e.index;
        }
    });
    const auto got = d.down(idx, 4);
    EXPECT_EQ(got[0], 500);
    EXPECT_EQ(got[1], 2500);
    EXPECT_EQ(got[2], 1500);
    EXPECT_EQ(got[3], 500);
}

TYPED_TEST(DeviceCoreLevel1Test, TeamLoopsAndTilesCoverTheirOutputsOnce) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int WG = 64, nout = 21, m = 13, nn = 7;
    int* hits = d.template alloc<int>(nout, 0);
    int* trips = d.template alloc<int>(WG, 0);
    int* tiles = d.template alloc<int>(m * nn, 0);
    d.template run<WG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const dv::Teams<WG, 8> ex{it};  // 8 teams
        int t = 0;
        dv::for_each_team_owned(ex, nout, [&](int o, bool live) {
            ++t;
            if (live && ex.lane() == 0) sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(hits[o]) += 1;
        });
        trips[it.get_local_linear_id()] = t;
        const dv::WorkGroup<WG, 32> wg{it};
        dv::for_each_owned_tiles<4, 3>(wg, m, nn, [&](int i0, int j0, int lr, int lc) {
            for (int j = 0; j < lc; ++j)
                for (int i = 0; i < lr; ++i)
                    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device>(tiles[(j0 + j) * m + i0 + i]) += 1;
        });
    });
    const auto h = d.down(hits, nout), tr = d.down(trips, WG), tl = d.down(tiles, m * nn);
    for (int o = 0; o < nout; ++o) EXPECT_EQ(h[o], 1) << o;
    for (int g = 0; g < WG; ++g) EXPECT_EQ(tr[g], 3) << "lockstep trips, item " << g;  // ceil(21 / 8)
    for (int e = 0; e < m * nn; ++e) EXPECT_EQ(tl[e], 1) << e;
    (void)sizeof(T);
}

}  // namespace
