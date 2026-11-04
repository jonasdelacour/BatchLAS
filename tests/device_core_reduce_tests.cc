// SPDX-License-Identifier: MIT
// device_core_reduce_tests: reduce.hh (design A.2) against host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts through every shuffle;
// partial mask ranges (Lo > 1, Hi < 32) so a butterfly that crosses its block is visible; both
// orders; the leader-only contract of shift_reduce checked on the leader alone; argmax ties
// (Lowest and Highest), NaN (First and Skip), empty candidates, and a WG = 1024 launch where
// every one of 1024 items must be bit-identical to its period-3 representative (SLM scratch).
// Red set (each break was made once, 2026-11-04, and this suite failed): the butterfly mask loop
// starting at Hi instead of Hi / 2; v2.2 (AtomicMinSlot): fetch_min replaced by fetch_max.
#include <tests/device_core_common.hh>
#include <batchlas/device/reduce.hh>

#include <climits>
#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> class DeviceCoreReduceTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreReduceTest, AllTypes);

struct Plus {
    template <class T> T operator()(const T& a, const T& b) const { return dv::Arith<T>::add(a, b); }
};

// butterfly<Lo, Hi> on a sub-group: lane l ends with the sum over lanes that agree with l
// outside bits [log Lo, log Hi).
template <class T, int Lo, int Hi, class Order> void check_butterfly(Dev& d) {
    constexpr int SG = 32, groups = 64;
    const auto x_h = vals<T>(SG * groups, 11 + Lo + Hi);
    T* x = d.up(x_h);
    T* out = d.template alloc<T>(x_h.size(), T(0));
    d.template run<SG>(groups, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const int g = static_cast<int>(it.get_global_linear_id());
        out[g] = dv::butterfly<Lo, Hi, Order>(sg, x[g], Plus{});
    });
    const auto got = d.down(out, x_h.size());
    for (int grp = 0; grp < groups; ++grp)
        for (int l = 0; l < SG; ++l) {
            constexpr int range = (Hi - 1) & ~(Lo - 1);  // the lane bits the masks flip
            LD<T> ref = 0;
            for (int m = 0; m < SG; ++m)
                if (((m ^ l) & ~range) == 0) ref += wide(x_h[grp * SG + m]);
            EXPECT_LE(err(got[grp * SG + l], ref), tol<T>(Hi / Lo)) << "Lo " << Lo << " Hi " << Hi << " lane " << l;
        }
}

TYPED_TEST(DeviceCoreReduceTest, ButterflyCoversExactlyItsMaskRange) {
    using T = TypeParam;
    check_butterfly<T, 1, 32, dv::bfly::Down>(this->dev);
    check_butterfly<T, 1, 32, dv::bfly::Asc>(this->dev);
    check_butterfly<T, 2, 16, dv::bfly::Down>(this->dev);
    check_butterfly<T, 4, 8, dv::bfly::Asc>(this->dev);
    check_butterfly<T, 1, 4, dv::bfly::Down>(this->dev);
}

TYPED_TEST(DeviceCoreReduceTest, ShuffleXorMovesBothPartsAndShiftReduceFeedsTheLeader) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int SG = 32;
    const auto x_h = vals<T>(SG, 21);
    T* x = d.up(x_h);
    T* sh = d.template alloc<T>(SG, T(0));
    T* red = d.template alloc<T>(SG, T(0));
    d.template run<SG>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const int l = static_cast<int>(sg.get_local_linear_id());
        sh[l] = dv::shuffle_xor(sg, x[l], 5u);
        red[l] = dv::shift_reduce<8>(sg, x[l], Plus{});
    });
    const auto s = d.down(sh, SG), r = d.down(red, SG);
    for (int l = 0; l < SG; ++l) EXPECT_TRUE(same_bits(s[l], x_h[l ^ 5])) << l;
    for (int blk = 0; blk < SG; blk += 8) {  // leader-only: lanes 0, 8, 16, 24
        LD<T> ref = 0;
        for (int m = blk; m < blk + 8; ++m) ref += wide(x_h[m]);
        EXPECT_LE(err(r[blk], ref), tol<T>(8)) << blk;
    }
}

TEST(DeviceCoreReduceHost, ArgmaxMergeTieAndNanRules) {
    using A = dv::ArgMax<float>;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const A none{-1, -std::numeric_limits<float>::infinity()};
    const A a{3, 2.f}, b{1, 2.f}, c{5, 7.f}, n1{4, nan}, n2{2, nan};
    auto low = [](A x, A y) { return dv::argmax_merge<dv::tie::Lowest, dv::nan::First>(x, y); };
    auto high = [](A x, A y) { return dv::argmax_merge<dv::tie::Highest, dv::nan::First>(x, y); };
    auto skip = [](A x, A y) { return dv::argmax_merge<dv::tie::Lowest, dv::nan::Skip>(x, y); };
    EXPECT_EQ(low(a, b).index, 1);
    EXPECT_EQ(low(b, a).index, 1);
    EXPECT_EQ(high(a, b).index, 3);
    EXPECT_EQ(high(b, a).index, 3);
    EXPECT_EQ(low(a, c).index, 5);
    EXPECT_EQ(low(none, a).index, 3);
    EXPECT_EQ(low(a, none).index, 3);
    EXPECT_EQ(low(c, n1).index, 4);
    EXPECT_EQ(low(n1, n2).index, 2);
    EXPECT_EQ(skip(c, n1).index, 5);
    EXPECT_EQ(skip(n1, a).index, 3);
}

// argmax of |x| over a WG = 1024 work-group through allreduce(ex, Scratch, ArgMax, merge): ties
// planted at two indices; 1024 items, items repeat with period 3 and must be bit-identical.
TYPED_TEST(DeviceCoreReduceTest, WorkGroupArgmaxThroughScratchIsUniformAndBitIdentical) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int WG = 1024, n = 1000, items = 1024, period = 3;
    std::vector<T> x_h(std::size_t(n) * period);
    for (int p = 0; p < period; ++p)
        for (int i = 0; i < n; ++i) x_h[p * n + i] = val<T>(p * 7919 + i);
    // Ties: the maximum |re| + |im| appears twice in pattern 0 and 1; pattern 2 holds a NaN.
    x_h[0 * n + 100] = x_h[0 * n + 900] = dv::lift<T>(R(9));
    x_h[1 * n + 50] = dv::lift<T>(R(-9));
    x_h[1 * n + 950] = dv::lift<T>(R(9));
    x_h[2 * n + 600] = dv::qnan<T>();
    T* x = d.up(x_h);
    int* lo = d.template alloc<int>(items, -7);
    int* hi = d.template alloc<int>(items, -7);
    int* sk = d.template alloc<int>(items, -7);
    using AM = dv::ArgMax<R>;
    constexpr int slots = dv::scratch_elements<dv::op::Reduce, dv::WorkGroup<WG>, AM>();
    d.template run<WG>(items, sizeof(AM) * slots, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const T* xb = x + (b % period) * n;
        const dv::Scratch<AM, slots> scr{reinterpret_cast<AM*>(lm)};
        const int i = ex.lid();
        const AM mine = i < n ? AM{i, dv::abs1(xb[i])} : AM{-1, R(0)};
        const AM l = dv::allreduce(ex, scr, mine, dv::ArgMaxMerge<dv::tie::Lowest, dv::nan::First>{});
        const AM h = dv::allreduce(ex, scr, mine, dv::ArgMaxMerge<dv::tie::Highest, dv::nan::First>{});
        const AM s = dv::allreduce(ex, scr, mine, dv::ArgMaxMerge<dv::tie::Lowest, dv::nan::Skip>{});
        if (i == 517) {  // a non-leader item reads the uniform result
            lo[b] = l.index;
            hi[b] = h.index;
            sk[b] = s.index;
        }
    });
    const auto L = d.down(lo, items), H = d.down(hi, items), S = d.down(sk, items);
    const int want_lo[period] = {100, 50, 600}, want_hi[period] = {900, 950, 600};
    for (int b = 0; b < items; ++b) {
        EXPECT_EQ(L[b], want_lo[b % period]) << b;
        EXPECT_EQ(H[b], want_hi[b % period]) << b;
        if (b % period != 2) EXPECT_EQ(S[b], want_lo[b % period]) << b;  // Skip == Lowest without a NaN
    }
    // pattern 2 under Skip: the largest finite key (host argmax, lowest index on ties).
    int best = -1;
    R bv = -1;
    for (int i = 0; i < n; ++i) {
        const T v = x_h[2 * n + i];
        const R k = std::abs(std::real(v)) + std::abs(std::imag(v));
        if (!std::isnan(k) && k > bv) bv = k, best = i;
    }
    for (int b = 2; b < items; b += period) EXPECT_EQ(S[b], best) << b;
}


// v2.2: AtomicMinSlot on a WG = 1024 work-group: every item offers a value, the minimum (held by
// one item of the last sub-group) wins; no offer leaves INT_MAX (or_zero: 0).
TYPED_TEST(DeviceCoreReduceTest, AtomicMinSlotKeepsTheMinimumOffer) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int WG = 1024;
    int* out = d.template alloc<int>(3, -1);
    d.template run<WG>(1, sizeof(int), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<WG, 32> ex{it};
        const auto slot = dv::atomic_min_slot(reinterpret_cast<int*>(lm));
        slot.reset(ex);
        ex.barrier();
        slot.offer(ex.lid() == 1000 ? 7 : 7 + (ex.lid() * 37) % 991 + 1);
        ex.barrier();
        const int v = slot.value();
        ex.barrier();
        slot.reset(ex);
        ex.barrier();
        if (ex.leader()) {
            out[0] = v;
            out[1] = slot.value();
            out[2] = slot.or_zero();
        }
    });
    const auto o = d.down(out, 3);
    EXPECT_EQ(o[0], 7);
    EXPECT_EQ(o[1], INT_MAX);
    EXPECT_EQ(o[2], 0);
    (void)sizeof(T);
}

}  // namespace
