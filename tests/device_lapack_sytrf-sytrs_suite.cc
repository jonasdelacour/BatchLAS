// SPDX-License-Identifier: MIT
// sytrf / sytrs device routine internals (batchlas/device/lapack/sytrf.hh and sytrs.hh, their
// detail / detail pieces) against long-double host references.
// The shared ops the kernels use (RtGroup, SubGroup, for_each_owned over a Range, lacpy,
// for_each_owned_2d, abs1, arith::Fma) are covered by device_core_ops_tests.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld, the index-reversed view
// (rev = true, the Upper route), complex data with nonzero imaginary parts, the unread triangle
// and the trsv diagonal NaN-poisoned, batch 1024 with period-3 data where every item must be
// bit-identical to its representative, arg-max ties inside one owner and across owners and
// sub-groups at widths 32 / 128 and the 256 ceiling (kSytrfWgMax) and on SubGroup<32> inside a
// wider work-group, NaN keys, an empty range, the Bunch-Kaufman thresholds straddled by one ulp,
// a 2 x 2 block across the lane-31 / 32 register boundary, output slots pre-poisoned.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/lapack/sytrf.hh>
#include <batchlas/device/lapack/sytrs.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

// Folded into device_lapack_sytrf_tests (one binary per routine): its own namespace keeps the two
// translation units' helper and kernel names apart.
namespace device_lapack_sytrf_sytrs_suite {

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class T> T val(std::uint64_t i) {
    if constexpr (is_cplx<T>) return T(uval<real_of<T>>(2 * i), uval<real_of<T>>(2 * i + 1));
    else return uval<T>(i);
}
template <class T> T nanv() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(item pointer)), copied to every item b from b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < std::min(kPeriod, items); ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(h.size(), q());
        q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    template <int WG, class F> void run(int batch, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceLapackSytrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceLapackSytrfTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// Every item equal to its representative, bit for bit.
template <class T> int twins_differ(const std::vector<T>& got, std::size_t per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (std::size_t e = 0; e < per_item; ++e) bad += !same_bits(got[b * per_item + e], got[(b % kPeriod) * per_item + e]);
    return bad;
}

// ---- the reversed views ---------------------------------------------------------------------

// device::lacpy<Lower> from a reversed RevMat (ld n + 5; its physical strict lower triangle and padding
// rows NaN: never read) into a plain one (ld n + 1, pre-poisoned: the strict upper never written),
// then column 0 through a reversed RevVec. n = 37 at width 64.
TYPED_TEST(DeviceLapackSytrfTest, RevMatRevVecLowerCopy) {
    using T = TypeParam;
    constexpr int n = 37, lg = n + 5, ls = n + 1;
    const std::size_t pg = std::size_t(lg) * n, ps = std::size_t(ls) * n;
    const auto hg = batched<T>(int(pg), 11, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < lg; ++i)
                if (i > j || i >= n) p[std::size_t(i) + std::size_t(j) * lg] = nanv<T>();
    });
    std::vector<T> hs(std::size_t(kBatch) * ps, nanv<T>()), hv(std::size_t(kBatch) * n, nanv<T>());
    T* gp = this->dev.up(hg);
    T* sp = this->dev.up(hs);
    T* vp = this->dev.up(hv);
    this->dev.template run<64>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const device::RtGroup<> g(it);
        const auto G = device::detail::rev_mat(gp + b * std::ptrdiff_t(pg), lg, n, true);
        const auto S = device::detail::rev_mat(sp + b * std::ptrdiff_t(ps), ls, n, false);
        device::lacpy<device::Part::Lower, Transpose::NoTrans, device::alg::DivMod>(g, G, S);
        device::sync<device::Scope::Group>(g, device::Mem::Global);
        const device::detail::RevVec<T> v{vp + b * n, n, true};
        device::for_each_owned(g, n, [&](int i) { v(i) = S(i, 0); });
    });
    const auto gs = this->dev.down(sp, hs.size());
    const auto gv = this->dev.down(vp, hv.size());
    EXPECT_EQ(twins_differ(gs, ps), 0);
    EXPECT_EQ(twins_differ(gv, std::size_t(n)), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* g = hg.data() + b * pg;
        const T* s = gs.data() + b * ps;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ls; ++i) {
                const T want = i >= j && i < n ? g[std::size_t(n - 1 - i) + std::size_t(n - 1 - j) * lg] : nanv<T>();
                bad += !same_bits(s[std::size_t(i) + std::size_t(j) * ls], want);
            }
        for (int i = 0; i < n; ++i) bad += !same_bits(gv[std::size_t(b) * n + (n - 1 - i)], s[std::size_t(i)]);
    }
    EXPECT_EQ(bad, 0) << "logical (i, j) of the reversal is physical (n-1-i, n-1-j)";
}

// ---- the deterministic arg-max and sums -----------------------------------------------------

enum IamaxSlot { kLow, kHigh, kSkip, kEmpty, kNanLow, kNanHigh, kISlots };
enum SumSlot { kVal, kEmptyVal, kS0, kS1, kSEmpty, kSSlots };

// x: three tied maxima (|re| + |im| = 4) at 10, 42, 138 (one owner at width 32; 10 / 138 one owner
// at 128); y: NaN at 20 (real part) and 77 (imaginary part), 1e30 at 50. Sums: x^T z and z^T z
// over [60, n), as ldl_solve composes them (for_each_owned, then sum_reduce), and an empty range.
template <class T, int WG, bool SgOnly>
void run_iamax(Dev& dev, int n, const std::vector<T>& hx, const std::vector<T>& hy, const std::vector<T>& hz, std::vector<int>& oi,
               std::vector<T>& ov) {
    using R = real_of<T>;
    using A = device::arith_t<device::arith::Fma, T>;
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    T* z = dev.up(hz);
    std::vector<int> hoi(std::size_t(kBatch) * kISlots, -7);
    std::vector<T> hov(std::size_t(kBatch) * kSSlots, nanv<T>());
    int* oip = dev.up(hoi);
    T* ovp = dev.up(hov);
    dev.template run<WG>(kBatch, (2 * sizeof(R) + sizeof(int)) * 32, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        if (SgOnly && sg.get_group_linear_id() != 0) return;
        R* sv = reinterpret_cast<R*>(lm);
        const device::detail::ArgMaxScratch<R> am{sv, reinterpret_cast<int*>(sv + 64)};
        const device::detail::SumScratch<R> ss{sv, sv + 32};
        const T* xb = x + b * n;
        const T* yb = y + b * n;
        const T* zb = z + b * n;
        auto xv = [=](int i) -> const T& { return xb[i]; };
        auto yv = [=](int i) -> const T& { return yb[i]; };
        auto body = [&](const auto& g) {
            const auto lo = device::detail::iamax_bk(g, am, xv, device::Range{3, n}, true);
            const auto hi = device::detail::iamax_bk(g, am, xv, device::Range{3, n}, false);
            const auto sk = device::detail::iamax_bk<true>(g, am, xv, device::Range{0, n}, true, lo.index);
            const auto em = device::detail::iamax_bk(g, am, xv, device::Range{n, n}, true);
            const auto nl = device::detail::iamax_bk(g, am, yv, device::Range{0, n}, true);
            const auto nh = device::detail::iamax_bk(g, am, yv, device::Range{0, n}, false);
            T p0 = A::zero(), p1 = A::zero(), pe = A::zero();
            device::for_each_owned(g, device::Range{60, n}, [&](int i) {
                p0 = A::add(p0, A::mul(xb[i], zb[i]));
                p1 = A::add(p1, A::mul(zb[i], zb[i]));
            });
            device::for_each_owned(g, device::Range{n, n}, [&](int i) { pe = A::add(pe, A::mul(xb[i], zb[i])); });
            const T s0 = device::detail::sum_reduce(g, ss, p0);
            const T s1 = device::detail::sum_reduce(g, ss, p1);
            const T se = device::detail::sum_reduce(g, ss, pe);
            if (g.leader()) {
                int* o = oip + b * kISlots;
                o[kLow] = lo.index, o[kHigh] = hi.index, o[kSkip] = sk.index, o[kEmpty] = em.index, o[kNanLow] = nl.index, o[kNanHigh] = nh.index;
                T* q = ovp + b * kSSlots;
                q[kVal] = T(lo.value), q[kEmptyVal] = T(em.value), q[kS0] = s0, q[kS1] = s1, q[kSEmpty] = se;
            }
        };
        if constexpr (SgOnly) body(device::SubGroup<32>{sg});
        else body(device::RtGroup<>(it));
    });
    oi = dev.down(oip, hoi.size());
    ov = dev.down(ovp, hov.size());
}

template <class T> void check_iamax(int n, const std::vector<T>& hx, const std::vector<T>& hz,
                                    const std::vector<int>& oi, const std::vector<T>& ov, const char* where) {
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < kISlots; ++k) bad += oi[std::size_t(b) * kISlots + k] != oi[std::size_t(b % kPeriod) * kISlots + k];
        for (int k = 0; k < kSSlots; ++k) bad += !same_bits(ov[std::size_t(b) * kSSlots + k], ov[std::size_t(b % kPeriod) * kSSlots + k]);
    }
    EXPECT_EQ(bad, 0) << where << ": items differ from their representative";
    for (int b = 0; b < kPeriod; ++b) {
        const int* o = oi.data() + std::size_t(b) * kISlots;
        const T* q = ov.data() + std::size_t(b) * kSSlots;
        EXPECT_EQ(o[kLow], 10) << where;
        EXPECT_EQ(o[kHigh], 138) << where;
        EXPECT_EQ(o[kSkip], 42) << where;
        EXPECT_EQ(o[kEmpty], -1) << where;
        EXPECT_EQ(o[kNanLow], 20) << where << " (NaN ranks as Inf)";
        EXPECT_EQ(o[kNanHigh], 77) << where;
        EXPECT_EQ(std::real(q[kVal]), real_of<T>(4)) << where;
        EXPECT_EQ(std::real(q[kEmptyVal]), real_of<T>(-1)) << where;
        EXPECT_TRUE(same_bits(q[kSEmpty], T(0))) << where << " an empty sum is an exact zero";
        LD<T> d0 = 0, d1 = 0;
        long double a0 = 0, a1 = 0;
        for (int i = 60; i < n; ++i) {
            const LD<T> xi = wide(hx[std::size_t(b) * n + i]), zi = wide(hz[std::size_t(b) * n + i]);
            d0 += xi * zi, d1 += zi * zi;
            a0 += std::abs(xi) * std::abs(zi), a1 += std::abs(zi) * std::abs(zi);
        }
        EXPECT_VERIFY(T, Check::blas, n, std::max(rel(q[kS0], d0, a0), rel(q[kS1], d1, a1))) << where << " sum_reduce";
    }
}

TYPED_TEST(DeviceLapackSytrfTest, IamaxBkTiesNanAndSums) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 150;
    auto mk = [](R re, R im) {
        if constexpr (is_cplx<T>) return T(re, im);
        else return T(re + (re < 0 ? -std::abs(im) : std::abs(im)));
    };
    const auto hx = batched<T>(n, 41, [&](T* p) {
        p[10] = mk(R(3), R(-1)), p[42] = mk(R(-1), R(3)), p[138] = mk(R(-2), R(2));
    });
    const auto hy = batched<T>(n, 42, [&](T* p) {
        p[20] = T(std::numeric_limits<R>::quiet_NaN());
        if constexpr (is_cplx<T>) p[77] = T(R(0.5), std::numeric_limits<R>::quiet_NaN());
        else p[77] = std::numeric_limits<R>::quiet_NaN();
        p[50] = T(R(1e30));
    });
    const auto hz = batched<T>(n, 43, [](T*) {});
    std::vector<int> oi;
    std::vector<T> ov;
    run_iamax<T, 32, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "RtGroup width 32");
    run_iamax<T, 128, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "RtGroup width 128");
    run_iamax<T, 256, false>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "RtGroup width 256 (kSytrfWgMax)");
    run_iamax<T, 64, true>(this->dev, n, hx, hy, hz, oi, ov);
    check_iamax(n, hx, hz, oi, ov, "SubGroup<32>, sub-group 0 of 64");
}

// ---- Bunch-Kaufman decisions and the 2 x 2 blocks -------------------------------------------

// bk_stage / bk_choice over cases straddling each threshold by one ulp (LAPACK ?SYTF2's tests
// written out here); the result is item 0's broadcast, read on item 31.
TYPED_TEST(DeviceLapackSytrfTest, BkStageAndChoiceThresholds) {
    using T = TypeParam;
    using R = real_of<T>;
    // The thresholds are built on the device's alpha (a float sqrt need not be correctly rounded).
    std::vector<R> ha(1, R(0));
    R* ap = this->dev.up(ha);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() == 0) ap[0] = device::detail::bk_alpha<R>();
    });
    const R alpha = this->dev.down(ap, 1)[0];
    EXPECT_NEAR(alpha, (R(1) + std::sqrt(R(17))) / R(8), 2 * std::numeric_limits<R>::epsilon());
    const R nan = std::numeric_limits<R>::quiet_NaN();
    auto below = [](R x) { return std::nextafter(x, R(0)); };
    auto above = [](R x) { return std::nextafter(x, R(1e30)); };
    // {absakk, colmax, rowmax, absimax}
    std::vector<std::array<R, 4>> cases = {
        {0, 0, 1, 1}, {nan, 1, 1, 1}, {nan, 0, 1, 1}, {1, 0, 1, 1},                              // zero / NaN / keep
        {alpha * 2, 2, 1, 1}, {below(alpha * 2), 2, 3, 1},                                       // keep at the line, row search below it
        {R(0.5), 1, 1, 1}, {R(0.1), 1, 1, alpha}, {R(0.1), 1, 1, below(alpha)},                   // absimax straddles alpha rowmax
        {R(0.1), 1, R(0.2), R(0.01)}, {above(alpha / 4), 1, 4, 0}, {alpha / 4, 1, 4, 0}, {below(alpha / 4), 1, 4, 0}};
    const int nc = static_cast<int>(cases.size());
    std::vector<R> hin;
    for (const auto& c : cases) hin.insert(hin.end(), c.begin(), c.end());
    std::vector<int> hout(std::size_t(nc) * 3, -9);
    R* in = this->dev.up(hin);
    int* out = this->dev.up(hout);
    this->dev.template run<64>(nc, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int c = static_cast<int>(it.get_group_linear_id());
        const device::RtGroup<> g(it);
        const R* p = in + 4 * c;
        // only item 0 sees the real inputs: the verdict must be its broadcast
        const bool l0 = g.leader();
        const R absakk = l0 ? p[0] : R(-5), colmax = l0 ? p[1] : R(7), rowmax = l0 ? p[2] : R(1), absimax = l0 ? p[3] : R(9);
        const R al = device::detail::bk_alpha<R>();
        const int stage = device::detail::bk_stage(g, absakk, colmax, al);
        int kp = 3, kstep = 1;
        if (stage == device::detail::kBkNeedRow) device::detail::bk_choice(g, absakk, colmax, rowmax, absimax, al, 3, 8, kp, kstep);
        int* o = out + 3 * c;
        if (it.get_local_linear_id() == 47) o[0] = stage, o[1] = kp, o[2] = kstep;
    });
    const auto got = this->dev.down(out, hout.size());
    for (int c = 0; c < nc; ++c) {
        const auto [akk, cmax, rmax, aimax] = cases[c];
        int stage = device::detail::kBkNeedRow, kp = 3, kstep = 1;
        if (std::max(akk, cmax) == R(0) || std::isnan(akk)) stage = device::detail::kBkZero;
        else if (akk >= alpha * cmax) stage = device::detail::kBkKeep;
        else if (akk >= alpha * cmax * (cmax / rmax)) kp = 3;
        else if (aimax >= alpha * rmax) kp = 8;
        else kp = 8, kstep = 2;
        EXPECT_EQ(got[3 * c], stage) << "case " << c;
        EXPECT_EQ(got[3 * c + 1], kp) << "case " << c;
        EXPECT_EQ(got[3 * c + 2], kstep) << "case " << c;
    }
}

// The 2 x 2 D solve both ways, ldl_dsolve_row per row (?SYTRS's form, the cta / blocked tiers)
// and BkBlock2 ([b0 b1] D^-1, the factor's form), against Cramer in long double; and the 1 x 1 row.
TYPED_TEST(DeviceLapackSytrfTest, LdlBlockSolves) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int per = 5;  // dkk, dk1k1, off, b0, b1
    auto h = batched<T>(per, 51, [](T* p) {
        p[0] *= R(0.2), p[1] *= R(0.2), p[2] = T(R(-2)) + p[2] * R(0.5);
    });
    std::vector<T> ho(std::size_t(kBatch) * 5, nanv<T>());
    T* in = this->dev.up(h);
    T* out = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() != 0) return;
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const T* p = in + b * per;
        T* o = out + b * 5;
        const T dkk = p[0], dk1 = p[1], off = p[2], b0 = p[3], b1 = p[4];
        o[0] = device::detail::ldl_dsolve_row(1, b0, b1, dkk, dk1, off);
        o[1] = device::detail::ldl_dsolve_row(2, b1, b0, dk1, dkk, off);
        const auto blk = device::detail::bk_block2(dkk, off, dk1);
        o[2] = blk.first(b0, b1);
        o[3] = blk.second(b0, b1);
        o[4] = device::detail::ldl_dsolve_row(0, b0, b0, dkk, dkk, dkk);
    });
    const auto got = this->dev.down(out, ho.size());
    EXPECT_EQ(twins_differ(got, 5), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h.data() + b * per;
        const T* o = got.data() + b * 5;
        const LD<T> a = wide(p[0]), d = wide(p[1]), e = wide(p[2]), x0 = wide(p[3]), x1 = wide(p[4]);
        const LD<T> det = a * d - e * e;
        const LD<T> r0 = (d * x0 - e * x1) / det, r1 = (a * x1 - e * x0) / det;
        const long double s0 = (std::abs(d) * std::abs(x0) + std::abs(e) * std::abs(x1)) / std::abs(det);
        const long double s1 = (std::abs(a) * std::abs(x1) + std::abs(e) * std::abs(x0)) / std::abs(det);
        worst = std::max({worst, rel(o[0], r0, s0), rel(o[1], r1, s1), rel(o[2], r0, s0), rel(o[3], r1, s1),
                          rel(o[4], x0 / a, std::abs(x0 / a))});
    }
    EXPECT_VERIFY(T, Check::blas, 8, worst);
}

// ---- register-distributed sweeps --------------------------------------------------------------

// trsv_frag NoTrans then Trans (unit lower; the diagonal and the strict upper triangle are NaN),
// and ldl_dsolve_frag with a 2 x 2 block on rows 31 / 32 (across the register boundary), n = 50,
// R = 2, one sub-group per item, ld n + 3.
TYPED_TEST(DeviceLapackSytrfTest, TrsvFragAndDsolveFrag) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 50, ld = n + 3, RR = 2;
    const std::size_t pL = std::size_t(ld) * n;
    auto hL = batched<T>(int(pL), 61, [](T* p) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                T& v = p[std::size_t(i) + std::size_t(j) * ld];
                v = i > j && i < n ? v * R(0.1) : nanv<T>();
            }
    });
    // D: kind by t % 5 = {0: 1 x 1, 1 / 2 and 3 / 4: 2 x 2}; t = 31 opens a block (31 % 5 == 1)
    auto hD = batched<T>(int(pL), 62, [](T* p) {
        for (int t = 0; t < n; ++t) {
            T& v = p[std::size_t(t) * (ld + 1)];
            v = t % 5 == 0 ? T(R(2)) + v * R(0.5) : v * R(0.2);
        }
    });
    std::vector<int> hk(n);
    for (int t = 0; t < n; ++t) hk[t] = t % 5 == 0 ? 0 : (t % 5) % 2 == 1 ? 1 : 2;
    auto he = batched<T>(n, 63, [](T* p) {
        for (int t = 0; t < n; ++t) p[t] = T(R(2.5)) + p[t] * R(0.5);
    });
    const auto hb = batched<T>(n, 64, [](T*) {});
    std::vector<T> ho(std::size_t(kBatch) * 3 * n, nanv<T>());
    T* Lp = this->dev.up(hL);
    T* Dp = this->dev.up(hD);
    int* kp = this->dev.up(hk);
    T* ep = this->dev.up(he);
    T* bp = this->dev.up(hb);
    T* op = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const auto L = device::mat(Lp + b * std::ptrdiff_t(pL), n, n, ld);
        const auto Dm = device::mat(Dp + b * std::ptrdiff_t(pL), n, n, ld);
        const T* bb = bp + b * n;
        T* o = op + b * 3 * n;
        const device::SubGroup<32> ex{sg};
        device::Frag<T, RR, device::Cyclic> x, y;
        for (int r = 0; r < RR; ++r) {
            const int t = lane + 32 * r;
            x[r] = t < n ? bb[t] : T(0);
            y[r] = x[r];
        }
        device::detail::trsv_frag<Transpose::NoTrans>(ex, L, n, x);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[lane + 32 * r] = x[r];
        device::detail::trsv_frag<Transpose::Trans>(ex, L, n, x);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[n + lane + 32 * r] = x[r];
        device::detail::ldl_dsolve_frag(ex, Dm, kp, ep + b * n, n, y);
        for (int r = 0; r < RR; ++r)
            if (lane + 32 * r < n) o[2 * n + lane + 32 * r] = y[r];
    });
    const auto got = this->dev.down(op, ho.size());
    EXPECT_EQ(twins_differ(got, 3 * std::size_t(n)), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* l = hL.data() + b * pL;
        const T* dd = hD.data() + b * pL;
        const T* e = he.data() + b * n;
        const T* o = got.data() + b * 3 * n;
        std::vector<LD<T>> y(n), z(n);
        std::vector<long double> sy_(n), sz(n);
        for (int t = 0; t < n; ++t) {
            y[t] = wide(hb[b * n + t]), sy_[t] = std::abs(y[t]);
            for (int k = 0; k < t; ++k) {
                y[t] -= wide(l[t + std::size_t(k) * ld]) * y[k];
                sy_[t] += std::abs(wide(l[t + std::size_t(k) * ld])) * sy_[k];
            }
        }
        for (int t = n - 1; t >= 0; --t) {
            z[t] = y[t], sz[t] = sy_[t];
            for (int i = t + 1; i < n; ++i) {
                z[t] -= wide(l[i + std::size_t(t) * ld]) * z[i];
                sz[t] += std::abs(wide(l[i + std::size_t(t) * ld])) * sz[i];
            }
        }
        for (int t = 0; t < n; ++t) worst = std::max({worst, rel(o[t], y[t], sy_[t]), rel(o[n + t], z[t], sz[t])});
        for (int t = 0; t < n;) {
            const LD<T> x0 = wide(hb[b * n + t]);
            if (hk[t] == 0) {
                const LD<T> d0 = wide(dd[std::size_t(t) * (ld + 1)]);
                worst = std::max(worst, rel(o[2 * n + t], x0 / d0, std::abs(x0 / d0)));
                t += 1;
                continue;
            }
            const LD<T> x1 = wide(hb[b * n + t + 1]), a = wide(dd[std::size_t(t) * (ld + 1)]),
                        d = wide(dd[std::size_t(t + 1) * (ld + 1)]), off = wide(e[t]);
            const LD<T> det = a * d - off * off;
            const long double s0 = (std::abs(d) * std::abs(x0) + std::abs(off) * std::abs(x1)) / std::abs(det);
            const long double s1 = (std::abs(a) * std::abs(x1) + std::abs(off) * std::abs(x0)) / std::abs(det);
            worst = std::max({worst, rel(o[2 * n + t], (d * x0 - off * x1) / det, s0), rel(o[2 * n + t + 1], (a * x1 - off * x0) / det, s1)});
            t += 2;
        }
    }
    EXPECT_VERIFY(T, Check::blas, n, worst);
}

// ---- ?SYCONV on one column ----------------------------------------------------------------------

// Logical ipiv {3, -5, -5, 5, 5} on 5 x 5, one column per item of a width-32 RtGroup. Expected by
// hand: column 0 gets rows (2 4) then (3 4): [a0 a1 a4 a2 a3]; column 1 (the 2 x 2 block at 1)
// moves L(2, 1) to e[1], zeroes it, then (3 4); column 2 gets (3 4); columns 3 and 4 unchanged.
TYPED_TEST(DeviceLapackSytrfTest, SyconvColumnByHand) {
    using T = TypeParam;
    constexpr int n = 5, ld = 7;
    std::vector<T> h(std::size_t(ld) * n);
    for (std::size_t e = 0; e < h.size(); ++e) h[e] = val<T>(700 + e);
    std::vector<T> he(n, nanv<T>());
    const std::vector<int> ps = {3, -5, -5, 5, 5};
    T* L = this->dev.up(h);
    T* E = this->dev.up(he);
    const int* P = this->dev.up(ps);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const device::RtGroup<> g(it);
        const auto M = device::mat(L, n, n, ld);
        device::for_each_owned(g, n, [&](int j) { device::syconv_col<T>(M, P, n, j, E); });
    });
    const auto got = this->dev.down(L, h.size());
    const auto ge = this->dev.down(E, he.size());
    auto H = [&](int i, int j) { return h[std::size_t(i) + std::size_t(j) * ld]; };
    std::vector<T> want = h;
    auto W = [&](int i, int j) -> T& { return want[std::size_t(i) + std::size_t(j) * ld]; };
    W(2, 0) = H(4, 0), W(3, 0) = H(2, 0), W(4, 0) = H(3, 0);
    W(2, 1) = T(0), W(3, 1) = H(4, 1), W(4, 1) = H(3, 1);
    W(3, 2) = H(4, 2), W(4, 2) = H(3, 2);
    int bad = 0;
    for (std::size_t e = 0; e < h.size(); ++e) bad += !same_bits(got[e], want[e]);
    EXPECT_EQ(bad, 0);
    EXPECT_TRUE(same_bits(ge[1], H(2, 1))) << "the 2 x 2 off-diagonal moves to e[1]";
    for (int j : {0, 2, 3, 4}) EXPECT_TRUE(std::isnan(std::real(ge[j]))) << "e[" << j << "] is written only for a block start";
}

}  // namespace

}  // namespace device_lapack_sytrf_sytrs_suite
