// SPDX-License-Identifier: MIT
// Device LA v2 extension for the bidiagonal SVD family (v2/ext/bdsqr-bdsdc.hh) against host
// references. Guards (docs/developer/agent-guide.md section 8): non-natural ld, inc and batch
// stride, padding rows poisoned and checked untouched, complex data (zrot and zdrot), outputs
// over NaN, batch 1024 with period-3 data where every item must be bit-identical to its
// representative, a WG = 1024 launch, argmin ties on one owner and across owners, NaN skipped
// by the fmax reduction, hand-derived sort permutations (no reference built from the op).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/bdsqr-bdsdc.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// kPeriod representatives copied to every item (b from b % kPeriod).
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> int period_diffs(const std::vector<T>& h, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) d += !same_bits(h[std::size_t(b) * per_item + i], h[std::size_t(b % kPeriod) * per_item + i]);
    return d;
}

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
    template <int WG, class F> void run(int batch, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class V2BidiagTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};
template <class T> class V2BidiagRealTest : public V2BidiagTest<T> {};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(V2BidiagTest, Types);
TYPED_TEST_SUITE(V2BidiagRealTest, RealTypes);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> LD<T> conj_h(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- rot on columns (Group executor) and rows (Thread); SMat views give the same bits ------
// Columns use s of type T (zrot for complex), rows use a real s (zdrot / drot).
TYPED_TEST(V2BidiagTest, RotColumnsAndRowsThroughMatAndSMat) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int WG = 64, m = 75, n = 29, ld = m + 5, per = ld * n;
    const R c1 = R(std::cos(0.7)), c2 = R(std::cos(-1.9));
    const R s2 = R(std::sin(-1.9));
    T s1;
    if constexpr (is_cplx<T>) s1 = T(R(std::sin(0.7) * 0.6), R(std::sin(0.7) * 0.8));
    else s1 = T(std::sin(0.7));
    auto hA = batched<T>(per, 11);
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = m; i < ld; ++i) hA[std::size_t(b) * per + j * ld + i] = T(R(1e6));  // padding poison
    T* A = this->dev.up(hA);
    T* B = this->dev.up(hA);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> wg{it};
        const auto Am = v2::mat(A + std::ptrdiff_t(b) * per, m, n, ld);
        const auto Bm = v2::mat(B + std::ptrdiff_t(b) * per, m, n, ld);
        v2::rot(wg, Am.col(0), Am.col(3), c1, s1);
        v2::rot(wg, v2::as_smat(Bm).col(0), v2::as_smat(Bm).col(3), c1, s1);
        wg.barrier(v2::Mem::Global);
        if (wg.leader()) {
            v2::rot(v2::Thread{}, Am.row(2), Am.row(m - 1), c2, s2);
            v2::rot(v2::Thread{}, v2::trans(Bm).col(2), v2::trans(Bm).col(m - 1), c2, s2);
        }
    });
    const auto gA = this->dev.down(A, hA.size());
    const auto gB = this->dev.down(B, hA.size());
    int diffAB = 0, pad = 0;
    for (std::size_t k = 0; k < gA.size(); ++k) diffAB += !same_bits(gA[k], gB[k]);
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = m; i < ld; ++i) pad += !same_bits(gA[std::size_t(b) * per + j * ld + i], T(R(1e6)));
    EXPECT_EQ(diffAB, 0) << "SMat (as_smat / trans) differs from Mat col / row";
    EXPECT_EQ(pad, 0) << "padding rows written";
    EXPECT_EQ(period_diffs(gA, per), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> ref(per);
        for (int k = 0; k < per; ++k) ref[k] = wide(hA[std::size_t(b) * per + k]);
        auto rot_ref = [&](int i0, int st0, int i1, int st1, int len, LD<T> s, long double c) {
            for (int k = 0; k < len; ++k) {
                const LD<T> x = ref[i0 + k * st0], y = ref[i1 + k * st1];
                ref[i0 + k * st0] = c * x + s * y;
                ref[i1 + k * st1] = c * y - conj_h<T>(s) * x;
            }
        };
        rot_ref(0, 1, 3 * ld, 1, m, wide(s1), c1);
        rot_ref(2, ld, m - 1, ld, n, LD<T>(s2), c2);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) worst = std::max(worst, rel(gA[std::size_t(b) * per + j * ld + i], ref[j * ld + i], 2.0L));
    }
    EXPECT_VERIFY(T, Check::blas, 4, worst);
}

// ---- fill / set_unit / interleave into a slice; untouched slots keep their NaN ----------------
TYPED_TEST(V2BidiagTest, FillSetUnitInterleaveSlice) {
    using T = TypeParam;
    constexpr int WG = 64, n = 101, inc = 3, ylen = 2 * n * inc + 4, per = 3 * n + ylen;
    const T alpha = val<T>(7);
    auto hin = batched<T>(2 * n, 21);  // a = [0, n), b = [n, 2n - 1)
    std::vector<T> ho(std::size_t(kBatch) * per, nan_v<T>());
    T* in = this->dev.up(hin);
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> wg{it};
        T* ob = o + std::ptrdiff_t(b) * per;
        const auto a = v2::vec(in + std::ptrdiff_t(b) * 2 * n, n);
        const auto bb = v2::vec(in + std::ptrdiff_t(b) * 2 * n + n, n - 1);
        v2::fill(wg, alpha, v2::vec(ob, n));
        v2::fill(wg, v2::Zero{}, v2::vec(ob + n, n));
        v2::set_unit(wg, v2::vec(ob + 2 * n, n), n - 2);
        v2::interleave(wg, a, bb, v2::slice(v2::vec(ob + 3 * n, ylen), 1, 2 * n - 1, inc));
    });
    const auto g = this->dev.down(o, ho.size());
    EXPECT_EQ(period_diffs(g, per), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* ob = g.data() + std::size_t(b) * per;
        const T* hb = hin.data() + std::size_t(b) * 2 * n;
        for (int i = 0; i < n; ++i) {
            bad += !same_bits(ob[i], alpha);
            bad += !same_bits(ob[n + i], T(0));
            bad += !same_bits(ob[2 * n + i], i == n - 2 ? T(1) : T(0));
        }
        for (int k = 0; k < ylen; ++k) {
            const T got = ob[3 * n + k];
            const int q = k - 1;
            if (q >= 0 && q % inc == 0 && q / inc < 2 * n - 1) {
                const int s = q / inc;
                bad += !same_bits(got, (s % 2 == 0) ? hb[s / 2] : hb[n + s / 2]);
            } else {
                bad += !std::isnan(std::real(got));  // outside the slice: never written
            }
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- sumsq_many over two interleaved slices (WG 64 and the WG 1024 ceiling) and dgmv --------
template <class T, int WG> void run_sumsq_dgmv(Dev& dev) {
    using R = real_of<T>;
    constexpr int n = 1500, ld = 2 * n + 3, per = 2 * ld, outs = 2 + 2 * n;
    const T alpha = val<T>(5);
    auto hz = batched<T>(per, 31);
    auto hd = batched<T>(n, 41);
    std::vector<T> ho(std::size_t(kBatch) * outs, nan_v<T>());
    T* z = dev.up(hz);
    T* d = dev.up(hd);
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> wg{it};
        const auto col = v2::mat(z + std::ptrdiff_t(b) * per, 2 * n, 2, ld).col(1);
        const auto x0 = v2::slice(col, 0, n, 2), x1 = v2::slice(col, 1, n, 2);
        const auto [p, q] = v2::sumsq_many<v2::Scope::Group>(wg, x0, x1);
        T* ob = o + std::ptrdiff_t(b) * outs;
        if (wg.leader()) ob[0] = T(p), ob[1] = T(q);
        const auto dv = v2::vec(d + std::ptrdiff_t(b) * n, n);
        v2::dgmv(wg, alpha, dv, x0, v2::vec(ob + 2, n));
        v2::dgmv(wg, v2::One{}, dv, x1, v2::vec(ob + 2 + n, n));
    });
    const auto g = dev.down(o, ho.size());
    EXPECT_EQ(period_diffs(g, outs), 0) << "WG " << WG;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* ob = g.data() + std::size_t(b) * outs;
        long double sp = 0, sq = 0;
        for (int r = 0; r < n; ++r) {
            const LD<T> zp = wide(hz[std::size_t(b) * per + ld + 2 * r]), zq = wide(hz[std::size_t(b) * per + ld + 2 * r + 1]);
            const LD<T> dr = wide(hd[std::size_t(b) * n + r]);
            sp += std::norm(zp), sq += std::norm(zq);
            const long double sc = std::abs(wide(alpha)) * std::abs(dr) * std::abs(zp);
            worst = std::max({worst, rel(ob[2 + r], wide(alpha) * dr * zp, sc), rel(ob[2 + n + r], dr * zq, std::abs(dr) * std::abs(zq))});
        }
        EXPECT_VERIFY(T, Check::blas, n, std::max(rel(ob[0], LD<T>(sp), sp), rel(ob[1], LD<T>(sq), sq))) << "WG " << WG << " sumsq";
    }
    EXPECT_VERIFY(T, Check::blas, 2, worst) << "WG " << WG << " dgmv";
}
TYPED_TEST(V2BidiagTest, SumsqManyAndDgmv) {
    run_sumsq_dgmv<TypeParam, 64>(this->dev);
    run_sumsq_dgmv<TypeParam, 1024>(this->dev);
}

// ---- fmax_reduce / amax_fmax / iminloc / ifirst / any_nan / zero_if on Thread and WorkGroup --
enum Slot { kFmax, kAmax, kImin, kIminAllMax, kFirst, kFirstNone, kNan, kNanClean, kZeroBelow2, kZeroBelowHalf, kSlots };

template <class R, class Ex> void searches(const Ex& ex, const R* x, const R* xc, R* e1, R* e2, int n, R* out) {
    const auto xv = v2::vec(x, n, 2);    // NaN at 3 and 40, minimum -5 at 69 and 5 (same owner on 64 items) and 70
    const auto cv = v2::vec(xc, n);      // NaN-free
    const R fm = v2::fmax_reduce(ex, n, [&](int i) { return xv(i); }, R(-100));
    const R am = v2::amax_fmax(ex, xv, R(0));
    const int imin = v2::iminloc(ex, n, [&](int i) { return xv(i); });
    const int iall = v2::iminloc(ex, n, [&](int) { return std::numeric_limits<R>::max(); });
    const int f1 = v2::ifirst(ex, cv, [](R v) { return v > R(0.9); });
    const int f2 = v2::ifirst(ex, cv, [](R v) { return v > R(2); });
    const bool n1 = v2::any_nan(ex, xv), n2 = v2::any_nan(ex, cv);
    const auto ev1 = v2::vec(e1, n), ev2 = v2::vec(e2, n);
    const bool z1 = v2::zero_if(ex, ev1, [&](int i) { return sycl::fabs(ev1(i)) <= R(2); });
    const bool z2 = v2::zero_if(ex, ev2, [&](int i) { return sycl::fabs(ev2(i)) <= R(0.5); });
    if (ex.leader()) {
        out[kFmax] = fm, out[kAmax] = am, out[kImin] = R(imin), out[kIminAllMax] = R(iall), out[kFirst] = R(f1);
        out[kFirstNone] = R(f2), out[kNan] = R(n1), out[kNanClean] = R(n2), out[kZeroBelow2] = R(z1), out[kZeroBelowHalf] = R(z2);
    }
}

TYPED_TEST(V2BidiagRealTest, SearchesAndReductions) {
    using R = TypeParam;
    constexpr int WG = 64, n = 150;
    std::vector<R> hx(std::size_t(kBatch) * 2 * n), hc(std::size_t(kBatch) * n), he(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b) {
        const int r = b % kPeriod;
        for (int i = 0; i < n; ++i) {
            hx[std::size_t(b) * 2 * n + 2 * i] = uval<R>(1000 * r + i);
            hx[std::size_t(b) * 2 * n + 2 * i + 1] = R(-77);  // the inc-2 gap: read means wrong stride
            hc[std::size_t(b) * n + i] = uval<R>(5000 + 1000 * r + i) * R(0.8);
            he[std::size_t(b) * n + i] = R(3) * uval<R>(9000 + 1000 * r + i);
        }
        R* xb = hx.data() + std::size_t(b) * 2 * n;
        xb[2 * 3] = xb[2 * 40] = std::numeric_limits<R>::quiet_NaN();
        xb[2 * 5] = xb[2 * 69] = xb[2 * 70] = R(-5);
        xb[2 * 100] = R(7) + R(r);
        hc[std::size_t(b) * n + 17 + r] = R(0.95);
    }
    auto run = [&](auto thread_c) {
        constexpr bool on_thread = decltype(thread_c)::value;
        R* x = this->dev.up(hx);
        R* c = this->dev.up(hc);
        R* e1 = this->dev.up(he);
        R* e2 = this->dev.up(he);
        std::vector<R> ho(std::size_t(kBatch) * kSlots, std::numeric_limits<R>::quiet_NaN());
        R* o = this->dev.up(ho);
        this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const std::size_t bo = std::size_t(b);
            if constexpr (on_thread) {
                if (it.get_local_linear_id() == 0)
                    searches(v2::Thread{}, x + bo * 2 * n, c + bo * n, e1 + bo * n, e2 + bo * n, n, o + bo * kSlots);
            } else {
                searches(v2::WorkGroup<WG>{it}, x + bo * 2 * n, c + bo * n, e1 + bo * n, e2 + bo * n, n, o + bo * kSlots);
            }
        });
        const auto g = this->dev.down(o, ho.size());
        const auto g1 = this->dev.down(e1, he.size());
        EXPECT_EQ(period_diffs(g, kSlots), 0);
        for (int r = 0; r < kPeriod; ++r) {
            const R* s = g.data() + std::size_t(r) * kSlots;
            EXPECT_EQ(s[kFmax], R(7) + R(r)) << "NaN must be skipped";
            EXPECT_EQ(s[kAmax], R(7) + R(r));
            EXPECT_EQ(s[kImin], R(5)) << "lowest index of the tied minimum";
            EXPECT_EQ(s[kIminAllMax], R(0));
            EXPECT_EQ(s[kFirst], R(17 + r));
            EXPECT_EQ(s[kFirstNone], R(-1));
            EXPECT_EQ(s[kNan], R(1));
            EXPECT_EQ(s[kNanClean], R(0));
            EXPECT_EQ(s[kZeroBelow2], R(0)) << "entries with |e| > 2 survive, so not all zeroed";
            EXPECT_EQ(s[kZeroBelowHalf], R(0));
            for (int i = 0; i < n; ++i) {
                const R h = he[std::size_t(r) * n + i];
                EXPECT_EQ(g1[std::size_t(r) * n + i], std::fabs(h) <= R(2) ? R(0) : h) << i;
            }
        }
    };
    run(std::true_type{});
    run(std::false_type{});
}

// zero_if returning true: every element negligible.
TYPED_TEST(V2BidiagRealTest, ZeroIfAllNegligible) {
    using R = TypeParam;
    constexpr int WG = 64, n = 70;
    std::vector<R> he(std::size_t(kBatch) * n);
    for (std::size_t k = 0; k < he.size(); ++k) he[k] = R(1e-3) * uval<R>(k % (kPeriod * n));
    R* e = this->dev.up(he);
    std::vector<R> ho(kBatch, R(-1));
    R* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> wg{it};
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * n, n);
        const bool all = v2::zero_if(wg, ev, [&](int i) { return sycl::fabs(ev(i)) <= R(1e-2); });
        if (wg.leader()) o[b] = R(all);
    });
    const auto g = this->dev.down(o, ho.size());
    const auto ge = this->dev.down(e, he.size());
    EXPECT_EQ(std::count(g.begin(), g.end(), R(1)), kBatch);
    EXPECT_EQ(std::count(ge.begin(), ge.end(), R(0)), std::ptrdiff_t(ge.size()));
}

// ---- select_sort with a caller swap: hand-derived permutations ------------------------------
TYPED_TEST(V2BidiagRealTest, SelectSortCarriesColumns) {
    using R = TypeParam;
    constexpr int n = 4, rows = 2, ld = 5;
    // values {1, 3, 3, 2}; descending: swaps (0,1), (1,2), (2,3) -> ids {1, 2, 3, 0}.
    // ascending: swaps (1,3) -> {1, 2, 3, 3}, ids {0, 3, 2, 1} (tie at 2,3 keeps order).
    std::vector<R> h(std::size_t(kBatch) * 2 * (n + rows * ld));
    const R vals[n] = {1, 3, 3, 2};
    const int per = n + rows * ld;
    for (int b = 0; b < 2 * kBatch; ++b)
        for (int j = 0; j < n; ++j) {
            h[std::size_t(b) * per + j] = vals[j];
            h[std::size_t(b) * per + n + j * ld] = R(j);
            h[std::size_t(b) * per + n + j * ld + 1] = R(10 + j);
        }
    R* d = this->dev.up(h);
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int b = static_cast<int>(it.get_group_linear_id());
        for (int k = 0; k < 2; ++k) {
            R* p = d + std::ptrdiff_t(2 * b + k) * per;
            const auto x = v2::vec(p, n);
            const auto C = v2::mat(p + n, rows, n, ld);
            auto sw = [&](int i, int j) { v2::swap(v2::Thread{}, C.col(i), C.col(j)); };
            if (k == 0) v2::select_sort<SortOrder::Descending>(v2::Thread{}, x, sw);
            else v2::select_sort<SortOrder::Ascending>(v2::Thread{}, x, sw);
        }
    });
    const auto g = this->dev.down(d, h.size());
    const R want_v[2][n] = {{3, 3, 2, 1}, {1, 2, 3, 3}};
    const int want_id[2][n] = {{1, 2, 3, 0}, {0, 3, 2, 1}};
    int bad = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < n; ++j) {
                const R* p = g.data() + std::size_t(2 * b + k) * per;
                bad += p[j] != want_v[k][j];
                bad += p[n + j * ld] != R(want_id[k][j]) || p[n + j * ld + 1] != R(10 + want_id[k][j]);
            }
    EXPECT_EQ(bad, 0);
}

// ---- interop: vec_item / mat_item address item b like the v1 views (non-natural ld, inc, stride)
TYPED_TEST(V2BidiagTest, InteropItemOperands) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 7, n = 5, ld = m + 3, mstride = ld * n + 11, vn = 9, inc = 2, vstride = vn * inc + 5;
    std::vector<T> hm(std::size_t(kBatch) * mstride, nan_v<T>()), hv(std::size_t(kBatch) * vstride, nan_v<T>());
    T* pm = this->dev.up(hm);
    T* pv = this->dev.up(hv);
    const KernelMatrixView<T, MatrixFormat::Dense> km(pm, m, n, ld, mstride, kBatch);
    const VectorView<T> vv(pv, vn, kBatch, inc, vstride);
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = v2::mat_item(km, b);
        const auto x = v2::vec_item(vv, b);
        for (int j = 0; j < A.cols(); ++j)
            for (int i = 0; i < A.rows(); ++i) A(i, j) = T(R(b * 1000 + j * 100 + i));
        for (int i = 0; i < x.size(); ++i) x(i) = T(R(-(b * 1000 + i)));
    });
    const auto gm = this->dev.down(pm, hm.size());
    const auto gv = this->dev.down(pv, hv.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < mstride; ++k) {
            const int j = k / ld, i = k % ld;
            const T got = gm[std::size_t(b) * mstride + k];
            if (j < n && i < m) bad += !same_bits(got, T(R(b * 1000 + j * 100 + i)));
            else bad += !std::isnan(std::real(got));
        }
        for (int k = 0; k < vstride; ++k) {
            const T got = gv[std::size_t(b) * vstride + k];
            if (k % inc == 0 && k / inc < vn) bad += !same_bits(got, T(R(-(b * 1000 + k / inc))));
            else bad += !std::isnan(std::real(got));
        }
    }
    EXPECT_EQ(bad, 0);
}

}  // namespace
