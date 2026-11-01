// SPDX-License-Identifier: MIT
// Device LA v2 extension ops of the tridiagonal eigen family
// (include/batchlas/blas/device/v2/ext/tridiag-stebz-stein.hh) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural inc and ld with a NaN-poisoned
// pad, complex data, batch 1024 at period 3 bit-identical to its representative, DynWorkGroup at
// local ranges 1, 7, 100, 256 and the device ceiling, graded eigenvalue inputs with analytic
// references, Sturm counts straddling an eigenvalue by one ulp, a pivot straddling the floor.
// Deliberate breaks to run when this first builds (one per axis, expected red set):
//   for_each_owned(DynWorkGroup) strides by 32          -> DynReductions (wg 7, 100)
//   gershgorin drops |e_{i-1}|                           -> DynReductions, SturmBisect
//   sturm_count counts q < 0 instead of q <= 0           -> SturmBisect (straddle)
//   gttrf never swaps                                    -> Gttrf (piv, residual)
//   gttrf floor uses <= tol                              -> Gttrf (straddle)
//   refl2 drops the factor 2                             -> House2Refl2
//   rot_phase ignores ph / gram2 drops the conj          -> Level1Extras (complex)
//   laset writes ld rows                                 -> Level1Extras (pad poison)
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

#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/tridiag-stebz-stein.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;
template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;

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
// kPeriod representatives copied to every item (item b is a twin of b % kPeriod).
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}
template <class T> int bitdiff_items(const std::vector<T>& got, int per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) bad += !same_bits(got[std::size_t(b) * per_item + i], got[std::size_t(b % kPeriod) * per_item + i]);
    return bad;
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
    template <class F> void run(int groups, int wg, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * wg, std::size_t(wg)), [=](sycl::nd_item<1> it) { f(it); });
         }).wait_and_throw();
    }
    int max_wg() { return static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)); }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class TseReal : public ::testing::Test {
protected:
    Dev dev;
};
template <class T> class TseAll : public ::testing::Test {
protected:
    Dev dev;
};
using RealTypes = ::testing::Types<float, double>;
using AllTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(TseReal, RealTypes);
TYPED_TEST_SUITE(TseAll, AllTypes);

// ---- DynWorkGroup + tridiagonal reductions ------------------------------------------------

// lanst_inf, gershgorin, amax_fmax are exact (min / max of the same T expressions as the host);
// dot and sumsq go through the core reductions with the runtime stride.
TYPED_TEST(TseReal, DynReductionsAtEveryRuntimeRange) {
    using T = TypeParam;
    constexpr int n = 133, incd = 2, ince = 3, slots = 7;
    auto hd = batched<T>(n * incd, 11), he = batched<T>((n - 1) * ince, 12);
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    std::vector<int> ranges = {1, 7, 100, 256};
    if (this->dev.max_wg() >= 1024) ranges.push_back(1024);
    for (int wg : ranges) {
        std::vector<T> ho(std::size_t(kBatch) * slots, std::numeric_limits<T>::quiet_NaN());
        T* o = this->dev.up(ho);
        this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::DynWorkGroup ex{it};
            const auto dv = v2::vec(d + std::ptrdiff_t(b) * n * incd, n, incd);
            const auto ev = v2::vec(e + std::ptrdiff_t(b) * (n - 1) * ince, n - 1, ince);
            const T ln = v2::lanst_inf(ex, dv, ev);
            const auto g = v2::gershgorin(ex, dv, ev);
            const T am = v2::amax_fmax(ex, dv), dt = v2::dot(ex, dv, dv), ss = v2::sumsq(ex, ev);
            if (ex.leader()) {
                T* ob = o + std::ptrdiff_t(b) * slots;
                ob[0] = ln, ob[1] = g.lo, ob[2] = g.hi, ob[3] = g.e2max, ob[4] = am, ob[5] = dt, ob[6] = ss;
            }
        });
        const auto got = this->dev.down(o, ho.size());
        EXPECT_EQ(bitdiff_items(got, slots), 0) << "wg " << wg;
        for (int b = 0; b < kPeriod; ++b) {
            const T* D = hd.data() + std::size_t(b) * n * incd;
            const T* E = he.data() + std::size_t(b) * (n - 1) * ince;
            T ln = 0, lo = std::numeric_limits<T>::max(), hi = -std::numeric_limits<T>::max(), e2 = 0, am = 0;
            long double dt = 0, ss = 0;
            for (int i = 0; i < n; ++i) {
                const T left = i > 0 ? std::fabs(E[(i - 1) * ince]) : T(0), right = i < n - 1 ? std::fabs(E[i * ince]) : T(0);
                const T di = D[i * incd];
                ln = std::max(ln, std::fabs(di) + left + right);
                lo = std::min(lo, di - (left + right));
                hi = std::max(hi, di + (left + right));
                e2 = std::max(e2, right * right);
                am = std::max(am, std::fabs(di));
                dt += (long double)di * di;
                if (i < n - 1) ss += (long double)E[i * ince] * E[i * ince];
            }
            const T* g = got.data() + std::size_t(b) * slots;
            EXPECT_EQ(g[0], ln) << "lanst_inf wg " << wg;
            EXPECT_EQ(g[1], lo) << "gershgorin lo wg " << wg;
            EXPECT_EQ(g[2], hi) << "gershgorin hi wg " << wg;
            EXPECT_EQ(g[3], e2) << "gershgorin e2max wg " << wg;
            EXPECT_EQ(g[4], am) << "amax_fmax wg " << wg;
            EXPECT_VERIFY(T, Check::blas, n, rel(g[5], dt, dt)) << "dot wg " << wg;
            EXPECT_VERIFY(T, Check::blas, n, rel(g[6], ss, ss)) << "sumsq wg " << wg;
        }
    }
}

// ---- Sturm count and bisection on graded inputs with analytic eigenvalues ------------------

// Item kind 0: Toeplitz (2, -1), lambda_j = 2 - 2 cos((j + 1) pi / (n + 1)).
// Item kind 1: diagonal d_i = (-1)^i 2^-i (graded over 2^-39), e = 0: lambda = sort(d), and the
// count at lambda_j is j + 1 while one ulp below it is j (q = 0 counts: dlaebz's pivmin clamp).
// Item kind 2: kind 0 scaled by 2^-60 with e_0 = 0 (a split, small magnitudes, pivmin path).
TYPED_TEST(TseReal, SturmBisectGradedAndStraddled) {
    using T = TypeParam;
    constexpr int n = 40, wg = 32;
    std::vector<T> hd(std::size_t(kBatch) * n), he(std::size_t(kBatch) * n, T(0));
    std::vector<std::vector<long double>> exact(kPeriod);
    for (int b = 0; b < kBatch; ++b) {
        const int kind = b % kPeriod;
        const long double s = kind == 2 ? std::ldexp(1.0L, -60) : 1.0L;
        for (int i = 0; i < n; ++i) {
            hd[std::size_t(b) * n + i] = kind == 1 ? T(std::ldexp(i % 2 ? -1.0L : 1.0L, -i)) : T(2 * s);
            if (i < n - 1) he[std::size_t(b) * n + i] = kind == 1 ? T(0) : T(-s);
        }
        if (kind == 2) he[std::size_t(b) * n] = T(0);
    }
    const long double pi = std::acos(-1.0L);
    for (int j = 0; j < n; ++j) exact[0].push_back(2 - 2 * std::cos((j + 1) * pi / (n + 1)));
    for (int i = 0; i < n; ++i) exact[1].push_back(hd[std::size_t(n) + i]);
    std::sort(exact[1].begin(), exact[1].end());
    {  // kind 2: eigenvalues of the 1x1 block {2s} and of the (n-1) Toeplitz block, scaled.
        const long double s = std::ldexp(1.0L, -60);
        exact[2].push_back(2 * s);
        for (int j = 0; j < n - 1; ++j) exact[2].push_back(s * (2 - 2 * std::cos((j + 1) * pi / n)));
        std::sort(exact[2].begin(), exact[2].end());
    }
    std::vector<T> hx(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j) hx[std::size_t(b) * n + j] = T(exact[b % kPeriod][j]);
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    T* x = this->dev.up(hx);
    std::vector<T> hl(std::size_t(kBatch) * n, std::numeric_limits<T>::quiet_NaN());
    std::vector<int> hc(std::size_t(kBatch) * n * 2, -1);
    T* lam = this->dev.up(hl);
    int* cnt = this->dev.up(hc);
    this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::DynWorkGroup ex{it};
        const v2::Thread th{};
        const auto dv = v2::vec(d + std::ptrdiff_t(b) * n, n);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * n, n - 1);
        const auto g = v2::gershgorin(ex, dv, ev);
        const T norm = sycl::max(sycl::fabs(g.lo), sycl::fabs(g.hi));
        const T pad = std::numeric_limits<T>::epsilon() * norm * T(2) + std::numeric_limits<T>::min();
        const T pivmin = std::numeric_limits<T>::min() * sycl::max(T(1), g.e2max);
        v2::for_each_owned(ex, n, [&](int j) {
            lam[std::ptrdiff_t(b) * n + j] = v2::bisect_eig(th, dv, ev, j, g.lo - pad, g.hi + pad, T(0), pivmin, 400);
            const T xj = x[std::ptrdiff_t(b) * n + j];
            cnt[(std::ptrdiff_t(b) * n + j) * 2] = v2::sturm_count(th, dv, ev, xj, pivmin);
            cnt[(std::ptrdiff_t(b) * n + j) * 2 + 1] = v2::sturm_count(th, dv, ev, sycl::nextafter(xj, -std::numeric_limits<T>::infinity()), pivmin);
        });
    });
    const auto gl = this->dev.down(lam, hl.size());
    const auto gc = this->dev.down(cnt, hc.size());
    EXPECT_EQ(bitdiff_items(gl, n), 0);
    EXPECT_EQ(bitdiff_items(gc, 2 * n), 0);
    for (int b = 0; b < kPeriod; ++b) {
        long double tn = 0, err = 0;
        for (int i = 0; i < n; ++i) tn = std::max<long double>(tn, std::fabs((long double)hd[std::size_t(b) * n + i]) + 2 * std::fabs((long double)he[std::size_t(b) * n + i]));
        for (int j = 0; j < n; ++j) err = std::max(err, std::fabs(gl[std::size_t(b) * n + j] - exact[b][j]));
        EXPECT_VERIFY(T, Check::values, n, static_cast<double>(err / tn)) << "bisect_eig kind " << b;
        if (b == 1) {
            for (int j = 0; j < n; ++j) {
                EXPECT_EQ(gc[(std::size_t(b) * n + j) * 2], j + 1) << "count at lambda_" << j;
                EXPECT_EQ(gc[(std::size_t(b) * n + j) * 2 + 1], j) << "count one ulp below lambda_" << j;
            }
        }
    }
}

// ---- gttrf / gttrs --------------------------------------------------------------------------

// Rows with |dl| > |d| take the interchange branch. Row 0 straddles the pivot floor: kind 2
// has d0 = tol / 2 (raised to tol), kind 1 has d0 = 2 tol (kept).
TYPED_TEST(TseReal, GttrfGttrsBothBranchesAndPivotFloor) {
    using T = TypeParam;
    constexpr int n = 37, wg = 32;
    const T tol = T(1e-3);
    auto hdl = batched<T>(n, 21), hd = batched<T>(n, 22), hdu = batched<T>(n, 23), hb = batched<T>(n, 24);
    for (int b = 0; b < kBatch; ++b) {
        T* D = hd.data() + std::size_t(b) * n;
        T* DL = hdl.data() + std::size_t(b) * n;
        for (int i = 0; i < n; ++i)
            if (i % 3 == 1) DL[i] = T(4) * (std::fabs(D[i]) + T(0.25));  // forces the interchange
            else D[i] += std::copysign(T(3), D[i]);                       // diagonally dominant
        if (b % kPeriod == 2) DL[0] = T(0), D[0] = tol / T(2);
        if (b % kPeriod == 1) DL[0] = T(0), D[0] = tol * T(2);
    }
    T* dl = this->dev.up(hdl);
    T* dd = this->dev.up(hd);
    T* du = this->dev.up(hdu);
    T* x = this->dev.up(hb);
    std::vector<T> h0(std::size_t(kBatch) * n, std::numeric_limits<T>::quiet_NaN());
    T* du2 = this->dev.up(h0);
    std::vector<unsigned char> hp(std::size_t(kBatch) * n, 7);
    unsigned char* piv = this->dev.up(hp);
    this->dev.run(kBatch / wg, wg, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const v2::Thread th{};
        const std::ptrdiff_t o = std::ptrdiff_t(b) * n;
        const auto vdl = v2::vec(dl + o, n - 1), vd = v2::vec(dd + o, n), vdu = v2::vec(du + o, n - 1);
        const auto vdu2 = v2::vec(du2 + o, n - 2), vp = v2::vec(piv + o, n - 1), vx = v2::vec(x + o, n);
        v2::gttrf(th, vdl, vd, vdu, vdu2, vp, tol);
        v2::gttrs(th, vdl, vd, vdu, vdu2, vp, vx);
    });
    const auto gx = this->dev.down(x, hb.size());
    const auto gd = this->dev.down(dd, hd.size());
    const auto gp = this->dev.down(piv, hp.size());
    EXPECT_EQ(bitdiff_items(gx, n), 0);
    for (int b = 0; b < kPeriod; ++b) {
        const std::size_t o = std::size_t(b) * n;
        int swaps = 0;
        for (int i = 0; i < n - 1; ++i) swaps += gp[o + i] == 1;
        EXPECT_GE(swaps, n / 3 - 1) << "the interchange branch ran";
        EXPECT_EQ(gp[o], (std::fabs(hdl[o]) > std::fabs(hd[o])) ? 1 : 0);
        if (b == 2) {
            EXPECT_EQ(gd[o], tol) << "pivot below the floor is raised to tol";
            continue;  // the system was perturbed on purpose
        }
        if (b == 1) EXPECT_EQ(gd[o], tol * T(2)) << "pivot above the floor is kept";
        long double res = 0, an = 0, xn = 0;
        for (int i = 0; i < n; ++i) {
            long double r = (long double)hd[o + i] * gx[o + i] - hb[o + i];
            if (i > 0) r += (long double)hdl[o + i - 1] * gx[o + i - 1];
            if (i < n - 1) r += (long double)hdu[o + i] * gx[o + i + 1];
            res = std::max(res, std::fabs(r));
            an = std::max(an, std::fabs((long double)hd[o + i]) + (i > 0 ? std::fabs((long double)hdl[o + i - 1]) : 0) +
                                  (i < n - 1 ? std::fabs((long double)hdu[o + i]) : 0));
            xn = std::max(xn, std::fabs((long double)gx[o + i]));
        }
        EXPECT_VERIFY(T, Check::solve, n, static_cast<double>(res / (an * xn))) << "gttrs residual item " << b;
    }
}

// ---- 2-element reflectors ---------------------------------------------------------------------

// house2 maps (a0, a1) to (-sign(a0) ||a||, 0) with unit v; refl2 on a scattered stripe (Thread,
// Refs) and on a column pair of a matrix with ld = m + 3 (DynWorkGroup) against long double.
TYPED_TEST(TseReal, House2Refl2) {
    using T = TypeParam;
    constexpr int m = 150, ld = m + 3, npair = 6, wg = 100;
    const std::array<T, 2 * npair> pairs = {T(3), T(4), T(-3), T(4), T(0), T(5), T(1e-3), T(1), T(5), T(0), T(-0.5), T(-7)};
    auto ha = batched<T>(2 * ld, 31);
    T* a = this->dev.up(ha);
    T* pr = this->dev.up(std::vector<T>(pairs.begin(), pairs.end()));
    std::vector<T> hv(std::size_t(kBatch) * 4 * npair, std::numeric_limits<T>::quiet_NaN());
    T* vo = this->dev.up(hv);
    this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::DynWorkGroup ex{it};
        T* ob = vo + std::ptrdiff_t(b) * 4 * npair;
        if (ex.leader()) {
            for (int p = 0; p < npair; ++p) {
                T x0 = pr[2 * p], x1 = pr[2 * p + 1];
                const auto v = v2::house2(x0, x1);
                v2::refl2(v2::Thread{}, v, v2::refs(x0), v2::refs(x1));
                ob[4 * p] = v[0], ob[4 * p + 1] = v[1], ob[4 * p + 2] = x0, ob[4 * p + 3] = x1;
            }
        }
        const auto A = v2::mat(a + std::ptrdiff_t(b) * 2 * ld, m, 2, ld);
        v2::refl2(ex, v2::house2(pr[0], pr[1]), A.col(0), A.col(1));
    });
    const auto gv = this->dev.down(vo, hv.size());
    const auto ga = this->dev.down(a, ha.size());
    EXPECT_EQ(bitdiff_items(gv, 4 * npair), 0);
    EXPECT_EQ(bitdiff_items(ga, 2 * ld), 0);
    for (int p = 0; p < npair; ++p) {
        const T* g = gv.data() + 4 * p;
        const long double a0 = pairs[2 * p], a1 = pairs[2 * p + 1], an = std::hypot(a0, a1);
        const long double alpha = a0 >= 0 ? -an : an;
        EXPECT_VERIFY(T, Check::blas, 2, rel(T(g[0] * g[0] + g[1] * g[1]), 1.0L, 1.0L)) << "unit v, pair " << p;
        EXPECT_VERIFY(T, Check::blas, 2, rel(g[2], alpha, an)) << "alpha, pair " << p;
        EXPECT_VERIFY(T, Check::blas, 2, rel(g[3], 0.0L, an)) << "annihilated, pair " << p;
    }
    const long double v0 = gv[0], v1 = gv[1];
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < m; ++r) {
            const long double x = ha[std::size_t(b) * 2 * ld + r], y = ha[std::size_t(b) * 2 * ld + ld + r];
            const long double t = x * v0 + y * v1, s = std::fabs(x) + std::fabs(y);
            worst = std::max({worst, rel(ga[std::size_t(b) * 2 * ld + r], x - 2 * v0 * t, s), rel(ga[std::size_t(b) * 2 * ld + ld + r], y - 2 * v1 * t, s)});
        }
    EXPECT_VERIFY(T, Check::blas, 2, worst);
    for (int b = 0; b < kBatch; ++b)
        for (int r = m; r < ld; ++r)
            for (int c = 0; c < 2; ++c) {
                const std::size_t k = std::size_t(b) * 2 * ld + std::size_t(c) * ld + r;
                EXPECT_TRUE(same_bits(ga[k], ha[k])) << "pad row " << r << " of column " << c << " written";
            }
}

// ---- elementwise / Jacobi pair, every type, three executors -----------------------------------

enum Slot { kSumsq, kGxx, kGyy, kGxy, kSlots };

// fill and laset over NaN (laset leaves the ld pad poisoned), rscal_div, sumsq, gram2, rot_phase.
template <class T, class MakeEx>
void level1_extras(Dev& dev, int wg, int groups, int items_per_group, MakeEx make_ex, const char* where) {
    using R = real_of<T>;
    constexpr int n = 133, incx = 2, m = 29, nc = 17, ld = m + 3;
    const R c = R(0.6), s = R(0.8), dv = R(3.25);
    const T a = val<T>(901), bb = val<T>(902);
    T ph = val<T>(903);
    if constexpr (is_cplx<T>) ph /= std::abs(ph);
    else ph = T(-1);
    auto hx = batched<T>(n * incx, 41), hy = batched<T>(n, 42);
    const T nan = T(std::numeric_limits<R>::quiet_NaN());
    std::vector<T> hf(std::size_t(kBatch) * 2 * n, nan), hA(std::size_t(kBatch) * ld * nc, nan), hs(std::size_t(kBatch) * kSlots, nan);
    std::vector<T> hq(std::size_t(kBatch) * 3 * n, nan);
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    T* f = dev.up(hf);
    T* A = dev.up(hA);
    T* so = dev.up(hs);
    T* q = dev.up(hq);
    dev.run(groups, wg, [=](sycl::nd_item<1> it) {
        const auto ex = make_ex(it);
        const int b = static_cast<int>(it.get_group_linear_id()) * items_per_group +
                      (items_per_group > 1 ? static_cast<int>(it.get_local_linear_id()) : 0);
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * incx, n, incx);
        const auto yv = v2::vec(y + std::ptrdiff_t(b) * n, n);
        const auto fv = v2::mat(f + std::ptrdiff_t(b) * 2 * n, n, 2, n);
        v2::fill(ex, v2::Scalar{a}, fv.col(0));
        v2::fill(ex, v2::Zero{}, fv.col(1));
        v2::laset(ex, v2::Scalar{a}, v2::Scalar{bb}, v2::mat(A + std::ptrdiff_t(b) * ld * nc, m, nc, ld));
        const auto qv = v2::mat(q + std::ptrdiff_t(b) * 3 * n, n, 3, n);
        v2::copy(ex, xv, qv.col(0));
        v2::rscal_div(ex, dv, qv.col(0));
        v2::copy(ex, xv, qv.col(1));
        v2::copy(ex, yv, qv.col(2));
        v2::rot_phase(ex, c, s, ph, qv.col(1), qv.col(2));
        const R ss = v2::sumsq(ex, xv);
        const auto g = v2::gram2(ex, xv, yv);
        if (ex.leader()) {
            T* ob = so + std::ptrdiff_t(b) * kSlots;
            ob[kSumsq] = T(ss), ob[kGxx] = T(g.xx), ob[kGyy] = T(g.yy), ob[kGxy] = g.xy;
        }
    });
    const auto gf = dev.down(f, hf.size()), gA = dev.down(A, hA.size()), gs = dev.down(so, hs.size()), gq = dev.down(q, hq.size());
    EXPECT_EQ(bitdiff_items(gf, 2 * n) + bitdiff_items(gA, ld * nc) + bitdiff_items(gs, kSlots) + bitdiff_items(gq, 3 * n), 0) << where;
    double worst = 0;
    int poison = 0, filled = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < nc; ++j)
            for (int r = m; r < ld; ++r) poison += !std::isnan(std::real(gA[(std::size_t(b) * nc + j) * ld + r]));
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < n; ++i) filled += !same_bits(gf[std::size_t(b) * 2 * n + i], a) + !same_bits(gf[std::size_t(b) * 2 * n + n + i], T(0));
        for (int j = 0; j < nc; ++j)
            for (int r = 0; r < m; ++r) filled += !same_bits(gA[(std::size_t(b) * nc + j) * ld + r], r == j ? bb : a);
        long double ss = 0, xx = 0, yy = 0;
        LD<T> xy = 0;
        for (int i = 0; i < n; ++i) {
            const LD<T> xi = LD<T>(hx[std::size_t(b) * n * incx + i * incx]), yi = LD<T>(hy[std::size_t(b) * n + i]);
            ss += std::norm(xi), xx += std::norm(xi), yy += std::norm(yi);
            if constexpr (is_cplx<T>) xy += std::conj(xi) * yi;
            else xy += xi * yi;
            const LD<T> yp = yi * LD<T>(ph), sc = std::abs(xi) + std::abs(yi);
            const T* qb = gq.data() + std::size_t(b) * 3 * n;
            worst = std::max({worst, rel(qb[i], xi / (long double)dv, std::abs(xi)), rel(qb[n + i], (long double)c * xi - (long double)s * yp, sc),
                              rel(qb[2 * n + i], (long double)s * xi + (long double)c * yp, sc)});
        }
        const T* g = gs.data() + std::size_t(b) * kSlots;
        worst = std::max({worst, rel(g[kSumsq], LD<T>(ss), ss), rel(g[kGxx], LD<T>(xx), xx), rel(g[kGyy], LD<T>(yy), yy),
                          rel(g[kGxy], xy, std::sqrt(xx * yy))});
    }
    EXPECT_EQ(poison, 0) << where << ": laset wrote the ld pad";
    EXPECT_EQ(filled, 0) << where << ": fill / laset values";
    EXPECT_VERIFY(T, Check::blas, n, worst) << where;
}

TYPED_TEST(TseAll, Level1ExtrasOnEveryExecutor) {
    using T = TypeParam;
    level1_extras<T>(this->dev, 100, kBatch, 1, [](sycl::nd_item<1> it) { return v2::DynWorkGroup{it}; }, "DynWorkGroup(100)");
    level1_extras<T>(this->dev, 64, kBatch, 1, [](sycl::nd_item<1> it) { return v2::WorkGroup<64>{it}; }, "WorkGroup<64>");
    level1_extras<T>(this->dev, 32, kBatch / 32, 32, [](sycl::nd_item<1>) { return v2::Thread{}; }, "Thread");
}

static_assert(v2::Refs<double, 3>{}.size() == 3);
static_assert(std::is_same_v<v2::elem_t<v2::Refs<float, 2>>, float>);

}  // namespace
