// SPDX-License-Identifier: MIT
// Device LA v2 ops private to the tridiagonal eigen kernels (src/ops/syev/tridiag/tridiag_v2.hh)
// against host references. Core ops (RtGroup, for_each_owned, allreduce, map, laset, dot, axpy,
// sumsq, scal) are covered by device_blas_v2_tests.
// Guards (docs/developer/agent-guide.md section 8): non-natural inc with a poisoned pad, batch
// 1024 at period 3 bit-identical to its representative, RtGroup at local ranges 1, 7, 100, 256
// and the device ceiling, graded eigenvalue inputs with analytic references, Sturm counts
// straddling an eigenvalue by one ulp, ld = m + 3 pad rows left untouched.
// Deliberate breaks to run when this first builds (one per axis, expected red set):
//   gershgorin drops |e_{i-1}|                 -> Reductions, SturmBisect
//   gershgorin reduces lo with maximum         -> Reductions
//   lanst_inf reads e(i) at i = n - 1          -> Reductions (poisoned pad)
//   sturm_count counts q < 0 instead of q <= 0 -> SturmBisect (straddle)
//   bisect_eig returns left, not the midpoint  -> SturmBisect (kind 2)
//   refl2 drops the factor 2                   -> House2Refl2
//   refl2(Thread, Refs) skips element N - 1    -> House2Refl2 (stripe)
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/blas/device/v2/dla.hh>
#include <src/ops/syev/tridiag/tridiag_v2.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
namespace tdv2 = batchlas::device::v2::tridiag;
using batchlas::verify::Check;

namespace {

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
// kPeriod representatives copied to every item (item b is a twin of b % kPeriod). With inc > 1
// the gaps hold `pad`, a value that wins every max / min an op that read it would take.
template <class T> std::vector<T> batched(int n, int inc, std::uint64_t seed, T pad) {
    std::vector<T> h(std::size_t(kBatch) * n * inc, pad);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) h[(std::size_t(b) * n + i) * inc] = uval<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> double rel(T got, long double ref, long double scale) {
    const long double d = std::fabs((long double)got - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<T>::min()));
}
template <class T> int bitdiff_items(const std::vector<T>& got, int per_item) {
    int bad = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) bad += !same_bits(got[std::size_t(b) * per_item + i], got[std::size_t(b % kPeriod) * per_item + i]);
    return bad;
}

// The test's queue and device buffers; no queue when no SYCL device is visible (the fixture skips).
struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
                                   : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
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
    int max_wg() { return static_cast<int>(ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)); }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class TridiagV2 : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
    }
    Dev dev;
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TridiagV2, RealTypes);

// ---- tridiagonal reductions on a runtime-width group ---------------------------------------

// lanst_inf and gershgorin are exact (min / max of the same T expressions as the host), so the
// device must match the host bit for bit at every local range.
TYPED_TEST(TridiagV2, ReductionsAtEveryRuntimeRange) {
    using T = TypeParam;
    constexpr int n = 133, incd = 2, ince = 3, slots = 4;
    const T poison = T(1e30);
    auto hd = batched<T>(n, incd, 11, poison), he = batched<T>(n, ince, 12, poison);
    for (int b = 0; b < kBatch; ++b) he[(std::size_t(b) * n + n - 1) * ince] = poison;  // e(n-1) is past the end
    T* d = this->dev.up(hd);
    T* e = this->dev.up(he);
    std::vector<int> ranges = {1, 7, 100, 256};
    if (this->dev.max_wg() > 256) ranges.push_back(std::min(this->dev.max_wg(), 1024));
    for (int wg : ranges) {
        std::vector<T> ho(std::size_t(kBatch) * slots, std::numeric_limits<T>::quiet_NaN());
        T* o = this->dev.up(ho);
        this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const v2::RtGroup<> ex{it};
            const auto dv = v2::vec(d + std::ptrdiff_t(b) * n * incd, n, incd);
            const auto ev = v2::vec(e + std::ptrdiff_t(b) * n * ince, n - 1, ince);
            const T ln = tdv2::lanst_inf(ex, dv, ev);
            const auto g = tdv2::gershgorin(ex, dv, ev);
            if (ex.leader()) {
                T* ob = o + std::ptrdiff_t(b) * slots;
                ob[0] = ln, ob[1] = g.lo, ob[2] = g.hi, ob[3] = g.e2max;
            }
        });
        const auto got = this->dev.down(o, ho.size());
        EXPECT_EQ(bitdiff_items(got, slots), 0) << "wg " << wg;
        for (int b = 0; b < kPeriod; ++b) {
            const T* D = hd.data() + std::size_t(b) * n * incd;
            const T* E = he.data() + std::size_t(b) * n * ince;
            T ln = 0, lo = std::numeric_limits<T>::max(), hi = -std::numeric_limits<T>::max(), e2 = 0;
            for (int i = 0; i < n; ++i) {
                const T left = i > 0 ? std::fabs(E[(i - 1) * ince]) : T(0), right = i < n - 1 ? std::fabs(E[i * ince]) : T(0);
                const T di = D[i * incd];
                ln = std::max(ln, std::fabs(di) + left + right);
                lo = std::min(lo, di - (left + right));
                hi = std::max(hi, di + (left + right));
                e2 = std::max(e2, right * right);
            }
            const T* g = got.data() + std::size_t(b) * slots;
            EXPECT_EQ(g[0], ln) << "lanst_inf wg " << wg;
            EXPECT_EQ(g[1], lo) << "gershgorin lo wg " << wg;
            EXPECT_EQ(g[2], hi) << "gershgorin hi wg " << wg;
            EXPECT_EQ(g[3], e2) << "gershgorin e2max wg " << wg;
            EXPECT_LT(g[0], T(16)) << "a poisoned pad was read, wg " << wg;
        }
    }
}

// ---- Sturm count and bisection on graded inputs with analytic eigenvalues ------------------

// Item kind 0: Toeplitz (2, -1), lambda_j = 2 - 2 cos((j + 1) pi / (n + 1)).
// Item kind 1: diagonal d_i = (-1)^i 2^-i (graded over 2^-39), e = 0: lambda = sort(d), and the
// count at lambda_j is j + 1 while one ulp below it is j (q = 0 counts: dlaebz's pivmin clamp).
// Item kind 2: kind 0 scaled by 2^-60 with e_0 = 0 (a split, small magnitudes, pivmin path).
TYPED_TEST(TridiagV2, SturmBisectGradedAndStraddled) {
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
        const v2::RtGroup<> ex{it};
        const v2::Thread th{};
        const auto dv = v2::vec(d + std::ptrdiff_t(b) * n, n);
        const auto ev = v2::vec(e + std::ptrdiff_t(b) * n, n - 1);
        const auto g = tdv2::gershgorin(ex, dv, ev);
        const T norm = sycl::max(sycl::fabs(g.lo), sycl::fabs(g.hi));
        const T pad = std::numeric_limits<T>::epsilon() * norm * T(2) + std::numeric_limits<T>::min();
        const T pivmin = std::numeric_limits<T>::min() * sycl::max(T(1), g.e2max);
        v2::for_each_owned(ex, n, [&](int j) {
            lam[std::ptrdiff_t(b) * n + j] = tdv2::bisect_eig(th, dv, ev, j, g.lo - pad, g.hi + pad, T(0), pivmin, 400);
            const T xj = x[std::ptrdiff_t(b) * n + j];
            cnt[(std::ptrdiff_t(b) * n + j) * 2] = tdv2::sturm_count(th, dv, ev, xj, pivmin);
            cnt[(std::ptrdiff_t(b) * n + j) * 2 + 1] = tdv2::sturm_count(th, dv, ev, sycl::nextafter(xj, -std::numeric_limits<T>::infinity()), pivmin);
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

// ---- 2-element reflectors ---------------------------------------------------------------------

// house2 maps (a0, a1) to (-sign(a0) ||a||, 0) with unit v; refl2 on scattered stripes (Thread,
// Refs of 1 and of 3) and on a column pair of a matrix with ld = m + 3 (RtGroup at 100) against
// long double.
TYPED_TEST(TridiagV2, House2Refl2) {
    using T = TypeParam;
    constexpr int m = 150, ld = m + 3, npair = 6, wg = 100, nstripe = 3;
    const std::array<T, 2 * npair> pairs = {T(3), T(4), T(-3), T(4), T(0), T(5), T(1e-3), T(1), T(5), T(0), T(-0.5), T(-7)};
    auto ha = batched<T>(2 * ld, 1, 31, T(0));
    T* a = this->dev.up(ha);
    T* pr = this->dev.up(std::vector<T>(pairs.begin(), pairs.end()));
    std::vector<T> hv(std::size_t(kBatch) * 4 * npair, std::numeric_limits<T>::quiet_NaN());
    T* vo = this->dev.up(hv);
    std::vector<T> hs(std::size_t(kBatch) * 2 * nstripe, std::numeric_limits<T>::quiet_NaN());
    T* so = this->dev.up(hs);
    this->dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup<> ex{it};
        T* ob = vo + std::ptrdiff_t(b) * 4 * npair;
        if (ex.leader()) {
            for (int p = 0; p < npair; ++p) {
                T x0 = pr[2 * p], x1 = pr[2 * p + 1];
                const auto v = tdv2::house2(x0, x1);
                tdv2::refl2(v2::Thread{}, v, tdv2::refs(x0), tdv2::refs(x1));
                ob[4 * p] = v[0], ob[4 * p + 1] = v[1], ob[4 * p + 2] = x0, ob[4 * p + 3] = x1;
            }
            // A 3-stripe: x = pairs 1..3 first halves, y = their second halves.
            T s[2 * nstripe];
            for (int i = 0; i < nstripe; ++i) s[i] = pr[2 * (i + 1)], s[nstripe + i] = pr[2 * (i + 1) + 1];
            tdv2::refl2(v2::Thread{}, tdv2::house2(pr[0], pr[1]), tdv2::refs(s[0], s[1], s[2]), tdv2::refs(s[3], s[4], s[5]));
            for (int i = 0; i < 2 * nstripe; ++i) so[std::ptrdiff_t(b) * 2 * nstripe + i] = s[i];
        }
        const auto A = v2::mat(a + std::ptrdiff_t(b) * 2 * ld, m, 2, ld);
        tdv2::refl2(ex, tdv2::house2(pr[0], pr[1]), A.col(0), A.col(1));
    });
    const auto gv = this->dev.down(vo, hv.size());
    const auto ga = this->dev.down(a, ha.size());
    const auto gs = this->dev.down(so, hs.size());
    EXPECT_EQ(bitdiff_items(gv, 4 * npair), 0);
    EXPECT_EQ(bitdiff_items(ga, 2 * ld), 0);
    EXPECT_EQ(bitdiff_items(gs, 2 * nstripe), 0);
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
    for (int i = 0; i < nstripe; ++i) {
        const long double x = pairs[2 * (i + 1)], y = pairs[2 * (i + 1) + 1];
        const long double t = x * v0 + y * v1, s = std::fabs(x) + std::fabs(y);
        worst = std::max({worst, rel(gs[i], x - 2 * v0 * t, s), rel(gs[nstripe + i], y - 2 * v1 * t, s)});
    }
    EXPECT_VERIFY(T, Check::blas, 2, worst) << "Refs<T, 3> stripe";
    worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < m; ++r) {
            const long double x = ha[std::size_t(b) * 2 * ld + r], y = ha[std::size_t(b) * 2 * ld + ld + r];
            const long double t = x * v0 + y * v1, s = std::fabs(x) + std::fabs(y);
            worst = std::max({worst, rel(ga[std::size_t(b) * 2 * ld + r], x - 2 * v0 * t, s), rel(ga[std::size_t(b) * 2 * ld + ld + r], y - 2 * v1 * t, s)});
        }
    EXPECT_VERIFY(T, Check::blas, 2, worst) << "column pair";
    for (int b = 0; b < kBatch; ++b)
        for (int r = m; r < ld; ++r)
            for (int c = 0; c < 2; ++c) {
                const std::size_t k = std::size_t(b) * 2 * ld + std::size_t(c) * ld + r;
                EXPECT_TRUE(same_bits(ga[k], ha[k])) << "pad row " << r << " of column " << c << " written";
            }
}

static_assert(tdv2::Refs<double, 3>{}.size() == 3);
static_assert(std::is_same_v<v2::elem_t<tdv2::Refs<float, 2>>, float>);

}  // namespace
