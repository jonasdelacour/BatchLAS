// SPDX-License-Identifier: MIT
// Device LA v2 syevx extension (include/batchlas/blas/device/v2/ext/syevx.hh) against host
// references. Guards (docs/developer/agent-guide.md section 8): run-time widths that are not a
// multiple of the sub-group (40, a partial sub-group) and the ceiling (1024); non-natural ld and
// inc; complex data with nonzero imaginary parts; NaN-poisoned outputs that laset / lacpy must
// overwrite; batch 1024 with period-3 data where every item must be bit-identical to its
// representative; count_le ties (x == v counts), NaN and Inf queries; an Inf imaginary part that a
// cross-term scale_re would turn into NaN; laset with alpha != beta on a non-square block.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/ext/syevx.hh>
#include <src/queue.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
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

// kPeriod representatives copied to every item b from representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(static_cast<std::size_t>(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < kBatch; ++b)
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
    // A run-time work-group size: what RtGroup exists for.
    template <class F> void run(int batch, int wg, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * wg, std::size_t(wg)),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SyevxTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SyevxTest, Types);

template <class T> long double abs2_wide(T v) {
    if constexpr (is_cplx<T>) {
        const long double r = v.real(), i = v.imag();
        return r * r + i * i;
    } else {
        return static_cast<long double>(v) * static_cast<long double>(v);
    }
}
template <class R> double rel_r(R got, long double ref) {
    const long double d = std::abs(static_cast<long double>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(std::abs(ref), std::numeric_limits<R>::min()));
}

// ---- 2-D elementwise and re-indexed operands --------------------------------------------------

// Output block of one item, columns: [0, n) lacpy forward | [n, 2n) lacpy column-reversed |
// [2n, 2n+2) laset(Zero, Zero) | [2n+2, 2n+5) laset(alpha, beta) | [2n+5, 3n+5) copy then
// in-place column reversal by swap_mat (n odd: the middle column stays). Then a vector part.
template <class T> void run_elementwise(Dev& dev, int wg) {
    using R = real_of<T>;
    constexpr int m = 37, n = 9, lda = m + 3, ldo = m + 5, ncol = 3 * n + 5, nx = 23, inc = 2;
    const T alpha = val<T>(7), beta = val<T>(8);
    const auto ha = batched<T>(n * lda, 11), hx = batched<T>(nx * inc, 12);
    const T nan = T(std::numeric_limits<R>::quiet_NaN());
    std::vector<T> ho(std::size_t(kBatch) * ncol * ldo, nan), hv(std::size_t(kBatch) * 2 * nx, nan);
    const T* a = dev.up(ha);
    const T* x = dev.up(hx);
    T* o = dev.up(ho);
    T* v = dev.up(hv);
    dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
        const v2::RtGroup<> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = v2::mat(a + std::ptrdiff_t(b) * n * lda, m, n, lda);
        const auto O = v2::mat(o + std::ptrdiff_t(b) * ncol * ldo, m, ncol, ldo);
        v2::lacpy(ex, v2::cols_flipped(A, false), O.sub(0, 0, m, n));
        v2::lacpy(ex, v2::cols_flipped(A, true), O.sub(0, n, m, n));
        v2::laset(ex, v2::Zero{}, v2::Zero{}, O.sub(0, 2 * n, m, 2));
        v2::laset(ex, v2::Scalar{alpha}, v2::Scalar{beta}, O.sub(0, 2 * n + 2, m, 3));
        const auto S = O.sub(0, 2 * n + 5, m, n);
        v2::lacpy(ex, A, S);
        v2::group_sync(ex);  // the swap reads columns another item copied
        constexpr int half = n / 2;
        v2::swap_mat(ex, S.sub(0, 0, m, half), v2::cols_flipped(S.sub(0, n - half, m, half), true));
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * nx * inc, nx, inc);
        v2::copy(ex, v2::flipped(xv, false), v2::vec(v + std::ptrdiff_t(b) * 2 * nx, nx));
        v2::copy(ex, v2::flipped(xv, true), v2::vec(v + std::ptrdiff_t(b) * 2 * nx + nx, nx));
    });
    const auto go = dev.down(o, ho.size()), gv = dev.down(v, hv.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* ab = ha.data() + std::size_t(b) * n * lda;
        const T* ob = go.data() + std::size_t(b) * ncol * ldo;
        auto at = [&](int r, int c) { return ab[r + std::size_t(c) * lda]; };
        for (int c = 0; c < ncol; ++c)
            for (int r = 0; r < m; ++r) {
                T want;
                if (c < n) want = at(r, c);
                else if (c < 2 * n) want = at(r, 2 * n - 1 - c);
                else if (c < 2 * n + 2) want = T(0);
                else if (c < 2 * n + 5) want = (r == c - (2 * n + 2)) ? beta : alpha;
                else want = at(r, n - 1 - (c - (2 * n + 5)));
                bad += !same_bits(ob[r + std::size_t(c) * ldo], want);
            }
        // ld padding rows stay poisoned: no op wrote past the block's rows.
        for (int c = 0; c < ncol; ++c) bad += !std::isnan(std::abs(ob[m + std::size_t(c) * ldo]));
        const T* xb = hx.data() + std::size_t(b) * nx * inc;
        const T* vb = gv.data() + std::size_t(b) * 2 * nx;
        for (int i = 0; i < nx; ++i) bad += !same_bits(vb[i], xb[i * inc]) + !same_bits(vb[nx + i], xb[(nx - 1 - i) * inc]);
    }
    EXPECT_EQ(bad, 0) << "wg " << wg;
}

TYPED_TEST(DeviceBlasV2SyevxTest, ElementwiseOnRunTimeWidths) {
    for (int wg : {40, 96, 1024}) run_elementwise<TypeParam>(this->dev, wg);
}

// ---- reductions ------------------------------------------------------------------------------

enum Slot { kSumsq, kSumsqMat, kMin, kMax, kLazy, kCol0, kSlots = kCol0 + 7 };

// Every reduction of item b on executor family `which`: 0 = RtGroup (run-time wg), 1 =
// WorkGroup<96>, 2 = Thread on the leader. Column sums come from RtGroup teams (one sub-group
// per column) on family 0 and from Thread loops elsewhere.
template <class T> std::vector<real_of<T>> run_reductions(Dev& dev, int which, int wg, const std::vector<T>& hx,
                                                          const std::vector<T>& hm) {
    using R = real_of<T>;
    using Ar = v2::Arith<T>;
    constexpr int n = 133, inc = 3, m = 19, nc = 7, ld = 23;
    const R lam = R(0.375);
    std::vector<R> ho(std::size_t(kBatch) * kSlots, std::numeric_limits<R>::quiet_NaN());
    const T* x = dev.up(hx);
    const T* mm = dev.up(hm);
    R* o = dev.up(ho);
    auto body = [=](const auto& ex, int b, R* ob) {
        const auto xv = v2::vec(x + std::ptrdiff_t(b) * n * inc, n, inc);
        const auto M = v2::mat(mm + std::ptrdiff_t(b) * nc * ld, m, nc, ld);
        const R s0 = v2::sumsq(ex, xv), s1 = v2::sumsq_mat(ex, M);
        R lo = std::numeric_limits<R>::max(), hi = -std::numeric_limits<R>::max();
        v2::for_each_owned(ex, n, [&](int i) {
            lo = sycl::min(lo, Ar::re(xv(i)));
            hi = sycl::max(hi, Ar::re(xv(i)));
        });
        lo = v2::allreduce(ex, lo, sycl::minimum<R>());
        hi = v2::allreduce(ex, hi, sycl::maximum<R>());
        const auto r = v2::lazy(m, [&](int i) { return Ar::sub(M(i, 0), v2::scale_re(lam, M(i, 1))); });
        const R s4 = v2::sumsq(ex, r);
        if (ex.leader()) ob[kSumsq] = s0, ob[kSumsqMat] = s1, ob[kMin] = lo, ob[kMax] = hi, ob[kLazy] = s4;
    };
    if (which == 0) {
        dev.run(kBatch, wg, [=](sycl::nd_item<1> it) {
            const v2::RtGroup<> ex{it};
            const int b = static_cast<int>(it.get_group_linear_id());
            R* ob = o + std::ptrdiff_t(b) * kSlots;
            body(ex, b, ob);
            const auto M = v2::mat(mm + std::ptrdiff_t(b) * nc * ld, m, nc, ld);
            const auto tm = ex.sub();
            for (int c = ex.team(); c < nc; c += ex.nteams()) {
                const R s = v2::sumsq(tm, M.col(c));
                if (tm.leader()) ob[kCol0 + c] = s;
            }
        });
    } else if (which == 1) {
        dev.run(kBatch, 96, [=](sycl::nd_item<1> it) {
            const v2::WorkGroup<96> ex{it};
            const int b = static_cast<int>(it.get_group_linear_id());
            R* ob = o + std::ptrdiff_t(b) * kSlots;
            body(ex, b, ob);
            const auto M = v2::mat(mm + std::ptrdiff_t(b) * nc * ld, m, nc, ld);
            for (int c = 0; c < nc; ++c) {
                const R s = v2::sumsq<v2::Scope::Group>(ex, M.col(c));
                if (ex.leader()) ob[kCol0 + c] = s;
            }
        });
    } else {
        dev.run(kBatch, 32, [=](sycl::nd_item<1> it) {
            if (it.get_local_linear_id() != 0) return;
            const v2::Thread ex{};
            const int b = static_cast<int>(it.get_group_linear_id());
            R* ob = o + std::ptrdiff_t(b) * kSlots;
            body(ex, b, ob);
            const auto M = v2::mat(mm + std::ptrdiff_t(b) * nc * ld, m, nc, ld);
            for (int c = 0; c < nc; ++c) ob[kCol0 + c] = v2::sumsq(ex, M.col(c));
        });
    }
    return dev.down(o, ho.size());
}

TYPED_TEST(DeviceBlasV2SyevxTest, SumsqAllreduceAndTeamsPerColumn) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 133, inc = 3, m = 19, nc = 7, ld = 23;
    const long double lam = 0.375L;
    const auto hx = batched<T>(n * inc, 21), hm = batched<T>(nc * ld, 22);
    struct Run { int which, wg; };
    for (const Run run : {Run{0, 40}, Run{0, 96}, Run{0, 1024}, Run{1, 96}, Run{2, 32}}) {
        const auto got = run_reductions<T>(this->dev, run.which, run.wg, hx, hm);
        int bitdiff = 0;
        for (int b = 0; b < kBatch; ++b)
            for (int s = 0; s < kSlots; ++s)
                bitdiff += !same_bits(got[std::size_t(b) * kSlots + s], got[std::size_t(b % kPeriod) * kSlots + s]);
        EXPECT_EQ(bitdiff, 0) << "items differ from their representative (family " << run.which << ", wg " << run.wg << ")";
        for (int b = 0; b < kPeriod; ++b) {
            const T* xb = hx.data() + std::size_t(b) * n * inc;
            const T* mb = hm.data() + std::size_t(b) * nc * ld;
            const R* g = got.data() + std::size_t(b) * kSlots;
            long double sx = 0, smat = 0, slazy = 0;
            R lo = std::numeric_limits<R>::max(), hi = -std::numeric_limits<R>::max();
            for (int i = 0; i < n; ++i) {
                sx += abs2_wide(xb[i * inc]);
                lo = std::min(lo, R(std::real(xb[i * inc])));
                hi = std::max(hi, R(std::real(xb[i * inc])));
            }
            for (int c = 0; c < nc; ++c) {
                long double sc = 0;
                for (int r = 0; r < m; ++r) sc += abs2_wide(mb[r + c * ld]);
                smat += sc;
                EXPECT_VERIFY(R, Check::blas, m, rel_r(g[kCol0 + c], sc)) << "column " << c << " family " << run.which;
            }
            for (int r = 0; r < m; ++r) {
                using LT = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
                const LT d = LT(mb[r]) - lam * LT(mb[r + ld]);
                slazy += std::norm(std::complex<long double>(d));
            }
            EXPECT_VERIFY(R, Check::blas, n, rel_r(g[kSumsq], sx)) << "sumsq family " << run.which << " wg " << run.wg;
            EXPECT_VERIFY(R, Check::blas, m * nc, rel_r(g[kSumsqMat], smat)) << "sumsq_mat family " << run.which;
            EXPECT_VERIFY(R, Check::blas, m, rel_r(g[kLazy], slazy)) << "lazy residual family " << run.which;
            EXPECT_TRUE(same_bits(g[kMin], lo) && same_bits(g[kMax], hi)) << "allreduce min/max family " << run.which;
        }
    }
}

// ---- scalar helpers and searching ----------------------------------------------------------------

TYPED_TEST(DeviceBlasV2SyevxTest, CountLeAndScalarHelpers) {
    using T = TypeParam;
    using R = real_of<T>;
    const R inf = std::numeric_limits<R>::infinity(), qnan = std::numeric_limits<R>::quiet_NaN();
    const std::vector<R> hx = {R(-3), R(-1), R(-1), R(0), R(2), R(2), R(2), R(5)};
    const std::vector<R> hq = {R(-4), R(-1), R(2), R(5), R(1), qnan, inf, -inf};
    const std::vector<int> want = {0, 3, 7, 8, 4, 0, 8, 0};
    const R* x = this->dev.up(hx);
    const R* qv = this->dev.up(hq);
    int* cnt = this->dev.up(std::vector<int>(hq.size() + 1, -7));
    T* sc = this->dev.up(std::vector<T>(4, T(0)));
    R* ab = this->dev.up(std::vector<R>(1, R(0)));
    const int nq = static_cast<int>(hq.size()), nx = static_cast<int>(hx.size());
    T z, hinf;
    if constexpr (is_cplx<T>) z = T(R(3), R(4)), hinf = T(R(1), inf);
    else z = T(3), hinf = T(inf);
    this->dev.run(1, 32, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const auto xv = v2::vec(x, nx);
        for (int q = 0; q < nq; ++q) cnt[q] = v2::count_le(xv, qv[q]);
        cnt[nq] = v2::count_le(v2::vec(x, 0), R(0));  // empty
        sc[0] = v2::scale_re(R(2), hinf);
        sc[1] = v2::scale_re(R(-0.5), z);
        sc[2] = v2::sub_scaled(z, R(2), hinf);
        sc[3] = v2::sub_scaled(z, R(-0.5), z);
        ab[0] = v2::abs2(z);
    });
    const auto gc = this->dev.down(cnt, hq.size() + 1);
    for (int q = 0; q < nq; ++q) EXPECT_EQ(gc[q], want[q]) << "query " << q;
    EXPECT_EQ(gc[nq], 0) << "empty array";
    const auto gs = this->dev.down(sc, 4);
    const auto ga = this->dev.down(ab, 1);
    if constexpr (is_cplx<T>) {
        // No cross terms: (2 + 0i)(1 + inf i) by the full product has 0 * inf = NaN in Re.
        EXPECT_EQ(gs[0].real(), R(2));
        EXPECT_EQ(gs[0].imag(), inf);
        EXPECT_TRUE(same_bits(gs[1], T(R(-1.5), R(-2))));
        EXPECT_EQ(gs[2].real(), R(1));
        EXPECT_EQ(gs[2].imag(), -inf);
        EXPECT_TRUE(same_bits(gs[3], T(R(4.5), R(6))));
        EXPECT_EQ(ga[0], R(25));
    } else {
        EXPECT_EQ(gs[0], inf);
        EXPECT_TRUE(same_bits(gs[1], T(R(-1.5))));
        EXPECT_EQ(gs[2], -inf);
        EXPECT_TRUE(same_bits(gs[3], T(R(4.5))));
        EXPECT_EQ(ga[0], R(9));
    }
}

}  // namespace
