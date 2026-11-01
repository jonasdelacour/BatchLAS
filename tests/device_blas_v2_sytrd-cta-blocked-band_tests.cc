// SPDX-License-Identifier: MIT
// Device LA v2 sytrd extension (include/batchlas/blas/device/v2/ext/sytrd-cta-blocked-band.hh)
// against long-double host references. Guards (docs/developer/agent-guide.md section 8):
// non-natural ld (tile ld P+3, global ld n+2 / n+5, row strides), complex data with nonzero
// imaginary parts, the unread triangle and Im(diag) poisoned with NaN, beta == Zero over NaN, a
// trailing block whose ownership origin is not 0, larfg's tau == 0 thresholds (x == 0 with a real
// and with a complex alpha, n == 1), batch 1024 with period-3 data where every item must be
// bit-identical to its representative (SLM paths), and a WG = 1024 launch.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/blas/device/v2/ext/sytrd-cta-blocked-band.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/dla_v2_partition.hh>
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(item pointer, representative)) copied to every item.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(h.data() + std::size_t(b) * per_item, b);
    }
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed, int items = kBatch) {
    return batched<T>(per_item, seed, [](T*, int) {}, items);
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> int bitdiff(const std::vector<T>& g, int per_item, int items = kBatch) {
    int d = 0;
    for (int b = kPeriod; b < items; ++b)
        for (int i = 0; i < per_item; ++i) d += !same_bits(g[std::size_t(b) * per_item + i], g[std::size_t(b % kPeriod) * per_item + i]);
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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2SytrdTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2SytrdTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> LD<T> cj(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> long double nanabs(LD<T> v) {
    const long double a = std::abs(v);
    return std::isnan(a) ? std::numeric_limits<long double>::infinity() : a;
}

// The Hermitian matrix whose UL triangle is stored in h (column-major, ld), real diagonal.
template <class T> std::vector<LD<T>> herm_from(const T* h, int n, int ld, Uplo ul) {
    std::vector<LD<T>> a(std::size_t(n) * n);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            const bool stored = ul == Uplo::Lower ? r >= c : r <= c;
            LD<T> e = stored ? wide(h[r + std::size_t(c) * ld]) : cj<T>(wide(h[c + std::size_t(r) * ld]));
            if (r == c) e = LD<T>(std::real(e));
            a[r + std::size_t(c) * n] = e;
        }
    return a;
}
// Poison everything the op must not read: the other triangle, Im(diag), rows past n.
template <class T> void poison_herm(T* h, int n, int ld, Uplo ul) {
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < ld; ++r) {
            const bool stored = r < n && (ul == Uplo::Lower ? r >= c : r <= c);
            if (!stored) h[r + std::size_t(c) * ld] = nan_v<T>();
            if (r == c) {
                if constexpr (is_cplx<T>) h[r + std::size_t(c) * ld] = T(std::real(h[r + std::size_t(c) * ld]), std::numeric_limits<real_of<T>>::quiet_NaN());
            }
        }
}

// ---- sytd2 on a resident tile ---------------------------------------------------------------

// ||A - Q T Q^H||_F / ||A||_F from LAPACK's sytd2 packing in o (n x n, ld ldo), d, e, tau.
template <class T>
double sytd2_residual(const std::vector<LD<T>>& a, const T* o, int ldo, const T* d, const T* e, const T* tau, int n, Uplo ul) {
    using L = LD<T>;
    std::vector<L> q(std::size_t(n) * n, L(0)), qv(n), v(n);
    for (int i = 0; i < n; ++i) q[i + std::size_t(i) * n] = L(1);
    auto apply = [&](int i) {  // Q := Q (I - tau_i v_i v_i^H)
        for (int r = 0; r < n; ++r) v[r] = L(0);
        if (ul == Uplo::Lower) {
            v[i + 1] = L(1);
            for (int r = i + 2; r < n; ++r) v[r] = wide(o[r + std::size_t(i) * ldo]);
        } else {
            v[i] = L(1);
            for (int r = 0; r < i; ++r) v[r] = wide(o[r + std::size_t(i + 1) * ldo]);
        }
        for (int r = 0; r < n; ++r) {
            qv[r] = L(0);
            for (int c = 0; c < n; ++c) qv[r] += q[r + std::size_t(c) * n] * v[c];
        }
        const L t = wide(tau[i]);
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) q[r + std::size_t(c) * n] -= t * qv[r] * cj<T>(v[c]);
    };
    if (ul == Uplo::Lower) for (int i = 0; i <= n - 2; ++i) apply(i);
    else for (int i = n - 2; i >= 0; --i) apply(i);
    std::vector<L> t(std::size_t(n) * n, L(0));
    for (int i = 0; i < n; ++i) t[i + std::size_t(i) * n] = wide(d[i]);
    for (int i = 0; i + 1 < n; ++i) {
        const L ei = wide(e[i]);
        if (ul == Uplo::Lower) t[i + 1 + std::size_t(i) * n] = ei, t[i + std::size_t(i + 1) * n] = cj<T>(ei);
        else t[i + std::size_t(i + 1) * n] = ei, t[i + 1 + std::size_t(i) * n] = cj<T>(ei);
    }
    long double num = 0, den = 0;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            L s = 0;
            for (int k = 0; k < n; ++k)
                for (int l = 0; l < n; ++l) s += q[r + std::size_t(k) * n] * t[k + std::size_t(l) * n] * cj<T>(q[c + std::size_t(l) * n]);
            const long double x = nanabs<T>(a[r + std::size_t(c) * n] - s);
            num += x * x;
            den += std::norm(a[r + std::size_t(c) * n]);
        }
    return static_cast<double>(std::sqrt(num / std::max<long double>(den, 1e-300L)));
}

// Executor factories: the src-side masked partition, the lockstep sub-group partition and a whole
// work-group (Fit) all run the same sytd2.
enum class ExKind { Sgp, Lockstep, WorkGroup };

template <class T, Uplo UL, int P, int WG, ExKind K> void sytd2_case(Dev& dev, int n) {
    constexpr int LDA = P + 3;
    constexpr int per_wg = K == ExKind::WorkGroup ? 1 : WG / P;
    constexpr int slot_elems = LDA * P + 2 * P;
    const int lda = n + 2, ldo = n + 1, per_a = lda * n;
    auto ha = batched<T>(per_a, 7 + 31 * int(UL) + P, [&](T* h, int) { poison_herm(h, n, lda, UL); });
    std::vector<T> ho(std::size_t(kBatch) * ldo * n, T(0)), hd(std::size_t(kBatch) * n, T(0)), he(std::size_t(kBatch) * n, T(0)),
        ht(std::size_t(kBatch) * n, T(0));
    T* a = dev.up(ha);
    T* o = dev.up(ho);
    T* d = dev.up(hd);
    T* e = dev.up(he);
    T* tu = dev.up(ht);
    dev.template run<WG>(kBatch / per_wg, sizeof(T) * per_wg * slot_elems, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const auto sg = it.get_sub_group();
        auto body = [&](const auto& ex, int slot) {
            const int b = static_cast<int>(it.get_group_linear_id()) * per_wg + slot;
            T* l = reinterpret_cast<T*>(lm) + slot * slot_elems;
            const v2::Mat<T, v2::Ext<P>, v2::Ext<P>, v2::Ext<LDA>, v2::Mem::Local> tile{l, {}, {}, {}};
            const auto v = v2::vec<v2::Mem::Local>(l + LDA * P, n), w = v2::vec<v2::Mem::Local>(l + LDA * P + P, n);
            T* eb = e + std::ptrdiff_t(b) * n;
            T* tb = tu + std::ptrdiff_t(b) * n;
            v2::lacpy_herm<UL>(v2::fit(ex), v2::mat(a + std::ptrdiff_t(b) * per_a, n, n, lda), tile);
            v2::team_barrier(ex);
            const T tl = v2::sytd2<UL>(ex, tile, v, w, n, [&](int i, const T& beta, const T& t) {
                if constexpr (UL == Uplo::Lower) {
                    eb[i] = beta;
                    tb[i] = t;
                }
            });
            v2::team_barrier(ex);
            const int lane = ex.lid();
            if (lane < n) d[std::ptrdiff_t(b) * n + lane] = tile(lane, lane);
            if constexpr (UL == Uplo::Upper) {
                if (lane < n - 1) {
                    eb[lane] = tile(lane, lane + 1);
                    tb[lane] = tl;
                }
            }
            v2::lacpy(v2::fit(ex), tile.sub(0, 0, n, n), v2::mat(o + std::ptrdiff_t(b) * ldo * n, n, n, ldo));
        };
        if constexpr (K == ExKind::Sgp) {
            const auto part = make_partition<P>(sg);
            body(v2::adopt_partition(part),
                 static_cast<int>(sg.get_group_linear_id()) * (32 / P) + static_cast<int>(part.get_group_linear_id()));
        } else if constexpr (K == ExKind::Lockstep) {
            const v2::Partition<P> ex{sg};
            body(ex, static_cast<int>(sg.get_group_linear_id()) * (32 / P) + static_cast<int>(sg.get_local_linear_id()) / P);
        } else {
            body(v2::WorkGroup<WG>{it}, 0);
        }
    });
    const auto go = dev.down(o, ho.size()), gd = dev.down(d, hd.size()), ge = dev.down(e, he.size()), gt = dev.down(tu, ht.size());
    EXPECT_EQ(bitdiff(go, ldo * n) + bitdiff(gd, n) + bitdiff(ge, n) + bitdiff(gt, n), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const auto aref = herm_from(ha.data() + std::size_t(b) * per_a, n, lda, UL);
        const double r = sytd2_residual<T>(aref, go.data() + std::size_t(b) * ldo * n, ldo, gd.data() + std::size_t(b) * n,
                                           ge.data() + std::size_t(b) * n, gt.data() + std::size_t(b) * n, n, UL);
        worst = std::isnan(r) ? std::numeric_limits<double>::infinity() : std::max(worst, r);
    }
    EXPECT_VERIFY(T, Check::factorization, std::max(n, 1), worst)
        << (UL == Uplo::Lower ? "lower" : "upper") << " P=" << P << " n=" << n << " kind=" << int(K);
}

TYPED_TEST(DeviceBlasV2SytrdTest, Sytd2OnResidentTile) {
    using T = TypeParam;
    sytd2_case<T, Uplo::Upper, 16, 64, ExKind::Sgp>(this->dev, 13);
    sytd2_case<T, Uplo::Lower, 16, 64, ExKind::Sgp>(this->dev, 13);
    sytd2_case<T, Uplo::Upper, 32, 64, ExKind::Sgp>(this->dev, 32);  // n at the partition ceiling
    sytd2_case<T, Uplo::Lower, 32, 64, ExKind::Sgp>(this->dev, 32);
    sytd2_case<T, Uplo::Upper, 8, 32, ExKind::Sgp>(this->dev, 2);
    sytd2_case<T, Uplo::Lower, 8, 32, ExKind::Sgp>(this->dev, 1);  // no step: d only
    sytd2_case<T, Uplo::Upper, 16, 32, ExKind::Lockstep>(this->dev, 11);
    sytd2_case<T, Uplo::Lower, 16, 32, ExKind::Lockstep>(this->dev, 11);
    sytd2_case<T, Uplo::Upper, 48, 64, ExKind::WorkGroup>(this->dev, 40);  // a 48-row tile on 64 items
    sytd2_case<T, Uplo::Lower, 48, 64, ExKind::WorkGroup>(this->dev, 40);
}

// ---- larfg on a RowFrag vector ----------------------------------------------------------------

// Representative 0: random; 1: x = 0 with a real alpha (tau = 0, beta = alpha bit for bit);
// 2: x = 0 with a complex alpha (a reflector for n > 1, tau = 0 at n = 1).
template <class T> void larfg_case(Dev& dev, int n, int ai) {
    constexpr int P = 8, WG = 32;
    auto hx = batched<T>(P, 300 + 17 * n + ai, [&](T* h, int rep) {
        if (rep == 0) return;
        for (int i = 0; i < P; ++i)
            if (i != ai) h[i] = T(0);
        if (rep == 1) h[ai] = T(std::real(h[ai]));
    });
    std::vector<T> hy(std::size_t(kBatch) * P, nan_v<T>()), ht(kBatch, nan_v<T>());
    T* x = dev.up(hx);
    T* y = dev.up(hy);
    T* t = dev.up(ht);
    dev.template run<WG>(kBatch / (WG / P), 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const auto ex = v2::SgpPartition<P>{sg};
        const int b = static_cast<int>(it.get_group_linear_id()) * (WG / P) + static_cast<int>(sg.get_local_linear_id()) / P;
        const int lane = ex.lid();
        const T* xb = x + std::ptrdiff_t(b) * P;
        const bool act = lane < n && lane != ai;
        T alpha = lane == ai ? xb[lane] : T(0);
        T xv = act ? xb[lane] : T(0);
        const T tau = v2::larfg_rowfrag(ex, n, ai, alpha, xv, act);
        if (lane == ai) y[std::ptrdiff_t(b) * P + lane] = alpha;
        else if (act) y[std::ptrdiff_t(b) * P + lane] = xv;
        if (lane == 0) t[b] = tau;
    });
    const auto gy = dev.down(y, hy.size()), gt = dev.down(t, ht.size());
    EXPECT_EQ(bitdiff(gt, 1) + bitdiff(gy, P), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        using L = LD<T>;
        const T* xin = hx.data() + std::size_t(b) * P;
        const T* yout = gy.data() + std::size_t(b) * P;
        const L tau = wide(gt[b]);
        if (n <= 1 || b == 1) {  // identity: tau = 0 and alpha unchanged, exactly
            EXPECT_TRUE(gt[b] == T(0)) << "n=" << n << " rep " << b;
            EXPECT_TRUE(same_bits(yout[ai], xin[ai])) << "n=" << n << " rep " << b;
            continue;
        }
        // H^H y = (beta at ai, 0 elsewhere), H = I - tau v v^H, v(ai) = 1.
        std::vector<L> v(n), z(n);
        long double ny = 0;
        for (int i = 0; i < n; ++i) v[i] = i == ai ? L(1) : wide(yout[i]), ny += std::norm(wide(xin[i]));
        L vy = 0;
        for (int i = 0; i < n; ++i) vy += cj<T>(v[i]) * wide(xin[i]);
        for (int i = 0; i < n; ++i) {
            z[i] = wide(xin[i]) - cj<T>(tau) * v[i] * vy;
            const L want = i == ai ? wide(yout[ai]) : L(0);
            worst = std::max(worst, static_cast<double>(nanabs<T>(z[i] - want) / std::sqrt(ny)));
        }
        if (b == 2 && is_cplx<T>) EXPECT_FALSE(gt[b] == T(0)) << "a complex alpha with x = 0 still reflects (n > 1)";
    }
    EXPECT_VERIFY(T, Check::factorization, std::max(n, 1), worst) << "larfg n=" << n << " alpha item " << ai;
}

TYPED_TEST(DeviceBlasV2SytrdTest, LarfgRowFrag) {
    using T = TypeParam;
    for (int n : {1, 2, 7, 8}) {
        larfg_case<T>(this->dev, n, 0);
        larfg_case<T>(this->dev, n, n - 1);
    }
}

// ---- Fit ops on a trailing block ---------------------------------------------------------------

// On Fit<WorkGroup<64>> with origin o (item lid owns row lid - o of the trailing block):
// lacpy_herm<UL> into a 32 x 32 tile (ld 35; items 32..63 own nothing), then on B = tile(o:N, o:N), m = N - o:
//   y := alpha B x (y NaN-poisoned, beta = Zero);  s := x^H y;  y += -s x (axpy through Fit);
//   C := her2_full(alpha, x, y, B) on a copy;  her2_mirror<UL>(NegOne, x, y, B) in place.
// Every row outside the block must keep its bits.
template <class T, Uplo UL> void fit_case(Dev& dev) {
    using L = LD<T>;
    constexpr int WG = 64, NT = 32, LDT = NT + 3, N = 24, o = 5, m = N - o, lda = N + 5;
    const T alpha = val<T>(901);
    auto ha = batched<T>(lda * N, 500 + int(UL), [&](T* h, int) { poison_herm(h, N, lda, UL); });
    auto hx = batched<T>(m, 600);
    const int per_out = 2 * NT * NT + 2 * m + 1;
    std::vector<T> ho(std::size_t(kBatch) * per_out, T(0));
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* out = dev.up(ho);
    dev.template run<WG>(kBatch, sizeof(T) * (2 * LDT * NT + 2 * m), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroup<WG> wg{it};
        const auto ex = v2::fit(wg);
        const auto fo = ex.at(o);
        T* l = reinterpret_cast<T*>(lm);
        const v2::Mat<T, v2::Ext<NT>, v2::Ext<NT>, v2::Ext<LDT>, v2::Mem::Local> t1{l, {}, {}, {}}, t2{l + LDT * NT, {}, {}, {}};
        const auto xl = v2::vec<v2::Mem::Local>(l + 2 * LDT * NT, m), yl = v2::vec<v2::Mem::Local>(l + 2 * LDT * NT + m, m);
        T* ob = out + std::ptrdiff_t(b) * per_out;
        v2::lacpy_herm<UL>(ex, v2::mat(a + std::ptrdiff_t(b) * lda * N, N, N, lda), t1);
        v2::copy(fo, v2::vec(x + std::ptrdiff_t(b) * m, m), xl);
        v2::map(fo, yl, [](T) { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }, yl);
        wg.barrier();
        v2::lacpy(ex, t1, t2);
        const auto B1 = t1.sub(o, o, m, m), B2 = t2.sub(o, o, m, m);
        v2::gemv<Transpose::NoTrans>(fo, alpha, B1, v2::assume_published(xl), v2::Zero{}, yl);  // reads t1 rows of others
        const T s = v2::dotc(fo, xl, yl);
        v2::axpy(fo, v2::Arith<T>::neg(s), xl, yl);
        wg.barrier();
        if (ex.lid() == 0) ob[2 * NT * NT + 2 * m] = s;
        v2::copy(fo, yl, v2::vec(ob + 2 * NT * NT + m, m));
        v2::copy(fo, xl, v2::vec(ob + 2 * NT * NT, m));
        v2::her2_full(fo, alpha, v2::assume_published(xl), v2::assume_published(yl), B2);
        v2::her2_mirror<UL>(fo, v2::NegOne{}, v2::assume_published(xl), v2::assume_published(yl), B1);
        wg.barrier();
        v2::lacpy(ex, t1, v2::mat(ob, NT, NT, NT));
        v2::lacpy(ex, t2, v2::mat(ob + NT * NT, NT, NT, NT));
    });
    const auto g = dev.down(out, ho.size());
    EXPECT_EQ(bitdiff(g, per_out), 0) << "items differ from their representative";
    double worst = 0;
    long double scale = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* gb = g.data() + std::size_t(b) * per_out;
        const auto A = herm_from(ha.data() + std::size_t(b) * lda * N, N, lda, UL);
        auto at = [&](int r, int c) { return r < N && c < N ? A[r + std::size_t(c) * N] : L(0); };
        std::vector<L> xv(m), yv(m, L(0));
        for (int i = 0; i < m; ++i) xv[i] = wide(hx[std::size_t(b) * m + i]);
        long double gscale = 0;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) yv[r] += wide(alpha) * at(o + r, o + c) * xv[c], gscale += std::abs(at(o + r, o + c) * xv[c]);
        }
        gscale *= std::abs(wide(alpha));
        L s = 0;
        for (int i = 0; i < m; ++i) s += cj<T>(xv[i]) * yv[i];
        for (int i = 0; i < m; ++i) yv[i] -= s * xv[i];
        scale = std::max<long double>(gscale * (1 + std::abs(s)), 1e-30L);
        worst = std::max(worst, double(nanabs<T>(wide(gb[2 * NT * NT + 2 * m]) - s) / scale));
        for (int i = 0; i < m; ++i) worst = std::max(worst, double(nanabs<T>(wide(gb[2 * NT * NT + m + i]) - yv[i]) / scale));
        // Use the device's y in the rank-2 references: they then test only the update itself.
        std::vector<L> yd(m);
        for (int i = 0; i < m; ++i) yd[i] = wide(gb[2 * NT * NT + m + i]);
        const L al = wide(alpha);
        for (int c = 0; c < NT; ++c)
            for (int r = 0; r < NT; ++r) {
                const bool in = r >= o && r < N && c >= o && c < N;
                const L base = at(r, c);
                L full = base, mir = base;
                if (in) {
                    const int i = r - o, j = c - o;
                    full = base + al * xv[i] * cj<T>(yd[j]) + cj<T>(al) * yd[i] * cj<T>(xv[j]);
                    mir = base - (xv[i] * cj<T>(yd[j]) + yd[i] * cj<T>(xv[j]));
                    if (r == c) mir = L(std::real(mir));
                }
                const long double sc = 1 + std::abs(base) + 2 * std::abs(al) * std::abs(xv[std::min(m - 1, std::max(0, r - o))]) * std::abs(yd[std::min(m - 1, std::max(0, c - o))]);
                worst = std::max(worst, double(nanabs<T>(wide(gb[r + std::size_t(c) * NT]) - mir) / sc));
                worst = std::max(worst, double(nanabs<T>(wide(gb[NT * NT + r + std::size_t(c) * NT]) - full) / sc));
                if (!in && !same_bits(gb[r + std::size_t(c) * NT], T(base))) worst = std::numeric_limits<double>::infinity();
            }
    }
    EXPECT_VERIFY(T, Check::blas, m, worst) << (UL == Uplo::Lower ? "lower" : "upper");
}

TYPED_TEST(DeviceBlasV2SytrdTest, FitOpsOnTrailingBlock) {
    fit_case<TypeParam, Uplo::Lower>(this->dev);
    fit_case<TypeParam, Uplo::Upper>(this->dev);
}

// ---- dot2c: the her2k entry ------------------------------------------------------------------

// A(r, c) -= dot2c(V.row(r), W.row(c), W.row(r), V.row(c)) for the lower triangle of an m x m A,
// V, W m x k with ld m + 5; on Thread (one item per entry), WorkGroup<64> and WorkGroup<1024>.
template <class T, int WG, bool PerEntryThread> void dot2c_case(Dev& dev, int m, int k) {
    const int ld = m + 5, lda = m + 2, per_vw = ld * k, per_a = lda * m;
    auto hv = batched<T>(per_vw, 700 + k), hw = batched<T>(per_vw, 800 + k), ha = batched<T>(per_a, 900 + k);
    T* v = dev.up(hv);
    T* w = dev.up(hw);
    T* a = dev.up(ha);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto V = v2::mat(static_cast<const T*>(v) + std::ptrdiff_t(b) * per_vw, m, k, ld);
        const auto W = v2::mat(static_cast<const T*>(w) + std::ptrdiff_t(b) * per_vw, m, k, ld);
        const auto A = v2::mat(a + std::ptrdiff_t(b) * per_a, m, m, lda);
        if constexpr (PerEntryThread) {
            for (int e = static_cast<int>(it.get_local_linear_id()); e < m * m; e += WG) {
                const int r = e % m, c = e / m;
                if (r < c) continue;
                const T acc = v2::dot2c(v2::Thread{}, V.row(r), W.row(c), W.row(r), V.row(c));
                A(r, c) = v2::Arith<T>::sub(A(r, c), acc);
            }
        } else {
            const v2::WorkGroup<WG> ex{it};
            for (int c = 0; c < m; ++c)
                for (int r = c; r < m; ++r) {
                    const T acc = v2::dot2c(ex, V.row(r), W.row(c), W.row(r), V.row(c));
                    if (ex.leader()) A(r, c) = v2::Arith<T>::sub(A(r, c), acc);
                }
        }
    });
    const auto g = dev.down(a, ha.size());
    EXPECT_EQ(bitdiff(g, per_a), 0) << "items differ from their representative";
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        auto e = [&](const std::vector<T>& h, int r, int c) { return wide(h[std::size_t(b) * per_vw + r + std::size_t(c) * ld]); };
        for (int c = 0; c < m; ++c)
            for (int r = 0; r < m; ++r) {
                const LD<T> base = wide(ha[std::size_t(b) * per_a + r + std::size_t(c) * lda]);
                LD<T> want = base;
                long double sc = std::abs(base);
                if (r >= c)
                    for (int q = 0; q < k; ++q) {
                        const LD<T> t = e(hv, r, q) * cj<T>(e(hw, c, q)) + e(hw, r, q) * cj<T>(e(hv, c, q));
                        want -= t;
                        sc += std::abs(t);
                    }
                worst = std::max(worst, double(nanabs<T>(wide(g[std::size_t(b) * per_a + r + std::size_t(c) * lda]) - want) / sc));
            }
    }
    EXPECT_VERIFY(T, Check::blas, k, worst) << "dot2c WG=" << WG << (PerEntryThread ? " Thread" : " WorkGroup");
}

TYPED_TEST(DeviceBlasV2SytrdTest, Dot2cHer2kEntry) {
    using T = TypeParam;
    dot2c_case<T, 64, true>(this->dev, 9, 13);
    dot2c_case<T, 64, false>(this->dev, 5, 70);    // k > WG: two owned elements per item
    dot2c_case<T, 1024, false>(this->dev, 3, 1100);  // the launch ceiling
}

}  // namespace
