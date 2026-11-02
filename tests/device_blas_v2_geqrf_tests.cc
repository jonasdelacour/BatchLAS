// SPDX-License-Identifier: MIT
// Device LA v2 QR extension (include/batchlas/blas/device/v2/ext/geqrf.hh) against host references.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld and inc, complex data with
// nonzero imaginary parts, a complex alpha over a zero tail (a reflector) next to a real one (the
// identity), a tail scaled past sqrt(max) so an unscaled norm overflows, NaN poison in the
// implied unit head and above it, bit-exact butterfly orders against a host simulation, batch
// 1024 with period-4 data where every item must match its representative, a WG = 1024 launch,
// and every executor: DynWorkGroup, DynSubGroup, DynTeams, Teams, WorkGroup, SgpPartition.
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
#include <batchlas/blas/device/v2/ext/geqrf.hh>
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
template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;

constexpr int kBatch = 1024;
constexpr int kPeriod = 4;

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
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T nan_of() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives (edited by fix(rep, item pointer)) copied to every item b from b % kPeriod.
template <class T, class Fix> std::vector<T> batched(int per_item, std::uint64_t seed, Fix fix) {
    std::vector<T> h(static_cast<std::size_t>(kBatch) * per_item);
    for (int b = 0; b < kPeriod; ++b) {
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
        fix(b, h.data() + std::size_t(b) * per_item);
    }
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
// Each launch declares its WG so ptxas caps registers (complex<double> geqr2 at 1024 needs <= 64).
// NVPTX only, as BATCHLAS_LAUNCH_BOUNDS (whose min_work_groups_per_cu a lambda rejects): .maxntid,
// plus .reqntid where the configure probe says it is lowered (src/ops/gemm/kernels/register_128x128.hh).
#if !BATCHLAS_SYCL_IMPL_ACPP && defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
#if BATCHLAS_NVPTX_REQD_WG_OK
#define TEST_WG_BOUND(n) [[intel::max_work_group_size(1, 1, n), sycl::reqd_work_group_size(n)]]
#else
#define TEST_WG_BOUND(n) [[intel::max_work_group_size(1, 1, n)]]
#endif
#else
#define TEST_WG_BOUND(n)
#endif

template <class T> int period_bitdiff(const std::vector<T>& got, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i)
            d += !same_bits(got[std::size_t(b) * per_item + i], got[std::size_t(b % kPeriod) * per_item + i]);
    return d;
}

template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}
template <class T> LD<T> conj_h(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
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
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32)
                                                                                 TEST_WG_BOUND(WG) {
                 f(it, batchlas::util::get_raw_ptr(lm));
             });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GeqrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    }
    Dev dev;
};
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GeqrfTest, Types);

// Host zlarfg in long double: {beta, tau, v = x / (alpha - beta)}; tau = 0 is the identity.
template <class T> void host_larfg(LD<T> alpha, std::vector<LD<T>>& x, LD<T>& beta, LD<T>& tau) {
    long double sc = std::abs(alpha), xn2 = 0;  // scaled: long double may be double (aarch64)
    for (const auto& e : x) sc = std::max(sc, std::abs(e));
    for (const auto& e : x) xn2 += sc > 0 ? std::norm(e / sc) : 0.0L;
    const long double ai = is_cplx<T> ? std::imag(LD<T>(alpha)) : 0.0L, ar = std::real(LD<T>(alpha));
    if (xn2 == 0 && ai == 0) {
        beta = alpha;
        tau = LD<T>(0);
        return;
    }
    const long double nrm = sc * std::sqrt(std::norm(alpha / sc) + xn2);
    const long double b = ar >= 0 ? -nrm : nrm;
    beta = LD<T>(b);
    if constexpr (is_cplx<T>) tau = LD<T>((b - ar) / b, -ai / b);
    else tau = LD<T>((b - ar) / b);
    for (auto& e : x) e = e / (alpha - beta);
}

// ---- scalar arithmetic ---------------------------------------------------------------------

// mul_fma / div_smith / recip_smith; slot 3 divides by a value whose |b|^2 overflows (Smith's
// reason to exist): the naive (c - di) / (c^2 + d^2) returns 0 there.
TYPED_TEST(DeviceBlasV2GeqrfTest, ScalarArithmetic) {
    using T = TypeParam;
    using R = real_of<T>;
    const R big = std::is_same_v<R, float> ? R(1e30f) : R(1e300);
    auto h = batched<T>(4, 7, [&](int rep, T* p) {
        if (rep == 3) p[1] = p[1] * big;
    });
    T* in = this->dev.up(h);
    std::vector<T> ho(std::size_t(kBatch) * 3, nan_of<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<32>(kBatch / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const T* p = in + std::ptrdiff_t(b) * 4;
        o[b * 3 + 0] = v2::mul_fma(p[0], p[2]);
        o[b * 3 + 1] = v2::div_smith(p[0], p[1]);
        o[b * 3 + 2] = v2::recip_smith(p[1]);
    });
    const auto got = this->dev.down(o, ho.size());
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const LD<T> a(h[b * 4]), d(h[b * 4 + 1]), c(h[b * 4 + 2]);
        worst = std::max({worst, rel(got[b * 3], a * c, std::abs(a) * std::abs(c)), rel(got[b * 3 + 1], a / d, std::abs(a / d)),
                          rel(got[b * 3 + 2], LD<T>(1) / d, std::abs(LD<T>(1) / d))});
    }
    EXPECT_EQ(period_bitdiff(got, 3), 0);
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// ---- xLARFG on Cyclic executors -------------------------------------------------------------

// Per item: alpha, x (n at inc 3). Representatives: 0 random, 1 zero tail + real alpha (identity),
// 2 tail scaled past sqrt(max) (an unscaled sum of squares overflows), 3 zero tail + complex alpha
// (a reflector for complex types, the identity for real ones).
template <class T, int WG, class MakeEx> void larfg_case(Dev& dev, MakeEx make_ex, const char* what) {
    using R = real_of<T>;
    constexpr int n = 45, inc = 3, per = 1 + n * inc;
    const R big = std::is_same_v<R, float> ? R(1e30f) : R(1e300);
    auto h = batched<T>(per, 11, [&](int rep, T* p) {
        if (rep == 1 || rep == 3) {
            for (int i = 0; i < n; ++i) p[1 + i * inc] = T(0);
            if (rep == 1) p[0] = T(std::real(p[0]));
        }
        if (rep == 2)
            for (int i = 0; i < n; ++i) p[1 + i * inc] = p[1 + i * inc] * big;
    });
    T* a = dev.up(h);
    std::vector<T> ho(std::size_t(kBatch) * 2, nan_of<T>());
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto ex = make_ex(it);
        T* p = a + std::ptrdiff_t(b) * per;
        const auto x = v2::vec(p + 1, n, inc);
        const T alpha = p[0];
        const auto hh = v2::larfg_scalars<v2::Conv::Lapack>(ex, alpha, x);
        if (!hh.identity) v2::scal_or_div(ex, hh, x);
        if (ex.leader()) {
            o[b * 2] = hh.identity ? alpha : hh.beta;
            o[b * 2 + 1] = hh.tau;
        }
    });
    const auto got = dev.down(o, ho.size());
    const auto gx = dev.down(a, h.size());
    EXPECT_EQ(period_bitdiff(got, 2), 0) << what;
    EXPECT_EQ(period_bitdiff(gx, per), 0) << what;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h.data() + std::size_t(b) * per;
        std::vector<LD<T>> x(n);
        for (int i = 0; i < n; ++i) x[i] = LD<T>(p[1 + i * inc]);
        LD<T> beta, tau;
        host_larfg<T>(LD<T>(p[0]), x, beta, tau);
        const T gb = got[b * 2], gt = got[b * 2 + 1];
        if (tau == LD<T>(0)) {
            EXPECT_TRUE(same_bits(gt, T(0)) && same_bits(gb, p[0])) << what << ": identity expected, rep " << b;
            for (int i = 0; i < n; ++i) EXPECT_TRUE(same_bits(gx[std::size_t(b) * per + 1 + i * inc], p[1 + i * inc])) << what;
            continue;
        }
        if constexpr (is_cplx<T>) EXPECT_EQ(std::imag(gb), real_of<T>(0)) << what << ": beta must be real (zlarfg)";
        worst = std::max({worst, rel(gb, beta, std::abs(beta)), rel(gt, tau, 1.0L)});
        for (int i = 0; i < n; ++i) worst = std::max(worst, rel(gx[std::size_t(b) * per + 1 + i * inc], x[i], 1.0L + std::abs(x[i])));
    }
    EXPECT_VERIFY(T, Check::blas, n, worst) << what;
}

TYPED_TEST(DeviceBlasV2GeqrfTest, LarfgScalarsCyclic) {
    using T = TypeParam;
    larfg_case<T, 64>(this->dev, [](sycl::nd_item<1> it) { return v2::DynWorkGroup{it}; }, "DynWorkGroup@64");
    larfg_case<T, 1024>(this->dev, [](sycl::nd_item<1> it) { return v2::DynWorkGroup{it}; }, "DynWorkGroup@1024");
    larfg_case<T, 32>(this->dev, [](sycl::nd_item<1> it) { return v2::DynSubGroup{it}; }, "DynSubGroup");
    larfg_case<T, 64>(this->dev, [](sycl::nd_item<1> it) { return v2::WorkGroup<64>{it}; }, "WorkGroup<64>");
}

// ---- larf<Left, SgPerCol> -------------------------------------------------------------------

// C (m x nc, ld m + 5) := (I - tau v v^H) C with v = unit_head(v, head = 2): v(0..2) are NaN and
// must not be read; rows 0, 1 of C are outside the reflector and must keep their bits.
template <class T, int WG, class MakeTeams> void larf_case(Dev& dev, MakeTeams make_teams, const char* what) {
    constexpr int m = 41, nc = 11, ldc = m + 5, head = 2, per = ldc * nc + m + 1;
    auto h = batched<T>(per, 23, [&](int, T* p) {
        for (int i = 0; i <= head; ++i) p[ldc * nc + i] = nan_of<T>();
    });
    T* d = dev.up(h);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        T* p = d + std::ptrdiff_t(b) * per;
        const auto teams = make_teams(it);
        const v2::WideMat<T> c{p, m, nc, ldc};
        v2::larf<Side::Left, v2::SgPerCol>(teams, p[ldc * nc + m], v2::unit_head(v2::vec(p + ldc * nc, m), head), c);
    });
    const auto got = dev.down(d, h.size());
    EXPECT_EQ(period_bitdiff(got, per), 0) << what;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h.data() + std::size_t(b) * per;
        const T* g = got.data() + std::size_t(b) * per;
        const LD<T> tau(p[ldc * nc + m]);
        for (int c = 0; c < nc; ++c) {
            LD<T> w(p[c * ldc + head]);
            long double s = std::abs(w);
            for (int r = head + 1; r < m; ++r) {
                w += conj_h<T>(LD<T>(p[ldc * nc + r])) * LD<T>(p[c * ldc + r]);
                s += std::abs(LD<T>(p[ldc * nc + r])) * std::abs(LD<T>(p[c * ldc + r]));
            }
            for (int r = 0; r < head; ++r) EXPECT_TRUE(same_bits(g[c * ldc + r], p[c * ldc + r])) << what << ": row above head";
            for (int r = head; r < m; ++r) {
                const LD<T> vr = r == head ? LD<T>(1) : LD<T>(p[ldc * nc + r]);
                const LD<T> ref = LD<T>(p[c * ldc + r]) - tau * vr * w;
                worst = std::max(worst, rel(g[c * ldc + r], ref, std::abs(LD<T>(p[c * ldc + r])) + std::abs(tau * vr) * s));
            }
        }
    }
    EXPECT_VERIFY(T, Check::blas, m, worst) << what;
}

TYPED_TEST(DeviceBlasV2GeqrfTest, LarfLeftSgPerCol) {
    using T = TypeParam;
    larf_case<T, 128>(this->dev, [](sycl::nd_item<1> it) { return v2::Teams<128>{it}; }, "Teams<128>");
    larf_case<T, 96>(this->dev, [](sycl::nd_item<1> it) { return v2::DynTeams<false>{it}; }, "DynTeams@96");
    larf_case<T, 32>(this->dev, [](sycl::nd_item<1> it) { return v2::DynTeams<true>{it}; }, "DynTeams<Whole>@32");
}

// ---- butterfly orders, bit-exact ------------------------------------------------------------

// XorAsc on SgpPartition<8> (masks 1, 2, 4) and DynSubGroup's descending run-time butterfly
// (masks 16 .. 1), each against a host simulation in the same precision: an order change flips bits.
TYPED_TEST(DeviceBlasV2GeqrfTest, ButterflyOrdersAreBitExact) {
    using T = TypeParam;
    using R = real_of<T>;
    auto h = batched<T>(32, 31, [](int, T*) {});
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = h[i] * T(R(1 + (i % 32 % 7) * 1000));  // by lane: keeps period 4
    T* in = this->dev.up(h);
    std::vector<T> ho(std::size_t(kBatch) * 32 * 3, nan_of<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int l = static_cast<int>(it.get_local_linear_id());
        const T v = in[b * 32 + l];
        const v2::DynSubGroup sg{it};
        o[(b * 32 + l) * 3] = sg.team_reduce(v);
        if constexpr (!is_cplx<T>) {
            const v2::SgpPartition<8> part{it.get_sub_group()};
            o[(b * 32 + l) * 3 + 1] = v2::team_reduce_with<v2::alg::XorAsc>(part, v, sycl::plus<R>());
            o[(b * 32 + l) * 3 + 2] = v2::team_reduce_with<v2::alg::XorAsc>(part, v, v2::Fmax{});
        }
    });
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<T> desc(h.begin() + b * 32, h.begin() + b * 32 + 32), asc = desc, mx = desc;
        for (int off = 16; off > 0; off >>= 1) {
            std::vector<T> nx(32);
            for (int l = 0; l < 32; ++l) nx[l] = v2::Arith<T>::add(desc[l], desc[l ^ off]);
            desc = nx;
        }
        for (int mask = 1; mask < 8; mask <<= 1) {
            std::vector<T> na(32), nm(32);
            for (int l = 0; l < 32; ++l) {
                na[l] = v2::Arith<T>::add(asc[l], asc[l ^ mask]);
                if constexpr (!is_cplx<T>) nm[l] = std::fmax(mx[l], mx[l ^ mask]);
            }
            asc = na;
            mx = nm;
        }
        for (int l = 0; l < 32; ++l) {
            const T* g = got.data() + (std::size_t(b) * 32 + l) * 3;
            bad += !same_bits(g[0], desc[l]);
            if constexpr (!is_cplx<T>) bad += !same_bits(g[1], asc[l]) + !same_bits(g[2], mx[l]);
        }
    }
    EXPECT_EQ(bad, 0) << "a butterfly ran in a different order than its tag names";
    EXPECT_EQ(period_bitdiff(got, 32 * 3), 0);
}

// ---- geqr2 composed from the ops ------------------------------------------------------------

// The panel body of src/ops/geqrf/geqrf_cta_device.hh on global memory (m x n, ld m + 4).
template <class Ex, class T> void geqr2_v2(const Ex& ex, T* p, int m, int n, int ld, T* tau) {
    const auto teams = v2::teams_of(ex);
    const v2::WideMat<T> a{p, m, n, ld};
    for (int j = 0; j < std::min(m, n); ++j) {
        ex.barrier();
        const T alpha = a(j, j);
        const auto x = a.col(j).tail(j + 1);
        const auto h = v2::larfg_scalars<v2::Conv::Lapack>(ex, alpha, x);
        ex.barrier();
        if (!h.identity) {
            if (ex.leader()) a(j, j) = h.beta;
            v2::scal_or_div(ex, h, x);
        }
        if (ex.leader()) tau[j] = h.tau;
        ex.barrier();
        if (h.identity) continue;
        v2::larf<Side::Left, v2::SgPerCol>(teams, v2::Arith<T>::conj(h.tau), v2::unit_head(a.col(j).tail(j), 0),
                                           a.sub(j, j + 1, m - j, n - j - 1));
    }
}

// ||A - Q R|| / ||A|| with Q = H_0 .. H_{k-1} rebuilt on the host. Rep 1: column 0 has a zero
// tail and a real diagonal (identity reflector); rep 3: same with a complex diagonal.
template <class T, int WG, class MakeEx> void geqr2_case(Dev& dev, MakeEx make_ex, const char* what) {
    constexpr int m = 37, n = 13, ld = m + 4, k = n, per = ld * n + k;
    auto h = batched<T>(per, 41, [&](int rep, T* p) {
        if (rep == 1 || rep == 3) {
            for (int r = 1; r < m; ++r) p[r] = T(0);
            if (rep == 1) p[0] = T(std::real(p[0]));
        }
    });
    T* d = dev.up(h);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        T* p = d + std::ptrdiff_t(it.get_group_linear_id()) * per;
        geqr2_v2(make_ex(it), p, m, n, ld, p + ld * n);
    });
    const auto got = dev.down(d, h.size());
    EXPECT_EQ(period_bitdiff(got, per), 0) << what;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* a0 = h.data() + std::size_t(b) * per;
        const T* g = got.data() + std::size_t(b) * per;
        std::vector<LD<T>> X(std::size_t(m) * n, LD<T>(0));
        long double an = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r) {
                if (r <= c) X[c * m + r] = LD<T>(g[c * ld + r]);
                an += std::norm(LD<T>(a0[c * ld + r]));
            }
        for (int j = k - 1; j >= 0; --j) {
            const LD<T> tj(g[ld * n + j]);
            if constexpr (is_cplx<T>) EXPECT_EQ(std::imag(g[j * ld + j]), real_of<T>(0)) << what << ": R(j,j) must be real";
            for (int c = 0; c < n; ++c) {
                LD<T> w = X[c * m + j];
                for (int r = j + 1; r < m; ++r) w += conj_h<T>(LD<T>(g[j * ld + r])) * X[c * m + r];
                X[c * m + j] -= tj * w;
                for (int r = j + 1; r < m; ++r) X[c * m + r] -= tj * LD<T>(g[j * ld + r]) * w;
            }
        }
        long double res = 0;
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r) res += std::norm(X[c * m + r] - LD<T>(a0[c * ld + r]));
        worst = std::max(worst, static_cast<double>(std::sqrt(res / an)));
        if (b == 1) EXPECT_TRUE(same_bits(g[ld * n], T(0))) << what << ": real alpha over a zero tail is the identity";
        if (b == 3 && is_cplx<T>) EXPECT_FALSE(same_bits(g[ld * n], T(0))) << what << ": a complex alpha needs a reflector";
    }
    EXPECT_VERIFY(T, Check::factorization, m, worst) << what;
}

TYPED_TEST(DeviceBlasV2GeqrfTest, Geqr2ComposedFromOps) {
    using T = TypeParam;
    geqr2_case<T, 64>(this->dev, [](sycl::nd_item<1> it) { return v2::DynWorkGroup{it}; }, "DynWorkGroup@64");
    geqr2_case<T, 1024>(this->dev, [](sycl::nd_item<1> it) { return v2::DynWorkGroup{it}; }, "DynWorkGroup@1024");
    geqr2_case<T, 32>(this->dev, [](sycl::nd_item<1> it) { return v2::DynSubGroup{it}; }, "DynSubGroup");
}

// ---- RowFrag reflector apply ---------------------------------------------------------------

// One reflector (column j = 2, tau) applied to a row-fragment panel two ways: team partials in
// slots + ger (DynWorkGroup, 64 items, rows >= m dead), and the local-tile column-sum form on
// SgpPartition<8> (four 8 x 8 panels per sub-group, chunk C = 4). Host: (I - tau v v^H) A.
template <class T> void host_frag_apply(const T* a, int m, int n, int j, T tau, std::vector<LD<T>>& out) {
    out.assign(std::size_t(m) * n, LD<T>(0));
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < m; ++r) out[c * m + r] = LD<T>(a[c * m + r]);
    auto v = [&](int r) { return r < j ? LD<T>(0) : (r == j ? LD<T>(1) : LD<T>(a[j * m + r])); };
    for (int c = j + 1; c < n; ++c) {
        LD<T> w(0);
        for (int r = 0; r < m; ++r) w += conj_h<T>(v(r)) * out[c * m + r];
        for (int r = 0; r < m; ++r) out[c * m + r] -= LD<T>(tau) * v(r) * w;
    }
}

TYPED_TEST(DeviceBlasV2GeqrfTest, RowFragApply) {
    using T = TypeParam;
    using A = v2::Arith<T>;
    constexpr int N = 8, j = 2;
    // (a) slots: m = 50 rows of 64 items, n = 7 of N columns.
    {
        constexpr int WG = 64, m = 50, n = 7, per = m * n + 1;
        auto h = batched<T>(per, 51, [](int, T*) {});
        T* d = this->dev.up(h);
        this->dev.template run<WG>(kBatch, sizeof(T) * (WG / 32) * N, [=](sycl::nd_item<1> it, unsigned char* lm) {
            T* p = d + std::ptrdiff_t(it.get_group_linear_id()) * per;
            const v2::DynWorkGroup ex{it};
            T* sw = reinterpret_cast<T*>(lm);
            const int r = ex.lid();
            const bool live = r < m;
            T rA[N];
#pragma clang loop unroll(full)
            for (int c = 0; c < N; ++c) rA[c] = (live && c < n) ? p[c * m + r] : A::zero();
            const T v = (!live || r < j) ? A::zero() : (r == j ? A::one() : rA[j]);
            v2::gemv_rowfrag_partials<N>(ex, A::conj(v), rA, j + 1, n, sw, 0, N);
            ex.barrier();
            v2::ger_rowfrag_slots<N>(p[m * n], v, rA, j + 1, n, sw, 0, ex.nteams_dyn(), N);
#pragma clang loop unroll(full)
            for (int c = 0; c < N; ++c)
                if (live && c < n) p[c * m + r] = rA[c];
        });
        const auto got = this->dev.down(d, h.size());
        EXPECT_EQ(period_bitdiff(got, per), 0) << "slots";
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            std::vector<LD<T>> ref;
            host_frag_apply(h.data() + std::size_t(b) * per, m, n, j, h[std::size_t(b) * per + m * n], ref);
            for (int i = 0; i < m * n; ++i) worst = std::max(worst, rel(got[std::size_t(b) * per + i], ref[i], 1.0L + std::abs(ref[i]) * m));
        }
        EXPECT_VERIFY(T, Check::blas, m, worst) << "slots";
    }
    // (b) local tile on SgpPartition<8>: n = 7, columns chunked by C = 4, tile ld C + 1.
    {
        constexpr int C = 4, SLD = C + 1, n = 7, per = N * N + 1, PB = N * SLD + C;
        auto h = batched<T>(per, 61, [](int, T*) {});
        T* d = this->dev.up(h);
        this->dev.template run<32>(kBatch / 4, sizeof(T) * 4 * PB, [=](sycl::nd_item<1> it, unsigned char* lm) {
            const auto sg = it.get_sub_group();
            const v2::SgpPartition<N> ex{sg};
            const int lane = ex.lid();
            const int slot = static_cast<int>(sg.get_local_linear_id()) / N;
            T* p = d + (std::ptrdiff_t(it.get_group_linear_id()) * 4 + slot) * per;
            T* sP = reinterpret_cast<T*>(lm) + slot * PB;
            T* sy = sP + N * SLD;
            T rA[N];
#pragma clang loop unroll(full)
            for (int c = 0; c < N; ++c) rA[c] = c < n ? p[c * N + lane] : A::zero();
            const T vr = lane < j ? A::zero() : (lane == j ? A::one() : rA[j]);
#pragma clang loop unroll(full)
            for (int k0 = 0; k0 < N; k0 += C) {
                if (k0 + C <= j + 1 || k0 >= n) continue;
                v2::frag_scale_to_row<C>(A::conj(vr), rA, k0, j + 1, n, sP, lane * SLD);
                sycl::group_barrier(sg);
                v2::colsum_colperlane<C, N>(lane, A::conj(p[N * N]), sP, SLD, j, sy);
                sycl::group_barrier(sg);
                v2::frag_rank1<C>(vr, sy, rA, k0, j + 1, n);
            }
#pragma clang loop unroll(full)
            for (int c = 0; c < N; ++c)
                if (c < n) p[c * N + lane] = rA[c];
        });
        const auto got = this->dev.down(d, h.size());
        EXPECT_EQ(period_bitdiff(got, per), 0) << "tile";
        double worst = 0;
        for (int b = 0; b < kPeriod; ++b) {
            std::vector<LD<T>> ref;
            host_frag_apply(h.data() + std::size_t(b) * per, N, n, j, A::conj(h[std::size_t(b) * per + N * N]), ref);
            for (int i = 0; i < N * n; ++i) worst = std::max(worst, rel(got[std::size_t(b) * per + i], ref[i], 1.0L + std::abs(ref[i]) * N));
        }
        EXPECT_VERIFY(T, Check::blas, N, worst) << "tile";
    }
}

}  // namespace
