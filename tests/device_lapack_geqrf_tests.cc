// SPDX-License-Identifier: MIT
// The geqr2 leaves' helpers (include/batchlas/device/lapack/geqrf.hh detail, plus the core
// abs_inf / op::FMax / butterfly forms they call) against host references; the other core ops the
// leaves use (RtGroup, mat_wide, for_each_owned, FmaArith, larfg_scalars) are covered by
// device_core_ops_tests and device_core_*. Guards (docs/developer/agent-guide.md section 8): non-natural ld,
// complex data with nonzero imaginary parts, finite poison (1e18) where v's unit head and the rows
// above it sit, values past sqrt(max), NaN dropped by the max butterflies, bit-exact butterfly
// orders against a host simulation, batch 1024 with period-4 data where every item must match its
// representative, a WG = 1024 launch, and the executors the leaves run on (RtGroup<Group>,
// RtGroup<Sub>, SgpPartition).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/geqrf.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;
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
// Finite poison: a read of it moves the result far past any tolerance without a NaN.
template <class T> T poison_of() {
    if constexpr (is_cplx<T>) return T(real_of<T>(1e18), real_of<T>(-1e18));
    else return T(1e18);
}

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
// Each launch declares its WG so ptxas caps registers at the launch size (test-only spelling of
// BATCHLAS_LAUNCH_BOUNDS, whose min_work_groups_per_cu a lambda rejects): .maxntid, plus .reqntid
// where the configure probe says it is lowered (src/ops/gemm/kernels/register_128x128.hh).
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

// No queue when no SYCL device is visible (the fixture skips).
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

template <class T> class DeviceLapackGeqrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceLapackGeqrfTest, Types);

// ---- Arith<sycl_device::Cx>: the predicates larfg_scalars reads on the kernels' scalar ----------
template <class R> void cx_predicates() {
    using Cx = sycl_device::Cx<R>;
    using A = device::Arith<Cx>;
    const R inf = std::numeric_limits<R>::infinity(), nan = std::numeric_limits<R>::quiet_NaN();
    const R den = std::numeric_limits<R>::denorm_min();
    const Cx cases[] = {{R(0), R(0)}, {R(-0.0), R(0)}, {R(0), R(-0.0)}, {den, R(0)}, {R(0), den}, {inf, R(0)},
                        {R(0), -inf}, {nan, R(1)}, {R(1), nan}, {R(2), R(-3)}};
    for (const Cx& c : cases) {
        EXPECT_EQ(A::is_zero(c), sycl_device::dev_is_zero(c)) << c.re << " " << c.im;
        EXPECT_EQ(A::isfinite(c), sycl_device::dev_isfinite(c)) << c.re << " " << c.im;
    }
}
TEST(DeviceLapackGeqrfCx, ArithPredicatesMatchDeviceScalar) {
    cx_predicates<float>();
    cx_predicates<double>();
}

// ---- xLARFG's two-pass scalars ----------------------------------------------------------------
// abs_inf = max(|re|, |im|) and abs2_scaled = |a / s|^2 on values past sqrt(max) (|a|^2 would
// overflow), and op::FMax dropping a NaN on either side.
TYPED_TEST(DeviceLapackGeqrfTest, NormScalars) {
    using T = TypeParam;
    using R = real_of<T>;
    const R big = std::is_same_v<R, float> ? R(1e30f) : R(1e300);
    auto h = batched<T>(2, 7, [&](int rep, T* p) {
        if (rep >= 2) p[0] = p[0] * big;
    });
    T* in = this->dev.up(h);
    std::vector<R> ho(std::size_t(kBatch) * 4, std::numeric_limits<R>::quiet_NaN());
    R* o = this->dev.up(ho);
    this->dev.template run<32>(kBatch / 32, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_global_linear_id());
        const T a = in[std::ptrdiff_t(b) * 2];
        const R s = device::abs_inf(a);
        o[b * 4 + 0] = s;
        o[b * 4 + 1] = device::detail::abs2_scaled(a, s);
        o[b * 4 + 2] = device::op::FMax{}(std::numeric_limits<R>::quiet_NaN(), s);
        o[b * 4 + 3] = device::op::FMax{}(s, std::numeric_limits<R>::quiet_NaN());
    });
    const auto got = this->dev.down(o, ho.size());
    EXPECT_EQ(period_bitdiff(got, 4), 0);
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const LD<T> a(h[std::size_t(b) * 2]);
        long double s = std::abs(std::real(a));
        if constexpr (is_cplx<T>) s = std::max(s, std::abs(std::imag(a)));
        const long double a2 = std::norm(a / s);
        worst = std::max({worst, rel<R>(got[b * 4], s, s), rel<R>(got[b * 4 + 1], a2, 1.0L)});
        EXPECT_TRUE(same_bits(got[b * 4 + 2], got[b * 4]) && same_bits(got[b * 4 + 3], got[b * 4])) << "Fmax kept a NaN, rep " << b;
    }
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// ---- butterfly orders, bit-exact ------------------------------------------------------------
// The run-time butterfly (masks 16 .. 1 over the sub-group range) and butterfly<1, 8, bfly::Asc> on
// SgpPartition<8> (masks 1, 2, 4), each against a host simulation in the same precision: an order change flips
// bits. Lane 5 of every item is NaN in the max inputs: the Fmax butterflies must drop it.
TYPED_TEST(DeviceLapackGeqrfTest, ButterflyOrdersAreBitExact) {
    using T = TypeParam;
    using R = real_of<T>;
    using A = device::Arith<T>;
    auto h = batched<T>(32, 31, [](int, T*) {});
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = h[i] * T(R(1 + (i % 32 % 7) * 1000));  // by lane: keeps period 4
    T* in = this->dev.up(h);
    constexpr int K = 4;
    std::vector<T> ho(std::size_t(kBatch) * 32 * K, nan_of<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const int l = static_cast<int>(it.get_local_linear_id());
        const T v = in[b * 32 + l];
        T* out = o + std::ptrdiff_t(b * 32 + l) * K;
        const auto sg = it.get_sub_group();
        const int lanes = static_cast<int>(sg.get_local_linear_range());
        out[0] = device::butterfly(sg, v, [](const T& p, const T& q) { return A::add(p, q); }, 1, lanes);
        if constexpr (!is_cplx<T>) {
            const T vm = l == 5 ? std::numeric_limits<T>::quiet_NaN() : v;
            const device::SgpPartition<8> part{it.get_sub_group()};
            out[1] = device::butterfly(sg, vm, device::op::FMax{}, 1, lanes);
            out[2] = device::butterfly<1, 8, device::bfly::Asc>(part, v, sycl::plus<R>());
            out[3] = device::butterfly<1, 8, device::bfly::Asc>(part, vm, device::op::FMax{});
        }
    });
    const auto got = this->dev.down(o, ho.size());
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<T> desc(h.begin() + b * 32, h.begin() + b * 32 + 32), dmx = desc, asc = desc, amx = desc;
        if constexpr (!is_cplx<T>) dmx[5] = amx[5] = std::numeric_limits<T>::quiet_NaN();
        for (int off = 16; off > 0; off >>= 1) {
            std::vector<T> nd(32), nm(32);
            for (int l = 0; l < 32; ++l) {
                nd[l] = A::add(desc[l], desc[l ^ off]);
                if constexpr (!is_cplx<T>) nm[l] = std::fmax(dmx[l], dmx[l ^ off]);
            }
            desc = nd;
            dmx = nm;
        }
        for (int mask = 1; mask < 8; mask <<= 1) {
            std::vector<T> na(32), nm(32);
            for (int l = 0; l < 32; ++l) {
                na[l] = A::add(asc[l], asc[l ^ mask]);
                if constexpr (!is_cplx<T>) nm[l] = std::fmax(amx[l], amx[l ^ mask]);
            }
            asc = na;
            amx = nm;
        }
        for (int l = 0; l < 32; ++l) {
            const T* g = got.data() + (std::size_t(b) * 32 + l) * K;
            bad += !same_bits(g[0], desc[l]);
            if constexpr (!is_cplx<T>) bad += !same_bits(g[1], dmx[l]) + !same_bits(g[2], asc[l]) + !same_bits(g[3], amx[l]);
        }
    }
    EXPECT_EQ(bad, 0) << "a butterfly ran in a different order than its name says, or kept a NaN";
    EXPECT_EQ(period_bitdiff(got, 32 * K), 0);
}

// ---- larf_left_head ------------------------------------------------------------------------
// A (m x nc, ld m + 5) := (I - tau v v^H) A on rows j.., columns j+1.., v = [1; A(j+1:, j)],
// j = 2. A(0:j, j) and the head A(j, j) hold finite poison: never read as v. Columns 0..j and
// rows above j keep their bits.
template <class T, int WG, device::Scope S> void larf_case(Dev& dev, const char* what) {
    constexpr int m = 41, nc = 11, ld = m + 5, j = 2, per = ld * nc + 1;
    auto h = batched<T>(per, 23, [&](int, T* p) {
        for (int r = 0; r <= j; ++r) p[j * ld + r] = poison_of<T>();
    });
    T* d = dev.up(h);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        T* p = d + std::ptrdiff_t(it.get_group_linear_id()) * per;
        const device::RtGroup<S> ex{it};
        device::detail::larf_left_head(ex, ex.team_size(), p[ld * nc], device::mat_wide(p, m, nc, ld), j, m, nc);
    });
    const auto got = dev.down(d, h.size());
    EXPECT_EQ(period_bitdiff(got, per), 0) << what;
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* p = h.data() + std::size_t(b) * per;
        const T* g = got.data() + std::size_t(b) * per;
        const LD<T> tau(p[ld * nc]);
        auto v = [&](int r) { return r == j ? LD<T>(1) : LD<T>(p[j * ld + r]); };
        for (int c = 0; c < nc; ++c) {
            if (c <= j) {
                for (int r = 0; r < m; ++r) EXPECT_TRUE(same_bits(g[c * ld + r], p[c * ld + r])) << what << ": column " << c;
                continue;
            }
            LD<T> w(0);
            long double s = 0;
            for (int r = j; r < m; ++r) {
                w += conj_h<T>(v(r)) * LD<T>(p[c * ld + r]);
                s += std::abs(v(r)) * std::abs(LD<T>(p[c * ld + r]));
            }
            for (int r = 0; r < j; ++r) EXPECT_TRUE(same_bits(g[c * ld + r], p[c * ld + r])) << what << ": row above head";
            for (int r = j; r < m; ++r) {
                const LD<T> ref = LD<T>(p[c * ld + r]) - tau * v(r) * w;
                worst = std::max(worst, rel(g[c * ld + r], ref, std::abs(LD<T>(p[c * ld + r])) + std::abs(tau * v(r)) * s));
            }
        }
    }
    EXPECT_VERIFY(T, Check::blas, m, worst) << what;
}

TYPED_TEST(DeviceLapackGeqrfTest, LarfLeftHead) {
    using T = TypeParam;
    larf_case<T, 96, device::Scope::Group>(this->dev, "RtGroup<Group>@96");
    larf_case<T, 1024, device::Scope::Group>(this->dev, "RtGroup<Group>@1024");
    larf_case<T, 32, device::Scope::Sub>(this->dev, "RtGroup<Sub>@32");
}

// ---- team slots --------------------------------------------------------------------------------
// WG 1024 (32 teams): each team's run-time butterfly sum (and Fmax for real types) put in a slot, a
// barrier, then folded in slot order; bit-exact against the host simulation.
TYPED_TEST(DeviceLapackGeqrfTest, TeamSlotFolds) {
    using T = TypeParam;
    using A = device::Arith<T>;
    constexpr int WG = 1024, NT = WG / 32;
    auto h = batched<T>(WG, 71, [](int, T*) {});
    T* in = this->dev.up(h);
    std::vector<T> ho(std::size_t(kBatch) * 2, nan_of<T>());
    T* o = this->dev.up(ho);
    this->dev.template run<WG>(kBatch, sizeof(T) * 2 * NT, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const device::RtGroup<device::Scope::Group> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        T* slots = reinterpret_cast<T*>(lm);
        const T v = in[std::ptrdiff_t(b) * WG + ex.lid()];
        const auto add = [](const T& p, const T& q) { return A::add(p, q); };
        device::detail::team_slot_put(ex, slots, 0, device::butterfly(ex.sub(), v, add, 1, ex.team_size()));
        if constexpr (!is_cplx<T>) device::detail::team_slot_put(ex, slots, NT, device::butterfly(ex.sub(), sycl::fabs(v), device::op::FMax{}, 1, ex.team_size()));
        ex.barrier();
        if (ex.leader()) {
            o[b * 2] = device::detail::slot_fold_sum(slots, 0, ex.nteams());
            if constexpr (!is_cplx<T>) o[b * 2 + 1] = device::detail::slot_fold_max(slots, NT, ex.nteams());
        }
    });
    const auto got = this->dev.down(o, ho.size());
    EXPECT_EQ(period_bitdiff(got, 2), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        T sum = A::zero(), mx{};
        for (int t = 0; t < NT; ++t) {
            std::vector<T> s(h.begin() + std::size_t(b) * WG + t * 32, h.begin() + std::size_t(b) * WG + t * 32 + 32), x = s;
            if constexpr (!is_cplx<T>)
                for (auto& e : x) e = std::fabs(e);
            for (int off = 16; off > 0; off >>= 1) {
                std::vector<T> ns(32), nx(32);
                for (int l = 0; l < 32; ++l) {
                    ns[l] = A::add(s[l], s[l ^ off]);
                    if constexpr (!is_cplx<T>) nx[l] = std::fmax(x[l], x[l ^ off]);
                }
                s = ns;
                x = nx;
            }
            sum = A::add(sum, s[0]);
            if constexpr (!is_cplx<T>) mx = t == 0 ? x[0] : std::fmax(mx, x[0]);
        }
        bad += !same_bits(got[b * 2], sum);
        if constexpr (!is_cplx<T>) bad += !same_bits(got[b * 2 + 1], mx);
    }
    EXPECT_EQ(bad, 0) << "a slot fold ran in another order than slot order";
}

// ---- RowFrag reflector apply ---------------------------------------------------------------
// One reflector (column j = 2, tau) applied to a row-fragment panel two ways: team partials in
// slots + ger (RtGroup<Group>, 64 items, rows >= m dead), and the local-tile column-sum form on
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

TYPED_TEST(DeviceLapackGeqrfTest, RowFragApply) {
    using T = TypeParam;
    using A = device::Arith<T>;
    constexpr int N = 8, j = 2;
    // (a) slots: m = 50 rows of 64 items, n = 7 of N columns.
    {
        constexpr int WG = 64, m = 50, n = 7, per = m * n + 1;
        auto h = batched<T>(per, 51, [](int, T*) {});
        T* d = this->dev.up(h);
        this->dev.template run<WG>(kBatch, sizeof(T) * (WG / 32) * N, [=](sycl::nd_item<1> it, unsigned char* lm) {
            T* p = d + std::ptrdiff_t(it.get_group_linear_id()) * per;
            const device::RtGroup<device::Scope::Group> ex{it};
            T* sw = reinterpret_cast<T*>(lm);
            const int r = ex.lid();
            const bool live = r < m;
            T rA[N];
#pragma clang loop unroll(full)
            for (int c = 0; c < N; ++c) rA[c] = (live && c < n) ? p[c * m + r] : A::zero();
            const T v = (!live || r < j) ? A::zero() : (r == j ? A::one() : rA[j]);
            device::detail::gemv_rowfrag_partials<N>(ex, A::conj(v), rA, j + 1, n, sw, 0, N);
            ex.barrier();
            device::detail::ger_rowfrag_slots<N>(p[m * n], v, rA, j + 1, n, sw, 0, ex.nteams(), N);
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
            const device::SgpPartition<N> ex{sg};
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
                device::detail::frag_scale_to_row<C>(A::conj(vr), rA, k0, j + 1, n, sP, lane * SLD);
                sycl::group_barrier(sg);
                device::detail::colsum_colperlane<C, N>(lane, A::conj(p[N * N]), sP, SLD, j, sy);
                sycl::group_barrier(sg);
                device::detail::frag_rank1<C>(vr, sy, rA, k0, j + 1, n);
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
