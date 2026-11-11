// SPDX-License-Identifier: MIT
// The public device getrf (include/batchlas/device/lapack/getrf.hh) called from a user-style
// kernel against the host routes whose body it is: tier::Reg vs getrf_tiny_dispatch and
// tier::Cta vs getrf_cta_dispatch, BIT-IDENTICAL (factor bytes incl. pads, int32 pivots, info).
// Guards (agent-guide section 8): each host route reached through its own dispatch (no table);
// ld = n + 3 and a batch stride past ld * n, pads poisoned with a large finite value; the user's
// std::complex with nonzero imaginary parts (the host re-types to its POD Cx); batch 1024 of
// period-3 data, every item bit-identical to its representative; pivots against a long-double
// ?GETF2 oracle on inputs with a 4x pivot margin and a singular representative (info = n by
// construction); orders straddling each tier bound; a launch at the Cta<128> ceiling; with_tier's
// pick per n, its local-memory / fp64 / sub-group filters and its throw; buffer_size by hand; and
// a deliberate break (a dead item must be the ONLY mismatch, so the comparison can fail).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/ops/getrf/getrf_native.hh>
#include <src/queue.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/resident_capacity.hh>

using namespace batchlas;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;
template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr float kPadPoison = 1234.5f;
constexpr int kGap = 5;  // batch stride = ld * n + kGap

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
template <class T> long double cabs1(LD<T> v) {
    if constexpr (is_cplx<T>) return std::fabs(v.real()) + std::fabs(v.imag());
    else return std::fabs(v);
}

// A = P0 L0 U0, |L0(i,j)| <= 0.17 below a unit diagonal, |U0(j,j)| >= 1: partial pivoting picks
// P0's rows with a 4x margin (cabs1 included). `singular` zeroes U0's last column: that Schur
// column stays EXACTLY zero and info = n.
template <class T> std::vector<T> lu_input(int n, int ld, std::uint64_t seed, bool singular) {
    using R = real_of<T>;
    std::vector<LD<T>> L(std::size_t(n) * n, LD<T>(0)), U(std::size_t(n) * n, LD<T>(0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            L[i + std::size_t(j) * n] = i == j ? LD<T>(1) : LD<T>(val<T>(seed + 7u * i + 1000u * j)) * (long double)0.17;
    for (int j = 0; j < n; ++j)
        for (int c = j; c < n; ++c) {
            LD<T> u = LD<T>(val<T>(seed + 99991u + 13u * j + 1777u * c));
            if (c == j) u = LD<T>(T(R(1.5) + R(0.5) * uval<R>(seed + 31u * j))) + u * (long double)0.1;
            U[j + std::size_t(c) * n] = (singular && c == n - 1) ? LD<T>(0) : u;
        }
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[mix(seed + 555u + i) % std::uint64_t(i + 1)]);
    std::vector<T> a(std::size_t(ld) * n, T(R(kPadPoison)));
    for (int i = 0; i < n; ++i)
        for (int c = 0; c < n; ++c) {
            LD<T> s = 0;
            for (int j = 0; j < n; ++j) s += L[i + std::size_t(j) * n] * U[j + std::size_t(c) * n];
            a[perm[i] + std::size_t(c) * ld] = static_cast<T>(s);
        }
    return a;
}

// LAPACK ?GETF2 in long double, cabs1 metric: 1-based pivots and info.
template <class T> void oracle(const T* a0, int n, int ld, std::vector<int>& ipiv, int& info) {
    std::vector<LD<T>> a(std::size_t(ld) * n);
    for (std::size_t i = 0; i < a.size(); ++i) a[i] = LD<T>(a0[i]);
    auto at = [&](int i, int j) -> LD<T>& { return a[i + std::size_t(j) * ld]; };
    ipiv.assign(n, 0);
    info = 0;
    for (int k = 0; k < n; ++k) {
        int p = k;
        long double best = -1;
        for (int i = k; i < n; ++i)
            if (cabs1<T>(at(i, k)) > best) best = cabs1<T>(at(i, k)), p = i;
        ipiv[k] = p + 1;
        for (int c = 0; c < n; ++c) std::swap(at(k, c), at(p, c));
        const LD<T> d = at(k, k);
        if (d == LD<T>(0)) {
            if (info == 0) info = k + 1;
            continue;
        }
        for (int i = k + 1; i < n; ++i) at(i, k) /= d;
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < n; ++i) at(i, j) -= at(i, k) * at(k, j);
    }
}

// One batch, run twice from the same bytes: by the host route and by the device op.
template <class T> struct Prob {
    int n, ld, stride;
    std::vector<T> a0;
    UnifiedVector<T> a_host, a_dev;
    UnifiedVector<std::int64_t> piv_host;
    UnifiedVector<std::int32_t> piv_dev, info_host, info_dev;
    Prob(int n_, std::uint64_t seed)
        : n(n_), ld(n_ + 3), stride((n_ + 3) * n_ + kGap), a0(std::size_t(stride) * kBatch, T(real_of<T>(kPadPoison))),
          a_host(a0.size()), a_dev(a0.size()), piv_host(std::size_t(n_) * kBatch, 0), piv_dev(std::size_t(n_) * kBatch, -1),
          info_host(kBatch, -12345), info_dev(kBatch, -12345) {
        for (int r = 0; r < kPeriod; ++r) {
            const auto a = lu_input<T>(n, ld, seed + 104729u * r, r == 1);
            for (int b = r; b < kBatch; b += kPeriod) std::copy(a.begin(), a.end(), a0.begin() + std::size_t(stride) * b);
        }
        std::copy(a0.begin(), a0.end(), a_host.data());
        std::copy(a0.begin(), a0.end(), a_dev.data());
    }
    std::int32_t host_piv(std::size_t i) const { return reinterpret_cast<const std::int32_t*>(piv_host.data())[i]; }
};

enum class Route { Tiny, Cta };

template <class T> void run_host(Queue& ctx, Prob<T>& p, Route r) {
    const MatrixView<T, MatrixFormat::Dense> A(p.a_host.data(), p.n, p.n, p.ld, p.stride, kBatch);
    if (r == Route::Tiny) {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, sycl_getrf::getrf_tiny_buffer_size<T>(ctx, A)));
        (void)sycl_getrf::getrf_tiny_dispatch<T>(ctx, A, p.piv_host.to_span(), ws.to_span(), p.info_host.to_span());
    } else {
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, sycl_getrf::getrf_cta_buffer_size<T>(ctx, A)));
        (void)sycl_getrf::getrf_cta_dispatch<T>(ctx, A, p.piv_host.to_span(), ws.to_span(), p.info_host.to_span());
    }
    ctx.wait();
}

// The user's kernel: Reg on 64-item groups of Partition<part_size>, Cta on one WorkGroup per
// matrix; `dead` is a matrix the kernel passes live = false for (the deliberate break).
template <class Tier, class T> void run_device(Queue& ctx, Prob<T>& p, int dead = -1) {
    using I = device::tier_info<device::routine::getrf, Tier, T>;
    T* a = p.a_dev.data();
    std::int32_t* piv = p.piv_dev.data();
    std::int32_t* info = p.info_dev.data();
    const int n = p.n, ld = p.ld, stride = p.stride, batch = kBatch;
    sycl::queue& q = sycl_queue(ctx);
    if constexpr (device::tier::is_reg_v<Tier>) {
        constexpr int P = I::part_size, WG = 64, per = WG / P;
        const int groups = (batch + per - 1) / per;
        q.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const device::Partition<P> px{it.get_sub_group()};
             const int b = static_cast<int>(it.get_group_linear_id()) * per + static_cast<int>(it.get_local_linear_id()) / P;
             const bool live = b < batch && b != dead;
             const int bb = live ? b : 0;
             const auto A = device::mat_wide(a + std::ptrdiff_t(bb) * stride, n, n, ld);
             const std::int32_t r = device::getrf<Tier>(px, A, piv + std::ptrdiff_t(bb) * n, device::Workspace{}, live);
             if (live && px.leader()) info[bb] = r;
         }).wait_and_throw();
    } else {
        constexpr int WG = I::wg_size;
        const device::WorkspaceSize ws = device::getrf_buffer_size<Tier, T>(n);
        q.submit([&](sycl::handler& h) {
             sycl::local_accessor<std::byte, 1> lm(sycl::range<1>(ws.local_bytes), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 const device::WorkGroup<WG, 32> ex{it};
                 const int b = static_cast<int>(it.get_group_linear_id());
                 const device::Workspace w{lm.template get_multi_ptr<sycl::access::decorated::yes>(), nullptr};
                 const auto A = device::mat_wide(a + std::ptrdiff_t(b) * stride, n, n, ld);
                 const std::int32_t r = device::getrf<Tier>(ex, A, piv + std::ptrdiff_t(b) * n, w, b != dead);
                 if (b != dead && ex.leader()) info[b] = r;
             });
         }).wait_and_throw();
    }
}

// Items whose bytes (matrix, its pads and the gap after it), pivots or info differ between the two runs.
template <class T> std::vector<int> mismatches(const Prob<T>& p) {
    std::vector<int> bad;
    for (int b = 0; b < kBatch; ++b) {
        bool same = std::memcmp(p.a_host.data() + std::size_t(p.stride) * b, p.a_dev.data() + std::size_t(p.stride) * b,
                                sizeof(T) * std::size_t(p.stride)) == 0;
        for (int k = 0; k < p.n; ++k) same = same && p.host_piv(std::size_t(b) * p.n + k) == p.piv_dev[std::size_t(b) * p.n + k];
        same = same && p.info_host[b] == p.info_dev[b];
        if (!same) bad.push_back(b);
    }
    return bad;
}

// The device run on its own: every item bit-identical to its representative, and the
// representatives' pivots and info those of the oracle.
template <class T> void check_device(const Prob<T>& p, const char* what) {
    int bitdiff = 0;
    for (int b = kPeriod; b < kBatch; ++b) {
        const int r = b % kPeriod;
        bitdiff += std::memcmp(p.a_dev.data() + std::size_t(p.stride) * b, p.a_dev.data() + std::size_t(p.stride) * r,
                               sizeof(T) * std::size_t(p.ld) * p.n) != 0;
        for (int k = 0; k < p.n; ++k) bitdiff += p.piv_dev[std::size_t(b) * p.n + k] != p.piv_dev[std::size_t(r) * p.n + k];
        bitdiff += p.info_dev[b] != p.info_dev[r];
    }
    EXPECT_EQ(bitdiff, 0) << what << " n=" << p.n << ": items differ from their representative";
    for (int r = 0; r < kPeriod; ++r) {
        std::vector<int> hp;
        int hi = 0;
        oracle(p.a0.data() + std::size_t(p.stride) * r, p.n, p.ld, hp, hi);
        for (int k = 0; k < p.n; ++k) EXPECT_EQ(p.piv_dev[std::size_t(r) * p.n + k], hp[k]) << what << " n=" << p.n << " rep " << r << " ipiv[" << k << "]";
        EXPECT_EQ(p.info_dev[r], hi) << what << " n=" << p.n << " rep " << r;
        EXPECT_EQ(p.info_dev[r], r == 1 ? p.n : 0) << what << " n=" << p.n << " rep " << r << " info by construction";
    }
}

// The host cta route's own ceiling (its occupancy-scaled local-memory gate, getrf_cta.cc): above
// it the host route throws, so only the device op runs there, against the oracle.
template <class T> int host_cta_cap(Queue& ctx) {
    const auto lm = static_cast<std::size_t>(ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
    return sycl_getrf::getrf_cta_max_n_for_slm<T>(resident::device_slm_budget(lm));
}
template <class T> bool host_runs(Queue& ctx, int n, Route r) { return r == Route::Tiny || n <= host_cta_cap<T>(ctx); }

// Whether one work-group of the tier launches with its workspace at order n: acpp Metal's 32 KiB
// less its wg * 8 B runtime scratch stops complex<float> Cta<64> below n = 64.
template <class Tier, class T> bool device_holds(Queue& ctx, int n) {
    using I = device::tier_info<device::routine::getrf, Tier, T>;
    return device::getrf_buffer_size<Tier, T>(n).local_bytes <= batchlas::impl::local_mem_bytes(sycl_queue(ctx).get_device(), I::wg_size);
}

template <class Tier, class T> void expect_thin_launcher(Queue& ctx, int n, Route route, const char* what) {
    Prob<T> p(n, 0x5eedu + 977u * n);
    const bool host = host_runs<T>(ctx, n, route);
    if (host) run_host(ctx, p, route);
    run_device<Tier>(ctx, p);
    if (host) {
        const auto bad = mismatches(p);
        EXPECT_TRUE(bad.empty()) << what << " n=" << n << ": " << bad.size() << " items differ from the host route, first " << (bad.empty() ? -1 : bad[0]);
    }
    check_device(p, what);
}

template <class Tier> std::string tier_name() {
    if constexpr (device::tier::is_reg_v<Tier>) return "Reg<" + std::to_string(Tier::n_max) + ">";
    else if constexpr (device::tier::is_blocked_v<Tier>) return "Blocked<" + std::to_string(Tier::nb) + ">";
    else return "Cta<" + std::to_string(Tier::n_max) + ">";
}
template <class T> std::string picked(const device::DeviceLimits& lim, int n) {
    std::string s;
    try {
        device::with_tier<device::routine::getrf, T>(lim, n, [&]<class Tier>(Tier) { s = tier_name<Tier>(); });
    } catch (const batchlas::invalid_argument& e) {
        s = e.what();
    }
    return s;
}

template <class T> class DeviceApiGetrfTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (ctx.device().type != DeviceType::GPU) GTEST_SKIP() << "the host tiny and cta routes are GPU only";
        if (!ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "the tiers launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(sycl_queue(ctx).get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Queue ctx{Device::default_device()};
};
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceApiGetrfTest, Types);

// Orders straddle the partition sizes (4|5, 8|9, 16|17) and the column bound of each tier,
// and n = 5 also runs on Reg<32> (a tier wider than the order).
TYPED_TEST(DeviceApiGetrfTest, RegTierIsTheTinyRouteBitForBit) {
    using T = TypeParam;
    constexpr int cap = device::detail::reg_cap<T>();
    for (int n : {3, 4, 5, 8}) expect_thin_launcher<device::tier::Reg<8>, T>(this->ctx, n, Route::Tiny, "Reg<8>");
    for (int n : {9, 15, 16}) expect_thin_launcher<device::tier::Reg<16>, T>(this->ctx, n, Route::Tiny, "Reg<16>");
    if constexpr (cap >= 32) {
        for (int n : {5, 17, 26, 31, 32}) expect_thin_launcher<device::tier::Reg<32>, T>(this->ctx, n, Route::Tiny, "Reg<32>");
    }
}

TYPED_TEST(DeviceApiGetrfTest, CtaTierIsTheCtaRouteBitForBit) {
    using T = TypeParam;
    const int cap = host_cta_cap<T>(this->ctx);  // straddled: the last order the host route takes
    ASSERT_GE(cap, 17) << "the host cta route takes no order this test compares";
    int top = 64;  // the largest order this device holds, launched at that ceiling
    while (top > cap && !device_holds<device::tier::Cta<64>, T>(this->ctx, top)) --top;
    for (int n : {17, 33, 47, cap, cap + 1, 63, 64, top}) {
        if (n <= top) expect_thin_launcher<device::tier::Cta<64>, T>(this->ctx, n, Route::Cta, "Cta<64>");
    }
}

// The advertised ceiling, launched: Cta<128> at n = 128 wherever the device's local memory holds it.
TYPED_TEST(DeviceApiGetrfTest, CtaTierLaunchesAtItsCeiling) {
    using T = TypeParam;
    const auto need = device::getrf_buffer_size<device::tier::Cta<128>, T>(128).local_bytes;
    if (!device_holds<device::tier::Cta<128>, T>(this->ctx, 128)) GTEST_SKIP() << "Cta<128> needs " << need << " bytes of local memory";
    Prob<T> p(128, 0xce11u);
    run_device<device::tier::Cta<128>>(this->ctx, p);
    check_device(p, "Cta<128>");
}

// C.9's example: the host picks the tier for a run-time n, the kernel instantiates only that one.
TYPED_TEST(DeviceApiGetrfTest, WithTierLaunchesThePickedTier) {
    using T = TypeParam;
    for (int n : {13, 40}) {
        Prob<T> p(n, 0x7e11u + n);
        const Route r = n <= 32 && n <= device::detail::reg_cap<T>() ? Route::Tiny : Route::Cta;
        const bool host = host_runs<T>(this->ctx, n, r);
        if (host) run_host(this->ctx, p, r);
        device::with_tier<device::routine::getrf, T>(sycl_queue(this->ctx).get_device(), n,
                                             [&]<class Tier>(Tier) { run_device<Tier>(this->ctx, p); });
        if (host) EXPECT_TRUE(mismatches(p).empty()) << "with_tier n=" << n;
        check_device(p, "with_tier");
    }
}

// The deliberate break: the comparison must see a matrix the device op did not factor, and
// nothing else.
TYPED_TEST(DeviceApiGetrfTest, DeliberateBreakADeadItemIsTheOnlyMismatch) {
    using T = TypeParam;
    {
        Prob<T> p(12, 0xdeadu);
        run_host(this->ctx, p, Route::Tiny);
        run_device<device::tier::Reg<16>>(this->ctx, p, 517);
        EXPECT_EQ(mismatches(p), std::vector<int>{517}) << "Reg<16>";
    }
    {
        Prob<T> p(std::min(40, host_cta_cap<T>(this->ctx)), 0xbeefu);
        run_host(this->ctx, p, Route::Cta);
        run_device<device::tier::Cta<64>>(this->ctx, p, 3);
        EXPECT_EQ(mismatches(p), std::vector<int>{3}) << "Cta<64>";
    }
}

// Host only: with_tier's pick and filters on fixed device limits, buffer_size by hand.
TEST(DeviceApiGetrfTier, WithTierPicksTheFirstFittingTier) {
    const device::DeviceLimits full{99 * 1024, {32}, 1024, true};
    const std::vector<std::pair<int, std::string>> f = {{1, "Reg<8>"},   {8, "Reg<8>"},    {9, "Reg<16>"},  {16, "Reg<16>"},
                                                        {17, "Reg<32>"}, {32, "Reg<32>"},  {33, "Cta<64>"}, {64, "Cta<64>"},
                                                        {65, "Cta<96>"}, {96, "Cta<96>"},  {97, "Cta<128>"}, {128, "Cta<128>"}};
    for (const auto& [n, want] : f) EXPECT_EQ(picked<float>(full, n), want) << "float n=" << n;
    EXPECT_EQ(picked<float>(full, 129), "Blocked<32>") << "past the last Cta tier, any n";
    EXPECT_EQ(picked<float>(full, 5000), "Blocked<32>");
    EXPECT_EQ(picked<float>(full, 0), "getrf: no device tier admits n=0");
    EXPECT_EQ(picked<std::complex<double>>(full, 16), "Reg<16>");
    EXPECT_EQ(picked<std::complex<double>>(full, 17), "Cta<64>") << "complex<double> has no Reg<32>";

    const device::DeviceLimits small{32 * 1024, {32}, 1024, true};
    EXPECT_EQ(picked<float>(small, 80), "Cta<96>") << "81 * 80 * 4 + 256 = 26176 bytes fit 32 KB";
    EXPECT_EQ(picked<float>(small, 100), "Blocked<32>") << "Cta<128>'s 101 * 100 * 4 + 256 = 40656 bytes do not fit; Blocked's 2464 do";
    EXPECT_EQ(picked<float>(device::DeviceLimits{2 * 1024, {32}, 1024, true}, 100), "getrf: no device tier admits n=100")
        << "Blocked's 2464 bytes do not fit 2 KB";
    EXPECT_EQ(picked<double>(device::DeviceLimits{99 * 1024, {32}, 1024, false}, 5), "getrf: no device tier admits n=5");
    EXPECT_EQ(picked<float>(device::DeviceLimits{99 * 1024, {16}, 1024, true}, 5), "getrf: no device tier admits n=5");
    EXPECT_EQ(picked<float>(device::DeviceLimits{99 * 1024, {32}, 128, true}, 64), "getrf: no device tier admits n=64")
        << "Cta<64> launches 256 items";
}

TEST(DeviceApiGetrfTier, BufferSizeIsThePlanByHand) {
    using device::WorkspaceSize;
    // float n = 64: tile 65 * 64 * 4 = 16640, slots 32 * 4 + 32 * 4.
    EXPECT_EQ((device::getrf_buffer_size<device::tier::Cta<64>, float>(64)), (WorkspaceSize{16896, 0, 16}));
    // complex<double> n = 33: tile 33 * 33 * 16 = 17424, slots 32 * 8 + 32 * 4.
    EXPECT_EQ((device::getrf_buffer_size<device::tier::Cta<64>, std::complex<double>>(33)), (WorkspaceSize{17808, 0, 16}));
    // float n = 7: tile 7 * 7 * 4 = 196 -> 208 aligned, slots 128 + 128, total 464.
    EXPECT_EQ((device::getrf_buffer_size<device::tier::Cta<64>, float>(7)), (WorkspaceSize{464, 0, 16}));
    EXPECT_EQ((device::getrf_buffer_size<device::tier::Reg<32>, float>(32)), (WorkspaceSize{}));
    EXPECT_EQ((device::getrf_buffer_size<device::tier::upto<20>, float>(20)), (WorkspaceSize{}));
    static_assert(std::is_same_v<device::resolve_tier_t<device::routine::getrf, float, device::tier::upto<20>>, device::tier::Reg<32>>);
    static_assert(std::is_same_v<device::resolve_tier_t<device::routine::getrf, float, device::tier::upto<33>>, device::tier::Cta<64>>);
    static_assert(std::is_same_v<device::resolve_tier_t<device::routine::getrf, std::complex<double>, device::tier::upto<20>>, device::tier::Cta<64>>);
    static_assert(std::is_same_v<device::resolve_tier_t<device::routine::getrf, float, device::tier::upto<129>>, device::tier::Blocked<32>>);
    static_assert(device::HasTier<device::routine::getrf, device::tier::Reg<16>, std::complex<double>>);
    static_assert(!device::HasTier<device::routine::getrf, device::tier::Reg<32>, std::complex<double>>);
    static_assert(device::tier_info<device::routine::getrf, device::tier::Reg<5>, float>::part_size == 8);
    static_assert(device::tier_info<device::routine::getrf, device::tier::Cta<64>, float>::wg_size == 256);
    static_assert(device::tier_info<device::routine::getrf, device::tier::Cta<128>, float>::wg_size == 512);
    static_assert(!device::tier_info<device::routine::getrf, device::tier::Reg<8, true>, float>::admits(7));
}

}  // namespace
