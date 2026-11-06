// SPDX-License-Identifier: MIT
// device_api_sb2st_tests: the public device op device::sb2st (include/batchlas/device/lapack/sb2st.hh)
// called from a user-style kernel (one SgpPartition<P> per matrix, the caller's Workspace), against
// the host route sytrd_sb2st on the same band.
// Guards (docs/developer/agent-guide.md section 8):
//   - pinned route: BATCHLAS_SB2ST_SUBGROUP=1 makes the host throw unless its sub-group kernel runs
//     (kd <= 1: the kd0 / kd1 kernels; kd > 32: the work-group kernel, unpinned, tolerance only);
//   - both tiers BIT-IDENTICAL to the pinned host route (memcmp of d, e, tau), batch 1024 with
//     period-3 data where every item must equal its representative bit for bit;
//   - ld = kd + 4 with NaN-poisoned pad rows that must survive; d / e / tau poisoned before the
//     call, a poisoned tail that must survive; complex data with Im(diag) != 0; one graded item;
//   - kd > P (kd = 12 on P = 8), kd = 32 on P = 32 at n = N for Cta<64>; n = N + 1 returns -1
//     uniformly and leaves every output untouched while Blocked runs it;
//   - with_tier straddles the local-memory boundary exactly (need, need - 1) and the N bound.
// Red set (make each break once on the GPU box before calling this suite green): finalize_tridiag
// without the phase chain (complex BitIdentical fails); givens_on staging rows 0..kd-1 only
// (BitIdentical, Cta); the Cta n > N check removed (CtaBoundReturnsMinusOne).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/sb2st.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace dv = batchlas::device;

namespace {

template <class T> using real_of = typename dv::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = dv::Arith<T>::is_complex;

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr int kTail = 8;  // poisoned elements after the last item of every output

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
template <class T> T poison() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

template <class V> int bitdiff(const std::vector<V>& a, const std::vector<V>& b) {
    int d = 0;
    for (std::size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(V)) != 0;
    return d + static_cast<int>(a.size() != b.size());
}
template <class V> int bitdiff_vs_rep(const std::vector<V>& got, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i)
            d += std::memcmp(&got[std::size_t(b) * per_item + i], &got[std::size_t(b % kPeriod) * per_item + i], sizeof(V)) != 0;
    return d;
}
template <class V> int poison_lost(const std::vector<V>& got, std::size_t from) {
    const V p = poison<V>();
    int d = 0;
    for (std::size_t i = from; i < got.size(); ++i) d += std::memcmp(&got[i], &p, sizeof(V)) != 0;
    return d;
}

// kBatch lower bands (ld x n, rows kd + 1 .. ld - 1 poisoned, entries past the matrix 0), item b a
// copy of representative b % kPeriod; representative 1 is graded (D A D, D = diag(2^(-8 k / n))).
template <class T> std::vector<T> make_band(int n, int kd, int ld, std::uint64_t seed) {
    using R = real_of<T>;
    const std::size_t per = std::size_t(ld) * n;
    std::vector<T> h(per * kBatch);
    for (int r = 0; r < kPeriod; ++r) {
        T* a = h.data() + std::size_t(r) * per;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ld; ++i) {
                T& x = a[i + std::size_t(j) * ld];
                if (i > kd) x = poison<T>();
                else if (j + i >= n) x = T(0);
                else {
                    x = val<T>(seed + std::uint64_t(r) * 1000003u + std::uint64_t(j) * ld + i);
                    if (r == 1) x = x * R(std::exp2(-8.0 * (j + i) / n) * std::exp2(-8.0 * j / n));
                }
            }
    }
    for (int b = kPeriod; b < kBatch; ++b) std::copy_n(h.data() + std::size_t(b % kPeriod) * per, per, h.data() + std::size_t(b) * per);
    return h;
}

template <class T> struct Out {
    std::vector<real_of<T>> d, e;
    std::vector<T> tau, ab;
    std::vector<int> info;
};

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_device<T>(std::max<std::size_t>(h.size(), 1), q());
        if (!h.empty()) q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        if (n) q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T, int P, class Tier> class UserSb2stKernel;
struct alignas(16) Chunk16 {
    unsigned char b[16];
};

// The user's kernel: work-groups of 32, 32 / P matrices each, each partition its own local slice.
template <class T, int P, class Tier> Out<T> run_device(Dev& dev, const std::vector<T>& band, int n, int kd, int ld) {
    using R = real_of<T>;
    const int nm1 = std::max(0, n - 1);
    const dv::WorkspaceSize sz = dv::sb2st_buffer_size<Uplo::Lower, Tier, T>(n, kd);
    T* ab = dev.up(band);
    R* d = dev.up(std::vector<R>(std::size_t(kBatch) * n + kTail, poison<R>()));
    R* e = dev.up(std::vector<R>(std::size_t(kBatch) * nm1 + kTail, poison<R>()));
    T* tau = dev.up(std::vector<T>(std::size_t(kBatch) * nm1 + kTail, poison<T>()));
    int* info = dev.up(std::vector<int>(kBatch, 12345));
    std::byte* gws = dev.up(std::vector<std::byte>(sz.global_bytes * kBatch));
    constexpr int WG = 32, per_wg = WG / P;
    const std::size_t lbytes = sz.local_bytes;
    const std::size_t l16 = std::max<std::size_t>((lbytes * per_wg + 15) / 16, 1);
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<Chunk16, 1> slm(sycl::range<1>(l16), h);
           h.parallel_for<UserSb2stKernel<T, P, Tier>>(
               sycl::nd_range<1>(std::size_t(kBatch / per_wg) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                   const auto sg = it.get_sub_group();
                   const dv::SgpPartition<P> pex{sg};
                   const int part = static_cast<int>(sg.get_local_linear_id()) / P;
                   const int b = static_cast<int>(it.get_group(0)) * per_wg + part;
                   std::byte* lb = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get_raw()) +
                                   std::size_t(part) * lbytes;
                   const dv::Workspace ws{
                       sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(lb),
                       dv::global_slice(gws, sz, b)};
                   const auto A = dv::mat(ab + std::ptrdiff_t(b) * ld * n, kd + 1, n, ld);
                   const auto D = dv::vec(d + std::ptrdiff_t(b) * n, n);
                   const auto E = dv::vec(e + std::ptrdiff_t(b) * nm1, nm1);
                   const auto Tau = dv::vec(tau + std::ptrdiff_t(b) * nm1, nm1);
                   const int r = dv::sb2st<Uplo::Lower, Tier>(pex, A, D, E, Tau, kd, ws);
                   if (pex.leader()) info[b] = r;
               });
       }).wait_and_throw();
    return {dev.down(d, std::size_t(kBatch) * n + kTail), dev.down(e, std::size_t(kBatch) * nm1 + kTail),
            dev.down(tau, std::size_t(kBatch) * nm1 + kTail), dev.down(ab, band.size()), dev.down(info, kBatch)};
}

// The host route on a device copy of the same band (its input is not overwritten).
template <class T> Out<T> run_host(Dev& dev, const std::vector<T>& band, int n, int kd, int ld) {
    using R = real_of<T>;
    const int nm1 = std::max(0, n - 1);
    T* ab = dev.up(band);
    R* d = dev.up(std::vector<R>(std::size_t(kBatch) * n + kTail, poison<R>()));
    R* e = dev.up(std::vector<R>(std::size_t(kBatch) * nm1 + kTail, poison<R>()));
    T* tau = dev.up(std::vector<T>(std::size_t(kBatch) * nm1 + kTail, poison<T>()));
    const MatrixView<T, MatrixFormat::Dense> abv(ab, kd + 1, n, ld, ld * n, kBatch);
    const VectorView<R> dvw(d, n, kBatch, 1, n);
    const VectorView<R> evw(e, nm1, kBatch, 1, nm1);
    const VectorView<T> tvw(tau, nm1, kBatch, 1, nm1);
    UnifiedVector<std::byte> ws(sytrd_sb2st_buffer_size(dev.ctx, abv, dvw, evw, tvw, Uplo::Lower, kd, 16));
    sytrd_sb2st(dev.ctx, abv, dvw, evw, tvw, Uplo::Lower, kd, ws.to_span(), 16).wait();
    return {dev.down(d, std::size_t(kBatch) * n + kTail), dev.down(e, std::size_t(kBatch) * nm1 + kTail),
            dev.down(tau, std::size_t(kBatch) * nm1 + kTail), {}, std::vector<int>(kBatch, 0)};
}

template <class T> class DeviceApiSb2stTest : public ::testing::Test {
public:
    bool fits(std::size_t local_bytes_per_wg) {
        return local_bytes_per_wg <= dev.q().get_device().template get_info<sycl::info::device::local_mem_size>();
    }
    Dev dev;

protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "sb2st's partition chase runs at sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceApiSb2stTest, Types);

// Device op (one tier) vs the pinned host route, bit for bit, plus the band's poisoned pad rows and
// the outputs' poisoned tails.
template <class T, int P, class Tier, class Fx> void expect_bit_identical(Fx& fx, int n, int kd, const Out<T>& host, const std::vector<T>& band) {
    const int ld = kd + 4, nm1 = std::max(0, n - 1);
    if constexpr (dv::sb2st_detail::is_cta_v<Tier>) {
        if (!fx.fits(dv::sb2st_buffer_size<Uplo::Lower, Tier, T>(n, kd).local_bytes * (32 / P))) {
            std::printf("[   NOTE   ] Cta tier at n=%d kd=%d does not fit this device's local memory; not run\n", n, kd);
            return;
        }
    }
    const Out<T> got = run_device<T, P, Tier>(fx.dev, band, n, kd, ld);
    EXPECT_EQ(std::count(got.info.begin(), got.info.end(), 0), kBatch);
    EXPECT_EQ(bitdiff(got.d, host.d), 0) << "d";
    EXPECT_EQ(bitdiff(got.e, host.e), 0) << "e";
    EXPECT_EQ(bitdiff(got.tau, host.tau), 0) << "tau";
    EXPECT_EQ(bitdiff_vs_rep(got.d, n), 0);
    EXPECT_EQ(bitdiff_vs_rep(got.e, nm1), 0);
    EXPECT_EQ(poison_lost(got.d, std::size_t(kBatch) * n), 0);
    EXPECT_EQ(poison_lost(got.e, std::size_t(kBatch) * nm1), 0);
    EXPECT_EQ(poison_lost(got.tau, std::size_t(kBatch) * nm1), 0);
    int pads = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = kd + 1; i < ld; ++i) {
                const std::size_t k = std::size_t(b) * ld * n + std::size_t(j) * ld + i;
                pads += std::memcmp(&got.ab[k], &band[k], sizeof(T)) != 0;
            }
    EXPECT_EQ(pads, 0) << "pad rows of the band";
}

TYPED_TEST(DeviceApiSb2stTest, BothTiersBitIdenticalToThePinnedSubGroupRoute) {
    using T = TypeParam;
    ScopedEnvVar pin("BATCHLAS_SB2ST_SUBGROUP", "1");
    struct Case {
        int n, kd;
    };
    // P = 4: kd 3; P = 8: kd 6 and kd 12 > P; P = 32: kd 32 (the sub-group route's ceiling), n = N.
    for (const Case cs : {Case{24, 3}, Case{40, 6}, Case{33, 12}, Case{64, 32}}) {
        SCOPED_TRACE("n=" + std::to_string(cs.n) + " kd=" + std::to_string(cs.kd));
        const int ld = cs.kd + 4;
        const auto band = make_band<T>(cs.n, cs.kd, ld, 77 + cs.n);
        const Out<T> host = run_host<T>(this->dev, band, cs.n, cs.kd, ld);
        if (cs.kd <= 3) {
            expect_bit_identical<T, 4, dv::tier::Cta<64>>(*this, cs.n, cs.kd, host, band);
            expect_bit_identical<T, 4, dv::tier::Blocked<32>>(*this, cs.n, cs.kd, host, band);
        } else if (cs.kd <= 12) {
            expect_bit_identical<T, 8, dv::tier::Cta<64>>(*this, cs.n, cs.kd, host, band);
            expect_bit_identical<T, 8, dv::tier::Blocked<32>>(*this, cs.n, cs.kd, host, band);
        } else {
            expect_bit_identical<T, 32, dv::tier::Cta<64>>(*this, cs.n, cs.kd, host, band);
            expect_bit_identical<T, 32, dv::tier::Blocked<32>>(*this, cs.n, cs.kd, host, band);
        }
    }
}

// kd = 0 and kd = 1 take no chase: the device op equals the host's kd0 / kd1 kernels.
TYPED_TEST(DeviceApiSb2stTest, NarrowBandsBitIdenticalToTheHostExtracts) {
    using T = TypeParam;
    for (const int kd : {0, 1}) {
        SCOPED_TRACE("kd=" + std::to_string(kd));
        const int n = 10, ld = kd + 4;
        const auto band = make_band<T>(n, kd, ld, 5 + kd);
        const Out<T> host = run_host<T>(this->dev, band, n, kd, ld);
        expect_bit_identical<T, 4, dv::tier::Cta<64>>(*this, n, kd, host, band);
    }
}

// n = N + 1 for Cta<N>: info -1 on every item, nothing written; Blocked runs the same band.
TYPED_TEST(DeviceApiSb2stTest, CtaBoundReturnsMinusOneAndBlockedRunsIt) {
    using T = TypeParam;
    ScopedEnvVar pin("BATCHLAS_SB2ST_SUBGROUP", "1");
    const int n = 65, kd = 3, ld = kd + 4;
    const auto band = make_band<T>(n, kd, ld, 901);
    if (this->fits(dv::sb2st_buffer_size<Uplo::Lower, dv::tier::Cta<64>, T>(n, kd).local_bytes * 8)) {
        const Out<T> got = run_device<T, 4, dv::tier::Cta<64>>(this->dev, band, n, kd, ld);
        EXPECT_EQ(std::count(got.info.begin(), got.info.end(), -1), kBatch);
        EXPECT_EQ(poison_lost(got.d, 0), 0);
        EXPECT_EQ(poison_lost(got.e, 0), 0);
        EXPECT_EQ(poison_lost(got.tau, 0), 0);
        EXPECT_EQ(bitdiff(got.ab, band), 0);
    } else {
        std::printf("[   NOTE   ] Cta<64> at n=65 does not fit this device's local memory; the -1 check did not run\n");
    }
    const Out<T> host = run_host<T>(this->dev, band, n, kd, ld);
    expect_bit_identical<T, 4, dv::tier::Blocked<32>>(*this, n, kd, host, band);
}

// kd > 32: the host runs its work-group kernel (another body), so d and e agree to rounding only.
TYPED_TEST(DeviceApiSb2stTest, WideBandMatchesTheWorkGroupRoute) {
    using T = TypeParam;
    using R = real_of<T>;
    ScopedEnvVar unpin("BATCHLAS_SB2ST_SUBGROUP", "0");
    const int n = 64, kd = 40, ld = kd + 4, nm1 = n - 1;
    const auto band = make_band<T>(n, kd, ld, 4242);
    const Out<T> host = run_host<T>(this->dev, band, n, kd, ld);
    const Out<T> got = run_device<T, 32, dv::tier::Blocked<32>>(this->dev, band, n, kd, ld);
    EXPECT_EQ(std::count(got.info.begin(), got.info.end(), 0), kBatch);
    const double tol = 64.0 * n * std::numeric_limits<R>::epsilon() * (kd + 1);
    int bad = 0;
    for (std::size_t i = 0; i < std::size_t(kBatch) * n; ++i) bad += !(std::fabs(double(got.d[i]) - double(host.d[i])) <= tol);
    for (std::size_t i = 0; i < std::size_t(kBatch) * nm1; ++i)
        bad += !(std::fabs(std::fabs(double(got.e[i])) - std::fabs(double(host.e[i]))) <= tol);
    EXPECT_EQ(bad, 0) << "tol " << tol;
    EXPECT_EQ(bitdiff_vs_rep(got.d, n), 0);
}

// ---- host-side: the workspace plan and with_tier --------------------------------------------
using Lower8 = dv::routine::sb2st<Uplo::Lower, 8>;

constexpr auto kCta = dv::sb2st_detail::layout<dv::tier::Cta<64>, std::complex<double>>(37, 5);
static_assert(kCta.c.offset == 0 && kCta.work.offset % 16 == 0 && kCta.work.offset >= kCta.c.count * sizeof(double));
static_assert(kCta.ab.offset % 16 == 0 && kCta.ab.offset >= kCta.work.offset + kCta.work.count * sizeof(std::complex<double>));
static_assert(kCta.size.local_bytes >= kCta.ab.offset + kCta.ab.count * sizeof(std::complex<double>) && kCta.size.global_bytes == 0);
constexpr auto kBlk = dv::sb2st_detail::layout<dv::tier::Blocked<32>, float>(37, 5);
static_assert(kBlk.size.local_bytes == 0 && kBlk.size.global_bytes >= (37 + 5 + 2) * 2 * sizeof(float));
static_assert(dv::sb2st_buffer_size<Uplo::Lower, dv::tier::Cta<64>, float>(10, 1) == dv::WorkspaceSize{});
static_assert(dv::sb2st_buffer_size<Uplo::Lower, dv::tier::Blocked<32>, float>(0, 5) == dv::WorkspaceSize{});
static_assert(std::is_same_v<dv::resolve_tier_t<Lower8, float, dv::tier::upto<4096>>, dv::tier::Cta<4096>>);
static_assert(std::is_same_v<dv::resolve_tier_t<Lower8, float, dv::tier::upto<4097>>, dv::tier::Blocked<32>>);

template <class T> std::string pick(const dv::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        dv::with_tier<Lower8, T>(lim, n, [&]<class Tier>(Tier) { got = dv::sb2st_detail::is_cta_v<Tier> ? "cta" : "blocked"; });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string("throw:") + e.what();
    }
    return got;
}

TEST(DeviceApiSb2stHost, WithTierStraddlesLocalMemoryAndTheCtaBound) {
    const int n = 200;
    const std::size_t need = dv::sb2st_buffer_size<Uplo::Lower, dv::tier::Cta<4096>, float>(n, 8).local_bytes;
    EXPECT_EQ(need, (dv::tier_info<Lower8, dv::tier::Cta<4096>, float>::buffer_size(n).local_bytes));
    dv::DeviceLimits lim{need, {32}, 1024, true};
    EXPECT_EQ(pick<float>(lim, n), "cta");
    lim.local_mem_bytes = need - 1;
    EXPECT_EQ(pick<float>(lim, n), "blocked");
    lim.local_mem_bytes = std::size_t(1) << 40;
    EXPECT_EQ(pick<float>(lim, 4096), "cta");
    EXPECT_EQ(pick<float>(lim, 4097), "blocked");
    lim.sub_group_sizes = {16};
    EXPECT_EQ(pick<float>(lim, n), "throw:sb2st: no device tier admits n=200");
}

}  // namespace
