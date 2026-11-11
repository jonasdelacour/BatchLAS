// SPDX-License-Identifier: MIT
// device_api_sygst_tests: the public in-kernel sygst (include/batchlas/device/lapack/sygst.hh, v22
// design C.4-C.10) called from a user-style kernel, one matrix per work-group, with its tier picked
// by with_tier, against the host route pinned to blocked:nb=32, which runs the same ?sygs2 body
// (sygs2_group, ipw blocks per group) for n <= 32: BIT-IDENTICAL, plus a long-double reference
// (the bit-identity alone is self-referential: a break in the shared body moves both sides).
// Guards (agent-guide section 8): the pin is reached (coverage `reached` row); ld = n + 3 for A,
// n + 2 for B, batch strides off ld * n; complex data with Im(diag) = 1e3 poisoned in A and B
// (the op must read both diagonals as real) and a result diagonal that must be exactly real;
// NaN in A's and B's other triangles and pads, which must stay bit-for-bit (A) and B whole; batch
// 1024 with period-3 data, every item bit-identical to its representative; n at every tier
// ceiling (8, 16, 32) and one past (9, 17), n = 1, and n = 5 under tier::upto<32> (a tier wider
// than n); itypes 1, 2, 3 and both uplos; with_tier's pick at every boundary, a local-memory
// filter straddled (complex<double> n = 31 fits 32 KB, n = 32 does not) and the throw past 32;
// buffer_size against the layout written out by hand; -2 / -3 returns that leave A untouched.
// tier::Blocked<NB> (any n, LAPACK's schedule in one work-group of 256 on the in-group L3): against
// the host route pinned to blocked:nb=32 (the same step order through separate kernels) to the
// factorization tolerance, and against the long-double reference; n = 33, 63, 64, 65 (NB
// multiples +-1), Blocked<16> over three blocks (16, 16, 1), Blocked<32> at n = 32 (one block)
// bit-identical to Cta<32>; the global workspace poisoned with NaN bytes (H and W are written
// before they are read); B's diagonal real (potrf's factor: the trsm / trmm steps read it as
// stored, on both routes, as LAPACK's do), A's Im(diag) still poisoned. Cta<N> on 64 and 256 items
// (n = 13, 31) bit-identical to its wg_size launch: red on Metal before sygs2_one.
// Deliberate breaks (to record when first run; expected red sets): (1) sygst() passing
// `UL != Uplo::Lower` to sygs2_group: every BitIdentical case red (device vs host); (2) in
// sygs2_group's first sweep, F stored with `fat<Inverse>(fs, row, c)` in the Upper branch: the
// Upper reference checks red on both paths, the bit-identity green (the shared body); (3) in
// sygst_detail::blocked, itype 1 Lower, the second gemm dropped: Blocked itype 1 Lower red
// against host and reference at every n > 32, Upper and n <= 32 green (recorded on sm_89, all
// four types: Blocked<32> n = 33, 63, 64, 65 and Blocked<16> n = 33 red, nothing else).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>
#include <tests/test_support/coverage_rows.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include <batchlas/blas/functions/sygst.hh>
#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/sygst.hh>
#include <src/ops/sygst/choice.hh>
#include <src/queue.hh>
#include <src/select/select.hh>

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;
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
template <class T> T nan_of() {
    const real_of<T> q = std::numeric_limits<real_of<T>>::quiet_NaN();
    if constexpr (is_cplx<T>) return T(q, q);
    else return q;
}
template <class T> T with_imag(real_of<T> re, real_of<T> im) {
    if constexpr (is_cplx<T>) return T(re, im);
    else return re;
}
template <class V> V conj_h(V v) {
    if constexpr (std::is_arithmetic_v<V>) return v;
    else return std::conj(v);
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
bool in_tri(Uplo ul, int i, int j) { return ul == Uplo::Lower ? i >= j : i <= j; }

// One problem: A and B with ld, batch stride and poison as the header says; kPeriod representatives
// copied to every item b from b % kPeriod. b_im: Im of B's diagonal (0 for the Blocked tier).
template <class T> struct Problem {
    int n, lda, ldb;
    std::ptrdiff_t sa, sb;
    Uplo ul;
    std::vector<T> a, b;
    Problem(int n_, Uplo ul_, real_of<T> b_im = real_of<T>(1e3))
        : n(n_), lda(n_ + 3), ldb(n_ + 2), sa(std::ptrdiff_t(lda) * n_ + 5), sb(std::ptrdiff_t(ldb) * n_ + 1), ul(ul_) {
        a.assign(std::size_t(sa) * kBatch, nan_of<T>());
        b.assign(std::size_t(sb) * kBatch, nan_of<T>());
        for (int r = 0; r < kPeriod; ++r)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    if (!in_tri(ul, i, j)) continue;
                    const std::uint64_t s = std::uint64_t(r) * 100003u + std::uint64_t(i + j * n);
                    T& ea = a[std::size_t(r) * sa + i + std::size_t(j) * lda];
                    T& eb = b[std::size_t(r) * sb + i + std::size_t(j) * ldb];
                    ea = i == j ? with_imag<T>(std::real(val<T>(s)), real_of<T>(1e3)) : val<T>(s);
                    eb = i == j ? with_imag<T>(real_of<T>(1.5) + real_of<T>(0.5) * std::real(val<T>(s + 7)), b_im)
                                : val<T>(s + 7) * real_of<T>(0.5 / n);
                }
        for (int r = kPeriod; r < kBatch; ++r) {
            std::copy_n(a.data() + std::size_t(r % kPeriod) * sa, sa, a.data() + std::size_t(r) * sa);
            std::copy_n(b.data() + std::size_t(r % kPeriod) * sb, sb, b.data() + std::size_t(r) * sb);
        }
    }
};

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible() ? std::optional<Queue>(std::in_place, Device::default_device())
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
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

// The user's kernel: one matrix per work-group of the tier's wg_size, the op's info per item; the
// global workspace (Blocked) is batch slices of global_bytes, poisoned with NaN bytes. Wg > 0
// launches a wider group than the tier's wg_size (Cta: any group of at least N items).
template <int Itype, Uplo UL, class Tier, int Wg = 0, class T>
void device_call(Dev& dev, T* A, const T* B, const Problem<T>& p, int* info) {
    using Tr = device::resolve_tier_t<device::routine::sygst<Itype, UL>, T, Tier>;  // Tier may be tier::upto
    constexpr int WG = Wg > 0 ? Wg : device::tier_info<device::routine::sygst<Itype, UL>, Tr, T>::wg_size;
    const device::WorkspaceSize ws = device::sygst_buffer_size<Itype, UL, Tier, T>(p.n);
    std::byte* gws = nullptr;
    if (ws.global_bytes) {
        gws = sycl::malloc_device<std::byte>(ws.global_bytes * kBatch, dev.q());
        dev.q().memset(gws, 0xff, ws.global_bytes * kBatch).wait();
    }
    const int n = p.n, lda = p.lda, ldb = p.ldb;
    const std::ptrdiff_t sa = p.sa, sb = p.sb;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(std::max<std::size_t>(ws.local_bytes, 16)), h);
           h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
               const device::WorkGroup<WG, 32> ex{it};
               const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
               const int r = device::sygst<Itype, UL, Tier>(ex, device::mat(A + b * sa, n, n, lda), device::mat(B + b * sb, n, n, ldb),
                                                         device::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(),
                                                                       device::global_slice(gws, ws, b)});
               if (ex.leader()) info[b] = r;
           });
       }).wait_and_throw();
    if (gws) sycl::free(gws, dev.q());
}

template <class Tier> std::string tier_name() {
    if constexpr (Tier::n_max < 0) return "Blocked<" + std::to_string(Tier::nb) + ">";
    else return "Cta<" + std::to_string(Tier::n_max) + ">";
}
using CtaTiers = device::tier::list<device::tier::Cta<8>, device::tier::Cta<16>, device::tier::Cta<32>>;

// C = F^-1 A F^-H (itype 1) or F^H A F (2, 3) in long double, F = L or U^H from B's ul triangle.
template <class T> std::vector<LD<T>> reference(const Problem<T>& p, int r, int itype) {
    const int n = p.n;
    using W = LD<T>;
    std::vector<W> a(std::size_t(n) * n), f(std::size_t(n) * n, W(0));
    auto at = [&](std::vector<W>& m, int i, int j) -> W& { return m[std::size_t(i) + std::size_t(j) * n]; };
    const T* pa = p.a.data() + std::size_t(r) * p.sa;
    const T* pb = p.b.data() + std::size_t(r) * p.sb;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const bool own = in_tri(p.ul, i, j);
            const W x = W(own ? pa[i + std::size_t(j) * p.lda] : pa[j + std::size_t(i) * p.lda]);
            at(a, i, j) = i == j ? W(std::real(x)) : own ? x : conj_h(x);
            if (i > j) at(f, i, j) = p.ul == Uplo::Lower ? W(pb[i + std::size_t(j) * p.ldb]) : conj_h(W(pb[j + std::size_t(i) * p.ldb]));
            if (i == j) at(f, i, i) = W(std::real(pb[i + std::size_t(i) * p.ldb]));
        }
    auto lower_solve = [&](std::vector<W>& x) {  // x := F^-1 x, column by column
        for (int c = 0; c < n; ++c)
            for (int i = 0; i < n; ++i) {
                W s = at(x, i, c);
                for (int k = 0; k < i; ++k) s -= at(f, i, k) * at(x, k, c);
                at(x, i, c) = s / at(f, i, i);
            }
    };
    auto herm_t = [&](const std::vector<W>& x) {
        std::vector<W> y(x.size());
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) y[std::size_t(i) + std::size_t(j) * n] = conj_h(x[std::size_t(j) + std::size_t(i) * n]);
        return y;
    };
    if (itype == 1) {
        lower_solve(a);         // X = F^-1 A
        auto y = herm_t(a);     // X^H
        lower_solve(y);         // F^-1 X^H = C^H
        return herm_t(y);
    }
    std::vector<W> af(std::size_t(n) * n, W(0)), c(std::size_t(n) * n, W(0));
    for (int j = 0; j < n; ++j)
        for (int k = j; k < n; ++k)
            for (int i = 0; i < n; ++i) at(af, i, j) += at(a, i, k) * at(f, k, j);  // A F
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            for (int k = i; k < n; ++k) at(c, i, j) += conj_h(at(f, k, i)) * at(af, k, j);  // F^H (A F)
    return c;
}

template <class T> class DeviceApiSygstTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "the Cta tiers launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }

    // Host route pinned to blocked:nb=32 (one diagonal-block launch for n <= 32); false when the
    // queue cannot run it (blocked needs a GPU backend).
    bool host_call(T* A, const T* B, const Problem<T>& p, int itype, std::string* reached) {
        const ops::sygst::SygstChoice c{ops::sygst::Blocked{32}};
        const MatrixView<T, MatrixFormat::Dense> Av(A, p.n, p.n, p.lda, static_cast<int>(p.sa), kBatch);
        const MatrixView<T, MatrixFormat::Dense> Bv(const_cast<T*>(B), p.n, p.n, p.ldb, static_cast<int>(p.sb), kBatch);
        bool ran = true;
        const auto rows = test_support::reached_rows([&] {
            const select::ScopedPin<ops::sygst::SygstChoice> pin("sygst", c);
            try {
                const std::size_t need = sygst_buffer_size(*dev.ctx, Av, Bv, itype, p.ul);
                std::byte* ws = need ? sycl::malloc_device<std::byte>(need, dev.q()) : nullptr;
                (void)sygst(*dev.ctx, Av, Bv, itype, p.ul, Span<std::byte>(ws, need));
                dev.ctx->wait();
                if (ws) sycl::free(ws, dev.q());
            } catch (const std::exception&) {
                ran = false;
            }
        });
        for (const auto& r : rows)
            if (r.op == "sygst") *reached = r.spelling;
        EXPECT_TRUE(!ran || *reached == select::to_string(c)) << "pinned " << select::to_string(c) << ", reached '" << *reached << "'";
        return ran;
    }

    template <int Itype, Uplo UL> void bit_identical(int n) {
        Problem<T> p(n, UL);
        const std::string at = "itype " + std::to_string(Itype) + (UL == Uplo::Lower ? " L" : " U") + " n " + std::to_string(n);
        T* Ah = dev.up(p.a);
        T* Ad = dev.up(p.a);
        const T* B = dev.up(p.b);
        int* info = dev.up(std::vector<int>(kBatch, -99));
        std::string reached;
        if (!host_call(Ah, B, p, Itype, &reached)) GTEST_SKIP() << "the pinned blocked route cannot run on this queue";
        std::string tier = "none";
        device::with_tier<device::routine::sygst<Itype, UL>, T, CtaTiers>(dev.q().get_device(), n, [&]<class Tier>(Tier) {
            tier = tier_name<Tier>();
            device_call<Itype, UL, Tier>(dev, Ad, B, p, info);
        });
        EXPECT_EQ(tier, n <= 8 ? "Cta<8>" : n <= 16 ? "Cta<16>" : "Cta<32>") << at;
        const auto gh = dev.down(Ah, p.a.size()), gd = dev.down(Ad, p.a.size());
        const auto gi = dev.down(info, kBatch);
        EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch) << at << ": info";
        EXPECT_EQ(std::memcmp(gh.data(), gd.data(), gh.size() * sizeof(T)), 0) << at << ": device call vs pinned host route";
        EXPECT_EQ(std::memcmp(dev.down(B, p.b.size()).data(), p.b.data(), p.b.size() * sizeof(T)), 0) << at << ": B written";
        check_result(p, gd, Itype, at);
    }

    // tier::Blocked<NB> against the host blocked:nb=32 route (tolerance; skipped where that route
    // cannot run) and the reference; info 0 on every item, B untouched.
    template <int Itype, Uplo UL, int NB = 32> void blocked_vs_host(int n) {
        Problem<T> p(n, UL, real_of<T>(0));
        const std::string at = "Blocked<" + std::to_string(NB) + "> itype " + std::to_string(Itype) + (UL == Uplo::Lower ? " L" : " U") +
                               " n " + std::to_string(n);
        T* Ah = dev.up(p.a);
        T* Ad = dev.up(p.a);
        const T* B = dev.up(p.b);
        int* info = dev.up(std::vector<int>(kBatch, -99));
        std::string tier = "none";
        device::with_tier<device::routine::sygst<Itype, UL>, T, device::tier::list<device::tier::Blocked<NB>>>(dev.q().get_device(), n, [&]<class Tier>(Tier) {
            tier = tier_name<Tier>();
            device_call<Itype, UL, Tier>(dev, Ad, B, p, info);
        });
        ASSERT_EQ(tier, "Blocked<" + std::to_string(NB) + ">") << at;
        const auto gd = dev.down(Ad, p.a.size());
        const auto gi = dev.down(info, kBatch);
        EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch) << at << ": info";
        EXPECT_EQ(std::memcmp(dev.down(B, p.b.size()).data(), p.b.data(), p.b.size() * sizeof(T)), 0) << at << ": B written";
        check_result(p, gd, Itype, at);
        std::string reached;
        if (!host_call(Ah, B, p, Itype, &reached)) return;  // blocked:nb=32 needs a GPU backend
        const auto gh = dev.down(Ah, p.a.size());
        long double num = 0, den = 0;
        for (int r = 0; r < kPeriod; ++r)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    if (!in_tri(p.ul, i, j)) continue;
                    const std::size_t e = std::size_t(r) * p.sa + i + std::size_t(j) * p.lda;
                    num += std::norm(LD<T>(LD<T>(gd[e]) - LD<T>(gh[e]))), den += std::norm(LD<T>(gh[e]));
                }
        const double diff = static_cast<double>(std::sqrt(num / std::max<long double>(den, std::numeric_limits<long double>::min())));
        EXPECT_VERIFY(T, Check::factorization, n, std::isnan(diff) ? std::numeric_limits<double>::infinity() : diff) << at << ": vs host";
    }

    // Untouched poison, twins, real diagonal and the reference, on the device call's result.
    void check_result(const Problem<T>& p, const std::vector<T>& got, int itype, const std::string& at) {
        int touched = 0, twins = 0, imag = 0;
        for (std::size_t b = 0; b < std::size_t(kBatch); ++b)
            for (std::size_t e = 0; e < std::size_t(p.sa); ++e) {
                const std::size_t k = b * p.sa + e;
                const int i = static_cast<int>(e % p.lda), j = static_cast<int>(e / p.lda);
                const bool stored = j < p.n && i < p.n && in_tri(p.ul, i, j);
                if (!stored) touched += !same_bits(got[k], p.a[k]);
                if (b >= std::size_t(kPeriod)) twins += !same_bits(got[k], got[(b % kPeriod) * p.sa + e]);
                if constexpr (is_cplx<T>)
                    if (stored && i == j) imag += std::imag(got[k]) != real_of<T>(0);
            }
        EXPECT_EQ(touched, 0) << at << ": the other triangle or a pad written";
        EXPECT_EQ(twins, 0) << at << ": items differ from their representative";
        EXPECT_EQ(imag, 0) << at << ": a result diagonal not real";
        double worst = 0;
        for (int r = 0; r < kPeriod; ++r) {
            const auto c = reference(p, r, itype);
            long double num = 0, den = 0;
            for (int j = 0; j < p.n; ++j)
                for (int i = 0; i < p.n; ++i) {
                    if (!in_tri(p.ul, i, j)) continue;
                    const LD<T> ref = c[std::size_t(i) + std::size_t(j) * p.n];
                    const LD<T> g = LD<T>(got[std::size_t(r) * p.sa + i + std::size_t(j) * p.lda]);
                    num += std::norm(LD<T>(g - ref)), den += std::norm(ref);
                }
            const long double e = std::sqrt(num / std::max<long double>(den, std::numeric_limits<long double>::min()));
            worst = std::max(worst, std::isnan(static_cast<double>(e)) ? std::numeric_limits<double>::infinity() : static_cast<double>(e));
        }
        EXPECT_VERIFY(T, Check::factorization, p.n, worst) << at;
    }

    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceApiSygstTest, Types);

TYPED_TEST(DeviceApiSygstTest, BitIdenticalToPinnedHostRouteItype1) {
    for (int n : {1, 8, 9, 16, 17, 31, 32}) {
        this->template bit_identical<1, Uplo::Lower>(n);
        this->template bit_identical<1, Uplo::Upper>(n);
    }
}

TYPED_TEST(DeviceApiSygstTest, BitIdenticalToPinnedHostRouteItype2And3) {
    for (int n : {5, 16, 32}) {
        this->template bit_identical<2, Uplo::Lower>(n);
        this->template bit_identical<2, Uplo::Upper>(n);
    }
    this->template bit_identical<3, Uplo::Upper>(13);
}

TYPED_TEST(DeviceApiSygstTest, BlockedMatchesHostBlockedRouteItype1) {
    for (int n : {33, 63, 64, 65}) {
        this->template blocked_vs_host<1, Uplo::Lower>(n);
        this->template blocked_vs_host<1, Uplo::Upper>(n);
    }
}

TYPED_TEST(DeviceApiSygstTest, BlockedMatchesHostBlockedRouteItype2And3) {
    for (int n : {33, 65}) {
        this->template blocked_vs_host<2, Uplo::Lower>(n);
        this->template blocked_vs_host<2, Uplo::Upper>(n);
    }
    this->template blocked_vs_host<3, Uplo::Upper>(64);
}

// Blocked<16> (the 32 KB fallback) over blocks of 16, 16 and 1; its host comparison is still nb = 32.
TYPED_TEST(DeviceApiSygstTest, Blocked16OverThreeBlocks) {
    this->template blocked_vs_host<1, Uplo::Lower, 16>(33);
    this->template blocked_vs_host<2, Uplo::Upper, 16>(33);
}

// Blocked<32> at n = 32 is one ?sygs2 block, the Cta<32> body on a 256-item group: same bits.
TYPED_TEST(DeviceApiSygstTest, BlockedOneBlockIsBitIdenticalToCta) {
    using T = TypeParam;
    Problem<T> p(32, Uplo::Upper);
    T* Ac = this->dev.up(p.a);
    T* Ab = this->dev.up(p.a);
    const T* B = this->dev.up(p.b);
    int* info = this->dev.up(std::vector<int>(kBatch, -99));
    device_call<1, Uplo::Upper, device::tier::Cta<32>>(this->dev, Ac, B, p, info);
    device_call<1, Uplo::Upper, device::tier::Blocked<32>>(this->dev, Ab, B, p, info);
    const auto gc = this->dev.down(Ac, p.a.size()), gb = this->dev.down(Ab, p.a.size());
    EXPECT_EQ(std::memcmp(gc.data(), gb.data(), gc.size() * sizeof(T)), 0);
    const auto gi = this->dev.down(info, kBatch);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch);
}

// Cta<N> on a group of several sub-groups (64, 256 items) with n < 32: same bits as on wg_size. On
// Metal the ?sygs2 block ran on the whole group before, and a partly active sub-group freed the
// others early (every column but those of sub-group 0 wrong).
TYPED_TEST(DeviceApiSygstTest, CtaOnWiderGroupIsBitIdentical) {
    using T = TypeParam;
    Problem<T> p(13, Uplo::Lower);
    T* A0 = this->dev.up(p.a);
    T* A1 = this->dev.up(p.a);
    const T* B = this->dev.up(p.b);
    int* info = this->dev.up(std::vector<int>(kBatch, -99));
    device_call<1, Uplo::Lower, device::tier::Cta<16>>(this->dev, A0, B, p, info);
    device_call<1, Uplo::Lower, device::tier::Cta<16>, 64>(this->dev, A1, B, p, info);
    const auto g0 = this->dev.down(A0, p.a.size()), g1 = this->dev.down(A1, p.a.size());
    EXPECT_EQ(std::memcmp(g0.data(), g1.data(), g0.size() * sizeof(T)), 0) << "Cta<16> n 13 on 64 items";
    this->check_result(p, g1, 1, "Cta<16> n 13 on 64 items");
    Problem<T> q(31, Uplo::Upper);
    T* B0 = this->dev.up(q.a);
    T* B1 = this->dev.up(q.a);
    const T* Bq = this->dev.up(q.b);
    device_call<2, Uplo::Upper, device::tier::Cta<32>>(this->dev, B0, Bq, q, info);
    device_call<2, Uplo::Upper, device::tier::Cta<32>, 256>(this->dev, B1, Bq, q, info);
    const auto h0 = this->dev.down(B0, q.a.size()), h1 = this->dev.down(B1, q.a.size());
    EXPECT_EQ(std::memcmp(h0.data(), h1.data(), h0.size() * sizeof(T)), 0) << "Cta<32> n 31 on 256 items";
    this->check_result(q, h1, 2, "Cta<32> n 31 on 256 items");
    const auto gi = this->dev.down(info, kBatch);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), 0), kBatch);
}

// tier::upto<32> on n = 5: Cta<32> (a 32-row fragment) over a 5 x 5 matrix, same bits as Cta<8>.
TYPED_TEST(DeviceApiSygstTest, WiderTierThanNIsBitIdentical) {
    using T = TypeParam;
    using Up = device::resolve_tier_t<device::routine::sygst<1, Uplo::Lower>, T, device::tier::upto<32>>;
    static_assert(std::is_same_v<Up, device::tier::Cta<32>>);
    Problem<T> p(5, Uplo::Lower);
    T* A8 = this->dev.up(p.a);
    T* A32 = this->dev.up(p.a);
    const T* B = this->dev.up(p.b);
    int* info = this->dev.up(std::vector<int>(kBatch, -99));
    device_call<1, Uplo::Lower, device::tier::Cta<8>>(this->dev, A8, B, p, info);
    device_call<1, Uplo::Lower, device::tier::upto<32>>(this->dev, A32, B, p, info);
    const auto g8 = this->dev.down(A8, p.a.size()), g32 = this->dev.down(A32, p.a.size());
    EXPECT_EQ(std::memcmp(g8.data(), g32.data(), g8.size() * sizeof(T)), 0);
    this->check_result(p, g32, 1, "upto<32> n 5");
}

// n > N returns -2 and B smaller than n x n returns -3, on every item, A untouched.
TYPED_TEST(DeviceApiSygstTest, BadArgumentsReturnMinusIAndTouchNothing) {
    using T = TypeParam;
    Problem<T> p(9, Uplo::Lower);
    T* A = this->dev.up(p.a);
    const T* B = this->dev.up(p.b);
    int* info = this->dev.up(std::vector<int>(kBatch, -99));
    device_call<1, Uplo::Lower, device::tier::Cta<8>>(this->dev, A, B, p, info);
    auto gi = this->dev.down(info, kBatch);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), -2), kBatch) << "n = 9 on Cta<8>";
    EXPECT_EQ(std::memcmp(this->dev.down(A, p.a.size()).data(), p.a.data(), p.a.size() * sizeof(T)), 0);
    const int n = p.n, lda = p.lda, ldb = p.ldb;
    const std::ptrdiff_t sa = p.sa, sb = p.sb;
    const device::WorkspaceSize ws = device::sygst_buffer_size<1, Uplo::Lower, device::tier::Cta<16>, T>(n);
    this->dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(ws.local_bytes), h);
           h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * 32, 32), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
               const device::WorkGroup<32, 32> ex{it};
               const std::ptrdiff_t b = static_cast<std::ptrdiff_t>(it.get_group_linear_id());
               const int r = device::sygst<1, Uplo::Lower, device::tier::Cta<16>>(
                   ex, device::mat(A + b * sa, n, n, lda), device::mat(B + b * sb, n - 1, n, ldb),
                   device::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(), nullptr});
               if (ex.leader()) info[b] = r;
           });
       }).wait_and_throw();
    gi = this->dev.down(info, kBatch);
    EXPECT_EQ(std::count(gi.begin(), gi.end(), -3), kBatch) << "B of n - 1 rows";
    EXPECT_EQ(std::memcmp(this->dev.down(A, p.a.size()).data(), p.a.data(), p.a.size() * sizeof(T)), 0);
}

// Host side only: with_tier at every boundary, the local-memory filter straddled (Cta<32>, then
// Blocked<32> -> Blocked<16>), the throws, and buffer_size against the layout written out (F: n
// lines of whole 16-byte packs plus one; S: ld n + 1; Blocked: max(that block, the L3 staging)
// local, H then W global).
TEST(DeviceApiSygstTiers, WithTierBoundariesFiltersAndBufferSize) {
    using R = device::routine::sygst<1, Uplo::Lower>;
    auto pick = [](const device::DeviceLimits& lim, int n) {
        std::string got = "none";
        try {
            device::with_tier<R, std::complex<double>>(lim, n, [&]<class Tier>(Tier) { got = tier_name<Tier>(); });
        } catch (const batchlas::invalid_argument& e) {
            got = std::string("throw:") + e.what();
        }
        return got;
    };
    const device::DeviceLimits big{std::size_t(48) << 10, {32}, 1024, true}, metal{std::size_t(32) << 10, {32}, 1024, true};
    using P = std::pair<int, const char*>;
    for (auto [n, want] : {P{0, "Cta<8>"}, P{8, "Cta<8>"}, P{9, "Cta<16>"}, P{16, "Cta<16>"}, P{17, "Cta<32>"}, P{32, "Cta<32>"},
                           P{33, "Blocked<32>"}, P{1000, "Blocked<32>"}})
        EXPECT_EQ(pick(big, n), want) << "n " << n;
    EXPECT_EQ(pick(metal, 31), "Cta<32>");
    EXPECT_EQ(pick(metal, 32), "Blocked<16>");
    EXPECT_EQ(pick(metal, 33), "Blocked<16>");
    EXPECT_EQ(pick(device::DeviceLimits{std::size_t(4) << 10, {32}, 1024, true}, 40), "throw:sygst: no device tier admits n=40");
    EXPECT_EQ(pick(device::DeviceLimits{std::size_t(48) << 10, {32}, 128, true}, 40), "throw:sygst: no device tier admits n=40");
    EXPECT_EQ(pick(device::DeviceLimits{std::size_t(48) << 10, {16}, 1024, true}, 4), "throw:sygst: no device tier admits n=4");
    using UpTo = device::resolve_tier_t<R, float, device::tier::upto<64>>;
    static_assert(std::is_same_v<UpTo, device::tier::Blocked<32>>);
    auto blocked_hand = [](int n, int nb, int itype, std::size_t elem) {
        const int kb = std::min(n, nb);
        const std::size_t v = elem >= 16 ? 1 : 16 / elem, ldf = (std::size_t(kb) + v - 1) / v * v + v;
        const std::size_t blk = std::size_t(kb) * (ldf + std::size_t(kb) + 1) * elem;
        const std::size_t stage = (std::size_t(33) * 8 + std::size_t(9) * 32) * elem;  // 32 x 32 x 8 tile: A (ld 33), B (ld 9)
        const auto r16 = [](std::size_t b) { return (b + 15) / 16 * 16; };
        std::size_t g = 0;
        if (n > nb) g = r16(std::size_t(nb) * nb * elem) + (itype != 1 ? std::size_t(n) * nb * elem : 0);
        return std::pair<std::size_t, std::size_t>{r16(std::max(blk, stage)), r16(g)};
    };
    for (int n : {1, 16, 32, 33, 65, 100}) {
        const auto f1 = device::sygst_buffer_size<1, Uplo::Lower, device::tier::Blocked<32>, float>(n);
        const auto c2 = device::sygst_buffer_size<2, Uplo::Upper, device::tier::Blocked<16>, std::complex<float>>(n);
        const auto z3 = device::sygst_buffer_size<3, Uplo::Lower, device::tier::Blocked<32>, std::complex<double>>(n);
        EXPECT_EQ(std::pair(f1.local_bytes, f1.global_bytes), blocked_hand(n, 32, 1, 4)) << n;
        EXPECT_EQ(std::pair(c2.local_bytes, c2.global_bytes), blocked_hand(n, 16, 2, 8)) << n;
        EXPECT_EQ(std::pair(z3.local_bytes, z3.global_bytes), blocked_hand(n, 32, 3, 16)) << n;
    }
    auto hand = [](int n, std::size_t elem) {
        const std::size_t v = elem >= 16 ? 1 : 16 / elem, ldf = (std::size_t(n) + v - 1) / v * v + v;
        const std::size_t bytes = std::size_t(n) * (ldf + std::size_t(n) + 1) * elem;
        return (bytes + 15) / 16 * 16;
    };
    for (int n : {1, 3, 8, 13, 31, 32}) {
        EXPECT_EQ((device::sygst_buffer_size<1, Uplo::Lower, device::tier::Cta<32>, float>(n).local_bytes), hand(n, 4)) << n;
        EXPECT_EQ((device::sygst_buffer_size<2, Uplo::Upper, device::tier::Cta<32>, std::complex<float>>(n).local_bytes), hand(n, 8)) << n;
        EXPECT_EQ((device::sygst_buffer_size<1, Uplo::Upper, device::tier::Cta<32>, std::complex<double>>(n).local_bytes), hand(n, 16)) << n;
        EXPECT_EQ((device::sygst_buffer_size<1, Uplo::Lower, device::tier::Cta<32>, double>(n).global_bytes), 0u) << n;
    }
}

}  // namespace
