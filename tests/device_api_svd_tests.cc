// SPDX-License-Identifier: MIT
// device_api_svd_tests: the public device SVD-family ops (include/batchlas/device/lapack/) called
// from a user-style kernel, against the host route each one backs (v22 design C.10).
//   gebrd<tier::Cta<P>>  gebrd_cta is a thin launcher over it: the device call (a WG of 128, four
//                        sub-groups, the tier picked by with_tier) and the host route (a WG of 32)
//                        must agree BIT FOR BIT on A (pads and stride gaps included), d, e, tauq,
//                        taup (inc / stride gaps included); and Q B P^T must reproduce A.
//   with_tier / upto / gebrd_buffer_size  the tier per n at every Cta boundary, each device filter
//                        on its own, the op-named throw, buffer_size == the carved plan.
// Guards (agent guide section 8): ld = n + 3, a matrix stride and vector strides that are not
// ld * n / n, d with inc 2; pads, gaps and outputs poisoned with a large FINITE value the kernel
// would accept; batch 1024 with period-3 items bit-identical to their representative (SLM tiles
// shared by four sub-groups); n straddling every tier width (4/5, 16/17) and the ceiling n = 32;
// the reconstruction is a long-double host product, not the device code; EXPECT_VERIFY only.
// Red set: building with -DBATCHLAS_DEVICE_API_SVD_BREAK drops the per-partition workspace offset
// in the user kernel (all partitions share one tile: the slicing error a caller can make); every
// GebrdCta case must fail on the bit-identity check. (Record the run at the gate: no build here.)
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/gebrd.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace dv = batchlas::device;
using batchlas::verify::Check;

namespace {

template <typename T, Backend B> struct ApiConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
constexpr int kPeriod = 3;
constexpr int kWg = 128;      // the user kernel's work-group: four sub-groups share the SLM

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
template <class R> R uval(std::uint64_t i) { return static_cast<R>(static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0); }
template <class R> R poison() { return static_cast<R>(7.5e29); }  // finite, in range: a read would be used

template <class T> struct DevBuf {
    sycl::queue* q = nullptr;
    T* p = nullptr;
    std::size_t n = 0;
    DevBuf(sycl::queue& qq, const std::vector<T>& h) : q(&qq), p(sycl::malloc_device<T>(std::max<std::size_t>(h.size(), 1), qq)), n(h.size()) {
        if (n) q->memcpy(p, h.data(), n * sizeof(T)).wait();
    }
    DevBuf(const DevBuf&) = delete;
    DevBuf& operator=(const DevBuf&) = delete;
    ~DevBuf() { sycl::free(p, *q); }
    std::vector<T> down() const {
        std::vector<T> h(n);
        if (n) q->memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
};

template <class T> long first_diff(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], sizeof(T)) != 0) return static_cast<long>(i);
    return -1;
}

// One batch of gebrd inputs and output slots with non-natural layouts, pads poisoned.
template <class T> struct GebrdData {
    int n, ld, sa, sd, se, st;
    std::vector<T> a, d, e, tq, tp;
    explicit GebrdData(int nn) : n(nn), ld(nn + 3), sa((nn + 3) * nn + 7), sd(2 * nn + 5), se(std::max(nn - 1, 0) + 4), st(nn + 2) {
        a.assign(std::size_t(sa) * kBatch, poison<T>());
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    a[std::size_t(b) * sa + std::size_t(j) * ld + i] = uval<T>(std::uint64_t(b % kPeriod) * 1000003u + std::uint64_t(j) * n + i);
        d.assign(std::size_t(sd) * kBatch, poison<T>());
        e.assign(std::size_t(se) * kBatch, poison<T>());
        tq.assign(std::size_t(st) * kBatch, poison<T>());
        tp.assign(std::size_t(st) * kBatch, poison<T>());
    }
};

// ||A0 - Q B P^T||_F / ||A0||_F of item b, in long double: Q = H(0)..H(n-1), H(i) = I - tauq_i v v^T
// (v(i) = 1, v(i+1:) = A(i+1:, i)); P = G(0)..G(n-2), G(i) = I - taup_i u u^T (u(i+1) = 1,
// u(i+2:) = A(i, i+2:)); B upper bidiagonal (d, e).
template <class T> double gebrd_residual(const GebrdData<T>& in, const std::vector<T>& a, const std::vector<T>& d,
                                         const std::vector<T>& e, const std::vector<T>& tq, const std::vector<T>& tp, int b) {
    using L = long double;
    const int n = in.n;
    auto A = [&](int i, int j) { return L(a[std::size_t(b) * in.sa + std::size_t(j) * in.ld + i]); };
    std::vector<L> m(std::size_t(n) * n, L(0));  // column-major, ld n
    auto M = [&](int i, int j) -> L& { return m[std::size_t(j) * n + i]; };
    for (int i = 0; i < n; ++i) {
        M(i, i) = L(d[std::size_t(b) * in.sd + std::size_t(i) * 2]);
        if (i + 1 < n) M(i, i + 1) = L(e[std::size_t(b) * in.se + i]);
    }
    std::vector<L> v(n);
    for (int i = n - 2; i >= 0; --i) {  // M := M G(i), G(n-2) first
        const L tau = L(tp[std::size_t(b) * in.st + i]);
        for (int j = 0; j < n; ++j) v[j] = j < i + 1 ? L(0) : (j == i + 1 ? L(1) : A(i, j));
        for (int r = 0; r < n; ++r) {
            L s = 0;
            for (int j = 0; j < n; ++j) s += M(r, j) * v[j];
            for (int j = 0; j < n; ++j) M(r, j) -= tau * s * v[j];
        }
    }
    for (int i = n - 1; i >= 0; --i) {  // M := H(i) M, H(n-1) first
        const L tau = L(tq[std::size_t(b) * in.st + i]);
        for (int j = 0; j < n; ++j) v[j] = j < i ? L(0) : (j == i ? L(1) : A(j, i));
        for (int c = 0; c < n; ++c) {
            L s = 0;
            for (int j = 0; j < n; ++j) s += v[j] * M(j, c);
            for (int j = 0; j < n; ++j) M(j, c) -= tau * s * v[j];
        }
    }
    L num = 0, den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const L a0 = L(in.a[std::size_t(b) * in.sa + std::size_t(j) * in.ld + i]);
            num += (a0 - M(i, j)) * (a0 - M(i, j));
            den += a0 * a0;
        }
    return static_cast<double>(std::sqrt(num) / std::max(std::sqrt(den), std::numeric_limits<L>::min()));
}

template <typename Config> class DeviceApiSvd : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (test_utils::lacks_fp64<T>(sycl_queue(*this->ctx).get_device())) GTEST_SKIP() << "device has no fp64";
    }
    bool sg32() const { return this->ctx->device().supports_sub_group_size(32); }
    sycl::queue& q() { return sycl_queue(*this->ctx); }

    // The user-style kernel: SgpPartition<P> per matrix, the tier visited by with_tier, the
    // workspace a local byte accessor sliced per partition, dead partitions with live = false.
    template <class R> void device_gebrd(GebrdData<R>& g, DevBuf<R>& a, DevBuf<R>& d, DevBuf<R>& e, DevBuf<R>& tq, DevBuf<R>& tp) {
        const MatrixView<R, MatrixFormat::Dense> av(a.p, g.n, g.n, g.ld, g.sa, kBatch);
        const auto akv = av.kernel_view();
        const VectorView<R> dvw(d.p, g.n, kBatch, 2, g.sd), evw(e.p, g.n - 1, kBatch, 1, g.se);
        const VectorView<R> tqv(tq.p, g.n, kBatch, 1, g.st), tpv(tp.p, g.n, kBatch, 1, g.st);
        const int n = g.n;
        dv::with_tier<dv::routine::gebrd, R>(q().get_device(), n, [&]<class Tier>(Tier) {
            constexpr int P = Tier::n_max;
            constexpr int mats = kWg / P;
            const std::size_t lb = dv::gebrd_buffer_size<Tier, R>(n).local_bytes;
            const int groups = (kBatch + mats - 1) / mats;
            q().submit([&](sycl::handler& h) {
                   sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(lb * mats), h);
                   h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * kWg, kWg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                       const auto sg = it.get_sub_group();
                       const dv::SgpPartition<P> ex{sg};
                       const int part = static_cast<int>(sg.get_group_linear_id()) * (32 / P) +
                                        static_cast<int>(ex.part.get_group_linear_id());
                       const int b = static_cast<int>(it.get_group_linear_id()) * mats + part;
                       const bool live = b < kBatch;
                       const int pb = live ? b : 0;
#ifdef BATCHLAS_DEVICE_API_SVD_BREAK
                       const std::size_t slice = 0;
#else
                       const std::size_t slice = std::size_t(part) * lb;
#endif
                       const dv::Workspace w{slm.template get_multi_ptr<sycl::access::decorated::yes>() + slice, nullptr};
                       (void)dv::gebrd<Tier>(ex, it, dv::item(akv, pb, n, n), dv::item(dvw, pb), dv::item(evw, pb), dv::item(tqv, pb),
                                             dv::item(tpv, pb), w, live);
                   });
               }).wait_and_throw();
        });
    }
};

using Types = typename test_utils::backend_types<ApiConfig>::type;
TYPED_TEST_SUITE(DeviceApiSvd, Types);

// gebrd<tier::Cta<P>> from a user kernel vs the gebrd_cta host route, bit for bit, at every tier
// boundary and the ceiling; then Q B P^T against A for the representatives.
TYPED_TEST(DeviceApiSvd, GebrdCtaDeviceCallIsBitIdenticalToTheHostRoute) {
    using T = typename TypeParam::ScalarType;
    constexpr Backend B = TestFixture::B;
    if constexpr (dv::Arith<T>::is_complex) {
        GTEST_SKIP() << "gebrd tier::Cta is real only (the throw is covered by WithTierPicksTheCtaWidthPerN)";
    } else {
        if (!this->sg32()) GTEST_SKIP() << "gebrd tier::Cta needs sub-group size 32";
        for (int n : {1, 4, 5, 16, 17, 32}) {
            SCOPED_TRACE("n " + std::to_string(n));
            GebrdData<T> g(n);
            DevBuf<T> a1(this->q(), g.a), d1(this->q(), g.d), e1(this->q(), g.e), q1(this->q(), g.tq), p1(this->q(), g.tp);
            DevBuf<T> a2(this->q(), g.a), d2(this->q(), g.d), e2(this->q(), g.e), q2(this->q(), g.tq), p2(this->q(), g.tp);
            (void)gebrd_cta<B, T>(*this->ctx, MatrixView<T, MatrixFormat::Dense>(a1.p, n, n, g.ld, g.sa, kBatch),
                                  VectorView<T>(d1.p, n, kBatch, 2, g.sd), VectorView<T>(e1.p, n - 1, kBatch, 1, g.se),
                                  VectorView<T>(q1.p, n, kBatch, 1, g.st), VectorView<T>(p1.p, n, kBatch, 1, g.st));
            this->ctx->wait();
            this->device_gebrd(g, a2, d2, e2, q2, p2);
            const auto ha = a1.down(), hd = d1.down(), he = e1.down(), hq = q1.down(), hp = p1.down();
            const auto ga = a2.down(), gd = d2.down(), ge = e2.down(), gq = q2.down(), gp = p2.down();
            EXPECT_EQ(first_diff(ha, ga), -1) << "A (pads and stride gaps included)";
            EXPECT_EQ(first_diff(hd, gd), -1) << "d (inc 2 gaps included)";
            EXPECT_EQ(first_diff(he, ge), -1) << "e";
            EXPECT_EQ(first_diff(hq, gq), -1) << "tauq";
            EXPECT_EQ(first_diff(hp, gp), -1) << "taup";
            // Items repeat with period kPeriod: each must equal its representative bit for bit.
            long bad = 0;
            for (int b = kPeriod; b < kBatch; ++b) {
                const int r = b % kPeriod;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        bad += std::memcmp(&ga[std::size_t(b) * g.sa + std::size_t(j) * g.ld + i],
                                           &ga[std::size_t(r) * g.sa + std::size_t(j) * g.ld + i], sizeof(T)) != 0;
                for (int i = 0; i < n; ++i)
                    bad += std::memcmp(&gd[std::size_t(b) * g.sd + 2 * i], &gd[std::size_t(r) * g.sd + 2 * i], sizeof(T)) != 0;
            }
            EXPECT_EQ(bad, 0) << "an item differs from its representative (SLM race across sub-groups)";
            double worst = 0;
            for (int b = 0; b < kPeriod; ++b) worst = batchlas::verify::nanmax(worst, gebrd_residual(g, ga, gd, ge, gq, gp, b));
            EXPECT_VERIFY(T, Check::factorization, n, worst);
        }
    }
}

// with_tier picks Cta<4|8|16|32> by n, each filter alone removes the tier, nothing admits n = 33
// (or any complex n), and the throw names gebrd and n.
TYPED_TEST(DeviceApiSvd, WithTierPicksTheCtaWidthPerN) {
    using T = typename TypeParam::ScalarType;
    auto pick = [](const dv::DeviceLimits& lim, int n) {
        int got = 0;
        try {
            dv::with_tier<dv::routine::gebrd, T>(lim, n, [&]<class Tier>(Tier) { got = Tier::n_max; });
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("gebrd: no device tier admits n=" + std::to_string(n)), std::string::npos) << e.what();
            got = -1;
        }
        return got;
    };
    const dv::DeviceLimits big{std::size_t(100) << 10, {16, 32}, 1024, true};
    if constexpr (dv::Arith<T>::is_complex) {
        EXPECT_EQ(pick(big, 4), -1) << "tier::Cta is real only";
    } else {
        const std::pair<int, int> want[] = {{1, 4}, {4, 4}, {5, 8}, {8, 8}, {9, 16}, {16, 16}, {17, 32}, {32, 32}, {33, -1}, {0, -1}};
        for (const auto& [n, p] : want) EXPECT_EQ(pick(big, n), p) << "n " << n;
        constexpr std::size_t t32 = std::size_t(32) * 32 * sizeof(T);
        EXPECT_EQ(pick({t32, {32}, 1024, true}, 17), 32) << "a local memory of exactly one Cta<32> tile";
        EXPECT_EQ(pick({t32 - 1, {32}, 1024, true}, 17), -1) << "one byte short of a Cta<32> tile";
        EXPECT_EQ(pick({t32 - 1, {32}, 1024, true}, 16), 16) << "Cta<16> still fits";
        EXPECT_EQ(pick({std::size_t(32) << 10, {16}, 1024, true}, 4), -1) << "no sub-group size 32";
        EXPECT_EQ(pick({std::size_t(32) << 10, {32}, 16, true}, 4), -1) << "a work-group ceiling below 32";
        constexpr int want_no_fp64 = std::is_same_v<T, double> ? -1 : 4;
        EXPECT_EQ(pick({std::size_t(32) << 10, {32}, 1024, false}, 4), want_no_fp64) << "no fp64";
    }
}

// upto<N> resolves to the smallest admitting width; buffer_size is the carved tile (P x P of T at
// offset 0, rounded to the plan's alignment), the same for every n the tier admits.
TYPED_TEST(DeviceApiSvd, UptoAndBufferSizeMatchThePlan) {
    using T = typename TypeParam::ScalarType;
    using dv::tier::Cta;
    if constexpr (!dv::Arith<T>::is_complex) {
        static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::gebrd, T, dv::tier::upto<4>>, Cta<4>>);
        static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::gebrd, T, dv::tier::upto<5>>, Cta<8>>);
        static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::gebrd, T, dv::tier::upto<17>>, Cta<32>>);
        static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::gebrd, T, Cta<16>>, Cta<16>>);
        auto check = [](auto tier) {
            using Tier = decltype(tier);
            constexpr int P = Tier::n_max;
            constexpr auto plan = dv::gebrd_detail::cta_plan<P, T>();
            static_assert(plan.tile.offset == 0 && plan.tile.count == std::size_t(P) * P);
            for (int n = 1; n <= P; ++n) {
                const auto sz = dv::gebrd_buffer_size<Tier, T>(n);
                EXPECT_EQ(sz.local_bytes, std::size_t(P) * P * sizeof(T)) << "P " << P << " n " << n;
                EXPECT_EQ(sz.global_bytes, 0u);
                EXPECT_EQ(sz, plan.size);
            }
        };
        check(Cta<4>{});
        check(Cta<8>{});
        check(Cta<16>{});
        check(Cta<32>{});
    }
}

}  // namespace
