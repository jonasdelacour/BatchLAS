// SPDX-License-Identifier: MIT
// device_api_svd_tests: the public device SVD-family ops (include/batchlas/device/lapack/) called
// from a user-style kernel, against the host route each one backs (v22 design C.10).
//   gebrd<tier::Cta<P>>  gebrd_cta is a thin launcher over it: the device call (a WG of 128, four
//                        sub-groups, the tier picked by with_tier) and the host route (a WG of 32)
//                        must agree BIT FOR BIT on A (pads and stride gaps included), d, e, tauq,
//                        taup (inc / stride gaps included); and Q B P^T must reproduce A.
//   with_tier / upto / gebrd_buffer_size  the tier per n at every Cta boundary, each device filter
//                        on its own, the op-named throw, buffer_size == the carved plan.
//   tier::Blocked (one matrix per work-group of 256, the caller's global slice per matrix):
//     bdsqr              vs the host bdsqr (bdsqr<Thread>, one item per matrix) on uniform, graded with
//                        an interior and a bottom zero diagonal (both zero-shift chases) and NaN
//                        items: info per item, s, and U diag(s) V^H == B from identity seeds;
//                        the Thread executor (one item per matrix) BIT FOR BIT the host's.
//     bdsdc             vs the host bdsdc (host stedc; the device runs the in-group one at 2n) on
//                        uniform, graded and exact-zero-d items (repair): info, s, U S V^H == B,
//                        orthogonality, the values-only s bit for bit the with-vectors s.
//     gebrd + ormbr      Q (ormbr 'Q' Left on I) B P^H (ormbr 'P' Right ConjTrans on I) == A, Q and
//                        P^H unitary, the bidiagonal's singular values vs the host gebrd_blocked's,
//                        m >= n across NB = 16.
//     gesvd              vs the host route pinned to blocked with BIDIAG=bdsqr: info per item, S,
//                        U S V^H == A, U and V^H unitary; m > n, m < n (the adjoint path), All /
//                        Thin / None, pads of every output unchanged.
//     with_tier          gesvd: Blocked<16> for every n >= 1, each device filter, the throw;
//                        gesvd_buffer_size == the driver plan, its local bytes independent of m, n.
// Guards (agent guide section 8): ld = n + 3, a matrix stride and vector strides that are not
// ld * n / n, d with inc 2; pads, gaps and outputs poisoned with a large FINITE value the kernel
// would accept; batch 1024 with period-3 items bit-identical to their representative (SLM tiles
// shared by four sub-groups); n straddling every tier width (4/5, 16/17) and the ceiling n = 32;
// the reconstruction is a long-double host product, not the device code; EXPECT_VERIFY only.
// Red set: building with -DBATCHLAS_DEVICE_API_SVD_BREAK shifts the workspace of odd partitions
// by one slice in the user kernel (partitions 1 and 2 then share a tile: the slicing error a
// caller can make); every GebrdCta case must fail on the bit-identity check (gate: all six n).
// -DBATCHLAS_DEVICE_API_SVD_BLOCKED_BREAK moves every odd work-group's global slice half a slice
// down (neighbours overlap: the per-matrix offset a caller can get wrong); the Blocked gesvd and
// gebrd cases must fail their period / residual checks (bdsqr's is n-1 rotations, so n >= 3; on
// CUDA bdsqr and bdsdc then fault on a neighbour's indices: the run aborts, also red).
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
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/bdsdc.hh>
#include <batchlas/device/lapack/bdsqr.hh>
#include <batchlas/device/lapack/gebrd.hh>
#include <batchlas/device/lapack/gesvd.hh>
#include <batchlas/device/lapack/ormbr.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/util/env.hh>
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
    // workspace the work-group's local byte accessor (gebrd slices it per partition), dead
    // partitions with live = false.
    template <class R> void device_gebrd(GebrdData<R>& g, DevBuf<R>& a, DevBuf<R>& d, DevBuf<R>& e, DevBuf<R>& tq, DevBuf<R>& tp) {
        const MatrixView<R, MatrixFormat::Dense> av(a.p, g.n, g.n, g.ld, g.sa, kBatch);
        const auto akv = av.kernel_view();
        const VectorView<R> dvw(d.p, g.n, kBatch, 2, g.sd), evw(e.p, g.n - 1, kBatch, 1, g.se);
        const VectorView<R> tqv(tq.p, g.n, kBatch, 1, g.st), tpv(tp.p, g.n, kBatch, 1, g.st);
        const int n = g.n;
        using CtaTiers = dv::tier::list<dv::tier::Cta<4>, dv::tier::Cta<8>, dv::tier::Cta<16>, dv::tier::Cta<32>>;
        dv::with_tier<dv::routine::gebrd, R, CtaTiers>(q().get_device(), n, [&]<class Tier>(Tier) {
            constexpr int P = Tier::n_max;
            constexpr int mats = kWg / P;
            const std::size_t lb = dv::gebrd_buffer_size<Tier, R>(n).local_bytes;
            const int groups = (kBatch + mats - 1) / mats;
            q().submit([&](sycl::handler& h) {
                   sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(lb * (mats + 1)), h);
                   h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * kWg, kWg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                       const auto sg = it.get_sub_group();
                       const dv::SgpPartition<P> ex{sg};
                       const int part = static_cast<int>(sg.get_group_linear_id()) * (32 / P) +
                                        static_cast<int>(ex.part.get_group_linear_id());
                       const int b = static_cast<int>(it.get_group_linear_id()) * mats + part;
                       const bool live = b < kBatch;
                       const int pb = live ? b : 0;
#ifdef BATCHLAS_DEVICE_API_SVD_BREAK
                       const std::size_t slice = (part & 1) ? lb : 0;
#else
                       const std::size_t slice = 0;
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
    } else if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "the gebrd_cta host route has no CPU image (batchlas_extensions_cta is GPU-only)";
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

// with_tier picks Cta<4|8|16|32> by n and Blocked<16> (reported as 1016) above 32 and for every
// complex n, each filter alone removes a tier, nothing admits n = 0, and the throw names gebrd and n.
TYPED_TEST(DeviceApiSvd, WithTierPicksTheCtaWidthPerN) {
    using T = typename TypeParam::ScalarType;
    auto pick = [](const dv::DeviceLimits& lim, int n) {
        int got = 0;
        try {
            dv::with_tier<dv::routine::gebrd, T>(lim, n, [&]<class Tier>(Tier) {
                if constexpr (requires { Tier::nb; }) got = 1000 + Tier::nb;
                else got = Tier::n_max;
            });
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("gebrd: no device tier admits n=" + std::to_string(n)), std::string::npos) << e.what();
            got = -1;
        }
        return got;
    };
    const dv::DeviceLimits big{std::size_t(100) << 10, {16, 32}, 1024, true};
    if constexpr (dv::Arith<T>::is_complex) {
        EXPECT_EQ(pick(big, 4), 1016) << "tier::Cta is real only: complex takes tier::Blocked";
        EXPECT_EQ(pick(big, 0), -1);
    } else {
        const std::pair<int, int> want[] = {{1, 4}, {4, 4}, {5, 8}, {8, 8}, {9, 16}, {16, 16}, {17, 32}, {32, 32}, {33, 1016}, {0, -1}};
        for (const auto& [n, p] : want) EXPECT_EQ(pick(big, n), p) << "n " << n;
        constexpr std::size_t t32 = std::size_t(32) * 32 * sizeof(T);
        static_assert(dv::gebrd_buffer_size<dv::tier::Blocked<16>, T>(17).local_bytes < t32 - 1);
        EXPECT_EQ(pick({t32, {32}, 1024, true}, 17), 32) << "a local memory of exactly one Cta<32> tile";
        EXPECT_EQ(pick({t32 - 1, {32}, 1024, true}, 17), 1016) << "one byte short of a Cta<32> tile: Blocked<16> fits";
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
        static_assert(std::is_same_v<dv::resolve_tier_t<dv::routine::gebrd, T, dv::tier::upto<33>>, dv::tier::Blocked<16>>);
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

// ==== tier::Blocked: one matrix per work-group of 256 =========================================
constexpr int kBwg = 256;
using Blk = dv::tier::Blocked<16>;
using LC = std::complex<long double>;
template <class T> using real_of = typename dv::Arith<T>::real_t;

template <class T> LC lc(const T& v) {
    if constexpr (dv::Arith<T>::is_complex) return LC(v.real(), v.imag());
    else return LC(static_cast<long double>(v), 0.0L);
}
template <class T> T sval(std::uint64_t i) {
    if constexpr (dv::Arith<T>::is_complex) return T(uval<real_of<T>>(i), uval<real_of<T>>(i ^ 0x9e3779b9u));
    else return uval<T>(i);
}

// kBatch column-major m x n matrices at ld = m + 3, stride ld * n + 5, everything poisoned.
template <class T> struct Mats {
    int m, n, ld, stride;
    std::vector<T> h;
    Mats(int mm, int nn) : m(mm), n(nn), ld(std::max(mm, 1) + 3), stride((std::max(mm, 1) + 3) * std::max(nn, 1) + 5),
                           h(std::size_t(stride) * kBatch, poison<T>()) {}
    T& at(int b, int i, int j) { return h[std::size_t(b) * stride + std::size_t(j) * ld + i]; }
    const T& at(int b, int i, int j) const { return h[std::size_t(b) * stride + std::size_t(j) * ld + i]; }
    bool inside(std::size_t e) const {
        const std::size_t r = e % std::size_t(stride);
        return int(r / std::size_t(ld)) < n && int(r % std::size_t(ld)) < m;
    }
    MatrixView<T, MatrixFormat::Dense> view(T* p) const { return MatrixView<T, MatrixFormat::Dense>(p, m, n, ld, stride, kBatch); }
    void fill_random(std::uint64_t seed) {
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) at(b, i, j) = sval<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + std::uint64_t(j) * m + i);
    }
    void fill_identity() {
        for (int b = 0; b < kBatch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) at(b, i, j) = i == j ? T(1) : T(0);
    }
    Mats with(std::vector<T> got) const {
        Mats r = *this;
        r.h = std::move(got);
        return r;
    }
};
// Elements outside the m x n matrices that differ from `before`, bit for bit.
template <class T> long pads_changed(const Mats<T>& before, const Mats<T>& after) {
    long bad = 0;
    for (std::size_t e = 0; e < before.h.size(); ++e)
        if (!before.inside(e) && std::memcmp(&before.h[e], &after.h[e], sizeof(T)) != 0) ++bad;
    return bad;
}
// Items differing from their period representative (r < kPeriod; skip_rep excluded), bit for bit.
template <class T> long period_diffs(const Mats<T>& M, int skip_rep = -1) {
    long bad = 0;
    for (int b = kPeriod; b < kBatch; ++b) {
        if (b % kPeriod == skip_rep) continue;
        for (int j = 0; j < M.n; ++j)
            for (int i = 0; i < M.m; ++i) bad += std::memcmp(&M.at(b, i, j), &M.at(b % kPeriod, i, j), sizeof(T)) != 0;
    }
    return bad;
}
// max |(Q^H Q - I)(a, c)| over Q's columns (cols) or |(Q Q^H - I)(a, c)| over its rows.
template <class T> double unitary_defect(const Mats<T>& Q, int b, bool cols) {
    const int p = cols ? Q.n : Q.m, len = cols ? Q.m : Q.n;
    long double worst = 0;
    for (int a = 0; a < p; ++a)
        for (int c = 0; c < p; ++c) {
            LC sum = 0;
            for (int r = 0; r < len; ++r)
                sum += cols ? std::conj(lc(Q.at(b, r, a))) * lc(Q.at(b, r, c)) : lc(Q.at(b, a, r)) * std::conj(lc(Q.at(b, c, r)));
            worst = std::max(worst, std::abs(sum - LC(a == c ? 1.0L : 0.0L)));
        }
    return static_cast<double>(worst);
}
// ||A0 - U diag(s) Vh||_F / ||A0||_F of item b (A0(i, j) m x n) over the leading k columns of U
// and rows of Vh.
template <class T, class R, class F>
double svd_residual(F A0, int m, int n, const Mats<T>& U, const Mats<T>& Vh, const std::vector<R>& s, int ss, int b) {
    const int k = std::min(m, n);
    long double num = 0, den = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            LC x = 0;
            for (int p = 0; p < k; ++p) x += lc(U.at(b, i, p)) * static_cast<long double>(s[std::size_t(b) * ss + p]) * lc(Vh.at(b, p, j));
            const LC a = lc(A0(i, j));
            num += std::norm(a - x);
            den += std::norm(a);
        }
    return static_cast<double>(std::sqrt(num) / std::max(std::sqrt(den), std::numeric_limits<long double>::min()));
}
// Singular values (ascending, long double) of item b's upper bidiagonal (d at inc 2, stride sd; e
// at stride se): bisection on the Golub-Kahan tridiagonal (zero diagonal, off-diagonal d0 e0 d1
// ..), whose n largest eigenvalues they are; Sturm counts, so no rotation can mask a difference.
template <class R> std::vector<long double> bidiag_svals(const std::vector<R>& d, const std::vector<R>& e, int sd, int se, int b, int n) {
    std::vector<long double> f(std::size_t(2 * n - 1));
    for (int i = 0; i < n; ++i) {
        f[std::size_t(2 * i)] = d[std::size_t(b) * sd + 2 * i];
        if (i + 1 < n) f[std::size_t(2 * i + 1)] = e[std::size_t(b) * se + i];
    }
    long double hi = 0;
    for (int i = 0; i < 2 * n; ++i)
        hi = std::max(hi, (i > 0 ? std::fabs(f[i - 1]) : 0.0L) + (i < 2 * n - 1 ? std::fabs(f[i]) : 0.0L));
    const auto below = [&](long double x) {  // eigenvalues of the 2n x 2n tridiagonal below x
        int cnt = 0;
        long double q = -x;
        for (int i = 0;; ++i) {
            if (q == 0) q = -std::numeric_limits<long double>::min();
            cnt += q < 0;
            if (i + 1 == 2 * n) break;
            q = -x - f[i] * f[i] / q;
        }
        return cnt;
    };
    std::vector<long double> s(n);
    for (int j = 0; j < n; ++j) {
        long double lo = 0, up = hi * 1.01L + std::numeric_limits<long double>::min();
        for (int it = 0; it < 200 && up - lo > std::numeric_limits<long double>::epsilon() * up; ++it) {
            const long double mid = (lo + up) / 2;
            (below(mid) >= n + j + 1 ? up : lo) = mid;
        }
        s[j] = (lo + up) / 2;
    }
    return s;
}
// The user-style launch: kBatch work-groups of kBwg, group b's global slice at b * global_bytes
// (BLOCKED_BREAK: odd groups half a slice lower), f(ex, b, ws) per group.
template <class F> void launch_blocked(sycl::queue& q, const dv::WorkspaceSize& sz, F f) {
    std::byte* g = sycl::malloc_device<std::byte>(sz.global_bytes * kBatch + 64, q);
    q.submit([&](sycl::handler& h) {
         sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(std::max<std::size_t>(sz.local_bytes, 16)), h);
         h.parallel_for(sycl::nd_range<1>(std::size_t(kBatch) * kBwg, kBwg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
             const dv::WorkGroup<kBwg, 32> ex{it};
             const int b = static_cast<int>(it.get_group_linear_id());
             std::byte* gb = dv::global_slice(g, sz, b);
#ifdef BATCHLAS_DEVICE_API_SVD_BLOCKED_BREAK
             if (b & 1) gb -= sz.global_bytes / 32 * 16;
#endif
             f(ex, b, dv::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(), gb});
         });
     }).wait_and_throw();
    sycl::free(g, q);
}

bool blocked_runs(const sycl::queue& q) {
    const auto dev = q.get_device();
    const auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
    return std::find(sgs.begin(), sgs.end(), std::size_t(32)) != sgs.end() &&
           dev.get_info<sycl::info::device::max_work_group_size>() >= std::size_t(kBwg);
}

// bdsqr<tier::Blocked<1>> on a work-group vs the host bdsqr (one item per matrix), from identity
// U and V^H. Representatives: 0 uniform; 1 graded d with an interior and a bottom zero (both
// zero-shift chases); 2 a NaN in d (info 1 on both).
TYPED_TEST(DeviceApiSvd, BdsqrBlockedMatchesTheHostBdsqr) {
    using T = typename TypeParam::ScalarType;
    constexpr Backend B = TestFixture::B;
    if constexpr (dv::Arith<T>::is_complex) {
        GTEST_SKIP() << "bdsqr is real (gebrd hands it a real bidiagonal for every T)";
    } else {
        if (!blocked_runs(this->q())) GTEST_SKIP() << "tier::Blocked needs sub-group size 32 and a work-group of 256";
        for (int n : {1, 2, 3, 31, 32, 33, 100}) {
            SCOPED_TRACE("n " + std::to_string(n));
            const int sd = n + 2, se = n + 1, ss = n;
            std::vector<T> d(std::size_t(sd) * kBatch, poison<T>()), e(std::size_t(se) * kBatch, poison<T>());
            for (int b = 0; b < kBatch; ++b) {
                const int r = b % kPeriod;
                for (int i = 0; i < n; ++i) {
                    const T g = r == 1 ? static_cast<T>(std::pow(10.0, -6.0 * i / n)) : T(1);
                    d[std::size_t(b) * sd + i] = g * (T(1.5) + uval<T>(std::uint64_t(r) * 7919u + i));
                    if (i + 1 < n) e[std::size_t(b) * se + i] = g * uval<T>(std::uint64_t(r) * 7919u + 5000u + i);
                }
                if (r == 1 && n >= 3) d[std::size_t(b) * sd + n / 2] = T(0);
                if (r == 1) d[std::size_t(b) * sd + n - 1] = T(0);
                if (r == 2) d[std::size_t(b) * sd] = std::numeric_limits<T>::quiet_NaN();
            }
            Mats<T> U(n, n), Vh(n, n);
            U.fill_identity();
            Vh.fill_identity();
            DevBuf<T> d1(this->q(), d), e1(this->q(), e), u1(this->q(), U.h), v1(this->q(), Vh.h);
            DevBuf<T> d2(this->q(), d), e2(this->q(), e), u2(this->q(), U.h), v2(this->q(), Vh.h);
            DevBuf<T> s1(this->q(), std::vector<T>(std::size_t(ss) * kBatch, poison<T>())), s2(this->q(), std::vector<T>(std::size_t(ss) * kBatch, poison<T>()));
            DevBuf<std::int32_t> i1(this->q(), std::vector<std::int32_t>(kBatch, -7)), i2(this->q(), std::vector<std::int32_t>(kBatch, -7));
            {
                const VectorView<T> dv1(d1.p, n, kBatch, 1, sd), ev1(e1.p, std::max(n - 1, 0), kBatch, 1, se);
                const Span<T> sspan(s1.p, std::size_t(ss) * kBatch);
                UnifiedVector<std::byte> ws(bdsqr_buffer_size<T>(*this->ctx, dv1, ev1, sspan));
                (void)bdsqr<B, T>(*this->ctx, dv1, ev1, sspan, ws.to_span(), U.view(u1.p), Vh.view(v1.p), true,
                                  Span<std::int32_t>(i1.p, kBatch));
                this->ctx->wait();
            }
            T* const dp = d2.p;
            T* const ep = e2.p;
            T* const sp = s2.p;
            T* const up = u2.p;
            T* const vp = v2.p;
            std::int32_t* const ip = i2.p;
            const int ld = U.ld, st = U.stride;
            launch_blocked(this->q(), dv::bdsqr_buffer_size<dv::tier::Blocked<1>, T>(n), [=](const auto& ex, int b, const dv::Workspace& w) {
                const std::ptrdiff_t q = b;
                const int r = dv::bdsqr<dv::tier::Blocked<1>>(ex, dv::vec(dp + q * sd, n), dv::vec(ep + q * se, std::max(n - 1, 0)),
                                                              dv::vec(sp + q * ss, n), w, dv::mat(up + q * st, n, n, ld),
                                                              dv::mat(vp + q * st, n, n, ld));
                if (ex.leader()) ip[b] = r;
            });
            // The Thread executor from a one-item-per-matrix user kernel (v2 Mat operands, the
            // default params) vs the host launcher (v1 item operands, host-formed params): s, U,
            // V^H and info BIT FOR BIT.
            DevBuf<T> d3(this->q(), d), e3(this->q(), e), u3(this->q(), U.h), v3(this->q(), Vh.h);
            DevBuf<T> s3(this->q(), std::vector<T>(std::size_t(ss) * kBatch, poison<T>()));
            DevBuf<std::int32_t> i3(this->q(), std::vector<std::int32_t>(kBatch, -7));
            {
                T* const dq = d3.p;
                T* const eq = e3.p;
                T* const sq = s3.p;
                T* const uq = u3.p;
                T* const vq = v3.p;
                std::int32_t* const iq = i3.p;
                this->q()
                    .parallel_for(sycl::range<1>(std::size_t(kBatch)),
                                  [=](sycl::id<1> id) {
                                      const std::ptrdiff_t q = static_cast<std::ptrdiff_t>(id[0]);
                                      iq[q] = dv::bdsqr<dv::tier::Blocked<1>>(dv::Thread{}, dv::vec(dq + q * sd, n),
                                                                              dv::vec(eq + q * se, std::max(n - 1, 0)), dv::vec(sq + q * ss, n),
                                                                              dv::Workspace{}, dv::mat(uq + q * st, n, n, ld),
                                                                              dv::mat(vq + q * st, n, n, ld));
                                  })
                    .wait_and_throw();
            }
            const auto hi = i1.down(), gi = i2.down();
            const auto hs = s1.down(), gs = s2.down();
            EXPECT_EQ(first_diff(hs, s3.down()), -1) << "Thread s vs the host's";
            EXPECT_EQ(first_diff(u1.down(), u3.down()), -1) << "Thread U vs the host's";
            EXPECT_EQ(first_diff(v1.down(), v3.down()), -1) << "Thread V^H vs the host's";
            EXPECT_EQ(first_diff(hi, i3.down()), -1) << "Thread info vs the host's";
            const Mats<T> gu = U.with(u2.down()), gv = Vh.with(v2.down());
            long info_bad = 0;
            for (int b = 0; b < kBatch; ++b) info_bad += hi[b] != gi[b] || gi[b] != (b % kPeriod == 2 ? 1 : 0);
            EXPECT_EQ(info_bad, 0) << "info per item: host and device, 1 exactly on the NaN items";
            EXPECT_EQ(pads_changed(U, gu) + pads_changed(Vh, gv), 0) << "U / V^H pads";
            EXPECT_EQ(period_diffs(gu, 2) + period_diffs(gv, 2), 0) << "an item differs from its representative";
            double sv = 0, res = 0, orth = 0;
            for (int b = 0; b < 2; ++b) {
                long double smax = 0;
                for (int i = 0; i < n; ++i) smax = std::max<long double>(smax, std::fabs(hs[std::size_t(b) * ss + i]));
                for (int i = 0; i < n; ++i)
                    sv = batchlas::verify::nanmax(sv, double(std::fabs((long double)gs[std::size_t(b) * ss + i] - hs[std::size_t(b) * ss + i]) /
                                                             std::max(smax, std::numeric_limits<long double>::min())));
                const auto B0 = [&](int i, int j) {
                    return i == j ? d[std::size_t(b) * sd + i] : (j == i + 1 ? e[std::size_t(b) * se + i] : T(0));
                };
                res = batchlas::verify::nanmax(res, svd_residual(B0, n, n, gu, gv, gs, ss, b));
                orth = batchlas::verify::nanmax(orth, std::max(unitary_defect(gu, b, true), unitary_defect(gv, b, false)));
            }
            EXPECT_VERIFY(T, Check::values, n, sv);
            EXPECT_VERIFY(T, Check::eigen_residual, n, res);
            EXPECT_VERIFY(T, Check::orthogonality_rotations, n, orth);
            for (int b = 0; b < 2; ++b)
                for (int i = 0; i + 1 < n; ++i) EXPECT_GE(gs[std::size_t(b) * ss + i], gs[std::size_t(b) * ss + i + 1]) << "descending, item " << b;
        }
    }
}

// bdsdc<tier::Blocked<1>> on a work-group (the in-group stedc at order 2n) vs the host bdsdc (the
// host stedc): info, S descending, B == U S V^H, U and V^H orthogonal, and the values-only call.
// Representatives: 0 uniform; 1 graded (1e-3); 2 an exact zero d (a zero sigma: repair rebuilds
// its vectors).
TYPED_TEST(DeviceApiSvd, BdsdcBlockedMatchesTheHostBdsdc) {
    using T = typename TypeParam::ScalarType;
    constexpr Backend B = TestFixture::B;
    if constexpr (dv::Arith<T>::is_complex) {
        GTEST_SKIP() << "bdsdc is real (gebrd hands it a real bidiagonal for every T)";
    } else {
        if (!blocked_runs(this->q())) GTEST_SKIP() << "tier::Blocked needs sub-group size 32 and a work-group of 256";
        for (int n : {1, 2, 3, 17, 32, 33}) {
            SCOPED_TRACE("n " + std::to_string(n));
            const int sd = n + 2, se = n + 1, ss = n + 1;
            std::vector<T> d(std::size_t(sd) * kBatch, poison<T>()), e(std::size_t(se) * kBatch, poison<T>());
            for (int b = 0; b < kBatch; ++b) {
                const int r = b % kPeriod;
                for (int i = 0; i < n; ++i) {
                    const T g = r == 1 ? static_cast<T>(std::pow(10.0, -3.0 * i / n)) : T(1);
                    d[std::size_t(b) * sd + i] = g * (T(1.5) + uval<T>(std::uint64_t(r) * 7919u + i));
                    if (i + 1 < n) e[std::size_t(b) * se + i] = g * uval<T>(std::uint64_t(r) * 7919u + 5000u + i);
                }
                if (r == 2) d[std::size_t(b) * sd + n / 2] = T(0);
            }
            Mats<T> U(n, n), Vh(n, n);
            DevBuf<T> d1(this->q(), d), e1(this->q(), e), u1(this->q(), U.h), v1(this->q(), Vh.h), u2(this->q(), U.h), v2(this->q(), Vh.h);
            DevBuf<T> s1(this->q(), std::vector<T>(std::size_t(n) * kBatch, poison<T>()));
            DevBuf<T> s2(this->q(), std::vector<T>(std::size_t(ss) * kBatch, poison<T>())), s3(this->q(), std::vector<T>(std::size_t(ss) * kBatch, poison<T>()));
            DevBuf<std::int32_t> i1(this->q(), std::vector<std::int32_t>(kBatch, -7)), i2(this->q(), std::vector<std::int32_t>(kBatch, -7));
            {
                const VectorView<T> dv1(d1.p, n, kBatch, 1, sd), ev1(e1.p, std::max(n - 1, 0), kBatch, 1, se);
                const Span<T> sspan(s1.p, std::size_t(n) * kBatch);
                UnifiedVector<std::byte> ws(bdsdc_buffer_size<B, T>(*this->ctx, dv1, ev1, sspan, true));
                (void)bdsdc<B, T>(*this->ctx, dv1, ev1, sspan, ws.to_span(), U.view(u1.p), Vh.view(v1.p), true,
                                  Span<std::int32_t>(i1.p, kBatch));
                this->ctx->wait();
            }
            const T* const dp = d1.p;  // read only on both sides
            const T* const ep = e1.p;
            T* const sp = s2.p;
            T* const s3p = s3.p;
            T* const up = u2.p;
            T* const vp = v2.p;
            std::int32_t* const ip = i2.p;
            const int ld = U.ld, st = U.stride;
            launch_blocked(this->q(), dv::bdsdc_buffer_size<dv::tier::Blocked<1>, T>(n), [=](const auto& ex, int b, const dv::Workspace& w) {
                const std::ptrdiff_t q = b;
                const auto dd = dv::vec(dp + q * sd, n), ee = dv::vec(ep + q * se, std::max(n - 1, 0));
                const int r = dv::bdsdc(ex, dd, ee, dv::vec(sp + q * ss, n), w, dv::mat(up + q * st, n, n, ld), dv::mat(vp + q * st, n, n, ld));
                const auto none = dv::mat(static_cast<T*>(nullptr), 0, 0, 1);
                const int r3 = dv::bdsdc(ex, dd, ee, dv::vec(s3p + q * ss, n), w, none, none);
                if (ex.leader()) ip[b] = r | (r3 << 4);
            });
            const auto hi = i1.down(), gi = i2.down();
            const auto hs = s1.down(), gs = s2.down(), gs3 = s3.down();
            const Mats<T> gu = U.with(u2.down()), gv = Vh.with(v2.down());
            const T pz = poison<T>();
            long info_bad = 0, s_pads = 0, s_vals_only = 0;
            for (int b = 0; b < kBatch; ++b) {
                info_bad += hi[b] != 0 || gi[b] != 0;
                s_pads += std::memcmp(&gs[std::size_t(b) * ss + n], &pz, sizeof(T)) != 0;
                s_pads += std::memcmp(&gs3[std::size_t(b) * ss + n], &pz, sizeof(T)) != 0;
                for (int i = 0; i < n; ++i) s_vals_only += std::memcmp(&gs[std::size_t(b) * ss + i], &gs3[std::size_t(b) * ss + i], sizeof(T)) != 0;
            }
            EXPECT_EQ(info_bad, 0) << "info 0 on host and device for every item";
            EXPECT_EQ(s_pads, 0) << "S stride gaps";
            EXPECT_EQ(s_vals_only, 0) << "the values-only S is the with-vectors S bit for bit";
            EXPECT_EQ(pads_changed(U, gu) + pads_changed(Vh, gv), 0) << "U / V^H pads";
            EXPECT_EQ(period_diffs(gu) + period_diffs(gv), 0) << "an item differs from its representative";
            double sv = 0, res = 0, orth = 0;
            for (int b = 0; b < kPeriod; ++b) {
                long double smax = 0;
                for (int i = 0; i < n; ++i) smax = std::max<long double>(smax, std::fabs(hs[std::size_t(b) * n + i]));
                for (int i = 0; i < n; ++i)
                    sv = batchlas::verify::nanmax(sv, double(std::fabs((long double)gs[std::size_t(b) * ss + i] - hs[std::size_t(b) * n + i]) /
                                                             std::max(smax, std::numeric_limits<long double>::min())));
                const auto B0 = [&](int i, int j) {
                    return i == j ? d[std::size_t(b) * sd + i] : (j == i + 1 ? e[std::size_t(b) * se + i] : T(0));
                };
                res = batchlas::verify::nanmax(res, svd_residual(B0, n, n, gu, gv, gs, ss, b));
                orth = batchlas::verify::nanmax(orth, std::max(unitary_defect(gu, b, true), unitary_defect(gv, b, false)));
                for (int i = 0; i + 1 < n; ++i) EXPECT_GE(gs[std::size_t(b) * ss + i], gs[std::size_t(b) * ss + i + 1]) << "descending, item " << b;
            }
            EXPECT_VERIFY(T, Check::values, n, sv);
            EXPECT_VERIFY(T, Check::factorization, n, res);
            EXPECT_VERIFY(T, Check::orthogonality_rotations, n, orth);
        }
    }
}

// gebrd<tier::Blocked<16>> then ormbr on identities in the same user kernel: Q B P^H == A, Q and
// P^H unitary, |d| and |e| as the host gebrd_blocked's; m >= n straddling NB.
TYPED_TEST(DeviceApiSvd, GebrdBlockedAndOrmbrReconstructA) {
    using T = typename TypeParam::ScalarType;
    using R = real_of<T>;
    constexpr Backend B = TestFixture::B;
    if (!blocked_runs(this->q())) GTEST_SKIP() << "tier::Blocked needs sub-group size 32 and a work-group of 256";
    const std::pair<int, int> shapes[] = {{1, 1}, {15, 15}, {16, 16}, {17, 16}, {33, 33}, {50, 32}};
    for (const auto& [m, n] : shapes) {
        SCOPED_TRACE("m " + std::to_string(m) + " n " + std::to_string(n));
        Mats<T> A(m, n), Q(m, m), Ph(n, n);
        A.fill_random(11);
        Q.fill_identity();
        Ph.fill_identity();
        const int sd = 2 * n + 5, se = n + 3, st = n + 2;
        DevBuf<T> a(this->q(), A.h), qb(this->q(), Q.h), pb(this->q(), Ph.h);
        DevBuf<R> d(this->q(), std::vector<R>(std::size_t(sd) * kBatch, poison<R>())), e(this->q(), std::vector<R>(std::size_t(se) * kBatch, poison<R>()));
        DevBuf<T> tq(this->q(), std::vector<T>(std::size_t(st) * kBatch, poison<T>())), tp(this->q(), std::vector<T>(std::size_t(st) * kBatch, poison<T>()));
        T* const ap = a.p;
        T* const qp = qb.p;
        T* const pp = pb.p;
        R* const dp = d.p;
        R* const ep = e.p;
        T* const tqp = tq.p;
        T* const tpp = tp.p;
        const int lda = A.ld, sa = A.stride, ldq = Q.ld, sq = Q.stride, ldp = Ph.ld, sp = Ph.stride;
        launch_blocked(this->q(), dv::gebrd_buffer_size<Blk, T>(m, n), [=](const auto& ex, int b, const dv::Workspace& w) {
            const std::ptrdiff_t q = b;
            const auto Am = dv::mat(ap + q * sa, m, n, lda);
            const auto tauq = dv::vec(tqp + q * st, n), taup = dv::vec(tpp + q * st, n);
            (void)dv::gebrd<Blk>(ex, Am, dv::vec(dp + q * sd, n, 2), dv::vec(ep + q * se, std::max(n - 1, 0), 1), tauq, taup, w);
            (void)dv::ormbr<'Q', Side::Left, Transpose::NoTrans>(ex, Am, tauq, dv::mat(qp + q * sq, m, m, ldq), dv::Workspace{});
            (void)dv::ormbr<'P', Side::Right, Transpose::ConjTrans>(ex, Am, taup, dv::mat(pp + q * sp, n, n, ldp), dv::Workspace{});
        });
        const Mats<T> ga = A.with(a.down()), gq = Q.with(qb.down()), gp = Ph.with(pb.down());
        const auto gd = d.down(), ge = e.down();
        EXPECT_EQ(pads_changed(A, ga) + pads_changed(Q, gq) + pads_changed(Ph, gp), 0) << "A / Q / P^H pads";
        EXPECT_EQ(period_diffs(ga) + period_diffs(gq) + period_diffs(gp), 0) << "an item differs from its representative";
        double res = 0, orth = 0;
        for (int b = 0; b < kPeriod; ++b) {
            long double num = 0, den = 0;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    LC x = 0;  // (Q B P^H)(i, j), B(p, p) = d(p), B(p, p + 1) = e(p)
                    for (int p = 0; p < n; ++p) {
                        LC bp = 0;
                        bp += lc(gq.at(b, i, p)) * static_cast<long double>(gd[std::size_t(b) * sd + 2 * p]) * lc(gp.at(b, p, j));
                        if (p + 1 < n) bp += lc(gq.at(b, i, p)) * static_cast<long double>(ge[std::size_t(b) * se + p]) * lc(gp.at(b, p + 1, j));
                        x += bp;
                    }
                    num += std::norm(lc(A.at(b, i, j)) - x);
                    den += std::norm(lc(A.at(b, i, j)));
                }
            res = batchlas::verify::nanmax(res, double(std::sqrt(num) / std::max(std::sqrt(den), std::numeric_limits<long double>::min())));
            orth = batchlas::verify::nanmax(orth, std::max(unitary_defect(gq, b, true), unitary_defect(gp, b, false)));
        }
        EXPECT_VERIFY(T, Check::factorization, m, res);
        EXPECT_VERIFY(T, Check::orthogonality, m, orth);
        if constexpr (B != Backend::NETLIB) {
            DevBuf<T> a2(this->q(), A.h);
            DevBuf<R> d2(this->q(), std::vector<R>(std::size_t(sd) * kBatch, poison<R>())), e2(this->q(), std::vector<R>(std::size_t(se) * kBatch, poison<R>()));
            DevBuf<T> q2(this->q(), std::vector<T>(std::size_t(st) * kBatch, poison<T>())), p2(this->q(), std::vector<T>(std::size_t(st) * kBatch, poison<T>()));
            const auto av = A.view(a2.p);
            const VectorView<R> dv2(d2.p, n, kBatch, 2, sd), ev2(e2.p, std::max(n - 1, 0), kBatch, 1, se);
            const VectorView<T> qv2(q2.p, n, kBatch, 1, st), pv2(p2.p, n, kBatch, 1, st);
            UnifiedVector<std::byte> ws(gebrd_blocked_buffer_size<B, T>(*this->ctx, av, dv2, ev2, qv2, pv2, 16));
            (void)gebrd_blocked<B, T>(*this->ctx, av, dv2, ev2, qv2, pv2, ws.to_span(), 16);
            this->ctx->wait();
            // The bidiagonals' singular values, not d / e entrywise: B is a Lanczos-like product
            // whose entries move by far more than the backward error when two reductions sum in
            // different orders (the real host panel is its own kernel), its singular values do not.
            const auto hd = d2.down(), he = e2.down();
            double worst = 0;
            for (int b = 0; b < kPeriod; ++b) {
                const auto gs = bidiag_svals(gd, ge, sd, se, b, n), hs = bidiag_svals(hd, he, sd, se, b, n);
                const long double scale = std::max(hs.back(), std::numeric_limits<long double>::min());
                for (int i = 0; i < n; ++i) worst = batchlas::verify::nanmax(worst, double(std::fabs(gs[i] - hs[i]) / scale));
            }
            EXPECT_VERIFY(T, Check::values, m, worst);
        }
    }
}

// gesvd<JU, JVT, tier::Blocked<16>> from a user kernel vs the host route pinned to blocked over
// bdsqr, for one (m, n) and job pair.
template <SvdVectors JU, SvdVectors JVT, Backend B, class T> void check_gesvd_blocked(sycl::queue& dq, Queue& ctx, int m, int n) {
    using R = real_of<T>;
    const int k = std::min(m, n);
    const int um = JU == SvdVectors::None ? 0 : m, un = JU == SvdVectors::All ? m : (JU == SvdVectors::Thin ? k : 0);
    const int vm = JVT == SvdVectors::All ? n : (JVT == SvdVectors::Thin ? k : 0), vn = JVT == SvdVectors::None ? 0 : n;
    Mats<T> A(m, n), U(std::max(um, 1), std::max(un, 1)), Vh(std::max(vm, 1), std::max(vn, 1));
    A.fill_random(29);
    const int ss = k + 2;
    DevBuf<T> a1(dq, A.h), u1(dq, U.h), v1(dq, Vh.h), a2(dq, A.h), u2(dq, U.h), v2(dq, Vh.h);
    DevBuf<R> s1(dq, std::vector<R>(std::size_t(k) * kBatch, poison<R>())), s2(dq, std::vector<R>(std::size_t(ss) * kBatch, poison<R>()));
    DevBuf<std::int32_t> i1(dq, std::vector<std::int32_t>(kBatch, -7)), i2(dq, std::vector<std::int32_t>(kBatch, -7));
    {
        const batchlas::ScopedEnvVar route("BATCHLAS_GESVD_ROUTE", "blocked");
        const batchlas::ScopedEnvVar bidiag("BATCHLAS_GESVD_BIDIAG", "bdsqr");
        const auto av = A.view(a1.p);
        const auto uv = JU == SvdVectors::None ? MatrixView<T, MatrixFormat::Dense>() : U.view(u1.p);
        const auto vv = JVT == SvdVectors::None ? MatrixView<T, MatrixFormat::Dense>() : Vh.view(v1.p);
        const Span<R> sspan(s1.p, std::size_t(k) * kBatch);
        UnifiedVector<std::byte> ws(gesvd_buffer_size<B, T>(ctx, av, sspan, uv, vv, JU, JVT));
        (void)gesvd<B, T>(ctx, av, sspan, uv, vv, JU, JVT, ws.to_span(), Span<std::int32_t>(i1.p, kBatch));
        ctx.wait();
    }
    T* const ap = a2.p;
    T* const up = u2.p;
    T* const vp = v2.p;
    R* const sp = s2.p;
    std::int32_t* const ip = i2.p;
    const int lda = A.ld, sa = A.stride, ldu = U.ld, su = U.stride, ldv = Vh.ld, sv = Vh.stride;
    launch_blocked(dq, dv::gesvd_buffer_size<JU, JVT, Blk, T>(m, n), [=](const auto& ex, int b, const dv::Workspace& w) {
        const std::ptrdiff_t q = b;
        const int r = dv::gesvd<JU, JVT, Blk>(ex, dv::mat(ap + q * sa, m, n, lda), dv::vec(sp + q * ss, k), dv::mat(up + q * su, um, un, ldu),
                                              dv::mat(vp + q * sv, vm, vn, ldv), w);
        if (ex.leader()) ip[b] = r;
    });
    const auto hi = i1.down(), gi = i2.down();
    const auto hs = s1.down(), gs = s2.down();
    const Mats<T> gu = U.with(u2.down()), gv = Vh.with(v2.down());
    const R pz = poison<R>();
    long info_bad = 0, s_pads = 0, s_period = 0;
    for (int b = 0; b < kBatch; ++b) {
        info_bad += hi[b] != gi[b] || gi[b] != 0;
        for (int i = k; i < ss; ++i) s_pads += std::memcmp(&gs[std::size_t(b) * ss + i], &pz, sizeof(R)) != 0;
        if (b >= kPeriod)
            for (int i = 0; i < k; ++i) s_period += std::memcmp(&gs[std::size_t(b) * ss + i], &gs[std::size_t(b % kPeriod) * ss + i], sizeof(R)) != 0;
    }
    EXPECT_EQ(info_bad, 0) << "info: host and device 0 for every item";
    EXPECT_EQ(s_pads, 0) << "S stride gaps";
    EXPECT_EQ(s_period, 0) << "S of an item differs from its representative";
    double sv_err = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < k; ++i)
            sv_err = batchlas::verify::nanmax(sv_err, double(std::fabs((long double)gs[std::size_t(b) * ss + i] - hs[std::size_t(b) * k + i]) /
                                                             std::max<long double>(hs[std::size_t(b) * k], std::numeric_limits<long double>::min())));
    EXPECT_VERIFY(T, Check::values, std::max(m, n), sv_err);
    if constexpr (JU != SvdVectors::None && JVT != SvdVectors::None) {
        EXPECT_EQ(pads_changed(U, gu) + pads_changed(Vh, gv), 0) << "U / V^H pads";
        EXPECT_EQ(period_diffs(gu) + period_diffs(gv), 0) << "U / V^H of an item differs from its representative";
        double res = 0, orth = 0;
        for (int b = 0; b < kPeriod; ++b) {
            res = batchlas::verify::nanmax(res, svd_residual([&](int i, int j) { return A.at(b, i, j); }, m, n, gu, gv, gs, ss, b));
            orth = batchlas::verify::nanmax(orth, std::max(unitary_defect(gu, b, true), unitary_defect(gv, b, false)));
        }
        EXPECT_VERIFY(T, Check::factorization, std::max(m, n), res);
        EXPECT_VERIFY(T, Check::orthogonality, std::max(m, n), orth);
    } else {
        EXPECT_EQ(std::memcmp(gu.h.data(), U.h.data(), U.h.size() * sizeof(T)), 0) << "a None U is not touched";
        EXPECT_EQ(std::memcmp(gv.h.data(), Vh.h.data(), Vh.h.size() * sizeof(T)), 0) << "a None V^H is not touched";
    }
}

TYPED_TEST(DeviceApiSvd, GesvdBlockedMatchesThePinnedHostBlockedRoute) {
    using T = typename TypeParam::ScalarType;
    constexpr Backend B = TestFixture::B;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "the host blocked gesvd route is GPU-only";
    } else {
        if (!blocked_runs(this->q())) GTEST_SKIP() << "tier::Blocked needs sub-group size 32 and a work-group of 256";
        const std::pair<int, int> shapes[] = {{1, 1}, {17, 16}, {33, 33}, {48, 17}, {17, 48}};
        for (const auto& [m, n] : shapes) {
            SCOPED_TRACE("m " + std::to_string(m) + " n " + std::to_string(n));
            check_gesvd_blocked<SvdVectors::All, SvdVectors::All, B, T>(this->q(), *this->ctx, m, n);
            check_gesvd_blocked<SvdVectors::Thin, SvdVectors::Thin, B, T>(this->q(), *this->ctx, m, n);
            check_gesvd_blocked<SvdVectors::None, SvdVectors::None, B, T>(this->q(), *this->ctx, m, n);
        }
    }
}

// with_tier picks Blocked<16> for every n >= 1 and drops it under each device filter; the throw
// names gesvd and n; gesvd_buffer_size is the driver plan, its local bytes the same for all m, n.
TYPED_TEST(DeviceApiSvd, GesvdWithTierAndBufferSize) {
    using T = typename TypeParam::ScalarType;
    constexpr auto All = SvdVectors::All;
    using Ro = dv::routine::gesvd<All, All>;
    auto pick = [](const dv::DeviceLimits& lim, int n) {
        int got = 0;
        try {
            dv::with_tier<Ro, T>(lim, n, [&]<class Tier>(Tier) { got = Tier::nb; });
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("gesvd: no device tier admits n=" + std::to_string(n)), std::string::npos) << e.what();
            got = -1;
        }
        return got;
    };
    const std::size_t lb = dv::gesvd_buffer_size<All, All, Blk, T>(64, 64).local_bytes;
    const int no_fp64 = std::is_same_v<real_of<T>, double> ? -1 : 16;
    const dv::DeviceLimits big{std::size_t(100) << 10, {16, 32}, 1024, true};
    EXPECT_EQ(pick(big, 1), 16);
    EXPECT_EQ(pick(big, 1000), 16);
    EXPECT_EQ(pick(big, 0), -1);
    EXPECT_EQ(pick({lb, {32}, std::size_t(kBwg), true}, 64), 16) << "exactly the local bytes and a work-group of 256";
    EXPECT_EQ(pick({lb - 1, {32}, 1024, true}, 64), -1) << "one byte short";
    EXPECT_EQ(pick({lb, {16}, 1024, true}, 64), -1) << "no sub-group size 32";
    EXPECT_EQ(pick({lb, {32}, std::size_t(kBwg) - 1, true}, 64), -1) << "a work-group ceiling below 256";
    EXPECT_EQ(pick({lb, {32}, 1024, false}, 64), no_fp64) << "no fp64";
    static_assert(std::is_same_v<dv::resolve_tier_t<Ro, T, dv::tier::upto<4096>>, Blk>);
    const std::pair<int, int> shapes[] = {{1, 1}, {16, 17}, {64, 33}, {33, 64}};
    for (const auto& [m, n] : shapes) {
        const auto plan = dv::gesvd_detail::driver_plan<All, All, 16, T>(m, n);
        EXPECT_EQ((dv::gesvd_buffer_size<All, All, Blk, T>(m, n)), plan.size) << m << " x " << n;
        EXPECT_EQ(plan.size.local_bytes, lb) << "local bytes depend on neither m nor n";
        EXPECT_EQ(plan.at.count, m < n ? std::size_t(m) * n : 0u) << "A^H only on the m < n path";
        EXPECT_GE(plan.size.global_bytes, plan.bdsqr_global.offset + plan.bdsqr_global.count);
    }
}

}  // namespace
