// SPDX-License-Identifier: MIT
// device_api_stebz-stein_tests: the public device::stebz and device::stein (include/batchlas/
// device/lapack/{stebz,stein}.hh, tier::Blocked<1>) called from user-style kernels, against the
// host routes, which are thin launchers over the same bodies (design C.8, C.10).
// Guards (docs/developer/agent-guide.md section 8):
//   - bit-identical (memcmp, m and info included) to the host route. Each op has one host route
//     (no choice.hh), so there is no route to pin. The stebz kernel runs a compile-time
//     WorkGroup<256, 32> (bisection does not depend on the width); the stein kernel runs the
//     host's width (phase 2 reduces over the work-group)
//   - batch 1024 at period 3, every item bit-identical to its representative
//   - inc 2 and stride + 3 with poisoned pads; w's pads and slots past m, and Z's pad rows
//     (ld = n + 3), stay poisoned
//   - graded, split and repeated-eigenvalue inputs, and the 1-D Laplacian against its analytic
//     spectrum (Check::values)
//   - every range (All, Index descending, Value straddling an eigenvalue by its midpoints), n = 1,
//     2, 33, 300. At n = 300 there are more eigenvalues (and vectors) than the 256 items, so each
//     item owns two (the launch at the work-group ceiling)
//   - with_tier picks Blocked<1> and throws "<op>: no device tier admits n=0"; buffer_size
//     equals the carve plan
// Deliberate breaks, run on the 4090 (one per axis, red set observed):
//   stebz drops the descending flip               -> StebzMatchesHostRoute (float, double)
//   stebz sets m = found[1]                       -> StebzMatchesHostRoute (float, double)
//   stein iterate skips zeroing columns past kb   -> SteinMatchesHostRoute (float, double)
//   stein scratch() carves dd onto dl (op only)   -> SteinMatchesHostRoute (float, double)
//   stein iterate drops the j * 104729 seed term  -> none: the host route shares the body and
//     the vectors stay valid (accepted blind spot, as is a layout that moves offsets but keeps
//     sizes: the carve test pins total and alignment only)
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numbers>
#include <optional>
#include <string>
#include <vector>

#include <batchlas/blas/extensions.hh>
#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/stebz.hh>
#include <batchlas/device/lapack/stein.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace dv = batchlas::device;
using batchlas::verify::Check;
using Tier = dv::tier::Blocked<1>;

template <class T> struct UserStebz;
template <class T> struct UserStein;

namespace {

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr int kInc = 2;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
double uval(std::uint64_t i) { return static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0; }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T poison() { return std::numeric_limits<T>::max(); }  // wins every max an op that read it takes

// Representative r of an order-n tridiagonal: 0 the 1-D Laplacian (2, -1); 1 graded (d_i and
// e_i fall by 10 every 4 rows); 2 split (e_i = 0 every fifth row) with d repeating in pairs, so
// whole eigenvalues repeat exactly.
template <class T> void rep(int r, int n, std::vector<double>& d, std::vector<double>& e) {
    d.assign(n, 0.0);
    e.assign(std::max(n - 1, 0), 0.0);
    for (int i = 0; i < n; ++i) {
        if (r == 0) d[i] = 2.0;
        else if (r == 1) d[i] = std::pow(10.0, -(i / 4)) * (1.5 + uval(7 * n + i));
        else d[i] = uval(11 * n + i / 2);
    }
    for (int i = 0; i + 1 < n; ++i) {
        if (r == 0) e[i] = -1.0;
        else if (r == 1) e[i] = 0.5 * std::pow(10.0, -(i / 4)) * uval(13 * n + i);
        else e[i] = (i % 5 == 4) ? 0.0 : 0.25 * uval(17 * n + i);
    }
}

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
                                   : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
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
    int max_wg() { return static_cast<int>(ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)); }
    // Frees every buffer allocated since ptrs.size() was mark (n = 300 in double holds ~9 GB).
    void release(std::size_t mark) {
        for (std::size_t i = mark; i < ptrs.size(); ++i) sycl::free(ptrs[i], q());
        ptrs.resize(mark);
    }
    ~Dev() {
        if (ctx) release(0);
    }
    std::vector<void*> ptrs;
};

// The batch of one n: d, e at inc 2 and stride n * 2 + 3 (pads poisoned), item b a twin of
// b % kPeriod; the representatives in double for the analytic check.
template <class T> struct Problem {
    int n, sd, se;
    std::vector<T> d, e;
    std::vector<std::vector<double>> rd, re;
    explicit Problem(int n_) : n(n_), sd(n_ * kInc + 3), se(std::max(n_ - 1, 1) * kInc + 3) {
        d.assign(std::size_t(kBatch) * sd, poison<T>());
        e.assign(std::size_t(kBatch) * se, poison<T>());
        rd.resize(kPeriod);
        re.resize(kPeriod);
        for (int r = 0; r < kPeriod; ++r) rep<T>(r, n, rd[r], re[r]);
        for (int b = 0; b < kBatch; ++b) {
            const int r = b % kPeriod;
            for (int i = 0; i < n; ++i) d[std::size_t(b) * sd + i * kInc] = static_cast<T>(rd[r][i]);
            for (int i = 0; i + 1 < n; ++i) e[std::size_t(b) * se + i * kInc] = static_cast<T>(re[r][i]);
        }
    }
};

template <class T> class DeviceApiStebzStein : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceApiStebzStein, RealTypes);

struct Sel {
    EigenRangeType range;
    SortOrder order;
};

// Host StebzParams for selection s on order n: Index takes [1, n - 2] (all when n < 3); Value
// takes (vl, vu] with vl, vu the midpoints below the second and above the second-to-last
// Laplacian eigenvalue, so the count straddles an eigenvalue of representative 0 at each end.
template <class T> StebzParams<T> params_for(const Sel& s, int n) {
    StebzParams<T> p;
    p.range = s.range;
    p.order = s.order;
    if (s.range == EigenRangeType::Index) {
        p.il = n < 3 ? 0 : 1;
        p.iu = n < 3 ? n - 1 : n - 2;
    } else if (s.range == EigenRangeType::Value) {
        auto lap = [&](int k) { return 2.0 - 2.0 * std::cos(k * std::numbers::pi / (n + 1)); };
        p.vl = static_cast<T>(n < 3 ? -1.0 : 0.5 * (lap(1) + lap(2)));
        p.vu = static_cast<T>(n < 3 ? 5.0 : 0.5 * (lap(n - 1) + lap(n)));
    }
    return p;
}
template <class T> dv::stebz_detail::Params<T> device_params(const StebzParams<T>& p, int n) {
    const bool index = p.range == EigenRangeType::Index;
    return {p.range == EigenRangeType::Value,
            index ? p.il : 0,
            index ? (p.iu < 0 ? n - 1 : p.iu) : n - 1,
            p.vl,
            p.vu,
            p.abstol,
            p.order == SortOrder::Descending,
            std::max<std::int32_t>(1, p.max_iterations)};
}

// w at inc 1, stride n + 3 (pads and unused slots poisoned).
template <class T> struct WOut {
    int sw;
    T* p;
};

// The device op from a user kernel: WorkGroup<256, 32>, one matrix per work-group.
template <class T>
void user_stebz(Dev& dev, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& w, std::int32_t* m, std::int32_t* info,
                const dv::stebz_detail::Params<T>& p, int n) {
    constexpr int WG = 256;
    const dv::WorkspaceSize sz = dv::stebz_buffer_size<Tier, T>(n);
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(sz.local_bytes), h);
           h.parallel_for<UserStebz<T>>(sycl::nd_range<1>(std::size_t(kBatch) * WG, WG),
                                        [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                                            const dv::WorkGroup<WG, 32> ex{it};
                                            const int b = static_cast<int>(it.get_group_linear_id());
                                            int mb = -1;
                                            const int ib = dv::stebz<Tier>(ex, dv::item(d, b), dv::item(e, b), dv::item(w, b), mb,
                                                                           dv::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(), nullptr}, p);
                                            if (ex.leader()) {
                                                m[b] = mb;
                                                info[b] = ib;
                                            }
                                        });
       }).wait_and_throw();
}

// ---- stebz ----------------------------------------------------------------------------------

TYPED_TEST(DeviceApiStebzStein, StebzMatchesHostRoute) {
    using T = TypeParam;
    auto& dev = this->dev;
    const Sel sels[] = {{EigenRangeType::All, SortOrder::Ascending},
                        {EigenRangeType::Index, SortOrder::Descending},
                        {EigenRangeType::Value, SortOrder::Ascending}};
    for (int n : {1, 2, 33, 300}) {
        Problem<T> P(n);
        T* d = dev.up(P.d);
        T* e = dev.up(P.e);
        const VectorView<T> dview(d, n, kBatch, kInc, P.sd);
        const VectorView<T> eview(e, std::max(n - 1, 0), kBatch, kInc, P.se);
        const int sw = n + 3;
        for (const Sel& s : sels) {
            const auto hp = params_for<T>(s, n);
            const std::vector<T> w0(std::size_t(kBatch) * sw, poison<T>());
            T* wh = dev.up(w0);
            T* wd = dev.up(w0);
            std::int32_t* mh = dev.up(std::vector<std::int32_t>(kBatch, -7));
            std::int32_t* md = dev.up(std::vector<std::int32_t>(kBatch, -7));
            std::int32_t* info = dev.up(std::vector<std::int32_t>(kBatch, -7));
            stebz(*dev.ctx, dview, eview, VectorView<T>(wh, n, kBatch, 1, sw), Span<std::int32_t>(mh, kBatch), Span<std::byte>(), hp);
            dev.ctx->wait();
            user_stebz<T>(dev, dview, eview, VectorView<T>(wd, n, kBatch, 1, sw), md, info, device_params(hp, n), n);

            const std::string tag = "n=" + std::to_string(n) + " range=" + std::to_string(int(s.range));
            const auto gh = dev.down(wh, w0.size()), gd = dev.down(wd, w0.size());
            const auto mgh = dev.down(mh, kBatch), mgd = dev.down(md, kBatch), ig = dev.down(info, kBatch);
            int bad = 0;
            for (std::size_t i = 0; i < gh.size(); ++i) bad += !same_bits(gh[i], gd[i]);
            EXPECT_EQ(bad, 0) << tag << ": device w differs from the host route";
            EXPECT_EQ(std::memcmp(mgh.data(), mgd.data(), kBatch * sizeof(std::int32_t)), 0) << tag << ": m differs";
            for (int b = 0; b < kBatch; ++b) {
                ASSERT_EQ(ig[b], 0) << tag << " item " << b;
                ASSERT_EQ(mgd[b], mgd[b % kPeriod]) << tag << " item " << b;
                const int mb = mgd[b];
                ASSERT_GE(mb, s.range == EigenRangeType::Value ? 0 : 1) << tag;
                for (int i = 0; i < sw; ++i) {
                    const T got = gd[std::size_t(b) * sw + i];
                    if (i >= mb) ASSERT_TRUE(same_bits(got, poison<T>())) << tag << " item " << b << " slot " << i << " past m";
                    else ASSERT_TRUE(same_bits(got, gd[std::size_t(b % kPeriod) * sw + i])) << tag << " item " << b << " slot " << i;
                }
            }
            // Representative 0, the Laplacian: lambda_k = 2 - 2 cos(k pi / (n + 1)), ||T|| <= 4.
            if (n >= 2) {
                const int m0 = mgd[0];
                const auto p = device_params(hp, n);
                std::int64_t first = p.il;
                if (p.value_range) {
                    first = 0;
                    while (first < n && 2.0 - 2.0 * std::cos((first + 1) * std::numbers::pi / (n + 1)) <= double(p.vl)) ++first;
                }
                double err = 0;
                for (int s0 = 0; s0 < m0; ++s0) {
                    const int idx = p.descending ? m0 - 1 - s0 : s0;
                    const double ref = 2.0 - 2.0 * std::cos((first + idx + 1) * std::numbers::pi / (n + 1));
                    err = std::max(err, std::fabs(double(gd[s0]) - ref));
                }
                EXPECT_VERIFY(T, Check::values, n, err / 4.0) << tag << " (Laplacian)";
                if (s.range == EigenRangeType::Value && n >= 3) EXPECT_EQ(m0, n - 2) << tag << ": the straddled count";
            }
        }
    }
}

TYPED_TEST(DeviceApiStebzStein, StebzTierQueries) {
    using T = TypeParam;
    using R = dv::routine::stebz;
    static_assert(std::is_same_v<dv::resolve_tier_t<R, T, dv::tier::upto<1 << 20>>, Tier>);
    for (int n : {1, 2, 33, 300, 100000}) {
        std::string got = "none";
        dv::with_tier<R, T>(this->dev.q().get_device(), n, [&]<class Tr>(Tr) { got = std::is_same_v<Tr, Tier> ? "blocked1" : "other"; });
        EXPECT_EQ(got, "blocked1") << n;
        const auto sz = dv::stebz_buffer_size<Tier, T>(n);
        const auto L = dv::stebz_detail::layout<T>();
        EXPECT_EQ(sz, L.size);
        EXPECT_EQ(sz.global_bytes, 0u);
        EXPECT_GE(sz.local_bytes, L.found.offset + 2 * sizeof(std::int32_t));
        EXPECT_EQ(L.found.offset % alignof(std::int32_t), 0u);
    }
    try {
        dv::with_tier<R, T>(this->dev.q().get_device(), 0, [](auto) {});
        ADD_FAILURE() << "with_tier admitted n = 0";
    } catch (const batchlas::invalid_argument& ex) {
        EXPECT_NE(std::string(ex.what()).find("stebz: no device tier admits n=0"), std::string::npos) << ex.what();
    }
}

// ---- stein ----------------------------------------------------------------------------------

// The device op from a user kernel at the host's phase-2 width (RtGroup), one matrix per group,
// both phases in one kernel; counts may be null (all k).
template <class T>
void user_stein(Dev& dev, int wg, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& w, const std::int32_t* counts,
                const MatrixView<T, MatrixFormat::Dense>& Z, std::byte* gws, std::int32_t* info, std::uint32_t seed) {
    const int n = d.size(), k = Z.cols();
    const dv::WorkspaceSize sz = dv::stein_buffer_size<Tier, T>(n, k);
    const auto Zk = Z.kernel_view();
    const SteinParams<T> hp;
    const std::int32_t max_iter = std::max<std::int32_t>(1, hp.max_iterations);
    const T thr = hp.ortho_threshold;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(sz.local_bytes), h);
           h.parallel_for<UserStein<T>>(sycl::nd_range<1>(std::size_t(kBatch) * wg, wg), [=](sycl::nd_item<1> it) {
               const dv::RtGroup<> ex{it};
               const int b = static_cast<int>(it.get_group_linear_id());
               const int kb = counts ? std::clamp<int>(counts[b], 0, k) : k;
               const dv::stein_detail::Params<T> p{max_iter, thr, seed + static_cast<std::uint32_t>(std::int64_t(b) * 7919)};
               const auto dvec = dv::vec(d.data_ptr() + std::int64_t(b) * d.stride(), n, d.inc());
               const auto evec = dv::vec(e.data_ptr() + std::int64_t(b) * e.stride(), n - 1, e.inc());
               const auto Zb = dv::mat(Zk.data() + std::int64_t(b) * Zk.stride(), n, k, Zk.ld());
               const int ib = dv::stein<Tier>(ex, dvec, evec, dv::item(w, b), kb, Zb,
                                              dv::Workspace{slm.template get_multi_ptr<sycl::access::decorated::yes>(),
                                                            dv::global_slice(gws, sz, b)},
                                              p);
               if (ex.leader()) info[b] = ib;
           });
       }).wait_and_throw();
}

// stebz (All, then Value for per-item counts) feeds both stein routes; Z, including the
// poisoned pad rows and the zeroed columns past kb, must match bit for bit.
TYPED_TEST(DeviceApiStebzStein, SteinMatchesHostRoute) {
    using T = TypeParam;
    auto& dev = this->dev;
    for (int n : {1, 2, 33, 300}) {
        Problem<T> P(n);
        T* d = dev.up(P.d);
        T* e = dev.up(P.e);
        const VectorView<T> dview(d, n, kBatch, kInc, P.sd);
        const VectorView<T> eview(e, std::max(n - 1, 0), kBatch, kInc, P.se);
        for (const Sel s : {Sel{EigenRangeType::All, SortOrder::Ascending}, Sel{EigenRangeType::Value, SortOrder::Ascending}}) {
            const std::size_t mark = dev.ptrs.size();
            const bool with_counts = s.range == EigenRangeType::Value;
            const int k = n, sw = n + 3, ld = n + 3, sz = ld * k + 5;
            T* w = dev.up(std::vector<T>(std::size_t(kBatch) * sw, T(0)));
            std::int32_t* m = dev.up(std::vector<std::int32_t>(kBatch, 0));
            const VectorView<T> wview(w, n, kBatch, 1, sw);
            stebz(*dev.ctx, dview, eview, wview, Span<std::int32_t>(m, kBatch), Span<std::byte>(), params_for<T>(s, n));
            dev.ctx->wait();

            const std::vector<T> z0(std::size_t(kBatch) * sz, poison<T>());
            T* zh = dev.up(z0);
            T* zd = dev.up(z0);
            const MatrixView<T, MatrixFormat::Dense> Zh(zh, n, k, ld, sz, kBatch), Zd(zd, n, k, ld, sz, kBatch);
            const SteinParams<T> sp;
            const std::size_t bytes = stein_buffer_size(*dev.ctx, std::size_t(n), std::size_t(k), std::size_t(kBatch), sp);
            std::byte* hws = dev.up(std::vector<std::byte>(bytes));
            const Span<const std::int32_t> counts = with_counts ? Span<const std::int32_t>(m, kBatch) : stein_all_counts;
            stein(*dev.ctx, dview, eview, wview, std::size_t(k), counts, Zh, Span<std::byte>(hws, bytes), sp);
            dev.ctx->wait();

            const auto gsz = dv::stein_buffer_size<Tier, T>(n, k);
            std::byte* gws = dev.up(std::vector<std::byte>(gsz.global_bytes * kBatch));
            std::int32_t* info = dev.up(std::vector<std::int32_t>(kBatch, -7));
            const int wg = std::min({256, dev.max_wg(), std::max(n, 1)});
            user_stein<T>(dev, wg, dview, eview, wview, with_counts ? m : nullptr, Zd, gws, info, sp.seed);

            const std::string tag = "n=" + std::to_string(n) + (with_counts ? " counts" : " all");
            const auto gh = dev.down(zh, z0.size()), gd = dev.down(zd, z0.size());
            const auto ig = dev.down(info, kBatch);
            const auto mg = dev.down(m, kBatch);
            int bad = 0;
            for (std::size_t i = 0; i < gh.size(); ++i) bad += !same_bits(gh[i], gd[i]);
            EXPECT_EQ(bad, 0) << tag << ": device Z differs from the host route";
            for (int b = 0; b < kBatch; ++b) {
                ASSERT_EQ(ig[b], 0) << tag << " item " << b;
                const int kb = with_counts ? std::clamp(mg[b], 0, k) : k;
                for (int j = 0; j < k; ++j)
                    for (int i = 0; i < ld; ++i) {
                        const T got = gd[std::size_t(b) * sz + std::size_t(j) * ld + i];
                        if (i >= n) ASSERT_TRUE(same_bits(got, poison<T>())) << tag << " pad row " << i << " item " << b;
                        else if (j >= kb) ASSERT_EQ(got, T(0)) << tag << " column " << j << " past kb, item " << b;
                    }
            }
            dev.release(mark);
        }
    }
}

TYPED_TEST(DeviceApiStebzStein, SteinTierQueries) {
    using T = TypeParam;
    using R = dv::routine::stein;
    for (int n : {1, 2, 33, 300}) {
        std::string got = "none";
        dv::with_tier<R, T>(this->dev.q().get_device(), n, [&]<class Tr>(Tr) { got = std::is_same_v<Tr, Tier> ? "blocked1" : "other"; });
        EXPECT_EQ(got, "blocked1") << n;
        for (int k : {1, n}) {
            const auto sz = dv::stein_buffer_size<Tier, T>(n, k);
            const auto L = dv::stein_detail::layout<T>(n, k);
            EXPECT_EQ(sz, L.size);
            const std::size_t nk = std::size_t(n) * k;
            // Every global slot fits inside one matrix's slice and is aligned for its type.
            for (const auto* s : {&L.dl, &L.dd, &L.du, &L.du2, &L.x}) {
                EXPECT_LE(s->offset + nk * sizeof(T), sz.global_bytes);
                EXPECT_EQ(s->offset % alignof(T), 0u);
            }
            EXPECT_LE(L.piv.offset + nk, sz.global_bytes);
            EXPECT_EQ(sz.global_bytes % sz.align, 0u);
            EXPECT_GE(sz.local_bytes, sizeof(T));
        }
    }
    try {
        dv::with_tier<R, T>(this->dev.q().get_device(), 0, [](auto) {});
        ADD_FAILURE() << "with_tier admitted n = 0";
    } catch (const batchlas::invalid_argument& ex) {
        EXPECT_NE(std::string(ex.what()).find("stein: no device tier admits n=0"), std::string::npos) << ex.what();
    }
}

}  // namespace
