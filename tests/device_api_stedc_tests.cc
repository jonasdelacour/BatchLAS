// SPDX-License-Identifier: MIT
// device_api_stedc_tests: the public device::stedc (include/batchlas/device/lapack/stedc.hh,
// tier::Blocked<1>: in-group divide and conquer, any n) called from a user-style kernel, against
// the host stedc route (design C.10, Blocked tiers: tolerance plus info parity). The device op
// tears every off-diagonal (1 x 1 leaves), the host solves steqr leaves, so the two agree to
// rounding, not bit for bit; the merge bodies (laed2, laed3, laed4) are the same code.
// Guards (docs/developer/agent-guide.md section 8):
//   - eigenvalues against the host route (Check::values, scaled by ||T||_inf), the device
//     vectors by eigen_residual and orthogonality on the dense tridiagonal, the 1-D Laplacian
//     against its analytic spectrum (no self-referential reference)
//   - batch 1024 at period 3, every item's w and Z bit-identical to its representative
//   - d, e at inc 2 and stride + 3, w at stride n + 3, Z at ld = n + 3: pads poisoned with a large
//     finite value and still poisoned on return; d and e unchanged (the op does not write them)
//   - representatives: the Laplacian (e < 0: rho < 0 at every tear), graded (d, e fall by 10 every
//     4 rows, condition ~1e18 at n = 300) and split with repeated eigenvalues (exact deflation)
//   - n = 1, 2, 3, 33, 64, 65, 300: NB = 1 multiples +-1 around the 64-wide gemm tile, and n = 300
//     above the 256 items (each item owns two rows / roots per sub-group of 32)
//   - info parity straddled: max_sec_iter = 1 gives info 1 on both the device op and the host's
//     FusedCta merge (the arm that honours it), the default 50 gives 0 on both
//   - eigenvalues only (J = NoEigenVectors) bit-identical to the vectors run's w
//   - with_tier picks Blocked<1>, filters a 1 KB local memory and a sub-group size of 16, throws
//     "stedc: no device tier admits n=0"; buffer_size equals the carve plan
// Deliberate breaks, one per axis, each red on the 4090 (2026-11-06):
//   merge passes sign 1 to laed2_merge_vector -> StedcMatchesHostRoute (residual)
//   merge skips the final rank sort      -> StedcMatchesHostRoute (values)
//   bottom_up drops the - |e(i)| tear    -> StedcMatchesHostRoute (values, residual)
//   dc_layout carves tmp onto qp         -> StedcMatchesHostRoute (residual)
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
#include <batchlas/device/lapack/stedc.hh>
#include <src/queue.hh>

using namespace batchlas;
namespace dv = batchlas::device;
using batchlas::verify::Check;
using Tier = dv::tier::Blocked<1>;

template <class T, JobType J> struct UserStedc;

namespace {

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr int kInc = 2;
constexpr int kWg = dv::stedc_detail::wg_size;

std::uint64_t mix(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}
double uval(std::uint64_t i) { return static_cast<double>(mix(i) >> 11) * 0x1.0p-53 * 2.0 - 1.0; }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> T poison() { return std::numeric_limits<T>::max() / T(4); }  // large and finite

// Representative r of order n: 0 the 1-D Laplacian (2, -1); 1 graded (d_i and e_i fall by 10
// every 4 rows); 2 split (e_i = 0 every fifth row) with d repeating in pairs.
void rep(int r, int n, std::vector<double>& d, std::vector<double>& e) {
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
double inf_norm(const std::vector<double>& d, const std::vector<double>& e) {
    double m = 0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double l = i > 0 ? std::fabs(e[i - 1]) : 0.0, r = i + 1 < d.size() ? std::fabs(e[i]) : 0.0;
        m = std::max(m, std::fabs(d[i]) + l + r);
    }
    return std::max(m, 1e-300);
}

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
                                   : std::nullopt;
    sycl::queue& q() { return sycl_queue(*ctx); }
    // Shared USM: Check::eigen_residual / orthogonality read the views on the host.
    template <class T> T* up(const std::vector<T>& h) {
        T* p = sycl::malloc_shared<T>(std::max<std::size_t>(h.size(), 1), q());
        if (!h.empty()) q().memcpy(p, h.data(), h.size() * sizeof(T)).wait();
        ptrs.push_back(p);
        return p;
    }
    template <class T> std::vector<T> down(const T* p, std::size_t n) {
        std::vector<T> h(n);
        if (n) q().memcpy(h.data(), p, n * sizeof(T)).wait();
        return h;
    }
    void release(std::size_t mark) {
        for (std::size_t i = mark; i < ptrs.size(); ++i) sycl::free(ptrs[i], q());
        ptrs.resize(mark);
    }
    ~Dev() {
        if (ctx) release(0);
    }
    std::vector<void*> ptrs;
};

// The batch of one n. Device layout: d, e at inc 2 and stride n * 2 + 3, pads poisoned; host
// layout: natural (inc 1, stride n), the route's own copies (it tears d in place).
template <class T> struct Problem {
    int n, sd, se;
    std::vector<T> d, e, dn, en;
    std::vector<std::vector<double>> rd, re;
    explicit Problem(int n_) : n(n_), sd(n_ * kInc + 3), se(std::max(n_ - 1, 1) * kInc + 3) {
        d.assign(std::size_t(kBatch) * sd, poison<T>());
        e.assign(std::size_t(kBatch) * se, poison<T>());
        dn.assign(std::size_t(kBatch) * n, T(0));
        en.assign(std::size_t(kBatch) * std::max(n - 1, 1), T(0));
        rd.resize(kPeriod);
        re.resize(kPeriod);
        for (int r = 0; r < kPeriod; ++r) rep(r, n, rd[r], re[r]);
        for (int b = 0; b < kBatch; ++b) {
            const int r = b % kPeriod;
            for (int i = 0; i < n; ++i) {
                d[std::size_t(b) * sd + i * kInc] = static_cast<T>(rd[r][i]);
                dn[std::size_t(b) * n + i] = static_cast<T>(rd[r][i]);
            }
            for (int i = 0; i + 1 < n; ++i) {
                e[std::size_t(b) * se + i * kInc] = static_cast<T>(re[r][i]);
                en[std::size_t(b) * std::max(n - 1, 1) + i] = static_cast<T>(re[r][i]);
            }
        }
    }
    // The kPeriod representatives as dense n x n tridiagonals (ld n).
    std::vector<T> dense() const {
        std::vector<T> a(std::size_t(kPeriod) * n * n, T(0));
        for (int r = 0; r < kPeriod; ++r)
            for (int i = 0; i < n; ++i) {
                a[std::size_t(r) * n * n + i + std::size_t(i) * n] = static_cast<T>(rd[r][i]);
                if (i + 1 < n) {
                    a[std::size_t(r) * n * n + (i + 1) + std::size_t(i) * n] = static_cast<T>(re[r][i]);
                    a[std::size_t(r) * n * n + i + std::size_t(i + 1) * n] = static_cast<T>(re[r][i]);
                }
            }
        return a;
    }
};

template <class T> class DeviceApiStedc : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
        bool fits = false;
        try {
            dv::with_tier<dv::routine::stedc<>, T>(dev.q().get_device(), 1, [&](auto) { fits = true; });
        } catch (const batchlas::invalid_argument&) {
        }
        if (!fits) GTEST_SKIP() << "no stedc tier fits this device (work-group 256, sub-group 32)";
    }
    Dev dev;
};
using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceApiStedc, RealTypes);

// The device op from a user kernel: WorkGroup<256, 32>, one matrix per work-group; Z may be empty
// (J = NoEigenVectors).
template <class T, JobType J>
void user_stedc(Dev& dev, const VectorView<T>& d, const VectorView<T>& e, const VectorView<T>& w, const MatrixView<T, MatrixFormat::Dense>& Z,
                std::byte* gws, std::int32_t* info, std::int32_t max_sec_iter, int n) {
    const dv::WorkspaceSize sz = dv::stedc_buffer_size<J, Tier, T>(n);
    const auto Zk = Z.kernel_view();
    dv::stedc_detail::Params<T> p;
    p.max_sec_iter = max_sec_iter;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<std::byte, 1> slm(sycl::range<1>(sz.local_bytes), h);
           h.parallel_for<UserStedc<T, J>>(sycl::nd_range<1>(std::size_t(kBatch) * kWg, kWg),
                                           [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                                               const dv::WorkGroup<kWg, 32> ex{it};
                                               const int b = static_cast<int>(it.get_group_linear_id());
                                               const dv::Workspace ws{slm.template get_multi_ptr<sycl::access::decorated::yes>(),
                                                                      dv::global_slice(gws, sz, b)};
                                               int ib = 0;
                                               if constexpr (J == JobType::EigenVectors)
                                                   ib = dv::stedc<J, Tier>(ex, dv::item(d, b), dv::item(e, b), dv::item(w, b), ws, p,
                                                                           dv::item(Zk, b, n, n));
                                               else ib = dv::stedc<J, Tier>(ex, dv::item(d, b), dv::item(e, b), dv::item(w, b), ws, p);
                                               if (ex.leader()) info[b] = ib;
                                           });
       }).wait_and_throw();
}

// One device run on the problem's layouts: w (stride n + 3) and Z (ld n + 3) poisoned first.
template <class T> struct DeviceRun {
    int sw, ld, sz;
    T *w, *z;
    std::int32_t* info;
};
template <class T, JobType J> DeviceRun<T> run_device(Dev& dev, const Problem<T>& P, T* d, T* e, std::int32_t max_sec_iter) {
    const int n = P.n;
    DeviceRun<T> r{n + 3, n + 3, (n + 3) * n + 5, nullptr, nullptr, nullptr};
    r.w = dev.up(std::vector<T>(std::size_t(kBatch) * r.sw, poison<T>()));
    r.z = dev.up(std::vector<T>(std::size_t(kBatch) * r.sz, poison<T>()));
    r.info = dev.up(std::vector<std::int32_t>(kBatch, -7));
    const auto gsz = dv::stedc_buffer_size<J, Tier, T>(n);
    std::byte* gws = dev.up(std::vector<std::byte>(gsz.global_bytes * kBatch, std::byte{0xff}));
    const VectorView<T> dview(d, n, kBatch, kInc, P.sd);
    const VectorView<T> eview(e, std::max(n - 1, 0), kBatch, kInc, P.se);
    user_stedc<T, J>(dev, dview, eview, VectorView<T>(r.w, n, kBatch, 1, r.sw), MatrixView<T, MatrixFormat::Dense>(r.z, n, n, r.ld, r.sz, kBatch),
                     gws, r.info, max_sec_iter, n);
    return r;
}

// The host route on its natural layouts (FusedCta when max_sec_iter must reach the merge).
template <class T> struct HostRun {
    T *w, *z;
    std::int32_t* info;
};
template <class T> HostRun<T> run_host(Dev& dev, const Problem<T>& P, std::int32_t max_sec_iter, bool fused_cta) {
    const int n = P.n, ne = std::max(n - 1, 1);
    HostRun<T> r{};
    T* d = dev.up(P.dn);
    T* e = dev.up(P.en);
    r.w = dev.up(std::vector<T>(std::size_t(kBatch) * n, T(0)));
    std::vector<T> eye(std::size_t(kBatch) * n * n, T(0));
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < n; ++i) eye[std::size_t(b) * n * n + i + std::size_t(i) * n] = T(1);
    r.z = dev.up(eye);
    r.info = dev.up(std::vector<std::int32_t>(kBatch, -7));
    StedcParams<T> sp;
    sp.max_sec_iter = max_sec_iter;
    if (fused_cta) sp.merge_variant = StedcMergeVariant::FusedCta;
    const std::size_t bytes = stedc_buffer_size(*dev.ctx, std::size_t(n), std::size_t(kBatch), JobType::EigenVectors, sp);
    std::byte* hws = dev.up(std::vector<std::byte>(std::max<std::size_t>(bytes, 1)));
    (void)stedc(*dev.ctx, VectorView<T>(d, n, kBatch, 1, n), VectorView<T>(e, std::max(n - 1, 0), kBatch, 1, ne),
                VectorView<T>(r.w, n, kBatch, 1, n), Span<std::byte>(hws, bytes), JobType::EigenVectors, sp,
                MatrixView<T, MatrixFormat::Dense>(r.z, n, n, n, n * n, kBatch), Span<std::int32_t>(r.info, kBatch));
    dev.ctx->wait();
    return r;
}

// ---- device op vs host route ------------------------------------------------------------------

TYPED_TEST(DeviceApiStedc, StedcMatchesHostRoute) {
    using T = TypeParam;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    auto& dev = this->dev;
    for (int n : {1, 2, 3, 33, 64, 65, 300}) {
        const std::size_t mark = dev.ptrs.size();
        Problem<T> P(n);
        T* d = dev.up(P.d);
        T* e = dev.up(P.e);
        const auto D = run_device<T, JobType::EigenVectors>(dev, P, d, e, 50);
        const std::string tag = "n=" + std::to_string(n);

        const auto wd = dev.down(D.w, std::size_t(kBatch) * D.sw), zd = dev.down(D.z, std::size_t(kBatch) * D.sz);
        const auto id = dev.down(D.info, kBatch);
        EXPECT_EQ(dev.down(d, P.d.size()), P.d) << tag << ": d was written";
        EXPECT_EQ(dev.down(e, P.e.size()), P.e) << tag << ": e was written";
        for (int b = 0; b < kBatch; ++b) {
            ASSERT_EQ(id[b], 0) << tag << " item " << b;
            const int r = b % kPeriod;
            for (int i = 0; i < D.sw; ++i) {
                const T got = wd[std::size_t(b) * D.sw + i];
                if (i >= n) ASSERT_TRUE(same_bits(got, poison<T>())) << tag << " w pad " << i << " item " << b;
                else ASSERT_TRUE(same_bits(got, wd[std::size_t(r) * D.sw + i])) << tag << " w(" << i << ") item " << b;
            }
            for (int x = 0; x < D.sz; ++x) {
                const T got = zd[std::size_t(b) * D.sz + x];
                const bool pad = x >= D.ld * n || x % D.ld >= n;
                if (pad) ASSERT_TRUE(same_bits(got, poison<T>())) << tag << " Z pad " << x << " item " << b;
                else ASSERT_TRUE(same_bits(got, zd[std::size_t(r) * D.sz + x])) << tag << " Z[" << x << "] item " << b;
            }
        }
        if (n == 1) {
            for (int r = 0; r < kPeriod; ++r) {
                EXPECT_EQ(wd[std::size_t(r) * D.sw], static_cast<T>(P.rd[r][0])) << tag;
                EXPECT_EQ(zd[std::size_t(r) * D.sz], T(1)) << tag;
            }
            dev.release(mark);
            continue;
        }

        const auto H = run_host<T>(dev, P, 50, false);
        const auto wh = dev.down(H.w, std::size_t(kBatch) * n);
        const auto ih = dev.down(H.info, kBatch);
        for (int r = 0; r < kPeriod; ++r) {
            ASSERT_EQ(ih[r], 0) << tag << " host item " << r;
            double err = 0;
            for (int i = 0; i < n; ++i) err = std::max(err, std::fabs(double(wd[std::size_t(r) * D.sw + i]) - double(wh[std::size_t(r) * n + i])));
            EXPECT_VERIFY(T, Check::values, n, err / inf_norm(P.rd[r], P.re[r])) << tag << " rep " << r << " vs the host route";
        }
        // Representative 0, the Laplacian: lambda_k = 2 - 2 cos(k pi / (n + 1)), ||T|| <= 4.
        double lap = 0;
        for (int k = 0; k < n; ++k)
            lap = std::max(lap, std::fabs(double(wd[k]) - (2.0 - 2.0 * std::cos((k + 1) * std::numbers::pi / (n + 1)))));
        EXPECT_VERIFY(T, Check::values, n, lap / 4.0) << tag << " (Laplacian)";

        T* a = dev.up(P.dense());
        const MV A(a, n, n, n, n * n, kPeriod);
        const MV V(D.z, n, n, D.ld, D.sz, kBatch);
        const std::vector<int> items = {0, 1, 2};
        EXPECT_VERIFY(T, Check::eigen_residual, n, MV::eigen_residual(*dev.ctx, A, V, VectorView<T>(D.w, n, kBatch, 1, D.sw), items)) << tag;
        EXPECT_VERIFY(T, Check::orthogonality, n, MV::orthogonality(*dev.ctx, V, items)) << tag;
        dev.release(mark);
    }
}

// J = NoEigenVectors runs the same merges with Z in the workspace: w bit for bit.
TYPED_TEST(DeviceApiStedc, StedcValuesOnlyMatchesVectorsRun) {
    using T = TypeParam;
    auto& dev = this->dev;
    for (int n : {2, 65, 300}) {
        const std::size_t mark = dev.ptrs.size();
        Problem<T> P(n);
        T* d = dev.up(P.d);
        T* e = dev.up(P.e);
        const auto V = run_device<T, JobType::EigenVectors>(dev, P, d, e, 50);
        const auto N = run_device<T, JobType::NoEigenVectors>(dev, P, d, e, 50);
        const auto wv = dev.down(V.w, std::size_t(kBatch) * V.sw), wn = dev.down(N.w, std::size_t(kBatch) * N.sw);
        int bad = 0;
        for (std::size_t i = 0; i < wv.size(); ++i) bad += !same_bits(wv[i], wn[i]);
        EXPECT_EQ(bad, 0) << "n=" << n << ": NoEigenVectors w differs from EigenVectors w";
        const auto in = dev.down(N.info, kBatch);
        for (int b = 0; b < kBatch; ++b) ASSERT_EQ(in[b], 0) << "n=" << n << " item " << b;
        const auto zn = dev.down(N.z, std::size_t(kBatch) * N.sz);
        for (const T x : zn) ASSERT_TRUE(same_bits(x, poison<T>())) << "n=" << n << ": NoEigenVectors wrote the caller's Z";
        dev.release(mark);
    }
}

// The secular cap reaches info on both sides: 1 step leaves roots unconverged (info 1), the
// default 50 converges (info 0). The host's FusedCta merge is the arm that honours the cap.
TYPED_TEST(DeviceApiStedc, StedcInfoParityStraddlesTheSecularCap) {
    using T = TypeParam;
    auto& dev = this->dev;
    if (dev.ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the host route's FusedCta merge is a GPU arm";
    const int n = 64;
    for (const std::int32_t cap : {1, 50}) {
        const std::size_t mark = dev.ptrs.size();
        Problem<T> P(n);
        T* d = dev.up(P.d);
        T* e = dev.up(P.e);
        const auto D = run_device<T, JobType::EigenVectors>(dev, P, d, e, cap);
        const auto H = run_host<T>(dev, P, cap, true);
        const auto id = dev.down(D.info, kBatch), ih = dev.down(H.info, kBatch);
        const std::int32_t want = cap == 1 ? 1 : 0;
        for (int b = 0; b < kBatch; b += kPeriod) {  // the Laplacian items
            EXPECT_EQ(id[b], want) << "cap " << cap << ": device info, item " << b;
            EXPECT_EQ(ih[b] != 0, want != 0) << "cap " << cap << ": host info, item " << b;
        }
        dev.release(mark);
    }
}

// ---- tier queries -----------------------------------------------------------------------------

TYPED_TEST(DeviceApiStedc, StedcTierQueries) {
    using T = TypeParam;
    using R = dv::routine::stedc<JobType::EigenVectors>;
    using RN = dv::routine::stedc<JobType::NoEigenVectors>;
    static_assert(std::is_same_v<dv::resolve_tier_t<R, T, dv::tier::upto<1 << 20>>, Tier>);
    const auto lim = dv::DeviceLimits::of(this->dev.q().get_device());
    for (int n : {1, 2, 65, 300, 5000}) {
        std::string got = "none";
        dv::with_tier<R, T>(lim, n, [&]<class Tr>(Tr) { got = std::is_same_v<Tr, Tier> ? "blocked1" : "other"; });
        EXPECT_EQ(got, "blocked1") << n;
        const auto sz = dv::stedc_buffer_size<JobType::EigenVectors, Tier, T>(n);
        const auto szn = dv::stedc_buffer_size<JobType::NoEigenVectors, Tier, T>(n);
        const auto L = dv::stedc_detail::dc_layout<T>(n, false);
        EXPECT_EQ(sz, L.size);
        EXPECT_EQ(szn, dv::stedc_detail::dc_layout<T>(n, true).size);
        EXPECT_GE(szn.global_bytes, sz.global_bytes + std::size_t(n) * n * sizeof(T)) << n;
        EXPECT_GE(sz.local_bytes, dv::l3::smem_bytes<dv::stedc_detail::gemm_cfg<T>>() + sizeof(std::int32_t));
        const std::size_t nn = std::size_t(n) * n;
        for (const auto* s : {&L.qp, &L.tmp}) {
            EXPECT_LE(s->offset + nn * sizeof(T), sz.global_bytes) << n;
            EXPECT_EQ(s->offset % alignof(T), 0u);
        }
        EXPECT_LE(L.rank.offset + std::size_t(n) * sizeof(std::int32_t), sz.global_bytes) << n;
        EXPECT_EQ(sz.global_bytes % sz.align, 0u);
    }
    // Filtered: a local memory below the gemm's staging, and no sub-group of 32.
    for (const bool small_slm : {true, false}) {
        auto mock = lim;
        if (small_slm) mock.local_mem_bytes = 1024;
        else mock.sub_group_sizes = {16};
        try {
            dv::with_tier<RN, T>(mock, 64, [](auto) {});
            ADD_FAILURE() << "with_tier admitted a device without " << (small_slm ? "the local memory" : "sub-group 32");
        } catch (const batchlas::invalid_argument& ex) {
            EXPECT_NE(std::string(ex.what()).find("stedc: no device tier admits n=64"), std::string::npos) << ex.what();
        }
    }
    try {
        dv::with_tier<R, T>(lim, 0, [](auto) {});
        ADD_FAILURE() << "with_tier admitted n = 0";
    } catch (const batchlas::invalid_argument& ex) {
        EXPECT_NE(std::string(ex.what()).find("stedc: no device tier admits n=0"), std::string::npos) << ex.what();
    }
}

}  // namespace
