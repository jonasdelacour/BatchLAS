// SPDX-License-Identifier: MIT
// device_api_sytrd_tests: device::sytrd<UL, tier::Cta<N>> (include/batchlas/device/lapack/sytrd.hh)
// called from a user-style kernel against the host sytrd_cta route, the thin launcher over the
// same body (sytrd_cta is the route itself: no choice.hh, nothing to pin; the kernel is
// Sytd2CTAKernel). The tier comes from with_tier on the run-time n, as a user would pick it.
// Guards (agent-guide section 8):
//   - bit-identity (memcmp) of A, d, e, tau and info for float, double and both complex types,
//     both triangles, n at every tier boundary (1, 4, 5, 8, 9, 16, 17, 31, 32 = the ceiling);
//   - batch 1024 (32 / N problems per sub-group: every tile but the first at a non-zero local
//     offset) with period-3 items, each item bit-identical to its representative;
//   - ld = n + 3 with NaN poison in the pad rows (must come back untouched) and in the triangle
//     sytrd must not read (overwritten, never read: a read leaks NaN into d / e);
//   - complex Im(diag) != 0 (dropped, as {c,z}hetrd does);
//   - with_tier picks Cta<4|8|16|32> and throws "sytrd: ..." past 32; buffer_size equals the
//     independent count (N^2 + 2N) T; n > N returns -2 and writes nothing.
// Deliberate break (a negative control that runs every time): the device call with the OTHER
// triangle must differ from the host route, so the comparison can see a wrong UL.
#include <tests/device_core_common.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/extensions.hh>
#include <batchlas/device/lapack/sytrd.hh>
#include <batchlas/device/sg_partition.hh>

#include <cmath>
#include <limits>
#include <string>

using namespace device_core_test;

namespace {

#if BATCHLAS_HAS_CUDA_BACKEND
constexpr Backend kB = Backend::CUDA;
#elif BATCHLAS_HAS_ROCM_BACKEND
constexpr Backend kB = Backend::ROCM;
#else
constexpr Backend kB = Backend::NETLIB;
#endif

constexpr int kBatch = 1024;
constexpr int kPeriod = 3;
constexpr int kNs[] = {1, 4, 5, 8, 9, 16, 17, 31, 32};

template <Uplo UL> using Rt = dv::routine::sytrd<UL>;

template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// Item b repeats representative b % kPeriod. UL's triangle holds data (complex Im(diag) != 0),
// the other triangle and the pad rows hold NaN.
template <class T> std::vector<T> input(int n, int ld, Uplo ul) {
    std::vector<T> a(std::size_t(ld) * n * kBatch, nan_v<T>());
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                if (ul == Uplo::Upper ? r <= c : r >= c)
                    a[std::size_t(b) * ld * n + r + std::size_t(c) * ld] = val<T>(std::uint64_t(b % kPeriod) * 1000003u + r + 97u * c);
    return a;
}

template <Uplo UL, class Tier, class T> class ApiSytrdKernel;

// The user's kernel: one problem per partition of N lanes, the tile in the work-group's local
// memory at this partition's slice, the tier fixed at compile time.
template <Uplo UL, class Tier, class T> void launch_device(Dev& dev, T* A, int n, int ld, T* d, T* e, T* tau, int* info) {
    constexpr int N = Tier::n_max;
    constexpr int ppw = 32 / N;
    constexpr std::size_t lbytes = dv::sytrd_buffer_size<UL, Tier, T>(N).local_bytes;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<sycl::uint4, 1> slm(sycl::range<1>((lbytes * ppw + 15) / 16), h);
           h.parallel_for<ApiSytrdKernel<UL, Tier, T>>(
               sycl::nd_range<1>(std::size_t(kBatch / ppw) * 32, 32), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                   const auto part = make_partition<N>(it.get_sub_group());
                   const auto ex = dv::wrap(part);
                   const int slot = static_cast<int>(part.get_group_linear_id());
                   const int b = static_cast<int>(it.get_group_linear_id()) * ppw + slot;
                   std::byte* raw = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get_raw());
                   const dv::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                              raw + std::size_t(slot) * lbytes),
                                          nullptr};
                   const int nm1 = n > 1 ? n - 1 : 0;
                   const int r = dv::sytrd<UL, Tier>(ex, dv::mat(A + std::size_t(b) * ld * n, n, n, ld), dv::vec(d + std::size_t(b) * n, n),
                                                     dv::vec(e + std::size_t(b) * nm1, nm1), dv::vec(tau + std::size_t(b) * nm1, nm1), ws);
                   if (ex.leader()) info[b] = r;
               });
       }).wait_and_throw();
}

template <class T> class DeviceApiSytrdTest : public DeviceCoreTest<T> {
  protected:
    void SetUp() override {
        DeviceCoreTest<T>::SetUp();
        if (this->IsSkipped()) return;
        // The CTA routes build without CPU targets (NO_CPU_TARGETS): only a GPU reaches them.
        if (!this->dev.q().get_device().has(sycl::aspect::gpu)) GTEST_SKIP() << "sytrd_cta runs on GPUs only";
    }

  public:
    struct Result {
        std::vector<T> a, d, e, tau;
        std::vector<int> info;
    };

    Result host(int n, Uplo ul, const std::vector<T>& a0, int ld) {
        Dev& dev = this->dev;
        const int nm1 = n > 1 ? n - 1 : 0;
        T* A = dev.up(a0);
        T* d = dev.alloc<T>(std::size_t(n) * kBatch, nan_v<T>());
        T* e = dev.alloc<T>(std::size_t(nm1) * kBatch + 1, nan_v<T>());
        T* tau = dev.alloc<T>(std::size_t(nm1) * kBatch + 1, nan_v<T>());
        MatrixView<T, MatrixFormat::Dense> Av(A, n, n, ld, ld * n, kBatch);
        VectorView<T> dv_(d, n, kBatch, 1, n), ev(e, nm1, kBatch, 1, nm1), tv(tau, nm1, kBatch, 1, nm1);
        std::vector<std::byte> none(1);
        (void)sytrd_cta<kB, T>(*dev.ctx, Av, dv_, ev, tv, ul, Span<std::byte>(none.data(), 1), 1);
        dev.ctx->wait();
        // e and tau come back with their NaN end pad: n = 1 would be a zero-byte copy, which Metal rejects.
        return {dev.down(A, a0.size()), dev.down(d, std::size_t(n) * kBatch), dev.down(e, std::size_t(nm1) * kBatch + 1),
                dev.down(tau, std::size_t(nm1) * kBatch + 1), std::vector<int>(kBatch, 0)};
    }

    template <Uplo UL> Result device(int n, const std::vector<T>& a0, int ld) {
        Dev& dev = this->dev;
        const int nm1 = n > 1 ? n - 1 : 0;
        T* A = dev.up(a0);
        T* d = dev.alloc<T>(std::size_t(n) * kBatch, nan_v<T>());
        T* e = dev.alloc<T>(std::size_t(nm1) * kBatch + 1, nan_v<T>());
        T* tau = dev.alloc<T>(std::size_t(nm1) * kBatch + 1, nan_v<T>());
        int* info = dev.alloc<int>(kBatch, 7);
        dv::with_tier<Rt<UL>, T>(dev.q().get_device(), n,
                                 [&]<class Tier>(Tier) { launch_device<UL, Tier, T>(dev, A, n, ld, d, e, tau, info); });
        return {dev.down(A, a0.size()), dev.down(d, std::size_t(n) * kBatch), dev.down(e, std::size_t(nm1) * kBatch + 1),
                dev.down(tau, std::size_t(nm1) * kBatch + 1), dev.down(info, kBatch)};
    }
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceApiSytrdTest, Types);

template <class T> int bitdiff(const std::vector<T>& x, const std::vector<T>& y) {
    int k = 0;
    for (std::size_t i = 0; i < x.size(); ++i) k += !same_bits(x[i], y[i]);
    return k + static_cast<int>(x.size() != y.size());
}
template <class T> int period_diff(const std::vector<T>& g, std::size_t per) {
    int k = 0;
    for (std::size_t b = kPeriod; b < std::size_t(kBatch); ++b)
        for (std::size_t i = 0; i < per; ++i) k += !same_bits(g[b * per + i], g[(b % kPeriod) * per + i]);
    return k;
}

template <class T, Uplo UL, class TF> void bit_identity(TF& t) {
    for (int n : kNs) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3;
        const auto a0 = input<T>(n, ld, UL);
        const auto h = t.host(n, UL, a0, ld);
        const auto g = t.template device<UL>(n, a0, ld);
        EXPECT_EQ(bitdiff(g.a, h.a), 0);
        EXPECT_EQ(bitdiff(g.d, h.d), 0);
        EXPECT_EQ(bitdiff(g.e, h.e), 0);
        EXPECT_EQ(bitdiff(g.tau, h.tau), 0);
        EXPECT_EQ(bitdiff(g.info, h.info), 0);
        EXPECT_EQ(period_diff(g.a, std::size_t(ld) * n), 0);
        // The pad rows come back untouched; d is finite (the poisoned triangle was never read).
        int pad = 0, nonfinite = 0;
        for (int b = 0; b < kBatch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = n; r < ld; ++r) {
                    const std::size_t k = std::size_t(b) * ld * n + r + std::size_t(c) * ld;
                    pad += !same_bits(g.a[k], a0[k]);
                }
        for (const T& x : g.d) nonfinite += !std::isfinite(std::abs(x));
        EXPECT_EQ(pad, 0);
        EXPECT_EQ(nonfinite, 0);
    }
}

TYPED_TEST(DeviceApiSytrdTest, CtaBitIdenticalToHostRouteLower) { bit_identity<TypeParam, Uplo::Lower>(*this); }
TYPED_TEST(DeviceApiSytrdTest, CtaBitIdenticalToHostRouteUpper) { bit_identity<TypeParam, Uplo::Upper>(*this); }

// The negative control: reading the other triangle (NaN-poisoned here) cannot match the host.
TYPED_TEST(DeviceApiSytrdTest, WrongTriangleIsSeen) {
    using T = TypeParam;
    const int n = 17, ld = n + 3;
    const auto a0 = input<T>(n, ld, Uplo::Lower);
    const auto h = this->host(n, Uplo::Lower, a0, ld);
    const auto g = this->template device<Uplo::Upper>(n, a0, ld);
    EXPECT_GT(bitdiff(g.d, h.d), 0);
}

TYPED_TEST(DeviceApiSytrdTest, ArgumentCheckReturnsMinus2AndWritesNothing) {
    using T = TypeParam;
    const int n = 5, ld = n + 3;  // tier::Cta<4> cannot hold n = 5
    const auto a0 = input<T>(n, ld, Uplo::Lower);
    Dev& dev = this->dev;
    T* A = dev.up(a0);
    T* d = dev.alloc<T>(std::size_t(n) * kBatch, nan_v<T>());
    T* e = dev.alloc<T>(std::size_t(n - 1) * kBatch, nan_v<T>());
    T* tau = dev.alloc<T>(std::size_t(n - 1) * kBatch, nan_v<T>());
    int* info = dev.alloc<int>(kBatch, 7);
    launch_device<Uplo::Lower, dv::tier::Cta<4>, T>(dev, A, n, ld, d, e, tau, info);
    const auto got = dev.down(info, kBatch);
    EXPECT_EQ(std::count(got.begin(), got.end(), -2), kBatch);
    EXPECT_EQ(bitdiff(dev.down(A, a0.size()), a0), 0);
}

TEST(DeviceApiSytrdTier, WithTierPicksTheSmallestCtaAndThrowsPast32) {
    const dv::DeviceLimits lim{48 * 1024, {32}, 1024, true};
    for (int n = 1; n <= 32; ++n) {
        int got = 0;
        dv::with_tier<Rt<Uplo::Lower>, float>(lim, n, [&]<class Tier>(Tier) { got = Tier::n_max; });
        EXPECT_EQ(got, n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32) << "n=" << n;
    }
    try {
        dv::with_tier<Rt<Uplo::Upper>, double>(lim, 33, [&]<class Tier>(Tier) {});
        ADD_FAILURE() << "with_tier admitted n = 33";
    } catch (const batchlas::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("sytrd: no device tier admits n=33"), std::string::npos) << e.what();
    }
    // No sub-group size 32: nothing admits.
    const dv::DeviceLimits sg16{48 * 1024, {16}, 1024, true};
    EXPECT_THROW((dv::with_tier<Rt<Uplo::Lower>, float>(sg16, 4, [&]<class Tier>(Tier) {})), batchlas::invalid_argument);
}

template <int N, class T> void expect_size() {
    const auto s = dv::sytrd_buffer_size<Uplo::Lower, dv::tier::Cta<N>, T>(N);
    EXPECT_EQ(s.local_bytes, (std::size_t(N) * N + 2 * N) * sizeof(T)) << "N=" << N;
    EXPECT_EQ(s.global_bytes, 0u);
    static_assert(dv::sytrd_buffer_size<Uplo::Upper, dv::tier::upto<N>, T>(N) == dv::sytrd_buffer_size<Uplo::Upper, dv::tier::Cta<N>, T>(N));
}
TEST(DeviceApiSytrdTier, BufferSizeIsTileVW) {
    expect_size<4, float>();
    expect_size<8, double>();
    expect_size<16, std::complex<float>>();
    expect_size<32, std::complex<double>>();
}

}  // namespace
