// SPDX-License-Identifier: MIT
// device_api_steqr_tests: device::steqr<J, tier::Cta<N>> (include/batchlas/device/lapack/steqr.hh)
// called from a user-style kernel against the host steqr_cta route, the thin launcher over the
// same body (steqr_cta is the route itself: no choice.hh, nothing to pin; the kernel is
// SteqrCTAKernel, reached on a GPU only: elsewhere steqr_cta falls back to steqr_wg, so the test
// skips). The tier comes from with_tier on the run-time n.
// Guards (agent-guide section 8):
//   - bit-identity (memcmp) of the eigenvalues, Q (with vectors) and info for float and double,
//     both jobz, n at every tier boundary (1, 2, 4, 5, 8, 9, 16, 17, 31, 32 = the ceiling), the
//     three shift / update combinations and a one-sweep budget (info = 1 must occur: straddled);
//   - batch 1024 (32 / N problems per sub-group, the Q tiles at non-zero local offsets) with
//     period-3 items: uniform, graded (d, e scaled by 2^-3i) and a split straddle (one e at 1/16
//     and one at 16x the dsteqr negligibility bound, one exact zero);
//   - Q with ld = n + 3, NaN in the pad rows (untouched) and a non-identity start matrix (the
//     rotations accumulate into it: back_transform = true on the host);
//   - with_tier picks Cta<4|8|16|32> and throws "steqr: ..." past 32; buffer_size equals N^2 T
//     with vectors and 0 without; n > N returns -2 and writes nothing.
// Deliberate break (a negative control that runs every time): the device call with the other
// shift strategy must differ from the host route, so the comparison can see a wrong knob.
#include <tests/device_core_common.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/extensions.hh>
#include <batchlas/device/lapack/steqr.hh>
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
constexpr int kNs[] = {1, 2, 4, 5, 8, 9, 16, 17, 31, 32};

template <JobType J> using Rt = dv::routine::steqr<J>;

template <class T> T nan_v() { return std::numeric_limits<T>::quiet_NaN(); }

struct Case {
    SteqrShiftStrategy shift;
    SteqrUpdateScheme update;
    int max_sweeps;
};
constexpr Case kCases[] = {{SteqrShiftStrategy::Lapack, SteqrUpdateScheme::EXP, 50},
                           {SteqrShiftStrategy::Wilkinson, SteqrUpdateScheme::EXP, 50},
                           {SteqrShiftStrategy::Lapack, SteqrUpdateScheme::PG, 50},
                           {SteqrShiftStrategy::Lapack, SteqrUpdateScheme::EXP, 1}};

// Representative r of item b (b % kPeriod): 0 uniform, 1 graded, 2 the split straddle.
template <class T> void fill(int n, std::vector<T>& d, std::vector<T>& e, std::vector<T>& q, int ld) {
    const int nm1 = n > 1 ? n - 1 : 0;
    d.assign(std::size_t(n) * kBatch, T(0));
    e.assign(std::size_t(nm1) * kBatch + 1, T(0));
    q.assign(std::size_t(ld) * n * kBatch, nan_v<T>());
    // dsteqr's EPS2 (eps = lamch('E') = epsilon / 2), the negligibility bound's scale.
    const T eps = std::numeric_limits<T>::epsilon() / 2;
    for (int b = 0; b < kBatch; ++b) {
        const int r = b % kPeriod;
        const std::uint64_t s = std::uint64_t(r) * 1000003u;
        T* db = d.data() + std::size_t(b) * n;
        T* eb = e.data() + std::size_t(b) * nm1;
        for (int i = 0; i < n; ++i) db[i] = val<T>(s + i) + (r == 1 ? T(0) : T(0.25));
        for (int i = 0; i < nm1; ++i) eb[i] = val<T>(s + 500 + i);
        if (r == 1) {
            for (int i = 0; i < n; ++i) db[i] *= std::ldexp(T(1), -3 * i);
            for (int i = 0; i < nm1; ++i) eb[i] *= std::ldexp(T(1), -3 * i - 1);
        }
        if (r == 2 && n >= 4) {
            const auto bound = [&](int j) { return eps * std::sqrt(std::abs(db[j]) * std::abs(db[j + 1])); };
            eb[nm1 / 3] = bound(nm1 / 3) / 4;               // e^2 = bound^2 / 16: negligible
            eb[(2 * nm1) / 3] = bound((2 * nm1) / 3) * 4;   // 16 x: not negligible
            eb[nm1 / 2] = T(0);
        }
        for (int c = 0; c < n; ++c)
            for (int i = 0; i < n; ++i) q[std::size_t(b) * ld * n + i + std::size_t(c) * ld] = val<T>(s + 9000 + i + 37u * c);
    }
}

template <JobType J, class Tier, class T> class ApiSteqrKernel;

// The user's kernel: one problem per unmasked partition of N lanes (every lane of the sub-group
// calls), Q at this partition's slice of the work-group's local memory.
template <JobType J, class Tier, class T>
void launch_device(Dev& dev, int n, int ld, T* d, T* e, T* q, int* info, dv::steqr_opts opts) {
    constexpr int N = Tier::n_max;
    constexpr int ppw = 32 / N;
    constexpr std::size_t lbytes = dv::steqr_buffer_size<J, Tier, T>(N).local_bytes;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<sycl::uint4, 1> slm(sycl::range<1>(std::max<std::size_t>((lbytes * ppw + 15) / 16, 1)), h);
           h.parallel_for<ApiSteqrKernel<J, Tier, T>>(
               sycl::nd_range<1>(std::size_t(kBatch / ppw) * 32, 32), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                   const auto part = make_partition<N, false>(it.get_sub_group());
                   const auto ex = dv::wrap(part);
                   const int slot = static_cast<int>(part.get_group_linear_id());
                   const int b = static_cast<int>(it.get_group_linear_id()) * ppw + slot;
                   std::byte* raw = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get_raw());
                   const dv::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                              raw + std::size_t(slot) * lbytes),
                                          nullptr};
                   const int nm1 = n > 1 ? n - 1 : 0;
                   const int r = dv::steqr<J, Tier>(ex, dv::vec(d + std::size_t(b) * n, n), dv::vec(e + std::size_t(b) * nm1, nm1),
                                                    dv::mat(q + std::size_t(b) * ld * n, n, n, ld), ws, opts);
                   if (ex.leader()) info[b] = r;
               });
       }).wait_and_throw();
}

template <class T> class DeviceApiSteqrTest : public DeviceCoreTest<T> {
  protected:
    void SetUp() override {
        DeviceCoreTest<T>::SetUp();
        if (this->IsSkipped()) return;
        if (!this->dev.q().get_device().has(sycl::aspect::gpu)) GTEST_SKIP() << "steqr_cta's CTA kernel runs on GPUs only";
    }

  public:
    struct Result {
        std::vector<T> w, q;
        std::vector<int> info;
    };

    Result host(int n, int ld, JobType jobz, const Case& c, const std::vector<T>& d0, const std::vector<T>& e0, const std::vector<T>& q0) {
        Dev& dev = this->dev;
        const int nm1 = n > 1 ? n - 1 : 0;
        T* d = dev.up(d0);
        T* e = dev.up(e0);
        T* q = dev.up(q0);
        T* w = dev.alloc<T>(std::size_t(n) * kBatch, nan_v<T>());
        int* info = dev.alloc<int>(kBatch, 0);
        VectorView<T> dv_(d, n, kBatch, 1, n), ev(e, nm1, kBatch, 1, nm1), wv(w, n, kBatch, 1, n);
        MatrixView<T, MatrixFormat::Dense> Qv(q, n, n, ld, ld * n, kBatch);
        SteqrParams<T> p;
        p.sort = false;
        p.back_transform = true;
        p.max_sweeps = static_cast<std::size_t>(c.max_sweeps);
        p.cta_shift_strategy = c.shift;
        p.cta_update_scheme = c.update;
        const std::size_t bytes = steqr_cta_buffer_size<T>(*dev.ctx, dv_, ev, wv, jobz, p);
        std::byte* ws = dev.alloc<std::byte>(bytes, std::byte{0});
        (void)steqr_cta<kB, T>(*dev.ctx, dv_, ev, wv, Span<std::byte>(ws, bytes), jobz, p, Qv, Span<int32_t>(info, kBatch));
        dev.ctx->wait();
        return {dev.down(w, std::size_t(n) * kBatch), dev.down(q, q0.size()), dev.down(info, kBatch)};
    }

    template <JobType J>
    Result device(int n, int ld, dv::steqr_opts o, const std::vector<T>& d0, const std::vector<T>& e0, const std::vector<T>& q0) {
        Dev& dev = this->dev;
        T* d = dev.up(d0);
        T* e = dev.up(e0);
        T* q = dev.up(q0);
        int* info = dev.alloc<int>(kBatch, 7);
        dv::with_tier<Rt<J>, T>(dev.q().get_device(), n,
                                [&]<class Tier>(Tier) { launch_device<J, Tier, T>(dev, n, ld, d, e, q, info, o); });
        return {dev.down(d, std::size_t(n) * kBatch), dev.down(q, q0.size()), dev.down(info, kBatch)};
    }
};

using RealTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(DeviceApiSteqrTest, RealTypes);

template <class T> int bitdiff(const std::vector<T>& x, const std::vector<T>& y) {
    int k = 0;
    for (std::size_t i = 0; i < x.size() && i < y.size(); ++i) k += !same_bits(x[i], y[i]);
    return k + static_cast<int>(x.size() != y.size());
}

template <class T, JobType J, class TF> void bit_identity(TF& t) {
    for (const Case& c : kCases) {
        int failures = 0;
        for (int n : kNs) {
            SCOPED_TRACE("n=" + std::to_string(n) + " shift=" + std::to_string(int(c.shift)) + " update=" +
                         std::to_string(int(c.update)) + " sweeps=" + std::to_string(c.max_sweeps));
            const int ld = n + 3;
            std::vector<T> d0, e0, q0;
            fill<T>(n, d0, e0, q0, ld);
            const auto h = t.host(n, ld, J, c, d0, e0, q0);
            const auto g = t.template device<J>(n, ld, dv::steqr_opts{c.max_sweeps, c.shift, c.update}, d0, e0, q0);
            EXPECT_EQ(bitdiff(g.w, h.w), 0);
            EXPECT_EQ(bitdiff(g.info, h.info), 0);
            if constexpr (J == JobType::EigenVectors) EXPECT_EQ(bitdiff(g.q, h.q), 0);  // pads included
            else EXPECT_EQ(bitdiff(g.q, q0), 0);                                       // z never touched
            failures += static_cast<int>(std::count(h.info.begin(), h.info.end(), 1));
        }
        // The one-sweep budget must fail somewhere, the full budget nowhere (info parity both ways).
        if (c.max_sweeps == 1) EXPECT_GT(failures, 0);
        else EXPECT_EQ(failures, 0);
    }
}

TYPED_TEST(DeviceApiSteqrTest, CtaBitIdenticalToHostRouteVectors) { bit_identity<TypeParam, JobType::EigenVectors>(*this); }
TYPED_TEST(DeviceApiSteqrTest, CtaBitIdenticalToHostRouteValues) { bit_identity<TypeParam, JobType::NoEigenVectors>(*this); }

// The negative control: the other shift strategy cannot match the host's bits.
TYPED_TEST(DeviceApiSteqrTest, WrongShiftIsSeen) {
    using T = TypeParam;
    const int n = 17, ld = n + 3;
    std::vector<T> d0, e0, q0;
    fill<T>(n, d0, e0, q0, ld);
    const Case c = kCases[0];
    const auto h = this->host(n, ld, JobType::EigenVectors, c, d0, e0, q0);
    const auto g = this->template device<JobType::EigenVectors>(
        n, ld, dv::steqr_opts{c.max_sweeps, SteqrShiftStrategy::Wilkinson, c.update}, d0, e0, q0);
    EXPECT_GT(bitdiff(g.q, h.q), 0);
}

TYPED_TEST(DeviceApiSteqrTest, ArgumentCheckReturnsMinus2AndWritesNothing) {
    using T = TypeParam;
    const int n = 5, ld = n + 3;  // tier::Cta<4> cannot hold n = 5
    std::vector<T> d0, e0, q0;
    fill<T>(n, d0, e0, q0, ld);
    Dev& dev = this->dev;
    T* d = dev.up(d0);
    T* e = dev.up(e0);
    T* q = dev.up(q0);
    int* info = dev.alloc<int>(kBatch, 7);
    launch_device<JobType::EigenVectors, dv::tier::Cta<4>, T>(dev, n, ld, d, e, q, info, {});
    const auto got = dev.down(info, kBatch);
    EXPECT_EQ(std::count(got.begin(), got.end(), -2), kBatch);
    EXPECT_EQ(bitdiff(dev.down(d, d0.size()), d0), 0);
    EXPECT_EQ(bitdiff(dev.down(q, q0.size()), q0), 0);
}

TEST(DeviceApiSteqrTier, WithTierPicksTheSmallestCtaAndThrowsPast32) {
    const dv::DeviceLimits lim{48 * 1024, {32}, 1024, true};
    for (int n = 1; n <= 32; ++n) {
        int got = 0;
        dv::with_tier<Rt<JobType::EigenVectors>, double>(lim, n, [&]<class Tier>(Tier) { got = Tier::n_max; });
        EXPECT_EQ(got, n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32) << "n=" << n;
    }
    try {
        dv::with_tier<Rt<JobType::NoEigenVectors>, float>(lim, 33, [&]<class Tier>(Tier) {});
        ADD_FAILURE() << "with_tier admitted n = 33";
    } catch (const batchlas::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("steqr: no device tier admits n=33"), std::string::npos) << e.what();
    }
    // No fp64: a double problem has no tier.
    const dv::DeviceLimits no64{48 * 1024, {32}, 1024, false};
    EXPECT_THROW((dv::with_tier<Rt<JobType::EigenVectors>, double>(no64, 4, [&]<class Tier>(Tier) {})), batchlas::invalid_argument);
}

TEST(DeviceApiSteqrTier, BufferSizeIsQOnlyWithVectors) {
    EXPECT_EQ((dv::steqr_buffer_size<JobType::EigenVectors, dv::tier::Cta<4>, float>(4).local_bytes), 4u * 4u * sizeof(float));
    EXPECT_EQ((dv::steqr_buffer_size<JobType::EigenVectors, dv::tier::Cta<32>, double>(32).local_bytes), 32u * 32u * sizeof(double));
    EXPECT_EQ((dv::steqr_buffer_size<JobType::NoEigenVectors, dv::tier::Cta<16>, float>(16).local_bytes), 0u);
    EXPECT_EQ((dv::steqr_buffer_size<JobType::EigenVectors, dv::tier::Cta<8>, float>(8).global_bytes), 0u);
    static_assert(dv::steqr_buffer_size<JobType::EigenVectors, dv::tier::upto<9>, float>(9) ==
                  dv::steqr_buffer_size<JobType::EigenVectors, dv::tier::Cta<16>, float>(9));
}

}  // namespace
