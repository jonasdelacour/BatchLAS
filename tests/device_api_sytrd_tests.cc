// SPDX-License-Identifier: MIT
// device_api_sytrd_tests: device::sytrd<UL, tier::Cta<N>> and sytrd<Lower, tier::Blocked<NB>>
// (include/batchlas/device/lapack/sytrd.hh; the Blocked cases are described above their section)
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
//   - with_tier picks Cta<4|8|16|32> and throws "sytrd: ..." past 32 for Upper (Lower goes on to
//     Blocked<32>); buffer_size equals the
//     independent count (N^2 + 2N) T; n > N returns -2 and writes nothing.
// Deliberate break (a negative control that runs every time): the device call with the OTHER
// triangle must differ from the host route, so the comparison can see a wrong UL.
#include <tests/device_core_common.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/device/lapack/steqr.hh>
#include <batchlas/device/lapack/sytrd.hh>
#include <batchlas/device/sg_partition.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

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
        // Every listed tier is instantiated; n <= 32 must reach a Cta one (Blocked is tested below).
        dv::with_tier<Rt<UL>, T>(dev.q().get_device(), n, [&]<class Tier>(Tier) {
            if constexpr (Tier::n_max > 0) launch_device<UL, Tier, T>(dev, A, n, ld, d, e, tau, info);
            else ADD_FAILURE() << "with_tier picked tier::Blocked for n = " << n;
        });
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

// ---- tier::Blocked<NB>: Uplo::Lower, any n, one matrix per WorkGroup<256> ------------------------
// device::sytrd<Lower, tier::Blocked<32>> runs the host sytrd_blocked's panel loop in one
// work-group: the SAME xLATRD body (the host latrd legacy kernel is its thin launcher), the
// trailing update on the in-group her2k instead of the host's her2k / syr2k / gemm pair (and the
// host may take the grid latrd at large n), so the comparison is a tolerance one:
//   - d and e against the host sytrd_blocked (block 32: the same panel boundaries) within 64 n eps
//     ||A||_F, n across NB multiples +-1 (33, 63, 64, 65, 97) and 160 (five panels);
//   - an independent residual: Q^H A0 Q - T and Q^H Q - I in long double from the device's
//     reflectors and tau, Q = H(0) ... H(n-2) (not the host route: no self-reference);
//   - batch 1024 at n <= 65 (the SLM stage and the per-matrix global slice at every offset), each
//     item bit-identical to its period-3 representative; ld = n + 3 with NaN pads, the strictly
//     upper triangle past the superdiagonal NaN and untouched, d finite;
//   - complex Im(diag) != 0 gives the bits of Im(diag) == 0 (dropped, as zhetrd);
//   - one user kernel chaining sytrd<Blocked> and steqr<NoEigenVectors, Blocked<1>> (the in-group
//     syev values path: Re(d), |e|) against the host syev's eigenvalues; info 0;
//   - with_tier picks Blocked<32> past 32 for Lower; buffer_size equals the independent count.
// Deliberate break (runs every time): the residual of the device output against A0 with one
// element moved by 1e-2 must exceed the tolerance, so the residual check can see a wrong matrix.
template <class T> using LDm = std::vector<LD<T>>;

constexpr int kNB = 32;
using BlockedTier = dv::tier::Blocked<kNB>;
constexpr int kBlockedNs[] = {33, 63, 64, 65, 97, 160};
int blocked_batch(int n) { return n <= 65 ? kBatch : 96; }

template <class T> T conj_of(T x) {
    if constexpr (is_cplx<T>) return std::conj(x);
    else return x;
}

// Item b repeats representative b % kPeriod; the lower triangle holds data (Im(diag) = im_diag
// times a value), the rest and the pad rows NaN.
template <class T> std::vector<T> input_lower(int n, int ld, int batch, bool im_diag) {
    std::vector<T> a(std::size_t(ld) * n * batch, nan_v<T>());
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = c; r < n; ++r) {
                T v = val<T>(std::uint64_t(b % kPeriod) * 1000003u + r + 97u * c);
                if (r == c) {
                    if constexpr (is_cplx<T>) v = T(v.real(), im_diag ? v.imag() : real_of<T>(0));
                }
                a[std::size_t(b) * ld * n + r + std::size_t(c) * ld] = v;
            }
    return a;
}

template <class T> struct BlockedOut {
    std::vector<T> a, d, e, tau;
    std::vector<int> info;
};

template <class T> BlockedOut<T> blocked_device(Dev& dev, int n, int ld, int batch, const std::vector<T>& a0) {
    using R = Rt<Uplo::Lower>;
    constexpr int WG = dv::tier_info<R, BlockedTier, T>::wg_size;
    const int nm1 = n - 1;
    T* A = dev.up(a0);
    T* d = dev.alloc<T>(std::size_t(n) * batch, nan_v<T>());
    T* e = dev.alloc<T>(std::size_t(nm1) * batch + 1, nan_v<T>());
    T* tau = dev.alloc<T>(std::size_t(nm1) * batch + 1, nan_v<T>());
    int* info = dev.alloc<int>(batch, 7);
    const dv::WorkspaceSize sz = dv::sytrd_buffer_size<Uplo::Lower, BlockedTier, T>(n);
    std::byte* g = dev.alloc<std::byte>(sz.global_bytes * std::size_t(batch) + sz.align, std::byte{0});
    dev.run<WG>(batch, sz.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const dv::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                   reinterpret_cast<std::byte*>(lm)),
                               dv::global_slice(g, sz, b)};
        const int r = dv::sytrd<Uplo::Lower, BlockedTier>(ex, dv::mat(A + std::size_t(b) * ld * n, n, n, ld), dv::vec(d + std::size_t(b) * n, n),
                                                          dv::vec(e + std::size_t(b) * nm1, nm1), dv::vec(tau + std::size_t(b) * nm1, nm1), ws);
        if (ex.leader()) info[b] = r;
    });
    return {dev.down(A, a0.size()), dev.down(d, std::size_t(n) * batch), dev.down(e, std::size_t(nm1) * batch + 1),
            dev.down(tau, std::size_t(nm1) * batch + 1), dev.down(info, batch)};
}

template <class T> BlockedOut<T> blocked_host(Dev& dev, int n, int ld, int batch, const std::vector<T>& a0) {
    const int nm1 = n - 1;
    T* A = dev.up(a0);
    T* d = dev.alloc<T>(std::size_t(n) * batch, nan_v<T>());
    T* e = dev.alloc<T>(std::size_t(nm1) * batch + 1, nan_v<T>());
    T* tau = dev.alloc<T>(std::size_t(nm1) * batch + 1, nan_v<T>());
    MatrixView<T, MatrixFormat::Dense> Av(A, n, n, ld, ld * n, batch);
    VectorView<T> dv_(d, n, batch, 1, n), ev(e, nm1, batch, 1, nm1), tv(tau, nm1, batch, 1, nm1);
    const std::size_t bytes = sytrd_blocked_buffer_size<kB, T>(*dev.ctx, Av, dv_, ev, tv, Uplo::Lower, kNB);
    std::byte* ws = dev.alloc<std::byte>(std::max<std::size_t>(bytes, 1), std::byte{0});
    (void)sytrd_blocked<kB, T>(*dev.ctx, Av, dv_, ev, tv, Uplo::Lower, Span<std::byte>(ws, bytes), kNB);
    dev.ctx->wait();
    return {dev.down(A, a0.size()), dev.down(d, std::size_t(n) * batch), dev.down(e, std::size_t(nm1) * batch + 1),
            dev.down(tau, std::size_t(nm1) * batch + 1), std::vector<int>(batch, 0)};
}

// The Hermitian A0 of item q (full, from its lower triangle, Im(diag) dropped) in long double.
template <class T> LDm<T> herm_of(const std::vector<T>& a0, int n, int ld, int q) {
    LDm<T> h(std::size_t(n) * n);
    for (int c = 0; c < n; ++c)
        for (int r = c; r < n; ++r) {
            LD<T> v = wide(a0[std::size_t(q) * ld * n + r + std::size_t(c) * ld]);
            if (r == c) v = LD<T>(std::real(v));
            h[r + std::size_t(c) * n] = v;
            h[c + std::size_t(r) * n] = conj_of(v);
        }
    return h;
}

// max |Q^H A0 Q - T| / ||A0||_F and max |Q^H Q - I| for item q of a device output.
template <class T> std::pair<double, double> residual(const BlockedOut<T>& o, const LDm<T>& a0, int n, int ld, int q) {
    using W = LD<T>;
    const std::size_t nn = std::size_t(n) * n;
    LDm<T> Q(nn, W(0));
    for (int i = 0; i < n; ++i) Q[i + std::size_t(i) * n] = W(1);
    // Q := H(i) Q for i = n-2 .. 0: H(i) = I - tau(i) v v^H, v(i+1) = 1, v(r > i+1) = A(r, i).
    for (int i = n - 2; i >= 0; --i) {
        const W t = wide(o.tau[std::size_t(q) * (n - 1) + i]);
        std::vector<W> v(n, W(0));
        v[i + 1] = W(1);
        for (int r = i + 2; r < n; ++r) v[r] = wide(o.a[std::size_t(q) * ld * n + r + std::size_t(i) * ld]);
        for (int c = 0; c < n; ++c) {
            W s(0);
            for (int r = i + 1; r < n; ++r) s += conj_of(v[r]) * Q[r + std::size_t(c) * n];
            for (int r = i + 1; r < n; ++r) Q[r + std::size_t(c) * n] -= t * v[r] * s;
        }
    }
    LDm<T> AQ(nn, W(0));
    for (int c = 0; c < n; ++c)
        for (int k = 0; k < n; ++k)
            for (int r = 0; r < n; ++r) AQ[r + std::size_t(c) * n] += a0[r + std::size_t(k) * n] * Q[k + std::size_t(c) * n];
    long double fro = 0, worst = 0, orth = 0;
    for (const W& x : a0) fro += std::norm(x);
    fro = std::sqrt(fro);
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            W m(0), o2(0);
            for (int k = 0; k < n; ++k) {
                m += conj_of(Q[k + std::size_t(r) * n]) * AQ[k + std::size_t(c) * n];
                o2 += conj_of(Q[k + std::size_t(r) * n]) * Q[k + std::size_t(c) * n];
            }
            W t(0);
            if (r == c) t = wide(o.d[std::size_t(q) * n + r]);
            if (r == c + 1) t = wide(o.e[std::size_t(q) * (n - 1) + c]);
            if (c == r + 1) t = conj_of(wide(o.e[std::size_t(q) * (n - 1) + r]));
            worst = std::max(worst, std::abs(m - t));
            orth = std::max(orth, std::abs(o2 - (r == c ? W(1) : W(0))));
        }
    return {static_cast<double>(worst / fro), static_cast<double>(orth)};
}

template <class T> class DeviceApiSytrdBlockedTest : public DeviceApiSytrdTest<T> {};
TYPED_TEST_SUITE(DeviceApiSytrdBlockedTest, Types);

TYPED_TEST(DeviceApiSytrdBlockedTest, BlockedMatchesHostBlockedRouteWithinTolerance) {
    using T = TypeParam;
    for (int n : kBlockedNs) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3, batch = blocked_batch(n), nm1 = n - 1;
        const auto a0 = input_lower<T>(n, ld, batch, false);
        const auto g = blocked_device<T>(this->dev, n, ld, batch, a0);
        const auto h = blocked_host<T>(this->dev, n, ld, batch, a0);
        EXPECT_EQ(std::count(g.info.begin(), g.info.end(), 0), batch);
        double worst_d = 0, worst_e = 0;
        for (int q = 0; q < kPeriod; ++q) {
            const auto a = herm_of<T>(a0, n, ld, q);
            long double fro = 0;
            for (const auto& x : a) fro += std::norm(x);
            const double s = static_cast<double>(std::sqrt(fro));
            for (int i = 0; i < n; ++i) worst_d = std::max(worst_d, err(g.d[std::size_t(q) * n + i], wide(h.d[std::size_t(q) * n + i])) / s);
            for (int i = 0; i < nm1; ++i)
                worst_e = std::max(worst_e, err(g.e[std::size_t(q) * nm1 + i], wide(h.e[std::size_t(q) * nm1 + i])) / s);
            const auto [res, orth] = residual(g, a, n, ld, q);
            EXPECT_LE(res, tol<T>(n)) << "item " << q;
            EXPECT_LE(orth, tol<T>(n)) << "item " << q;
        }
        EXPECT_LE(worst_d, tol<T>(n));
        EXPECT_LE(worst_e, tol<T>(n));
        if (n <= 65) {
            EXPECT_EQ(period_diff(g.a, std::size_t(ld) * n), 0);
            EXPECT_EQ(period_diff(g.d, std::size_t(n)), 0);
        }
        // Pads and the strictly upper triangle past the superdiagonal: untouched; d finite.
        int untouched = 0, nonfinite = 0;
        for (int b = 0; b < batch; ++b)
            for (int c = 0; c < n; ++c)
                for (int r = 0; r < ld; ++r) {
                    if (r >= c - 1 && r < n) continue;
                    const std::size_t k = std::size_t(b) * ld * n + r + std::size_t(c) * ld;
                    untouched += !same_bits(g.a[k], a0[k]);
                }
        for (const T& x : g.d) nonfinite += !std::isfinite(std::abs(x));
        EXPECT_EQ(untouched, 0);
        EXPECT_EQ(nonfinite, 0);
    }
}

TYPED_TEST(DeviceApiSytrdBlockedTest, ImaginaryDiagonalIsDropped) {
    using T = TypeParam;
    if constexpr (!is_cplx<T>) {
        GTEST_SKIP() << "real T has no imaginary diagonal";
    } else {
        const int n = 65, ld = n + 3, batch = 3;
        const auto with = blocked_device<T>(this->dev, n, ld, batch, input_lower<T>(n, ld, batch, true));
        const auto without = blocked_device<T>(this->dev, n, ld, batch, input_lower<T>(n, ld, batch, false));
        EXPECT_EQ(bitdiff(with.a, without.a), 0);
        EXPECT_EQ(bitdiff(with.d, without.d), 0);
        EXPECT_EQ(bitdiff(with.e, without.e), 0);
        EXPECT_EQ(bitdiff(with.tau, without.tau), 0);
    }
}

// The negative control: the residual against a moved A0 must fail.
TYPED_TEST(DeviceApiSytrdBlockedTest, ResidualSeesAWrongMatrix) {
    using T = TypeParam;
    const int n = 65, ld = n + 3, batch = 3;
    const auto a0 = input_lower<T>(n, ld, batch, false);
    const auto g = blocked_device<T>(this->dev, n, ld, batch, a0);
    auto a = herm_of<T>(a0, n, ld, 0);
    a[(n - 1) + 0] += LD<T>(1e-2L);
    a[std::size_t(n - 1) * n] += LD<T>(1e-2L);
    EXPECT_GT(residual(g, a, n, ld, 0).first, tol<T>(n));
}

// The in-group syev values path in ONE user kernel: sytrd<Blocked>, Re(d) / |e| (the Hermitian
// tridiagonal's real form), steqr<NoEigenVectors, Blocked<1>>; eigenvalues against host syev.
TYPED_TEST(DeviceApiSytrdBlockedTest, ChainedWithSteqrGivesTheHostEigenvalues) {
    using T = TypeParam;
    using Re = real_of<T>;
    using RS = dv::routine::steqr<JobType::NoEigenVectors>;
    using ST = dv::tier::Blocked<1>;
    constexpr int WG = dv::tier_info<Rt<Uplo::Lower>, BlockedTier, T>::wg_size;
    static_assert(WG == dv::tier_info<RS, ST, Re>::wg_size, "both Blocked tiers run on one WorkGroup<256>");
    Dev& dev = this->dev;
    for (int n : {64, 97}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3, batch = 6, nm1 = n - 1;
        const auto a0 = input_lower<T>(n, ld, batch, false);
        T* A = dev.up(a0);
        T* d = dev.alloc<T>(std::size_t(n) * batch, nan_v<T>());
        T* e = dev.alloc<T>(std::size_t(nm1) * batch, nan_v<T>());
        T* tau = dev.alloc<T>(std::size_t(nm1) * batch, nan_v<T>());
        Re* w = dev.alloc<Re>(std::size_t(n) * batch, std::numeric_limits<Re>::quiet_NaN());
        Re* er = dev.alloc<Re>(std::size_t(nm1) * batch, std::numeric_limits<Re>::quiet_NaN());
        int* info = dev.alloc<int>(batch, 7);
        const dv::WorkspaceSize s1 = dv::sytrd_buffer_size<Uplo::Lower, BlockedTier, T>(n);
        const dv::WorkspaceSize s2 = dv::steqr_buffer_size<JobType::NoEigenVectors, ST, Re>(n);
        std::byte* g1 = dev.alloc<std::byte>(s1.global_bytes * std::size_t(batch) + s1.align, std::byte{0});
        std::byte* g2 = dev.alloc<std::byte>(s2.global_bytes * std::size_t(batch) + s2.align, std::byte{0});
        dev.run<WG>(batch, s1.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
            const dv::WorkGroup<WG> ex{it};
            const int b = static_cast<int>(it.get_group_linear_id());
            const auto loc = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                reinterpret_cast<std::byte*>(lm));
            const auto db = dv::vec(d + std::size_t(b) * n, n);
            const auto eb = dv::vec(e + std::size_t(b) * nm1, nm1);
            const auto wb = dv::vec(w + std::size_t(b) * n, n);
            const auto erb = dv::vec(er + std::size_t(b) * nm1, nm1);
            int r = dv::sytrd<Uplo::Lower, BlockedTier>(ex, dv::mat(A + std::size_t(b) * ld * n, n, n, ld), db, eb,
                                                        dv::vec(tau + std::size_t(b) * nm1, nm1), dv::Workspace{loc, dv::global_slice(g1, s1, b)});
            dv::for_each_owned(ex, n, [&](int i) {
                if constexpr (is_cplx<T>) {
                    wb(i) = db(i).real();
                    if (i < nm1) erb(i) = sycl::hypot(eb(i).real(), eb(i).imag());
                } else {
                    wb(i) = db(i);
                    if (i < nm1) erb(i) = eb(i);
                }
            });
            ex.barrier(dv::Mem::Global);
            r += dv::steqr<JobType::NoEigenVectors, ST>(ex, wb, erb, dv::mat(w, 0, 0, 1), dv::Workspace{loc, dv::global_slice(g2, s2, b)});
            if (ex.leader()) info[b] = r;
        });
        auto got = dev.down(w, std::size_t(n) * batch);
        const auto inf = dev.down(info, batch);
        EXPECT_EQ(std::count(inf.begin(), inf.end(), 0), batch);
        T* Ah = dev.up(a0);
        Re* wh = dev.alloc<Re>(std::size_t(n) * batch, std::numeric_limits<Re>::quiet_NaN());
        MatrixView<T, MatrixFormat::Dense> Av(Ah, n, n, ld, ld * n, batch);
        (void)syev<kB>(*dev.ctx, Av, Span<Re>(wh, std::size_t(n) * batch), SyevOptions{JobType::NoEigenVectors, Uplo::Lower});
        dev.ctx->wait();
        const auto ref = dev.down(wh, std::size_t(n) * batch);
        double worst = 0;
        for (int b = 0; b < batch; ++b) {
            std::sort(got.begin() + std::size_t(b) * n, got.begin() + std::size_t(b + 1) * n);
            const auto a = herm_of<T>(a0, n, ld, b % kPeriod);
            long double fro = 0;
            for (const auto& x : a) fro += std::norm(x);
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, err(got[std::size_t(b) * n + i], wide(ref[std::size_t(b) * n + i])) / static_cast<double>(std::sqrt(fro)));
        }
        EXPECT_LE(worst, tol<T>(n));
    }
}

TEST(DeviceApiSytrdTier, WithTierPicksBlockedPast32ForLower) {
    const dv::DeviceLimits lim{48 * 1024, {32}, 1024, true};
    for (int n : {33, 64, 1000, 5000}) {
        int got = 0;
        dv::with_tier<Rt<Uplo::Lower>, std::complex<double>>(lim, n, [&]<class Tier>(Tier) { got = Tier::n_max; });
        EXPECT_EQ(got, -1) << "n=" << n;  // Blocked<32>: local memory O(NB), any n
    }
    // Work-groups capped below 256: only the Cta tiers remain.
    const dv::DeviceLimits wg128{48 * 1024, {32}, 128, true};
    EXPECT_THROW((dv::with_tier<Rt<Uplo::Lower>, float>(wg128, 33, [&]<class Tier>(Tier) {})), batchlas::invalid_argument);
    static_assert(std::is_same_v<dv::resolve_tier_t<Rt<Uplo::Lower>, float, dv::tier::upto<4096>>, BlockedTier>);
}

template <class T> void expect_blocked_size(int n) {
    const auto r16 = [](std::size_t b) { return (b + 15) / 16 * 16; };
    const std::size_t s = sizeof(T);
    const std::size_t wc = r16(r16(std::size_t(n) * kNB * s) + n * s);  // W, then v, then wcol
    const std::size_t wip = r16(kNB * s);
    const std::size_t stage = r16(wip + kNB * s);
    const std::size_t elems = 1 * ((32 + 1) * 8 + (8 + 1) * 32);  // RegTiled<32, 32, 8>: A and B slabs
    const auto got = dv::sytrd_buffer_size<Uplo::Lower, BlockedTier, T>(n);
    EXPECT_EQ(got.global_bytes, r16(wc + n * s)) << "n=" << n;
    EXPECT_EQ(got.local_bytes, r16(stage + elems * s)) << "n=" << n;
}
TEST(DeviceApiSytrdTier, BlockedBufferSizeIsWvwAndStage) {
    expect_blocked_size<float>(33);
    expect_blocked_size<double>(160);
    expect_blocked_size<std::complex<float>>(1);
    expect_blocked_size<std::complex<double>>(4096);
}

}  // namespace
