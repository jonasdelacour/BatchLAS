// SPDX-License-Identifier: MIT
// device_lapack_syev_tests: device::syev<J, UL, tier::Cta<N>> and syev<J, UL, tier::Blocked<NB>>
// (include/batchlas/device/lapack/syev.hh) called from a user-style kernel.
// Cta: the host syev_cta_fused kernel is the thin launcher over the same body (route cta_fused;
// called directly, nothing to pin), so the comparison is bit identity, plus an independent
// long-double residual so the check is not only self-referential:
//   - memcmp of W, A (vectors) and info for float, double and both complex types, both
//     triangles, both jobs, n at every tier boundary (1, 4, 5, 8, 9, 16, 17, 31, 32 = the ceiling);
//   - batch 1024 with period-3 items (every tile but the first at a non-zero local offset), each
//     item bit-identical to its representative;
//   - ld = n + 3 with NaN in the pad rows (untouched) and in the triangle syev must not read;
//   - complex Im(diag) != 0 (dropped, as {c,z}heev does).
// Blocked: sytrd<Lower, Blocked<32>> + steqr<J, Blocked<1>> + ormqr in one work-group, a
// tolerance comparison: W against a graded spectrum built on the host (A = H D H^H, H a
// Householder reflector, D spanning 1 .. 1e-3) and against the host syev on random input;
// residual ||A0 V - V W|| and ||V^H V - I|| in long double; ascending W; batch items
// bit-identical to their period-3 representative; NaN pads untouched; Upper input mirrored.
// Deliberate breaks (run every time): the device call with the OTHER triangle must differ from
// the host route; the residual against a moved A0 must exceed the tolerance.
#include <tests/device_core_common.hh>

#include <batchlas/backend_config.h>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/device/lapack/syev.hh>
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
constexpr JobType kV = JobType::EigenVectors;
constexpr JobType kN = JobType::NoEigenVectors;

template <JobType J, Uplo UL> using Rt = batchlas::device::routine::syev<J, UL>;

template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> T conj_of(T x) {
    if constexpr (is_cplx<T>) return std::conj(x);
    else return x;
}

// Item b repeats representative b % kPeriod. UL's triangle holds data (complex Im(diag) != 0),
// the other triangle and the pad rows hold NaN.
template <class T> std::vector<T> input(int n, int ld, Uplo ul, int batch) {
    std::vector<T> a(std::size_t(ld) * n * batch, nan_v<T>());
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r)
                if (ul == Uplo::Upper ? r <= c : r >= c)
                    a[std::size_t(b) * ld * n + r + std::size_t(c) * ld] = val<T>(std::uint64_t(b % kPeriod) * 1000003u + r + 97u * c);
    return a;
}

// The Hermitian matrix item b stores (its UL triangle; Im(diag) dropped), in long double.
template <class T> std::vector<LD<T>> herm_of(const std::vector<T>& a, int n, int ld, Uplo ul, int b) {
    std::vector<LD<T>> h(std::size_t(n) * n);
    const T* p = a.data() + std::size_t(b) * ld * n;
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            const bool stored = ul == Uplo::Upper ? r <= c : r >= c;
            LD<T> v = stored ? wide(p[r + std::size_t(c) * ld]) : wide(conj_of(p[c + std::size_t(r) * ld]));
            if (r == c) v = LD<T>(std::real(v));
            h[r + std::size_t(c) * n] = v;
        }
    return h;
}

// ||A V - V diag(w)||_F / ||A||_F and ||V^H V - I||_F of item b's output (V over A, ld).
template <class T>
std::pair<double, double> residual(const std::vector<LD<T>>& h, const std::vector<T>& v, const std::vector<real_of<T>>& w, int n, int ld, int b) {
    const T* V = v.data() + std::size_t(b) * ld * n;
    const real_of<T>* W = w.data() + std::size_t(b) * n;
    long double fro = 0, res = 0, orth = 0;
    for (const auto& x : h) fro += std::norm(x);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            LD<T> s = -wide(V[i + std::size_t(j) * ld]) * static_cast<long double>(W[j]);
            for (int k = 0; k < n; ++k) s += h[i + std::size_t(k) * n] * wide(V[k + std::size_t(j) * ld]);
            res += std::norm(s);
            LD<T> g = i == j ? LD<T>(-1) : LD<T>(0);
            for (int k = 0; k < n; ++k) g += conj_of(wide(V[k + std::size_t(i) * ld])) * wide(V[k + std::size_t(j) * ld]);
            orth += std::norm(g);
        }
    return {static_cast<double>(std::sqrt(res / std::max(fro, 1e-300L))), static_cast<double>(std::sqrt(orth))};
}

template <class T> int bitdiff(const std::vector<T>& x, const std::vector<T>& y) {
    int k = 0;
    for (std::size_t i = 0; i < x.size(); ++i) k += !same_bits(x[i], y[i]);
    return k + static_cast<int>(x.size() != y.size());
}
template <class T> int period_diff(const std::vector<T>& g, std::size_t per, int batch) {
    int k = 0;
    for (std::size_t b = kPeriod; b < std::size_t(batch); ++b)
        for (std::size_t i = 0; i < per; ++i) k += !same_bits(g[b * per + i], g[(b % kPeriod) * per + i]);
    return k;
}
template <class T> int pad_diff(const std::vector<T>& g, const std::vector<T>& a0, int n, int ld, int batch) {
    int k = 0;
    for (int b = 0; b < batch; ++b)
        for (int c = 0; c < n; ++c)
            for (int r = n; r < ld; ++r) {
                const std::size_t i = std::size_t(b) * ld * n + r + std::size_t(c) * ld;
                k += !same_bits(g[i], a0[i]);
            }
    return k;
}

template <class T> struct Out {
    std::vector<T> a;
    std::vector<real_of<T>> w;
    std::vector<int> info;
};

template <JobType J, Uplo UL, class Tier, class T> class ApiSyevCtaKernel;

// The user's kernel: one problem per partition of N lanes, the tiles in the work-group's local
// memory at this partition's slice, the tier fixed at compile time.
template <JobType J, Uplo UL, class Tier, class T> void launch_cta(Dev& dev, T* A, int n, int ld, real_of<T>* w, int* info) {
    constexpr int N = Tier::n_max;
    constexpr int ppw = 32 / N;
    constexpr std::size_t lbytes = batchlas::device::syev_buffer_size<J, UL, Tier, T>(N).local_bytes;
    dev.q().submit([&](sycl::handler& h) {
           sycl::local_accessor<sycl::uint4, 1> slm(sycl::range<1>((lbytes * ppw + 15) / 16), h);
           h.parallel_for<ApiSyevCtaKernel<J, UL, Tier, T>>(
               sycl::nd_range<1>(std::size_t(kBatch / ppw) * 32, 32), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                   const auto part = make_partition<N>(it.get_sub_group());
                   const auto ex = batchlas::device::wrap(part);
                   const int slot = static_cast<int>(part.get_group_linear_id());
                   const int b = static_cast<int>(it.get_group_linear_id()) * ppw + slot;
                   std::byte* raw = reinterpret_cast<std::byte*>(slm.template get_multi_ptr<sycl::access::decorated::no>().get_raw());
                   const batchlas::device::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                              raw + std::size_t(slot) * lbytes),
                                          nullptr};
                   const int r = batchlas::device::syev<J, UL, Tier>(ex, batchlas::device::mat(A + std::size_t(b) * ld * n, n, n, ld), batchlas::device::vec(w + std::size_t(b) * n, n), ws);
                   if (ex.leader()) info[b] = r;
               });
       }).wait_and_throw();
}

template <class T> class DeviceApiSyevCtaTest : public DeviceCoreTest<T> {
  protected:
    void SetUp() override {
        DeviceCoreTest<T>::SetUp();
        if (this->IsSkipped()) return;
        // The CTA routes build without CPU targets (NO_CPU_TARGETS): only a GPU reaches them.
        if (!this->dev.q().get_device().has(sycl::aspect::gpu)) GTEST_SKIP() << "syev_cta_fused runs on GPUs only";
    }

  public:
    Out<T> host(int n, JobType J, Uplo ul, const std::vector<T>& a0, int ld) {
        Dev& dev = this->dev;
        T* A = dev.up(a0);
        real_of<T>* w = dev.alloc<real_of<T>>(std::size_t(n) * kBatch, std::numeric_limits<real_of<T>>::quiet_NaN());
        int* info = dev.alloc<int>(kBatch, 7);
        MatrixView<T, MatrixFormat::Dense> Av(A, n, n, ld, ld * n, kBatch);
        (void)syev_cta_fused<kB, T>(*dev.ctx, Av, Span<real_of<T>>(w, std::size_t(n) * kBatch), J, ul, Span<std::byte>(), SteqrParams<T>{}, 0,
                                    Span<int32_t>(info, kBatch));
        dev.ctx->wait();
        return {dev.down(A, a0.size()), dev.down(w, std::size_t(n) * kBatch), dev.down(info, kBatch)};
    }

    template <JobType J, Uplo UL> Out<T> device(int n, const std::vector<T>& a0, int ld) {
        Dev& dev = this->dev;
        T* A = dev.up(a0);
        real_of<T>* w = dev.alloc<real_of<T>>(std::size_t(n) * kBatch, std::numeric_limits<real_of<T>>::quiet_NaN());
        int* info = dev.alloc<int>(kBatch, 7);
        // Every listed tier is instantiated; n <= 32 must reach a Cta one (Blocked is tested below).
        batchlas::device::with_tier<Rt<J, UL>, T>(dev.q().get_device(), n, [&]<class Tier>(Tier) {
            if constexpr (Tier::n_max > 0) launch_cta<J, UL, Tier, T>(dev, A, n, ld, w, info);
            else ADD_FAILURE() << "with_tier picked tier::Blocked for n = " << n;
        });
        return {dev.down(A, a0.size()), dev.down(w, std::size_t(n) * kBatch), dev.down(info, kBatch)};
    }
};

TYPED_TEST_SUITE(DeviceApiSyevCtaTest, AllTypes);

template <class T, JobType J, Uplo UL, class TF> void bit_identity(TF& t) {
    for (int n : kNs) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3;
        const auto a0 = input<T>(n, ld, UL, kBatch);
        const auto h = t.host(n, J, UL, a0, ld);
        const auto g = t.template device<J, UL>(n, a0, ld);
        EXPECT_EQ(bitdiff(g.w, h.w), 0);
        EXPECT_EQ(bitdiff(g.a, h.a), 0);
        EXPECT_EQ(bitdiff(g.info, h.info), 0);
        EXPECT_EQ(std::count(g.info.begin(), g.info.end(), 0), kBatch);
        EXPECT_EQ(period_diff(g.w, std::size_t(n), kBatch), 0);
        EXPECT_EQ(pad_diff(g.a, a0, n, ld, kBatch), 0);
        // Ascending, and (vectors) an independent residual on the representatives.
        for (int b = 0; b < kPeriod; ++b) {
            EXPECT_TRUE(std::is_sorted(g.w.begin() + std::size_t(b) * n, g.w.begin() + std::size_t(b + 1) * n));
            if constexpr (J == kV) {
                const auto [res, orth] = residual<T>(herm_of(a0, n, ld, UL, b), g.a, g.w, n, ld, b);
                EXPECT_LE(res, tol<T>(n));
                EXPECT_LE(orth, tol<T>(n));
            }
        }
    }
}

TYPED_TEST(DeviceApiSyevCtaTest, VectorsLowerBitIdenticalToHostRoute) { bit_identity<TypeParam, kV, Uplo::Lower>(*this); }
TYPED_TEST(DeviceApiSyevCtaTest, VectorsUpperBitIdenticalToHostRoute) { bit_identity<TypeParam, kV, Uplo::Upper>(*this); }
TYPED_TEST(DeviceApiSyevCtaTest, ValuesLowerBitIdenticalToHostRoute) { bit_identity<TypeParam, kN, Uplo::Lower>(*this); }
TYPED_TEST(DeviceApiSyevCtaTest, ValuesUpperBitIdenticalToHostRoute) { bit_identity<TypeParam, kN, Uplo::Upper>(*this); }

// The negative control: reading the other triangle (NaN-poisoned here) cannot match the host.
TYPED_TEST(DeviceApiSyevCtaTest, WrongTriangleIsSeen) {
    using T = TypeParam;
    const int n = 17, ld = n + 3;
    const auto a0 = input<T>(n, ld, Uplo::Lower, kBatch);
    const auto h = this->host(n, kN, Uplo::Lower, a0, ld);
    const auto g = this->template device<kN, Uplo::Upper>(n, a0, ld);
    EXPECT_GT(bitdiff(g.w, h.w), 0);
}

TYPED_TEST(DeviceApiSyevCtaTest, ArgumentCheckReturnsMinus2AndWritesNothing) {
    using T = TypeParam;
    const int n = 5, ld = n + 3;  // tier::Cta<4> cannot hold n = 5
    const auto a0 = input<T>(n, ld, Uplo::Lower, kBatch);
    Dev& dev = this->dev;
    T* A = dev.up(a0);
    real_of<T>* w = dev.alloc<real_of<T>>(std::size_t(n) * kBatch, std::numeric_limits<real_of<T>>::quiet_NaN());
    int* info = dev.alloc<int>(kBatch, 7);
    launch_cta<kV, Uplo::Lower, batchlas::device::tier::Cta<4>, T>(dev, A, n, ld, w, info);
    const auto got = dev.down(info, kBatch);
    EXPECT_EQ(std::count(got.begin(), got.end(), -2), kBatch);
    EXPECT_EQ(bitdiff(dev.down(A, a0.size()), a0), 0);
    int written = 0;
    for (real_of<T> x : dev.down(w, std::size_t(n) * kBatch)) written += !std::isnan(x);
    EXPECT_EQ(written, 0);
}

TEST(DeviceApiSyevTier, WithTierPicksTheSmallestCtaThenBlocked) {
    const batchlas::device::DeviceLimits lim{48 * 1024, {32}, 1024, true};
    for (int n : {1, 4, 5, 8, 9, 16, 17, 32, 33, 64, 1000}) {
        int got = 0;
        batchlas::device::with_tier<Rt<kV, Uplo::Upper>, std::complex<double>>(lim, n, [&]<class Tier>(Tier) { got = Tier::n_max; });
        EXPECT_EQ(got, n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : n <= 32 ? 32 : -1) << "n=" << n;
    }
    // Work-groups capped below 256: past 32 nothing admits.
    const batchlas::device::DeviceLimits wg128{48 * 1024, {32}, 128, true};
    try {
        batchlas::device::with_tier<Rt<kN, Uplo::Lower>, float>(wg128, 33, [&]<class Tier>(Tier) {});
        ADD_FAILURE() << "with_tier admitted n = 33 at 128 items";
    } catch (const batchlas::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("syev: no device tier admits n=33"), std::string::npos) << e.what();
    }
}

// The Cta workspace: the fused kernel's per-problem tiles (independent count).
template <JobType J, int N, class T> void expect_cta_size() {
    const bool fused = J == kV && !is_cplx<T>;
    const std::size_t lda = fused ? N + 1 : N;
    std::size_t bytes = (lda * N + 2 * N) * sizeof(T);
    if (J == kV && is_cplx<T>) bytes = (bytes + 15) / 16 * 16 + std::size_t(N + 1) * N * sizeof(real_of<T>);
    const auto s = batchlas::device::syev_buffer_size<J, Uplo::Lower, batchlas::device::tier::Cta<N>, T>(N);
    EXPECT_EQ(s.local_bytes, (bytes + 15) / 16 * 16) << "N=" << N;
    EXPECT_EQ(s.global_bytes, 0u);
}
TEST(DeviceApiSyevTier, CtaBufferSizeIsTheFusedTiles) {
    expect_cta_size<kV, 4, float>();
    expect_cta_size<kN, 8, double>();
    expect_cta_size<kV, 16, std::complex<float>>();
    expect_cta_size<kV, 32, std::complex<double>>();
    expect_cta_size<kN, 32, std::complex<float>>();
}

// ---- tier::Blocked<NB>: any n, one matrix per WorkGroup<256> ---------------------------------
constexpr int kNB = 32;
using BlockedTier = batchlas::device::tier::Blocked<kNB>;
constexpr int kBlockedBatch = 6;

template <JobType J, Uplo UL, class T> Out<T> blocked_device(Dev& dev, int n, int ld, int batch, const std::vector<T>& a0) {
    using R = Rt<J, UL>;
    constexpr int WG = batchlas::device::tier_info<R, BlockedTier, T>::wg_size;
    T* A = dev.up(a0);
    real_of<T>* w = dev.alloc<real_of<T>>(std::size_t(n) * batch, std::numeric_limits<real_of<T>>::quiet_NaN());
    int* info = dev.alloc<int>(batch, 7);
    const batchlas::device::WorkspaceSize sz = batchlas::device::syev_buffer_size<J, UL, BlockedTier, T>(n);
    std::byte* g = dev.alloc<std::byte>(sz.global_bytes * std::size_t(batch) + sz.align, std::byte{0});
    dev.run<WG>(batch, sz.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const batchlas::device::WorkGroup<WG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const batchlas::device::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                   reinterpret_cast<std::byte*>(lm)),
                               batchlas::device::global_slice(g, sz, b)};
        const int r = batchlas::device::syev<J, UL, BlockedTier>(ex, batchlas::device::mat(A + std::size_t(b) * ld * n, n, n, ld), batchlas::device::vec(w + std::size_t(b) * n, n), ws);
        if (ex.leader()) info[b] = r;
    });
    return {dev.down(A, a0.size()), dev.down(w, std::size_t(n) * batch), dev.down(info, batch)};
}

// A = H D H^H with H = I - 2 u u^H / (u^H u) and D(i) = 10^(-3 i / (n - 1)) (sign alternating):
// a graded spectrum known exactly; item b uses u from representative b % kPeriod. UL's triangle
// holds data, the rest NaN.
template <class T> std::vector<T> graded(int n, int ld, Uplo ul, int batch, std::vector<long double>& spectrum) {
    spectrum.assign(n, 0);
    for (int i = 0; i < n; ++i) spectrum[i] = (i % 2 ? -1.0L : 1.0L) * std::pow(10.0L, n > 1 ? -3.0L * i / (n - 1) : 0.0L);
    std::vector<T> a(std::size_t(ld) * n * batch, nan_v<T>());
    for (int q = 0; q < kPeriod; ++q) {
        std::vector<LD<T>> u(n);
        long double uu = 0;
        for (int i = 0; i < n; ++i) {
            u[i] = wide(val<T>(std::uint64_t(q) * 7919u + i));
            uu += std::norm(u[i]);
        }
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < n; ++r) {
                if (ul == Uplo::Upper ? r > c : r < c) continue;
                LD<T> s = 0;
                for (int k = 0; k < n; ++k) {
                    const LD<T> hr = (r == k ? LD<T>(1) : LD<T>(0)) - 2.0L * u[r] * conj_of(u[k]) / uu;
                    const LD<T> hc = (c == k ? LD<T>(1) : LD<T>(0)) - 2.0L * u[c] * conj_of(u[k]) / uu;
                    s += hr * spectrum[k] * conj_of(hc);
                }
                if (r == c) s = LD<T>(std::real(s));
                for (int b = q; b < batch; b += kPeriod) a[std::size_t(b) * ld * n + r + std::size_t(c) * ld] = static_cast<T>(s);
            }
    }
    std::sort(spectrum.begin(), spectrum.end());
    return a;
}

template <class T> class DeviceApiSyevBlockedTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceApiSyevBlockedTest, AllTypes);

template <class T, Uplo UL> void blocked_graded(Dev& dev) {
    for (int n : {1, 2, 33, 64, 97}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3;
        std::vector<long double> spec;
        const auto a0 = graded<T>(n, ld, UL, kBlockedBatch, spec);
        const auto g = blocked_device<kV, UL, T>(dev, n, ld, kBlockedBatch, a0);
        const auto v = blocked_device<kN, UL, T>(dev, n, ld, kBlockedBatch, a0);
        EXPECT_EQ(std::count(g.info.begin(), g.info.end(), 0), kBlockedBatch);
        EXPECT_EQ(std::count(v.info.begin(), v.info.end(), 0), kBlockedBatch);
        EXPECT_EQ(period_diff(g.w, std::size_t(n), kBlockedBatch), 0);
        EXPECT_EQ(period_diff(g.a, std::size_t(ld) * n, kBlockedBatch), 0);
        EXPECT_EQ(pad_diff(g.a, a0, n, ld, kBlockedBatch), 0);
        double worst = 0, worst_v = 0;
        for (int b = 0; b < kPeriod && b < kBlockedBatch; ++b)
            for (int i = 0; i < n; ++i) {
                worst = std::max(worst, std::abs(static_cast<double>(g.w[std::size_t(b) * n + i] - spec[i])));
                worst_v = std::max(worst_v, std::abs(static_cast<double>(v.w[std::size_t(b) * n + i] - spec[i])));
            }
        EXPECT_LE(worst, tol<T>(n));
        EXPECT_LE(worst_v, tol<T>(n));
        for (int b = 0; b < kPeriod; ++b) {
            const auto [res, orth] = residual<T>(herm_of(a0, n, ld, UL, b), g.a, g.w, n, ld, b);
            EXPECT_LE(res, tol<T>(n));
            EXPECT_LE(orth, tol<T>(n));
        }
    }
}
TYPED_TEST(DeviceApiSyevBlockedTest, GradedSpectrumLower) { blocked_graded<TypeParam, Uplo::Lower>(this->dev); }
TYPED_TEST(DeviceApiSyevBlockedTest, GradedSpectrumUpper) { blocked_graded<TypeParam, Uplo::Upper>(this->dev); }

// Random input against the host syev's eigenvalues (another route: no self-reference).
TYPED_TEST(DeviceApiSyevBlockedTest, ValuesMatchTheHostSyev) {
    using T = TypeParam;
    using Re = real_of<T>;
    Dev& dev = this->dev;
    for (int n : {40, 160}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        const int ld = n + 3;
        auto a0 = input<T>(n, ld, Uplo::Lower, kBlockedBatch);
        // Real diagonal: the host route is the reference, and not every host route drops Im(diag).
        if constexpr (is_cplx<T>)
            for (int b = 0; b < kBlockedBatch; ++b)
                for (int i = 0; i < n; ++i) {
                    T& x = a0[std::size_t(b) * ld * n + i + std::size_t(i) * ld];
                    x = T(x.real(), 0);
                }
        const auto g = blocked_device<kN, Uplo::Lower, T>(dev, n, ld, kBlockedBatch, a0);
        T* Ah = dev.up(a0);
        Re* wh = dev.alloc<Re>(std::size_t(n) * kBlockedBatch, std::numeric_limits<Re>::quiet_NaN());
        MatrixView<T, MatrixFormat::Dense> Av(Ah, n, n, ld, ld * n, kBlockedBatch);
        (void)syev<kB>(*dev.ctx, Av, Span<Re>(wh, std::size_t(n) * kBlockedBatch), SyevOptions{kN, Uplo::Lower});
        dev.ctx->wait();
        const auto ref = dev.down(wh, std::size_t(n) * kBlockedBatch);
        double worst = 0;
        for (int b = 0; b < kBlockedBatch; ++b) {
            long double fro = 0;
            for (const auto& x : herm_of(a0, n, ld, Uplo::Lower, b)) fro += std::norm(x);
            for (int i = 0; i < n; ++i)
                worst = std::max(worst, err(g.w[std::size_t(b) * n + i], wide(ref[std::size_t(b) * n + i])) / static_cast<double>(std::sqrt(fro)));
        }
        EXPECT_LE(worst, tol<T>(n));
        EXPECT_EQ(std::count(g.info.begin(), g.info.end(), 0), kBlockedBatch);
    }
}

// Random input with Im(diag) != 0 (complex): the residual against the matrix with Im(diag)
// dropped, as {c,z}heev reads it; both triangles.
TYPED_TEST(DeviceApiSyevBlockedTest, RandomInputResidualBothTriangles) {
    using T = TypeParam;
    for (Uplo ul : {Uplo::Lower, Uplo::Upper}) {
        const int n = 65, ld = n + 3;
        const auto a0 = input<T>(n, ld, ul, kBlockedBatch);
        const auto g = ul == Uplo::Lower ? blocked_device<kV, Uplo::Lower, T>(this->dev, n, ld, kBlockedBatch, a0)
                                         : blocked_device<kV, Uplo::Upper, T>(this->dev, n, ld, kBlockedBatch, a0);
        EXPECT_EQ(std::count(g.info.begin(), g.info.end(), 0), kBlockedBatch);
        for (int b = 0; b < kPeriod; ++b) {
            EXPECT_TRUE(std::is_sorted(g.w.begin() + std::size_t(b) * n, g.w.begin() + std::size_t(b + 1) * n));
            const auto [res, orth] = residual<T>(herm_of(a0, n, ld, ul, b), g.a, g.w, n, ld, b);
            EXPECT_LE(res, tol<T>(n)) << "upper=" << (ul == Uplo::Upper);
            EXPECT_LE(orth, tol<T>(n)) << "upper=" << (ul == Uplo::Upper);
        }
    }
}

// The negative control: the residual against a moved A0 must fail.
TYPED_TEST(DeviceApiSyevBlockedTest, ResidualSeesAWrongMatrix) {
    using T = TypeParam;
    const int n = 33, ld = n + 3;
    std::vector<long double> spec;
    const auto a0 = graded<T>(n, ld, Uplo::Lower, kBlockedBatch, spec);
    const auto g = blocked_device<kV, Uplo::Lower, T>(this->dev, n, ld, kBlockedBatch, a0);
    auto h = herm_of(a0, n, ld, Uplo::Lower, 0);
    h[n - 1] += LD<T>(1);
    h[std::size_t(n - 1) * n] += LD<T>(1);
    EXPECT_GT(residual<T>(h, g.a, g.w, n, ld, 0).first, tol<T>(n));
}

TEST(DeviceApiSyevTier, BlockedBufferSizeHoldsTheStages) {
    using T = std::complex<float>;
    for (int n : {1, 33, 1000}) {
        const auto s = batchlas::device::syev_buffer_size<kV, Uplo::Lower, BlockedTier, T>(n);
        const auto s1 = batchlas::device::sytrd_buffer_size<Uplo::Lower, BlockedTier, T>(n);
        const auto s2 = batchlas::device::steqr_buffer_size<kV, batchlas::device::tier::Blocked<1>, float>(n);
        // d, e, tau, Re(d), |e|, S, Z, rank, then at least the largest stage.
        const std::size_t own = (3 * sizeof(T) + 2 * sizeof(float) + sizeof(T) + sizeof(int)) * std::size_t(n) + std::size_t(n) * n * sizeof(T);
        EXPECT_GE(s.global_bytes, own + std::max(s1.global_bytes, s2.global_bytes)) << "n=" << n;
        EXPECT_GE(s.local_bytes, std::max(s1.local_bytes, s2.local_bytes)) << "n=" << n;
    }
}

}  // namespace
