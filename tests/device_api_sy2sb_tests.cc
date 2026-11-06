// SPDX-License-Identifier: MIT
// device_api_sy2sb_tests: the public device op device::sy2sb (include/batchlas/device/lapack/sy2sb.hh)
// called from a user-style kernel (one WorkGroup<256> per matrix, the caller's Workspace), against
// the host route sytrd_sy2sb (a multi-kernel pipeline: tolerance and info parity, not bits), and
// sy2sb + sb2st fused in ONE user kernel against host sy2sb + sb2st and against the input itself.
// Guards (docs/developer/agent-guide.md section 8):
//   - straddled thresholds: n <= kd, n == kd + 1 (a 1 x 1 panel: complex still reflects), n % kd in
//     {0, 1, >= 2} (the short last panel's strip), n - kd = 32 and 33 (one and two L3 tiles),
//     kd = 1, kd = 33 and kd = 40 > NB; both tiers (Blocked<32>, Blocked<64>);
//   - lda = n + 3, ldab = kd + 4, batch strides past the matrices; the strict upper triangle of A,
//     every pad, AB and tau NaN before the call; the upper triangle and pads must survive bit for
//     bit; the global workspace is NaN-filled (beta == 0 products must not read it);
//   - complex data with Im(diag) != 0 (ignored as the host ignores it); a graded representative;
//   - batch 1024 with period-3 data: every item bit-identical to its representative;
//   - the fused tridiagonal checked against trace and Frobenius norm of the input (no
//     self-reference) and its Sturm eigenvalues against the host pipeline's;
//   - bad arguments return -i uniformly and write nothing; with_tier filters on local memory,
//     work-group size, sub-group size and fp64, and buffer_size equals the carve.
// Red set (make each break once on the GPU box before calling this suite green): drop the strip
// update in two_sided_lower (StripCases: n % kd >= 2 fails); hemm -> symm for a complex T
// (complex Parity fails); lacpy diag_fill::Unit -> Keep (every reduced case fails).
#include <tests/device_core_common.hh>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/device/lapack/sb2st.hh>
#include <batchlas/device/lapack/sy2sb.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace device_core_test;
using batchlas::Uplo;

namespace {

constexpr int kWG = 256;
constexpr int kPeriod = 3;
using Blk32 = dv::tier::Blocked<32>;
using Blk64 = dv::tier::Blocked<64>;
static_assert(dv::tier_info<dv::routine::sy2sb<Uplo::Lower>, Blk32, float>::wg_size == kWG);
static_assert(dv::tier_info<dv::routine::sy2sb<Uplo::Lower>, Blk64, std::complex<double>>::wg_size == kWG);

template <class T> T nan_of() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> double re_of(T v) {
    if constexpr (is_cplx<T>) return double(v.real());
    else return double(v);
}

// items Hermitian n x n matrices, item b a copy of representative b % kPeriod (representative 1
// graded: D A D, D = diag(2^(-8 k / n))); A, AB and tau with non-natural ld and batch strides, all
// pads, the strict upper triangle, AB and tau NaN.
template <class T> struct Problem {
    int n, kd, items, lda, ldab, ntau;
    std::size_t sa, sab, stau;
    std::vector<T> a, ab, tau;
    Problem(int n_, int kd_, int items_, std::uint64_t seed)
        : n(n_), kd(kd_), items(items_), lda(n_ + 3), ldab(kd_ + 4), ntau(std::max(0, n_ - kd_)), sa(std::size_t(n_ + 3) * n_ + 5),
          sab(std::size_t(kd_ + 4) * n_ + 3), stau(std::size_t(std::max(0, n_ - kd_)) + 2), a(sa * items_, nan_of<T>()),
          ab(sab * items_, nan_of<T>()), tau(stau * items_, nan_of<T>()) {
        for (int r = 0; r < kPeriod; ++r)
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i) {
                    T x = val<T>(seed * 1000003u + std::uint64_t(r) * 7919u + std::uint64_t(j) * n + i);
                    if (r == 1) x = x * real_of<T>(std::exp2(-8.0 * (i + j) / n));
                    a[std::size_t(r) * sa + i + std::size_t(j) * lda] = x;
                }
        for (int b = kPeriod; b < items; ++b) std::copy_n(a.data() + std::size_t(b % kPeriod) * sa, sa, a.data() + std::size_t(b) * sa);
        for (int r = 0; r < kPeriod; ++r) {  // ||A||_F as a Hermitian matrix (lower triangle, real diagonal)
            long double s = 0;
            for (int j = 0; j < n; ++j) {
                s += std::pow(static_cast<long double>(re_of(at(r, j, j))), 2);
                for (int i = j + 1; i < n; ++i) s += 2 * std::pow(static_cast<long double>(std::abs(at(r, i, j))), 2);
            }
            fros[r] = std::max(std::sqrt(double(s)), 1e-30);
        }
    }
    T at(int b, int i, int j) const { return a[std::size_t(b) * sa + i + std::size_t(j) * lda]; }
    double fro(int b) const { return fros[b % kPeriod]; }
    double fros[kPeriod] = {};
};

template <class T> struct Out {
    std::vector<T> a, ab, tau;
    std::vector<int> info;
};

// Argument overrides for the bad-argument cases (defaults: the problem's own shapes).
struct Args {
    int kd = -1000, ab_rows = -1, tau_len = -1, a_cols = -1;
};

template <class T, class Tier> Out<T> run_device(Dev& dev, const Problem<T>& p, Args g = {}) {
    const int n = p.n, lda = p.lda, ldab = p.ldab;
    const int kd = g.kd == -1000 ? p.kd : g.kd;
    const int ab_rows = g.ab_rows < 0 ? p.kd + 1 : g.ab_rows, tau_len = g.tau_len < 0 ? p.ntau : g.tau_len;
    const int a_cols = g.a_cols < 0 ? n : g.a_cols;
    const std::size_t sa = p.sa, sab = p.sab, stau = p.stau;
    const dv::WorkspaceSize sz = dv::sy2sb_buffer_size<Uplo::Lower, Tier, T>(n, std::max(kd, 0));
    T* a = dev.up(p.a);
    T* ab = dev.up(p.ab);
    T* tau = dev.up(p.tau);
    int* info = dev.up(std::vector<int>(p.items, 12345));
    std::byte* gws = dev.up(std::vector<std::byte>(std::max<std::size_t>(sz.global_bytes * p.items, 1), std::byte{0xff}));
    dev.template run<kWG>(p.items, sz.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<kWG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const dv::Workspace ws{sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
                                   reinterpret_cast<std::byte*>(lm)),
                               dv::global_slice(gws, sz, b)};
        const int r = dv::sy2sb<Uplo::Lower, Tier>(ex, dv::mat(a + sa * b, n, a_cols, lda), dv::mat(ab + sab * b, ab_rows, n, ldab),
                                                   dv::vec(tau + stau * b, tau_len), kd, ws);
        if (ex.leader()) info[b] = r;
        ex.barrier(dv::Mem::Global);
        if (r != ex.bcast(r, 0)) info[b] = -999;  // not uniform
    });
    return {dev.down(a, p.a.size()), dev.down(ab, p.ab.size()), dev.down(tau, p.tau.size()), dev.down(info, std::size_t(p.items))};
}

template <class T> Out<T> run_host(Dev& dev, const Problem<T>& p) {
    T* a = dev.up(p.a);
    T* ab = dev.up(p.ab);
    T* tau = dev.up(p.tau);
    const MatrixView<T, MatrixFormat::Dense> av(a, p.n, p.n, p.lda, int(p.sa), p.items);
    const MatrixView<T, MatrixFormat::Dense> abv(ab, p.kd + 1, p.n, p.ldab, int(p.sab), p.items);
    const VectorView<T> tv(tau, p.ntau, p.items, 1, int(p.stau));
    UnifiedVector<std::byte> ws(sytrd_sy2sb_buffer_size(*dev.ctx, av, abv, tv, Uplo::Lower, p.kd));
    sytrd_sy2sb(*dev.ctx, av, abv, tv, Uplo::Lower, p.kd, ws.to_span()).wait();
    return {dev.down(a, p.a.size()), dev.down(ab, p.ab.size()), dev.down(tau, p.tau.size()), std::vector<int>(p.items, 0)};
}

// Element e of item b's slice is inside the written region: A's lower triangle, AB's kd + 1
// rows, tau's ntau entries.
template <class T> bool in_a(const Problem<T>& p, std::size_t e) {
    const std::size_t r = e % p.sa;
    return r < std::size_t(p.lda) * p.n && int(r % p.lda) < p.n && int(r % p.lda) >= int(r / p.lda);
}
template <class T> bool in_ab(const Problem<T>& p, std::size_t e) {
    const std::size_t r = e % p.sab;
    return r < std::size_t(p.ldab) * p.n && int(r % p.ldab) <= p.kd;
}
template <class T> bool in_tau(const Problem<T>& p, std::size_t e) { return int(e % p.stau) < p.ntau; }

// Outside the written regions the output equals the input bit for bit.
template <class T, class In> int untouched_lost(const Problem<T>& p, const std::vector<T>& before, const std::vector<T>& after, In inside) {
    int bad = 0;
    for (std::size_t e = 0; e < after.size(); ++e) bad += !inside(p, e) && !same_bits(before[e], after[e]);
    return bad;
}
template <class T> int vs_rep(const std::vector<T>& got, std::size_t stride, int items) {
    int bad = 0;
    for (int b = kPeriod; b < items; ++b)
        for (std::size_t k = 0; k < stride; ++k) bad += !same_bits(got[std::size_t(b) * stride + k], got[std::size_t(b % kPeriod) * stride + k]);
    return bad;
}
// Diagonal elements of A and AB (row 0). A Hermitian diagonal is real: its imaginary part is not
// part of the matrix (her2k writes 0, the host's gemm-pair fallback leaves rounding there).
template <class T> bool a_diag(const Problem<T>& p, std::size_t e) { return (e % p.sa) % (p.lda + 1) == 0; }
template <class T> bool ab_diag(const Problem<T>& p, std::size_t e) { return (e % p.sab) % p.ldab == 0; }
template <class T> bool no_diag(const Problem<T>&, std::size_t) { return false; }

// The worst |got - ref| / scale over the written region; scale = ||A||_F (floored at floor) or 1.
template <class T, class In, class Dg>
double worst(const Problem<T>& p, const std::vector<T>& got, const std::vector<T>& ref, std::size_t stride, In inside, Dg diag, double floor) {
    double w = 0;
    for (std::size_t e = 0; e < got.size(); ++e) {
        if (!inside(p, e)) continue;
        const double s = floor > 0 ? std::max(p.fro(int(e / stride)), floor) : 1.0;
        const double d = diag(p, e) ? std::fabs(re_of(got[e]) - re_of(ref[e])) : err(got[e], wide(ref[e]));
        w = batchlas::verify::nanmax(w, d / s);
    }
    return w;
}

// Device vs host: info, AB, tau and A's lower triangle within tolerance; everything else unchanged.
template <class T, class Tier> void expect_parity(Dev& dev, const Problem<T>& p) {
    SCOPED_TRACE("n=" + std::to_string(p.n) + " kd=" + std::to_string(p.kd));
    const Out<T> host = run_host(dev, p);
    const Out<T> got = run_device<T, Tier>(dev, p);
    EXPECT_EQ(std::count(got.info.begin(), got.info.end(), 0), p.items);
    const double tol = 16.0 * std::max(p.n, 1) * double(std::numeric_limits<real_of<T>>::epsilon());
    const bool reduced = p.kd > 0 && p.n > p.kd;
    // Reflectors and tau are forward quantities of columns every earlier panel updated: their bound
    // grows like sqrt(panels) (float n = 100, kd = 1, graded: 1.4 tol at 99 panels on sm_89).
    const double tol_v = tol * std::sqrt(double(reduced ? (p.n - 1) / p.kd : 1));
    EXPECT_LE(worst(p, got.ab, host.ab, p.sab, in_ab<T>, ab_diag<T>, 1e-30), tol) << "AB";
    EXPECT_LE(worst(p, got.a, host.a, p.sa, in_a<T>, a_diag<T>, 1.0), tol_v) << "A lower (band and reflectors)";
    if (reduced) {
        EXPECT_LE(worst(p, got.tau, host.tau, p.stau, in_tau<T>, no_diag<T>, 0.0), 4 * tol_v) << "tau";
    } else {
        EXPECT_EQ(untouched_lost(p, p.tau, got.tau, no_diag<T>), 0) << "tau is not written without a reduction";
    }
    if constexpr (is_cplx<T>) {  // LAPACK's her2k: the reduced diagonal is real
        int im = 0;
        for (int b = 0; b < p.items && reduced; ++b)
            for (int j = p.kd; j < p.n; ++j) im += got.ab[std::size_t(b) * p.sab + std::size_t(j) * p.ldab].imag() != 0;
        EXPECT_EQ(im, 0) << "Im AB(0, j >= kd)";
    }
    EXPECT_EQ(untouched_lost(p, p.a, got.a, in_a<T>), 0) << "A upper / pads";
    EXPECT_EQ(untouched_lost(p, p.ab, got.ab, in_ab<T>), 0) << "AB pads";
    EXPECT_EQ(untouched_lost(p, p.tau, got.tau, in_tau<T>), 0) << "tau pads";
    EXPECT_EQ(vs_rep(got.a, p.sa, p.items) + vs_rep(got.ab, p.sab, p.items) + vs_rep(got.tau, p.stau, p.items), 0);
}

template <class T> class DeviceApiSy2sbTest : public DeviceCoreTest<T> {};

// The host references sytrd_sy2sb / sytrd_sb2st are GPU routes (their own suites skip on a CPU
// device; there sy2sb's workspace sizing and sb2st's sub-group kernel are missing).
bool host_reference_runs(Dev& dev) { return dev.q().get_device().is_gpu(); }


TYPED_TEST_SUITE(DeviceApiSy2sbTest, AllTypes);

struct Case {
    int n, kd;
};

TYPED_TEST(DeviceApiSy2sbTest, ParityWithTheHostRoute) {
    using T = TypeParam;
    if (!host_reference_runs(this->dev)) GTEST_SKIP() << "host sytrd_sy2sb / sb2st references run on a GPU only";
    for (const Case c : {Case{1, 3}, Case{7, 0}, Case{8, 8}, Case{9, 8}, Case{24, 8}, Case{25, 8}, Case{41, 9}, Case{42, 9}, Case{100, 1},
                         Case{97, 33}})
        expect_parity<T, Blk32>(this->dev, Problem<T>(c.n, c.kd, 6, 31 + c.n));
    expect_parity<T, Blk64>(this->dev, Problem<T>(130, 16, 6, 17));
}

// pk < kd on the last panel: its strip left of A22 takes Q^H from the left.
TYPED_TEST(DeviceApiSy2sbTest, StripCases) {
    using T = TypeParam;
    if (!host_reference_runs(this->dev)) GTEST_SKIP() << "host sytrd_sy2sb / sb2st references run on a GPU only";
    for (const Case c : {Case{27, 8}, Case{45, 13}, Case{72, 40}}) expect_parity<T, Blk32>(this->dev, Problem<T>(c.n, c.kd, 6, 5 + c.kd));
    expect_parity<T, Blk64>(this->dev, Problem<T>(72, 40, 6, 9));
}

TYPED_TEST(DeviceApiSy2sbTest, Batch1024ItemsBitIdenticalToTheirRepresentative) {
    using T = TypeParam;
    if (!host_reference_runs(this->dev)) GTEST_SKIP() << "host sytrd_sy2sb / sb2st references run on a GPU only";
    expect_parity<T, Blk32>(this->dev, Problem<T>(40, 6, 1024, 123));
}

TYPED_TEST(DeviceApiSy2sbTest, BadArgumentsReturnMinusIAndWriteNothing) {
    using T = TypeParam;
    const Problem<T> p(20, 4, 3, 77);
    for (const auto& [g, want] : {std::pair{Args{-1, -1, -1, -1}, -4}, std::pair{Args{-1000, -1, -1, 21}, -1},
                                  std::pair{Args{-1000, 4, -1, -1}, -2}, std::pair{Args{-1000, -1, 15, -1}, -3}}) {
        SCOPED_TRACE("want " + std::to_string(want));
        const Out<T> got = run_device<T, Blk32>(this->dev, p, g);
        EXPECT_EQ(std::count(got.info.begin(), got.info.end(), want), p.items);
        EXPECT_TRUE(std::equal(got.a.begin(), got.a.end(), p.a.begin(), same_bits<T>));
        EXPECT_TRUE(std::equal(got.ab.begin(), got.ab.end(), p.ab.begin(), same_bits<T>));
        EXPECT_TRUE(std::equal(got.tau.begin(), got.tau.end(), p.tau.begin(), same_bits<T>));
    }
}

// ---- sy2sb + sb2st in one user kernel ------------------------------------------------------

template <class R> struct Tri {
    std::vector<R> d, e;
    std::vector<int> info;
};

// Eigenvalues of the symmetric tridiagonal (d, e) ascending: Sturm bisection in long double.
std::vector<long double> sturm_eigs(const std::vector<long double>& d, const std::vector<long double>& e) {
    const int n = int(d.size());
    long double lo = 0, hi = 0;
    for (int i = 0; i < n; ++i) {
        const long double r = (i > 0 ? std::fabs(e[i - 1]) : 0) + (i + 1 < n ? std::fabs(e[i]) : 0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
    }
    auto below = [&](long double x) {
        int c = 0;
        long double q = 1;
        for (int i = 0; i < n; ++i) {
            q = d[i] - x - (i > 0 ? e[i - 1] * e[i - 1] / q : 0);
            if (q == 0) q = -std::numeric_limits<long double>::min();
            c += q < 0;
        }
        return c;
    };
    std::vector<long double> w(n);
    for (int k = 0; k < n; ++k) {
        long double a = lo, b = hi;
        for (int it = 0; it < 128; ++it) {
            const long double m = (a + b) / 2;
            (below(m) <= k ? a : b) = m;
        }
        w[k] = (a + b) / 2;
    }
    return w;
}

// The x86 AOT compiler (spir64_x86_64, clang 22 e79d489d) segfaults in its
// sycl-kernel-reduce-cross-barrier-values pass on the complex<double> chase of this kernel, with
// or without the sy2sb before it: SPIR images compile it out and the case skips on their devices.
#if defined(__SYCL_DEVICE_ONLY__) && (defined(__SPIR__) || defined(__SPIRV__))
template <class T> constexpr bool kChaseCompiled = !std::is_same_v<T, std::complex<double>>;
#else
template <class T> constexpr bool kChaseCompiled = true;
#endif
template <class T> bool chase_on_spir_device(Dev& dev) {
    const sycl::backend be = dev.q().get_device().get_backend();
    return std::is_same_v<T, std::complex<double>> && (be == sycl::backend::opencl || be == sycl::backend::ext_oneapi_level_zero);
}

template <class T, class Tier> Tri<real_of<T>> run_fused(Dev& dev, const Problem<T>& p) {
    using R = real_of<T>;
    const int n = p.n, kd = p.kd, lda = p.lda, ldab = p.ldab, ntau = p.ntau, nm1 = std::max(0, n - 1);
    const std::size_t sa = p.sa, sab = p.sab, stau = p.stau;
    const dv::WorkspaceSize s1 = dv::sy2sb_buffer_size<Uplo::Lower, Tier, T>(n, kd);
    const dv::WorkspaceSize s2 = dv::sb2st_buffer_size<Uplo::Lower, Blk32, T>(n, kd);
    T* a = dev.up(p.a);
    T* ab = dev.up(p.ab);
    T* tau = dev.up(p.tau);
    R* d = dev.alloc<R>(std::size_t(p.items) * n, nan_of<R>());
    R* e = dev.alloc<R>(std::size_t(p.items) * nm1 + 1, nan_of<R>());
    T* tau2 = dev.alloc<T>(std::size_t(p.items) * nm1 + 1, nan_of<T>());
    int* info = dev.up(std::vector<int>(p.items, 12345));
    std::byte* g1 = dev.up(std::vector<std::byte>(std::max<std::size_t>(s1.global_bytes * p.items, 1), std::byte{0xff}));
    std::byte* g2 = dev.up(std::vector<std::byte>(std::max<std::size_t>(s2.global_bytes * p.items, 1), std::byte{0xff}));
    dev.template run<kWG>(p.items, s1.local_bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<kWG> ex{it};
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto local = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(
            reinterpret_cast<std::byte*>(lm));
        const auto AB = dv::mat(ab + sab * b, kd + 1, n, ldab);
        int r = dv::sy2sb<Uplo::Lower, Tier>(ex, dv::mat(a + sa * b, n, n, lda), AB, dv::vec(tau + stau * b, ntau), kd,
                                             dv::Workspace{local, dv::global_slice(g1, s1, b)});
        (void)d, (void)e, (void)tau2, (void)g2, (void)s2;  // the same captures in every image
        if constexpr (kChaseCompiled<T>) {
            if (ex.team() == 0) {  // one sub-group chases the band the work-group just produced
                const dv::SgpPartition<32> pex{it.get_sub_group()};
                r |= dv::sb2st<Uplo::Lower, Blk32>(pex, AB, dv::vec(d + std::ptrdiff_t(b) * n, n), dv::vec(e + std::ptrdiff_t(b) * nm1, nm1),
                                                   dv::vec(tau2 + std::ptrdiff_t(b) * nm1, nm1), kd,
                                                   dv::Workspace{local, dv::global_slice(g2, s2, b)});
            }
        }
        ex.barrier(dv::Mem::Global);
        if (ex.leader()) info[b] = r;
    });
    return {dev.down(d, std::size_t(p.items) * n), dev.down(e, std::size_t(p.items) * nm1), dev.down(info, std::size_t(p.items))};
}

template <class T> Tri<real_of<T>> run_host_two_stage(Dev& dev, const Problem<T>& p) {
    using R = real_of<T>;
    const int n = p.n, nm1 = std::max(0, n - 1);
    T* a = dev.up(p.a);
    T* ab = dev.up(p.ab);
    T* tau = dev.up(p.tau);
    R* d = dev.alloc<R>(std::size_t(p.items) * n, nan_of<R>());
    R* e = dev.alloc<R>(std::size_t(p.items) * nm1 + 1, nan_of<R>());
    T* tau2 = dev.alloc<T>(std::size_t(p.items) * nm1 + 1, nan_of<T>());
    const MatrixView<T, MatrixFormat::Dense> av(a, n, n, p.lda, int(p.sa), p.items);
    const MatrixView<T, MatrixFormat::Dense> abv(ab, p.kd + 1, n, p.ldab, int(p.sab), p.items);
    const VectorView<T> tv(tau, p.ntau, p.items, 1, int(p.stau));
    UnifiedVector<std::byte> ws1(sytrd_sy2sb_buffer_size(*dev.ctx, av, abv, tv, Uplo::Lower, p.kd));
    sytrd_sy2sb(*dev.ctx, av, abv, tv, Uplo::Lower, p.kd, ws1.to_span()).wait();
    const VectorView<R> dvw(d, n, p.items, 1, n);
    const VectorView<R> evw(e, nm1, p.items, 1, nm1);
    const VectorView<T> t2(tau2, nm1, p.items, 1, nm1);
    UnifiedVector<std::byte> ws2(sytrd_sb2st_buffer_size(*dev.ctx, abv, dvw, evw, t2, Uplo::Lower, p.kd, 16));
    sytrd_sb2st(*dev.ctx, abv, dvw, evw, t2, Uplo::Lower, p.kd, ws2.to_span(), 16).wait();
    return {dev.down(d, std::size_t(p.items) * n), dev.down(e, std::size_t(p.items) * nm1), std::vector<int>(p.items, 0)};
}

TYPED_TEST(DeviceApiSy2sbTest, FusedTwoStageMatchesTheInputAndTheHostPipeline) {
    using T = TypeParam;
    if (!host_reference_runs(this->dev)) GTEST_SKIP() << "host sytrd_sy2sb / sb2st references run on a GPU only";
    if (chase_on_spir_device<T>(this->dev)) GTEST_SKIP() << "complex<double> chase compiled out of SPIR images (x86 AOT crash)";
    const double eps = double(std::numeric_limits<real_of<T>>::epsilon());
    for (const Case c : {Case{9, 8}, Case{27, 8}, Case{70, 32}}) {
        SCOPED_TRACE("n=" + std::to_string(c.n) + " kd=" + std::to_string(c.kd));
        const Problem<T> p(c.n, c.kd, 6, 300 + c.n);
        const auto got = run_fused<T, Blk32>(this->dev, p);
        const auto host = run_host_two_stage<T>(this->dev, p);
        EXPECT_EQ(std::count(got.info.begin(), got.info.end(), 0), p.items);
        const int n = p.n, nm1 = n - 1;
        double w_inv = 0, w_eig = 0;
        for (int b = 0; b < p.items; ++b) {
            std::vector<long double> dg(n), eg(nm1), dh(n), eh(nm1);
            long double tr_a = 0, tr_t = 0, f_t = 0;
            for (int i = 0; i < n; ++i) {
                dg[i] = got.d[std::size_t(b) * n + i], dh[i] = host.d[std::size_t(b) * n + i];
                tr_a += re_of(p.at(b, i, i)), tr_t += dg[i], f_t += dg[i] * dg[i];
            }
            for (int i = 0; i < nm1; ++i) eg[i] = got.e[std::size_t(b) * nm1 + i], eh[i] = host.e[std::size_t(b) * nm1 + i], f_t += 2 * eg[i] * eg[i];
            const double fa = std::max(p.fro(b), 1e-30);
            w_inv = batchlas::verify::nanmax(w_inv, double(std::fabs(tr_t - tr_a)) / fa);
            w_inv = batchlas::verify::nanmax(w_inv, std::fabs(std::sqrt(double(f_t)) - fa) / fa);
            const auto lg = sturm_eigs(dg, eg), lh = sturm_eigs(dh, eh);
            for (int k = 0; k < n; ++k) w_eig = batchlas::verify::nanmax(w_eig, double(std::fabs(lg[k] - lh[k])) / fa);
        }
        EXPECT_LE(w_inv, 16.0 * n * eps) << "trace / Frobenius norm of the input";
        EXPECT_LE(w_eig, 16.0 * n * eps) << "eigenvalues vs the host pipeline";
    }
}

// ---- host side: the workspace plan and with_tier ------------------------------------------

using Lower16 = dv::routine::sy2sb<Uplo::Lower, 16>;
constexpr auto kLay = dv::sy2sb_detail::panel_layout<Blk32, std::complex<double>>(37, 5);
static_assert(kLay.smem.space == dv::ws::Space::Local && kLay.smem.count == std::size_t(dv::l3::smem_elems<dv::sy2sb_detail::cfg_t<Blk32, std::complex<double>>>()));
static_assert(kLay.size.local_bytes >= kLay.smem.count * sizeof(std::complex<double>));
static_assert(kLay.taup.offset == 0 && kLay.v.offset % 16 == 0 && kLay.v.offset >= 5 * sizeof(std::complex<double>));
static_assert(kLay.m2.offset + 5 * 5 * sizeof(std::complex<double>) <= kLay.size.global_bytes && kLay.v.count == 32 * 5);
static_assert(dv::sy2sb_buffer_size<Uplo::Lower, Blk32, float>(8, 8) == dv::WorkspaceSize{});
static_assert(dv::sy2sb_buffer_size<Uplo::Lower, Blk32, float>(9, 0) == dv::WorkspaceSize{});

template <class T> std::string pick(const dv::DeviceLimits& lim, int n) {
    std::string got = "none";
    try {
        dv::with_tier<Lower16, T>(lim, n, [&]<class Tier>(Tier) { got = "blocked" + std::to_string(Tier::nb); });
    } catch (const batchlas::invalid_argument& e) {
        got = std::string("throw:") + e.what();
    }
    return got;
}

TEST(DeviceApiSy2sbHost, WithTierFiltersAndBufferSizeIsTheCarve) {
    const int n = 100;
    const dv::WorkspaceSize need = dv::sy2sb_buffer_size<Uplo::Lower, Blk32, float>(n, 16);
    EXPECT_EQ(need, (dv::tier_info<Lower16, Blk32, float>::buffer_size(n)));
    EXPECT_EQ(need, (dv::sy2sb_detail::panel_layout<Blk32, float>(n, 16).size));
    dv::DeviceLimits lim{need.local_bytes, {32}, 1024, true};
    EXPECT_EQ(pick<float>(lim, n), "blocked32");
    lim.local_mem_bytes = need.local_bytes - 1;
    EXPECT_EQ(pick<float>(lim, n), "throw:sy2sb: no device tier admits n=100");
    lim.local_mem_bytes = 0;
    EXPECT_EQ(pick<float>(lim, 1), "blocked32");  // n <= kd: nothing is reduced, nothing is needed
    lim.local_mem_bytes = std::size_t(1) << 20;
    lim.max_work_group_size = kWG - 1;
    EXPECT_EQ(pick<float>(lim, n), "throw:sy2sb: no device tier admits n=100");
    lim.max_work_group_size = 1024;
    lim.sub_group_sizes = {16};
    EXPECT_EQ(pick<float>(lim, n), "throw:sy2sb: no device tier admits n=100");
    lim.sub_group_sizes = {32};
    lim.fp64 = false;
    EXPECT_EQ(pick<double>(lim, n), "throw:sy2sb: no device tier admits n=100");
}

}  // namespace
