// SPDX-License-Identifier: MIT
// The potrf CTA / LPANEL kernels' pieces and their potf2 distributions (detail in
// include/batchlas/device/lapack/potrf.hh)
// against long-double host references; the core ops they compose with are device_core_ops_tests'.
// Guards (docs/developer/agent-guide.md section 8): non-natural ld for every operand, complex data
// with nonzero imaginary parts and a poisoned Im(diag), a large finite poison in every unread
// triangle and pad row that must survive bit-for-bit, both storage triangles, runtime executor
// widths (RtGroup at both scopes, Slot<> narrower than the work-group), batch 1024 with
// period-3 data where every item must be bit-identical to its representative (all SLM paths), info
// at a negative and at a NaN pivot in the first and a later panel, a trsm whose L diagonal is
// poisoned (the diagonal must come from the separate real vector), a herk through a rescaled prefix
// table (Rt0 > Rt), and the kernels' arithmetic (arith::Fma on sycl_device::Cx) bit-identical both
// to the std::complex run and to the src/sycl/device_scalar.hh forms the kernels used before.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

// Folded into device_lapack_potrf_tests (one binary per routine): its own namespace keeps the two
// translation units' helper and kernel names apart.
namespace device_lapack_potrf_cta_lpanel_suite {

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;
template <class T> using DevT = typename sycl_device::DevMap<T>::type;

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
template <class T> T poison() {
    if constexpr (is_cplx<T>) return T(real_of<T>(1e6), real_of<T>(-1e6));
    else return T(1e6);
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class V> V cj(V v) {
    if constexpr (std::is_same_v<V, std::complex<long double>>) return std::conj(v);
    else return v;
}
template <class T> real_of<T> im_of(T v) {
    if constexpr (is_cplx<T>) return v.imag();
    else return real_of<T>(0);
}
double rel_err(long double d, long double scale) {
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<long double>::min()));
}

struct Dev {
    std::optional<Queue> ctx = test_utils::sycl_device_visible()
                                   ? std::optional<Queue>(std::in_place, Device::default_device())
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
    template <int WG, class F> void run(int groups, std::size_t local_bytes, F f) {
        q().submit([&](sycl::handler& h) {
             // 16-byte elements: gemv_n_slm_bcast reads its tile as 16-byte vectors.
             sycl::local_accessor<sycl::vec<float, 4>, 1> lm(sycl::range<1>(local_bytes / 16 + 2), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                 f(it, reinterpret_cast<unsigned char*>(batchlas::util::get_raw_ptr(lm)));
             });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceLapackPotrfCtaTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "the potrf ops launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};
using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceLapackPotrfCtaTest, Types);

// ---- the kernels' arithmetic --------------------------------------------------------------------

enum ArSlot { kMac, kMul, kSubMul, kScaleReal, kDivReal, kRealify, kArSlots };

// Every form the private ops use, on device type D, against the device_scalar.hh helper it
// replaced (sub(mul()) stays two roundings), and the std::complex run against the Cx one: memcmp.
TYPED_TEST(DeviceLapackPotrfCtaTest, ArithIsBitIdenticalToDeviceScalar) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    constexpr int WG = 64, N = 64 * 64;
    std::vector<T> h(std::size_t(N) * 4);
    for (std::size_t i = 0; i < h.size(); ++i) h[i] = val<T>(7 + i);
    const T* in = this->dev.up(h);
    std::vector<T> zero(std::size_t(N) * kArSlots * 3, T(0));
    T* out = this->dev.up(zero);
    this->dev.template run<WG>(N / WG, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int g = static_cast<int>(it.get_global_linear_id());
        const D* x = reinterpret_cast<const D*>(in) + 4 * g;
        D* o = reinterpret_cast<D*>(out) + std::ptrdiff_t(g) * kArSlots * 3;
        using A = device::arith_t<device::arith::Fma, D>;
        const R s = sycl_device::dev_real(x[3]) + R(2);
        D acc = x[2];
        A::mac(acc, x[0], x[1]);
        o[kMac] = acc;
        o[kMul] = A::mul(x[0], A::conj(x[1]));
        o[kSubMul] = A::sub(x[2], A::mul(x[0], A::conj(x[1])));
        o[kScaleReal] = A::scale_real(s, x[0]);
        o[kDivReal] = A::div_real(x[0], s);
        o[kRealify] = A::make(A::re(x[1]), R(0));
        D ref = x[2];
        sycl_device::fma_acc(ref, x[0], x[1]);
        o[kArSlots + kMac] = ref;
        o[kArSlots + kMul] = sycl_device::dev_mul(x[0], sycl_device::dev_conj(x[1]));
        o[kArSlots + kSubMul] = sycl_device::dev_sub(x[2], sycl_device::dev_mul(x[0], sycl_device::dev_conj(x[1])));
        o[kArSlots + kScaleReal] = sycl_device::dev_mul_real(x[0], s);
        o[kArSlots + kDivReal] = sycl_device::dev_div_real(x[0], s);
        o[kArSlots + kRealify] = sycl_device::dev_from_real<D>(sycl_device::dev_real(x[1]));
        using S = device::arith_t<device::arith::Fma, T>;  // std::complex (or the real type itself)
        const T* xs = reinterpret_cast<const T*>(x);
        T* os = reinterpret_cast<T*>(o + 2 * kArSlots);
        T sacc = xs[2];
        S::mac(sacc, xs[0], xs[1]);
        os[kMac] = sacc;
        os[kMul] = S::mul(xs[0], S::conj(xs[1]));
        os[kSubMul] = S::sub(xs[2], S::mul(xs[0], S::conj(xs[1])));
        os[kScaleReal] = S::scale_real(s, xs[0]);
        os[kDivReal] = S::div_real(xs[0], s);
        os[kRealify] = S::make(S::re(xs[1]), R(0));
    });
    const auto got = this->dev.down(out, zero.size());
    int diff_dev = 0, diff_std = 0;
    for (int g = 0; g < N; ++g)
        for (int k = 0; k < kArSlots; ++k) {
            const T* o = got.data() + std::size_t(g) * kArSlots * 3;
            diff_dev += !same_bits(o[k], o[kArSlots + k]);
            diff_std += !same_bits(o[k], o[2 * kArSlots + k]);
        }
    EXPECT_EQ(diff_dev, 0) << "arith::Fma on D differs from device_scalar.hh";
    EXPECT_EQ(diff_std, 0) << "arith::Fma on std::complex differs from Cx";
}

// ---- Hermitian test matrices ------------------------------------------------------------------

// kBatch items of an n x n Hermitian positive definite matrix at leading dimension ld, stored in
// the `upper` (or lower) triangle; the other strict triangle and the pad rows hold poison, and
// Im(diag) is nonzero (it must be ignored). Representative 1 has A(kf1, kf1) = -1000 (info
// kf1 + 1), representative 2 has A(kf2, kf2) = NaN (info kf2 + 1). a holds the true matrices.
template <class T> struct HpdBatch {
    int n, ld;
    bool upper;
    std::vector<T> h;
    std::array<std::vector<LD<T>>, kPeriod> a;
    std::array<int, kPeriod> info{};
};
template <class T> HpdBatch<T> make_hpd(int n, int ld, bool upper, int kf1, int kf2, std::uint64_t seed) {
    using R = real_of<T>;
    HpdBatch<T> m{n, ld, upper, std::vector<T>(std::size_t(kBatch) * ld * n), {}, {}};
    for (int b = 0; b < kPeriod; ++b) {
        std::vector<LD<T>> bm(std::size_t(n) * n), am(std::size_t(n) * n);
        for (int i = 0; i < n * n; ++i) bm[i] = wide(val<T>(seed + 1000003u * b + i));
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < n; ++c) {
                LD<T> s = 0;
                for (int k = 0; k < n; ++k) s += bm[i + k * n] * cj(bm[c + k * n]);
                am[i + c * n] = s / static_cast<long double>(n) + (i == c ? LD<T>(1) : LD<T>(0));
            }
        for (int c = 0; c < n; ++c) am[c + c * n] = LD<T>(std::real(am[c + c * n]));
        T* p = m.h.data() + std::size_t(b) * ld * n;
        for (int c = 0; c < n; ++c)
            for (int i = 0; i < ld; ++i) p[i + c * ld] = poison<T>();
        for (int c = 0; c < n; ++c)
            for (int i = c; i < n; ++i) {
                T v = static_cast<T>(am[i + c * n]);
                if (i == c) {
                    if (b == 1 && c == kf1) v = T(R(-1000));
                    if (b == 2 && c == kf2) v = T(std::numeric_limits<R>::quiet_NaN());
                    if constexpr (is_cplx<T>) v = T(v.real(), R(0.375));
                }
                if (upper) p[c + i * ld] = is_cplx<T> ? static_cast<T>(cj(wide(v))) : v;
                else p[i + c * ld] = v;
            }
        m.a[b] = am;
    }
    m.info = {0, kf1 + 1, kf2 + 1};
    for (int b = kPeriod; b < kBatch; ++b)
        std::copy_n(m.h.data() + std::size_t(b % kPeriod) * ld * n, std::size_t(ld) * n, m.h.data() + std::size_t(b) * ld * n);
    return m;
}

// Checks one factor: items bit-identical to their representative, info exact, the unread
// triangle and pad rows still poison, Im(diag(L)) == 0, and the componentwise backward error of
// L L^H = A for the definite representative.
template <class T> void check_factor(const HpdBatch<T>& m, const std::vector<T>& got, const std::vector<int>& info, const char* where) {
    const int n = m.n, ld = m.ld;
    int bitdiff = 0, infobad = 0, poisonbad = 0, diagbad = 0;
    for (int b = 0; b < kBatch; ++b) {
        const T* g = got.data() + std::size_t(b) * ld * n;
        const T* g0 = got.data() + std::size_t(b % kPeriod) * ld * n;
        for (int i = 0; i < ld * n; ++i) bitdiff += !same_bits(g[i], g0[i]);
        infobad += info[b] != m.info[b % kPeriod];
    }
    auto L = [&](const T* g, int i, int c) -> LD<T> {  // i >= c
        return m.upper ? cj(wide(g[c + i * ld])) : wide(g[i + c * ld]);
    };
    const T* g = got.data();
    for (int c = 0; c < n; ++c)
        for (int i = 0; i < ld; ++i) {
            const bool stored = i < n && (m.upper ? i <= c : i >= c);
            if (!stored) poisonbad += !same_bits(g[i + c * ld], poison<T>());
        }
    double worst = 0;
    for (int c = 0; c < n; ++c) {
        diagbad += im_of(g[c + c * ld]) != real_of<T>(0);
        for (int i = c; i < n; ++i) {
            LD<T> s = 0;
            long double sc = 0;
            for (int k = 0; k <= c; ++k) {
                s += L(g, i, k) * cj(L(g, c, k));
                sc += std::abs(L(g, i, k)) * std::abs(L(g, c, k));
            }
            worst = std::max(worst, rel_err(std::abs(s - m.a[0][i + c * n]), sc));
        }
    }
    EXPECT_EQ(bitdiff, 0) << where << ": items differ from their representative";
    EXPECT_EQ(infobad, 0) << where << ": info";
    EXPECT_EQ(poisonbad, 0) << where << ": the unread triangle or a pad row was written";
    EXPECT_EQ(diagbad, 0) << where << ": Im(diag(L)) != 0";
    EXPECT_VERIFY(T, Check::factorization, n, worst) << where << " n=" << n;
}

// ---- CTA: lacpy_herm_lower, tri_tile_prefix_init, potf2_sg_slm, trsm_rowfrag, herk_tiles, composed
// as potrf_cta_body does (core scal and lacpy around them), on RtGroup at both scopes ------------

template <class T, class D, int NB, int TS, int WG>
void run_cta(Dev& dev, const HpdBatch<T>& m, std::vector<T>& out, std::vector<int>& info) {
    using R = real_of<T>;
    constexpr device::Scope kScope = WG == 32 ? device::Scope::Sub : device::Scope::Group;
    const int n = m.n, ldg = m.ld, lda = n + 2;
    const int Rt0 = ((n > NB ? n - NB : 0) + TS - 1) / TS;
    const std::size_t s_bytes = std::size_t(lda) * n * sizeof(D);
    const std::size_t bytes = s_bytes + NB * sizeof(R) + std::size_t(Rt0 + 2) * sizeof(int);
    D* a = reinterpret_cast<D*>(dev.up(m.h));
    int* inf = dev.up(std::vector<int>(kBatch, -7));
    const bool upper = m.upper;
    dev.template run<WG>(kBatch, bytes, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        D* S = reinterpret_cast<D*>(lm);
        R* diag = reinterpret_cast<R*>(lm + s_bytes);
        int* off = reinterpret_cast<int*>(lm + s_bytes + NB * sizeof(R));
        int* fail = off + Rt0 + 1;
        auto bar = [&] {
            if constexpr (WG == 32) sycl::group_barrier(it.get_sub_group());
            else sycl::group_barrier(it.get_group());
        };
        const device::RtGroup<kScope> ex(it);
        D* Ag = a + std::ptrdiff_t(b) * ldg * n;
        const auto A = device::detail::slm_mat(S, lda, n, lda);
        const device::Vec<R, device::Ext<NB>, device::Unit, device::Mem::Local> dg{diag, {}};
        device::detail::lacpy_herm_lower(ex, device::detail::HermLower<D>{Ag, ldg, upper}, A, n);
        device::scal(ex, device::Zero{}, dg);
        device::detail::tri_tile_prefix_init(ex, off, Rt0);
        if (ex.leader()) *fail = 0;
        bar();
        bool ok = true;
        for (int j = 0; j < n; j += NB) {
            const int ib = std::min(NB, n - j), m2 = n - j - ib;
            const auto A11 = A.sub(j, j, ib, ib);
            const auto A21 = A.sub(j + ib, j, m2, ib);
            if (ok && it.get_sub_group().get_group_linear_id() == 0)
                device::potf2_sg_slm<NB, device::detail::InvSqrt>(device::SubGroup<32>{it.get_sub_group()}, A11, ib, dg, fail, j);
            bar();
            ok = (*fail == 0);
            if (ok && m2 > 0) device::detail::trsm_rowfrag<Side::Right, Uplo::Lower, Transpose::ConjTrans, NB>(ex, A11, dg, A21, ib);
            bar();
            if (ok && m2 > 0) {
                const int Rt = (m2 + TS - 1) / TS;
                device::detail::herk_tiles<Uplo::Lower, TS>(ex, device::NegOne{}, A21, device::One{}, A.sub(j + ib, j + ib, m2, m2),
                                                 device::detail::TriTilePrefix{off, Rt, Rt0 - Rt});
            }
            bar();
        }
        const auto G = device::mat_wide(Ag, n, n, ldg);
        if (upper) device::lacpy<device::Part::Upper, Transpose::ConjTrans>(ex, A.sub(0, 0, n, n), G);
        else device::lacpy<device::Part::Lower>(ex, A.sub(0, 0, n, n), G);
        if (ex.leader()) inf[b] = *fail;
    });
    out = dev.down(reinterpret_cast<const T*>(a), m.h.size());
    info = dev.down(inf, kBatch);
}

template <class T, int TS, int WG> void cta_case(Dev& dev, int n, bool upper) {
    const auto m = make_hpd<T>(n, n + 3, upper, n / 2, std::min(n - 1, 9), 11 + n);
    std::vector<T> out, out_dev;
    std::vector<int> info, info_dev;
    run_cta<T, T, 8, TS, WG>(dev, m, out, info);
    run_cta<T, DevT<T>, 8, TS, WG>(dev, m, out_dev, info_dev);
    const std::string where = std::string("cta ") + (upper ? "upper" : "lower") + " WG=" + std::to_string(WG);
    check_factor(m, out, info, where.c_str());
    int cxdiff = 0;
    for (std::size_t i = 0; i < out.size(); ++i) cxdiff += !same_bits(out[i], out_dev[i]);
    EXPECT_EQ(cxdiff, 0) << where << ": the device scalar (Cx) run differs from std::complex";
    EXPECT_EQ(info, info_dev) << where;
}

TYPED_TEST(DeviceLapackPotrfCtaTest, CtaRightLookingFactor) {
    using T = TypeParam;
    for (int n : {1, 7, 8, 9, 37}) {
        for (bool upper : {false, true}) {
            cta_case<T, 4, 32>(this->dev, n, upper);   // the SubGroup scope: one sub-group per matrix
            cta_case<T, 2, 128>(this->dev, n, upper);  // the WorkGroup scope, potf2 on sub-group 0
        }
    }
}

// ---- trsm_rowfrag: X := X L^-H, L's stored diagonal poisoned ---------------------------------

TYPED_TEST(DeviceLapackPotrfCtaTest, TrsmRowfragRightLowerConjTrans) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int NB = 8, WG = 64;
    for (int kb : {3, 8}) {
        const int m = 70, ldx = m + 5, ldl = NB + 1, items = 48;  // items < WG: a runtime stride
        std::vector<T> hx(std::size_t(kBatch) * ldx * kb), hl(std::size_t(kBatch) * ldl * kb, poison<T>());
        std::vector<R> hd(std::size_t(kBatch) * NB);
        for (int b = 0; b < kBatch; ++b) {
            const int r = b % kPeriod;
            for (int i = 0; i < ldx * kb; ++i) hx[std::size_t(b) * ldx * kb + i] = val<T>(31 + 7919u * r + i);
            for (int c = 0; c < kb; ++c) {
                hd[std::size_t(b) * NB + c] = R(1.5) + R(c % 3) * R(0.25);
                for (int q = 0; q < c; ++q) hl[std::size_t(b) * ldl * kb + c + q * ldl] = val<T>(57 + 104729u * r + c * kb + q);
            }
        }
        T* x = this->dev.up(hx);
        const T* l = this->dev.up(hl);
        const R* d = this->dev.up(hd);
        this->dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
            const int b = static_cast<int>(it.get_group_linear_id());
            const int lid = static_cast<int>(it.get_local_linear_id());
            if (lid >= items) return;
            const device::Slot<> ex{lid, items};
            const auto L = device::mat(l + std::ptrdiff_t(b) * ldl * kb, kb, kb, ldl);
            const auto X = device::mat(x + std::ptrdiff_t(b) * ldx * kb, m, kb, ldx);
            const device::Vec<const R, device::Ext<NB>> dv{d + std::ptrdiff_t(b) * NB, {}};
            device::detail::trsm_rowfrag<Side::Right, Uplo::Lower, Transpose::ConjTrans, NB>(ex, L, dv, X, kb);
        });
        const auto got = this->dev.down(x, hx.size());
        int bitdiff = 0;
        for (int b = 0; b < kBatch; ++b)
            for (int i = 0; i < ldx * kb; ++i)
                bitdiff += !same_bits(got[std::size_t(b) * ldx * kb + i], got[std::size_t(b % kPeriod) * ldx * kb + i]);
        double worst = 0;
        int padbad = 0;
        for (int b = 0; b < kPeriod; ++b) {
            const T* g = got.data() + std::size_t(b) * ldx * kb;
            const T* x0 = hx.data() + std::size_t(b) * ldx * kb;
            const T* l0 = hl.data() + std::size_t(b) * ldl * kb;
            for (int c = 0; c < kb; ++c)
                for (int i = m; i < ldx; ++i) padbad += !same_bits(g[i + c * ldx], x0[i + c * ldx]);
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < kb; ++c) {
                    // residual of X_new L^H = X: sum_{q<c} Xn(r,q) conj(L(c,q)) + Xn(r,c) d(c) - X(r,c)
                    LD<T> s = wide(g[r + c * ldx]) * static_cast<long double>(hd[std::size_t(b) * NB + c]);
                    long double sc = std::abs(s) + std::abs(wide(x0[r + c * ldx]));
                    for (int q = 0; q < c; ++q) {
                        const LD<T> t = wide(g[r + q * ldx]) * cj(wide(l0[c + q * ldl]));
                        s += t;
                        sc += std::abs(t);
                    }
                    worst = std::max(worst, rel_err(std::abs(s - wide(x0[r + c * ldx])), sc));
                }
        }
        EXPECT_EQ(bitdiff, 0) << "kb=" << kb;
        EXPECT_EQ(padbad, 0) << "kb=" << kb << ": rows >= m were written";
        EXPECT_VERIFY(T, Check::blas, kb, worst) << "trsm_rowfrag kb=" << kb;
    }
}

// ---- herk_tiles + TriTilePrefix: C := C - A A^H on the lower triangle -----------------------

template <class T, int TS> void herk_case(Dev& dev, int mm, int k) {
    using R = real_of<T>;
    constexpr int WG = 64;
    const int lda = mm + 2, ldc = mm + 3;
    const int Rt = (mm + TS - 1) / TS, Rt0 = Rt + 3;
    std::vector<T> ha(std::size_t(kBatch) * lda * k), hc(std::size_t(kBatch) * ldc * mm);
    for (int b = 0; b < kBatch; ++b) {
        const int r = b % kPeriod;
        for (int i = 0; i < lda * k; ++i) ha[std::size_t(b) * lda * k + i] = val<T>(101 + 7919u * r + i);
        for (int c = 0; c < mm; ++c)
            for (int i = 0; i < ldc; ++i) {
                T v = (i >= c && i < mm) ? val<T>(503 + 104729u * r + i + c * ldc) : poison<T>();
                if (i == c) v = T(v) + T(R(4));
                hc[std::size_t(b) * ldc * mm + i + c * ldc] = v;
            }
    }
    const T* a = dev.up(ha);
    T* cc = dev.up(hc);
    dev.template run<WG>(kBatch, std::size_t(Rt0 + 1) * sizeof(int), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> ex{it};
        int* off = reinterpret_cast<int*>(lm);
        device::detail::tri_tile_prefix_init(ex, off, Rt0);
        ex.barrier();
        const auto A = device::mat(a + std::ptrdiff_t(b) * lda * k, mm, k, lda);
        const auto C = device::mat(cc + std::ptrdiff_t(b) * ldc * mm, mm, mm, ldc);
        device::detail::herk_tiles<Uplo::Lower, TS>(ex, device::NegOne{}, A, device::One{}, C, device::detail::TriTilePrefix{off, Rt, Rt0 - Rt});
    });
    const auto got = dev.down(cc, hc.size());
    int bitdiff = 0, poisonbad = 0, diagbad = 0;
    double worst = 0;
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < ldc * mm; ++i)
            bitdiff += !same_bits(got[std::size_t(b) * ldc * mm + i], got[std::size_t(b % kPeriod) * ldc * mm + i]);
    for (int b = 0; b < kPeriod; ++b) {
        const T* g = got.data() + std::size_t(b) * ldc * mm;
        const T* c0 = hc.data() + std::size_t(b) * ldc * mm;
        const T* a0 = ha.data() + std::size_t(b) * lda * k;
        for (int c = 0; c < mm; ++c)
            for (int i = 0; i < ldc; ++i) {
                if (i < c || i >= mm) {
                    poisonbad += !same_bits(g[i + c * ldc], c0[i + c * ldc]);
                    continue;
                }
                LD<T> s = wide(c0[i + c * ldc]);
                long double sc = std::abs(s);
                for (int p = 0; p < k; ++p) {
                    const LD<T> t = wide(a0[i + p * lda]) * cj(wide(a0[c + p * lda]));
                    s -= t;
                    sc += std::abs(t);
                }
                if (i == c) {
                    s = LD<T>(std::real(s));
                    diagbad += im_of(g[i + c * ldc]) != R(0);
                }
                worst = std::max(worst, rel_err(std::abs(wide(g[i + c * ldc]) - s), sc));
            }
    }
    EXPECT_EQ(bitdiff, 0) << "TS=" << TS;
    EXPECT_EQ(poisonbad, 0) << "TS=" << TS << ": the strict upper triangle or a pad row was written";
    EXPECT_EQ(diagbad, 0) << "TS=" << TS << ": Im(diag(C)) != 0";
    EXPECT_VERIFY(T, Check::blas, k, worst) << "herk_tiles TS=" << TS << " m=" << mm;
}

TYPED_TEST(DeviceLapackPotrfCtaTest, HerkTilesLowerThroughRescaledPrefix) {
    using T = TypeParam;
    herk_case<T, 4>(this->dev, 37, 8);
    herk_case<T, 2>(this->dev, 37, 5);
    herk_case<T, 4>(this->dev, 3, 8);  // one tile, trimmed both ways
}

// ---- LPANEL: frag_load_row_herm_lower, lacpy<DivMod, EdgeZeroPad> of conj_view, gemv_n_slm_bcast,
// frag_store_row_herm_lower, potf2_panel_rows, composed as potrf_lpanel_body does on a Slot<>
// slot narrower than the work-group (the packed-G case) --------------------------------------

template <class T, class D, int NB, int WG, int SLOT>
void run_lpanel(Dev& dev, const HpdBatch<T>& m, std::vector<T>& out, std::vector<int>& info) {
    static_assert(WG % SLOT == 0, "whole slots");
    constexpr int G = WG / SLOT;
    const int n = m.n, ldg = m.ld, slda = n + 1;
    const std::size_t b_bytes = std::size_t(NB) * NB * sizeof(D);
    const std::size_t per = b_bytes + std::size_t(slda) * NB * sizeof(D) + 16;
    D* a = reinterpret_cast<D*>(dev.up(m.h));
    int* inf = dev.up(std::vector<int>(kBatch, -7));
    dev.template run<WG>(kBatch / G, per * G, [=](sycl::nd_item<1> it, unsigned char* lm) {
        using Ar = device::arith_t<device::arith::Fma, D>;
        const int lid = static_cast<int>(it.get_local_linear_id());
        const int slot = lid / SLOT, tid = lid % SLOT;
        const int b = static_cast<int>(it.get_group_linear_id()) * G + slot;
        unsigned char* mine = lm + per * slot;
        D* sB = reinterpret_cast<D*>(mine);
        D* sA = reinterpret_cast<D*>(mine + b_bytes);
        int* fail = reinterpret_cast<int*>(mine + b_bytes + std::size_t(slda) * NB * sizeof(D));
        const device::Slot<> ex{tid, SLOT};
        const auto A = device::mat_wide(a + std::ptrdiff_t(b) * ldg * n, n, n, ldg);
        const auto P = device::detail::slm_mat(sA, n, NB, slda);
        const auto B = device::detail::slm_tile<NB, NB>(sB);
        auto bar = [&] { sycl::group_barrier(it.get_group()); };
        if (ex.leader()) *fail = 0;
        bar();
        const int row = tid;
        bool ok = true;
        for (int j = 0; j < n; j += NB) {
            const int ib = std::min(NB, n - j);
            const bool panel_ok = ok, live = ok && row >= j && row < n;
            device::Frag<D, NB> rp, rS;
            device::detail::frag_load_row_herm_lower<NB>(rp, A, row, j, ib, live);
            device::fill(rS, Ar::zero());
            for (int k = 0; k < j; k += NB) {
                const int kb = std::min(NB, j - k);
                device::lacpy<device::Part::Full, Transpose::NoTrans, device::alg::DivMod, device::alg::EdgeZeroPad>(ex, device::conj_view(A.sub(j, k, ib, kb)), B);
                bar();
                if (live) {
                    device::Frag<D, NB> rA;
                    device::load_row(rA, A.sub(0, k, n, kb), row, kb);
                    device::detail::gemv_n_slm_bcast<NB>(rS, rA, sB);
                }
                bar();
            }
            if (live) device::detail::frag_store_row_herm_lower<NB>(P, row, j, ib, [&](int i) { return Ar::sub(rp[i], rS[i]); });
            bar();
            ok = device::potf2_panel_rows<device::detail::InvSqrt>(ex, P, j, ib, n, ok, fail, bar);
            if (panel_ok) device::lacpy<device::Part::Lower>(ex, P.sub(j, 0, n - j, ib), A.sub(j, j, n - j, ib));
            bar();
        }
        if (ex.leader()) inf[b] = *fail;
    });
    out = dev.down(reinterpret_cast<const T*>(a), m.h.size());
    info = dev.down(inf, kBatch);
}

TYPED_TEST(DeviceLapackPotrfCtaTest, LpanelLeftLookingFactor) {
    using T = TypeParam;
    for (int n : {1, 5, 8, 33, 61}) {
        const auto m = make_hpd<T>(n, n + 4, false, n / 2, std::min(n - 1, 12), 77 + n);
        std::vector<T> out, out_dev;
        std::vector<int> info, info_dev;
        run_lpanel<T, T, 8, 128, 64>(this->dev, m, out, info);  // two matrices per work-group
        run_lpanel<T, DevT<T>, 8, 128, 64>(this->dev, m, out_dev, info_dev);
        check_factor(m, out, info, "lpanel");
        int cxdiff = 0;
        for (std::size_t i = 0; i < out.size(); ++i) cxdiff += !same_bits(out[i], out_dev[i]);
        EXPECT_EQ(cxdiff, 0) << "lpanel n=" << n << ": the device scalar (Cx) run differs from std::complex";
        EXPECT_EQ(info, info_dev) << "lpanel n=" << n;
    }
}

}  // namespace

}  // namespace device_lapack_potrf_cta_lpanel_suite
