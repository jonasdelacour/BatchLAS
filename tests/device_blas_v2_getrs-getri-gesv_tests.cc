// SPDX-License-Identifier: MIT
// The device LA ops getrs / getri / gesv run on (getrs_detail in lapack/getrs.hh
// and gesv_detail in lapack/gesv.hh, and the core ones they took over: SafeRecip, laswp_index,
// AtomicMinSlot, shift_reduce, conj_if, op_view) against host references, as the LU kernels call
// them; RtGroup, lacpy, laswp, select and Partition::bcast are covered by device_blas_v2_tests.
// Guards (docs/developer/agent-guide.md section 8): SafeRecip bit for bit against getf2's
// open-coded sycl_device scaling at zero, -0, denormal, inf and both sides of Smith's branch;
// shift_reduce bit for bit against a host replay of the shift-down order;
// non-natural ld and sub-view offsets; complex data with nonzero imaginary parts and conjugated
// views; the unread triangle (and a Unit diagonal) poisoned with NaN; ld pads and a column past
// nrhs poisoned and checked untouched; batch 1024 with period-3 data where every item must match
// its representative bit for bit; run-time widths 64, 160 (not a power of two) and 1024 (the
// ceiling).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <batchlas/device.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/lapack/gesv.hh>
#include <batchlas/device/lapack/getrs.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace lu = batchlas::device::getrs_detail;
namespace lug = batchlas::device::gesv_detail;
namespace sd = batchlas::sycl_device;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;
template <class T> using DevT = typename sd::DevMap<T>::type;

constexpr int kBatch = 1024;  // saturating batch: items repeat with period kPeriod
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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }

// kPeriod representatives built by make(b, i), copied to every item b from b % kPeriod.
template <class T, class Make> std::vector<T> batched(int per_item, Make make, int items = kBatch) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < kPeriod; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = make(b, i);
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}

template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }

// Items whose bits differ from their representative's.
template <class T> int twin_diffs(const std::vector<T>& h, int per_item, int items = kBatch) {
    int d = 0;
    for (int b = kPeriod; b < items; ++b)
        for (int i = 0; i < per_item; ++i)
            d += !same_bits(h[std::size_t(b) * per_item + i], h[std::size_t(b % kPeriod) * per_item + i]);
    return d;
}

struct Dev {
    Queue ctx{Device::default_device()};
    sycl::queue& q() { return sycl_queue(ctx); }
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
             sycl::local_accessor<unsigned char, 1> lm(sycl::range<1>(std::max<std::size_t>(local_bytes, 16)), h);
             h.parallel_for(sycl::nd_range<1>(std::size_t(groups) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it, batchlas::util::get_raw_ptr(lm)); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2LuTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2LuTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> T conj_h(T v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// True when the device flushes a denormal product to zero (Metal); the CUDA build keeps them.
template <class T> bool device_flushes(Dev& dev) {
    using R = real_of<T>;
    std::vector<R> h{std::numeric_limits<R>::denorm_min() * R(4), R(1)};
    R* p = dev.up(h);
    dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        if (it.get_local_linear_id() == 0) p[0] = p[0] * p[1];
    });
    return dev.down(p, 2)[0] == R(0);
}

// ---- getf2 pivot scaling (gesv) ------------------------------------------------------------

// SafeRecip against gesv_tiny's legacy open-coded scaling (sycl_device), bit for bit: the
// multiply arm, the divide arm (denormal, huge) and the singular arm (0, -0), both sides of
// Smith's branch for complex.
TYPED_TEST(DeviceBlasV2LuTest, SafeRecipIsGetf2ScalingBitForBit) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    constexpr int n = 96, kSlots = 2;
    const R inf = std::numeric_limits<R>::infinity(), dn = std::numeric_limits<R>::denorm_min();
    const R big = std::numeric_limits<R>::max() / R(4), tiny = std::numeric_limits<R>::min();
    std::vector<T> ha(n), hb(n);
    for (int i = 0; i < n; ++i) ha[i] = val<T>(10 + i), hb[i] = val<T>(500 + i);
    const std::array<T, 8> edge{T(0), T(dn), T(inf), T(big), T(tiny), T(-dn), T(1), T(R(-0.0))};
    for (int i = 0; i < 8; ++i) hb[i] = edge[i];
    if constexpr (is_cplx<T>) {
        hb[8] = T(R(1e-3), R(7)), hb[9] = T(R(7), R(1e-3)), hb[10] = T(R(0), big), hb[11] = T(R(-3), R(-3));
        hb[12] = T(R(0), dn), hb[13] = T(dn, dn), hb[14] = T(R(0), R(1));
    }
    T* a = this->dev.up(ha);
    T* bb = this->dev.up(hb);
    std::vector<T> ho(std::size_t(2) * n * kSlots, nan_v<T>());
    std::vector<int> hf(std::size_t(2) * n, -1);
    T* o = this->dev.up(ho);
    int* f = this->dev.up(hf);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const D* ad = reinterpret_cast<const D*>(a);
        const D* bd = reinterpret_cast<const D*>(bb);
        D* got = reinterpret_cast<D*>(o);
        D* ref = got + n * kSlots;
        for (int i = static_cast<int>(it.get_local_linear_id()); i < n; i += 32) {
            const D x = ad[i], y = bd[i];
            const auto sr = v2::SafeRecip<D>::make(y);
            got[i * kSlots + 0] = sr.apply(x, x);                    // gesv step 4: keep x if singular
            got[i * kSlots + 1] = sr.apply(x, v2::Arith<D>::zero()); // gesv step 6: zero if singular
            f[2 * i] = sr.zero;
            const bool zero = sd::dev_is_zero(y);
            const D rc = sd::dev_recip(y);
            const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
            D s1 = x, s2 = D{};
            if (use_mul) s1 = sd::dev_mul(x, rc), s2 = sd::dev_mul(x, rc);
            else if (!zero) s1 = sd::dev_div(x, y), s2 = sd::dev_div(x, y);
            ref[i * kSlots + 0] = s1, ref[i * kSlots + 1] = s2;
            f[2 * i + 1] = zero;
        }
    });
    const auto h = this->dev.down(o, ho.size());
    const auto fl = this->dev.down(f, hf.size());
    int diff = 0, fdiff = 0;
    for (int i = 0; i < n * kSlots; ++i) diff += !same_bits(h[i], h[n * kSlots + i]);
    for (int i = 0; i < n; ++i) fdiff += fl[2 * i] != fl[2 * i + 1];
    EXPECT_EQ(diff, 0) << "SafeRecip differs from the sycl_device getf2 scaling";
    EXPECT_EQ(fdiff, 0) << "SafeRecip::zero";
    EXPECT_TRUE(fl[0] == 1 && fl[2 * 7] == 1) << "0 and -0 are exact zeros";
}

// ---- views and the row gather on a run-time width ------------------------------------------

// Per item: src (ldA x NA) with the view at (i0, j0), m x n. out regions (each poisoned):
// [lapmr_copy | core lacpy of conj_if(trans(src))], ld ldB / ldT.
template <class T, int WG> void gather_case(Dev& dev) {
    using D = DevT<T>;
    constexpr int m = 37, n = 5, i0 = 2, j0 = 1, ldA = m + 6, NA = n + 2, ldB = m + 5, ldT = n + 2;
    constexpr int kSrc = ldA * NA, kOut = ldB * n + ldT * m;
    auto perm = [](int i) { return (i * 7 + 3) % m; };
    auto hs = batched<T>(kSrc, [](int b, int i) { return val<T>(std::uint64_t(b) * 7919u + i); });
    std::vector<T> ho(std::size_t(kBatch) * kOut, nan_v<T>());
    T* s = dev.up(hs);
    T* o = dev.up(ho);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup<> ex(it);  // the width is the launch's, a run-time value
        const auto src = v2::mat_wide(reinterpret_cast<const D*>(s) + std::size_t(b) * kSrc, m + i0, NA, ldA).sub(i0, j0, m, n);
        D* ob = reinterpret_cast<D*>(o) + std::size_t(b) * kOut;
        lu::lapmr_copy(ex, src, perm, v2::mat_wide(ob, m, n, ldB));
        v2::lacpy<v2::Part::Full, Transpose::NoTrans, v2::alg::DivMod>(ex, v2::conj_if(v2::op_view<Transpose::Trans>(src), true),
                                                                     v2::mat_wide(ob + ldB * n, n, m, ldT));
    });
    const auto h = dev.down(o, ho.size());
    EXPECT_EQ(twin_diffs(h, kOut), 0) << "WG " << WG << ": items differ from their representative";
    int bad = 0, pad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* sb = hs.data() + std::size_t(b) * kSrc;
        const T* g = h.data() + std::size_t(b) * kOut;
        auto A = [&](int i, int j) { return sb[(i0 + i) + std::size_t(j0 + j) * ldA]; };
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < ldB; ++i) {
                const T v = g[i + j * ldB];
                if (i >= m) pad += !std::isnan(std::real(v));
                else bad += !same_bits(v, A(perm(i), j));
            }
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < ldT; ++j) {
                const T v = g[ldB * n + j + i * ldT];
                if (j >= n) pad += !std::isnan(std::real(v));
                else bad += !same_bits(v, conj_h(A(i, j)));
            }
    }
    EXPECT_EQ(bad, 0) << "WG " << WG << ": lapmr_copy / conj_if(trans) values";
    EXPECT_EQ(pad, 0) << "WG " << WG << ": a write outside the view";
}

TYPED_TEST(DeviceBlasV2LuTest, LapmrCopyAndConjTransViewsOnRunTimeWidth) {
    gather_case<TypeParam, 64>(this->dev);
    gather_case<TypeParam, 160>(this->dev);
    gather_case<TypeParam, 1024>(this->dev);
}

// ---- interchanges (getri) ------------------------------------------------------------------

// 0-based ?getrf-shaped pivots (p in [k, n)), about a third of them p == k.
std::vector<int> pivots(int n, int b) {
    std::vector<int> p(n);
    for (int k = 0; k < n; ++k) p[k] = (mix(std::uint64_t(b) * 131u + k) % 3 == 0) ? k : k + int(mix(77u + k + 1000u * b) % (n - k));
    return p;
}
void host_laswp(std::vector<int>& x, const std::vector<int>& p, int k0, int k1, bool forward) {
    auto step = [&](int k) {
        if (p[k] != k) std::swap(x[k], x[p[k]]);
    };
    if (forward)
        for (int k = k0; k < k1; ++k) step(k);
    else
        for (int k = k1 - 1; k >= k0; --k) step(k);
}

// laswp_index, both orders on the full list and Forward on a sub-range, for every i against
// laswp applied to the identity on the host (another walk: it moves values, not one index).
TEST(DeviceBlasV2LuPerm, LaswpIndexIsLaswpOfTheIdentity) {
    Dev dev;
    if (!dev.ctx.device().supports_sub_group_size(32)) GTEST_SKIP() << "v2 tests launch sub-group size 32";
    constexpr int WG = 64, n = 45, k0 = 3, k1 = n - 4, kOut = 3 * n;
    std::vector<int> hp(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b) {
        const auto p = pivots(n, b % kPeriod);
        for (int k = 0; k < n; ++k) hp[std::size_t(b) * n + k] = p[k] + 1;  // 1-based on the wire
    }
    std::vector<int> ho(std::size_t(kBatch) * kOut, -7);
    int* pv = dev.up(hp);
    int* o = dev.up(ho);
    dev.run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup<> ex(it);
        const auto ipiv = v2::vec(pv + std::size_t(b) * n, n);
        int* ob = o + std::size_t(b) * kOut;
        v2::for_each_owned(ex, n, [&](int i) {
            ob[i] = v2::laswp_index<v2::dir::Forward>(ipiv, 0, n, i);
            ob[n + i] = v2::laswp_index<v2::dir::Backward>(ipiv, 0, n, i);
            ob[2 * n + i] = v2::laswp_index<v2::dir::Forward>(ipiv, k0, k1, i);
        });
    });
    const auto h = dev.down(o, ho.size());
    EXPECT_EQ(twin_diffs(h, kOut), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const auto p = pivots(n, b);
        const int* g = h.data() + std::size_t(b) * kOut;
        for (int pass = 0; pass < 3; ++pass) {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[i] = i;
            host_laswp(id, p, pass == 2 ? k0 : 0, pass == 2 ? k1 : n, pass != 1);
            for (int pos = 0; pos < n; ++pos) bad += g[pass * n + id[pos]] != pos;  // where i landed
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- info: the first exact zero on a diagonal (getri) --------------------------------------

// Pattern b % 4: no zero (a complex item carries (0, 1), which is not zero), zero at 0, zeros at
// 77 and 200 (the second a -0), zero at n - 1. n > WG, so every item strides twice.
TYPED_TEST(DeviceBlasV2LuTest, AtomicMinSlotFirstExactZero) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    constexpr int WG = 256, n = 300, kPat = 4;
    auto zero_at = [](int pat, int i) { return (pat == 1 && i == 0) || (pat == 2 && (i == 77 || i == 200)) || (pat == 3 && i == n - 1); };
    std::vector<T> hd(std::size_t(kPat) * n);
    for (int pat = 0; pat < kPat; ++pat)
        for (int i = 0; i < n; ++i) {
            T v = val<T>(std::uint64_t(pat) * 1000u + i) + T(R(2));
            if (zero_at(pat, i)) v = T(R(pat == 2 && i == 200 ? -0.0 : 0.0));
            if constexpr (is_cplx<T>)
                if (pat == 0 && i == 5) v = T(R(0), R(1));
            hd[std::size_t(pat) * n + i] = v;
        }
    std::vector<T> hdiag(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b) std::copy_n(hd.data() + std::size_t(b % kPat) * n, n, hdiag.data() + std::size_t(b) * n);
    std::vector<int> hi(kBatch, -1);
    T* dg = this->dev.up(hdiag);
    int* info = this->dev.up(hi);
    this->dev.template run<WG>(kBatch, sizeof(int), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup<> ex(it);
        const auto slot = v2::atomic_min_slot(reinterpret_cast<int*>(lm));
        // ld 0 makes element (i, i) dg[i]: the diagonal alone is stored.
        const auto a = v2::mat_wide(reinterpret_cast<const D*>(dg) + std::size_t(b) * n, n, n, 0);
        slot.reset(ex);
        ex.barrier(v2::Mem::Global);
        v2::for_each_owned(ex, n, [&](int i) {
            if (sd::dev_is_zero(a(i, i))) slot.offer(i + 1);
        });
        ex.barrier(v2::Mem::Global);
        if (ex.leader()) info[b] = slot.or_zero();
    });
    const auto h = this->dev.down(info, hi.size());
    const int want[kPat] = {0, 1, 78, n};
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) bad += h[b] != want[b % kPat];
    EXPECT_EQ(bad, 0) << "info: item 0..3 = " << h[0] << " " << h[1] << " " << h[2] << " " << h[3];
}

// ---- sub-group sums and triangular solves (getrs) ------------------------------------------

// Lane 0's total against a host replay of the shift-down order (off = 16, 8, .., 1), bit for bit:
// another order (a butterfly, reduce_over_group) rounds differently on these inputs.
TYPED_TEST(DeviceBlasV2LuTest, ShiftReduceIsTheShiftDownOrderBitForBit) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    auto hx = batched<T>(32, [](int b, int i) { return val<T>(std::uint64_t(b) * 401u + i) * T(R(1 + (i % 5) * 1000)); });
    std::vector<T> ho(kBatch, nan_v<T>());
    T* x = this->dev.up(hx);
    T* o = this->dev.up(ho);
    this->dev.template run<32>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const auto add = [](D u, D v) { return v2::Arith<D>::add(u, v); };
        const D s = v2::shift_reduce<32>(sg, reinterpret_cast<const D*>(x)[std::size_t(b) * 32 + lane], add);
        if (lane == 0) reinterpret_cast<D*>(o)[b] = s;
    });
    const auto h = this->dev.down(o, ho.size());
    EXPECT_EQ(twin_diffs(h, 1), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        std::array<R, 32> re{}, im{};
        for (int l = 0; l < 32; ++l) re[l] = std::real(hx[std::size_t(b) * 32 + l]), im[l] = std::imag(hx[std::size_t(b) * 32 + l]);
        for (int off = 16; off > 0; off >>= 1)
            for (int l = 0; l + off < 32; ++l) re[l] = re[l] + re[l + off], im[l] = im[l] + im[l + off];
        bad += !same_bits(std::real(h[b]), re[0]);
        if constexpr (is_cplx<T>) bad += !same_bits(std::imag(h[b]), im[0]);
    }
    EXPECT_EQ(bad, 0) << "lane 0 is not the shift-down sum";
}

// A jb x jb block at ld jb + 1 whose op(A) (A, or A^H when Tr) is UL triangular: |diag| in [1, 2],
// off-diagonal scaled by 1/jb; the other triangle, and the diagonal when Unit, are NaN.
template <class T, Uplo UL, Diag DG, bool Tr> std::vector<T> tri_blocks(int jb, int ld) {
    using R = real_of<T>;
    return batched<T>(ld * jb, [=](int b, int e) {
        const int i = e % ld, j = e / ld;
        if (i >= jb) return nan_v<T>();
        const int oi = Tr ? j : i, oj = Tr ? i : j;  // op(A)(oi, oj) is stored at (i, j)
        const bool stored = UL == Uplo::Lower ? oi >= oj : oi <= oj;
        if (!stored || (DG == Diag::Unit && i == j)) return nan_v<T>();
        const T v = val<T>(std::uint64_t(b) * 65537u + e + 11u);
        if (i == j) return T(R(1.5)) + T(R(0.5)) * v;
        return v / T(R(jb));
    });
}
template <class T, Uplo UL, Diag DG, bool Tr> LD<T> op_at(const T* a, int ld, int i, int j) {
    if (i == j && DG == Diag::Unit) return LD<T>(1);
    if (UL == Uplo::Lower ? i < j : i > j) return LD<T>(0);
    return Tr ? wide(conj_h(a[j + std::size_t(i) * ld])) : wide(a[i + std::size_t(j) * ld]);
}
// Backward error of x for op(A) x = rhs, componentwise in the blas scale.
template <class T, Uplo UL, Diag DG, bool Tr> double tri_residual(const T* a, int ld, int jb, const T* x, const T* rhs) {
    double worst = 0;
    for (int i = 0; i < jb; ++i) {
        LD<T> s = 0;
        long double sc = std::abs(wide(rhs[i]));
        for (int j = 0; j < jb; ++j) {
            const LD<T> e = op_at<T, UL, DG, Tr>(a, ld, i, j);
            s += e * wide(x[j]), sc += std::abs(e) * std::abs(wide(x[j]));
        }
        worst = std::max(worst, rel(T(0), s - wide(rhs[i]), sc));
    }
    return worst;
}

// trsm_lane with two sub-groups taking the columns round-robin; Tr solves with conj_if(trans(A)).
template <class T, Uplo UL, Diag DG, bool Tr> void trsm_lane_case(Dev& dev, int jb) {
    using D = DevT<T>;
    constexpr int WG = 64, nrhs = 3;
    const int ld = jb + 1, ldy = jb + 2;
    auto ha = tri_blocks<T, UL, DG, Tr>(jb, ld);
    auto hy = batched<T>(ldy * nrhs, [=](int b, int e) { return e % ldy < jb ? val<T>(std::uint64_t(b) * 7u + e + 900u) : nan_v<T>(); });
    T* a = dev.up(ha);
    T* y = dev.up(hy);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        const v2::SubGroup<32> sx{sg};
        const auto am = v2::mat_wide(reinterpret_cast<const D*>(a) + std::size_t(b) * ld * jb, jb, jb, ld);
        const auto ym = v2::mat_wide(reinterpret_cast<D*>(y) + std::size_t(b) * ldy * nrhs, jb, nrhs, ldy);
        const int c0 = static_cast<int>(sg.get_group_linear_id());
        if constexpr (Tr) lu::trsm_lane<UL, DG>(sx, v2::conj_if(v2::op_view<Transpose::Trans>(am), true), ym, c0, 2);
        else lu::trsm_lane<UL, DG>(sx, am, ym, c0, 2);
    });
    const auto h = dev.down(y, hy.size());
    EXPECT_EQ(twin_diffs(h, ldy * nrhs), 0) << "jb " << jb;
    double worst = 0;
    int pad = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int c = 0; c < nrhs; ++c) {
            const std::size_t off = std::size_t(b) * ldy * nrhs + std::size_t(c) * ldy;
            worst = std::max(worst, tri_residual<T, UL, DG, Tr>(ha.data() + std::size_t(b) * ld * jb, ld, jb, h.data() + off, hy.data() + off));
            for (int i = jb; i < ldy; ++i) pad += !std::isnan(std::real(h[off + i]));
        }
    EXPECT_VERIFY(T, Check::solve, jb, worst) << "UL " << int(UL) << " DG " << int(DG) << " Tr " << Tr << " jb " << jb;
    EXPECT_EQ(pad, 0) << "jb " << jb << ": a write past the block";
}

TYPED_TEST(DeviceBlasV2LuTest, TrsmLaneEveryTriangleDiagAndTranspose) {
    using T = TypeParam;
    for (int jb : {1, 5, 16, 32}) {
        trsm_lane_case<T, Uplo::Lower, Diag::Unit, false>(this->dev, jb);
        trsm_lane_case<T, Uplo::Upper, Diag::NonUnit, false>(this->dev, jb);
        trsm_lane_case<T, Uplo::Lower, Diag::NonUnit, true>(this->dev, jb);
        trsm_lane_case<T, Uplo::Upper, Diag::Unit, true>(this->dev, jb);
        trsm_lane_case<T, Uplo::Lower, Diag::NonUnit, false>(this->dev, jb);
        trsm_lane_case<T, Uplo::Upper, Diag::NonUnit, true>(this->dev, jb);
    }
}

// ---- block update (getrs) ------------------------------------------------------------------

// C -= A X (NoTrans) and C2 -= conj(A)^T X2 (Trans of conj_if), NR = 4 accumulators with
// nrhs = 3: the fourth column of each C is poisoned and must stay NaN.
template <class T, int WG> void gemm_nr_case(Dev& dev) {
    using D = DevT<T>;
    constexpr int NR = 4, nrhs = 3, m = 70, k = 37, ldA = m + 4, ldX = m + 1, ldC = m + 5;
    constexpr int kA = ldA * (k + 3), kX = ldX * nrhs, kC = ldC * NR;
    auto ha = batched<T>(kA, [](int b, int e) { return val<T>(std::uint64_t(b) * 31u + e + 7u); });
    auto hx = batched<T>(kX, [](int b, int e) { return val<T>(std::uint64_t(b) * 37u + e + 70000u); });
    auto hc = batched<T>(2 * kC, [](int b, int e) { return (e % kC) >= ldC * nrhs ? nan_v<T>() : val<T>(std::uint64_t(b) * 41u + e + 9000u); });
    T* a = dev.up(ha);
    T* x = dev.up(hx);
    T* c = dev.up(hc);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::RtGroup<> ex(it);
        const auto am = v2::mat_wide(reinterpret_cast<const D*>(a) + std::size_t(b) * kA, ldA, k + 3, ldA);
        const auto xm = v2::mat_wide(reinterpret_cast<const D*>(x) + std::size_t(b) * kX, m, nrhs, ldX);
        D* cb = reinterpret_cast<D*>(c) + std::size_t(b) * 2 * kC;
        // A view (1, 2) of m x k; X rows (3, k) for NoTrans; the full m rows for Trans.
        lu::gemm_nr<Transpose::NoTrans, NR>(ex, v2::NegOne{}, am.sub(1, 2, m, k), xm.sub(3, 0, k, nrhs), v2::One{},
                                            v2::mat_wide(cb, m, nrhs, ldC));
        lu::gemm_nr<Transpose::Trans, NR>(ex, v2::NegOne{}, v2::conj_if(am.sub(1, 2, m, k), true), xm, v2::One{},
                                          v2::mat_wide(cb + kC, k, nrhs, ldC));
    });
    const auto h = dev.down(c, hc.size());
    EXPECT_EQ(twin_diffs(h, 2 * kC), 0) << "WG " << WG;
    double worst = 0;
    int pad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* A = ha.data() + std::size_t(b) * kA;
        const T* X = hx.data() + std::size_t(b) * kX;
        const T* C0 = hc.data() + std::size_t(b) * 2 * kC;
        const T* G = h.data() + std::size_t(b) * 2 * kC;
        auto Av = [&](int i, int j) { return wide(A[(1 + i) + std::size_t(2 + j) * ldA]); };
        for (int r = 0; r < nrhs; ++r) {
            for (int i = 0; i < m; ++i) {
                LD<T> s = wide(C0[i + r * ldC]);
                long double sc = std::abs(s);
                for (int kk = 0; kk < k; ++kk) s -= Av(i, kk) * wide(X[3 + kk + r * ldX]), sc += std::abs(Av(i, kk) * wide(X[3 + kk + r * ldX]));
                worst = std::max(worst, rel(G[i + r * ldC], s, sc));
            }
            for (int t = 0; t < k; ++t) {
                LD<T> s = wide(C0[kC + t + r * ldC]);
                long double sc = std::abs(s);
                for (int i = 0; i < m; ++i) {
                    const LD<T> e = conj_h(Av(i, t)) * wide(X[i + r * ldX]);
                    s -= e, sc += std::abs(e);
                }
                worst = std::max(worst, rel(G[kC + t + r * ldC], s, sc));
            }
        }
        for (int e = ldC * nrhs; e < kC; ++e) pad += !std::isnan(std::real(G[e])) + !std::isnan(std::real(G[kC + e]));
    }
    EXPECT_VERIFY(T, Check::blas, m, worst) << "WG " << WG;
    EXPECT_EQ(pad, 0) << "WG " << WG << ": a column past nrhs was written";
}

TYPED_TEST(DeviceBlasV2LuTest, GemmNrRowOwnerAndSgPerColOnRunTimeWidth) {
    gemm_nr_case<TypeParam, 128>(this->dev);
    gemm_nr_case<TypeParam, 160>(this->dev);
    gemm_nr_case<TypeParam, 1024>(this->dev);
}

// ---- register fragments over a partition (gesv) --------------------------------------------

// Two 16-lane partitions per sub-group, one problem each: ger_frag_bcast from source lane 3 on
// columns [2, kend); inactive lanes, the source lane and columns outside keep their bits.
TYPED_TEST(DeviceBlasV2LuTest, GerFragBcastRowUpdate) {
    using T = TypeParam;
    using D = DevT<T>;
    constexpr int P = 16, NC = 16, kItems = 2 * kBatch, kSrcLane = 3, k0 = 2;
    auto kend_of = [](int it) { return it % 3 == 2 ? 11 : NC - 1; };
    auto act_of = [](int l) { return (l % 3 != 0) && l != kSrcLane; };
    std::vector<T> hin(std::size_t(kItems) * P * NC);
    for (std::size_t e = 0; e < hin.size(); ++e) hin[e] = val<T>((e / (P * NC)) % kPeriod * 100003u + e % (P * NC) + 123456u);
    std::vector<T> hout(hin.size(), nan_v<T>());
    T* in = this->dev.up(hin);
    T* out = this->dev.up(hout);
    this->dev.template run<32>(kItems / 2, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const auto part = make_partition<P>(sg);
        const auto px = v2::wrap(part);
        const int item = static_cast<int>(it.get_group_linear_id()) * 2 + static_cast<int>(part.get_group_linear_id());
        const int lane = static_cast<int>(part.get_local_linear_id());
        const D* L = reinterpret_cast<const D*>(in) + (std::size_t(item) * P + lane) * NC;
        D g[NC];
        for (int k = 0; k < NC; ++k) g[k] = L[k];
        lug::ger_frag_bcast(px, g[0], g, static_cast<std::uint32_t>(kSrcLane), k0, kend_of(item), act_of(lane));
        D* O = reinterpret_cast<D*>(out) + (std::size_t(item) * P + lane) * NC;
        for (int k = 0; k < NC; ++k) O[k] = g[k];
    });
    const auto h = this->dev.down(out, hout.size());
    EXPECT_EQ(twin_diffs(h, P * NC, kItems), 0);
    double worst = 0;
    int bad = 0;
    for (int it = 0; it < kPeriod; ++it) {
        const T* G3 = hin.data() + (std::size_t(it) * P + kSrcLane) * NC;
        for (int l = 0; l < P; ++l) {
            const T* Gl = hin.data() + (std::size_t(it) * P + l) * NC;
            const T* O = h.data() + (std::size_t(it) * P + l) * NC;
            for (int k = 0; k < NC; ++k) {
                if (act_of(l) && k >= k0 && k < kend_of(it)) {
                    const LD<T> ref = wide(Gl[k]) - wide(Gl[0]) * wide(G3[k]);
                    worst = std::max(worst, rel(O[k], ref, std::abs(wide(Gl[k])) + std::abs(wide(Gl[0]) * wide(G3[k]))));
                } else {
                    bad += !same_bits(O[k], Gl[k]);
                }
            }
        }
    }
    EXPECT_EQ(bad, 0) << "unchanged bits (inactive lanes, source lane, columns outside [k0, kend))";
    EXPECT_VERIFY(T, Check::blas, NC, worst);
}

}  // namespace
