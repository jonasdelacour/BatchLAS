// SPDX-License-Identifier: MIT
// Device LA v2 LU-solve ops (include/batchlas/blas/device/v2/ext/getrs-getri-gesv.hh) against host
// references. Guards (docs/developer/agent-guide.md section 8): FusedArith bit for bit against
// src/sycl/device_scalar.hh on both sides of Smith's branch, zero, denormal and inf; non-natural ld
// and sub-view offsets everywhere; complex data with nonzero imaginary parts and conjugated views;
// the unread triangle (and a Unit diagonal) poisoned with NaN; ld pads and a column past nrhs
// poisoned and checked untouched; batch 1024 with period-3 data where every item must match its
// representative bit for bit (local-memory paths); run-time widths 64, 160 (not a power of two)
// and 1024 (the ceiling); the packed potrs solve bit-identical to the unpacked one.
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

#include <batchlas/blas/device/v2/ext/getrs-getri-gesv.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <src/kernels/common/sg_partition/sg_partition.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device::v2;
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

// ---- arithmetic ----------------------------------------------------------------------------

enum ArithSlot { kMul, kSub, kDiv, kRecip, kFma, kConj, kOne, kSolve, kSolveZ, kArith };

// FusedArith<Cx> and FusedArith<std::complex> against the sycl_device functions the legacy kernels
// call (and RecipOrDiv against gesv_tiny's open-coded getf2 scaling), bit for bit.
TYPED_TEST(DeviceBlasV2LuTest, FusedArithIsDeviceScalarBitForBit) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    constexpr int n = 96;
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
    std::vector<T> ho(std::size_t(3) * n * kArith, nan_v<T>());
    std::vector<int> hf(std::size_t(4) * n, -1);
    T* o = this->dev.up(ho);
    int* f = this->dev.up(hf);
    this->dev.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        using FA = v2::FusedArith<D>;
        using FS = v2::FusedArith<T>;
        const D* ad = reinterpret_cast<const D*>(a);
        const D* bd = reinterpret_cast<const D*>(bb);
        D* got = reinterpret_cast<D*>(o);
        D* ref = got + n * kArith;
        T* std_got = o + 2 * n * kArith;
        for (int i = static_cast<int>(it.get_local_linear_id()); i < n; i += 32) {
            const D x = ad[i], y = bd[i], c = ad[(i + 1) % n];
            D t = c;
            FA::fma_acc(t, x, y);
            const auto sr = v2::RecipOrDiv<D>::make(y);
            D* g = got + i * kArith;
            g[kMul] = FA::mul(x, y), g[kSub] = FA::sub(x, y), g[kDiv] = FA::div(x, y), g[kRecip] = FA::recip(y);
            g[kFma] = t, g[kConj] = FA::conj(x), g[kOne] = FA::one();
            g[kSolve] = sr.solve(x, x), g[kSolveZ] = sr.solve(x, FA::zero());
            // The legacy spellings (gesv_tiny steps 4 and 6 for the two solves).
            D u = c;
            sd::fma_acc(u, x, y);
            const bool zero = sd::dev_is_zero(y);
            const D rc = sd::dev_recip(y);
            const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
            D s1 = x, s2 = D{};
            if (use_mul) s1 = sd::dev_mul(x, rc), s2 = sd::dev_mul(x, rc);
            else if (!zero) s1 = sd::dev_div(x, y), s2 = sd::dev_div(x, y);
            D* r = ref + i * kArith;
            r[kMul] = sd::dev_mul(x, y), r[kSub] = sd::dev_sub(x, y), r[kDiv] = sd::dev_div(x, y), r[kRecip] = rc;
            r[kFma] = u, r[kConj] = sd::dev_conj(x), r[kOne] = sd::dev_one<D>(), r[kSolve] = s1, r[kSolveZ] = s2;
            f[4 * i + 0] = FA::is_zero(y), f[4 * i + 1] = sd::dev_is_zero(y);
            f[4 * i + 2] = FA::isfinite(y), f[4 * i + 3] = sd::dev_isfinite(y);
            // std::complex through the same trait: the same bits as the POD.
            const T xs = a[i], ys = bb[i];
            T ts = a[(i + 1) % n];
            FS::fma_acc(ts, xs, ys);
            T* s = std_got + i * kArith;
            s[kMul] = FS::mul(xs, ys), s[kSub] = FS::sub(xs, ys), s[kDiv] = FS::div(xs, ys), s[kRecip] = FS::recip(ys);
            s[kFma] = ts, s[kConj] = FS::conj(xs), s[kOne] = FS::one();
            s[kSolve] = v2::RecipOrDiv<T>::make(ys).solve(xs, xs), s[kSolveZ] = v2::RecipOrDiv<T>::make(ys).solve(xs, FS::zero());
        }
    });
    const auto h = this->dev.down(o, ho.size());
    const auto fl = this->dev.down(f, hf.size());
    int diff = 0, sdiff = 0, fdiff = 0;
    for (int i = 0; i < n * kArith; ++i) {
        diff += !same_bits(h[i], h[n * kArith + i]);
        sdiff += !same_bits(h[2 * n * kArith + i], h[n * kArith + i]);
    }
    for (int i = 0; i < n; ++i) fdiff += (fl[4 * i] != fl[4 * i + 1]) + (fl[4 * i + 2] != fl[4 * i + 3]);
    EXPECT_EQ(diff, 0) << "FusedArith<DevMap<T>> differs from src/sycl/device_scalar.hh";
    EXPECT_EQ(sdiff, 0) << "FusedArith<T> differs from the POD scalar";
    EXPECT_EQ(fdiff, 0) << "is_zero / isfinite";
    EXPECT_TRUE(fl[0] == 1 && fl[4 * 7] == 1) << "0 and -0 are exact zeros";
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

// ---- elementwise on a run-time width -------------------------------------------------------

// Per item: src (ldA x NA) with the view at (i0, j0), m x n. out regions (each poisoned):
// [packed round trip | Walk copy | lapmr_copy | conj(trans) | core copy of a column], ld ldB.
template <class T, int WG> void elementwise_case(Dev& dev) {
    using D = DevT<T>;
    constexpr int m = 37, n = 5, i0 = 2, j0 = 1, ldA = m + 6, NA = n + 2, ldB = m + 5, ldT = n + 2;
    constexpr int kSrc = ldA * NA, kReg = ldB * n, kOut = 3 * kReg + ldT * m + ldB;
    auto perm = [](int i) { return (i * 7 + 3) % m; };
    auto hs = batched<T>(kSrc, [](int b, int i) { return val<T>(std::uint64_t(b) * 7919u + i); });
    std::vector<T> ho(std::size_t(kBatch) * kOut, nan_v<T>());
    T* s = dev.up(hs);
    T* o = dev.up(ho);
    const int width = WG;  // a run-time value inside the kernel
    dev.template run<WG>(kBatch, sizeof(T) * m * n, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroupRt<32> ex{it, static_cast<int>(it.get_local_id(0)), width, width / 32};
        const v2::MatZ<const D> src = v2::MatZ<const D>{reinterpret_cast<const D*>(s) + std::size_t(b) * kSrc, m + i0, NA, ldA}.sub(i0, j0, m, n);
        D* ob = reinterpret_cast<D*>(o) + std::size_t(b) * kOut;
        const v2::MatZ<D, true> y{reinterpret_cast<D*>(lm), m, n, m};
        v2::lacpy_flat(ex, src, y);
        ex.barrier();
        v2::lacpy_flat(ex, y, v2::MatZ<D>{ob, m, n, ldB});
        v2::lacpy_flat<v2::alg::Walk>(ex, src, v2::MatZ<D>{ob + kReg, m, n, ldB});
        v2::lapmr_copy(ex, src, perm, v2::MatZ<D>{ob + 2 * kReg, m, n, ldB});
        v2::lacpy_flat(ex, v2::conj_if(v2::trans(src), true), v2::MatZ<D>{ob + 3 * kReg, n, m, ldT});
        v2::copy(ex, v2::vec(src.col_ptr(2), m), v2::vec(ob + 3 * kReg + ldT * m, m));
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
                const bool in = i < m;
                for (int r = 0; r < 3; ++r) {
                    const T v = g[r * kReg + i + j * ldB];
                    if (!in) { pad += !std::isnan(std::real(v)); continue; }
                    bad += !same_bits(v, r == 2 ? A(perm(i), j) : A(i, j));
                }
            }
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < ldT; ++j) {
                const T v = g[3 * kReg + j + i * ldT];
                if (j >= n) pad += !std::isnan(std::real(v));
                else bad += !same_bits(v, conj_h(A(i, j)));
            }
        for (int i = 0; i < ldB; ++i) {
            const T v = g[3 * kReg + ldT * m + i];
            if (i >= m) pad += !std::isnan(std::real(v));
            else bad += !same_bits(v, A(i, 2));
        }
    }
    EXPECT_EQ(bad, 0) << "WG " << WG << ": lacpy_flat / lapmr_copy / conj(trans) / copy values";
    EXPECT_EQ(pad, 0) << "WG " << WG << ": a write outside the view";
}

TYPED_TEST(DeviceBlasV2LuTest, LacpyWalkLapmrOnRunTimeWidth) {
    elementwise_case<TypeParam, 64>(this->dev);
    elementwise_case<TypeParam, 160>(this->dev);
    elementwise_case<TypeParam, 1024>(this->dev);
}

// ---- interchanges --------------------------------------------------------------------------

// 0-based ?getrf-shaped pivots (p in [k, n)), about a third of them p == k.
std::vector<int> pivots(int n, int b) {
    std::vector<int> p(n);
    for (int k = 0; k < n; ++k) p[k] = (mix(std::uint64_t(b) * 131u + k) % 3 == 0) ? k : k + int(mix(77u + k + 1000u * b) % (n - k));
    return p;
}
template <class V> void host_laswp(V& x, const std::vector<int>& p, int k0, int k1, bool forward) {
    auto step = [&](int k) {
        if (p[k] != k) std::swap(x[k], x[p[k]]);
    };
    if (forward)
        for (int k = k0; k < k1; ++k) step(k);
    else
        for (int k = k1 - 1; k >= k0; --k) step(k);
}

// laswp_cols both orders on a packed local Y, laswp on a sub-range by one item, perm_index for
// every i against the inverse of laswp applied to the identity (not the same walk).
TYPED_TEST(DeviceBlasV2LuTest, LaswpLaswpColsPermIndex) {
    using T = TypeParam;
    using D = DevT<T>;
    constexpr int WG = 64, n = 45, nrhs = 3, ldx = n + 3, k0 = 3, k1 = n - 4;
    constexpr int kOutT = 3 * n * nrhs + n, kOutI = 2 * n;
    auto hx = batched<T>(ldx * nrhs, [](int b, int i) { return val<T>(std::uint64_t(b) * 104729u + i); });
    std::vector<int> hp(std::size_t(kBatch) * n);
    for (int b = 0; b < kBatch; ++b) {
        const auto p = pivots(n, b % kPeriod);
        for (int k = 0; k < n; ++k) hp[std::size_t(b) * n + k] = p[k] + 1;  // 1-based on the wire
    }
    std::vector<T> hot(std::size_t(kBatch) * kOutT, nan_v<T>());
    std::vector<int> hoi(std::size_t(kBatch) * kOutI, -7);
    T* x = this->dev.up(hx);
    int* pv = this->dev.up(hp);
    T* ot = this->dev.up(hot);
    int* oi = this->dev.up(hoi);
    const int width = WG;
    this->dev.template run<WG>(kBatch, sizeof(T) * n * nrhs, [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroupRt<32> ex{it, static_cast<int>(it.get_local_id(0)), width, 2};
        const v2::MatZ<const D> xm{reinterpret_cast<const D*>(x) + std::size_t(b) * ldx * nrhs, n, nrhs, ldx};
        const v2::MatZ<D, true> y{reinterpret_cast<D*>(lm), n, nrhs, n};
        const v2::Ipiv1 ipiv{pv + std::size_t(b) * n};
        D* ob = reinterpret_cast<D*>(ot) + std::size_t(b) * kOutT;
        for (int pass = 0; pass < 2; ++pass) {
            v2::lacpy_flat(ex, xm, y);
            ex.barrier();
            if (pass == 0) v2::laswp_cols<ApplyOrder::Forward>(ex, y, ipiv);
            else v2::laswp_cols<ApplyOrder::Backward>(ex, y, ipiv);
            ex.barrier();
            v2::lacpy_flat(ex, y, v2::MatZ<D, true>{ob + pass * n * nrhs, n, nrhs, n});
            ex.barrier();
        }
        v2::lacpy_flat(ex, xm, y);
        ex.barrier();
        if (ex.leader()) v2::laswp<ApplyOrder::Forward>(v2::Thread{}, v2::vec<v2::Mem::Local>(y.col_ptr(1), n), ipiv, k0, k1);
        ex.barrier();
        v2::lacpy_flat(ex, y, v2::MatZ<D, true>{ob + 2 * n * nrhs, n, nrhs, n});
        v2::for_each_owned(ex, n, [&](int i) {
            oi[std::size_t(b) * kOutI + i] = v2::perm_index<ApplyOrder::Forward>(ipiv, 0, n, i);
            oi[std::size_t(b) * kOutI + n + i] = v2::perm_index<ApplyOrder::Backward>(ipiv, 0, n, i);
        });
    });
    const auto ht = this->dev.down(ot, hot.size());
    const auto hi = this->dev.down(oi, hoi.size());
    EXPECT_EQ(twin_diffs(ht, kOutT), 0);
    EXPECT_EQ(twin_diffs(hi, kOutI), 0);
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const auto p = pivots(n, b);
        const T* g = ht.data() + std::size_t(b) * kOutT;
        const int* gi = hi.data() + std::size_t(b) * kOutI;
        for (int c = 0; c < nrhs; ++c) {
            std::vector<T> col(hx.begin() + std::size_t(b) * ldx * nrhs + c * ldx, hx.begin() + std::size_t(b) * ldx * nrhs + c * ldx + n);
            auto f = col, bk = col, sub = col;
            host_laswp(f, p, 0, n, true);
            host_laswp(bk, p, 0, n, false);
            if (c == 1) host_laswp(sub, p, k0, k1, true);
            auto rt = f;
            host_laswp(rt, p, 0, n, false);  // the inverse recovers the input
            for (int i = 0; i < n; ++i) {
                bad += !same_bits(g[c * n + i], f[i]) + !same_bits(g[n * nrhs + c * n + i], bk[i]);
                bad += !same_bits(g[2 * n * nrhs + c * n + i], sub[i]) + !same_bits(rt[i], col[i]);
            }
        }
        for (int fwd = 0; fwd < 2; ++fwd) {
            std::vector<int> id(n);
            for (int i = 0; i < n; ++i) id[i] = i;
            host_laswp(id, p, 0, n, fwd == 0);
            for (int pos = 0; pos < n; ++pos) bad += gi[fwd * n + id[pos]] != pos;  // where i landed
        }
    }
    EXPECT_EQ(bad, 0);
}

// ---- sub-group triangular solves -----------------------------------------------------------

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
        const v2::MatZ<const D> am{reinterpret_cast<const D*>(a) + std::size_t(b) * ld * jb, jb, jb, ld};
        const v2::MatZ<D> ym{reinterpret_cast<D*>(y) + std::size_t(b) * ldy * nrhs, jb, nrhs, ldy};
        const int c0 = static_cast<int>(sg.get_group_linear_id());
        if constexpr (Tr) v2::trsm_lane<UL, DG>(sx, v2::conj_if(v2::trans(am), true), ym, c0, 2);
        else v2::trsm_lane<UL, DG>(sx, am, ym, c0, 2);
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

// The packed potrs solve (several right-hand sides per sub-group) against trsm_lane, bit for bit:
// each column's arithmetic is the same, only the lanes holding it differ.
template <class T, Uplo UL> void slots_case(Dev& dev, int jb) {
    using D = DevT<T>;
    constexpr int WG = 64, nrhs = 10, cs = 2;
    const int ld = jb + 1, ldy = jb + 3;
    const int w = jb <= 4 ? 4 : jb <= 8 ? 8 : jb <= 16 ? 16 : 32;
    const int P = std::min((nrhs + cs - 1) / cs, 32 / w);
    auto ha = tri_blocks<T, Uplo::Lower, Diag::NonUnit, false>(jb, ld);
    auto hy = batched<T>(ldy * nrhs, [=](int b, int e) { return e % ldy < jb ? val<T>(std::uint64_t(b) * 5u + e + 4000u) : nan_v<T>(); });
    std::vector<T> hy2 = hy;
    T* a = dev.up(ha);
    T* y1 = dev.up(hy);
    T* y2 = dev.up(hy2);
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto sg = it.get_sub_group();
        const v2::SubGroup<32> sx{sg};
        const int lane = static_cast<int>(sg.get_local_linear_id()), c0 = static_cast<int>(sg.get_group_linear_id());
        const v2::MatZ<const D> am{reinterpret_cast<const D*>(a) + std::size_t(b) * ld * jb, jb, jb, ld};
        const v2::MatZ<D> m1{reinterpret_cast<D*>(y1) + std::size_t(b) * ldy * nrhs, jb, nrhs, ldy};
        const v2::MatZ<D> m2{reinterpret_cast<D*>(y2) + std::size_t(b) * ldy * nrhs, jb, nrhs, ldy};
        const v2::LaneSlots sl{w, P, lane % w, lane / w};
        if constexpr (UL == Uplo::Lower) {
            v2::trsm_lane_slots<Uplo::Lower>(sx, am, m1, c0, cs, sl);
            v2::trsm_lane<Uplo::Lower, Diag::NonUnit>(sx, am, m2, c0, cs);
        } else {
            // The backward potrs form: the upper operator is the stored lower block transposed.
            v2::trsm_lane_slots<Uplo::Upper>(sx, v2::trans(am), m1, c0, cs, sl);
            v2::trsm_lane<Uplo::Upper, Diag::NonUnit>(sx, v2::trans(am), m2, c0, cs);
        }
    });
    const auto h1 = dev.down(y1, hy.size());
    const auto h2 = dev.down(y2, hy.size());
    int diff = 0;
    for (std::size_t i = 0; i < h1.size(); ++i) diff += !same_bits(h1[i], h2[i]);
    EXPECT_EQ(diff, 0) << "UL " << int(UL) << " jb " << jb << " w " << w << " P " << P;
    double worst = 0;
    for (int c = 0; c < nrhs; ++c) {
        const std::size_t off = std::size_t(c) * ldy;
        if constexpr (UL == Uplo::Lower)
            worst = std::max(worst, tri_residual<T, Uplo::Lower, Diag::NonUnit, false>(ha.data(), ld, jb, h1.data() + off, hy.data() + off));
    }
    EXPECT_VERIFY(T, Check::solve, jb, worst) << "jb " << jb;
}

TYPED_TEST(DeviceBlasV2LuTest, TrsmLaneSlotsMatchesUnpackedBitForBit) {
    for (int jb : {3, 7, 16, 32}) {
        slots_case<TypeParam, Uplo::Lower>(this->dev, jb);
        slots_case<TypeParam, Uplo::Upper>(this->dev, jb);
    }
}

// ---- block update --------------------------------------------------------------------------

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
    const int width = WG;
    dev.template run<WG>(kBatch, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroupRt<32> ex{it, static_cast<int>(it.get_local_id(0)), width, width / 32};
        const auto am = v2::MatZ<const D>{reinterpret_cast<const D*>(a) + std::size_t(b) * kA, ldA, k + 3, ldA};
        const auto xm = v2::MatZ<const D>{reinterpret_cast<const D*>(x) + std::size_t(b) * kX, m, nrhs, ldX};
        D* cb = reinterpret_cast<D*>(c) + std::size_t(b) * 2 * kC;
        // A view (1, 2) of m x k; X rows (3, k) for NoTrans; the full m rows for Trans.
        v2::gemm_nr<Transpose::NoTrans, NR>(ex, v2::NegOne{}, am.sub(1, 2, m, k), xm.sub(3, 0, k, nrhs), v2::One{},
                                            v2::MatZ<D>{cb, m, nrhs, ldC});
        v2::gemm_nr<Transpose::Trans, NR>(ex, v2::NegOne{}, v2::conj_if(am.sub(1, 2, m, k), true), xm, v2::One{},
                                          v2::MatZ<D>{cb + kC, k, nrhs, ldC});
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

// ---- info: the first exact zero on a diagonal ----------------------------------------------

// Pattern b % 4: no zero (a complex item carries (0, 1), which is not zero), zero at 0, zeros at
// 77 and 200, zero at n - 1. n > WG, so every item strides twice.
TYPED_TEST(DeviceBlasV2LuTest, LocalMinSlotFirstExactZero) {
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
    const int width = WG;
    this->dev.template run<WG>(kBatch, sizeof(int), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const v2::WorkGroupRt<32> ex{it, static_cast<int>(it.get_local_linear_id()), width, width / 32};
        const v2::LocalMinSlot slot{reinterpret_cast<int*>(lm)};
        // ld 0 makes element (i, i) dg[i]: the diagonal alone is stored.
        const v2::MatZ<const D> a{reinterpret_cast<const D*>(dg) + std::size_t(b) * n, n, n, 0};
        slot.reset(ex);
        ex.group_barrier();
        v2::for_each_owned(ex, n, [&](int i) {
            if (v2::FusedArith<D>::is_zero(a(i, i))) slot.offer(i + 1);
        });
        ex.group_barrier();
        if (ex.leader()) info[b] = slot.info();
    });
    const auto h = this->dev.down(info, hi.size());
    const int want[kPat] = {0, 1, 78, n};
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) bad += h[b] != want[b % kPat];
    EXPECT_EQ(bad, 0) << "info: item 0..3 = " << h[0] << " " << h[1] << " " << h[2] << " " << h[3];
}

// ---- register fragments over a partition (the tiny tier) -----------------------------------

// Two 16-lane partitions per sub-group, one problem each. ger_frag_bcast on a random fragment,
// then U X = Y on rows relabelled by a permutation (lane l holds row rowid[l]); U's strict lower
// part is NaN (never read into a kept value); one item has an exact zero pivot (x_i := 0), one a
// denormal pivot (the divide arm); nrhs 1 leaves the pad column's x at zero.
TYPED_TEST(DeviceBlasV2LuTest, FragGerAndRelabelledUpperSolve) {
    using T = TypeParam;
    using D = DevT<T>;
    using R = real_of<T>;
    constexpr int P = 16, NC = 16, NR = 2, kItems = 2 * kBatch, kStride = NC + 2 * NR + NC + 2;
    // per lane: a[NC] | b[NR] | ger[NC] | rowid | row_lane  (inputs);  out: x[NR] | ger[NC]
    auto shape = [](int it) { return it % 6; };  // 0,1: n 16; 2: n 11; 3: zero pivot; 4: denormal pivot; 5: nrhs 1
    auto n_of = [&](int it) { return shape(it) == 2 ? 11 : P; };
    auto nrhs_of = [&](int it) { return shape(it) == 5 ? 1 : 2; };
    auto rowid_of = [&](int it, int l) { return l < n_of(it) ? int((l * 5 + it % 7) % n_of(it)) : l; };  // gcd(5, n) = 1
    // The denormal row has no off-diagonal and y = d (3 + k), so x = 3 + k through the divide arm.
    const T dn4 = T(std::numeric_limits<R>::denorm_min() * R(4));
    auto U = [&](int it, int r, int c) -> T {
        if (c < r) return nan_v<T>();
        if (c >= n_of(it)) return T(0);
        if (r == c) {
            if (shape(it) == 3 && r == 6) return T(0);
            if (shape(it) == 4 && r == 9) return dn4;
            return T(R(1.25)) + T(R(0.5)) * val<T>(std::uint64_t(it) * 997u + r);
        }
        if (shape(it) == 4 && r == 9) return T(0);
        return val<T>(std::uint64_t(it) * 1009u + r * NC + c) / T(R(4));
    };
    auto Yv = [&](int it, int r, int k) {
        if (r >= n_of(it) || k >= nrhs_of(it)) return nan_v<T>();
        if (shape(it) == 4 && r == 9) return dn4 * T(R(3 + k));
        return val<T>(std::uint64_t(it) * 3u + r * NR + k + 50000u);
    };
    const bool ftz = device_flushes<T>(this->dev);
    std::vector<T> hin(std::size_t(kItems) * P * kStride, T(0));
    for (int it = 0; it < kItems; ++it)
        for (int l = 0; l < P; ++l) {
            T* L = hin.data() + (std::size_t(it) * P + l) * kStride;
            const int row = rowid_of(it, l);
            for (int k = 0; k < NC; ++k) L[k] = U(it, row, k);
            for (int k = 0; k < NR; ++k) L[NC + k] = Yv(it, row, k);
            for (int k = 0; k < NC; ++k) L[NC + NR + k] = val<T>(std::uint64_t(it) * 17u + l * NC + k + 123456u);
            L[NC + NR + NC] = T(R(row));
            int holder = row;
            for (int q = 0; q < P; ++q)
                if (rowid_of(it, q) == l) holder = q;
            L[NC + NR + NC + 1] = T(R(holder));  // row_lane at lane l: the lane holding row l
        }
    constexpr int kOut = NR + NC;
    std::vector<T> hout(std::size_t(kItems) * P * kOut, nan_v<T>());
    T* in = this->dev.up(hin);
    T* out = this->dev.up(hout);
    this->dev.template run<32>(kItems / 2, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto sg = it.get_sub_group();
        const auto part = make_partition<P>(sg);
        const int item = static_cast<int>(it.get_group_linear_id()) * 2 + static_cast<int>(part.get_group_linear_id());
        const int lane = static_cast<int>(part.get_local_linear_id());
        const int sh = item % 6, n = sh == 2 ? 11 : P, nrhs = sh == 5 ? 1 : 2;
        const D* L = reinterpret_cast<const D*>(in) + (std::size_t(item) * P + lane) * kStride;
        D a[NC], g[NC], bv[NR], x[NR];
        for (int k = 0; k < NC; ++k) a[k] = L[k], g[k] = L[NC + NR + k];
        for (int k = 0; k < NR; ++k) bv[k] = L[NC + k], x[k] = v2::FusedArith<D>::zero();
        const int rowid = static_cast<int>(v2::FusedArith<D>::re(L[NC + NR + NC]));
        const int row_lane = static_cast<int>(v2::FusedArith<D>::re(L[NC + NR + NC + 1]));
        const bool act = (lane % 3 != 0) && lane != 3;
        v2::ger_frag_bcast(part, g[0], g, 3u, 2, n - 1, act);
        v2::trsm_frag_upper_relabel(part, a, bv, x, rowid, row_lane, n, nrhs);
        D* O = reinterpret_cast<D*>(out) + (std::size_t(item) * P + lane) * kOut;
        for (int k = 0; k < NR; ++k) O[k] = x[k];
        for (int k = 0; k < NC; ++k) O[NR + k] = g[k];
    });
    const auto h = this->dev.down(out, hout.size());
    double worst = 0;
    int bad = 0;
    for (int it = 0; it < kItems; ++it) {
        if (ftz && shape(it) == 4) continue;  // the denormal pivot is a zero there
        const int n = n_of(it), nrhs = nrhs_of(it);
        auto lane_of_row = [&](int r) {
            for (int q = 0; q < P; ++q)
                if (rowid_of(it, q) == r) return q;
            return -1;
        };
        // ger: lanes with act get g[k] -= g[0] * g@3[k] for 2 <= k < n - 1; the rest keep their bits.
        const T* G3 = hin.data() + (std::size_t(it) * P + 3) * kStride + NC + NR;
        for (int l = 0; l < P; ++l) {
            const T* Gl = hin.data() + (std::size_t(it) * P + l) * kStride + NC + NR;
            const T* O = h.data() + (std::size_t(it) * P + l) * kOut + NR;
            const bool act = (l % 3 != 0) && l != 3;
            for (int k = 0; k < NC; ++k) {
                if (act && k >= 2 && k < n - 1) {
                    const LD<T> ref = wide(Gl[k]) - wide(Gl[0]) * wide(G3[k]);
                    worst = std::max(worst, rel(O[k], ref, std::abs(wide(Gl[k])) + std::abs(wide(Gl[0]) * wide(G3[k]))));
                } else {
                    bad += !same_bits(O[k], Gl[k]);
                }
            }
        }
        // back substitution in long double with the singular rule x_i := 0 when U_ii == 0.
        for (int k = 0; k < NR; ++k) {
            std::vector<LD<T>> xs(n, LD<T>(0));
            for (int i = n - 1; i >= 0; --i) {
                LD<T> s = wide(Yv(it, i, k));
                for (int j = i + 1; j < n; ++j) s -= wide(U(it, i, j)) * xs[j];
                const T d = U(it, i, i);
                xs[i] = (d == T(0)) ? LD<T>(0) : s / wide(d);
            }
            for (int i = 0; i < n; ++i) {
                const T got = h[(std::size_t(it) * P + lane_of_row(i)) * kOut + k];
                if (k >= nrhs) {
                    bad += !same_bits(got, T(0));  // the pad column is never solved
                    continue;
                }
                long double sc = std::abs(xs[i]);
                for (int j = i + 1; j < n; ++j) sc += std::abs(wide(U(it, i, j)) * xs[j]) / std::max<long double>(std::abs(wide(U(it, i, i))), 1e-300L);
                if (U(it, i, i) == T(0)) bad += !same_bits(got, T(0));
                else worst = std::max(worst, rel(got, xs[i], sc));
            }
        }
    }
    EXPECT_EQ(bad, 0) << "unchanged bits (inactive lanes, pad column, singular row)";
    EXPECT_VERIFY(T, Check::solve, P, worst);
}

static_assert(v2::is_rt_group_v<v2::WorkGroupRt<32>> && !v2::is_rt_group_v<v2::WorkGroup<64>>);
static_assert(v2::FusedArith<sd::Cx<float>>::is_complex && !v2::FusedArith<double>::is_complex);
static_assert(std::is_same_v<v2::FusedArith<std::complex<double>>::real_t, double>);

}  // namespace
