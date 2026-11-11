// SPDX-License-Identifier: MIT
// The compositions of bdsdc's Golub-Kahan kernels (BdsdcBuildGk, BdsdcExtract) against host
// references: core interleave / slice / map_many over the gesvd_glue adaptors (item_n, col_of,
// row_of, sub_of; src/ops/gesvd/gesvd-ormbr_v2.hh). Guards (agent guide section 8): non-natural
// ld, inc and batch stride, poisoned slots checked untouched, complex data, a disabled term over
// poison, batch 1024 with period-3 data where every item must be bit-identical to its
// representative, n straddling the WG (1 or 2 owned elements), a WG = 1024 launch, and both
// interleave tails (b one shorter than a, and equal).
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <vector>

#include <batchlas/device.hh>
#include <src/ops/gesvd/gesvd-ormbr_v2.hh>
#include <src/queue.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
using batchlas::verify::Check;

namespace {

template <class T> using real_of = typename device::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = device::Arith<T>::is_complex;

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
template <class T> T nan_v() { return T(std::numeric_limits<real_of<T>>::quiet_NaN()); }
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> bool is_nan(const T& v) { return std::isnan(std::real(v)); }

// kPeriod representatives copied to every item (b from b % kPeriod).
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed) {
    std::vector<T> h(std::size_t(kBatch) * per_item);
    for (int b = 0; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b % kPeriod) * 1000003u + i);
    return h;
}
template <class T> int period_diffs(const std::vector<T>& h, int per_item) {
    int d = 0;
    for (int b = kPeriod; b < kBatch; ++b)
        for (int i = 0; i < per_item; ++i) d += !same_bits(h[std::size_t(b) * per_item + i], h[std::size_t(b % kPeriod) * per_item + i]);
    return d;
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
    template <int WG, class F> void run(int batch, F f) {
        q().submit([&](sycl::handler& h) {
             h.parallel_for(sycl::nd_range<1>(std::size_t(batch) * WG, WG),
                            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { f(it); });
         }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceLapackBdsdcTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "device tests launch sub-group size 32";
        if (test_utils::lacks_fp64<T>(dev.q().get_device())) GTEST_SKIP() << "device has no fp64";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceLapackBdsdcTest, Types);

template <class T> using LD = std::conditional_t<is_cplx<T>, std::complex<long double>, long double>;
template <class T> LD<T> wide(T v) { return LD<T>(v); }
template <class T> double rel(T got, LD<T> ref, long double scale) {
    const long double d = std::abs(LD<T>(got) - ref);
    if (std::isnan(d)) return std::numeric_limits<double>::infinity();
    return static_cast<double>(d / std::max<long double>(scale, std::numeric_limits<real_of<T>>::min()));
}

// ---- interleave from item_n operands into a slice of an item vector -------------------------
// Thread runs on lane 0 (BdsdcBuildGk's executor); WG 64 with n = 101 owns 1 or 2 per item.
template <class T, int WG, bool OnThread> void run_interleave(Dev& dev, int nb) {
    constexpr int n = 101, ainc = 2, astride = 2 * n * ainc + 3, step = 3, ylen = 2 * n * step + 4;
    const int written = n + nb;
    auto ha = batched<T>(astride, 21), hb = batched<T>(astride, 22);
    std::vector<T> hy(std::size_t(kBatch) * ylen, nan_v<T>());
    T* pa = dev.up(ha);
    T* pb = dev.up(hb);
    T* py = dev.up(hy);
    const VectorView<T> av(pa, n, kBatch, ainc, astride), bv(pb, n, kBatch, ainc, astride);
    const VectorView<T> yv(py, ylen, kBatch, 1, ylen);
    dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto y = device::slice(device::item(yv, b), 1, 2 * n, step);
        if constexpr (OnThread) {
            if (it.get_local_linear_id() == 0) device::interleave(device::Thread{}, device::gesvd_glue::item_n(av, b, n), device::gesvd_glue::item_n(bv, b, nb), y);
        } else {
            device::interleave(device::WorkGroup<WG>{it}, device::gesvd_glue::item_n(av, b, n), device::gesvd_glue::item_n(bv, b, nb), y);
        }
    });
    const auto g = dev.down(py, hy.size());
    EXPECT_EQ(period_diffs(g, ylen), 0) << "WG " << WG << " nb " << nb;
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* yb = g.data() + std::size_t(b) * ylen;
        const T* ab = ha.data() + std::size_t(b) * astride;
        const T* bb = hb.data() + std::size_t(b) * astride;
        for (int k = 0; k < ylen; ++k) {
            const int q = k - 1;
            if (q >= 0 && q % step == 0 && q / step < written) {
                const int s = q / step;
                bad += !same_bits(yb[k], (s % 2 == 0) ? ab[(s / 2) * ainc] : bb[(s / 2) * ainc]);
            } else {
                bad += !is_nan(yb[k]);
            }
        }
    }
    EXPECT_EQ(bad, 0) << "WG " << WG << " nb " << nb;
}
TYPED_TEST(DeviceLapackBdsdcTest, InterleaveItemNIntoSlice) {
    for (int nb : {100, 101}) {
        run_interleave<TypeParam, 32, true>(this->dev, nb);
        run_interleave<TypeParam, 64, false>(this->dev, nb);
    }
}

// ---- map_many of scaled terms over slices of a matrix column, into a row and a column -------
// Op 0 writes row 1 of O (stride ldo), op 1 column 2 of P with alpha One, op 2 is disabled and
// its column 0 of P must keep its poison. n = 1500 straddles WG 64 and the WG 1024 ceiling.
template <class T, int WG> void run_dgmv_many(Dev& dev) {
    constexpr int n = 1500, ldz = 2 * n + 3, zstride = 2 * ldz + 5, sinc = 2, sstride = 2 * n * sinc + 7;
    constexpr int ldo = 5, ostride = ldo * n + 7, ldp = n + 2, pstride = 3 * ldp + 9;
    const T alpha = val<T>(5), off_alpha = val<T>(6);
    auto hz = batched<T>(zstride, 31), hs = batched<T>(sstride, 41);
    std::vector<T> ho(std::size_t(kBatch) * ostride, nan_v<T>()), hp(std::size_t(kBatch) * pstride, nan_v<T>());
    T* pz = dev.up(hz);
    T* ps = dev.up(hs);
    T* po = dev.up(ho);
    T* pp = dev.up(hp);
    const KernelMatrixView<T, MatrixFormat::Dense> zm(pz, 2 * n, 2, ldz, zstride, kBatch);
    const KernelMatrixView<T, MatrixFormat::Dense> om(po, 3, n, ldo, ostride, kBatch);
    const KernelMatrixView<T, MatrixFormat::Dense> pm(pp, n, 3, ldp, pstride, kBatch);
    const VectorView<T> sv(ps, 2 * n, kBatch, sinc, sstride);
    dev.template run<WG>(kBatch, [=](sycl::nd_item<1> it) {
        const int b = static_cast<int>(it.get_group_linear_id());
        const device::WorkGroup<WG> wg{it};
        const auto z = device::gesvd_glue::col_of(device::item(zm, b, 2 * n, 2), 1);
        const auto s = device::item(sv, b);
        const auto zv = device::slice(z, 0, n, 2), zu = device::slice(z, 1, n, 2);
        const auto s0 = device::slice(s, 0, n, 2), s1 = device::slice(s, 1, n, 2);
        const auto P = device::item(pm, b, n, 3);
        // BdsdcExtract's terms: y(k) := alpha * (d(k) x(k)) when on, every output in one pass.
        const auto scaled = [](bool on, auto al, const auto& d, const auto& x, const auto& y) {
            return [=](int k) {
                if (on) y(k) = device::tag_mul(device::as_tag(al), device::Arith<T>::mul(static_cast<T>(d(k)), static_cast<T>(x(k))));
            };
        };
        const auto orow = device::gesvd_glue::row_of(device::item(om, b, 3, n), 1);
        device::map_many(wg, orow.size(), scaled(true, alpha, s0, zv, orow), scaled(true, device::One{}, s1, zu, device::gesvd_glue::col_of(P, 2)),
                     scaled(false, off_alpha, s0, zu, device::gesvd_glue::col_of(P, 0)));
    });
    const auto go = dev.down(po, ho.size());
    const auto gp = dev.down(pp, hp.size());
    EXPECT_EQ(period_diffs(go, ostride) + period_diffs(gp, pstride), 0) << "WG " << WG;
    double worst = 0;
    int bad = 0;
    for (int b = 0; b < kPeriod; ++b) {
        const T* zb = hz.data() + std::size_t(b) * zstride + ldz;  // column 1
        const T* sb = hs.data() + std::size_t(b) * sstride;
        const T* ob = go.data() + std::size_t(b) * ostride;
        const T* pb = gp.data() + std::size_t(b) * pstride;
        for (int r = 0; r < n; ++r) {
            const LD<T> zp = wide(zb[2 * r]), zq = wide(zb[2 * r + 1]);
            const LD<T> dp = wide(sb[(2 * r) * sinc]), dq = wide(sb[(2 * r + 1) * sinc]);
            worst = std::max({worst, rel(ob[r * ldo + 1], wide(alpha) * dp * zp, std::abs(wide(alpha) * dp * zp)),
                              rel(pb[2 * ldp + r], dq * zq, std::abs(dq * zq))});
        }
        for (int k = 0; k < ostride; ++k) bad += (k % ldo != 1 || k / ldo >= n) && !is_nan(ob[k]);
        for (int k = 0; k < pstride; ++k) bad += (k / ldp != 2 || k % ldp >= n) && !is_nan(pb[k]);
    }
    EXPECT_EQ(bad, 0) << "WG " << WG << " a slot outside the enabled outputs was written";
    EXPECT_VERIFY(T, Check::blas, 2, worst) << "WG " << WG;
}
TYPED_TEST(DeviceLapackBdsdcTest, DgmvManyLinesAndSlices) {
    run_dgmv_many<TypeParam, 64>(this->dev);
    run_dgmv_many<TypeParam, 1024>(this->dev);
}

// ---- item_n, row_of and col_of address item b like the host views (non-natural ld, inc, stride) -----
TYPED_TEST(DeviceLapackBdsdcTest, ItemNRowColAddressing) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int m = 7, n = 5, ld = m + 3, mstride = ld * n + 11, vn = 9, vk = 6, inc = 2, vstride = vn * inc + 5;
    constexpr int r0 = 4, c0 = 3;
    std::vector<T> hm(std::size_t(kBatch) * mstride, nan_v<T>()), hv(std::size_t(kBatch) * vstride, nan_v<T>());
    T* pm = this->dev.up(hm);
    T* pv = this->dev.up(hv);
    const KernelMatrixView<T, MatrixFormat::Dense> km(pm, m, n, ld, mstride, kBatch);
    const VectorView<T> vv(pv, vn, kBatch, inc, vstride);
    this->dev.template run<32>(kBatch, [=](sycl::nd_item<1> it) {
        if (it.get_local_linear_id() != 0) return;
        const int b = static_cast<int>(it.get_group_linear_id());
        const auto A = device::item(km, b, m, n);
        const auto x = device::gesvd_glue::item_n(vv, b, vk);
        const auto rw = device::gesvd_glue::row_of(device::gesvd_glue::sub_of(A, 0, 0, m, n - 1), r0);
        const auto cl = device::gesvd_glue::col_of(A, c0);
        for (int j = 0; j < rw.size(); ++j) rw(j) = T(R(b * 1000 + 100 + j));
        for (int i = 0; i < cl.size(); ++i) cl(i) = T(R(b * 1000 + 200 + i));
        for (int i = 0; i < x.size(); ++i) x(i) = T(R(-(b * 1000 + i)));
    });
    const auto gm = this->dev.down(pm, hm.size());
    const auto gv = this->dev.down(pv, hv.size());
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        for (int k = 0; k < mstride; ++k) {
            const int j = k / ld, i = k % ld;
            const T got = gm[std::size_t(b) * mstride + k];
            if (j == c0 && i < m) bad += !same_bits(got, T(R(b * 1000 + 200 + i)));
            else if (i == r0 && j < n - 1) bad += !same_bits(got, T(R(b * 1000 + 100 + j)));
            else bad += !is_nan(got);
        }
        for (int k = 0; k < vstride; ++k) {
            const T got = gv[std::size_t(b) * vstride + k];
            if (k % inc == 0 && k / inc < vk) bad += !same_bits(got, T(R(-(b * 1000 + k / inc))));
            else bad += !is_nan(got);
        }
    }
    EXPECT_EQ(bad, 0);
}

}  // namespace
