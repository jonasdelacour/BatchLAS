// SPDX-License-Identifier: MIT
// Native gemv's private device LA v2 ops (src/ops/gemv/gemv_native_v2.hh: store_fma / RtScalar,
// conj_if, Member, fold_down, leader_sum) in the five body shapes they serve, against host
// references; the core ops they compose with are covered by device_blas_v2_tests. Every result is
// checked twice: bit for bit against a host emulation of the documented rounding (std::fma chain,
// shift-down tree) and with EXPECT_VERIFY against a long-double reference, so a rounding slip and
// a wrong formula fail separately. Guards (agent-guide section 8): ld = rows + 5, batch strides
// past the natural one, x inc 3, y inc 2 with its gaps checked untouched, complex data, ConjTrans
// through a launch-time flag, beta == 0 over a NaN y, alpha == 0 over a NaN in A, 256-item
// work-groups of 8 sub-groups, 1029 items (>= 1024, and the partitioned shape has a tail
// sub-group), period-3 items bit-identical.
#include <gtest/gtest.h>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>
#include <vector>

#include <batchlas/device.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/ops/gemv/gemv_native_v2.hh>
#include <src/queue.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

using namespace batchlas;
namespace v2 = batchlas::device;
namespace dla = batchlas::sycl_gemv::dla;
using batchlas::verify::Check;
using Fma = v2::arith::Fma;

namespace {

template <class T> using real_of = typename v2::Arith<T>::real_t;
template <class T> constexpr bool is_cplx = v2::Arith<T>::is_complex;
template <class T> using Dv = typename sycl_device::DevMap<T>::type;

constexpr int kItems = 1029;  // 11 * 1029 is not a multiple of 4: the W = 4 partition shape has a tail
constexpr int kPeriod = 3;
constexpr int kWG = 256, kSg = 32, kSgPerWg = kWG / kSg;

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
// kPeriod representatives, item b a copy of representative b % kPeriod.
template <class T> std::vector<T> batched(int per_item, std::uint64_t seed, int items = kItems) {
    std::vector<T> h(static_cast<std::size_t>(items) * per_item);
    for (int b = 0; b < std::min(kPeriod, items); ++b)
        for (int i = 0; i < per_item; ++i) h[std::size_t(b) * per_item + i] = val<T>(seed + std::uint64_t(b) * 1000003u + i);
    for (int b = kPeriod; b < items; ++b)
        std::copy_n(h.data() + std::size_t(b % kPeriod) * per_item, per_item, h.data() + std::size_t(b) * per_item);
    return h;
}
template <class T> bool same_bits(const T& a, const T& b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template <class T> Dv<T> to_dev(T v) {
    Dv<T> d;
    std::memcpy(&d, &v, sizeof(T));
    return d;
}

// No queue when no SYCL device is visible (the fixture skips).
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
    // kWG-item work-groups; f(it, unit) with unit = the work-item (PerSg false) or the sub-group
    // (PerSg true) index of the flattened launch. f owns its bounds test.
    template <bool PerSg, class F> void run(std::int64_t units, F f) {
        const std::int64_t per_wg = PerSg ? kSgPerWg : kWG;
        const std::int64_t groups = (units + per_wg - 1) / per_wg;
        q().parallel_for(sycl::nd_range<1>(std::size_t(groups) * kWG, kWG), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
               const std::int64_t u = PerSg ? std::int64_t(it.get_group_linear_id()) * kSgPerWg + it.get_sub_group().get_group_linear_id()
                                            : std::int64_t(it.get_global_linear_id());
               f(it, u);
           }).wait_and_throw();
    }
    ~Dev() {
        for (void* p : ptrs) sycl::free(p, q());
    }
    std::vector<void*> ptrs;
};

template <class T> class DeviceBlasV2GemvNativeTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!dev.ctx) GTEST_SKIP() << "no SYCL device visible";
        if (!dev.ctx->device().supports_sub_group_size(32)) GTEST_SKIP() << "the gemv ops shuffle a 32-lane sub-group";
    }
    Dev dev;
};

using Types = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(DeviceBlasV2GemvNativeTest, Types);

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

// The documented rounding, emulated on the host.
template <class T> void host_mac(T& acc, T a, T b) {
    if constexpr (is_cplx<T>) {
        real_of<T> re = acc.real(), im = acc.imag();
        re = std::fma(a.real(), b.real(), re);
        re = std::fma(-a.imag(), b.imag(), re);
        im = std::fma(a.real(), b.imag(), im);
        im = std::fma(a.imag(), b.real(), im);
        acc = T(re, im);
    } else {
        acc = std::fma(a, b, acc);
    }
}
// Shift-down fold of 32 lane values: lane l += lane l + stride*w for w = Count/2 .. 1 while the
// source lane exists (beyond it the device value is unspecified and no head lane reads it).
template <class T> std::vector<T> host_fold(std::vector<T> v, int count, int stride) {
    for (int w = count / 2; w >= 1; w >>= 1) {
        std::vector<T> nv = v;
        for (int l = 0; l + stride * w < kSg; ++l) {
            if constexpr (is_cplx<T>) nv[l] = T(v[l].real() + v[l + stride * w].real(), v[l].imag() + v[l + stride * w].imag());
            else nv[l] = v[l] + v[l + stride * w];
        }
        v = nv;
    }
    return v;
}

// ---- the five gemv body shapes, composed from core and the private ops ---------------------------

enum class Shape { RowThread, RowSegment, ColThread, ColSubGroup, ColPartition };

struct Dims {
    int m, n, members;  // members: the lanes sharing one output's reduction
};
constexpr Dims dims_of(Shape s) {
    switch (s) {
        case Shape::RowThread:    return {13, 29, 1};
        case Shape::RowSegment:   return {5, 29, 4};    // gemv_seg_width(5) = 4
        case Shape::ColThread:    return {29, 13, 1};
        case Shape::ColSubGroup:  return {45, 11, 32};  // red_len > 32: lanes wrap
        case Shape::ColPartition: return {29, 11, 8};   // W = 4 teams of L = 8
    }
    return {0, 0, 0};
}

// y := alpha op(A) x + beta y in shape S, with A's NaN poison (alpha == 0) or y's (beta == 0).
template <class T, Shape S> void body_case(Dev& dev, Transpose tr, bool alpha_zero, bool beta_zero, const char* where) {
    using D = Dv<T>;
    constexpr Dims dm = dims_of(S);
    constexpr int m = dm.m, n = dm.n, P = dm.members, ld = m + 5, xinc = 3, yinc = 2;
    constexpr bool notrans = S == Shape::RowThread || S == Shape::RowSegment;
    const int out = notrans ? m : n, red = notrans ? n : m;
    const int sa = ld * n + 7, sx = red * xinc + 2, sy = out * yinc + 3;
    const bool conj = tr == Transpose::ConjTrans;
    const T alpha = alpha_zero ? T(0) : val<T>(5), beta = beta_zero ? T(0) : val<T>(6);
    auto ha = batched<T>(sa, 31), hx = batched<T>(sx, 32), hy = batched<T>(sy, 33);
    const T nan = T(std::numeric_limits<real_of<T>>::quiet_NaN());
    if (alpha_zero)
        for (int b = 0; b < kItems; ++b) ha[std::size_t(b) * sa + 1 + std::size_t(2) * ld] = nan;  // read only when the dot runs
    if (beta_zero)
        for (int b = 0; b < kItems; ++b)
            for (int r = 0; r < out; ++r) hy[std::size_t(b) * sy + std::size_t(r) * yinc] = nan;
    const D* a = reinterpret_cast<const D*>(dev.up(ha));
    const D* x = reinterpret_cast<const D*>(dev.up(hx));
    D* y = reinterpret_cast<D*>(dev.up(hy));
    const auto al = dla::rt_scalar(to_dev(alpha), alpha_zero), be = dla::rt_scalar(to_dev(beta), beta_zero);
    const std::int64_t total = std::int64_t(out) * kItems;
    auto A = [=](std::int64_t b) { return v2::mat_wide(a + b * sa, m, n, ld); };
    auto X = [=](std::int64_t b) { return v2::vec_wide(x + b * sx, red, xinc); };
    auto Y = [=](std::int64_t b) { return v2::vec_wide(y + b * sy, out, yinc); };

    if constexpr (S == Shape::RowThread || S == Shape::ColThread) {
        dev.template run<false>(total, [=](sycl::nd_item<1>, std::int64_t g) {
            if (g >= total) return;
            const std::int64_t b = g / out;
            const int i = int(g - b * out);
            D acc{};
            if constexpr (S == Shape::RowThread) {
                if (!al.zero) acc = v2::dot<v2::Scope::Auto, Fma>(v2::Thread{}, A(b).row(i), X(b));
            } else {
                if (!al.zero) acc = v2::dot<v2::Scope::Auto, Fma>(v2::Thread{}, dla::conj_if(A(b).col(i), conj), X(b));
            }
            dla::store_fma(al, acc, be, Y(b)(i));
        });
    } else if constexpr (S == Shape::RowSegment) {
        dev.template run<true>(kItems, [=](sycl::nd_item<1> it, std::int64_t b) {
            if (b >= kItems) return;
            const auto sg = it.get_sub_group();
            const int lane = int(sg.get_local_linear_id());
            const int i = lane % m, jsub = lane / m;
            D acc{};
            if (!al.zero && jsub < P) acc = v2::detail::dot_partial<false, Fma>(dla::Member<P>{jsub}, A(b).row(i), X(b));
            acc = dla::fold_down<P>(sg, acc, v2::Dyn{m});
            if (lane < m) dla::store_fma(al, acc, be, Y(b)(i));
        });
    } else if constexpr (S == Shape::ColSubGroup) {
        dev.template run<true>(total, [=](sycl::nd_item<1> it, std::int64_t g) {
            if (g >= total) return;
            const v2::SubGroup<32> team{it.get_sub_group()};
            const std::int64_t b = g / out;
            const int j = int(g - b * out);
            D acc{};
            if (!al.zero) acc = v2::detail::dot_partial<false, Fma>(team, dla::conj_if(A(b).col(j), conj), X(b));
            acc = dla::leader_sum(team, acc);
            if (team.leader()) dla::store_fma(al, acc, be, Y(b)(j));
        });
    } else {
        constexpr int W = kSg / P;
        dev.template run<true>((total + W - 1) / W, [=](sycl::nd_item<1> it, std::int64_t sgid) {
            const std::int64_t base = sgid * W;
            if (base >= total) return;
            const auto sg = it.get_sub_group();
            const v2::Partition<P> team{sg};
            const std::int64_t g = base + int(sg.get_local_linear_id()) / P;
            const bool active = g < total;
            const std::int64_t gc = active ? g : total - 1;
            const std::int64_t b = gc / out;
            const int j = int(gc - b * out);
            D acc{};
            if (!al.zero && active) acc = v2::detail::dot_partial<false, Fma>(team, dla::conj_if(A(b).col(j), conj), X(b));
            acc = dla::leader_sum(team, acc);
            if (team.leader() && active) dla::store_fma(al, acc, be, Y(b)(j));
        });
    }

    const auto got = dev.down(reinterpret_cast<const T*>(y), hy.size());
    int bitdiff = 0, gapdiff = 0, emudiff = 0;
    for (int b = 0; b < kItems; ++b)
        for (int k = 0; k < sy; ++k) {
            const std::size_t e = std::size_t(b) * sy + k;
            bitdiff += !same_bits(got[e], got[std::size_t(b % kPeriod) * sy + k]);
            if (k % yinc != 0 || k / yinc >= out) gapdiff += !same_bits(got[e], hy[e]);
        }
    double worst = 0;
    for (int b = 0; b < kPeriod; ++b)
        for (int r = 0; r < out; ++r) {
            auto opa = [&](int k) {
                const T e = notrans ? ha[std::size_t(b) * sa + r + std::size_t(k) * ld] : ha[std::size_t(b) * sa + k + std::size_t(r) * ld];
                return conj ? conj_h(e) : e;
            };
            auto xk = [&](int k) { return hx[std::size_t(b) * sx + std::size_t(k) * xinc]; };
            const T y0 = hy[std::size_t(b) * sy + std::size_t(r) * yinc];
            std::vector<T> part(kSg, T(0));
            if (!alpha_zero)
                for (int t = 0; t < P; ++t)
                    for (int k = t; k < red; k += P) host_mac(part[t], opa(k), xk(k));
            const T acc = host_fold(part, P, 1)[0];  // member t's partial sits at fold index t
            T emu = T(0);
            host_mac(emu, alpha, acc);
            if (!beta_zero) host_mac(emu, beta, y0);
            const T g = got[std::size_t(b) * sy + std::size_t(r) * yinc];
            emudiff += !same_bits(g, emu);
            LD<T> ref = 0;
            long double s = 0;
            if (!alpha_zero)
                for (int k = 0; k < red; ++k) ref += wide(opa(k)) * wide(xk(k)), s += std::abs(wide(opa(k))) * std::abs(wide(xk(k)));
            ref = wide(alpha) * ref + (beta_zero ? LD<T>(0) : wide(beta) * wide(y0));
            s = std::abs(wide(alpha)) * s + (beta_zero ? 0 : std::abs(wide(beta)) * std::abs(wide(y0)));
            worst = std::max(worst, rel(g, ref, s));
        }
    EXPECT_EQ(bitdiff, 0) << where << ": items differ from their period-3 representative";
    EXPECT_EQ(gapdiff, 0) << where << ": y written outside y(k * yinc), k < len";
    EXPECT_EQ(emudiff, 0) << where << ": not the fma chain + shift-down tree bit for bit";
    EXPECT_VERIFY(T, Check::blas, red, worst) << where;
}

TYPED_TEST(DeviceBlasV2GemvNativeTest, GemvBodiesComposeBitExact) {
    using T = TypeParam;
    constexpr auto N = Transpose::NoTrans, Tt = Transpose::Trans, C = Transpose::ConjTrans;
    auto& d = this->dev;
    body_case<T, Shape::RowThread>(d, N, false, false, "RowThread");
    body_case<T, Shape::RowThread>(d, N, true, false, "RowThread alpha=0");
    body_case<T, Shape::RowSegment>(d, N, false, false, "RowSegment");
    body_case<T, Shape::RowSegment>(d, N, false, true, "RowSegment beta=0");
    body_case<T, Shape::ColThread>(d, C, false, false, "ColThread C");
    body_case<T, Shape::ColThread>(d, Tt, false, true, "ColThread T beta=0");
    body_case<T, Shape::ColSubGroup>(d, C, false, false, "ColSubGroup C");
    body_case<T, Shape::ColSubGroup>(d, Tt, true, false, "ColSubGroup T alpha=0");
    body_case<T, Shape::ColPartition>(d, C, false, false, "ColPartition C");
    body_case<T, Shape::ColPartition>(d, Tt, false, false, "ColPartition T");
    body_case<T, Shape::ColPartition>(d, C, false, true, "ColPartition C beta=0");
}

// fold_down over runtime strides the bodies do not use, and leader_sum on both team kinds.
TYPED_TEST(DeviceBlasV2GemvNativeTest, FoldDownStridesAndLeaderSum) {
    using T = TypeParam;
    using D = Dv<T>;
    constexpr int kCases = 7, kItemsF = 64;
    const auto hv = batched<T>(kSg, 41, kItemsF);
    const D* v = reinterpret_cast<const D*>(this->dev.up(hv));
    std::vector<T> hout(std::size_t(kItemsF) * kCases * kSg, T(0));
    D* o = reinterpret_cast<D*>(this->dev.up(hout));
    this->dev.template run<true>(kItemsF, [=](sycl::nd_item<1> it, std::int64_t b) {
        if (b >= kItemsF) return;
        const auto sg = it.get_sub_group();
        const int l = int(sg.get_local_linear_id());
        const D x = v[b * kSg + l];
        D* ob = o + b * kCases * kSg + l;
        ob[0 * kSg] = dla::fold_down<2>(sg, x, v2::Dyn{13});
        ob[1 * kSg] = dla::fold_down<4>(sg, x, v2::Dyn{7});
        ob[2 * kSg] = dla::fold_down<8>(sg, x, v2::Dyn{3});
        ob[3 * kSg] = dla::fold_down<16>(sg, x, v2::Dyn{2});
        ob[4 * kSg] = dla::fold_down<32>(sg, x);
        ob[5 * kSg] = dla::leader_sum(v2::Partition<4>{sg}, x);
        ob[6 * kSg] = dla::leader_sum(v2::SubGroup<32>{sg}, x);
    });
    const auto got = this->dev.down(reinterpret_cast<const T*>(o), hout.size());
    const int count[kCases] = {2, 4, 8, 16, 32, 4, 32}, stride[kCases] = {13, 7, 3, 2, 1, 1, 1};
    int diff = 0, checked = 0;
    for (int b = 0; b < kItemsF; ++b)
        for (int c = 0; c < kCases; ++c) {
            const std::vector<T> lanes(hv.begin() + std::ptrdiff_t(b) * kSg, hv.begin() + std::ptrdiff_t(b + 1) * kSg);
            const auto ref = host_fold(lanes, count[c], stride[c]);
            for (int l = 0; l < kSg; ++l) {
                const bool head = c == 5 ? l % 4 == 0 : l < stride[c];  // the lanes whose sum is defined
                if (!head) continue;
                ++checked;
                diff += !same_bits(got[(std::size_t(b) * kCases + c) * kSg + l], ref[l]);
            }
        }
    EXPECT_EQ(checked, kItemsF * (13 + 7 + 3 + 2 + 1 + 8 + 1));
    EXPECT_EQ(diff, 0) << "fold_down / leader_sum is not the shift-down tree bit for bit";
    // The tree is a sum: every head equals its members' total to the blas bound.
    double worst = 0;
    for (int l = 0; l < 13; ++l) {
        const T* in = hv.data();
        LD<T> ref = wide(in[l]) + wide(in[l + 13]);
        worst = std::max(worst, rel(got[std::size_t(0) * kSg + l], ref, std::abs(wide(in[l])) + std::abs(wide(in[l + 13]))));
    }
    EXPECT_VERIFY(T, Check::blas, 2, worst);
}

// store_fma with v2 tags: Zero never reads y (NaN), One adds y exactly, NegOne subtracts it.
TYPED_TEST(DeviceBlasV2GemvNativeTest, StoreFmaTags) {
    using T = TypeParam;
    using D = Dv<T>;
    constexpr int n = 97;
    const auto ha = batched<T>(n, 51, 1), hy = batched<T>(n, 52, 1);
    const T alpha = val<T>(53);
    std::vector<T> hz(n, T(std::numeric_limits<real_of<T>>::quiet_NaN()));
    const D* a = reinterpret_cast<const D*>(this->dev.up(ha));
    D* y1 = reinterpret_cast<D*>(this->dev.up(hy));
    D* y2 = reinterpret_cast<D*>(this->dev.up(hy));
    D* z = reinterpret_cast<D*>(this->dev.up(hz));
    const D ad = to_dev(alpha);
    this->dev.template run<false>(n, [=](sycl::nd_item<1>, std::int64_t i) {
        if (i >= n) return;
        dla::store_fma(v2::Scalar<D>{ad}, a[i], v2::Zero{}, z[i]);
        dla::store_fma(v2::Scalar<D>{ad}, a[i], v2::One{}, y1[i]);
        dla::store_fma(v2::One{}, a[i], v2::NegOne{}, y2[i]);
    });
    const auto gz = this->dev.down(reinterpret_cast<const T*>(z), n);
    const auto g1 = this->dev.down(reinterpret_cast<const T*>(y1), n);
    const auto g2 = this->dev.down(reinterpret_cast<const T*>(y2), n);
    int diff = 0;
    for (int i = 0; i < n; ++i) {
        T ez = T(0), e1 = T(0);
        host_mac(ez, alpha, ha[i]);
        host_mac(e1, alpha, ha[i]);
        e1 = e1 + hy[i];
        const T e2 = (T(0) + ha[i]) - hy[i];
        diff += !same_bits(gz[i], ez) + !same_bits(g1[i], e1) + !same_bits(g2[i], e2);
    }
    EXPECT_EQ(diff, 0);
}

}  // namespace
