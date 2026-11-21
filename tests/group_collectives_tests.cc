// SPDX-License-Identifier: MIT
// batchlas::portable group collectives (include/batchlas/util/group-collectives.hh): exactness of
// work-group / joint reductions, on the default device and the CPU leg.
#include <gtest/gtest.h>
#include <batchlas/verify/tolerance.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <batchlas/blas/linalg.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/util/group-collectives.hh>

#include <src/queue.hh>

using namespace batchlas;

namespace {

std::uint64_t test_hash(std::uint64_t i) { return (i + 1) * 0x9E3779B97F4A7C15ull; }

// A positive integer below 2^(digits-1) / count: any partial sum of `count` of them is exact in Real,
// so a group sum must be EXACT in any order, and a full sum keeps ~digits significant bits, so a
// result rounded on its way out cannot pass.
template <typename Real>
Real exact_sum_value(std::uint64_t i, std::uint64_t count) {
    const std::uint64_t m = (std::uint64_t(1) << (std::numeric_limits<Real>::digits - 1)) / count;
    return static_cast<Real>(m / 2 + (test_hash(i) >> 11) % (m / 2 + 1));
}

// A signed value in [-0.5, 0.5) with a full mantissa, for max/min.
template <typename Real>
Real full_mantissa_value(std::uint64_t i) {
    constexpr int kBits = std::numeric_limits<Real>::digits;
    return static_cast<Real>(std::ldexp(static_cast<double>(test_hash(i) >> (64 - kBits)), -kBits)) - Real(0.5);
}

template <typename T>
int check_portable_group_reductions(Queue& ctx, size_t wg, const char* where) {
    constexpr size_t kBatch = 1024;
    const size_t jlen = 2 * wg + 5;
    UnifiedVector<T> in(kBatch * wg), fin(kBatch * wg), jin(kBatch * jlen), jneg_in(kBatch * 3);
    UnifiedVector<T> item_sum(kBatch * wg), item_max(kBatch * wg), grp(kBatch * 8);
    for (size_t i = 0; i < in.size(); ++i) in[i] = exact_sum_value<T>(i, wg);
    for (size_t i = 0; i < fin.size(); ++i) fin[i] = full_mantissa_value<T>(i + 3333);
    for (size_t i = 0; i < jin.size(); ++i) jin[i] = exact_sum_value<T>(i + 7777, jlen);
    for (size_t i = 0; i < jneg_in.size(); ++i) jneg_in[i] = -std::abs(full_mantissa_value<T>(i)) - T(0.25);
    T* pin = in.data();
    T* pf = fin.data();
    T* pj = jin.data();
    T* pn = jneg_in.data();
    T* psum = item_sum.data();
    T* pmax = item_max.data();
    T* pg = grp.data();
    ctx->parallel_for(sycl::nd_range<1>(kBatch * wg, wg), [=](sycl::nd_item<1> it) {
        const auto g = it.get_group();
        const size_t b = g.get_group_linear_id();
        const size_t gid = it.get_global_linear_id();
        const T x = pin[gid];
        psum[gid] = batchlas::portable::reduce_over_group(g, x, sycl::plus<T>());
        pmax[gid] = batchlas::portable::reduce_over_group(g, pf[gid], sycl::maximum<T>());
        const T mn = batchlas::portable::reduce_over_group(g, pf[gid], sycl::minimum<T>());
        T* row = pj + b * jlen;
        const T js = batchlas::portable::joint_reduce(g, row, row + jlen, T(0), sycl::plus<T>());
        const T jneg = batchlas::portable::joint_reduce(g, pn + b * 3, pn + b * 3 + 3, sycl::maximum<T>());
        const sycl::vec<T, 3> v = batchlas::portable::reduce_over_group(
            g, sycl::vec<T, 3>(x, T(2) * x, x + T(1)), sycl::plus<sycl::vec<T, 3>>());
        const std::complex<T> c = batchlas::portable::reduce_over_group(
            g, std::complex<T>(x, T(-2) * x), sycl::plus<std::complex<T>>());
        if (it.get_local_linear_id() == 0) {
            const T r[8] = {mn, js, jneg, v[0], v[1], v[2], c.real(), c.imag()};
            for (int k = 0; k < 8; ++k) pg[b * 8 + k] = r[k];
        }
    });
    ctx.wait_and_throw();
    int bad = 0;
    for (size_t b = 0; b < kBatch && bad < 8; ++b) {
        long double s = 0, js = 0;
        T mx = fin[b * wg], mn = fin[b * wg];
        for (size_t i = 0; i < wg; ++i) {
            s += in[b * wg + i];
            mx = std::max(mx, fin[b * wg + i]);
            mn = std::min(mn, fin[b * wg + i]);
        }
        for (size_t i = 0; i < jlen; ++i) js += jin[b * jlen + i];
        const T jneg = std::max({jneg_in[b * 3], jneg_in[b * 3 + 1], jneg_in[b * 3 + 2]});
        const T want[8] = {mn, T(js), jneg, T(s), T(2 * s), T(s + wg), T(s), T(-2 * s)};
        for (int k = 0; k < 8; ++k) {
            bad += grp[b * 8 + k] != want[k];
            EXPECT_EQ(grp[b * 8 + k], want[k]) << where << " wg=" << wg << " group " << b << " result " << k;
        }
        for (size_t i = 0; i < wg; ++i) {
            bad += item_sum[b * wg + i] != T(s) || item_max[b * wg + i] != mx;
            EXPECT_EQ(item_sum[b * wg + i], T(s)) << where << " wg=" << wg << " group " << b << " item " << i;
            EXPECT_EQ(item_max[b * wg + i], mx) << where << " wg=" << wg << " group " << b << " item " << i;
        }
    }
    return bad;
}

template <typename Real>
void check_portable_group_reductions_on(Queue& ctx, const char* where) {
    const size_t max_wg = std::min<size_t>(1024, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    for (size_t wg : {size_t(1), size_t(16), size_t(32), size_t(33), size_t(100), size_t(256), max_wg}) {
        if (wg <= max_wg && check_portable_group_reductions<Real>(ctx, wg, where) != 0) return;
    }
}

}  // namespace

// AdaptiveCpp 25.10 rounds a WORK-GROUP floating reduce to ~16 (float) / ~42 (double) mantissa
// bits and pads joint_reduce(maximum) with a positive identity; batchlas::portable fixes both.
// The CPU leg is where acpp's sub-group is one item, so a work-group has up to 1024 sub-groups.
TEST(GroupCollectivesTest, PortableGroupReductionsAreExact) {
    Queue gpu(Device::default_device());
    check_portable_group_reductions_on<float>(gpu, "default device");
    if (!test_utils::kNativeGpuLacksFp64) check_portable_group_reductions_on<double>(gpu, "default device");
#if BATCHLAS_HAS_CPU_TARGET
    std::unique_ptr<Queue> cpu;
    try {
        cpu = std::make_unique<Queue>(Device("cpu"));
    } catch (const std::exception& e) {
        std::cout << "[ INFO ] cpu leg not run: " << e.what() << std::endl;
        return;
    }
    check_portable_group_reductions_on<float>(*cpu, "cpu device");
    check_portable_group_reductions_on<double>(*cpu, "cpu device");
#endif
}

namespace {

// Integers of mixed magnitude (1 .. 2^42, either sign) plus one 2^50 per group. Every partial sum of
// a group stays below 2^53 in magnitude, so ANY summation order is exact in double and the sum must
// equal the int64 sum bit for bit; the sums keep about 50 significant bits, so a reduce that rounds
// on the way (acpp's SSCP broadcast keeps ~42) or loses a partial cannot pass.
std::int64_t mixed_magnitude_integer(std::uint64_t i, std::uint64_t in_group) {
    if (in_group == 1) return std::int64_t(1) << 50;
    const std::uint64_t h = test_hash(i * 7 + 3);
    const std::int64_t v = static_cast<std::int64_t>(1 + ((h >> 20) % 1023)) << ((h >> 40) % 33);
    return (h >> 63) ? -v : v;
}

int significant_bits(std::int64_t v) {
    std::uint64_t a = static_cast<std::uint64_t>(v < 0 ? -v : v);
    if (a == 0) return 0;
    while (!(a & 1)) a >>= 1;
    int bits = 0;
    for (; a != 0; a >>= 1) ++bits;
    return bits;
}

}  // namespace

namespace {

// Group b's source value: hashed bits, with -0 at b % 7 == 3 and a NaN payload at b % 7 == 5
// (a sum-based broadcast loses both; bit identity is checked).
template <typename T> struct real_of {
    using type = T;
};
template <typename R> struct real_of<std::complex<R>> {
    using type = R;
};

template <typename T> T bcast_value(std::uint64_t b) {
    using R = typename real_of<T>::type;
    const auto real = [](std::uint64_t h) {
        if constexpr (std::is_integral_v<R>) return static_cast<R>(h >> 33);
        else return static_cast<R>(std::ldexp(static_cast<double>(h >> 20), -30)) - R(1000);
    };
    R re = real(test_hash(b)), im = real(test_hash(b + 99991));
    if constexpr (!std::is_integral_v<R>) {
        if (b % 7 == 3) re = R(-0.0);
        if (b % 7 == 5) re = std::numeric_limits<R>::quiet_NaN();
    }
    if constexpr (std::is_same_v<T, R>) return re;
    else return T(re, im);
}

// Every item of every group must get the source's bits: the source at item 0, 17 and wg - 1,
// as a linear id and as a sycl::id<1>. Each item first holds its own distinct value, so an item
// the broadcast misses keeps a wrong one. ReqdSg32: launched at reqd_sub_group_size(32), where
// DPC++'s CPU device reached only lanes 0..15 of each sub-group (2026-11-21).
template <typename T, bool ReqdSg32> int check_portable_broadcast(Queue& ctx, size_t wg, const char* where) {
    constexpr size_t kBatch = 256, kSrc = 3;
    const size_t srcs[kSrc] = {0, std::min<size_t>(17, wg - 1), wg - 1};
    UnifiedVector<T> mine(kBatch * wg), out(kBatch * wg * kSrc * 2);
    for (size_t b = 0; b < kBatch; ++b)
        for (size_t i = 0; i < wg; ++i) mine[b * wg + i] = bcast_value<T>(b * 4099 + i + 1);
    const T* pm = mine.data();
    T* po = out.data();
    const size_t s0 = srcs[0], s1 = srcs[1], s2 = srcs[2];
    const auto body = [=](sycl::nd_item<1> it) {
        const auto g = it.get_group();
        const size_t gid = it.get_global_linear_id();
        const T x = pm[gid];
        const size_t o = gid * kSrc * 2;
        po[o + 0] = batchlas::portable::group_broadcast(g, x);
        po[o + 1] = batchlas::portable::group_broadcast(g, x, s1);
        po[o + 2] = batchlas::portable::group_broadcast(g, x, s2);
        po[o + 3] = batchlas::portable::group_broadcast(g, x, sycl::id<1>(s0));
        po[o + 4] = batchlas::portable::group_broadcast(g, x, sycl::id<1>(s1));
        po[o + 5] = batchlas::portable::group_broadcast(g, x, sycl::id<1>(s2));
    };
    if constexpr (ReqdSg32)
        ctx->parallel_for(sycl::nd_range<1>(kBatch * wg, wg), [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) { body(it); });
    else
        ctx->parallel_for(sycl::nd_range<1>(kBatch * wg, wg), body);
    ctx.wait_and_throw();
    int bad = 0;
    for (size_t b = 0; b < kBatch && bad < 8; ++b)
        for (size_t i = 0; i < wg && bad < 8; ++i)
            for (size_t k = 0; k < kSrc * 2; ++k) {
                const T want = mine[b * wg + srcs[k % kSrc]];
                const bool ok = std::memcmp(&out[(b * wg + i) * kSrc * 2 + k], &want, sizeof(T)) == 0;
                bad += !ok;
                EXPECT_TRUE(ok) << where << " wg=" << wg << " reqd_sg32=" << ReqdSg32 << " group " << b << " item " << i
                                << " source " << srcs[k % kSrc] << (k < kSrc ? " (linear id)" : " (sycl::id)");
            }
    return bad;
}

template <typename T> void check_portable_broadcast_on(Queue& ctx, const char* where) {
    if (std::is_same_v<typename real_of<T>::type, double> && !sycl_queue(ctx).get_device().has(sycl::aspect::fp64)) return;
    const auto sgs = sycl_queue(ctx).get_device().get_info<sycl::info::device::sub_group_sizes>();
    const bool sg32 = BATCHLAS_SYCL_IMPL_ACPP || std::find(sgs.begin(), sgs.end(), size_t(32)) != sgs.end();
    const size_t max_wg = std::min<size_t>(1024, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    for (size_t wg : {size_t(32), size_t(64), size_t(96), size_t(256), max_wg}) {
        if (wg > max_wg) continue;
        if (check_portable_broadcast<T, false>(ctx, wg, where) != 0) return;
        if (sg32 && check_portable_broadcast<T, true>(ctx, wg, where) != 0) return;
    }
}

template <typename T> void check_portable_broadcast_all(const char* type) {
    Queue gpu(Device::default_device());
    SCOPED_TRACE(type);
    check_portable_broadcast_on<T>(gpu, "default device");
#if BATCHLAS_HAS_CPU_TARGET
    std::unique_ptr<Queue> cpu;
    try {
        cpu = std::make_unique<Queue>(Device("cpu"));
    } catch (const std::exception& e) {
        std::cout << "[ INFO ] cpu leg not run: " << e.what() << std::endl;
        return;
    }
    check_portable_broadcast_on<T>(*cpu, "cpu device");
#endif
}

}  // namespace

// portable::group_broadcast over a work-group reaches every item with the source's exact bits,
// for 4-, 8- and 16-byte T, at reqd_sub_group_size(32) and without. Red with `using
// sycl::group_broadcast` (the bare DPC++ call): the CPU leg at reqd_sub_group_size(32).
TEST(GroupCollectivesTest, PortableWorkGroupBroadcastReachesEveryItem) {
    check_portable_broadcast_all<std::int32_t>("int32");
    check_portable_broadcast_all<float>("float");
    check_portable_broadcast_all<double>("double");
    check_portable_broadcast_all<std::complex<float>>("complex<float>");
    check_portable_broadcast_all<std::complex<double>>("complex<double>");
}

// P0.6: a double WORK-GROUP sum through batchlas::portable is exact on every SYCL implementation.
// Only acpp generic (SSCP) rounds its own reduce, so only there does disabling kExactWgReducePass
// turn this red; on multipass CUDA it guards the replacement alone. The second sum (mirrored,
// negated) only checks that a second reduce in one kernel returns its own sum: it is NOT a race
// guard, since dropping the trailing barrier stays green on both flows.
// evidence: docs/design/sycl-implementations.md#sycl-impl-cuda-aot
TEST(GroupCollectivesTest, PortableWorkGroupDoubleSumIsExact) {
    if (test_utils::kNativeGpuLacksFp64) GTEST_SKIP() << "no device fp64";
    Queue ctx(Device::default_device());
    constexpr size_t kBatch = 1024;
    const size_t max_wg = std::min<size_t>(1024, ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    for (size_t wg : {size_t(64), size_t(96), size_t(256), max_wg}) {
        if (wg > max_wg) continue;
        UnifiedVector<double> in(kBatch * wg), out(kBatch * wg), out_neg(kBatch * wg);
        std::vector<std::int64_t> want(kBatch, 0);
        int strong = 0;
        for (size_t b = 0; b < kBatch; ++b) {
            for (size_t i = 0; i < wg; ++i) {
                const std::int64_t v = mixed_magnitude_integer(b * wg + i, i);
                in[b * wg + i] = static_cast<double>(v);
                want[b] += v;
            }
            strong += significant_bits(want[b]) >= 46;
        }
        ASSERT_GT(strong, static_cast<int>(kBatch / 2)) << "inputs no longer need 46+ significant bits";
        const double* pin = in.data();
        double* po = out.data();
        double* pn = out_neg.data();
        ctx->parallel_for(sycl::nd_range<1>(kBatch * wg, wg), [=](sycl::nd_item<1> it) {
            const auto g = it.get_group();
            const size_t gid = it.get_global_linear_id();
            const size_t mirror = g.get_group_linear_id() * wg + (wg - 1 - it.get_local_linear_id());
            po[gid] = batchlas::portable::reduce_over_group(g, pin[gid], sycl::plus<double>());
            pn[gid] = batchlas::portable::reduce_over_group(g, -pin[mirror], sycl::plus<double>());
        });
        ctx.wait_and_throw();
        int bad = 0;
        for (size_t b = 0; b < kBatch && bad < 8; ++b) {
            const double w = static_cast<double>(want[b]);
            const double wn = -w;
            for (size_t i = 0; i < wg && bad < 8; ++i) {
                const bool ok = std::memcmp(&out[b * wg + i], &w, sizeof(double)) == 0 &&
                                std::memcmp(&out_neg[b * wg + i], &wn, sizeof(double)) == 0;
                bad += !ok;
                EXPECT_TRUE(ok) << "wg=" << wg << " group " << b << " item " << i << ": got " << out[b * wg + i]
                                << " and " << out_neg[b * wg + i] << ", exact " << want[b];
            }
        }
    }
}
