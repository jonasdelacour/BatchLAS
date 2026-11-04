// SPDX-License-Identifier: MIT
#pragma once
// stedc-private device LA v2 helpers (beyond include/batchlas/device), on v2::RtGroup,
// v2::WorkGroup and v2 partitions. Tests: tests/device_blas_v2_stedc_tests.cc.

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::stedc_v2 {

namespace v2 = batchlas::device;

template <class Ex> inline constexpr bool is_group_v = v2::is_work_group_v<Ex>;
template <int SG> inline constexpr bool is_group_v<v2::RtGroup<v2::Scope::Group, SG>> = true;

// The legacy sycl::group_barrier on a work-group, nothing on a team. Not v2::sync: that fences
// only local memory on a work-group (these barriers also order global Q and v) and is a
// sub-group barrier on a partition, where the legacy loops had none.
template <class Ex> BATCHLAS_DLA_INLINE void group_sync(const Ex& ex) {
    if constexpr (is_group_v<Ex>) sycl::group_barrier(ex.it.get_group());
}

// The leader's v in every item: a loop exit on a reduced value goes through this, or a
// partition may diverge and deadlock in its next shuffle. Work-group: the legacy broadcast
// (RtGroup::bcast takes 4/8-byte words only off NVPTX; these are bools).
template <class Ex, class T> BATCHLAS_DLA_INLINE T uniform(const Ex& ex, T v) {
    if constexpr (v2::is_thread_v<Ex>) return v;
    else if constexpr (is_group_v<Ex>) return batchlas::portable::group_broadcast(ex.it.get_group(), v, 0);
    else return ex.bcast(v, 0u);
}

// portable::joint_reduce over a contiguous x: item lid reads x(lid), x(lid + width), ... (its
// Cyclic elements, so an owned write right before needs no barrier); an empty x gives T{}.
template <class Ex, class X, class Op> BATCHLAS_DLA_INLINE auto fold(const Ex& ex, const X& x, Op op) {
    static_assert(is_group_v<Ex>, "stedc_v2::fold: a work-group executor");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(x.inc)>, v2::Unit>, "stedc_v2::fold: x must be contiguous");
    const auto first = x.p + x.off;
    return batchlas::portable::joint_reduce(ex.it.get_group(), first, first + x.size(), op);
}

template <class R> struct ScaledProd {
    R mant;  // in [0.5, 1) unless the product is 0, Inf or NaN
    std::int32_t expo;
};

// prod over i < n of f(i) as mant * 2^expo: per-item frexp renormalisation (exact), mantissas
// multiplied in rounds of 32 items (partials >= 2^-33). In range: the naive product, bitwise.
template <class Ex, class F> BATCHLAS_DLA_INLINE auto prod_scaled(const Ex& ex, int n, F&& f) {
    using R = std::remove_cvref_t<decltype(f(0))>;
    R mant = R(1);
    std::int32_t expo = 0;
    v2::for_each_owned(ex, n, [&](int i) {
        mant *= f(i);
        if (mant != R(0) && sycl::isfinite(mant)) {
            int e = 0;
            mant = sycl::frexp(mant, &e);
            expo += e;
        }
    });
    R mant_all = R(1);
    std::int32_t expo_all = v2::allreduce(ex, expo);
    const int width = v2::detail::size_of(ex);
    const int lane = ex.lid();
    for (std::int32_t r0 = 0; r0 < width; r0 += 32) {
        const bool mine = lane >= r0 && lane < r0 + 32;
        mant_all *= v2::allreduce(ex, mine ? mant : R(1), sycl::multiplies<R>());
        if (mant_all != R(0) && sycl::isfinite(mant_all)) {
            int e = 0;
            mant_all = sycl::frexp(mant_all, &e);
            expo_all += e;
        }
    }
    return ScaledProd<R>{mant_all, expo_all};
}

// sqrt(|mant| 2^expo), split so neither the square root's argument nor its result leaves range.
template <class R> BATCHLAS_DLA_INLINE R sqrt_scaled(ScaledProd<R> q) {
    const std::int32_t half = q.expo >> 1;  // floor(expo / 2)
    const std::int32_t rem = q.expo - 2 * half;
    const R mag = sycl::sqrt(sycl::fabs(q.mant) * (rem ? R(2) : R(1)));
    return sycl::ldexp(mag, half);
}

}  // namespace batchlas::stedc_v2
