// SPDX-License-Identifier: MIT
#pragma once
// stedc-private v2 helpers core lacks. Tests: tests/device_blas_v2_stedc_tests.cc.

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::stedc_v2 {

namespace v2 = batchlas::device;

template <class Ex> inline constexpr bool is_group_v = v2::is_work_group_v<Ex>;
template <int SG> inline constexpr bool is_group_v<v2::RtGroup<v2::Scope::Group, SG>> = true;

// The leader's v in every item (a loop exit on a reduced value, else a partition may diverge and
// deadlock); work-group: the legacy broadcast of a bool (core's group bcast moves 4/8-byte words).
template <class Ex, class T> BATCHLAS_DLA_INLINE T uniform(const Ex& ex, T v) {
    if constexpr (v2::is_thread_v<Ex>) return v;
    else if constexpr (is_group_v<Ex>) return batchlas::portable::group_broadcast(ex.it.get_group(), v, 0);
    else return ex.uniform(v);
}

// portable::joint_reduce over a contiguous x (core allreduce folds in another order); empty: T{}.
template <class Ex, class X, class Op> BATCHLAS_DLA_INLINE auto fold(const Ex& ex, const X& x, Op op) {
    static_assert(is_group_v<Ex>, "stedc_v2::fold: a work-group executor");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(x.inc)>, v2::Unit>, "stedc_v2::fold: x must be contiguous");
    const auto first = x.p + x.off;
    return batchlas::portable::joint_reduce(ex.it.get_group(), first, first + x.size(), op);
}

}  // namespace batchlas::stedc_v2
