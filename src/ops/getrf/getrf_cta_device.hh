// SPDX-License-Identifier: MIT
#pragma once

// GETRF scalar helpers and launch constants; the panel's device code is getrf_v2.hh.

#include <src/sycl/device_scalar.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

using batchlas::sycl_device::Cx;

template <typename D> struct RealOf        { using type = D; };
template <typename R> struct RealOf<Cx<R>> { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

template <typename D>
inline D lu_zero() {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{real_of<D>(0), real_of<D>(0)};
    } else {
        return D(0);
    }
}

// LAPACK's cabs1: |Re| + |Im|; deliberately not the modulus (the pivot metric).
template <typename D>
inline real_of<D> lu_cabs1(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return sycl::fabs(a.re) + sycl::fabs(a.im);
    } else {
        return sycl::fabs(a);
    }
}

// CONSTANT, not a function of wg width: capacity query, fit predicate and launcher agree.
inline constexpr int kLuRedSlots = 32;

// The group the panel's barriers synchronise. SubGroup makes G-packing legal; WorkGroup on a
// packed launch is a RACE (wrong answers), so the launcher derives the two together.
enum class LuScope { SubGroup, WorkGroup };

}  // namespace batchlas::getrf_native
