// SPDX-License-Identifier: MIT
#pragma once

// The native GETRF panel's device code: v2::getf2 (v2/ext/getrf.hh), whose pivot butterfly,
// cabs1 metric, info rule and barriers are load-bearing. evidence: docs/perf/lu.md#the-shape-of-the-getrf-device-code

#include <src/sycl/device_scalar.hh>
#include <src/kernels/common/dla_v2_scalar.hh>

#include <batchlas/blas/device/v2/ext/getrf.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

using batchlas::sycl_device::Cx;
namespace v2 = ::batchlas::device::v2;

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

// LAPACK's cabs1, never the modulus; v2::abs1's spelling, kept for gesv_tiny.cc.
template <typename D>
inline real_of<D> lu_cabs1(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return sycl::fabs(a.re) + sycl::fabs(a.im);
    } else {
        return sycl::fabs(a);
    }
}

// CONSTANT, not a function of wg width: the capacity query, the fit predicate and the
// launcher must agree on the SLM footprint.
inline constexpr int kLuRedSlots = 32;

// The phase barriers' group. SubGroup makes G-packing legal; WorkGroup on a packed launch is a
// RACE, so the launcher derives the two together. The width is the launch's (run time).
enum class LuScope { SubGroup, WorkGroup };

template <LuScope SC>
using LuExec = v2::DynGroup<SC == LuScope::SubGroup ? v2::Scope::Sub : v2::Scope::Group>;

// ?GETF2 for `kmax` steps; `piv_base` (global first row) makes ipiv/info global and 1-based.
// `info_item` is READ too (first failure wins across panels): launchers zero it first.
template <LuScope SC, typename MA, typename ValAcc, typename IdxAcc>
inline void getf2_panel_device(const LuExec<SC>& ex, const MA& A, int kmax,
                               int* piv_item, int piv_base, int32_t* info_item,
                               ValAcc rval, IdxAcc ridx) {
    int32_t info = ex.leader() ? *info_item : 0;
    info = v2::getf2(ex, A, kmax, v2::Ipiv{piv_item, piv_base}, info, rval, ridx);
    if (ex.leader()) *info_item = info;
}

}  // namespace batchlas::getrf_native
