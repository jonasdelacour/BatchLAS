// SPDX-License-Identifier: MIT
#pragma once
// ?GETRI's device op: A^-1 from getrf's factor, out of place as the host op (C must not alias A).
// The host blocked route's steps (getri_blocked.cc): C := the permutation getrf applied, then
// C := U^-1 L^-1 C on the in-group trsm. tier::Blocked only: any n, getrf's executor and plan.

#include <cstdint>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getrs.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace routine {
struct getri {
    static constexpr const char* name = "getri";
};
}  // namespace routine

namespace detail {

// C := P (C(i, perm(i)) = 1, zero elsewhere; row i walked back through the interchanges, the
// host route's formula) and the first exact zero U(i, i), 1-based, or 0. Every item scans the
// diagonal, so the result is uniform with no reduction. Ends on a global barrier.
template <class Ex, class MA, class MC, class P>
BATCHLAS_DEVICE_INLINE std::int32_t permute_identity(const Ex& ex, const MA& a, const MC& c, const P& ipiv) {
    using D = typename MC::value_type;
    const int n = c.rows();
    laset(ex, Zero{}, Zero{}, c);
    ex.barrier(Mem::Global);
    for_each_owned(ex, n, [&](int i) BATCHLAS_DEVICE_BODY { c(i, laswp_index<dir::Backward>(ipiv, 0, n, i)) = Arith<D>::one(); });
    ex.barrier(Mem::Global);
    std::int32_t info = 0;
    for (int i = 0; i < n; ++i)
        if (info == 0 && Arith<D>::is_zero(a(i, i))) info = i + 1;
    return info;
}

}  // namespace detail

/// getri's Blocked<NB>: any n; getrf's Blocked executor and local plan (no global workspace).
template <int NB, class Alg, class T>
struct tier_info<routine::getri, tier::Blocked<NB, Alg>, T> : tier_info<routine::getrf, tier::Blocked<NB, Alg>, T> {};
template <class T> struct tiers<routine::getri, T> {
    using type = tier::list<tier::Blocked<32>>;
};

/// The workspace of one n x n inversion for getri<Tier> (host and device; constexpr).
template <class Tier, class T> constexpr WorkspaceSize getri_buffer_size(int n) {
    return tier_info<routine::getri, resolve_tier_t<routine::getri, T, Tier>, T>::buffer_size(n);
}

/// @brief ?GETRI in a kernel: \f$C := A^{-1}\f$ from getrf's factor of A (A not modified).
///
/// Collective as getrf; A, ipiv visible to all items on entry, C visible to all on return. As
/// the host blocked route, the solves run whatever info says: for info > 0, C is not meaningful.
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A and C n x n of any size in
///   global memory, C not aliasing A; `ws.local` >= getri_buffer_size<Tier, T>(n) bytes.
/// @param[in]  ex    the executor
/// @param[in]  a     getrf's L and U of this matrix
/// @param[out] c     this matrix's inverse
/// @param[in]  ipiv  getrf's n pivots, 1-based int32
/// @param[in]  ws    this matrix's workspace slice
/// @param[in]  live  false: a dead matrix of the launch (executor-uniform)
/// @return LAPACK's info, uniform across `ex`: 0, or the first k with \f$U_{kk} = 0\f$ exactly
template <class Tier, class Ex, class MA, class MC>
BATCHLAS_DEVICE_INLINE std::int32_t getri(const Ex& ex, const MA& a, const MC& c, const std::int32_t* ipiv, const Workspace& ws = {},
                                       bool live = true) {
    using D = detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::getri, D, Tier>;
    static_assert(HasTier<routine::getri, Tr, D>, "getri: this tier is not offered for this scalar type");
    static_assert(std::is_same_v<D, detail::lu_value_t<MC>>, "getri: A and C hold one scalar type");
    using Cfg = detail::lu_blocked_cfg_t<D, Tr>;
    static_assert(Ex::size == Cfg::threads, "getri: tier::Blocked runs on a WorkGroup of tier_info::wg_size items");
    if (!live) return 0;
    const auto ag = l3::global_of(a);
    const auto cg = l3::global_of(c);
    const int n = ag.n;
    constexpr auto pl = detail::lu_blocked_plan<D, Cfg>();
    const std::int32_t info = detail::permute_identity(ex, mat_wide(ag.p, n, n, ag.ld), mat_wide(cg.p, n, n, cg.ld),
                                                             detail::PivIn{ipiv});
    detail::lu_trsm_pair<Transpose::NoTrans, Tr::nb>(ex, ag, cg, l3::Smem<Cfg>{carve_local(ws, pl.smem).get_raw()});
    return info;
}

}  // namespace batchlas::device
