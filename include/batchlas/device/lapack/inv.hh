// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_lu
/// @brief The host `batchlas::inv` in a kernel: the inverse of one matrix, `A` read only.
///
/// The routine is inv(); its routine tag is routine::inv. It runs getrf() on a copy of `A` in
/// the caller's global workspace, then getri() into the output. Its one tier is
/// `tier::Blocked<32>` (any n): the local plan is getrf()'s, the global plan the n x n copy and
/// its n pivots.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getri.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace routine {
/// @ingroup api_device_lapack_lu
/// @brief The routine tag of inv(): names the routine to `with_tier`, `launch`, `tier_info`,
/// `tiers_t` and `buffer_size`.
struct inv {
    static constexpr const char* name = "inv";  ///< The routine's name, "inv".
};
}  // namespace routine

namespace detail {

// getrf's local slots first (the same offsets getrf carves), then the global copy (ld = n) and
// its pivots.
template <class Scalar> struct InvPlan {
    detail::LuBlockedPlan<Scalar> local;
    WsSlot<Scalar> lu;
    WsSlot<int> ipiv;
    WorkspaceSize size;
};
template <class Scalar, class Cfg> constexpr InvPlan<Scalar> inv_plan(int n) {
    Plan p;
    const auto local = detail::lu_blocked_take<Scalar, Cfg>(p);
    const auto lu = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    const auto ipiv = p.take<int>(WsSpace::Global, static_cast<std::size_t>(n));
    return {local, lu, ipiv, p.size()};
}

}  // namespace detail

/// @ingroup api_device_lapack_lu
/// @brief inv()'s blocked tier `tier::Blocked<NB>`: any n; getrf()'s Blocked executor and local
/// plan, and a global part per matrix of n * n elements plus n pivots.
///
/// `admits(n)` is n >= 1. `buffer_size(n)` grows with n through its global part.
template <int NB, class Alg, class Scalar> struct tier_info<routine::inv, tier::Blocked<NB, Alg>, Scalar> {
    using cfg = detail::lu_blocked_cfg_t<Scalar, tier::Blocked<NB, Alg>>;
    static_assert(std::is_same_v<typename cfg::value_type, Scalar>, "inv: tier::Blocked's l3 configuration is for another scalar type");
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::inv_plan<Scalar, cfg>(n).size; }
};
/// @ingroup api_device_lapack_lu
/// @brief inv()'s tiers for every `Scalar`: `Blocked<32>`.
template <class Scalar> struct tiers<routine::inv, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @ingroup api_device_lapack_lu
/// @brief The host inv in a kernel: \f$A_{\mathrm{inv}} := A^{-1}\f$, A not modified.
///
/// No argument is checked; the workspace pointers are used as given.
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A and Ainv n x n of any size
///   in global memory, not aliasing; `ws.local` and `ws.global` (this matrix's slice) of at
///   least buffer_size<routine::inv, Tier, T>(n) bytes each.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::inv, T>` that admits N.
///
/// @pre Collective as getrf(): every item calls with the same arguments; A visible to all items on
/// entry, Ainv on return.
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]  ex    the executor
/// @param[in]  a     this matrix; read only
/// @param[out] ainv  this matrix's inverse (not meaningful for info > 0)
/// @param[in]  ws    this matrix's workspace slice
/// @param[in]  live  false: a dead matrix of the launch (executor-uniform)
/// @return getrf()'s info, uniform across `ex` (the host inv reports none):
/// @retval 0  success, or `live` false
/// @retval k  (k > 0) \f$U_{kk}\f$ of the copy's factor is exactly zero: `ainv` is not meaningful
template <class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int inv(const Ex& ex, const MA& a, const MB& ainv, const Workspace& ws, bool live = true) {
    using D = detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::inv, D, Tier>;
    static_assert(HasTier<routine::inv, Tr, D>, "inv: this tier is not offered for this scalar type");
    using Cfg = detail::lu_blocked_cfg_t<D, Tr>;
    static_assert(Ex::size == Cfg::threads, "inv: tier::Blocked runs on a WorkGroup of tier_info::wg_size items");
    if (!live) return 0;
    const auto ag = l3::global_of(a);
    const int n = ag.n;
    const auto pl = detail::inv_plan<D, Cfg>(n);
    const auto w = mat_wide(carve_global(ws, pl.lu), n, n, n);
    const auto ipiv = vec(carve_global(ws, pl.ipiv), n);
    lacpy(ex, mat_wide(ag.p, n, n, ag.ld), w);
    ex.barrier(Mem::Global);
    const int info = getrf<Tr>(ex, w, ipiv, ws);
    (void)getri<Tr>(ex, w, ipiv, ainv, ws);
    return info;
}

}  // namespace batchlas::device
