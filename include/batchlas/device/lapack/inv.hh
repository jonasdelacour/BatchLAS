// SPDX-License-Identifier: MIT
#pragma once
// The device inv (host batchlas::inv): Ainv := A^-1 with A read only, as getrf on a copy of A in
// the caller's global workspace, then getri into Ainv. tier::Blocked only: any n; the local plan
// is getrf's, the global plan the n x n copy and its n pivots.

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

namespace batchlas::device::inline v2 {

namespace routine {
struct inv {
    static constexpr const char* name = "inv";
};
}  // namespace routine

namespace inv_detail {

// getrf's local slots first (the same offsets getrf carves), then the global copy (ld = n) and
// its pivots.
template <class T> struct InvPlan {
    getrf_detail::LuBlockedPlan<T> local;
    ws::Slot<T> lu;
    ws::Slot<std::int32_t> ipiv;
    WorkspaceSize size;
};
template <class T, class Cfg> constexpr InvPlan<T> inv_plan(int n) {
    Plan p;
    const auto local = getrf_detail::lu_blocked_take<T, Cfg>(p);
    const auto lu = p.take<T>(ws::Space::Global, static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    const auto ipiv = p.take<std::int32_t>(ws::Space::Global, static_cast<std::size_t>(n));
    return {local, lu, ipiv, p.size()};
}

}  // namespace inv_detail

/// inv's Blocked<NB>: any n; getrf's Blocked executor and local plan, global n * n + n pivots.
template <int NB, class Alg, class T> struct tier_info<routine::inv, tier::Blocked<NB, Alg>, T> {
    using cfg = getrf_detail::lu_blocked_cfg_t<T, tier::Blocked<NB, Alg>>;
    static_assert(std::is_same_v<typename cfg::value_type, T>, "inv: tier::Blocked's l3 configuration is for another scalar type");
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return inv_detail::inv_plan<T, cfg>(n).size; }
};
template <class T> struct tiers<routine::inv, T> {
    using type = tier::list<tier::Blocked<32>>;
};

/// The workspace of one n x n inversion for inv<Tier> (host and device; constexpr).
template <class Tier, class T> constexpr WorkspaceSize inv_buffer_size(int n) {
    return tier_info<routine::inv, resolve_tier_t<routine::inv, T, Tier>, T>::buffer_size(n);
}

/// @brief The host inv in a kernel: \f$A_{\mathrm{inv}} := A^{-1}\f$, A not modified.
///
/// Collective as getrf; A visible to all items on entry, Ainv on return.
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A and Ainv n x n of any size
///   in global memory, not aliasing; `ws.local` and `ws.global` (this matrix's slice) of at
///   least inv_buffer_size<Tier, T>(n) bytes each.
/// @param[in]  ex    the executor
/// @param[in]  a     this matrix; read only
/// @param[out] ainv  this matrix's inverse (not meaningful for info > 0)
/// @param[in]  ws    this matrix's workspace slice
/// @param[in]  live  false: a dead matrix of the launch (executor-uniform)
/// @return getrf's info, uniform across `ex` (the host inv reports none)
template <class Tier, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE std::int32_t inv(const Ex& ex, const MA& a, const MB& ainv, const Workspace& ws, bool live = true) {
    using D = getrf_detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::inv, D, Tier>;
    static_assert(HasTier<routine::inv, Tr, D>, "inv: this tier is not offered for this scalar type");
    using Cfg = getrf_detail::lu_blocked_cfg_t<D, Tr>;
    static_assert(Ex::size == Cfg::threads, "inv: tier::Blocked runs on a WorkGroup of tier_info::wg_size items");
    if (!live) return 0;
    const auto ag = l3::global_of(a);
    const int n = ag.n;
    const auto pl = inv_detail::inv_plan<D, Cfg>(n);
    const auto w = mat_wide(carve_global(ws, pl.lu), n, n, n);
    std::int32_t* const ipiv = carve_global(ws, pl.ipiv);
    lacpy(ex, mat_wide(ag.p, n, n, ag.ld), w);
    ex.barrier(Mem::Global);
    const std::int32_t info = getrf<Tr>(ex, w, ipiv, ws);
    (void)getri<Tr>(ex, w, ainv, ipiv, ws);
    return info;
}

}  // namespace batchlas::device
