// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_lu
/// @brief ?GETRI in a kernel: the inverse of one matrix from getrf()'s factor and pivots, out of
/// place.
///
/// The routine is getri(); its routine tag is routine::getri. Its one tier is `tier::Blocked<32>`
/// (any n), with getrf()'s Blocked executor and local workspace plan.
// The host blocked route's steps (getri_blocked.cc): C := the permutation getrf applied, then
// C := U^-1 L^-1 C on the in-group trsm.

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
/// @ingroup api_device_lapack_lu
/// @brief The routine tag of getri(): names the routine to `with_tier`, `launch`, `tier_info`,
/// `tiers_t` and `buffer_size`.
struct getri {
    static constexpr const char* name = "getri";  ///< The routine's name, "getri".
};
}  // namespace routine

namespace detail {

// C := P (C(i, perm(i)) = 1, zero elsewhere; row i walked back through the interchanges, the
// host route's formula) and the first exact zero U(i, i), 1-based, or 0. Every item scans the
// diagonal, so the result is uniform with no reduction. Ends on a global barrier.
template <class Ex, class MA, class P, class MC>
BATCHLAS_DEVICE_INLINE int permute_identity(const Ex& ex, const MA& a, const P& ipiv, const MC& c) {
    using D = typename MC::value_type;
    const int n = c.rows();
    laset(ex, Zero{}, Zero{}, c);
    ex.barrier(Mem::Global);
    for_each_owned(ex, n, [&](int i) BATCHLAS_DEVICE_BODY { c(i, laswp_index<Direct::Backward>(ipiv, 0, n, i)) = Arith<D>::one(); });
    ex.barrier(Mem::Global);
    int info = 0;
    for (int i = 0; i < n; ++i)
        if (info == 0 && Arith<D>::is_zero(a(i, i))) info = i + 1;
    return info;
}

}  // namespace detail

/// @ingroup api_device_lapack_lu
/// @brief getri()'s blocked tier `tier::Blocked<NB>`: any n; getrf()'s Blocked executor and local
/// plan, no global workspace.
template <int NB, class Alg, class Scalar>
struct tier_info<routine::getri, tier::Blocked<NB, Alg>, Scalar> : tier_info<routine::getrf, tier::Blocked<NB, Alg>, Scalar> {};
/// @ingroup api_device_lapack_lu
/// @brief getri()'s tiers for every `Scalar`: `Blocked<32>`.
template <class Scalar> struct tiers<routine::getri, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @ingroup api_device_lapack_lu
/// @brief ?GETRI in a kernel: \f$C := A^{-1}\f$ from getrf's factor of A (A not modified).
///
/// As the host blocked route, the solves run whatever info says: for info > 0, C is not
/// meaningful. No argument is checked.
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A and C n x n of any size in
///   global memory, C not aliasing A; `ws.local` >= buffer_size<routine::getri, Tier, T>(n) bytes,
///   no global workspace.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::getri, T>` that admits N.
///
/// @pre Collective as getrf(); A and ipiv visible to all items on entry, C visible to all on
/// return.
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]  ex    the executor
/// @param[in]  a     getrf's L and U of this matrix
/// @param[in]  ipiv  getrf's n pivots, 1-based (a Vec<int>)
/// @param[out] c     this matrix's inverse
/// @param[in]  ws    this matrix's workspace slice
/// @param[in]  live  false: a dead matrix of the launch (executor-uniform)
/// @return LAPACK's info, uniform across `ex`:
/// @retval 0  success, or `live` false
/// @retval k  (k > 0) \f$U_{kk}\f$ is exactly zero (the first such k): A is singular and C is not
///            meaningful
template <class Tier, class Ex, class MA, class PV, class MC>
BATCHLAS_DEVICE_INLINE int getri(const Ex& ex, const MA& a, const PV& ipiv, const MC& c, const Workspace& ws = {}, bool live = true) {
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
    const int info = detail::permute_identity(ex, mat_wide(ag.p, n, n, ag.ld), ipiv, mat_wide(cg.p, n, n, cg.ld));
    detail::lu_trsm_pair<Transpose::NoTrans, Tr::nb, Cfg>(ex, ag, cg, sub_workspace(ws, pl.smem));
    return info;
}

}  // namespace batchlas::device
