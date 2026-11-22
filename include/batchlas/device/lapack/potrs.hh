// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_chol
/// @brief ?POTRS in a kernel: the solve \f$A X = B\f$ for one matrix from potrf()'s Cholesky
/// factor, X overwriting B.
///
/// The routine is potrs(); its routine tag is routine::potrs. Its one tier is `tier::Blocked<32>`:
/// the two triangular solves run on the in-group level 3 trsm over NB-wide diagonal blocks.
// The small tiers (the host posv tiny and potrs fused bodies) land with WP7.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace detail {

// The Blocked plan: the L3 stage slot alone, at potrf's offset.
template <class Scalar> struct BlockedSolvePlan {
    WsSlot<Scalar> smem;
    WorkspaceSize size;
    static constexpr BlockedSolvePlan make() {
        Plan p;
        const auto s = p.take<Scalar>(WsSpace::Local,
                                      static_cast<std::size_t>(detail::PotrfBlockedCfg<Scalar>::smem_elems));
        return {s, p.size()};
    }
};
static_assert(BlockedSolvePlan<float>::make().smem.offset == detail::PotrfBlockedPlan<float, 32>::make().smem.offset);

// B := A^-1 B from the factor: Lower L (L^H X = Y after L Y = B), Upper U^H then U. Arguments
// already checked; n, nrhs >= 1.
template <Uplo UL, int NB, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE void potrs_blocked_solve(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    using T = typename MA::value_type;
    using Cfg = detail::PotrfBlockedCfg<T>;
    constexpr BlockedSolvePlan<T> pl = BlockedSolvePlan<T>::make();
    const Workspace sm = sub_workspace(ws, pl.smem);
    const int n = a.rows(), nrhs = b.cols();
    const auto A = mat(&a(0, 0), n, n, static_cast<int>(a.ld));
    const auto B = mat(&b(0, 0), n, nrhs, static_cast<int>(b.ld));
    if constexpr (UL == Uplo::Lower) {
        trsm<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit, Cfg, NB>(ex, One{}, A, B, sm);
        trsm<Side::Left, Uplo::Lower, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, One{}, A, B, sm);
    } else {
        trsm<Side::Left, Uplo::Upper, Transpose::ConjTrans, Diag::NonUnit, Cfg, NB>(ex, One{}, A, B, sm);
        trsm<Side::Left, Uplo::Upper, Transpose::NoTrans, Diag::NonUnit, Cfg, NB>(ex, One{}, A, B, sm);
    }
}

// The argument checks potrs and posv share (posv's ws is also argument 4).
template <class MA, class MB> BATCHLAS_DEVICE_INLINE int potrs_blocked_args(const MA& a, const MB& b, const Workspace& ws) {
    static_assert(std::is_same_v<typename MA::value_type, typename MB::value_type>, "potrs: A and B hold the same scalar type");
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (b.rows() != n || b.cols() < 0) return -3;
    if (n > 0 && b.cols() > 0 && !detail::blocked_ws_ok(ws, false)) return -4;
    return 0;
}

template <Uplo UL, int NB, class Alg, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE int potrs_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    using T = typename MA::value_type;
    static_assert(std::is_same_v<Alg, alg::Default>, "potrs: tier::Blocked has one algorithm");
    static_assert(NB >= 1, "potrs: tier::Blocked<NB> needs NB >= 1");
    static_assert(WG == detail::PotrfBlockedCfg<T>::threads && SG == 32, "potrs: tier::Blocked runs on WorkGroup<256, 32>");
    const int bad = potrs_blocked_args(a, b, ws);
    if (bad != 0) return bad;
    if (a.rows() == 0 || b.cols() == 0) return 0;
    potrs_blocked_solve<UL, NB>(ex, a, b, ws);
    return 0;
}
}  // namespace detail

namespace routine {
/// @ingroup api_device_lapack_chol
/// @brief The routine tag of potrs() for the triangle `UL`: names the routine to `with_tier`,
/// `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Uplo UL> struct potrs {
    static constexpr const char* name = "potrs";  ///< The routine's name, "potrs".
    static constexpr Uplo uplo = UL;              ///< The triangle that holds the factor.
};
}  // namespace routine

/// @ingroup api_device_lapack_chol
/// @brief potrs()'s blocked tier `tier::Blocked<NB>`: any n and nrhs, one matrix per 256-item
/// work-group, `A` and `B` in global memory.
///
/// The executor is `WorkGroup<256, 32>`. `admits(n)` is n >= 0. `buffer_size(n)` is the level 3
/// stage slot alone: local memory only, independent of n and nrhs (the same slot potrf()'s
/// Blocked plan starts with).
template <Uplo UL, int NB, class Alg, class Scalar> struct tier_info<routine::potrs<UL>, tier::Blocked<NB, Alg>, Scalar> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::BlockedSolvePlan<Scalar>::make().size; }
};
/// @ingroup api_device_lapack_chol
/// @brief potrs()'s tiers for every `Scalar`: `Blocked<32>`.
template <Uplo UL, class Scalar> struct tiers<routine::potrs<UL>, Scalar> {
    using type = tier::list<tier::Blocked<32>>;
};

/// @ingroup api_device_lapack_chol
/// @brief ?POTRS in a kernel: \f$B := A^{-1} B\f$ for one matrix, from potrf()'s factor in the
/// `UL` triangle of `A`.
///
/// `A`'s `UL` triangle holds \f$L\f$ (`Uplo::Lower`, \f$A = L L^H\f$) or \f$U\f$ (`Uplo::Upper`,
/// \f$A = U^H U\f$); the other triangle is not read and `A` is not written. The n x nrhs `B` is
/// overwritten with the solution X.
/// - `tier::Blocked<NB>`: `ex` a `WorkGroup<256, 32>`; any n and nrhs; `A` and `B` in global
///   memory. The small tiers are not offered yet.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::potrs<UL>, T>` that admits N.
///
/// Workspace: `ws` is this matrix's slice of `buffer_size<routine::potrs<UL>, Tier, T>(n)`: local
/// memory only, 16-byte aligned.
///
/// @pre Collective: every item of `ex` calls potrs() with the same arguments; `A` and `B` are
/// visible to every item on entry, and X is visible to every item on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle that holds the factor
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the executor that owns this matrix
/// @param[in]     A   potrf()'s factor of the n x n matrix
/// @param[in,out] B   the n x nrhs right-hand sides; overwritten with X
/// @param[in]     ws  this matrix's workspace slice
/// @return 0 or a negative argument number, the same on every item of `ex`:
/// @retval 0   success
/// @retval -2  `A` is not square
/// @retval -3  `B` does not have n rows
/// @retval -4  `ws.local` is null or not 16-byte aligned (checked when n and nrhs are positive)
template <Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int potrs(const Ex& ex, const MA& A, const MB& B, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::potrs<UL>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "potrs: tier::Blocked<NB> (the small tiers land with WP7)");
    return detail::potrs_blocked_op<UL, Tr::nb, typename Tr::alg_type>(ex, A, B, ws);
}

}  // namespace batchlas::device
