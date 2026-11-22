// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_ldl
/// @brief ?SYSV in a kernel: the Bunch-Kaufman factorization of one symmetric indefinite matrix,
/// then the solve with it, in one call.
///
/// The routine is sysv(); its routine tag is routine::sysv. Its tiers are `tier::Cta<32>` (both
/// halves fused in one local-memory tile) and `tier::Blocked<1>` (sytrf()'s then sytrs()'s
/// `tier::Blocked<1>`); `tier::Blocked<NB>` with NB > 1 is offered but not in `tiers_t`.
// The host route sysv `cta` launches the Cta tier's two bodies; sysv `compose` runs sytrf `wg` and
// sytrs `wg` (Blocked<1>) or sytrf `blocked` and sytrs `blocked` (Blocked<NB>).
// Tests: tests/device_lapack_sytrf_tests.cc.

#include <cstddef>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/lapack/sytrf.hh>
#include <batchlas/device/lapack/sytrs.hh>

namespace batchlas::device {

namespace routine {
/// @ingroup api_device_lapack_ldl
/// @brief The routine tag of sysv() for the triangle `UL`: names the routine to `with_tier`,
/// `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Uplo UL> struct sysv {
    static_assert(UL == Uplo::Lower || UL == Uplo::Upper, "sysv: Uplo::Lower or Uplo::Upper");
    static constexpr const char* name = "sysv";  ///< The routine's name, "sysv".
};
}  // namespace routine

namespace detail {

// The fused tile of sysv `cta`: the (N + 1) x N factor then e (N); the int part holds ipiv's
// copy [0, N), ps, perm, kind (N each), the validity flag at 4N and the factor's info at 4N + 1.
template <int N, class Scalar> struct SysvCtaPlan {
    WsSlot<Scalar> tile;
    WsSlot<int> idx;
    WorkspaceSize size;
};
template <int N, class Scalar> constexpr SysvCtaPlan<N, Scalar> sysv_cta_plan() {
    Plan p;
    const auto tile = p.take<Scalar>(WsSpace::Local, static_cast<std::size_t>(N + 1) * N + N);
    const auto idx = p.take<int>(WsSpace::Local, 4 * static_cast<std::size_t>(N) + 2);
    return {tile, idx, p.size()};
}

template <Uplo UL, int N, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE int sysv_run(tier::Cta<N>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using R = real_t<D>;
    constexpr bool rev = UL == Uplo::Upper;
    constexpr int ldt = N + 1;
    constexpr auto pl = sysv_cta_plan<N, D>();
    const int n = A.rows();
    D* const tp = carve_local(ws, pl.tile).get_raw();
    int* const pp = carve_local(ws, pl.idx).get_raw();
    const SubGroup<32> g{ex.sub()};
    detail::sytrf_cta_run(
        ex, g, detail::sytrf_team(ex) == 0, detail::sytrf_view(A, rev), detail::rev_mat(tp, ldt, n, false), n, rev,
        argmax_slots<Mem::Global, true>(static_cast<R*>(nullptr), static_cast<int*>(nullptr)),
        [&](int k, int v) {
            ipiv(k) = v;
            pp[k] = v;
        },
        [&](int st) { pp[4 * N + 1] = st; });
    sync<Scope::Group>(ex, Mem::Global);
    const int info = pp[4 * N + 1];
    if (info == 0) {
        const detail::Staged<D> L{tp, ldt, n, tp + ldt * N, pp + N, pp + 2 * N, pp + 3 * N};
        detail::sytrs_cta_run<1>(ex, ex.sub(), ex.lane(), detail::sytrf_team(ex), detail::sytrf_nteams(ex), L, pp,
                                         rev, pp + 4 * N, &B(0, 0), detail::sytrf_ld(B), B.cols());
    }
    sync<Scope::Group>(ex, Mem::Global);
    return info;
}

template <Uplo UL, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE int sysv_run(tier::Blocked<1>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    // Both halves carve the same workspace from offset 0: sytrf's ends on a barrier.
    const int info = detail::sytrf_run<UL>(tier::Blocked<1>{}, ex, A, ipiv, ws);
    if (info != 0) return info;
    detail::sytrs_run<UL>(tier::Blocked<1>{}, ex, A, ipiv, B, ws);
    return 0;
}

template <Uplo UL, int NB, class Ex, class MA, class PV, class MB>
    requires(NB > 1)
BATCHLAS_DEVICE_INLINE int sysv_run(tier::Blocked<NB>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    // As Blocked<1>: both halves carve from offset 0 of both spaces; sytrf's ends on a barrier.
    const int info = detail::sytrf_run<UL>(tier::Blocked<NB>{}, ex, A, ipiv, ws);
    if (info != 0) return info;
    detail::sytrs_run<UL>(tier::Blocked<NB>{}, ex, A, ipiv, B, ws);
    return 0;
}

constexpr WorkspaceSize sysv_max(const WorkspaceSize& a, const WorkspaceSize& b) {
    return {a.local_bytes > b.local_bytes ? a.local_bytes : b.local_bytes,
            a.global_bytes > b.global_bytes ? a.global_bytes : b.global_bytes, a.align > b.align ? a.align : b.align};
}

}  // namespace detail

/// @ingroup api_device_lapack_ldl
/// @brief sysv()'s tiers for every `Scalar`, smallest first: `Cta<32>`, `Blocked<1>`.
/// `Blocked<NB>` with NB > 1 is offered but not listed.
template <Uplo UL, class Scalar> struct tiers<routine::sysv<UL>, Scalar> {
    using type = tier::list<tier::Cta<32>, tier::Blocked<1>>;
};
/// @ingroup api_device_lapack_ldl
/// @brief sysv()'s local-memory tier `tier::Cta<N>`, N <= 32: sytrf()'s factor on sub-group 0 and
/// sytrs()'s converted solve, fused in one local-memory tile.
///
/// The executor is `WorkGroup<32, 32>`. `admits(n)` is 0 <= n <= N. `buffer_size(n, nrhs)` is
/// local memory only and depends on neither.
template <Uplo UL, int N, class Scalar> struct tier_info<routine::sysv<UL>, tier::Cta<N>, Scalar> {
    static_assert(N <= 32, "sysv tier::Cta<N>: N <= 32 (one sub-group factors)");
    static constexpr int wg_size = 32, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int, int = 1) { return detail::sysv_cta_plan<N, Scalar>().size; }
};
/// @ingroup api_device_lapack_ldl
/// @brief sysv()'s unblocked tier `tier::Blocked<1>`: any n, sytrf()'s then sytrs()'s
/// `tier::Blocked<1>` on one workspace.
///
/// The executor is `WorkGroup<256, 32>` (any executor of whole 32-wide sub-groups, at most 256
/// items). `admits(n)` is n >= 0. `buffer_size(n, nrhs)` is the larger of the two halves' plans:
/// local memory only, independent of n and nrhs.
template <Uplo UL, class Scalar> struct tier_info<routine::sysv<UL>, tier::Blocked<1>, Scalar> {
    static constexpr int wg_size = detail::sytrf_wg_max, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int, int = 1) {
        return detail::sysv_max(detail::sytrf_wg_plan<real_t<Scalar>>().size,
                                detail::sytrs_wg_plan<real_t<Scalar>>().size);
    }
};
/// @ingroup api_device_lapack_ldl
/// @brief sysv()'s blocked tier `tier::Blocked<NB>`, NB > 1: any n, sytrf()'s then sytrs()'s
/// `tier::Blocked<NB>` on one workspace.
///
/// The executor is `WorkGroup<256, 32>`. `admits(n)` is n >= 0. `buffer_size(n, nrhs)` is the
/// larger of the two halves' plans, in both spaces; its global part grows with n and nrhs.
template <Uplo UL, int NB, class Scalar> struct tier_info<routine::sysv<UL>, tier::Blocked<NB>, Scalar> {
    static constexpr int wg_size = detail::sytrf_blocked_wg, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n, int nrhs = 1) {
        return detail::sysv_max(detail::sytrf_blocked_plan<NB, Scalar>(n).size,
                                detail::sytrs_blocked_plan<Scalar>(n, nrhs).size);
    }
};

/// @ingroup api_device_lapack_ldl
/// @brief ?SYSV in a kernel: `A` := sytrf()'s factor, `ipiv` := its pivots, and
/// \f$B := A^{-1} B\f$ when the factor is nonsingular, for one matrix.
///
/// When info is 0, the n x nrhs `B` is overwritten with the solution X; otherwise `B` is left as
/// given, as LAPACK's. The executor needs sub-group size 32 and must be whole 32-wide sub-groups.
/// - `tier::Cta<N>`, N <= 32: `ex` a `WorkGroup<32, 32>`; n <= N; both halves in one local tile.
/// - `tier::Blocked<1>`: any n; `ex` at most 256 items.
/// - `tier::Blocked<NB>`, NB > 1: any n; `ex` a `WorkGroup<256, 32>`. Not in `tiers_t`.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::sysv<UL>, T>` that admits N.
///
/// Workspace: `ws` holds `buffer_size<routine::sysv<UL>, Tier, T>(n, nrhs)`: its local bytes,
/// and for `tier::Blocked<NB>` its global bytes for this matrix alone.
///
/// @pre Collective as sytrf(): every item of `ex` calls with the same arguments; `A`, `ipiv` and
/// `B` visible to every item on entry. All three are visible to every item on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle to factor
/// @tparam Tier  `tier::Cta<N>`, `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex    the executor that owns this matrix
/// @param[in,out] A     the n x n symmetric matrix; its `UL` triangle is overwritten with the
///                      factor
/// @param[out]    ipiv  the n pivots (a `Vec<int>`), as sytrf()'s
/// @param[in,out] B     the n x nrhs right-hand sides; overwritten with X when info is 0
/// @param[in]     ws    this matrix's workspace slice
/// @return sytrf()'s info or a negative LAPACK argument number, the same on every item of `ex`.
/// The numbers are ?SYTRS's, as ?SYSV numbers them (N is 2, LDA 5, B 7, LDB 8), not positions in
/// this call:
/// @retval 0   success: `A` holds the factor and `B` holds X
/// @retval k   (k > 0) D(k, k) is exactly zero; `B` is unchanged
/// @retval -1  `ex` has a run-time width that is not whole 32-wide sub-groups, or is wider than
///             `tier::Blocked<1>`'s 256 items
/// @retval -2  `A` is not square, or n is outside the tier
/// @retval -5  `A`'s leading dimension is less than max(1, n)
/// @retval -7  `B` does not have n rows
/// @retval -8  `B`'s leading dimension is less than max(1, n)
template <Uplo UL, class Tier, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE int sysv(const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    static_assert(std::is_same_v<D, std::remove_cv_t<typename MB::value_type>>, "sysv: A and B of one scalar type");
    using Tr = resolve_tier_t<routine::sysv<UL>, D, Tier>;
    static_assert(detail::sytrf_exec_ok<Ex>(), "sysv: an executor of whole 32-wide sub-groups");
    const int chk = detail::sytrs_check<Tr>(ex, A, B);
    if (chk != 0 || A.rows() == 0) return chk;
    return detail::sysv_run<UL>(Tr{}, ex, A, ipiv, B, ws);
}

}  // namespace batchlas::device
