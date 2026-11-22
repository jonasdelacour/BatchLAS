// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_chol
/// @brief ?POSV in a kernel: the Cholesky factorization of one Hermitian positive definite matrix,
/// then the solve with it, in one call on one workspace.
///
/// The routine is posv(); its routine tag is routine::posv. Its one tier is `tier::Blocked<32>`
/// (`tier::Blocked<16>` for `std::complex<double>`): potrf()'s Blocked tier, then potrs()'s, on
/// potrf()'s workspace plan.
// The small tiers land with WP7.

#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/lapack/potrs.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace detail {
template <Uplo UL, int NB, class Alg, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE int posv_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    static_assert(std::is_same_v<Alg, alg::Default>, "posv: tier::Blocked has one algorithm");
    static_assert(std::is_same_v<typename MA::value_type, typename MB::value_type>, "posv: A and B hold the same scalar type");
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (b.rows() != n || b.cols() < 0) return -3;
    if (n == 0) return 0;
    if (!detail::blocked_ws_ok(ws, false)) return -4;  // potrf would report it as its -3
    const int info = detail::potrf_blocked_op<UL, NB, Alg>(ex, a, ws);
    if (info != 0 || b.cols() == 0) return info;
    detail::potrs_blocked_solve<UL, NB>(ex, a, b, ws);
    return 0;
}
}  // namespace detail

namespace routine {
/// @ingroup api_device_lapack_chol
/// @brief The routine tag of posv() for the triangle `UL`: names the routine to `with_tier`,
/// `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Uplo UL> struct posv {
    static constexpr const char* name = "posv";  ///< The routine's name, "posv".
    static constexpr Uplo uplo = UL;             ///< The triangle that is read and factored.
};
}  // namespace routine

/// @ingroup api_device_lapack_chol
/// @brief posv()'s blocked tier `tier::Blocked<NB>`: any n and nrhs, one matrix per 256-item
/// work-group, `A` and `B` in global memory.
///
/// The executor is `WorkGroup<256, 32>`. `admits(n)` is n >= 0 when NB <= 32 (16 for
/// `std::complex<double>`), false otherwise. `buffer_size(n)` is potrf()'s Blocked plan: local
/// memory only, independent of n and nrhs; potrs() reuses its first slot.
template <Uplo UL, int NB, class Alg, class Scalar> struct tier_info<routine::posv<UL>, tier::Blocked<NB, Alg>, Scalar> {
    static constexpr int wg_size = detail::kBlockedWg;
    static constexpr int sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0 && NB <= detail::reg_max_n<Scalar>; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::PotrfBlockedPlan<Scalar, NB>::make().size; }
};
/// @ingroup api_device_lapack_chol
/// @brief posv()'s tiers for `Scalar`: `Blocked<32>`, or `Blocked<16>` for `std::complex<double>`.
template <Uplo UL, class Scalar> struct tiers<routine::posv<UL>, Scalar> {
    using type = tier::list<tier::Blocked<(detail::reg_max_n<Scalar> >= 32 ? 32 : 16)>>;
};

/// @ingroup api_device_lapack_chol
/// @brief ?POSV in a kernel: `A` := its Cholesky factor, then \f$B := A^{-1} B\f$ when the
/// factorization succeeded.
///
/// The `UL` triangle of `A` is read and overwritten with the factor, as potrf() does; the other
/// triangle is neither read nor written. When info is 0, the n x nrhs `B` is overwritten with
/// the solution X; otherwise `B` is unchanged.
/// - `tier::Blocked<NB>`: `ex` a `WorkGroup<256, 32>`; any n and nrhs; `A` and `B` in global
///   memory. The small tiers are not offered yet.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::posv<UL>, T>` that admits N.
///
/// Workspace: `ws` is this matrix's slice of `buffer_size<routine::posv<UL>, Tier, T>(n)`: local
/// memory only, 16-byte aligned.
///
/// @pre Collective: every item of `ex` calls posv() with the same arguments; `A` and `B` are
/// visible to every item on entry and on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle to factor
/// @tparam Tier  `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex  the executor that owns this matrix
/// @param[in,out] A   the n x n matrix; its `UL` triangle is overwritten with the factor
/// @param[in,out] B   the n x nrhs right-hand sides; overwritten with X when info is 0
/// @param[in]     ws  this matrix's workspace slice
/// @return LAPACK's info, the same on every item of `ex`:
/// @retval 0   success: `A` holds the factor and `B` holds X
/// @retval k   (k > 0) the leading minor of order k is not positive definite; `B` is unchanged
/// @retval -2  `A` is not square
/// @retval -3  `B` does not have n rows
/// @retval -4  `ws.local` is null or not 16-byte aligned
template <Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int posv(const Ex& ex, const MA& A, const MB& B, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::posv<UL>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "posv: tier::Blocked<NB> (the small tiers land with WP7)");
    return detail::posv_blocked_op<UL, Tr::nb, typename Tr::alg_type>(ex, A, B, ws);
}

}  // namespace batchlas::device
