// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xPOSV inside a kernel (v22 design C.2, C.4-C.7): A = L L^H (Lower) or U^H U (Upper) in
// place, then B := A^-1 B when the factorization succeeded. Public entry:
//   int posv<UL, Tier>(ex, A, B, ws)           potrf's info (uniform over ex), or -i
//   WorkspaceSize posv_buffer_size<UL, Tier, T>(n)
//   tier::Blocked<NB>  ex = WorkGroup<256, 32>, any n and nrhs: potrf's then potrs's Blocked tier
//                      on one workspace (potrf's plan, whose first slot potrs carves)
// The small tiers land with WP7. info = k > 0: the leading minor of order k is not positive
// definite, B unchanged; -2: A not square; -3: B's rows differ from n; -4: ws.local bad.

#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/lapack/potrs.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

namespace posv_detail {
template <Uplo UL, int NB, class Alg, int WG, int SG, class MA, class MB>
BATCHLAS_DEVICE_INLINE int posv_blocked_op(const WorkGroup<WG, SG>& ex, const MA& a, const MB& b, const Workspace& ws) {
    static_assert(std::is_same_v<Alg, alg::Default>, "posv: tier::Blocked has one algorithm");
    static_assert(std::is_same_v<typename MA::value_type, typename MB::value_type>, "posv: A and B hold the same scalar type");
    const int n = a.rows();
    if (n < 0 || a.cols() != n) return -2;
    if (b.rows() != n || b.cols() < 0) return -3;
    if (n == 0) return 0;
    if (!potrf_detail::blocked_ws_ok(ws, false)) return -4;  // potrf would report it as its -3
    const int info = potrf_detail::potrf_blocked_op<UL, NB, Alg>(ex, a, ws);
    if (info != 0 || b.cols() == 0) return info;
    potrs_detail::potrs_blocked_solve<UL, NB>(ex, a, b, ws);
    return 0;
}
}  // namespace posv_detail

namespace routine {
template <Uplo UL> struct posv {
    static constexpr const char* name = "posv";
    static constexpr Uplo uplo = UL;
};
}  // namespace routine

template <Uplo UL, int NB, class Alg, class T> struct tier_info<routine::posv<UL>, tier::Blocked<NB, Alg>, T> {
    static constexpr int wg_size = potrf_detail::kBlockedWg;
    static constexpr int sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && NB <= potrf_detail::reg_max_n<T>; }
    static constexpr WorkspaceSize buffer_size(int) { return potrf_detail::BlockedPlan<T, NB>::make().size; }
};
template <Uplo UL, class T> struct tiers<routine::posv<UL>, T> {
    using type = tier::list<tier::Blocked<(potrf_detail::reg_max_n<T> >= 32 ? 32 : 16)>>;
};

// The workspace of one matrix (host and device); nrhs does not enter it. Tier may be tier::upto<NMax>.
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize posv_buffer_size(int n) {
    using Tr = resolve_tier_t<routine::posv<UL>, T, Tier>;
    return tier_info<routine::posv<UL>, Tr, T>::buffer_size(n);
}

// A := its Cholesky factor (UL triangle read and overwritten, the other neither read nor written),
// then B := A^-1 B for the n x nrhs B; returns info (see the file head).
template <Uplo UL, class Tier, class Ex, class MA, class MB>
BATCHLAS_DEVICE_INLINE int posv(const Ex& ex, const MA& A, const MB& B, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::posv<UL>, T, Tier>;
    static_assert(potrf_detail::is_blocked_v<Tr>, "posv: tier::Blocked<NB> (the small tiers land with WP7)");
    return posv_detail::posv_blocked_op<UL, Tr::nb, typename Tr::alg_type>(ex, A, B, ws);
}

}  // namespace batchlas::device
