// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Forming Q inside a kernel: `orgqr`, LAPACK's xORGQR / xUNGQR on one matrix per
/// work-group.
///
/// `orgqr<Tier>(ex, a, tau, ws)` overwrites `geqrf`'s storage with the first n columns of
/// \f$Q = H_0 H_1 \cdots H_{n-1}\f$, for any m >= n, with `tier::Blocked<NB, Alg>`. Reference:
/// @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// As the host blocked route forms it (src/ops/orgqr/kernels/orgqr_blocked.cc): ormqr (Left,
// NoTrans) applied to an identity in the caller's global workspace, then copied over A. Q, V, T
// and larfb's W1 / W2 in the global workspace.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larfb.hh>
#include <batchlas/device/lapack/ormqr.hh>

namespace batchlas::device {

namespace routine {
/// @brief The routine tag of `orgqr`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_qr
struct orgqr {
    static constexpr const char* name = "orgqr";
};
}  // namespace routine

namespace detail {
template <class Scalar> struct OrgqrSlots {
    WsSlot<Scalar> q;
    detail::OrmqrSlots<Scalar> apply;
};
template <int NB, class Cfg, class Scalar> constexpr OrgqrSlots<Scalar> orgqr_take(Plan& p, int m, int n) {
    OrgqrSlots<Scalar> s{};
    s.q = p.take<Scalar>(WsSpace::Global,
                         static_cast<std::size_t>(m > 0 ? m : 0) * static_cast<std::size_t>(n > 0 ? n : 0));
    s.apply = detail::ormqr_take<Side::Left, NB, Cfg, Scalar>(p, m, n);
    return s;
}
template <int NB, class Cfg, class Scalar> constexpr WorkspaceSize orgqr_size(int m, int n) {
    Plan p;
    (void)orgqr_take<NB, Cfg, Scalar>(p, m, n);
    return p.size();
}
}  // namespace detail

/// @brief `orgqr` in `tier::Blocked<NB, A>`: one m x n matrix (m >= n) per work-group.
///
/// The executor is `WorkGroup<cfg::threads, 32>` with `cfg = larfb_cfg_t<A, Scalar>`, as for
/// `geqrf`; the kernel must run at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(sg_size)`).
/// `buffer_size(m, n)` is one matrix's local and global bytes (the global part holds an m x n Q
/// besides `ormqr`'s V, T and work); `buffer_size(n)` is the square case.
/// @ingroup api_device_lapack_qr
template <int NB, class A, class Scalar> struct tier_info<routine::orgqr, tier::Blocked<NB, A>, Scalar> {
    using cfg = larfb_cfg_t<A, Scalar>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return detail::orgqr_size<NB, cfg, Scalar>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
/// @brief The tiers `with_tier` tries for `orgqr`: `tier::Blocked<larfb_default_nb<Scalar>()>`.
/// @ingroup api_device_lapack_qr
template <class Scalar> struct tiers<routine::orgqr, Scalar> {
    using type = tier::list<tier::Blocked<larfb_default_nb<Scalar>()>>;
};

/// @brief xORGQR / xUNGQR inside a kernel: the m x n `a` (m >= n) := the first n columns of Q.
///
/// On entry `a` holds the n reflectors `geqrf` stored below its diagonal (the diagonal and upper
/// triangle are not read), and `tau` their scalars; on return `a` holds Q's first n columns,
/// which are orthonormal. Collective over `tier_info<routine::orgqr, Tier, T>::executor`, every
/// item with the same arguments; `a` and `tau` visible to every item on entry, `a` visible on
/// return. One tier per kernel (`with_tier<routine::orgqr, T>` on the host).
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam Tier  `tier::Blocked<NB, Alg>`, the only tier
/// @tparam Ex    the executor type, `WorkGroup<cfg::threads, 32>`
/// @tparam MA    a global matrix operand: `l3::Global`, a `Mat` over global memory, or
///               `device::item`
/// @tparam Tau   a unit-stride vector operand (`device::vec`), not a pointer
/// @param[in]     ex   the executor
/// @param[in,out] a    this matrix: the reflectors on entry, Q on return
/// @param[in]     tau  at least n scalars
/// @param[in]     ws   `ws.local` at least `local_bytes` and `ws.global` this matrix's slice of
///                     `global_bytes`, both of `buffer_size<routine::orgqr, Tier, T>(m, n)`
/// @return `info`, the same on every item
/// @retval 0   success, including n = 0
/// @retval -1  m < n (nothing written)
/// @retval -2  `tau` is shorter than n (nothing written)
/// @retval -3  `ws.global` is null (nothing written)
/// @ingroup api_device_lapack_qr
template <class Tier, class Ex, class MA, class Tau>
BATCHLAS_DEVICE_INLINE int orgqr(const Ex& ex, const MA& a, const Tau& tau, const Workspace& ws) {
    using T = std::remove_cvref_t<decltype(a(0, 0))>;
    using Tr = resolve_tier_t<routine::orgqr, T, Tier>;
    static_assert(HasTier<routine::orgqr, Tr, T>, "orgqr: this tier is not offered for this scalar type");
    static_assert(tier::is_blocked_v<Tr>, "orgqr: only tier::Blocked<NB> is offered");
    static_assert(!std::is_pointer_v<Tau>, "orgqr: tau is a vector operand (device::vec)");
    using Cfg = larfb_cfg_t<typename Tr::alg_type, T>;
    detail::larfb_check_ex<Cfg, Ex>();
    const auto ag = l3::global_of(a);
    const int m = ag.m, n = ag.n;
    if (m < n) return -1;
    if (static_cast<int>(tau.size()) < n) return -2;
    if (ws.global == nullptr) return -3;
    if (n <= 0) return 0;
    Plan p;
    const auto s = detail::orgqr_take<Tr::nb, Cfg, T>(p, m, n);
    T* const qb = carve_global(ws, s.q);
    const l3::Global<T> q{qb, m, m, n};
    laset(ex, Zero{}, One{}, mat(qb, m, n, m));
    ex.barrier(Mem::Global);  // the identity complete before the first block stages it
    detail::ormqr_wy_blocks<Side::Left, Transpose::NoTrans, Tr::nb, Cfg>(ex, ag, q, n, static_cast<const T*>(&tau(0)), s.apply, ws);
    lacpy(ex, mat(qb, m, n, m), mat(ag.p, m, n, ag.ld));
    ex.barrier(Mem::Global);  // Q visible on return
    return 0;
}

}  // namespace batchlas::device
