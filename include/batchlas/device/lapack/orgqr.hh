// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xORGQR / xUNGQR in a kernel: A := the first n columns of Q = H_1 ... H_n from geqrf's
// storage, as the host blocked route forms it (src/ops/orgqr/kernels/orgqr_blocked.cc): ormqr
// (Left, NoTrans) applied to an identity in the caller's global workspace, then copied over A.
//   tier::Blocked<NB, Alg>  any m >= n; Q, V, T and larfb's W1 / W2 in the global workspace

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larfb.hh>
#include <batchlas/device/lapack/ormqr.hh>

namespace batchlas::device {

namespace routine {
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

template <int NB, class A, class Scalar> struct tier_info<routine::orgqr, tier::Blocked<NB, A>, Scalar> {
    using cfg = larfb_cfg_t<A, Scalar>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return detail::orgqr_size<NB, cfg, Scalar>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
template <class Scalar> struct tiers<routine::orgqr, Scalar> {
    using type = tier::list<tier::Blocked<larfb_default_nb<Scalar>()>>;
};

/// @brief ?ORGQR / ?UNGQR in a kernel: a (m x n, m >= n) := the first n columns of Q.
///
/// a holds geqrf's n reflectors below its diagonal on entry (its diagonal and upper triangle are
/// not read) and Q, with orthonormal columns, on return. Collective over a
/// WorkGroup<tier_info<...>::wg_size, 32>; a and tau visible on entry, a visible on return.
/// @param ex   the executor
/// @param a    this matrix (l3::Global, a global Mat or device::item), overwritten with Q
/// @param tau  n scalars, a unit-stride vector operand (device::vec)
/// @param ws   ws.local >= local_bytes, ws.global this matrix's slice of global_bytes of
///             buffer_size<routine::orgqr, Tier, T>(m, n)
/// @return 0, or -i for a bad i-th argument after the executor (1: m < n, 2: tau shorter than n,
///         3: no global workspace), uniform
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
