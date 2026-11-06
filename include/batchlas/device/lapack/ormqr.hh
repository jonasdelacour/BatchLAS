// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xORMQR / xUNMQR in a kernel: C := op(Q) C or C op(Q), Q = H_1 ... H_k of geqrf's storage,
// as the host blocked route does it (src/ops/ormqr/kernels/ormqr_blocked.cc): per NB-wide block in
// application order, pack V, larft, larfb on the in-group Level 3. One matrix per work-group.
//   tier::Blocked<NB, Alg>  any m, n; V, T and larfb's W1 / W2 in the caller's global workspace,
//                           the l3 stage in its local memory; Alg: an l3::cfg preset or Default

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larfb.hh>

namespace batchlas::device::inline v2 {

namespace routine {
template <Side SD, Transpose TR> struct ormqr {
    static constexpr const char* name = "ormqr";
};
}  // namespace routine

namespace ormqr_detail {
template <class T> struct OrmqrSlots {
    ws::Slot<T> smem, v, t, work;
};
// ormqr's carve for C (m x n), k reflectors, block NB, appended to p (orgqr appends to its own).
template <Side SD, int NB, class Cfg, class T> constexpr OrmqrSlots<T> ormqr_take(Plan& p, int m, int n) {
    const std::size_t nq = static_cast<std::size_t>(SD == Side::Left ? m : n);
    OrmqrSlots<T> s{};
    s.smem = p.take<T>(ws::Space::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
    s.v = p.take<T>(ws::Space::Global, nq * NB);
    s.t = p.take<T>(ws::Space::Global, static_cast<std::size_t>(NB) * NB);
    s.work = p.take<T>(ws::Space::Global, larfb_work_elems<SD>(m, n, NB));
    return s;
}
template <Side SD, int NB, class Cfg, class T> constexpr WorkspaceSize ormqr_size(int m, int n) {
    Plan p;
    (void)ormqr_take<SD, NB, Cfg, T>(p, m, n);
    return p.size();
}

template <class Tier> inline constexpr bool ormqr_is_blocked_v = false;
template <int NB, class A> inline constexpr bool ormqr_is_blocked_v<tier::Blocked<NB, A>> = true;

// The blocks in application order: forward for Q^H C and C Q, backward for Q C and C Q^H.
template <Side SD, Transpose TR, int NB, class Cfg, class Ex, class T>
BATCHLAS_DLA_INLINE void ormqr_wy_blocks(const Ex& ex, const l3::Global<T>& a, const l3::Global<T>& c, int k, const T* tau,
                                         const OrmqrSlots<T>& s, const Workspace& w) {
    const l3::Smem<Cfg> sm{carve_local(w, s.smem).get_raw()};
    T* const vb = carve_global(w, s.v);
    T* const tb = carve_global(w, s.t);
    T* const wb = carve_global(w, s.work);
    constexpr bool forward = (SD == Side::Left) == (TR != Transpose::NoTrans);
    // i0 itself is the induction variable (a downward block count narrows to i27 on Metal).
    const int first = forward ? 0 : (k - 1) / NB * NB;
    for (int i0 = first; forward ? i0 < k : i0 >= 0; i0 += forward ? NB : -NB) {
        const int ib = k - i0 < NB ? k - i0 : NB;
        const l3::Global<T> cs = SD == Side::Left ? l3::Global<T>{c.p + i0, c.ld, c.m - i0, c.n}
                                                  : l3::Global<T>{c.p + static_cast<std::ptrdiff_t>(i0) * c.ld, c.ld, c.m, c.n - i0};
        larfb_from_panel<SD, TR, Cfg>(ex, a, i0, ib, tau, cs, vb, tb, NB, wb, sm);
    }
}
}  // namespace ormqr_detail

template <Side SD, Transpose TR, int NB, class A, class T> struct tier_info<routine::ormqr<SD, TR>, tier::Blocked<NB, A>, T> {
    using cfg = larfb_cfg_t<A, T>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    /// C of m x n (Q of order m for Left, n for Right); k never changes the workspace.
    static constexpr WorkspaceSize buffer_size(int m, int n) { return ormqr_detail::ormqr_size<SD, NB, cfg, T>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
template <Side SD, Transpose TR, class T> struct tiers<routine::ormqr<SD, TR>, T> {
    using type = tier::list<tier::Blocked<larfb_default_nb<T>()>>;
};

/// The workspace of one matrix for ormqr<SD, TR, Tier> with C of m x n (host and device; constexpr).
template <Side SD, Transpose TR, class Tier, class T> constexpr WorkspaceSize ormqr_buffer_size(int m, int n) {
    return tier_info<routine::ormqr<SD, TR>, resolve_tier_t<routine::ormqr<SD, TR>, T, Tier>, T>::buffer_size(m, n);
}

/// @brief ?ORMQR / ?UNMQR in a kernel: C := op(Q) C (Left) or C op(Q) (Right) for one matrix.
///
/// A holds geqrf's k = min(A.rows(), A.cols()) reflectors below its diagonal, Q = H_1 ... H_k of
/// order A.rows() (= C.rows() for Left, C.cols() for Right); op(Q) = Q (NoTrans) or Q^H
/// (ConjTrans; Trans on a real T). Collective over a WorkGroup<tier_info<...>::wg_size, 32>; A, C
/// and tau visible on entry, C visible on return; A and tau are not written. One tier per kernel
/// (with_tier<routine::ormqr<SD, TR>, T> on the host).
/// @param ex   the executor
/// @param a    this matrix's reflectors (l3::Global, a global Mat or device::item)
/// @param c    this matrix's C, overwritten
/// @param tau  k scalars: a T* or a unit-stride vector operand
/// @param ws   ws.local >= local_bytes, ws.global this matrix's slice of global_bytes of
///             ormqr_buffer_size<SD, TR, Tier, T>(C.rows(), C.cols())
/// @return 0, or -i for a bad i-th argument after the executor (1: A.rows() is not Q's order,
///         3: tau shorter than k, 4: no global workspace), uniform
template <Side SD, Transpose TR, class Tier, class Ex, class MA, class MC, class Tau>
BATCHLAS_DLA_INLINE int ormqr(const Ex& ex, const MA& A, const MC& C, const Tau& tau, const Workspace& ws) {
    using T = std::remove_cvref_t<decltype(A(0, 0))>;
    using Tr = resolve_tier_t<routine::ormqr<SD, TR>, T, Tier>;
    static_assert(HasTier<routine::ormqr<SD, TR>, Tr, T>, "ormqr: this tier is not offered for this scalar type");
    static_assert(ormqr_detail::ormqr_is_blocked_v<Tr>, "ormqr: only tier::Blocked<NB> is offered");
    static_assert(!(Arith<T>::is_complex && TR == Transpose::Trans), "ormqr: complex Trans; use ConjTrans");
    using Cfg = larfb_cfg_t<typename Tr::alg_type, T>;
    larfb_check_ex<Cfg, Ex>();
    const auto a = l3::global_of(A);
    const auto c = l3::global_of(C);
    const int k = a.m < a.n ? a.m : a.n;
    if (a.m != (SD == Side::Left ? c.m : c.n)) return -1;
    if constexpr (requires { tau.size(); }) {
        if (static_cast<int>(tau.size()) < k) return -3;
    }
    if (ws.global == nullptr) return -4;
    if (k <= 0 || c.m <= 0 || c.n <= 0) return 0;
    Plan p;
    const auto s = ormqr_detail::ormqr_take<SD, Tr::nb, Cfg, T>(p, c.m, c.n);
    ormqr_detail::ormqr_wy_blocks<SD, TR, Tr::nb, Cfg>(ex, a, c, k, static_cast<const T*>(larfb_tau_base(tau)), s, ws);
    return 0;
}

}  // namespace batchlas::device
