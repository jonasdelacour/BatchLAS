// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Applying Q inside a kernel: `ormqr`, LAPACK's xORMQR / xUNMQR on one matrix per
/// work-group.
///
/// `ormqr<SD, TR, Tier>(ex, a, tau, c, ws)` computes \f$\mathrm{op}(Q)\,C\f$ or
/// \f$C\,\mathrm{op}(Q)\f$ for the Q of `geqrf`'s storage, any m and n, with
/// `tier::Blocked<NB, Alg>`. Reference: @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// As the host blocked route does it (src/ops/ormqr/kernels/ormqr_blocked.cc): per NB-wide block in
// application order, pack V, larft, larfb on the in-group Level 3. V, T and larfb's W1 / W2 in the
// caller's global workspace, the l3 stage in its local memory.

#include <cstddef>
#include <type_traits>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/larfb.hh>

namespace batchlas::device {

namespace routine {
/// @brief The routine tag of `ormqr<SD, TR>`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_qr
template <Side SD, Transpose TR> struct ormqr {
    static constexpr const char* name = "ormqr";
};
}  // namespace routine

namespace detail {
template <class Scalar> struct OrmqrSlots {
    WsSlot<Scalar> smem, v, t, work;
};
// ormqr's carve for C (m x n), k reflectors, block NB, appended to p (orgqr appends to its own).
template <Side SD, int NB, class Cfg, class Scalar> constexpr OrmqrSlots<Scalar> ormqr_take(Plan& p, int m, int n) {
    const std::size_t nq = static_cast<std::size_t>(SD == Side::Left ? m : n);
    OrmqrSlots<Scalar> s{};
    s.smem = p.take<Scalar>(WsSpace::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
    s.v = p.take<Scalar>(WsSpace::Global, nq * NB);
    s.t = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(NB) * NB);
    s.work = p.take<Scalar>(WsSpace::Global, larfb_work_elems<SD>(m, n, NB));
    return s;
}
template <Side SD, int NB, class Cfg, class Scalar> constexpr WorkspaceSize ormqr_size(int m, int n) {
    Plan p;
    (void)ormqr_take<SD, NB, Cfg, Scalar>(p, m, n);
    return p.size();
}

// The blocks in application order: forward for Q^H C and C Q, backward for Q C and C Q^H.
template <Side SD, Transpose TR, int NB, class Cfg, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE void ormqr_wy_blocks(const Ex& ex, const l3::Global<Scalar>& a, const l3::Global<Scalar>& c, int k, const Scalar* tau,
                                         const OrmqrSlots<Scalar>& s, const Workspace& w) {
    const Workspace sm = sub_workspace(w, s.smem, s.work);
    Scalar* const vb = carve_global(w, s.v);
    Scalar* const tb = carve_global(w, s.t);
    constexpr bool forward = (SD == Side::Left) == (TR != Transpose::NoTrans);
    // i0 itself is the induction variable (a downward block count narrows to i27 on Metal).
    const int first = forward ? 0 : (k - 1) / NB * NB;
    for (int i0 = first; forward ? i0 < k : i0 >= 0; i0 += forward ? NB : -NB) {
        const int ib = k - i0 < NB ? k - i0 : NB;
        const l3::Global<Scalar> cs = SD == Side::Left ? l3::Global<Scalar>{c.p + i0, c.ld, c.m - i0, c.n}
                                                  : l3::Global<Scalar>{c.p + static_cast<std::ptrdiff_t>(i0) * c.ld, c.ld, c.m, c.n - i0};
        larfb_from_panel<SD, TR, Cfg>(ex, a, i0, ib, tau, cs, vb, tb, NB, sm);
    }
}
}  // namespace detail

/// @brief `ormqr<SD, TR>` in `tier::Blocked<NB, A>`: one C (any m x n) per work-group.
///
/// The executor is `WorkGroup<cfg::threads, 32>` with `cfg = larfb_cfg_t<A, Scalar>`, as for
/// `geqrf`; the kernel must run at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(sg_size)`).
/// `buffer_size(n)` is the square case of `buffer_size(m, n)`.
/// @ingroup api_device_lapack_qr
template <Side SD, Transpose TR, int NB, class A, class Scalar> struct tier_info<routine::ormqr<SD, TR>, tier::Blocked<NB, A>, Scalar> {
    using cfg = larfb_cfg_t<A, Scalar>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    /// @brief One matrix's local and global bytes for a C of m x n (Q of order m for Left, n for
    /// Right); the number of reflectors k does not change it.
    static constexpr WorkspaceSize buffer_size(int m, int n) { return detail::ormqr_size<SD, NB, cfg, Scalar>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
/// @brief The tiers `with_tier` tries for `ormqr<SD, TR>`: `tier::Blocked<larfb_default_nb<Scalar>()>`.
/// @ingroup api_device_lapack_qr
template <Side SD, Transpose TR, class Scalar> struct tiers<routine::ormqr<SD, TR>, Scalar> {
    using type = tier::list<tier::Blocked<larfb_default_nb<Scalar>()>>;
};

/// @brief xORMQR / xUNMQR inside a kernel: \f$C \leftarrow \mathrm{op}(Q)\,C\f$ (`Side::Left`)
/// or \f$C \leftarrow C\,\mathrm{op}(Q)\f$ (`Side::Right`) for one matrix.
///
/// `a` holds the k = min(a.rows(), a.cols()) reflectors `geqrf` stored below its diagonal, and
/// \f$Q = H_0 H_1 \cdots H_{k-1}\f$ has order a.rows(), which must be c.rows() (Left) or c.cols()
/// (Right). op(Q) is Q for `Transpose::NoTrans` and \f$Q^H\f$ for `Transpose::ConjTrans`
/// (`Trans` on a real type only). Collective over the tier's executor,
/// `tier_info<routine::ormqr<SD, TR>, Tier, T>::executor`, every item with the same arguments;
/// `a`, `tau` and `c` visible to every item on entry, `c` visible on return; `a` and `tau` are not
/// written. One tier per kernel (`with_tier<routine::ormqr<SD, TR>, T>` on the host).
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam SD    `Side::Left` or `Side::Right`
/// @tparam TR    `Transpose::NoTrans`, `Transpose::ConjTrans`, or `Transpose::Trans` for a real
///               type
/// @tparam Tier  `tier::Blocked<NB, Alg>`, the only tier
/// @tparam MA    a global matrix operand: `l3::Global`, a `Mat` over global memory, or
///               `device::item`
/// @tparam Tau   a unit-stride vector operand (`device::vec`), not a pointer
/// @tparam MC    a global matrix operand, as `MA`
/// @param[in]     ex   the executor
/// @param[in]     a    this matrix's reflectors
/// @param[in]     tau  at least k scalars
/// @param[in,out] c    this matrix's C, overwritten with the product
/// @param[in]     ws   `ws.local` at least `local_bytes` and `ws.global` this matrix's slice of
///                     `global_bytes`, both of
///                     `buffer_size<routine::ormqr<SD, TR>, Tier, T>(c.rows(), c.cols())`
/// @return `info`, the same on every item
/// @retval 0   success, including k = 0 or an empty C
/// @retval -1  a.rows() is not Q's order (nothing written)
/// @retval -2  `tau` is shorter than k (nothing written)
/// @retval -4  `ws.global` is null (nothing written)
/// @ingroup api_device_lapack_qr
template <Side SD, Transpose TR, class Tier, class Ex, class MA, class Tau, class MC>
BATCHLAS_DEVICE_INLINE int ormqr(const Ex& ex, const MA& a, const Tau& tau, const MC& c, const Workspace& ws) {
    using T = std::remove_cvref_t<decltype(a(0, 0))>;
    using Tr = resolve_tier_t<routine::ormqr<SD, TR>, T, Tier>;
    static_assert(HasTier<routine::ormqr<SD, TR>, Tr, T>, "ormqr: this tier is not offered for this scalar type");
    static_assert(tier::is_blocked_v<Tr>, "ormqr: only tier::Blocked<NB> is offered");
    static_assert(!(Arith<T>::is_complex && TR == Transpose::Trans), "ormqr: complex Trans; use ConjTrans");
    static_assert(!std::is_pointer_v<Tau>, "ormqr: tau is a vector operand (device::vec)");
    using Cfg = larfb_cfg_t<typename Tr::alg_type, T>;
    detail::larfb_check_ex<Cfg, Ex>();
    const auto ag = l3::global_of(a);
    const auto cg = l3::global_of(c);
    const int k = ag.m < ag.n ? ag.m : ag.n;
    if (ag.m != (SD == Side::Left ? cg.m : cg.n)) return -1;
    if (static_cast<int>(tau.size()) < k) return -2;
    if (ws.global == nullptr) return -4;
    if (k <= 0 || cg.m <= 0 || cg.n <= 0) return 0;
    Plan p;
    const auto s = detail::ormqr_take<SD, Tr::nb, Cfg, T>(p, cg.m, cg.n);
    detail::ormqr_wy_blocks<SD, TR, Tr::nb, Cfg>(ex, ag, cg, k, static_cast<const T*>(&tau(0)), s, ws);
    return 0;
}

}  // namespace batchlas::device
