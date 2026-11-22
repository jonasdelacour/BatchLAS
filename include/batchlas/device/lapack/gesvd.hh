// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The singular value decomposition inside a kernel: `gesvd`, LAPACK's xGESVD on one
/// matrix per work-group.
///
/// `gesvd<JU, JVT, Tier>(ex, A, S, U, Vh, ws)` computes \f$A = U \Sigma V^H\f$ for any m x n with
/// `tier::Blocked<NB>`: the host blocked route's pipeline, `gebrd`, then `bdsqr` into
/// identity-seeded U and \f$V^H\f$, then `ormbr` with Q and P. Reference:
/// @ref api_device_lapack_svd.
/// @ingroup api_device_lapack_svd
// detail: the driver's own steps on the real upper bidiagonal B = (d, e) of gebrd: bdgram forms
// B^T B or B B^T (the gesvd-via-tridiagonal route), bdmv_div recovers a left singular vector
// B x / sigma, tridiag_abs_offdiag makes a tridiagonal's off-diagonal non-negative by a diagonal
// sign similarity (Thread / Grid1D / WorkGroup executors through core for_each_owned).
// m < n solves A^H and returns the adjoints.

#include <cstddef>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/bdsqr.hh>
#include <batchlas/device/lapack/gebrd.hh>
#include <batchlas/device/lapack/ormbr.hh>
#include <batchlas/device/launch.hh>
#include <batchlas/device/level3/group.hh>

namespace batchlas::device::detail {

// The symmetric tridiagonal Gram matrix of the real upper bidiagonal B = (d, e), k = d.size():
// Side::Right is B^T B (order k), Side::Left is B B^T padded with zero rows to order td.size().
template <Side SD, class Ex, class D, class E, class TD, class TE>
BATCHLAS_DEVICE_INLINE void bdgram(const Ex& ex, const D& d, const E& e, const TD& td, const TE& te) {
    using T = elem_t<TD>;
    static_assert(!Arith<T>::is_complex, "gesvd_detail::bdgram: the bidiagonal of gebrd is real");
    const int k = d.size();
    if constexpr (SD == Side::Right) {
        for_each_owned(ex, td.size(), [&](int i) {
            const T di = d(i);
            T diag = di * di;
            if (i > 0) {
                const T e_prev = e(i - 1);
                diag += e_prev * e_prev;
            }
            td(i) = diag;
            if (i < k - 1) te(i) = di * e(i);
        });
    } else {
        const int order = td.size();
        for_each_owned(ex, order, [&](int i) {
            T diag = T(0);
            if (i < k) {
                const T di = d(i);
                diag += di * di;
                if (i < k - 1) {
                    const T ei = e(i);
                    diag += ei * ei;
                    te(i) = ei * d(i + 1);
                } else if (i < order - 1) {
                    te(i) = T(0);
                }
            } else if (i < order - 1) {
                te(i) = T(0);
            }
            td(i) = diag;
        });
    }
}

// y := B x / sigma for the real upper bidiagonal B = (d, e), k = d.size(), with y(i) = 0 for
// k <= i < y.size(); y := 0 when sigma <= tol (a zero singular value has no vector B x / sigma).
template <class Ex, class D, class E, class X, class Real, class Y>
BATCHLAS_DEVICE_INLINE void bdmv_div(const Ex& ex, const D& d, const E& e, const X& x, Real sigma, Real tol,
                                     const Y& y) {
    using T = elem_t<Y>;
    const int k = d.size();
    for_each_owned(ex, y.size(), [&](int i) {
        T value = T(0);
        if (sigma > tol && i < k) {
            value = d(i) * x(i);
            if (i < k - 1) value += e(i) * x(i + 1);
            value /= sigma;
        }
        y(i) = value;
    });
}

// e := |e| and s := the signs with s(0) = 1 for which diag(s) T diag(s) has off-diagonal |e|
// (T real symmetric tridiagonal of order s.size()). A prefix product: Thread only.
template <class Ex, class E, class S> BATCHLAS_DEVICE_INLINE void tridiag_abs_offdiag(const Ex&, const E& e, const S& s) {
    static_assert(is_thread_v<Ex>, "gesvd_detail::tridiag_abs_offdiag: a sequential scan, Thread only");
    using T = elem_t<S>;
    const int n = s.size();
    s(0) = T(1);
    for (int i = 0; i < n - 1; ++i) {
        const T ei = e(i);
        const T sgn = (ei >= T(0)) ? T(1) : T(-1);
        s(i + 1) = s(i) * sgn;
        e(i) = sycl::fabs(ei);
    }
}

}  // namespace batchlas::device::detail

namespace batchlas::device {

namespace routine {
/// @brief The routine tag of `gesvd<JU, JVT>`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_svd
template <SvdVectors JU, SvdVectors JVT> struct gesvd {
    static constexpr const char* name = "gesvd";
};
}  // namespace routine

namespace detail {
// One matrix's workspace for tier::Blocked<NB>; the sub-ops (gebrd, bdsqr) get their own slices,
// each carved by that op's plan from offset 0 of its slice.
template <class Scalar> struct GesvdBlockedPlan {
    WsSlot<Scalar> at, ub, vb;           // m < n: A^H (n x m), its U (n x vh_rows), its V^H (m x m)
    WsSlot<real_t<Scalar>> d, e;         // the bidiagonal: k, k - 1
    WsSlot<Scalar> tauq, taup;           // k each
    WsSlot<real_t<Scalar>> ur, vr;       // complex with vectors: bdsqr's real k x k U and V^H
    WsSlot<std::byte> gebrd_local, gebrd_global, bdsqr_local, bdsqr_global;
    WorkspaceSize size;
};
template <SvdVectors JU, SvdVectors JVT, int NB, class Scalar> constexpr GesvdBlockedPlan<Scalar> driver_plan(int m,
                                                                                                              int n) {
    using R = real_t<Scalar>;
    constexpr bool wu = JU != SvdVectors::None, wv = JVT != SvdVectors::None;
    const bool flip = m < n;
    const int k = flip ? m : n, mm = flip ? n : m;
    const std::size_t vh_rows = JVT == SvdVectors::All ? std::size_t(n) : std::size_t(k);
    const std::size_t kk = Arith<Scalar>::is_complex && (wu || wv) ? std::size_t(k) * std::size_t(k) : 0;
    Plan p;
    GesvdBlockedPlan<Scalar> b{};
    b.at = p.take<Scalar>(WsSpace::Global, flip ? std::size_t(n) * std::size_t(m) : 0);
    b.ub = p.take<Scalar>(WsSpace::Global, flip && wv ? std::size_t(n) * vh_rows : 0);
    b.vb = p.take<Scalar>(WsSpace::Global, flip && wu ? std::size_t(m) * std::size_t(m) : 0);
    b.d = p.take<R>(WsSpace::Global, std::size_t(k));
    b.e = p.take<R>(WsSpace::Global, std::size_t(k > 1 ? k - 1 : 1));
    b.tauq = p.take<Scalar>(WsSpace::Global, std::size_t(k));
    b.taup = p.take<Scalar>(WsSpace::Global, std::size_t(k));
    b.ur = p.take<R>(WsSpace::Global, kk);
    b.vr = p.take<R>(WsSpace::Global, kk);
    const WorkspaceSize g = device::buffer_size<routine::gebrd, tier::Blocked<NB>, Scalar>(mm, k);
    const WorkspaceSize q = device::buffer_size<routine::bdsqr, tier::Blocked<1>, R>(k);
    b.gebrd_local = p.take<std::byte>(WsSpace::Local, g.local_bytes);
    b.gebrd_global = p.take<std::byte>(WsSpace::Global, g.global_bytes);
    b.bdsqr_local = p.take<std::byte>(WsSpace::Local, q.local_bytes);
    b.bdsqr_global = p.take<std::byte>(WsSpace::Global, q.global_bytes);
    b.size = p.size();
    return b;
}
// The workspace a nested op carves from: its local and global slices of this matrix's.
BATCHLAS_DEVICE_INLINE Workspace nested_workspace(const Workspace& w, const WsSlot<std::byte>& l, const WsSlot<std::byte>& g) {
    return {w.local + static_cast<std::ptrdiff_t>(l.offset), w.global + g.offset};
}
// A Mat over T* (int32 index, Dyn ld) of a global matrix operand (Mat, ItemMat or l3::Global).
template <class M> BATCHLAS_DEVICE_INLINE auto flat_mat(const M& a) {
    const auto g = l3::global_of(a);
    return device::mat(g.p, g.m, g.n, g.ld);
}

// The m >= n problem: A (m x k) -> d, e, tauq, taup; bdsqr into the identity-seeded leading k x k
// of U (m x U.cols()) and V^H (k x k); then U := Q U and V^H := V^H P^H.
template <int NB, class Scalar, class Ex, class MA, class SV>
BATCHLAS_DEVICE_INLINE int gesvd_tall(const Ex& ex, const MA& A, const SV& S, const MA& U, const MA& Vh, bool want_u, bool want_vh,
                                   const Workspace& ws, const GesvdBlockedPlan<Scalar>& P) {
    const int k = A.cols();
    const auto d = device::vec(carve_global(ws, P.d), k);
    const auto e = device::vec(carve_global(ws, P.e), k > 1 ? k - 1 : 0);
    const auto tauq = device::vec(carve_global(ws, P.tauq), k);
    const auto taup = device::vec(carve_global(ws, P.taup), k);
    (void)gebrd<tier::Blocked<NB>>(ex, A, d, e, tauq, taup, nested_workspace(ws, P.gebrd_local, P.gebrd_global));
    const Workspace bws = nested_workspace(ws, P.bdsqr_local, P.bdsqr_global);
    if (!want_u && !want_vh) return bdsqr<tier::Blocked<1>>(ex, d, e, S, bws);
    if (want_u) laset(ex, Zero{}, One{}, U);
    if (want_vh) laset(ex, Zero{}, One{}, Vh);
    int info = 0;
    if constexpr (Arith<Scalar>::is_complex) {
        // B is real: bdsqr accumulates into real k x k blocks, widened into U and V^H after.
        const int ku = want_u ? k : 0, kv = want_vh ? k : 0;
        const auto ur = device::mat(carve_global(ws, P.ur), ku, ku, k);
        const auto vr = device::mat(carve_global(ws, P.vr), kv, kv, k);
        laset(ex, Zero{}, One{}, ur);
        laset(ex, Zero{}, One{}, vr);
        ex.barrier(Mem::Global);
        info = bdsqr<tier::Blocked<1>>(ex, d, e, S, ur, vr, bws);
        if (want_u) lacpy(ex, ur, U.sub(0, 0, k, k));
        if (want_vh) lacpy(ex, vr, Vh.sub(0, 0, k, k));
    } else {
        ex.barrier(Mem::Global);
        info = bdsqr<tier::Blocked<1>>(ex, d, e, S, U.sub(0, 0, want_u ? k : 0, want_u ? k : 0),
                                       Vh.sub(0, 0, want_vh ? k : 0, want_vh ? k : 0), bws);
    }
    ex.barrier(Mem::Global);
    if (want_u) (void)ormbr<Vect::Q, Side::Left, Transpose::NoTrans>(ex, A, tauq, U, Workspace{});
    if (want_vh) (void)ormbr<Vect::P, Side::Right, Transpose::ConjTrans>(ex, A, taup, Vh, Workspace{});
    return info;
}
}  // namespace detail

/// @brief The tiers `with_tier` tries for `gesvd<JU, JVT>`: `tier::Blocked<16>`.
/// @ingroup api_device_lapack_svd
template <SvdVectors JU, SvdVectors JVT, class Scalar> struct tiers<routine::gesvd<JU, JVT>, Scalar> {
    using type = tier::list<tier::Blocked<detail::blocked_nb>>;
};
/// @brief `gesvd<JU, JVT>` in `tier::Blocked<NB>` (NB: `gebrd`'s panel width): any m x n, every
/// scalar type, one matrix per `WorkGroup<256, 32>`.
///
/// The kernel must run at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(sg_size)`).
/// `buffer_size(n)` is the square problem's local and global bytes, `buffer_size(m, n)` any
/// other; they include the nested `gebrd` and `bdsqr` slices.
/// @ingroup api_device_lapack_svd
template <SvdVectors JU, SvdVectors JVT, int NB, class A, class Scalar> struct tier_info<routine::gesvd<JU, JVT>, tier::Blocked<NB, A>, Scalar> {
    static constexpr int wg_size = detail::blocked_wg, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::driver_plan<JU, JVT, NB, Scalar>(n, n).size; }
    static constexpr WorkspaceSize buffer_size(int m,
                                               int n) { return detail::driver_plan<JU, JVT, NB, Scalar>(m, n).size; }
};

/// @brief xGESVD inside a kernel: \f$A = U \Sigma V^H\f$ of one m x n matrix (any m, n >= 1).
///
/// `S` receives the k = min(m, n) singular values, descending; `U_` the m x m (`SvdVectors::All`)
/// or m x k (`SvdVectors::Thin`) left vectors as columns; `Vh_` the n x n (All) or k x n (Thin)
/// right vectors as rows. A `SvdVectors::None` job's operand is not touched: pass an empty
/// `device::mat`. `A_` is overwritten. Arguments in the host order (A, S, U, Vh, ws), with jobu
/// and jobvt as template parameters.
///
/// Collective over a `WorkGroup<256, 32>`, every item with the same arguments; `A_` visible to
/// every item on entry, outputs visible on return.
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam JU    the left vectors: `SvdVectors::None`, `Thin` or `All`
/// @tparam JVT   the right vectors: `SvdVectors::None`, `Thin` or `All`
/// @tparam Tier  `tier::Blocked<NB>`, the only tier
/// @tparam MA    a global matrix operand: a `Mat` over global memory, `device::item` (an
///               `ItemMat`) or `l3::Global`
/// @param[in]     ex   the executor
/// @param[in,out] A_   the matrix; overwritten
/// @param[out]    S    at least k reals, the singular values
/// @param[out]    U_   the left singular vectors, a global matrix operand as `A_`
/// @param[out]    Vh_  the right singular vectors as rows, a global matrix operand as `A_`
/// @param[in]     ws   `ws.local` and this matrix's `ws.global` slice of
///                     `device::buffer_size<routine::gesvd<JU, JVT>, Tier, T>(m, n)`
/// @return `info`, the same on every item (the host blocked route's values)
/// @retval 0   success
/// @retval 1   `bdsqr` did not converge, or a singular value is NaN
/// @retval -2  `S` holds fewer than k values (nothing written)
/// @retval -3  `U_` has the wrong shape for JU (nothing written)
/// @retval -4  `Vh_` has the wrong shape for JVT (nothing written)
/// @ingroup api_device_lapack_svd
template <SvdVectors JU, SvdVectors JVT, class Tier, class Ex, class MA, class SV, class MU, class MV>
BATCHLAS_DEVICE_INLINE int gesvd(const Ex& ex, const MA& A_, const SV& S, const MU& U_, const MV& Vh_, const Workspace& ws) {
    using T = typename MA::value_type;
    using Tr = resolve_tier_t<routine::gesvd<JU, JVT>, T, Tier>;
    static_assert(tier::is_blocked_v<Tr>, "gesvd: tier::Blocked<NB>");
    constexpr bool wu = JU != SvdVectors::None, wv = JVT != SvdVectors::None;
    const auto A = detail::flat_mat(A_);
    const int m = A.rows(), n = A.cols(), k = m < n ? m : n;
    const auto U = detail::flat_mat(U_);
    const auto Vh = detail::flat_mat(Vh_);
    if (S.size() < k) return -2;
    if (wu && (U.rows() != m || U.cols() != (JU == SvdVectors::All ? m : k))) return -3;
    if (wv && (Vh.rows() != (JVT == SvdVectors::All ? n : k) || Vh.cols() != n)) return -4;
    const auto P = detail::driver_plan<JU, JVT, Tr::nb, T>(m, n);
    if (m >= n) return detail::gesvd_tall<Tr::nb, T>(ex, A, S, U, Vh, wu, wv, ws, P);
    // A^H = U' S V'^H: U = V' and V^H = U'^H.
    const auto At = device::mat(carve_global(ws, P.at), n, m, n);
    lacpy<Part::Full, Transpose::ConjTrans>(ex, A, At);
    ex.barrier(Mem::Global);
    const auto Ub = device::mat(carve_global(ws, P.ub), n, wv ? Vh.rows() : 0, n);
    const auto Vb = device::mat(carve_global(ws, P.vb), wu ? m : 0, wu ? m : 0, m);
    const int info = detail::gesvd_tall<Tr::nb, T>(ex, At, S, Ub, Vb, wv, wu, ws, P);
    if (wu) lacpy<Part::Full, Transpose::ConjTrans>(ex, Vb, U);
    if (wv) lacpy<Part::Full, Transpose::ConjTrans>(ex, Ub, Vh);
    ex.barrier(Mem::Global);
    return info;
}

}  // namespace batchlas::device
