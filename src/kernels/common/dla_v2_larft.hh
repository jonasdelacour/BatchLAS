// SPDX-License-Identifier: MIT
#pragma once
// Family-private v2 ops of larft and ormqr_cta, forms core lacks (larf on a Frag; a distributed
// trmv) and larft's T recurrence. Everything else these kernels use comes from core.

#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/device.hh>

namespace batchlas::device::inline v2 {

// x := H x, H = I - tau v v^H (conj(tau) for H^H), x a register column one item owns; fully
// unrolled so x stays in registers. v Published. Native: the hand loop's arithmetic (strict).
template <class Ops = arith::Explicit, class T, int N, class L, class V>
BATCHLAS_DLA_INLINE void larf_frag(const Thread&, const T& tau, const V& vp, Frag<T, N, L>& x) {
    using A = arith_t<Ops, T>;
    static_assert(is_published_v<V>, "v2::larf_frag: v is read whole; pass assume_published(v) after the barrier that staged it");
    const auto& v = detail::unpub(vp);
    T w = A::zero();
BATCHLAS_UNROLL_FULL
    for (int i = 0; i < N; ++i) A::mac_conj(w, static_cast<T>(v(i)), x[i]);
    const T g = A::mul(tau, w);
BATCHLAS_UNROLL_FULL
    for (int i = 0; i < N; ++i) A::msub(x[i], static_cast<T>(v(i)), g);
}

// y := A x, A upper (strict lower never read), y not aliasing x. Item r forms y(r) from row r:
// y is Cyclic, so a later call reading y(r) as A(r, :) runs on its writer (no barrier).
template <class Ops = arith::Explicit, class Ex, class MA, class X, class Y>
BATCHLAS_DLA_INLINE void trmv_upper_row_owner(const Ex& ex, const MA& a, const X& xp, const Y& y) {
    using T = typename MA::value_type;
    using A = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const int n = a.rows();
    for_each_owned(ex, n, [&](int r) BATCHLAS_DLA_BODY {
        T acc = A::zero();
        A::mac(acc, a(r, r), x(r));
#pragma unroll 2  // 4 costs FP64 an occupancy step (REG 40 -> 48 in larft)
        for (int c = r + 1; c < n; ++c) A::mac(acc, a(r, c), x(c));
        y(r) = acc;
    });
}

// larft (Forward, Columnwise), V's unit diagonal stored. Two barriers: Y_j = -tau_j V^H v_j parks
// in T's strict lower row j; the trmv owner reads back only its rows; the scratch is cleared last.
template <class Ex, class MV, class MT, class VT>
BATCHLAS_DLA_INLINE void larft_forward_colwise(const Ex& ex, const MV& V, const MT& Tm, const VT& tau) {
    using T = typename MT::value_type;
    using A = Arith<T>;
    const int m = V.rows(), ib = V.cols();
    for (int j = 1; j < ib; ++j) {
        const T tj = tau(j);
        if (A::is_zero(tj)) continue;
        gemv<Transpose::ConjTrans>(ex, A::neg(tj), V.sub(j, 0, m - j, j), assume_published(V.col(j).tail(j)), Zero{},
                                   Tm.row(j).head(j));
    }
    ex.barrier(Mem::Global);

    using DiagVec = Vec<T, Dyn, Dyn, MT::mem, Cyclic, typename MT::base_type, typename MT::index_type>;
    copy(ex, tau, DiagVec{Tm.p, Dyn{ib}, Dyn{int(Tm.ld) + 1}, Tm.off});  // T(j, j) := tau_j
    for (int j = 1; j < ib; ++j) {  // T(0:j, j) := T(0:j, 0:j) Y_j, or 0 when tau_j == 0
        const auto t_col = Tm.col(j).head(j);
        if (A::is_zero(static_cast<T>(tau(j)))) map(ex, t_col, [] { return A::zero(); });
        else trmv_upper_row_owner(ex, Tm.sub(0, 0, j, j), assume_published(Tm.row(j).head(j)), t_col);
    }
    ex.barrier(Mem::Global);

    laset<Part::StrictLower>(ex, Zero{}, Zero{}, Tm);
}

}  // namespace batchlas::device
