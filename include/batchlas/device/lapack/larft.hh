// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The triangular factor of a block reflector inside a kernel: `larft`, LAPACK's xLARFT
/// (forward, columnwise).
///
/// `larft(ex, v, tau, t)` forms the upper triangular T of \f$H = I - V T V^H\f$ from a unit lower
/// V and its reflector scalars, on one work-group; `larfb` applies H. Reference:
/// @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// detail holds the one piece core has no spelling for yet (a distributed out-of-place upper trmv,
// one item per row).

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device {

namespace detail {

// y := A x, A upper (strict lower never read), y not aliasing x. Item r forms y(r) from row r:
// y is Cyclic, so a later call reading y(r) as A(r, :) runs on its writer (no barrier).
template <class Ops = arith::Explicit, class Ex, class MA, class X, class Y>
BATCHLAS_DEVICE_INLINE void trmv_upper_row_owner(const Ex& ex, const MA& a, const X& xp, const Y& y) {
    using T = typename MA::value_type;
    using A = arith_t<Ops, T>;
    detail::require_published<Ex, X>();
    const auto& x = detail::unpub(xp);
    const int n = a.rows();
    for_each_owned(ex, n, [&](int r) BATCHLAS_DEVICE_BODY {
        T acc = A::zero();
        A::mac(acc, a(r, r), x(r));
#pragma unroll 2  // 4 costs FP64 an occupancy step (REG 40 -> 48 in larft)
        for (int c = r + 1; c < n; ++c) A::mac(acc, a(r, c), x(c));
        y(r) = acc;
    });
}

}  // namespace detail

/// @brief xLARFT (Direct = Forward, Store = Columnwise) inside a kernel: the ib x ib upper
/// triangular `t` of the block reflector \f$H = H_0 H_1 \cdots H_{ib-1} = I - V T V^H\f$.
///
/// `v` is m x ib, unit lower trapezoidal with its unit diagonal stored (the part above the
/// diagonal is not read); `tau(j)` is reflector j's scalar. `t` receives T in its upper triangle and zeros in its strict
/// lower triangle; a zero `tau(j)` gives a zero column above `t(j, j)`. Netlib's argument order
/// (V, TAU, T). Collective over `ex`, every item with the same arguments; `v` and `tau` visible to
/// every item on entry. `t` is written by its owners: barrier (`Mem::Global`) before another item
/// reads it.
/// @pre `ex` is a `WorkGroup<WG, 32>` and the kernel runs at sub-group size 32 (its gemv packs
///      over 32 lanes).
/// @param[in]  ex   the executor
/// @param[in]  v    the m x ib reflector vectors, unit diagonal stored
/// @param[in]  tau  the ib reflector scalars
/// @param[out] t    the ib x ib triangular factor
/// @ingroup api_device_lapack_qr
// Two barriers: Y_j = -tau_j V^H v_j parks in t's strict lower row j; the trmv owner reads back
// only its rows; the scratch is cleared last.
template <class Ex, class MV, class VT, class MT>
BATCHLAS_DEVICE_INLINE void larft(const Ex& ex, const MV& v, const VT& tau, const MT& t) {
    using T = typename MT::value_type;
    using A = Arith<T>;
    const int m = v.rows(), ib = v.cols();
    for (int j = 1; j < ib; ++j) {
        const T tj = tau(j);
        if (A::is_zero(tj)) continue;
        gemv<Transpose::ConjTrans>(ex, A::neg(tj), v.sub(j, 0, m - j, j), assume_published(v.col(j).tail(j)), Zero{},
                                   t.row(j).head(j));
    }
    ex.barrier(Mem::Global);

    using DiagVec = Vec<T, Dyn, Dyn, MT::mem, Cyclic, typename MT::base_type, typename MT::index_type>;
    copy(ex, tau, DiagVec{t.p, Dyn{ib}, Dyn{int(t.ld) + 1}, t.off});  // t(j, j) := tau_j
    for (int j = 1; j < ib; ++j) {  // t(0:j, j) := t(0:j, 0:j) Y_j, or 0 when tau_j == 0
        const auto t_col = t.col(j).head(j);
        if (A::is_zero(static_cast<T>(tau(j)))) map(ex, t_col, [] { return A::zero(); });
        else detail::trmv_upper_row_owner(ex, t.sub(0, 0, j, j), assume_published(t.row(j).head(j)), t_col);
    }
    ex.barrier(Mem::Global);

    laset<Part::StrictLower>(ex, Zero{}, Zero{}, t);
}

}  // namespace batchlas::device
