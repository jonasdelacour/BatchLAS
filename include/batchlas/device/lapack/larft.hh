// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xLARFT (Direct = Forward, Store = Columnwise), the only larft: T of the block reflector
// H = I - V T V^H from a unit-lower V and tau, on one work-group. larft_detail holds the one
// piece core has no spelling for yet (a distributed out-of-place upper trmv, one item per row).

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device {

namespace larft_detail {

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

}  // namespace larft_detail

// T (ib x ib, upper) from V (m x ib, unit diagonal stored) and tau(0:ib). Two barriers: Y_j =
// -tau_j V^H v_j parks in T's strict lower row j; the trmv owner reads back only its rows; the
// scratch is cleared last. tau_j == 0 gives a zero column above T(j, j).
template <class Ex, class MV, class MT, class VT>
BATCHLAS_DLA_INLINE void larft(const Ex& ex, const MV& V, const MT& Tm, const VT& tau) {
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
        else larft_detail::trmv_upper_row_owner(ex, Tm.sub(0, 0, j, j), assume_published(Tm.row(j).head(j)), t_col);
    }
    ex.barrier(Mem::Global);

    laset<Part::StrictLower>(ex, Zero{}, Zero{}, Tm);
}

}  // namespace batchlas::device
