// SPDX-License-Identifier: MIT
#pragma once
// xSYTRD's unblocked reduction xSYTD2 / xHETD2 on a resident tile (one row per item), shared by
// sytrd_cta and the fused syev kernel, plus the CTA driver's tile copies. Strict port:
// arith::Native and the replaced kernel's expression shapes, so fp-contract fuses the same
// products. sytrd_detail holds the forms core does not reproduce bit for bit yet: dotc / gemv /
// her2 on a Fit executor (one guarded row, product-only dot, subtracted rank-2 term) and the
// row-owner Hermitian tile copy.

#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>

namespace batchlas::device::inline v2 {

namespace sytrd_detail {

// x1 conj(y1) + x2 conj(y2): one term of a rank-2 Hermitian update. Real: one expression, so
// it contracts to fma(x1, y1, x2 * y2) as the hand kernel's did.
template <class Ops, class T> BATCHLAS_DLA_INLINE T her2_term(const T& x1, const T& y1, const T& x2, const T& y2) {
    using Ar = arith_t<Ops, T>;
    if constexpr (Ar::is_complex) return Ar::add(Ar::mul(x1, Ar::conj(y1)), Ar::mul(x2, Ar::conj(y2)));
    else return x1 * y1 + x2 * y2;
}

// x^H y, one product per item (no accumulator), then the chunk reduction.
template <class Ops, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dotc(const Fit<Ex>& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using Ar = arith_t<Ops, T>;
    const int i = ex.lid();
    const T p = (i >= 0 && i < x.size()) ? Ar::mul(Ar::conj(x(i)), y(i)) : Ar::zero();
    return detail::scope_reduce<Scope::Auto>(ex, p);
}

// y(r) := (A x)(r) * alpha with item r computing output row r (core gemv's RowOwner without
// the loop); x is read across owners. alpha multiplies on the right, as `y *= tau` did.
template <class Ops, class Ex, class T, class MA, class X, class Y>
BATCHLAS_DLA_INLINE void gemv_rows(const Fit<Ex>& ex, const T& alpha, const MA& a, const Published<X>& xp, const Y& y) {
    using Ar = arith_t<Ops, T>;
    const auto& x = xp.v;
    const int m = a.rows(), n = a.cols(), r = ex.lid();
    if (r >= 0 && r < m) {
        T acc = Ar::zero();
        for (int c = 0; c < n; ++c) Ar::mac(acc, a(r, c), x(c));
        y(r) = Ar::mul(acc, alpha);
    }
}

// A := A - (x y^H + y x^H) on the FULL square A (both triangles), one row per item.
template <class Ops, class Ex, class X, class Y, class MA>
BATCHLAS_DLA_INLINE void her2_full_sub(const Fit<Ex>& ex, const Published<X>& xp, const Published<Y>& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = arith_t<Ops, T>;
    const auto& x = xp.v;
    const auto& y = yp.v;
    const int m = a.rows(), n = a.cols(), r = ex.lid();
    if (r >= 0 && r < m) {
        const T xr = x(r), yr = y(r);
        for (int c = 0; c < n; ++c) {
            const T xc = x(c), yc = y(c);
            a(r, c) = Ar::sub(static_cast<T>(a(r, c)), her2_term<Ops>(xr, yc, yr, xc));
        }
    }
}

// tile := the Hermitian matrix whose UL triangle is a (n x n, n = a.rows()), Im(diag) dropped,
// zero past n up to the tile's extent. Item r reads row r of the triangle (coalesced) and writes
// it and its conjugate mirror, so a barrier must precede reads of other rows. active == false
// (a masked-off batch item) zero-fills the tile's row and reads nothing.
template <Uplo UL, class Ops, class Ex, class MA, class MT>
BATCHLAS_DLA_INLINE void lacpy_herm(const Fit<Ex>& ex, const MA& a, const MT& tile, bool active) {
    using T = typename MT::value_type;
    using Ar = arith_t<Ops, T>;
    const int n = a.rows(), r = ex.lid();
    if (r < 0 || r >= tile.rows()) return;
    if (active && r < n) {
        for (int c = 0; c < n; ++c) {
            if (UL == Uplo::Upper ? r > c : r < c) continue;
            T v = a(r, c);
            if (r == c) {
                if constexpr (Ar::is_complex) v = T(Ar::re(v), real_t<T>(0));
            } else {
                tile(c, r) = Ar::conj(v);
            }
            tile(r, c) = v;
        }
        for (int c = n; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    } else {
        for (int c = 0; c < tile.cols(); ++c) tile(r, c) = Ar::zero();
    }
}

// B := A (m x n), item r copying row r (core lacpy owns elements 2-D; this is the row-owner
// form the hand kernels wrote, one guarded row per item).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy_rows(const Fit<Ex>& ex, const MA& a, const MB& b) {
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int c = 0; c < n; ++c) b(r, c) = a(r, c);
    });
}

// One sytd2 step on the active block blk: [alpha; x] = col one element per item (alpha on item
// ar), row is col's conjugate mirror. Reflector; mirrored write-back; then, unless tau == 0,
// v := reflector, w := A v tau, w += (-tau/2)(v^H w) v, A -= v w^H + w v^H.
template <Uplo UL, class Ops, class Ex, class Blk, class Col, class Row, class V, class W, class T, class F>
BATCHLAS_DLA_INLINE void sytd2_step(const Ex& ex, int i, int m, int ar, const Blk& blk, const Col& col, const Row& row,
                                    const V& v, const W& w, bool x_active, T& tau_lane, F& on_step) {
    using Ar = arith_t<Ops, T>;
    const Fit<Ex> fx{ex};
    const int lane = ex.lid();
    const auto vm = v.head(m);
    const auto wm = w.head(m);
    const bool is_alpha = lane == ar;
    T alpha = Ar::zero(), x = Ar::zero();
    if (lane < m) {
        const T e = col(lane);
        if (is_alpha) alpha = e;
        else x = e;
    }
    const T taui = v2::larfg<conv::Phase, Ops>(ex, m, ar, alpha, x, x_active);
    if (x_active) {  // both triangles: the block update reads the full block
        col(lane) = x;
        row(lane) = Ar::conj(x);
    }
    if (is_alpha) {
        col(ar) = alpha;
        row(ar) = Ar::conj(alpha);
        if constexpr (UL == Uplo::Upper) tau_lane = taui;
        on_step(i, alpha, taui);
    }
    if (taui != Ar::zero()) {
        v2::copy(fx, v2::unit_head<refl::QR, unit_fill::Keep>(col, ar), vm);
        v2::sync<scope::Part>(ex);
        if (is_alpha) {  // LAPACK's temporary unit; the block ops read v, not the column
            col(ar) = Ar::one();
            row(ar) = Ar::one();
        }
        v2::sync<scope::Part>(ex);
        sytrd_detail::gemv_rows<Ops>(fx, taui, blk, assume_published(vm), wm);
        v2::sync<scope::Part>(ex);
        const T alpha2 = T(-0.5) * taui * sytrd_detail::dotc<Ops>(fx, vm, wm);
        v2::axpy<Ops>(fx, Scalar<T>{alpha2}, vm, wm);
        v2::sync<scope::Part>(ex);
        sytrd_detail::her2_full_sub<Ops>(fx, assume_published(vm), assume_published(wm), blk);
        v2::sync<scope::Part>(ex);
        if (is_alpha) {
            col(ar) = alpha;
            row(ar) = Ar::conj(alpha);
        }
        v2::sync<scope::Part>(ex);
    } else if constexpr (UL == Uplo::Upper) {
        if (is_alpha) row(ar) = Ar::conj(alpha);
    }
}

}  // namespace sytrd_detail

// xSYTD2 / xHETD2 on the n x n leading block of a resident tile held in FULL storage (both
// triangles, real diagonal), one row per item (n <= Ex::size); v and w are length >= n scratch.
// Leaves LAPACK's packing, every element's conjugate mirror kept:
//   d(i) = A(i,i); Upper: e(i) = A(i,i+1), reflector i in A(0:i-1, i+1) (unit at row i);
//                  Lower: e(i) = A(i+1,i), reflector i in A(i+2:n-1, i) (unit at row i+1).
// on_step(i, beta, tau) runs on the item holding reflector i's alpha right after it is formed.
// Upper returns tau(lid) on item lid (0 past n-2); Lower returns 0 (tau goes through on_step).
// Step k touches only its active block, so column k is final once step k has run. COLLECTIVE.
template <Uplo UL, class Ops = arith::Native, class Ex, class MA, class V, class W, class F = NoOp>
BATCHLAS_DLA_INLINE typename MA::value_type sytd2(const Ex& ex, const MA& a, const V& v, const W& w, int n, F&& on_step = {}) {
    using T = typename MA::value_type;
    const int lane = ex.lid();
    T tau_lane = arith_t<Ops, T>::zero();
    if constexpr (UL == Uplo::Upper) {
        for (int k = n - 1; k >= 1; --k) {  // annihilate A(0:k-2, k); alpha = A(k-1, k)
            sytrd_detail::sytd2_step<UL, Ops>(ex, k - 1, k, k - 1, a.sub(0, 0, k, k), a.col(k).head(k), a.row(k).head(k), v, w,
                                              lane < k - 1, tau_lane, on_step);
            v2::sync<scope::Part>(ex);
        }
    } else {
        for (int i = 0; i <= n - 2; ++i) {  // annihilate A(i+2:n-1, i); alpha = A(i+1, i) on item 0
            const int s = i + 1, m = n - s;
            sytrd_detail::sytd2_step<UL, Ops>(ex, i, m, 0, a.sub(s, s, m, m), a.col(i).tail(s).head(m), a.row(i).tail(s).head(m),
                                              v, w, lane >= 1 && lane < m, tau_lane, on_step);
        }
    }
    return tau_lane;
}

}  // namespace batchlas::device
