// SPDX-License-Identifier: MIT
#pragma once
// sytrd-private device LA v2 ops for tiles that fit the executor (one row per item): the SYTD2
// shared by sytrd_cta and the fused syev kernel; shared ops are core's. Strict port: arith::Native
// and the replaced kernel's expression shapes, so fp-contract fuses the same products.

#include <cstdint>
#include <type_traits>

#include <batchlas/device.hh>
#include <src/kernels/common/dla_v2_partition.hh>

namespace batchlas::sytrd_v2 {

namespace v2 = batchlas::device;

// ---- executors ---------------------------------------------------------------------------
// Item lid owns element / row lid (n <= size); core copy / axpy find for_each_owned(Fit) by ADL.
template <class Ex> struct Fit {
    static constexpr int size = Ex::size, team_size = Ex::team_size, nteams = Ex::nteams;
    static constexpr bool sub_whole = Ex::sub_whole;
    static constexpr v2::Scope natural_scope = Ex::natural_scope;
    Ex ex;

    BATCHLAS_DLA_INLINE int lid() const { return ex.lid(); }
    BATCHLAS_DLA_INLINE int lane() const { return ex.lane(); }
    BATCHLAS_DLA_INLINE int team() const { return ex.team(); }
    BATCHLAS_DLA_INLINE bool leader() const { return ex.leader(); }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op op = {}) const {
        return ex.team_reduce(v, op);
    }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return ex.group_reduce(v, op);
    }
};
template <class Ex> BATCHLAS_DLA_INLINE Fit<Ex> fit(const Ex& ex) { return Fit<Ex>{ex}; }

// On a Fit executor an index has one owner and one guarded call (no loop, no stride).
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Fit<Ex>& ex, int n, F&& f) {
    const int i = ex.lid();
    if (i >= 0 && i < n) f(i);
}

// The hand kernels' chunk barrier: SubGroupPartition's own (masked on NVPTX), else sub-group.
template <class Ex> BATCHLAS_DLA_INLINE void part_barrier(const Ex& ex) {
    if constexpr (requires { ex.ex; }) {
        part_barrier(ex.ex);
    } else if constexpr (sycl::is_group_v<std::remove_cvref_t<decltype(ex.part)>>) {
        sycl::group_barrier(ex.part);
    } else {
        group_barrier(ex.part);  // ADL: SubGroupPartition (src/kernels/common)
    }
}

// The value item `src` holds, on every item of the chunk.
template <class Ex, class T> BATCHLAS_DLA_INLINE T bcast(const Ex& ex, const T& v, int src) {
    if constexpr (requires { ex.ex; }) return ex.ex.bcast(v, static_cast<std::uint32_t>(src));
    else return ex.bcast(v, static_cast<std::uint32_t>(src));
}

// ---- operands and element kernels ----------------------------------------------------------
// v with element k read as 1 (LAPACK's temporary unit of a stored reflector); v(k) is not read.
template <class V> struct ImpliedUnit {
    V v;
    int k;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        return i == k ? v2::Arith<value_type>::one() : static_cast<value_type>(v(i));
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE ImpliedUnit<V> implied_unit(V v, int k) { return {v, k}; }

// x1 conj(y1) + x2 conj(y2): one term of a rank-2 Hermitian update. Real: one expression, so
// it contracts to fma(x1, y1, x2 * y2) as the hand kernel's did.
template <class Ops, class T> BATCHLAS_DLA_INLINE T her2_term(const T& x1, const T& y1, const T& x2, const T& y2) {
    using Ar = v2::arith_t<Ops, T>;
    if constexpr (Ar::is_complex) return Ar::add(Ar::mul(x1, Ar::conj(y1)), Ar::mul(x2, Ar::conj(y2)));
    else return x1 * y1 + x2 * y2;
}

// ---- level 1 / 2 on Fit ------------------------------------------------------------------------
// x^H y, one product per item (no accumulator), then the chunk reduction.
template <class Ops, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto dotc(const Fit<Ex>& ex, const X& x, const Y& y) {
    using T = v2::elem_t<X>;
    using Ar = v2::arith_t<Ops, T>;
    const int i = ex.lid();
    const T p = (i >= 0 && i < x.size()) ? Ar::mul(Ar::conj(x(i)), y(i)) : Ar::zero();
    return v2::detail::scope_reduce<v2::Scope::Auto>(ex, p);
}

// y(r) := (A x)(r) * alpha with item r computing output row r (core gemv's RowOwner without
// the loop); x is read across owners. alpha multiplies on the right, as `y *= tau` did.
template <class Ops, class Ex, class T, class MA, class X, class Y>
BATCHLAS_DLA_INLINE void gemv_rows(const Fit<Ex>& ex, const T& alpha, const MA& a, const v2::Published<X>& xp, const Y& y) {
    using Ar = v2::arith_t<Ops, T>;
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
BATCHLAS_DLA_INLINE void her2_full_sub(const Fit<Ex>& ex, const v2::Published<X>& xp, const v2::Published<Y>& yp, const MA& a) {
    using T = typename MA::value_type;
    using Ar = v2::arith_t<Ops, T>;
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
    using Ar = v2::arith_t<Ops, T>;
    const int n = a.rows(), r = ex.lid();
    if (r < 0 || r >= tile.rows()) return;
    if (active && r < n) {
        for (int c = 0; c < n; ++c) {
            if (UL == Uplo::Upper ? r > c : r < c) continue;
            T v = a(r, c);
            if (r == c) {
                if constexpr (Ar::is_complex) v = T(Ar::re(v), v2::real_t<T>(0));
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

// B := A (m x n), item r copying row r (core lacpy owns elements 2-D Cyclic; this is the
// row-owner form the hand kernels wrote, one guarded row per item).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy_rows(const Fit<Ex>& ex, const MA& a, const MB& b) {
    const int n = a.cols();
    for_each_owned(ex, a.rows(), [&](int r) {
        for (int c = 0; c < n; ++c) b(r, c) = a(r, c);
    });
}

// DLARFG on a vector held one element per item (n <= Ex::size): item alpha_item holds alpha;
// items with x_active hold x (the others pass x = 0, x_active = false). On return alpha_item
// holds beta and the x items hold v (implicit unit); tau is returned on every item. ||x|| and
// alpha are replicated, so every item forms the scalars (no serial leader). n <= 1: tau = 0.
template <class Ops, class Ex, class T>
BATCHLAS_DLA_INLINE T larfg_rowfrag(const Ex& ex, int n, int alpha_item, T& alpha, T& x, bool x_active) {
    using Ar = v2::arith_t<Ops, T>;
    using R = v2::real_t<T>;
    const R xsq = x_active ? v2::abs2(x) : R(0);
    const R xnorm = sycl::sqrt(v2::detail::scope_reduce<v2::Scope::Auto>(ex, xsq));
    const T alpha_b = bcast(ex, alpha, alpha_item);
    const auto s = v2::larfg_scalars<v2::conv::Phase>(alpha_b, xnorm, n);
    if (ex.lid() == alpha_item) alpha = s.beta;
    else if (x_active && s.tau != Ar::zero()) x = Ar::mul(x, s.scale);
    return s.tau;
}

// ---- SYTD2 -------------------------------------------------------------------------------------
struct NoStep {
    template <class... A> BATCHLAS_DLA_INLINE void operator()(const A&...) const {}
};

namespace detail {
// One sytd2 step on the active block blk: [alpha; x] = col one element per item (alpha on item
// ar), row is col's conjugate mirror. Reflector; mirrored write-back; then, unless tau == 0,
// v := reflector, w := A v tau, w += (-tau/2)(v^H w) v, A -= v w^H + w v^H.
template <Uplo UL, class Ops, class Ex, class Blk, class Col, class Row, class V, class W, class T, class F>
BATCHLAS_DLA_INLINE void sytd2_step(const Ex& ex, int i, int m, int ar, const Blk& blk, const Col& col, const Row& row,
                                    const V& v, const W& w, bool x_active, T& tau_lane, F& on_step) {
    using Ar = v2::arith_t<Ops, T>;
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
    const T taui = larfg_rowfrag<Ops>(ex, m, ar, alpha, x, x_active);
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
        v2::copy(fx, implied_unit(col, ar), vm);
        part_barrier(ex);
        if (is_alpha) {  // LAPACK's temporary unit; the block ops read v, not the column
            col(ar) = Ar::one();
            row(ar) = Ar::one();
        }
        part_barrier(ex);
        gemv_rows<Ops>(fx, taui, blk, v2::assume_published(vm), wm);
        part_barrier(ex);
        const T alpha2 = T(-0.5) * taui * dotc<Ops>(fx, vm, wm);
        v2::axpy<Ops>(fx, v2::Scalar<T>{alpha2}, vm, wm);
        part_barrier(ex);
        her2_full_sub<Ops>(fx, v2::assume_published(vm), v2::assume_published(wm), blk);
        part_barrier(ex);
        if (is_alpha) {
            col(ar) = alpha;
            row(ar) = Ar::conj(alpha);
        }
        part_barrier(ex);
    } else if constexpr (UL == Uplo::Upper) {
        if (is_alpha) row(ar) = Ar::conj(alpha);
    }
}
}  // namespace detail

// xSYTD2 / xHETD2 on the n x n leading block of a resident tile held in FULL storage (both
// triangles, real diagonal), one row per item (n <= Ex::size); v and w are length >= n scratch.
// Leaves LAPACK's packing, every element's conjugate mirror kept:
//   d(i) = A(i,i); Upper: e(i) = A(i,i+1), reflector i in A(0:i-1, i+1) (unit at row i);
//                  Lower: e(i) = A(i+1,i), reflector i in A(i+2:n-1, i) (unit at row i+1).
// on_step(i, beta, tau) runs on the item holding reflector i's alpha right after it is formed.
// Upper returns tau(lid) on item lid (0 past n-2); Lower returns 0 (tau goes through on_step).
// Step k touches only its active block, so column k is final once step k has run.
template <Uplo UL, class Ops = v2::arith::Native, class Ex, class MA, class V, class W, class F = NoStep>
BATCHLAS_DLA_INLINE typename MA::value_type sytd2(const Ex& ex, const MA& a, const V& v, const W& w, int n, F&& on_step = {}) {
    using T = typename MA::value_type;
    const int lane = ex.lid();
    T tau_lane = v2::arith_t<Ops, T>::zero();
    if constexpr (UL == Uplo::Upper) {
        for (int k = n - 1; k >= 1; --k) {  // annihilate A(0:k-2, k); alpha = A(k-1, k)
            detail::sytd2_step<UL, Ops>(ex, k - 1, k, k - 1, a.sub(0, 0, k, k), a.col(k).head(k), a.row(k).head(k), v, w,
                                        lane < k - 1, tau_lane, on_step);
            part_barrier(ex);
        }
    } else {
        for (int i = 0; i <= n - 2; ++i) {  // annihilate A(i+2:n-1, i); alpha = A(i+1, i) on item 0
            const int s = i + 1, m = n - s;
            detail::sytd2_step<UL, Ops>(ex, i, m, 0, a.sub(s, s, m, m), a.col(i).tail(s).head(m), a.row(i).tail(s).head(m), v,
                                        w, lane >= 1 && lane < m, tau_lane, on_step);
        }
    }
    return tau_lane;
}

}  // namespace batchlas::sytrd_v2
