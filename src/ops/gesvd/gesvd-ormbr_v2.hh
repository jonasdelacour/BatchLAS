// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, the gesvd / ormbr private pieces: element executors for the host-launched glue
// kernels, adaptors over v2::item() views, and the bidiagonal / tridiagonal glue ops no other
// family uses. Core ops (lacpy, laset, mirror, map, copy) run on Elem / Elem2 / ElemIn2 through
// the for_each_owned / for_each_owned_2d overloads below, which argument-dependent lookup finds
// from inside the core templates (they are more specialised than core's generic loops).

#include <type_traits>

#include <batchlas/device.hh>

namespace batchlas::device::inline v2::gesvd_glue {
using device::dgmm;  // core level2.hh

// ---- element executors ------------------------------------------------------------------------
// Elem: index i of a launch that covers the extent exactly. Elem2: element (i, j) of such a 2-D
// launch. Neither loops nor tests a bound. Elem2::sub(r0, c0) is ElemIn2: the op addresses the
// block at (r0, c0) of the launch's grid, whose bounds are tested.
struct Elem {
    int i;
};
struct ElemIn2 {
    int i, j;    // absolute indices of the launch
    int r0, c0;  // origin of the block the op addresses
};
struct Elem2 {
    int i, j;
    BATCHLAS_DLA_INLINE ElemIn2 sub(int r0, int c0) const { return {i, j, r0, c0}; }
};

template <class U = unroll::Default, class F> BATCHLAS_DLA_INLINE void for_each_owned(const Elem& ex, int, F&& f) {
    f(ex.i);
}
template <class Alg = alg::Walk, class Idx = int, class F>
BATCHLAS_DLA_INLINE void for_each_owned_2d(const Elem2& ex, int, int, F&& f) {
    f(ex.i, ex.j);
}
// Column test first (the ormbr AQ kernel's order), then the rows and the far column bound.
template <class Alg = alg::Walk, class Idx = int, class F>
BATCHLAS_DLA_INLINE void for_each_owned_2d(const ElemIn2& ex, int m, int n, F&& f) {
    if (ex.j >= ex.c0) {
        if (ex.i >= ex.r0 && ex.i < ex.r0 + m && ex.j < ex.c0 + n) f(ex.i - ex.r0, ex.j - ex.c0);
    }
}

// ---- operand adaptors ---------------------------------------------------------------------------
// v2::item(VectorView, b) with n elements (the view's own size may be larger).
template <class T> BATCHLAS_DLA_INLINE ItemVec<T> item_n(const VectorView<T>& v, int b, int n) {
    ItemVec<T> x = item(v, b);
    x.n = n;
    return x;
}
// Column j, row i and the m x n block at (r0, c0) of a matrix operand: A(i, j), A(i, j),
// A(r0 + i, c0 + j), indexed through the operand's own operator() (item()'s v1 address).
template <class M> struct ColOf {
    using value_type = typename M::value_type;
    M a;
    int j;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return a.rows(); }
};
template <class M> struct RowOf {
    using value_type = typename M::value_type;
    M a;
    int i;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int j) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return a.cols(); }
};
template <class M> struct SubOf {
    using value_type = typename M::value_type;
    M a;
    int r0, c0, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(r0 + i, c0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE ColOf<M> col_of(const M& a, int j) { return {a, j}; }
template <class M> BATCHLAS_DLA_INLINE RowOf<M> row_of(const M& a, int i) { return {a, i}; }
template <class M> BATCHLAS_DLA_INLINE SubOf<M> sub_of(const M& a, int r0, int c0, int m, int n) {
    return {a, r0, c0, m, n};
}

// x(n - 1 - i) and x(p(i)) (a permutation gather, p.size() elements).
template <class V> struct Reversed {
    V v;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v((v.size() - 1) - i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V, class P> struct Gathered {
    V v;
    P p;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(p(i)); }
    BATCHLAS_DLA_INLINE int size() const { return p.size(); }
};
template <class V> BATCHLAS_DLA_INLINE Reversed<V> reversed(const V& v) { return {v}; }
template <class V, class P> BATCHLAS_DLA_INLINE Gathered<V, P> gathered(const V& v, const P& p) { return {v, p}; }

// y := Z x for the down-shift Z: y(0) = 0, y(i) = x(i - 1) for i <= x.size(), 0 past it.
template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void downshift(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<Y>;
    for_each_owned(ex, y.size(), [&](int i) {
        if (i == 0) y(i) = Arith<T>::zero();
        else y(i) = (i - 1 < x.size()) ? static_cast<T>(x(i - 1)) : Arith<T>::zero();
    });
}

// The symmetric tridiagonal Gram matrix of the real upper bidiagonal B = (d, e), k = d.size():
// Side::Right is B^T B (order k), Side::Left is B B^T padded with zero rows to order td.size().
template <Side SD, class Ex, class D, class E, class TD, class TE>
BATCHLAS_DLA_INLINE void bdgram(const Ex& ex, const D& d, const E& e, const TD& td, const TE& te) {
    using T = elem_t<TD>;
    static_assert(!Arith<T>::is_complex, "v2::bdgram: the bidiagonal of gebrd is real");
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
template <class Ex, class D, class E, class X, class R, class Y>
BATCHLAS_DLA_INLINE void bdmv_div(const Ex& ex, const D& d, const E& e, const X& x, R sigma, R tol, const Y& y) {
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
template <class Ex, class E, class S> BATCHLAS_DLA_INLINE void tridiag_abs_offdiag(const Ex&, const E& e, const S& s) {
    static_assert(is_thread_v<Ex>, "v2::tridiag_abs_offdiag: a sequential scan, Thread only");
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

}  // namespace batchlas::device::gesvd_glue
