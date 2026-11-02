// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension (gesvd / ormbr family): executors for host-launched glue kernels, operands
// that keep v1's batched index arithmetic, and the elementwise LAPACK-like ops those kernels are.
//
//   Elem / Elem2      one work-item per element of a launch that covers the extent exactly; no
//                     bounds test. Elem2::sub(r0, c0) is ElemIn2, which tests the sub-block's bounds.
//   item(view, b, ..) item b of a v1 VectorView / KernelMatrixView, indexed through the view's own
//                     operator() (same 64-bit arithmetic as the hand code; v2's vec()/mat() are int32).
//   strided(p, ..)    a raw pointer with a precomputed item base, same arithmetic as p[base + i + j*ld].
//   ops               fill laset lacpy omatcopy trcopy lacpy_strict mirror dgmm downshift bdgram
//                     bdmv_div tridiag_abs_offdiag any_of.

#include <cstddef>
#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2 {

// ---- element executors ------------------------------------------------------------------------
struct Elem {
    int i;
    static constexpr int size = 1, team_size = 1, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T team_reduce(T v, Op = {}) const { return v; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op = {}) const { return v; }
};

struct ElemIn2 {
    int i, j;    // absolute indices of the launch
    int r0, c0;  // origin of the sub-block the op addresses
};

struct Elem2 {
    int i, j;
    static constexpr int size = 1, team_size = 1, nteams = 1;
    static constexpr bool sub_whole = true;
    static constexpr Scope natural_scope = Scope::Sub;
    BATCHLAS_DLA_INLINE ElemIn2 sub(int r0, int c0) const { return {i, j, r0, c0}; }
};

// Elem owns index i of an exactly covering launch: f(i), never a loop or a bounds test.
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const Elem& ex, int, F&& f) { f(ex.i); }

// f(i, j) for every (i, j) < (m, n) this item owns: column loop outside, Cyclic rows inside.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned2(const Ex& ex, int m, int n, F&& f) {
    for (int j = 0; j < n; ++j) for_each_owned(ex, m, [&](int i) { f(i, j); });
}
template <class F> BATCHLAS_DLA_INLINE void for_each_owned2(const Elem2& ex, int, int, F&& f) { f(ex.i, ex.j); }
template <class F> BATCHLAS_DLA_INLINE void for_each_owned2(const ElemIn2& ex, int m, int n, F&& f) {
    if (ex.i >= ex.r0 && ex.j >= ex.c0 && ex.i < ex.r0 + m && ex.j < ex.c0 + n) f(ex.i - ex.r0, ex.j - ex.c0);
}

// The strict lower triangle i > j of an m x n block.
template <class Ex, class F> BATCHLAS_DLA_INLINE void for_each_owned_strict_lower(const Ex& ex, int m, int n, F&& f) {
    for (int j = 0; j < n; ++j) for_each_owned(ex, m - j - 1, [&](int t) { f(j + 1 + t, j); });
}
template <class F> BATCHLAS_DLA_INLINE void for_each_owned_strict_lower(const Elem2& ex, int, int, F&& f) {
    if (ex.i > ex.j) f(ex.i, ex.j);
}
// Column test first, then the triangle and the row bound: the triangle (i > j) together with the
// row bound makes i >= r0 and, for a square block, j < c0 + n redundant (folded by the compiler).
template <class F> BATCHLAS_DLA_INLINE void for_each_owned_strict_lower(const ElemIn2& ex, int m, int n, F&& f) {
    if (ex.j >= ex.c0) {
        if (ex.i >= ex.j + (ex.r0 - ex.c0) + 1 && ex.i < ex.r0 + m && ex.j < ex.c0 + n) f(ex.i - ex.r0, ex.j - ex.c0);
    }
}

// ---- operands ---------------------------------------------------------------------------------
template <class M> using melem_t = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;

template <class V> struct ItemVec {
    const V* v;
    int b, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return (*v)(i, b); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class V> struct ItemCol {
    const V* v;
    int j, b, m;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return (*v)(i, j, b); }
    BATCHLAS_DLA_INLINE int size() const { return m; }
};
template <class V> struct ItemRow {
    const V* v;
    int i, b, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int j) const { return (*v)(i, j, b); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class V> struct ItemMat {
    const V* v;
    int b, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return (*v)(i, j, b); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
    BATCHLAS_DLA_INLINE ItemCol<V> col(int j) const { return {v, j, b, m}; }
    BATCHLAS_DLA_INLINE ItemRow<V> row(int i) const { return {v, i, b, n}; }
};
// Item b of a v1 view, n elements (a VectorView) or m x n (a KernelMatrixView).
template <class V> BATCHLAS_DLA_INLINE ItemVec<V> item(const V& v, int b, int n) { return {&v, b, n}; }
template <class V> BATCHLAS_DLA_INLINE ItemMat<V> item(const V& v, int b, int m, int n) { return {&v, b, m, n}; }

template <class T> struct PtrVec {
    T* p;
    std::size_t base;
    int n;
    BATCHLAS_DLA_INLINE T& operator()(int i) const { return p[base + static_cast<std::size_t>(i)]; }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class T> struct PtrMat {
    T* p;
    std::ptrdiff_t base;
    int ld, m, n;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[base + i + j * ld]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class T> BATCHLAS_DLA_INLINE PtrVec<T> strided(T* p, std::size_t base, int n) { return {p, base, n}; }
template <class T> BATCHLAS_DLA_INLINE PtrMat<T> strided(T* p, std::ptrdiff_t base, int ld, int m, int n) {
    return {p, base, ld, m, n};
}

// A(r0 + i, c0 + j), m x n.
template <class M> struct Shifted {
    M a;
    int r0, c0, m, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(r0 + i, c0 + j); }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <class M> BATCHLAS_DLA_INLINE Shifted<M> shifted(const M& a, int r0, int c0, int m, int n) { return {a, r0, c0, m, n}; }

// x(n - 1 - i).
template <class V> struct Reversed {
    V v;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v((v.size() - 1) - i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE Reversed<V> reversed(const V& v) { return {v}; }

// x(p(i)), p.size() elements (a permutation gather).
template <class V, class P> struct Gathered {
    V v;
    P p;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(p(i)); }
    BATCHLAS_DLA_INLINE int size() const { return p.size(); }
};
template <class V, class P> BATCHLAS_DLA_INLINE Gathered<V, P> gathered(const V& v, const P& p) { return {v, p}; }

// ---- ops ----------------------------------------------------------------------------------------
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}

namespace detail {
// a * s for s of a's type, or a real s scaling a complex a (two multiplies, no complex product).
template <class T, class S> BATCHLAS_DLA_INLINE T scale(T a, S s) {
    using A = Arith<T>;
    if constexpr (A::is_complex && !Arith<S>::is_complex) return A::make(A::re(a) * s, A::im(a) * s);
    else return A::mul(a, static_cast<T>(s));
}
template <Transpose TR, class MA> BATCHLAS_DLA_INLINE auto op_at(const MA& a, int i, int j) {
    using T = melem_t<MA>;
    if constexpr (TR == Transpose::NoTrans) return static_cast<T>(a(i, j));
    else if constexpr (TR == Transpose::Trans) return static_cast<T>(a(j, i));
    else return Arith<T>::conj(a(j, i));
}
}  // namespace detail

// x := alpha
template <class Ex, class AS, class X> BATCHLAS_DLA_INLINE void fill(const Ex& ex, AS alpha, const X& x) {
    using T = elem_t<X>;
    const auto a = as_tag(alpha);
    for_each_owned(ex, x.size(), [&](int i) { x(i) = tag_value<T>(a); });
}

// A := beta on the diagonal, alpha off it (laset, uplo = all).
template <class Ex, class AS, class BS, class MA> BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha, BS beta, const MA& a) {
    using T = melem_t<MA>;
    const auto al = as_tag(alpha);
    const auto be = as_tag(beta);
    for_each_owned2(ex, a.rows(), a.cols(), [&](int i, int j) { a(i, j) = (i == j) ? tag_value<T>(be) : tag_value<T>(al); });
}

// B := A over B's extent (lacpy, uplo = all); the element type converts (real -> complex widens).
template <class Ex, class MA, class MB> BATCHLAS_DLA_INLINE void lacpy(const Ex& ex, const MA& a, const MB& b) {
    using T = melem_t<MB>;
    for_each_owned2(ex, b.rows(), b.cols(), [&](int i, int j) { b(i, j) = static_cast<T>(a(i, j)); });
}

// B := alpha op(A) (oneMKL omatcopy), distributed over A's elements: A(i, j) goes to B(j, i) for a
// transpose, so each item reads one element and writes one.
template <Transpose TR, class Ex, class AS, class MA, class MB>
BATCHLAS_DLA_INLINE void omatcopy(const Ex& ex, AS alpha, const MA& a, const MB& b) {
    using T = melem_t<MB>;
    const auto al = as_tag(alpha);
    for_each_owned2(ex, a.rows(), a.cols(), [&](int i, int j) {
        if constexpr (TR == Transpose::NoTrans) b(i, j) = tag_mul(al, static_cast<T>(a(i, j)));
        else if constexpr (TR == Transpose::Trans) b(j, i) = tag_mul(al, static_cast<T>(a(i, j)));
        else b(j, i) = tag_mul(al, Arith<T>::conj(a(i, j)));
    });
}

// B := the UL part of op(A) with a unit (DG = Unit) or op(A)'s (NonUnit) diagonal, zero in the other
// triangle; one store per element of B. Lower only (the reflector-panel form of LAPACK's lacpy + laset).
template <Uplo UL, Transpose TR, Diag DG, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void trcopy(const Ex& ex, const MA& a, const MB& b) {
    static_assert(UL == Uplo::Lower, "v2::trcopy: only the lower part is implemented");
    using T = melem_t<MB>;
    for_each_owned2(ex, b.rows(), b.cols(), [&](int i, int j) {
        T v = Arith<T>::zero();
        if constexpr (DG == Diag::Unit) {
            if (i == j) v = Arith<T>::one();
            else if (i > j) v = detail::op_at<TR>(a, i, j);
        } else {
            if (i >= j) v = detail::op_at<TR>(a, i, j);
        }
        b(i, j) = v;
    });
}

// B's strict UL triangle := op(A); every other element of B is left untouched. Lower only.
template <Uplo UL, Transpose TR, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void lacpy_strict(const Ex& ex, const MA& a, const MB& b) {
    static_assert(UL == Uplo::Lower, "v2::lacpy_strict: only the lower triangle is implemented");
    for_each_owned_strict_lower(ex, b.rows(), b.cols(), [&](int i, int j) { b(i, j) = detail::op_at<TR>(a, i, j); });
}

// A := its UL triangle mirrored into the other one, conjugated when Herm; in place, square A.
template <Uplo UL, bool Herm = true, class Ex, class MA> BATCHLAS_DLA_INLINE void mirror(const Ex& ex, const MA& a) {
    using T = melem_t<MA>;
    for_each_owned2(ex, a.rows(), a.cols(), [&](int i, int j) {
        if ((UL == Uplo::Lower) ? (i > j) : (i < j)) a(j, i) = Herm ? Arith<T>::conj(a(i, j)) : static_cast<T>(a(i, j));
    });
}

// A := diag(d) A (Side::Left) or A diag(d) (Side::Right); a real d scales a complex A by components.
template <Side SD, class Ex, class D, class MA> BATCHLAS_DLA_INLINE void dgmm(const Ex& ex, const D& d, const MA& a) {
    for_each_owned2(ex, a.rows(), a.cols(), [&](int i, int j) {
        const auto s = d(SD == Side::Left ? i : j);
        a(i, j) = detail::scale(static_cast<melem_t<MA>>(a(i, j)), s);
    });
}

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

// 1 when pred(i) holds for some i < n, else 0; OR-reduced over scope S (uniform result).
template <Scope S = Scope::Auto, class Ex, class P> BATCHLAS_DLA_INLINE int any_of(const Ex& ex, int n, P&& pred) {
    int acc = 0;
    for_each_owned(ex, n, [&](int i) { acc |= pred(i) ? 1 : 0; });
    return detail::scope_reduce<S>(ex, acc, sycl::bit_or<int>());
}

}  // namespace batchlas::device::v2
