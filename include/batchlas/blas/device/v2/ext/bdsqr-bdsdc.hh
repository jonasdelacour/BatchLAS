// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension used by the bidiagonal SVD kernels (bdsqr, bdsdc), general otherwise:
// a row/column-strided matrix operand, per-item operands straight from v1 views, plane
// rotations (BLAS rot), fills, an interleave, NaN-skipping max, zero-where, argmin over a
// functor, first-index search, multi-operand sums of squares, a diagonal-scaled copy and a
// selection sort with a caller swap. Core conventions hold: alpha first, Cyclic ownership, no
// barrier inside any op, every choice a type.

#include <array>
#include <climits>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::device::v2 {

// ---- operands ----------------------------------------------------------------------------

// (i, j) = p[i * rs + j * cs]. Column j is a Vec of stride rs, so the transpose of a Mat
// (rs = ld, cs = 1) and the Mat itself (rs = 1, cs = ld) are the same type and one code path.
template <class T, Mem M = Mem::Global> struct SMat {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    Dyn m;
    Dyn n;
    Dyn rs;
    Dyn cs;

    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i * int(rs) + j * int(cs)]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> col(int j) const { return {p + j * int(cs), m, rs}; }
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> row(int i) const { return {p + i * int(rs), n, cs}; }
};

template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE SMat<T, M> smat(T* p, int m, int n, int rs, int cs) {
    return {p, Dyn{m}, Dyn{n}, Dyn{rs}, Dyn{cs}};
}
template <class T, class R, class C, class Ld, Mem M, class L>
BATCHLAS_DLA_INLINE SMat<T, M> as_smat(const Mat<T, R, C, Ld, M, L>& a) {
    return {a.p, Dyn{a.rows()}, Dyn{a.cols()}, Dyn{1}, Dyn{int(a.ld)}};
}
// A^T without a copy.
template <class T, class R, class C, class Ld, Mem M, class L>
BATCHLAS_DLA_INLINE SMat<T, M> trans(const Mat<T, R, C, Ld, M, L>& a) {
    return {a.p, Dyn{a.cols()}, Dyn{a.rows()}, Dyn{int(a.ld)}, Dyn{1}};
}

// Lazy operands: each element is the parent's own element expression, so an address is computed
// exactly once, as the parent computes it (no folded pointer + stride the optimiser re-versions).
template <class V> using operand_value_t = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;

// v(start + k * step) for k < n.
template <class V> struct Sliced {
    using value_type = operand_value_t<V>;
    using layout = Cyclic;
    V v;
    int start;
    Dyn n;
    int step;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int k) const { return v(start + k * step); }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
};
template <class V> BATCHLAS_DLA_INLINE Sliced<V> slice(const V& v, int start, int n, int step) {
    return {v, start, Dyn{n}, step};
}

// Column j / row i of a matrix operand, first n elements.
template <class Mx> struct MatCol {
    using value_type = std::remove_cvref_t<decltype(std::declval<const Mx&>()(0, 0))>;
    using layout = Cyclic;
    Mx a;
    int j;
    Dyn n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    BATCHLAS_DLA_INLINE MatCol head(int k) const { return {a, j, Dyn{k}}; }
};
template <class Mx> struct MatRow {
    using value_type = std::remove_cvref_t<decltype(std::declval<const Mx&>()(0, 0))>;
    using layout = Cyclic;
    Mx a;
    int i;
    Dyn n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int j) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    BATCHLAS_DLA_INLINE MatRow head(int k) const { return {a, i, Dyn{k}}; }
};
template <class Self> struct LazyLines {
    BATCHLAS_DLA_INLINE MatCol<Self> col(int j) const {
        const Self& s = static_cast<const Self&>(*this);
        return {s, j, Dyn{s.rows()}};
    }
    BATCHLAS_DLA_INLINE MatRow<Self> row(int i) const {
        const Self& s = static_cast<const Self&>(*this);
        return {s, i, Dyn{s.cols()}};
    }
};

// Item b of a v1 view, addressed with the view's own expression: VectorView::at is
// p[(int64) i * inc + (int64) b * stride], KernelMatrixView is p[(int64) b * stride + j * ld + i].
// A matrix keeps the view's (capacity) rows and cols, not per-item active extents.
template <class T, Mem M = Mem::Global> struct VVec {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    int b;
    int stride;
    Dyn n;
    Dyn inc;
    BATCHLAS_DLA_INLINE T& operator()(int i) const {
        return p[static_cast<std::int64_t>(i) * int(inc) + static_cast<std::int64_t>(b) * stride];
    }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    BATCHLAS_DLA_INLINE VVec head(int k) const { return {p, b, stride, Dyn{k}, inc}; }
};
template <class T, Mem M = Mem::Global> struct VMat : LazyLines<VMat<T, M>> {
    using value_type = std::remove_cv_t<T>;
    using layout = Cyclic;
    static constexpr Mem mem = M;
    T* p;
    int b;
    int stride;
    Dyn m;
    Dyn n;
    Dyn ld;
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[static_cast<std::int64_t>(b) * stride + j * int(ld) + i]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
};
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE VVec<T, M> vec_item(const VectorView<T>& v, int b) {
    return {v.data_ptr(), b, v.stride(), Dyn{v.size()}, Dyn{v.inc()}};
}
template <Mem M = Mem::Global, class T>
BATCHLAS_DLA_INLINE VMat<T, M> mat_item(const KernelMatrixView<T, MatrixFormat::Dense>& a, int b) {
    return {{}, a.data(), b, a.stride(), Dyn{a.rows()}, Dyn{a.cols()}, Dyn{a.ld()}};
}

// ---- elementwise -------------------------------------------------------------------------

namespace detail {
#ifndef BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE  // integ/dla-v2-all: defined by several ext headers
#define BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
template <class T, class S> BATCHLAS_DLA_INLINE T tag_value(S s) {
    if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}
#endif  // BATCHLAS_DLA_V2_INTEG_DETAIL_TAG_VALUE
}  // namespace detail

// x := alpha (LAPACK laset on a vector).
template <class Ex, class A, class X> BATCHLAS_DLA_INLINE void fill(const Ex& ex, A alpha_, const X& x) {
    using T = elem_t<X>;
    const T v = detail::tag_value<T>(as_tag(alpha_));
    for_each_owned(ex, x.size(), [&](int i) { x(i) = v; });
}

// x := e_k.
template <class Ex, class X> BATCHLAS_DLA_INLINE void set_unit(const Ex& ex, const X& x, int k) {
    using T = elem_t<X>;
    for_each_owned(ex, x.size(), [&](int i) { x(i) = (i == k) ? Arith<T>::one() : Arith<T>::zero(); });
}

// y(2k) := a(k), y(2k+1) := b(k) for k < a.size() (b.size() is a.size() or a.size() - 1): the
// perfect shuffle, one loop over a.
template <class Ex, class A, class B, class Y>
BATCHLAS_DLA_INLINE void interleave(const Ex& ex, const A& a, const B& b, const Y& y) {
    const int nb = b.size();
    for_each_owned(ex, a.size(), [&](int k) {
        y(2 * k) = a(k);
        if (k < nb) y(2 * k + 1) = b(k);
    });
}

// y := alpha diag(d) x, rounded as alpha * (d_i x_i); d and x are read once per element.
template <class Ex, class A, class D, class X, class Y>
BATCHLAS_DLA_INLINE void dgmv(const Ex& ex, A alpha_, const D& d, const X& x, const Y& y) {
    using T = elem_t<Y>;
    const auto a = as_tag(alpha_);
    for_each_owned(ex, y.size(), [&](int i) {
        y(i) = tag_mul(a, Arith<T>::mul(static_cast<T>(d(i)), static_cast<T>(x(i))));
    });
}

// y := alpha diag(d) x when `on` (a deferred dgmv for dgmv_many).
template <class A, class D, class X, class Y> struct DgmvIf {
    bool on;
    A alpha;
    D d;
    X x;
    Y y;
};
template <class A, class D, class X, class Y>
BATCHLAS_DLA_INLINE DgmvIf<A, D, X, Y> dgmv_if(bool on, A alpha, const D& d, const X& x, const Y& y) {
    return {on, alpha, d, x, y};
}
// Every enabled dgmv_if in one pass over i < first.y.size() (the outputs share their length).
template <class Ex, class Op0, class... Ops> BATCHLAS_DLA_INLINE void dgmv_many(const Ex& ex, const Op0& op0, const Ops&... ops) {
    auto one = [](const auto& op, int i) {
        using T = elem_t<decltype(op.y)>;
        if (op.on) op.y(i) = tag_mul(as_tag(op.alpha), Arith<T>::mul(static_cast<T>(op.d(i)), static_cast<T>(op.x(i))));
    };
    for_each_owned(ex, op0.y.size(), [&](int i) {
        one(op0, i);
        (one(ops, i), ...);
    });
}

// x(i) := 0 where pred(i); true when pred held for every i < x.size().
template <Scope S = Scope::Auto, class Ex, class X, class P>
BATCHLAS_DLA_INLINE bool zero_if(const Ex& ex, const X& x, P&& pred) {
    using T = elem_t<X>;
    if constexpr (is_thread_v<Ex>) {
        bool all = true;
        for (int i = 0; i < x.size(); ++i) {
            if (pred(i)) x(i) = Arith<T>::zero();
            else all = false;
        }
        return all;
    } else {
        int all = 1;
        for_each_owned(ex, x.size(), [&](int i) {
            if (pred(i)) x(i) = Arith<T>::zero();
            else all = 0;
        });
        return detail::scope_reduce<S>(ex, all, sycl::minimum<int>()) != 0;
    }
}

// ---- plane rotations -----------------------------------------------------------------------

// BLAS rot: [x; y] := [c s; -conj(s) c] [x; y] with real c (drot; zdrot for real s; zrot).
// The real form rounds exactly as  x' = c x + s y,  y' = -s x + c y.
template <class Ex, class X, class Y, class C, class S>
BATCHLAS_DLA_INLINE void rot(const Ex& ex, const X& x, const Y& y, C c, S s) {
    using T = elem_t<X>;
    using A = Arith<T>;
    using R = real_t<T>;
    for_each_owned(ex, x.size(), [&](int i) {
        const T x0 = x(i);
        const T x1 = y(i);
        if constexpr (!A::is_complex) {
            x(i) = c * x0 + s * x1;
            y(i) = -s * x0 + c * x1;
        } else if constexpr (std::is_same_v<std::remove_cv_t<S>, R>) {
            x(i) = A::make(c * A::re(x0) + s * A::re(x1), c * A::im(x0) + s * A::im(x1));
            y(i) = A::make(-s * A::re(x0) + c * A::re(x1), -s * A::im(x0) + c * A::im(x1));
        } else {
            const T cx0 = A::make(c * A::re(x0), c * A::im(x0));
            const T cx1 = A::make(c * A::re(x1), c * A::im(x1));
            x(i) = A::add(cx0, A::mul(s, x1));
            y(i) = A::sub(cx1, A::mul(A::conj(s), x0));
        }
    });
}

// ---- reductions --------------------------------------------------------------------------

// max over i < n of f(i) by sycl::fmax, starting from init: a NaN f(i) is skipped (unlike amax,
// which propagates it). The cross-item step is a maximum of non-NaN partials.
template <Scope S = Scope::Auto, class Ex, class F, class R>
BATCHLAS_DLA_INLINE R fmax_reduce(const Ex& ex, int n, F&& f, R init) {
    R m = init;
    for_each_owned(ex, n, [&](int i) { m = sycl::fmax(m, static_cast<R>(f(i))); });
    return detail::scope_reduce<S>(ex, m, sycl::maximum<R>());
}
// max(init, max |re| + |im|) with fmax semantics.
template <Scope S = Scope::Auto, class Ex, class X, class R>
BATCHLAS_DLA_INLINE R amax_fmax(const Ex& ex, const X& x, R init) {
    return fmax_reduce<S>(ex, x.size(), [&](int i) { return abs1(x(i)); }, init);
}

// Index of the minimum of f(i), i < n; ties -> the lowest index. An item that never improves on
// max() reports its first index (lid), so an input of max/NaN values returns 0; n <= 0 -> 0.
template <Scope S = Scope::Auto, class Ex, class F> BATCHLAS_DLA_INLINE int iminloc(const Ex& ex, int n, F&& f) {
    using R = std::remove_cvref_t<decltype(f(0))>;
    R best = std::numeric_limits<R>::max();
    int bi = ex.lid();
    for_each_owned(ex, n, [&](int i) {
        const R v = f(i);
        if (v < best) {
            best = v;
            bi = i;
        }
    });
    const R m = detail::scope_reduce<S>(ex, best, sycl::minimum<R>());
    return detail::scope_reduce<S>(ex, best == m ? bi : n, sycl::minimum<int>());
}

// Lowest i with pred(x(i)), or -1. On Thread the scan stops at the first hit.
template <Scope S = Scope::Auto, class Ex, class X, class P> BATCHLAS_DLA_INLINE int ifirst(const Ex& ex, const X& x, P&& pred) {
    if constexpr (is_thread_v<Ex>) {
        for (int i = 0; i < x.size(); ++i)
            if (pred(x(i))) return i;
        return -1;
    } else {
        int k = INT_MAX;
        for_each_owned(ex, x.size(), [&](int i) {
            if (k == INT_MAX && pred(x(i))) k = i;
        });
        k = detail::scope_reduce<S>(ex, k, sycl::minimum<int>());
        return k == INT_MAX ? -1 : k;
    }
}

// True when some x(i) is NaN. On Thread the scan stops at the first NaN.
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE bool any_nan(const Ex& ex, const X& x) {
    if constexpr (is_thread_v<Ex>) {
        for (int i = 0; i < x.size(); ++i)
            if (!(x(i) == x(i))) return true;
        return false;
    } else {
        int f = 0;
        for_each_owned(ex, x.size(), [&](int i) { f |= !(x(i) == x(i)) ? 1 : 0; });
        return detail::scope_reduce<S>(ex, f, sycl::maximum<int>()) != 0;
    }
}

// {sum_i |x_k(i)|^2} for every operand, one pass over i < first.size(), then one scope-S
// reduction per operand in order: an unscaled nrm2^2 of each.
template <Scope S = Scope::Auto, class Ex, class X0, class... Xs>
BATCHLAS_DLA_INLINE auto sumsq_many(const Ex& ex, const X0& x0, const Xs&... xs) {
    using T = elem_t<X0>;
    using A = Arith<T>;
    using R = real_t<T>;
    constexpr int K = 1 + sizeof...(Xs);
    std::array<R, K> acc{};
    auto add = [](R& a, T v) {
        if constexpr (A::is_complex) {
            a += A::re(v) * A::re(v);
            a += A::im(v) * A::im(v);
        } else {
            a += v * v;
        }
    };
    for_each_owned(ex, x0.size(), [&](int i) {
        int k = 0;
        add(acc[k++], x0(i));
        (add(acc[k++], xs(i)), ...);
    });
    // One reduction per operand, in order (a braced list evaluates left to right).
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::array<R, K>{detail::scope_reduce<S>(ex, acc[I])...};
    }(std::make_index_sequence<K>{});
}

// ---- sorting -------------------------------------------------------------------------------

// Selection sort of x into order O on one item; swap(i, j) runs after each transposition so
// the caller permutes its vectors alongside (dbdsqr's sort: one swap per position). Ties keep
// the lower index; a NaN never wins a comparison.
template <SortOrder O, class X, class Sw> BATCHLAS_DLA_INLINE void select_sort(const Thread&, const X& x, Sw&& swap_fn) {
    using T = elem_t<X>;
    const int n = x.size();
    for (int i = 0; i < n; ++i) {
        int best = i;
        for (int j = i + 1; j < n; ++j) {
            if constexpr (O == SortOrder::Descending) {
                if (x(j) > x(best)) best = j;
            } else {
                if (x(j) < x(best)) best = j;
            }
        }
        if (best != i) {
            const T t = x(i);
            x(i) = x(best);
            x(best) = t;
            swap_fn(i, best);
        }
    }
}

}  // namespace batchlas::device::v2
