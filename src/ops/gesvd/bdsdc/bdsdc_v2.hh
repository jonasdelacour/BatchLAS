// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to bdsdc's Golub-Kahan kernels (BdsdcBuildGk, BdsdcExtract) until a
// second user moves them to core: a length-limited item(), lazy row/column lines of an item
// matrix, a strided slice, the perfect shuffle, and a fused pair of diagonal-scaled copies.
// Lazy operands index through the parent's own element expression, so every address is the v1
// view's expression (strict SASS parity with the hand kernels).

#include <type_traits>

#include <batchlas/device/interop.hh>
#include <batchlas/device/level1.hh>

namespace batchlas::device::inline v2::bdsdc {
using device::interleave;  // core level1.hh

// item(v, b) restricted to its first n elements.
template <class T> BATCHLAS_DLA_INLINE ItemVec<T> item_n(const VectorView<T>& v, int b, int n) {
    return {v.data_ptr(), b, v.stride(), v.inc(), n};
}

// Column j / row i of a matrix operand, first n elements.
template <class Mx> struct Col {
    using value_type = std::remove_cvref_t<decltype(std::declval<const Mx&>()(0, 0))>;
    using layout = Cyclic;
    Mx a;
    int j, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class Mx> struct Row {
    using value_type = std::remove_cvref_t<decltype(std::declval<const Mx&>()(0, 0))>;
    using layout = Cyclic;
    Mx a;
    int i, n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int j) const { return a(i, j); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class Mx> BATCHLAS_DLA_INLINE Col<Mx> col(const Mx& a, int j, int n) { return {a, j, n}; }
template <class Mx> BATCHLAS_DLA_INLINE Row<Mx> row(const Mx& a, int i, int n) { return {a, i, n}; }

// v(start + k * step) for k < n.
template <class V> struct Sliced {
    using value_type = elem_t<V>;
    using layout = Cyclic;
    V v;
    int start, n, step;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int k) const { return v(start + k * step); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class V> BATCHLAS_DLA_INLINE Sliced<V> slice(const V& v, int start, int n, int step) {
    return {v, start, n, step};
}

// y := alpha diag(d) x when `on`, rounded as alpha * (d_i x_i); a deferred op for dgmv_many.
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
// Every enabled dgmv_if in one pass over i < op0.y.size() (the outputs share their length).
template <class Ex, class Op0, class... Ops>
BATCHLAS_DLA_INLINE void dgmv_many(const Ex& ex, const Op0& op0, const Ops&... ops) {
    auto one = [](const auto& op, int i) {
        using T = elem_t<decltype(op.y)>;
        if (op.on) op.y(i) = tag_mul(as_tag(op.alpha), Arith<T>::mul(static_cast<T>(op.d(i)), static_cast<T>(op.x(i))));
    };
    for_each_owned(ex, op0.y.size(), [&](int i) {
        one(op0, i);
        (one(ops, i), ...);
    });
}

}  // namespace batchlas::device::bdsdc
