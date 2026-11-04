// SPDX-License-Identifier: MIT
#pragma once
// gebrd-private device LA v2 pieces the v2.1 core lacks: lacgv, the read-only adaptors
// conj_view / herm, and a gemv_acc y0 that adds nothing. The rest comes from v2/dla.hh.

#include <batchlas/device.hh>

namespace batchlas::gebrd_v2 {

namespace v2 = batchlas::device;

// x(i) := conj(x(i)) on every owned i (LAPACK lacgv); nothing for real T. No barrier.
template <class Ex, class X> BATCHLAS_DLA_INLINE void lacgv(const Ex& ex, const X& x) {
    using T = v2::elem_t<X>;
    if constexpr (v2::Arith<T>::is_complex) {
        v2::map(ex, x, [](const T& v) { return v2::Arith<T>::conj(v); }, x);
    }
}

// conj(v(i)), read only.
template <class V> struct ConjView {
    V v;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return v2::Arith<value_type>::conj(v(i)); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE ConjView<V> conj_view(const V& v) { return {v}; }

// A^H as a matrix operand: (i, j) reads conj(a(j, i)).
template <class M> struct HermView {
    M a;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i, int j) const { return v2::Arith<value_type>::conj(a(j, i)); }
    BATCHLAS_DLA_INLINE int rows() const { return a.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.rows(); }
};
template <class M> BATCHLAS_DLA_INLINE HermView<M> herm(const M& a) { return {a}; }

// -0 everywhere: as gemv_acc's y0 the add folds (-0 + x == x; +0 + -0 would be +0).
template <class T> struct NegZero {
    BATCHLAS_DLA_INLINE T operator()(int) const {
        using R = v2::real_t<T>;
        return v2::Arith<T>::make(-R(0), -R(0));
    }
};

}  // namespace batchlas::gebrd_v2
