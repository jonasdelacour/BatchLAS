// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 operands: thin int32 views whose extents, strides, memory space and layout are
// types. A Dyn extent is one int; an Ext<N> extent is free. Indexing is int32 (D7).

#include <type_traits>

#include <batchlas/blas/device/v2/core.hh>
#include <batchlas/blas/device/v2/exec.hh>

namespace batchlas::device::v2 {

template <class T, class N = Dyn, class Inc = Unit, Mem M = Mem::Global, class L = Cyclic> struct Vec {
    using value_type = std::remove_cv_t<T>;
    using layout = L;
    static constexpr Mem mem = M;
    T* p;
    N n;
    Inc inc{};

    BATCHLAS_DLA_INLINE T& operator()(int i) const { return p[i * int(inc)]; }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    // Elements [k, n).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L> tail(int k) const { return {p + k * int(inc), Dyn{int(n) - k}, inc}; }
    // Elements [0, k).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L> head(int k) const { return {p, Dyn{k}, inc}; }
};

template <class T, class R = Dyn, class C = Dyn, class Ld = Dyn, Mem M = Mem::Global, class L = Cyclic> struct Mat {
    using value_type = std::remove_cv_t<T>;
    using layout = L;
    static constexpr Mem mem = M;
    T* p;
    R m;
    C n;
    Ld ld;

    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i + j * int(ld)]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
    BATCHLAS_DLA_INLINE Vec<T, R, Unit, M> col(int j) const { return {p + j * int(ld), m}; }
    BATCHLAS_DLA_INLINE Vec<T, C, Ld, M> row(int i) const { return {p + i, n, ld}; }
    template <class R2, class C2> BATCHLAS_DLA_INLINE Mat<T, R2, C2, Ld, M, L> sub(int i, int j, R2 r, C2 c) const {
        return {p + i + j * int(ld), r, c, ld};
    }
    BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Ld, M, L> sub(int i, int j, int r, int c) const {
        return {p + i + j * int(ld), Dyn{r}, Dyn{c}, ld};
    }
    // The trailing block A(k:, k:).
    BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Ld, M, L> tail(int k) const { return sub(k, k, int(m) - k, int(n) - k); }
};

// A register fragment; every index into v is a compile-time constant in the op bodies.
template <class T, int N, class L = RowFrag> struct Frag {
    T v[N];
    BATCHLAS_DLA_INLINE T& operator[](int i) { return v[i]; }
    BATCHLAS_DLA_INLINE const T& operator[](int i) const { return v[i]; }
};

// D4: an operand every item of the executor may read. Only WorkGroup::publish (a barrier) or
// assume_published (input not written in this kernel, or written under a barrier the caller
// owns) make one; an op that reads across owners demands it at compile time.
template <class V> struct Published {
    V v;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
    BATCHLAS_DLA_INLINE Published<decltype(std::declval<V>().tail(0))> tail(int k) const { return {v.tail(k)}; }
};
template <class V> BATCHLAS_DLA_INLINE Published<V> assume_published(V v) { return {v}; }

template <int WG, int SG>
template <class V>
BATCHLAS_DLA_INLINE Published<V> WorkGroup<WG, SG>::publish(V v) const {
    barrier(V::mem == Mem::Local ? Mem::Local : Mem::Global);
    return {v};
}

template <class V> inline constexpr bool is_published_v = false;
template <class V> inline constexpr bool is_published_v<Published<V>> = true;

namespace detail {
template <class V> BATCHLAS_DLA_INLINE const V& unpub(const V& v) { return v; }
template <class V> BATCHLAS_DLA_INLINE const V& unpub(const Published<V>& v) { return v.v; }
// An operand read at indices other items own must be Published unless one item is the executor.
template <class Ex, class V> constexpr void require_published() {
    static_assert(is_published_v<V> || Ex::size == 1,
                  "v2: this operand is read across owners; pass ex.publish(v) or assume_published(v)");
}
}  // namespace detail

// A Householder vector whose element `head` is an implied 1 and whose elements above it are 0
// (LAPACK's unit-head convention): the stored value at `head` is never read.
template <class V> struct UnitHead {
    V v;
    int head;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        using A = Arith<value_type>;
        return i == head ? A::one() : (i > head ? static_cast<value_type>(v(i)) : A::zero());
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
    BATCHLAS_DLA_INLINE UnitHead<decltype(std::declval<V>().tail(0))> tail(int k) const { return {v.tail(k), head - k}; }
};
template <class V> BATCHLAS_DLA_INLINE UnitHead<V> unit_head(V v, int head) { return {v, head}; }

// Factories.
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, M> vec(T* p, int n) { return {p, Dyn{n}}; }
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> vec(T* p, int n, int inc) {
    return {p, Dyn{n}, Dyn{inc}};
}
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}

}  // namespace batchlas::device::v2
