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
    int off = 0;  // int32 element offset from p: sub-views never move the 64-bit pointer

    BATCHLAS_DLA_INLINE T& operator()(int i) const { return p[off + i * int(inc)]; }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    // Elements [k, n).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L> tail(int k) const { return {p, Dyn{int(n) - k}, inc, off + k * int(inc)}; }
    // Elements [0, k).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L> head(int k) const { return {p, Dyn{k}, inc, off}; }
};

template <class T, class R = Dyn, class C = Dyn, class Ld = Dyn, Mem M = Mem::Global, class L = Cyclic> struct Mat {
    using value_type = std::remove_cv_t<T>;
    using layout = L;
    static constexpr Mem mem = M;
    T* p;
    R m;
    C n;
    Ld ld;
    int off = 0;  // as Vec::off: the column-major index is one int32 sum, as in KernelMatrixView

    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[off + j * int(ld) + i]; }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
    BATCHLAS_DLA_INLINE Vec<T, R, Unit, M> col(int j) const { return {p, m, Unit{}, off + j * int(ld)}; }
    BATCHLAS_DLA_INLINE Vec<T, C, Ld, M> row(int i) const { return {p, n, ld, off + i}; }
    template <class R2, class C2> BATCHLAS_DLA_INLINE Mat<T, R2, C2, Ld, M, L> sub(int i, int j, R2 r, C2 c) const {
        return {p, r, c, ld, off + i + j * int(ld)};
    }
    BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Ld, M, L> sub(int i, int j, int r, int c) const {
        return {p, Dyn{r}, Dyn{c}, ld, off + i + j * int(ld)};
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
    template <class W = V> BATCHLAS_DLA_INLINE Published<decltype(std::declval<W>().tail(0))> tail(int k) const {
        return {v.tail(k)};
    }
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

// A(k:, k:) or x(k:) addressed in the parent's coordinates (LAPACK panel indexing). An op on a
// Trailing operand loops i in [k, n) with i owned by lid == (i - k) mod size: the same owners as
// the tail view, so the two compose without a barrier. The loop bounds keep the hand kernels'
// shape, and with it LLVM's unrolling (a 0-based tail loop unrolls differently: latrd P3).
namespace detail {
template <class V> constexpr auto operand_elem() {
    if constexpr (requires(const V& v) { v(0, 0); }) return std::remove_cvref_t<decltype(std::declval<const V&>()(0, 0))>{};
    else return std::remove_cvref_t<decltype(std::declval<const V&>()(0))>{};
}
template <class V> constexpr Mem operand_mem() {
    if constexpr (requires { V::mem; }) return V::mem;
    else return Mem::Global;  // a v1 view (KernelMatrixView, VectorView): global memory
}
}  // namespace detail

template <class V> struct Trailing {
    V v;
    int k;
    int n;  // one past the last row/element, in the parent's coordinates
    using value_type = decltype(detail::operand_elem<V>());
    static constexpr Mem mem = detail::operand_mem<V>();
    BATCHLAS_DLA_INLINE int first() const { return k; }
    BATCHLAS_DLA_INLINE int end() const { return n; }
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(i); }
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return v(i, j); }
};
// Any operand with operator()(i[, j]): a v2 view or a v1 view (KernelMatrixView after
// batch_item, whose plain-inline operator() a strict port of a hand kernel keeps).
template <class V> BATCHLAS_DLA_INLINE Trailing<V> trailing(V v, int k, int n) { return {v, k, n}; }
template <class V> BATCHLAS_DLA_INLINE Trailing<V> trailing(V v, int k) {
    if constexpr (requires { v.rows(); }) return {v, k, v.rows()};
    else return {v, k, v.size()};
}

template <class V> inline constexpr bool is_trailing_v = false;
template <class V> inline constexpr bool is_trailing_v<Trailing<V>> = true;

// Factories.
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, M> vec(T* p, int n) { return {p, Dyn{n}}; }
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> vec(T* p, int n, int inc) {
    return {p, Dyn{n}, Dyn{inc}};
}
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}

}  // namespace batchlas::device::v2
