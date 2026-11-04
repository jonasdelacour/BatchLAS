// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 operands: thin int32 views whose extents, strides, memory space and layout are
// types. A Dyn extent is one int; an Ext<N> extent is free. Indexing is int32 (D7).

#include <cstddef>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>

namespace batchlas::device::inline v2 {

// ---- address bases ------------------------------------------------------------------------
// How an operand reaches element k. B = T* (default); Aligned<T, A>: a pointer whose alignment
// the compiler is told at every access (16-byte vector LDS/STS survive a sub-view: sygst);
// sycl::local_accessor<T, 1>: the access stays in the shared address space with [R+UR]
// addressing. Never build a view from &acc[off]: that pointer is generic (sytrd cd Upper
// generic LD 61->109, syev_cta_fused). Idx is the type of the index sum: int (D7, the default)
// or std::ptrdiff_t, v1's 64-bit index math where a strict port needs it.
template <class T, int A> struct Aligned {
    static_assert(A > 0 && (A & (A - 1)) == 0, "v2::Aligned: A is a power of two");
    T* p;
    template <class I> BATCHLAS_DLA_INLINE T& operator[](I k) const {
        return static_cast<T*>(__builtin_assume_aligned(p, A))[k];
    }
};

namespace detail {
template <class Idx, class B> BATCHLAS_DLA_INLINE decltype(auto) at(const B& p, int off, int i, int inc) {
    if constexpr (std::is_same_v<Idx, int>) return p[off + i * inc];
    else return p[static_cast<Idx>(off) + static_cast<Idx>(i) * static_cast<Idx>(inc)];
}
template <class Idx, class B> BATCHLAS_DLA_INLINE decltype(auto) at2(const B& p, int off, int i, int j, int ld) {
    if constexpr (std::is_same_v<Idx, int>) return p[off + j * ld + i];
    else return p[static_cast<Idx>(off) + static_cast<Idx>(j) * static_cast<Idx>(ld) + static_cast<Idx>(i)];
}
}  // namespace detail

template <class T, class N = Dyn, class Inc = Unit, Mem M = Mem::Global, class L = Cyclic, class B = T*, class Idx = int>
struct Vec {
    using value_type = std::remove_cv_t<T>;
    using layout = L;
    using base_type = B;
    using index_type = Idx;
    static constexpr Mem mem = M;
    B p;
    N n;
    Inc inc{};
    int off = 0;  // int32 element offset from p: sub-views never move the 64-bit pointer

    BATCHLAS_DLA_INLINE T& operator()(int i) const { return detail::at<Idx>(p, off, i, int(inc)); }
    BATCHLAS_DLA_INLINE int size() const { return int(n); }
    // Elements [k, n).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L, B, Idx> tail(int k) const { return {p, Dyn{int(n) - k}, inc, off + k * int(inc)}; }
    // Elements [0, k).
    BATCHLAS_DLA_INLINE Vec<T, Dyn, Inc, M, L, B, Idx> head(int k) const { return {p, Dyn{k}, inc, off}; }
};

template <class T, class R = Dyn, class C = Dyn, class Ld = Dyn, Mem M = Mem::Global, class L = Cyclic, class B = T*,
          class Idx = int>
struct Mat {
    using value_type = std::remove_cv_t<T>;
    using layout = L;
    using base_type = B;
    using index_type = Idx;
    static constexpr Mem mem = M;
    B p;
    R m;
    C n;
    Ld ld;
    int off = 0;  // as Vec::off: the column-major index is one int32 sum, as in KernelMatrixView

    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return detail::at2<Idx>(p, off, i, j, int(ld)); }
    BATCHLAS_DLA_INLINE int rows() const { return int(m); }
    BATCHLAS_DLA_INLINE int cols() const { return int(n); }
    BATCHLAS_DLA_INLINE Vec<T, R, Unit, M, Cyclic, B, Idx> col(int j) const { return {p, m, Unit{}, off + j * int(ld)}; }
    BATCHLAS_DLA_INLINE Vec<T, C, Ld, M, Cyclic, B, Idx> row(int i) const { return {p, n, ld, off + i}; }
    template <class R2, class C2> BATCHLAS_DLA_INLINE Mat<T, R2, C2, Ld, M, L, B, Idx> sub(int i, int j, R2 r, C2 c) const {
        return {p, r, c, ld, off + i + j * int(ld)};
    }
    BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Ld, M, L, B, Idx> sub(int i, int j, int r, int c) const {
        return {p, Dyn{r}, Dyn{c}, ld, off + i + j * int(ld)};
    }
    // The trailing block A(k:, k:).
    BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Ld, M, L, B, Idx> tail(int k) const { return sub(k, k, int(m) - k, int(n) - k); }
};

// A register fragment; every index into v is a compile-time constant in the op bodies. Ops
// take Frag& in/out, never by value or as a return (sygst: a by-value Frag hoisted every load,
// REG 37 -> 70; trsm: a returned Frag doubled REG in local memory).
template <class T, int N, class L = RowFrag> struct Frag {
    static constexpr int size = N;
    T v[N];
    BATCHLAS_DLA_INLINE T& operator[](int i) { return v[i]; }
    BATCHLAS_DLA_INLINE const T& operator[](int i) const { return v[i]; }
};

// An R x C register tile (one item's block of a matrix); indices are compile-time constants.
template <class T, int R, int C> struct Tile {
    static constexpr int rows = R, cols = C;
    T v[R][C];
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) { return v[i][j]; }
    BATCHLAS_DLA_INLINE const T& operator()(int i, int j) const { return v[i][j]; }
};

// ---- Frag / Tile moves (compile-time-unrolled; a live bound masks the tail) ----------------
// f[k] := x(k) for k < live, else zero (x(k) is not read past live).
template <class T, int N, class L, class V> BATCHLAS_DLA_INLINE void load(Frag<T, N, L>& f, const V& x, int live = N) {
#pragma unroll
    for (int k = 0; k < N; ++k) f[k] = k < live ? static_cast<T>(x(k)) : Arith<T>::zero();
}
// x(k) := f[k] for k < live.
template <class T, int N, class L, class V> BATCHLAS_DLA_INLINE void store(const Frag<T, N, L>& f, const V& x, int live = N) {
#pragma unroll
    for (int k = 0; k < N; ++k)
        if (k < live) x(k) = f[k];
}
template <class T, int N, class L> BATCHLAS_DLA_INLINE void fill(Frag<T, N, L>& f, const T& v) {
#pragma unroll
    for (int k = 0; k < N; ++k) f[k] = v;
}
// RowFrag: f[j] := A(r, j) for j < live (the item's row r of a matrix).
template <class T, int N, class MA> BATCHLAS_DLA_INLINE void load_row(Frag<T, N, RowFrag>& f, const MA& a, int r, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j) f[j] = j < live ? static_cast<T>(a(r, j)) : Arith<T>::zero();
}
template <class T, int N, class MA> BATCHLAS_DLA_INLINE void store_row(const Frag<T, N, RowFrag>& f, const MA& a, int r, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j)
        if (j < live) a(r, j) = f[j];
}
// t(i, j) := A(i, j) inside the live_r x live_c corner, zero outside (A is not read there).
template <class T, int R, int C, class MA> BATCHLAS_DLA_INLINE void load(Tile<T, R, C>& t, const MA& a, int live_r = R, int live_c = C) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = (i < live_r && j < live_c) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
}
template <class T, int R, int C, class MA>
BATCHLAS_DLA_INLINE void store(const Tile<T, R, C>& t, const MA& a, int live_r = R, int live_c = C) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i)
            if (i < live_r && j < live_c) a(i, j) = t(i, j);
}
template <class T, int R, int C> BATCHLAS_DLA_INLINE void fill(Tile<T, R, C>& t, const T& v) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = v;
}

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

template <int P, bool Masked, class Impl>
template <class V>
BATCHLAS_DLA_INLINE Published<V> Partition<P, Masked, Impl>::publish(V v) const {
    barrier();
    return {v};
}
template <Scope S, int SG>
template <class V>
BATCHLAS_DLA_INLINE Published<V> RtGroup<S, SG>::publish(V v) const {
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
    static_assert(is_published_v<V> || is_thread_v<Ex>,
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
// 64-bit index math (v1's) on a raw pointer.
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M, Cyclic, T*, std::ptrdiff_t> vec_wide(T* p, int n, int inc = 1) {
    return {p, Dyn{n}, Dyn{inc}};
}
template <Mem M = Mem::Global, class T>
BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M, Cyclic, T*, std::ptrdiff_t> mat_wide(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}
// A pointer aligned to A bytes (the caller guarantees it).
template <int A, Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, M, Cyclic, Aligned<T, A>> vec_aligned(T* p, int n) {
    return {Aligned<T, A>{p}, Dyn{n}};
}
template <int A, Mem M = Mem::Global, class T>
BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M, Cyclic, Aligned<T, A>> mat_aligned(T* p, int m, int n, int ld) {
    return {Aligned<T, A>{p}, Dyn{m}, Dyn{n}, Dyn{ld}};
}
// Local memory through its accessor, at an int32 element offset (never &acc[off]).
template <class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Unit, Mem::Local, Cyclic, sycl::local_accessor<T, 1>> vec(const sycl::local_accessor<T, 1>& a, int n, int off = 0) {
    return {a, Dyn{n}, Unit{}, off};
}
template <class T>
BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, sycl::local_accessor<T, 1>> mat(const sycl::local_accessor<T, 1>& a, int m, int n, int ld, int off = 0) {
    return {a, Dyn{m}, Dyn{n}, Dyn{ld}, off};
}
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Vec<T, Dyn, Dyn, M> vec(T* p, int n, int inc) {
    return {p, Dyn{n}, Dyn{inc}};
}
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}

}  // namespace batchlas::device
