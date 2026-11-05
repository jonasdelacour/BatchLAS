// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 operands: thin int32 views whose extents, strides, memory space and layout are
// types. A Dyn extent is one int; an Ext<N> extent is free. Indexing is int32 (D7).

#include <cstddef>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/workspace.hh>

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
// The general form: refl::QR (zero / stored past the head as above) or refl::QL (xORG2L: stored
// above the head, zero below); unit_fill::Zero (the other side reads 0) or unit_fill::Keep (the
// other side reads the stored values: LAPACK's temporary unit of a stored reflector). In every
// form v(head) is never read.
namespace refl {
struct QR {};
struct QL {};
}  // namespace refl
namespace unit_fill {
struct Zero {};
struct Keep {};
}  // namespace unit_fill
template <class Dir, class Fill, class V> struct UnitHeadOf {
    static_assert(std::is_same_v<Dir, refl::QR> || std::is_same_v<Dir, refl::QL>, "unit_head: refl::QR or refl::QL");
    static_assert(std::is_same_v<Fill, unit_fill::Zero> || std::is_same_v<Fill, unit_fill::Keep>, "unit_head: unit_fill::Zero or Keep");
    V v;
    int head;
    using value_type = typename V::value_type;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const {
        using A = Arith<value_type>;
        if (i == head) return A::one();
        if constexpr (std::is_same_v<Fill, unit_fill::Keep>) return static_cast<value_type>(v(i));
        else if constexpr (std::is_same_v<Dir, refl::QR>) return i > head ? static_cast<value_type>(v(i)) : A::zero();
        else return i < head ? static_cast<value_type>(v(i)) : A::zero();
    }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class Dir, class Fill = unit_fill::Zero, class V> BATCHLAS_DLA_INLINE UnitHeadOf<Dir, Fill, V> unit_head(V v, int head) {
    return {v, head};
}

// ---- register vectors and staged operands (v22 design A.2) --------------------------------------
// One element per item, held in a register: element i belongs to the item with lid == i; the
// window [lo, hi) is the live part (owns(ex)). Every op on it reads x.v on the owning item only.
template <class T> struct LaneVec {
    using value_type = std::remove_cv_t<T>;
    using layout = OnePerItem;
    T& v;
    int lo;
    int hi;
    BATCHLAS_DLA_INLINE T& operator()(int) const { return v; }
    BATCHLAS_DLA_INLINE int size() const { return hi; }
    template <class Ex> BATCHLAS_DLA_INLINE bool owns(const Ex& ex) const {
        const int l = ex.lid();
        return l >= lo && l < hi;
    }
};
template <class T> BATCHLAS_DLA_INLINE LaneVec<T> lane_vec(T& v, int lo, int hi) { return {v, lo, hi}; }

// Element k of a lane-cyclic register vector (lane l holds elements l + size * r in f[r]), to
// every lane of the executor (one bcast; f[k / size] selected by unrolled compares). COLLECTIVE.
// k >= 0. A per-component select: a complex `v = f[r]` became a select of addresses into f, which
// kept sytrs' whole Frag in local memory (STL 56 -> 134).
template <class Ex, class T, int R, class L> BATCHLAS_DLA_HELPER T frag_bcast(const Ex& ex, const Frag<T, R, L>& f, int k) {
    constexpr int P = Ex::size;
    static_assert(P > 0 && (P & (P - 1)) == 0, "frag_bcast: a power-of-two executor");
    constexpr int S = __builtin_ctz(P);
    T v = f[0];
#pragma unroll
    for (int r = 1; r < R; ++r)
        v = select((k >> S) == r, f[r], v);
    return ex.bcast(v, k & (P - 1));
}
// The neighbours of every element of a lane-cyclic register vector: up[r] = element e + 1 and
// dn[r] = element e - 1 for this lane's element e = lane + size * r (wrapping lanes read the
// adjacent register of lane 0 / lane size - 1; the vector's two ends: unspecified). COLLECTIVE.
template <class Ex, class T, int R, class L>
BATCHLAS_DLA_INLINE void neighbor_exchange(const Ex& ex, const Frag<T, R, L>& x, Frag<T, R, L>& up, Frag<T, R, L>& dn) {
    constexpr int P = Ex::size;
    const int lane = ex.lane();
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const T u = ex.shuffle(x[r], (lane + 1) & (P - 1));
        const T d = ex.shuffle(x[r], (lane + P - 1) & (P - 1));
        const T hi = ex.bcast(x[r + 1 < R ? r + 1 : r], 0);
        const T lo = ex.bcast(x[r > 0 ? r - 1 : 0], P - 1);
        up[r] = select(lane == P - 1 && r + 1 < R, hi, u);
        dn[r] = select(lane == 0 && r > 0, lo, d);
    }
}

// RowFrag row moves on part P of a matrix, columns c0 .. c0 + N - 1: f[j] := A(r, c0 + j) for
// (r, c0 + j) in P and j < live; outside P, Herm reads conj(A(c0 + j, r)) (the stored mirror of a
// Hermitian matrix, Im(diag) dropped), else zero. store_row writes only (r, c0 + j) in P, j < live.
template <Part P, bool Herm = false, class T, int N, class MA>
BATCHLAS_DLA_INLINE void load_row(Frag<T, N, RowFrag>& f, const MA& a, int r, int c0, int live = N) {
    using A = Arith<T>;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int c = c0 + j;
        T v = A::zero();
        if (j < live) {
            if (detail::in_part<P>(r, c)) v = static_cast<T>(a(r, c));
            else if constexpr (Herm) v = A::conj(static_cast<T>(a(c, r)));
            if constexpr (Herm && A::is_complex)
                if (r == c) v = A::make(A::re(v), real_t<T>(0));
        }
        f[j] = v;
    }
}
template <Part P, class T, int N, class MA>
BATCHLAS_DLA_INLINE void store_row(const Frag<T, N, RowFrag>& f, const MA& a, int r, int c0, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j)
        if (j < live && detail::in_part<P>(r, c0 + j)) a(r, c0 + j) = f[j];
}

// A matrix with both strides named: element (i, j) at p[i * rs + j * cs] (a row-major tile is
// rs = C, cs = 1). Ext strides fold.
template <class T, class RS = Dyn, class CS = Unit, Mem M = Mem::Global, class B = T*> struct StridedMat {
    using value_type = std::remove_cv_t<T>;
    static constexpr Mem mem = M;
    B p;
    int m;
    int n;
    RS rs{};
    CS cs{};
    BATCHLAS_DLA_INLINE T& operator()(int i, int j) const { return p[i * int(rs) + j * int(cs)]; }
    BATCHLAS_DLA_INLINE int rows() const { return m; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <Mem M = Mem::Global, class T> BATCHLAS_DLA_INLINE StridedMat<T, Dyn, Dyn, M> mat_strided(T* p, int m, int n, int rs, int cs) {
    return {p, m, n, Dyn{rs}, Dyn{cs}};
}
// Local memory through a decorated local pointer (a Workspace slice, workspace.hh): the access
// stays in the shared address space. tile_local<R, C>: compile-time extents.
template <class T>
BATCHLAS_DLA_INLINE Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, local_mptr<T>> mat_local(local_mptr<T> p, int m, int n, int ld, int off = 0) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}, off};
}
template <int R, int C, class T>
BATCHLAS_DLA_INLINE Mat<T, Ext<R>, Ext<C>, Dyn, Mem::Local, Cyclic, local_mptr<T>> tile_local(local_mptr<T> p, int ld, int off = 0) {
    return {p, Ext<R>{}, Ext<C>{}, Dyn{ld}, off};
}

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
