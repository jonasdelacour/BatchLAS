// SPDX-License-Identifier: MIT
#pragma once
/** @file
 * @brief Device operands: Vec, Mat, Frag, Tile and the views and factories built on them.
 * @ingroup api_device_operands
 *
 * Operands are thin views whose extents, strides, memory space and layout are types: a Dyn
 * extent is one int, an `Ext<N>` extent is free. Indexing is int32 unless an operand's Idx says
 * otherwise.
 */

#include <cstddef>
#include <type_traits>

#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

/// @brief A pointer whose alignment the compiler is told at every access: an operand address base.
/// @ingroup api_device_operands
///
/// An operand reaches element k through its base B: `T*` (the default), `Aligned<T, A>`, or
/// `sycl::local_accessor<T, 1>`, whose accesses stay in the shared address space. Never build a
/// view from `&acc[off]`: that pointer is generic; pass the accessor and an offset instead.
/// @tparam T the element type
/// @tparam A the alignment in bytes, a power of two; the caller guarantees it
// Aligned keeps 16-byte vector LDS/STS through a sub-view (sygst); a view from &acc[off]
// compiled to generic loads (sytrd cd Upper generic LD 61->109, syev_cta_fused).
template <class T, int A> struct Aligned {
    static_assert(A > 0 && (A & (A - 1)) == 0, "Aligned: A is a power of two");
    T* p;  ///< The pointer, aligned to A bytes.
    /// @brief Element k, read through an A-aligned pointer.
    /// @param k the element index
    /// @return a reference to `p[k]`
    template <class I> BATCHLAS_DEVICE_INLINE T& operator[](I k) const {
        return static_cast<T*>(__builtin_assume_aligned(p, A))[k];
    }
};

namespace detail {
template <class Idx, class B> BATCHLAS_DEVICE_INLINE decltype(auto) at(const B& p, int off, int i, int inc) {
    if constexpr (std::is_same_v<Idx, int>) return p[off + i * inc];
    else return p[static_cast<Idx>(off) + static_cast<Idx>(i) * static_cast<Idx>(inc)];
}
template <class Idx, class B> BATCHLAS_DEVICE_INLINE decltype(auto) at2(const B& p, int off, int i, int j, int ld) {
    if constexpr (std::is_same_v<Idx, int>) return p[off + j * ld + i];
    else return p[static_cast<Idx>(off) + static_cast<Idx>(j) * static_cast<Idx>(ld) + static_cast<Idx>(i)];
}
}  // namespace detail

/// @brief A vector operand: n elements at p + off, stride inc.
/// @ingroup api_device_operands
///
/// Every element access is `p[off + i * inc]`, computed in Idx. An op distributes it over the
/// executor's layout L.
/// @tparam T the element type (const for a read-only operand)
/// @tparam N the extent type: Dyn or `Ext<N>`
/// @tparam Inc the stride type: Unit (1), `Ext<K>` or Dyn
/// @tparam M the memory space
/// @tparam L the layout: who owns element i (Cyclic by default)
/// @tparam B the address base: `T*`, `Aligned<T, A>` or `sycl::local_accessor<T, 1>`
/// @tparam Idx the index type of the address sum: int, or std::ptrdiff_t for 64-bit index math
template <class T, class N = Dyn, class Inc = Unit, Mem M = Mem::Global, class L = Cyclic, class B = T*, class Idx = int>
struct Vec {
    using value_type = std::remove_cv_t<T>;  ///< The element type without cv.
    using layout = L;  ///< The layout tag L.
    using base_type = B;  ///< The address base type B.
    using index_type = Idx;  ///< The index type Idx.
    static constexpr Mem mem = M;  ///< The memory space M.
    B p;  ///< The address base.
    N n;  ///< The number of elements.
    Inc inc{};  ///< The stride between elements.
    int off = 0;  ///< The int32 element offset from p: sub-views never move the 64-bit pointer.

    /// @brief Element i.
    /// @param i the index, 0 to size() - 1
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const { return detail::at<Idx>(p, off, i, int(inc)); }
    /// @brief The number of elements.
    BATCHLAS_DEVICE_INLINE int size() const { return int(n); }
    /// @brief Elements [k, n) as a vector of n - k elements.
    /// @param k the first element, 0 to n
    /// @return the tail view
    BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Inc, M, L, B, Idx> tail(int k) const { return {p, Dyn{int(n) - k}, inc, off + k * int(inc)}; }
    /// @brief Elements [0, k).
    /// @param k the number of elements
    /// @return the head view
    BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Inc, M, L, B, Idx> head(int k) const { return {p, Dyn{k}, inc, off}; }
};

/// @brief A column-major matrix operand: m x n elements at p + off, leading dimension ld.
/// @ingroup api_device_operands
///
/// Element (i, j) is `p[off + j * ld + i]`, one sum computed in Idx.
/// @tparam T the element type (const for a read-only operand)
/// @tparam R the row extent type: Dyn or `Ext<M>`
/// @tparam C the column extent type: Dyn or `Ext<N>`
/// @tparam Ld the leading-dimension type: Dyn or `Ext<K>`
/// @tparam M the memory space
/// @tparam L the layout: who owns row i (Cyclic by default)
/// @tparam B the address base: `T*`, `Aligned<T, A>`, `local_mptr<T>` or
///         `sycl::local_accessor<T, 1>`
/// @tparam Idx the index type of the address sum: int, or std::ptrdiff_t for 64-bit index math
template <class T, class R = Dyn, class C = Dyn, class Ld = Dyn, Mem M = Mem::Global, class L = Cyclic, class B = T*,
          class Idx = int>
struct Mat {
    using value_type = std::remove_cv_t<T>;  ///< The element type without cv.
    using layout = L;  ///< The layout tag L.
    using base_type = B;  ///< The address base type B.
    using index_type = Idx;  ///< The index type Idx.
    static constexpr Mem mem = M;  ///< The memory space M.
    B p;  ///< The address base.
    R m;  ///< The number of rows.
    C n;  ///< The number of columns.
    Ld ld;  ///< The leading dimension.
    int off = 0;  ///< The int32 element offset from p, as Vec::off.

    /// @brief Element (i, j).
    /// @param i the row, 0 to rows() - 1
    /// @param j the column, 0 to cols() - 1
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE T& operator()(int i, int j) const { return detail::at2<Idx>(p, off, i, j, int(ld)); }
    /// @brief The number of rows.
    BATCHLAS_DEVICE_INLINE int rows() const { return int(m); }
    /// @brief The number of columns.
    BATCHLAS_DEVICE_INLINE int cols() const { return int(n); }
    /// @brief Column j as a vector of rows() elements, stride 1.
    /// @param j the column
    /// @return the column view
    BATCHLAS_DEVICE_INLINE Vec<T, R, Unit, M, Cyclic, B, Idx> col(int j) const { return {p, m, Unit{}, off + j * int(ld)}; }
    /// @brief Row i as a vector of cols() elements, stride ld.
    /// @param i the row
    /// @return the row view
    BATCHLAS_DEVICE_INLINE Vec<T, C, Ld, M, Cyclic, B, Idx> row(int i) const { return {p, n, ld, off + i}; }
    /// @brief The r x c block at (i, j), with the extent types of r and c.
    /// @tparam R2 the row extent type
    /// @tparam C2 the column extent type
    /// @param i the block's first row
    /// @param j the block's first column
    /// @param r the block's rows
    /// @param c the block's columns
    /// @return the block view
    template <class R2, class C2> BATCHLAS_DEVICE_INLINE Mat<T, R2, C2, Ld, M, L, B, Idx> sub(int i, int j, R2 r, C2 c) const {
        return {p, r, c, ld, off + i + j * int(ld)};
    }
    /// @brief The r x c block at (i, j).
    /// @param i the block's first row
    /// @param j the block's first column
    /// @param r the block's rows
    /// @param c the block's columns
    /// @return the block view
    BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Ld, M, L, B, Idx> sub(int i, int j, int r, int c) const {
        return {p, Dyn{r}, Dyn{c}, ld, off + i + j * int(ld)};
    }
    /// @brief The trailing block A(k:, k:).
    /// @param k the first row and column
    /// @return the block view
    BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Ld, M, L, B, Idx> tail(int k) const { return sub(k, k, int(m) - k, int(n) - k); }
};

/// @brief A register fragment of N values; every index into v is a compile-time constant in the op
/// bodies.
/// @ingroup api_device_operands
///
/// Ops take `Frag&` in and out, never by value or as a return value.
/// @tparam T the element type
/// @tparam N the number of values
/// @tparam L the layout: RowFrag (lane l holds row l) by default
// A by-value Frag hoisted every load (sygst REG 37 -> 70); a returned Frag doubled REG in local
// memory (trsm).
template <class T, int N, class L = RowFrag> struct Frag {
    static constexpr int size = N;  ///< The number of values, N.
    T v[N];  ///< The values.
    /// @brief Value i.
    /// @param i the index, a compile-time constant in practice
    /// @return a reference to the value
    BATCHLAS_DEVICE_INLINE T& operator[](int i) { return v[i]; }
    /// @brief Value i.
    /// @param i the index, a compile-time constant in practice
    /// @return a const reference to the value
    BATCHLAS_DEVICE_INLINE const T& operator[](int i) const { return v[i]; }
};

/// @brief An R x C register tile: one item's block of a matrix; indices are compile-time
/// constants.
/// @ingroup api_device_operands
/// @tparam T the element type
/// @tparam R the rows
/// @tparam C the columns
template <class T, int R, int C> struct Tile {
    ///@{
    /// @brief The rows R and the columns C.
    static constexpr int rows = R, cols = C;
    ///@}
    /// @brief The values, row-major.
    T v[R][C];
    /// @brief Entry (i, j).
    /// @param i the row
    /// @param j the column
    /// @return a reference to the entry
    BATCHLAS_DEVICE_INLINE T& operator()(int i, int j) { return v[i][j]; }
    /// @brief Entry (i, j).
    /// @param i the row
    /// @param j the column
    /// @return a const reference to the entry
    BATCHLAS_DEVICE_INLINE const T& operator()(int i, int j) const { return v[i][j]; }
};

/// @brief Calls `f(i0 + x, j0 + y, t(x, y))` at every tile entry, row by row: the tile's global
/// coordinates.
/// @ingroup api_device_operands
/// @param t the tile
/// @param i0 the tile's first row in the matrix
/// @param j0 the tile's first column in the matrix
/// @param f the callable, `f(int i, int j, T& t_ij)`
template <class T, int R, int C, class F> BATCHLAS_DEVICE_INLINE void for_each(const Thread&, Tile<T, R, C>& t, int i0, int j0, F&& f) {
BATCHLAS_UNROLL_FULL
    for (int x = 0; x < R; ++x) {
BATCHLAS_UNROLL_FULL
        for (int y = 0; y < C; ++y) f(i0 + x, j0 + y, t(x, y));
    }
}

/// @brief `f[k] := x(k)` for k < live, else zero; x(k) is not read past live.
/// @ingroup api_device_operands
///
/// The Frag and Tile moves are unrolled at compile time; a live bound masks the tail.
/// @param f the fragment
/// @param x the source vector
/// @param live the number of elements to read
template <class T, int N, class L, class V> BATCHLAS_DEVICE_INLINE void load(Frag<T, N, L>& f, const V& x, int live = N) {
#pragma unroll
    for (int k = 0; k < N; ++k) f[k] = k < live ? static_cast<T>(x(k)) : Arith<T>::zero();
}
/// @brief `x(k) := f[k]` for k < live.
/// @ingroup api_device_operands
/// @param f the fragment
/// @param x the destination vector
/// @param live the number of elements to write
template <class T, int N, class L, class V> BATCHLAS_DEVICE_INLINE void store(const Frag<T, N, L>& f, const V& x, int live = N) {
#pragma unroll
    for (int k = 0; k < N; ++k)
        if (k < live) x(k) = f[k];
}
/// @brief Every value of f set to v.
/// @ingroup api_device_operands
/// @param f the fragment
/// @param v the value
template <class T, int N, class L> BATCHLAS_DEVICE_INLINE void fill(Frag<T, N, L>& f, const T& v) {
#pragma unroll
    for (int k = 0; k < N; ++k) f[k] = v;
}
/// @brief RowFrag: `f[j] := A(r, j)` for j < live, else zero: the item's row r of a matrix.
/// @ingroup api_device_operands
/// @param f the fragment
/// @param a the matrix
/// @param r the row
/// @param live the number of columns to read
template <class T, int N, class MA> BATCHLAS_DEVICE_INLINE void load_row(Frag<T, N, RowFrag>& f, const MA& a, int r, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j) f[j] = j < live ? static_cast<T>(a(r, j)) : Arith<T>::zero();
}
/// @brief RowFrag: `A(r, j) := f[j]` for j < live.
/// @ingroup api_device_operands
/// @param f the fragment
/// @param a the matrix
/// @param r the row
/// @param live the number of columns to write
template <class T, int N, class MA> BATCHLAS_DEVICE_INLINE void store_row(const Frag<T, N, RowFrag>& f, const MA& a, int r, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j)
        if (j < live) a(r, j) = f[j];
}
/// @brief `t(i, j) := A(i, j)` inside the live_r x live_c corner, zero outside; A is not read there.
/// @ingroup api_device_operands
/// @param t the tile
/// @param a the source matrix
/// @param live_r the rows to read
/// @param live_c the columns to read
template <class T, int R, int C, class MA> BATCHLAS_DEVICE_INLINE void load(Tile<T, R, C>& t, const MA& a, int live_r = R, int live_c = C) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = (i < live_r && j < live_c) ? static_cast<T>(a(i, j)) : Arith<T>::zero();
}
/// @brief `A(i, j) := t(i, j)` inside the live_r x live_c corner.
/// @ingroup api_device_operands
/// @param t the tile
/// @param a the destination matrix
/// @param live_r the rows to write
/// @param live_c the columns to write
template <class T, int R, int C, class MA>
BATCHLAS_DEVICE_INLINE void store(const Tile<T, R, C>& t, const MA& a, int live_r = R, int live_c = C) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i)
            if (i < live_r && j < live_c) a(i, j) = t(i, j);
}
/// @brief Every entry of t set to v.
/// @ingroup api_device_operands
/// @param t the tile
/// @param v the value
template <class T, int R, int C> BATCHLAS_DEVICE_INLINE void fill(Tile<T, R, C>& t, const T& v) {
#pragma unroll
    for (int j = 0; j < C; ++j)
#pragma unroll
        for (int i = 0; i < R; ++i) t(i, j) = v;
}

/// @brief An operand every item of the executor may read.
/// @ingroup api_device_operands
///
/// Only an executor's publish (a barrier) or assume_published (the input is not written in this
/// kernel, or is written under a barrier the caller owns) makes one. An op that reads across
/// owners demands it at compile time.
/// @tparam V the operand type
template <class V> struct Published {
    V v;  ///< The operand.
    using value_type = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;  ///< The operand's element type.
    /// @brief Element i of the operand.
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return v(i); }
    /// @brief The operand's size().
    BATCHLAS_DEVICE_INLINE int size() const { return v.size(); }
    /// @brief The operand's tail(k), still published.
    /// @param k the first element
    /// @return the published tail view
    template <class W = V> BATCHLAS_DEVICE_INLINE Published<decltype(std::declval<W>().tail(0))> tail(int k) const {
        return {v.tail(k)};
    }
};
/// @brief Marks v as readable by every item, with no barrier: the caller vouches for it.
/// @ingroup api_device_operands
///
/// v must not be written in this kernel, or be written under a barrier the caller owns.
/// @param v the operand
/// @return v as a Published operand
template <class V> BATCHLAS_DEVICE_INLINE Published<V> assume_published(V v) { return {v}; }

template <int WG, int SG>
template <class V>
BATCHLAS_DEVICE_INLINE Published<V> WorkGroup<WG, SG>::publish(V v) const {
    barrier(V::mem == Mem::Local ? Mem::Local : Mem::Global);
    return {v};
}

template <int P, bool Masked, class Impl>
template <class V>
BATCHLAS_DEVICE_INLINE Published<V> Partition<P, Masked, Impl>::publish(V v) const {
    barrier();
    return {v};
}
template <Scope S, int SG>
template <class V>
BATCHLAS_DEVICE_INLINE Published<V> RtGroup<S, SG>::publish(V v) const {
    barrier(V::mem == Mem::Local ? Mem::Local : Mem::Global);
    return {v};
}

/// @brief True when V is a Published operand.
/// @ingroup api_device_operands
/// @tparam V the operand type
template <class V> inline constexpr bool is_published_v = false;
/// @cond
template <class V> inline constexpr bool is_published_v<Published<V>> = true;
/// @endcond

namespace detail {
template <class V> BATCHLAS_DEVICE_INLINE const V& unpub(const V& v) { return v; }
template <class V> BATCHLAS_DEVICE_INLINE const V& unpub(const Published<V>& v) { return v.v; }
// An operand read at indices other items own must be Published unless one item is the executor.
template <class Ex, class V> constexpr void require_published() {
    static_assert(is_published_v<V> || is_thread_v<Ex>,
                  "device op: this operand is read across owners; pass ex.publish(v) or assume_published(v)");
}
}  // namespace detail

/// @brief A Householder vector whose element head is an implied 1 and whose elements above it are
/// 0: LAPACK's unit-head convention.
/// @ingroup api_device_operands
///
/// The stored value at head is never read.
/// @tparam V the stored vector's operand type
template <class V> struct UnitHead {
    V v;  ///< The stored vector.
    int head;  ///< The index of the implied 1.
    using value_type = typename V::value_type;  ///< The stored vector's element type.
    /// @brief Element i: 1 at head, the stored value below it, 0 above it.
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE value_type operator()(int i) const {
        using A = Arith<value_type>;
        return i == head ? A::one() : (i > head ? static_cast<value_type>(v(i)) : A::zero());
    }
    /// @brief The stored vector's size().
    BATCHLAS_DEVICE_INLINE int size() const { return v.size(); }
    /// @brief The same reflector on v(k:), its head moved to head - k.
    /// @param k the first element
    /// @return the tail view
    BATCHLAS_DEVICE_INLINE UnitHead<decltype(std::declval<V>().tail(0))> tail(int k) const { return {v.tail(k), head - k}; }
};
/// @brief The unit-head view of v: v(head) reads 1, the elements above it 0.
/// @ingroup api_device_operands
/// @param v the stored vector
/// @param head the index of the implied 1
/// @return the UnitHead view
template <class V> BATCHLAS_DEVICE_INLINE UnitHead<V> unit_head(V v, int head) { return {v, head}; }
/// @brief The side of the head a reflector is stored on: QR (below it, xGEQR2) or QL (above it,
/// xORG2L).
/// @ingroup api_device_operands
enum class Refl { QR, QL };
/// @brief A unit-head reflector view in its general form.
/// @ingroup api_device_operands
///
/// Refl::QR reads the stored values past the head and zero before it; Refl::QL reads them before
/// the head and zero past it. DiagFill::Zero makes the other side read 0; DiagFill::Keep makes it
/// read the stored values (LAPACK's temporary unit of a stored reflector). In every form v(head)
/// is never read.
/// @tparam Dir Refl::QR or Refl::QL
/// @tparam Fill DiagFill::Zero or DiagFill::Keep
/// @tparam V the stored vector's operand type
template <Refl Dir, DiagFill Fill, class V> struct UnitHeadOf {
    static_assert(Fill == DiagFill::Zero || Fill == DiagFill::Keep, "unit_head: DiagFill::Zero or Keep");
    V v;  ///< The stored vector.
    int head;  ///< The index of the implied 1.
    using value_type = typename V::value_type;  ///< The stored vector's element type.
    /// @brief Element i: 1 at head, else as Dir and Fill say.
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE value_type operator()(int i) const {
        using A = Arith<value_type>;
        if (i == head) return A::one();
        if constexpr (Fill == DiagFill::Keep) return static_cast<value_type>(v(i));
        else if constexpr (Dir == Refl::QR) return i > head ? static_cast<value_type>(v(i)) : A::zero();
        else return i < head ? static_cast<value_type>(v(i)) : A::zero();
    }
    /// @brief The stored vector's size().
    BATCHLAS_DEVICE_INLINE int size() const { return v.size(); }
    /// @brief The same reflector on v(k:).
    ///
    /// head - k may fall outside the tail; then every element is on one side of it.
    /// @param k the first element
    /// @return the tail view
    BATCHLAS_DEVICE_INLINE auto tail(int k) const { return UnitHeadOf<Dir, Fill, decltype(v.tail(k))>{v.tail(k), head - k}; }
};
/// @brief The unit-head view of v in the general form: `unit_head<Refl::QL>(v, head)`.
/// @ingroup api_device_operands
/// @tparam Dir Refl::QR or Refl::QL
/// @tparam Fill DiagFill::Zero (default) or DiagFill::Keep
/// @param v the stored vector
/// @param head the index of the implied 1
/// @return the UnitHeadOf view
template <Refl Dir, DiagFill Fill = DiagFill::Zero, class V> BATCHLAS_DEVICE_INLINE UnitHeadOf<Dir, Fill, V> unit_head(V v, int head) {
    return {v, head};
}

/// @brief One element per item, held in a register: element i belongs to the item with lid == i.
/// @ingroup api_device_operands
///
/// The window [lo, hi) is the live part (owns(ex)). Every op on it reads x.v on the owning item
/// only.
/// @tparam T the element type
template <class T> struct LaneVec {
    using value_type = std::remove_cv_t<T>;  ///< The element type without cv.
    using layout = OnePerItem;  ///< OnePerItem.
    T& v;  ///< This item's element.
    int lo;  ///< The first live index.
    int hi;  ///< One past the last live index.
    /// @brief This item's element, whatever the index (the op reads it on the owner only).
    /// @return a reference to v
    BATCHLAS_DEVICE_INLINE T& operator()(int) const { return v; }
    /// @brief The vector's size, hi.
    BATCHLAS_DEVICE_INLINE int size() const { return hi; }
    /// @brief True when the calling item's element is live: lo <= `ex.lid()` < hi.
    /// @param ex the executor
    template <class Ex> BATCHLAS_DEVICE_INLINE bool owns(const Ex& ex) const {
        const int l = ex.lid();
        return l >= lo && l < hi;
    }
};
/// @brief A LaneVec over this item's register v, live on items [lo, hi).
/// @ingroup api_device_operands
/// @param v this item's element
/// @param lo the first live index
/// @param hi one past the last live index
/// @return the LaneVec
template <class T> BATCHLAS_DEVICE_INLINE LaneVec<T> lane_vec(T& v, int lo, int hi) { return {v, lo, hi}; }

/// @brief COLLECTIVE: element k of a lane-cyclic register vector, on every lane of the executor.
/// @ingroup api_device_operands
///
/// Lane l holds elements l + size * r in `f[r]`. One bcast; `f[k / size]` is selected by unrolled
/// compares.
/// @tparam Ex the executor type; its size is a power of two
/// @param ex the executor
/// @param f the lane-cyclic fragment
/// @param k the element, k >= 0
/// @return element k
// A per-component select: a complex `v = f[r]` became a select of addresses into f, which
// kept sytrs' whole Frag in local memory (STL 56 -> 134).
template <class Ex, class T, int R, class L> BATCHLAS_DEVICE_HELPER T frag_bcast(const Ex& ex, const Frag<T, R, L>& f, int k) {
    constexpr int P = Ex::size;
    static_assert(P > 0 && (P & (P - 1)) == 0, "frag_bcast: a power-of-two executor");
    constexpr int S = __builtin_ctz(P);
    T v = f[0];
#pragma unroll
    for (int r = 1; r < R; ++r)
        v = select((k >> S) == r, f[r], v);
    return ex.bcast(v, k & (P - 1));
}
/// @brief COLLECTIVE: the neighbours of every element of a lane-cyclic register vector.
/// @ingroup api_device_operands
///
/// For this lane's element e = lane + size * r, `up[r]` is element e + 1 and `dn[r]` element e - 1;
/// lanes that wrap read the adjacent register of lane 0 or lane size - 1. At the vector's two
/// ends the result is unspecified.
/// @tparam Ex the executor type; its size is a power of two
/// @param ex the executor
/// @param x the lane-cyclic fragment
/// @param up receives the next elements
/// @param dn receives the previous elements
template <class Ex, class T, int R, class L>
BATCHLAS_DEVICE_INLINE void neighbor_exchange(const Ex& ex, const Frag<T, R, L>& x, Frag<T, R, L>& up, Frag<T, R, L>& dn) {
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

/// @brief RowFrag row load on part P of a matrix, columns c0 to c0 + N - 1.
/// @ingroup api_device_operands
///
/// `f[j] := A(r, c0 + j)` for (r, c0 + j) in P and j < live. Outside P, TS = ConjTrans reads
/// `conj(A(c0 + j, r))` (the stored mirror of a Hermitian matrix, the imaginary part of the
/// diagonal dropped) and TS = NoTrans reads zero.
/// @tparam P the part of the matrix that is stored
/// @tparam TS NoTrans (zero outside P) or ConjTrans (the Hermitian mirror)
/// @param f the fragment
/// @param a the matrix
/// @param r the row
/// @param c0 the first column
/// @param live the number of columns to read
template <Part P, Transpose TS = Transpose::NoTrans, class Scalar, int N, class MA>
BATCHLAS_DEVICE_INLINE void load_row(Frag<Scalar, N, RowFrag>& f, const MA& a, int r, int c0, int live = N) {
    static_assert(TS != Transpose::Trans, "load_row: TS is NoTrans (zero outside P) or ConjTrans (the Hermitian mirror)");
    constexpr bool herm = TS == Transpose::ConjTrans;
    using A = Arith<Scalar>;
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const int c = c0 + j;
        Scalar v = A::zero();
        if (j < live) {
            if (detail::in_part<P>(r, c)) v = static_cast<Scalar>(a(r, c));
            else if constexpr (herm) v = A::conj(static_cast<Scalar>(a(c, r)));
            if constexpr (herm && A::is_complex)
                if (r == c) v = A::make(A::re(v), real_t<Scalar>(0));
        }
        f[j] = v;
    }
}
/// @brief RowFrag row store on part P: writes only `A(r, c0 + j)` in P, for j < live.
/// @ingroup api_device_operands
/// @tparam P the part of the matrix to write
/// @param f the fragment
/// @param a the matrix
/// @param r the row
/// @param c0 the first column
/// @param live the number of columns to write
template <Part P, class T, int N, class MA>
BATCHLAS_DEVICE_INLINE void store_row(const Frag<T, N, RowFrag>& f, const MA& a, int r, int c0, int live = N) {
#pragma unroll
    for (int j = 0; j < N; ++j)
        if (j < live && detail::in_part<P>(r, c0 + j)) a(r, c0 + j) = f[j];
}

/// @brief A matrix with both strides named: element (i, j) at `p[i * rs + j * cs]`.
/// @ingroup api_device_operands
///
/// A row-major tile is rs = C, cs = 1. `Ext` strides fold at compile time.
/// @tparam T the element type
/// @tparam RS the row stride type
/// @tparam CS the column stride type
/// @tparam M the memory space
/// @tparam B the address base
template <class T, class RS = Dyn, class CS = Unit, Mem M = Mem::Global, class B = T*> struct StridedMat {
    using value_type = std::remove_cv_t<T>;  ///< The element type without cv.
    static constexpr Mem mem = M;  ///< The memory space M.
    B p;  ///< The address base.
    int m;  ///< The number of rows.
    int n;  ///< The number of columns.
    RS rs{};  ///< The row stride.
    CS cs{};  ///< The column stride.
    /// @brief Element (i, j).
    /// @param i the row
    /// @param j the column
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE T& operator()(int i, int j) const { return p[i * int(rs) + j * int(cs)]; }
    /// @brief The number of rows.
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    /// @brief The number of columns.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
/// @brief An m x n StridedMat at p with run-time strides.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param m the rows
/// @param n the columns
/// @param rs the row stride
/// @param cs the column stride
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE StridedMat<T, Dyn, Dyn, M> mat_strided(T* p, int m, int n, int rs, int cs) {
    return {p, m, n, Dyn{rs}, Dyn{cs}};
}
/// @brief An m x n matrix in local memory through a decorated local pointer (a Workspace slice).
/// @ingroup api_device_operands
///
/// The access stays in the shared address space.
/// @param p the decorated local pointer
/// @param m the rows
/// @param n the columns
/// @param ld the leading dimension
/// @param off the int32 element offset from p
/// @return the operand
template <class T>
BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, local_mptr<T>> mat_local(local_mptr<T> p, int m, int n, int ld, int off = 0) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}, off};
}
/// @brief An R x C matrix in local memory through a decorated local pointer, extents at compile
/// time.
/// @ingroup api_device_operands
/// @tparam R the rows
/// @tparam C the columns
/// @param p the decorated local pointer
/// @param ld the leading dimension
/// @param off the int32 element offset from p
/// @return the operand
template <int R, int C, class T>
BATCHLAS_DEVICE_INLINE Mat<T, Ext<R>, Ext<C>, Dyn, Mem::Local, Cyclic, local_mptr<T>> tile_local(local_mptr<T> p, int ld, int off = 0) {
    return {p, Ext<R>{}, Ext<C>{}, Dyn{ld}, off};
}

// The loop bounds keep the hand kernels' shape, and with it LLVM's unrolling (a 0-based tail loop
// unrolls differently: latrd P3).
namespace detail {
template <class V> constexpr auto operand_elem() {
    if constexpr (requires(const V& v) { v(0, 0); }) return std::remove_cvref_t<decltype(std::declval<const V&>()(0, 0))>{};
    else return std::remove_cvref_t<decltype(std::declval<const V&>()(0))>{};
}
template <class V> constexpr Mem operand_mem() {
    if constexpr (requires { V::mem; }) return V::mem;
    else return Mem::Global;  // a host view (KernelMatrixView, VectorView): global memory
}
}  // namespace detail

/// @brief A(k:, k:) or x(k:) addressed in the parent's coordinates: LAPACK panel indexing.
/// @ingroup api_device_operands
///
/// An op on a Trailing operand loops i in [k, n) with i owned by lid == (i - k) mod size: the same
/// owners as the tail view, so the two compose without a barrier.
/// @tparam V the parent operand type: a device view or a host view
template <class V> struct Trailing {
    V v;  ///< The parent operand.
    int k;  ///< The first row or element, in the parent's coordinates.
    int n;  ///< One past the last row or element, in the parent's coordinates.
    using value_type = decltype(detail::operand_elem<V>());  ///< The parent's element type.
    static constexpr Mem mem = detail::operand_mem<V>();  ///< The parent's memory space (Mem::Global for a host view).
    /// @brief The first index, k.
    BATCHLAS_DEVICE_INLINE int first() const { return k; }
    /// @brief One past the last index, n.
    BATCHLAS_DEVICE_INLINE int end() const { return n; }
    /// @brief Element i of the parent.
    /// @param i the index, k to n - 1
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return v(i); }
    /// @brief Element (i, j) of the parent.
    /// @param i the row, k to n - 1
    /// @param j the column
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const { return v(i, j); }
};
/// @brief v(k:n) in the parent's coordinates.
/// @ingroup api_device_operands
///
/// v is any operand with `operator()(i[, j])`: a device view or a host view (a KernelMatrixView
/// after batch_item).
/// @param v the parent operand
/// @param k the first index
/// @param n one past the last index
/// @return the Trailing view
template <class V> BATCHLAS_DEVICE_INLINE Trailing<V> trailing(V v, int k, int n) { return {v, k, n}; }
/// @brief v(k:) in the parent's coordinates, to v.rows() (a matrix) or v.size() (a vector).
/// @ingroup api_device_operands
/// @param v the parent operand
/// @param k the first index
/// @return the Trailing view
template <class V> BATCHLAS_DEVICE_INLINE Trailing<V> trailing(V v, int k) {
    if constexpr (requires { v.rows(); }) return {v, k, v.rows()};
    else return {v, k, v.size()};
}

/// @brief True when V is a Trailing view.
/// @ingroup api_device_operands
/// @tparam V the operand type
template <class V> inline constexpr bool is_trailing_v = false;
/// @cond
template <class V> inline constexpr bool is_trailing_v<Trailing<V>> = true;

/// @endcond

/// @brief A vector of n contiguous elements at p.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param n the number of elements
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Unit, M> vec(T* p, int n) { return {p, Dyn{n}}; }
/// @brief A vector of n elements at p, stride inc, with 64-bit index math.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param n the number of elements
/// @param inc the stride
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Dyn, M, Cyclic, T*, std::ptrdiff_t> vec_wide(T* p, int n, int inc = 1) {
    return {p, Dyn{n}, Dyn{inc}};
}
/// @brief A column-major m x n matrix at p with 64-bit index math.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param m the rows
/// @param n the columns
/// @param ld the leading dimension
/// @return the operand
template <Mem M = Mem::Global, class T>
BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, M, Cyclic, T*, std::ptrdiff_t> mat_wide(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}
/// @brief A vector of n contiguous elements at a pointer aligned to A bytes (the caller
/// guarantees it).
/// @ingroup api_device_operands
/// @tparam A the alignment in bytes, a power of two
/// @tparam M the memory space
/// @param p the first element
/// @param n the number of elements
/// @return the operand
template <int A, Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Unit, M, Cyclic, Aligned<T, A>> vec_aligned(T* p, int n) {
    return {Aligned<T, A>{p}, Dyn{n}};
}
/// @brief A column-major m x n matrix at a pointer aligned to A bytes (the caller guarantees it).
/// @ingroup api_device_operands
/// @tparam A the alignment in bytes, a power of two
/// @tparam M the memory space
/// @param p the first element
/// @param m the rows
/// @param n the columns
/// @param ld the leading dimension
/// @return the operand
template <int A, Mem M = Mem::Global, class T>
BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, M, Cyclic, Aligned<T, A>> mat_aligned(T* p, int m, int n, int ld) {
    return {Aligned<T, A>{p}, Dyn{m}, Dyn{n}, Dyn{ld}};
}
/// @brief A vector of n elements of local memory through its accessor, at an int32 element offset.
/// @ingroup api_device_operands
///
/// Pass the accessor, never `&acc[off]`: that pointer is generic.
/// @param a the local accessor
/// @param n the number of elements
/// @param off the int32 element offset
/// @return the operand
template <class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Unit, Mem::Local, Cyclic, sycl::local_accessor<T, 1>> vec(const sycl::local_accessor<T, 1>& a, int n, int off = 0) {
    return {a, Dyn{n}, Unit{}, off};
}
/// @brief A column-major m x n matrix of local memory through its accessor, at an int32 element
/// offset.
/// @ingroup api_device_operands
/// @param a the local accessor
/// @param m the rows
/// @param n the columns
/// @param ld the leading dimension
/// @param off the int32 element offset
/// @return the operand
template <class T>
BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, Mem::Local, Cyclic, sycl::local_accessor<T, 1>> mat(const sycl::local_accessor<T, 1>& a, int m, int n, int ld, int off = 0) {
    return {a, Dyn{m}, Dyn{n}, Dyn{ld}, off};
}
/// @brief A vector of n elements at p, stride inc.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param n the number of elements
/// @param inc the stride
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Vec<T, Dyn, Dyn, M> vec(T* p, int n, int inc) {
    return {p, Dyn{n}, Dyn{inc}};
}
/// @brief A column-major m x n matrix at p, leading dimension ld.
/// @ingroup api_device_operands
/// @tparam M the memory space
/// @param p the first element
/// @param m the rows
/// @param n the columns
/// @param ld the leading dimension
/// @return the operand
template <Mem M = Mem::Global, class T> BATCHLAS_DEVICE_INLINE Mat<T, Dyn, Dyn, Dyn, M> mat(T* p, int m, int n, int ld) {
    return {p, Dyn{m}, Dyn{n}, Dyn{ld}};
}

}  // namespace batchlas::device
