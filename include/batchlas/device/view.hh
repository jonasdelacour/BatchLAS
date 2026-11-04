// SPDX-License-Identifier: MIT
#pragma once
// Device LA lazy views (v22 design A.2 view.hh): index maps over any operand (a Vec-like x(i)
// with size(), or a Mat-like a(i, j) with rows() / cols()). A view holds its base by value and
// computes the index; nothing is copied. Views whose element is a function of the stored one
// (conj, op with ConjTrans, herm off its stored triangle) are read-only and return values.
// Also: the packed-triangle index map PackedTri and the Packet load/store unit.

#include <cstddef>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/enums.hh>
#include <batchlas/device/core.hh>

namespace batchlas::device::inline v2 {

namespace detail {
template <class V> inline constexpr bool is_vec_like_v = requires(const V& v) { v.size(); v(0); };
template <class M> inline constexpr bool is_mat_like_v = requires(const M& a) { a.rows(); a.cols(); a(0, 0); };
template <class V> using view_elem_t = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;
template <class M> using view_elem2_t = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;
}  // namespace detail

// ---- sub-operands as free functions (the members of Mat / Vec, for any operand) ----------
template <class M> BATCHLAS_DLA_INLINE auto row(const M& a, int i) { return a.row(i); }
template <class M> BATCHLAS_DLA_INLINE auto col(const M& a, int j) { return a.col(j); }
template <class M> BATCHLAS_DLA_INLINE auto window(const M& a, int i0, int j0, int m, int n) { return a.sub(i0, j0, m, n); }

// ---- flip: reversed index order ---------------------------------------------------------
namespace axis {
struct Rows {};
struct Cols {};
struct Both {};
}  // namespace axis

// Vector: x(n - 1 - i) when on (a run-time switch, uniform), else x(i).
template <class V> struct Flipped {
    V v;
    bool on;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(on ? v.size() - 1 - i : i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
// Matrix: rows (m - 1 - i), columns (n - 1 - j) or both reversed when on.
template <class Axis, class M> struct FlippedMat {
    M a;
    bool on;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (!std::is_same_v<Axis, axis::Cols>) i = on ? a.rows() - 1 - i : i;
        if constexpr (!std::is_same_v<Axis, axis::Rows>) j = on ? a.cols() - 1 - j : j;
        return a(i, j);
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
};
template <class Axis = axis::Rows, class X> BATCHLAS_DLA_INLINE auto flip(const X& x, bool on = true) {
    static_assert(std::is_same_v<Axis, axis::Rows> || std::is_same_v<Axis, axis::Cols> || std::is_same_v<Axis, axis::Both>,
                  "flip: axis::Rows, axis::Cols or axis::Both");
    if constexpr (detail::is_mat_like_v<X>) return FlippedMat<Axis, X>{x, on};
    else return Flipped<X>{x, on};
}

// ---- conj views ---------------------------------------------------------------------------
// conj(x(i)) (or conj(a(i, j))) when on, by value; the identity on real T.
template <class X> struct ConjView {
    X x;
    bool on;
    template <class... I> BATCHLAS_DLA_INLINE auto operator()(I... i) const {
        using T = std::remove_cvref_t<decltype(x(i...))>;
        const T v = x(i...);
        if constexpr (Arith<T>::is_complex) return on ? Arith<T>::conj(v) : v;
        else return v;
    }
    BATCHLAS_DLA_INLINE int size() const requires detail::is_vec_like_v<X> { return x.size(); }
    BATCHLAS_DLA_INLINE int rows() const requires detail::is_mat_like_v<X> { return x.rows(); }
    BATCHLAS_DLA_INLINE int cols() const requires detail::is_mat_like_v<X> { return x.cols(); }
};
template <class X> BATCHLAS_DLA_INLINE ConjView<X> conj_view(const X& x) { return {x, true}; }
// Compile-time switch: the operand itself when !C (no view, no branch).
template <class X> inline constexpr bool is_operand_v = detail::is_vec_like_v<X> || detail::is_mat_like_v<X>;
template <bool C, class X>
    requires is_operand_v<X>
BATCHLAS_DLA_INLINE auto conj_if(const X& x) {
    if constexpr (C) return ConjView<X>{x, true};
    else return x;
}
// Run-time switch (uniform): one select per element.
template <class X>
    requires is_operand_v<X>
BATCHLAS_DLA_INLINE ConjView<X> conj_if(const X& x, bool on) { return {x, on}; }

// ---- op(A) --------------------------------------------------------------------------------
// op(A)(i, j) for a compile-time Transpose: NoTrans is A itself; Trans swaps indices (writable);
// ConjTrans is conj(A(j, i)) by value.
template <Transpose TR, class M> struct OpView {
    M a;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (TR == Transpose::Trans) {
            return a(j, i);
        } else {
            using T = detail::view_elem2_t<M>;
            return Arith<T>::conj(static_cast<T>(a(j, i)));
        }
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.rows(); }
};
template <Transpose TR, class M> BATCHLAS_DLA_INLINE auto op_view(const M& a) {
    if constexpr (TR == Transpose::NoTrans) return a;
    else return OpView<TR, M>{a};
}
// The run-time form (uniform tr): by value, one branch per element.
template <class M> struct RtOpView {
    M a;
    Transpose tr;
    BATCHLAS_DLA_INLINE auto operator()(int i, int j) const {
        using T = detail::view_elem2_t<M>;
        if (tr == Transpose::NoTrans) return static_cast<T>(a(i, j));
        const T v = a(j, i);
        return tr == Transpose::ConjTrans ? Arith<T>::conj(v) : v;
    }
    BATCHLAS_DLA_INLINE int rows() const { return tr == Transpose::NoTrans ? a.rows() : a.cols(); }
    BATCHLAS_DLA_INLINE int cols() const { return tr == Transpose::NoTrans ? a.cols() : a.rows(); }
};
template <class M> BATCHLAS_DLA_INLINE RtOpView<M> op_view(const M& a, Transpose tr) { return {a, tr}; }

// ---- Hermitian / symmetric full view of one stored triangle -------------------------------
// herm_view<UL>(a)(i, j) = a(i, j) inside the stored UL triangle, conj(a(j, i)) outside (one
// load per element; the other triangle is never read). DiagReal = true returns (re(a(i, i)), 0)
// on the diagonal (LAPACK's assumption for a Hermitian matrix; it masks a poisoned Im(diag)).
// sym_view is the same without the conj (complex symmetric, and every real matrix).
template <Uplo UL, bool Conj, bool DiagReal, class M> struct HermView {
    M a;
    BATCHLAS_DLA_INLINE auto operator()(int i, int j) const {
        using T = detail::view_elem2_t<M>;
        using A = Arith<T>;
        const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
        const T v = stored ? static_cast<T>(a(i, j)) : static_cast<T>(a(j, i));
        if constexpr (A::is_complex && DiagReal) {
            if (i == j) return A::make(A::re(v), typename A::real_t(0));
        }
        if constexpr (A::is_complex && Conj) return stored ? v : A::conj(v);
        else return v;
    }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
};
template <Uplo UL, bool DiagReal = false, class M> BATCHLAS_DLA_INLINE HermView<UL, true, DiagReal, M> herm_view(const M& a) { return {a}; }
template <Uplo UL, class M> BATCHLAS_DLA_INLINE HermView<UL, false, false, M> sym_view(const M& a) { return {a}; }

// ---- packed triangles ----------------------------------------------------------------------
// The element map of an n x n triangle stored packed (no padding), LAPACK's AP for ByCols:
// Upper (i <= j): i + j (j + 1) / 2; Lower (i >= j): i + j (2n - j - 1) / 2. ByRows is the
// transposed traversal: Lower row i holds (i, 0..i): i (i + 1) / 2 + j; Upper row i holds
// (i, i..n-1). decode(k) inverts index exactly (an integer square root, then a +-1 fix).
namespace order {
struct ByCols {};
struct ByRows {};
}  // namespace order

template <Uplo UL, class Order = order::ByCols> struct PackedTri {
    static_assert(std::is_same_v<Order, order::ByCols> || std::is_same_v<Order, order::ByRows>, "PackedTri: order::ByCols or ByRows");
    // Number of stored elements.
    static constexpr int size(int n) { return n * (n + 1) / 2; }
    static constexpr int index(int i, int j, int n) {
        // ByRows of Lower == ByCols of Upper with (i, j) swapped, and the same for Upper.
        constexpr bool cols = std::is_same_v<Order, order::ByCols>;
        constexpr bool head = (UL == Uplo::Upper) == cols;  // a triangular-number prefix layout
        const int a = cols ? i : j, b = cols ? j : i;       // a: within the line, b: the line
        if constexpr (head) return a + b * (b + 1) / 2;
        else return a + b * (2 * n - b - 1) / 2;
    }
    struct RowCol {
        int i, j;
    };
    static RowCol decode(int k, int n) {
        constexpr bool cols = std::is_same_v<Order, order::ByCols>;
        constexpr bool head = (UL == Uplo::Upper) == cols;
        int b, a;
        if constexpr (head) {
            // largest b with b (b + 1) / 2 <= k
            b = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(k) + 1.0f) - 1.0f) * 0.5f);
            while (b * (b + 1) / 2 > k) --b;
            while ((b + 1) * (b + 2) / 2 <= k) ++b;
            a = k - b * (b + 1) / 2;
        } else {
            // line b starts at s(b) = b (2n - b + 1) / 2; count from the end: r = size - 1 - k
            const int r = n * (n + 1) / 2 - 1 - k;
            int c = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(r) + 1.0f) - 1.0f) * 0.5f);
            while (c * (c + 1) / 2 > r) --c;
            while ((c + 1) * (c + 2) / 2 <= r) ++c;
            b = n - 1 - c;
            a = b + (k - b * (2 * n - b - 1) / 2 - b);
        }
        return cols ? RowCol{a, b} : RowCol{b, a};
    }
};

// A packed view: (i, j) inside the UL triangle of a packed array; outside it is not addressable.
template <Uplo UL, class Order, class P> struct PackedMat {
    P p;  // T* or any base with p[k]
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return p[PackedTri<UL, Order>::index(i, j, n)]; }
    BATCHLAS_DLA_INLINE int rows() const { return n; }
    BATCHLAS_DLA_INLINE int cols() const { return n; }
};
template <Uplo UL, class Order = order::ByCols, class P> BATCHLAS_DLA_INLINE PackedMat<UL, Order, P> packed(P p, int n) { return {p, n}; }

// ---- Packet: one vector-width unit as a plain struct ---------------------------------------
// Bytes / sizeof(T) elements loaded or stored through a pointer the caller guarantees is
// Bytes-aligned (the compiler is told so; one 8/16-byte access where the target has it). A
// struct, not sycl::vec: Metal has no vector types in this position (design C7).
template <class T, int Bytes> struct Packet {
    static_assert(Bytes % sizeof(T) == 0 && (Bytes & (Bytes - 1)) == 0, "Packet: Bytes is a power of two multiple of sizeof(T)");
    static constexpr int n = Bytes / static_cast<int>(sizeof(T));
    T v[n];
    BATCHLAS_DLA_INLINE T& operator[](int k) { return v[k]; }
    BATCHLAS_DLA_INLINE const T& operator[](int k) const { return v[k]; }
    BATCHLAS_DLA_INLINE static Packet load(const T* p) {
        const T* a = static_cast<const T*>(__builtin_assume_aligned(p, Bytes));
        Packet r;
#pragma unroll
        for (int k = 0; k < n; ++k) r.v[k] = a[k];
        return r;
    }
    BATCHLAS_DLA_INLINE void store(T* p) const {
        T* a = static_cast<T*>(__builtin_assume_aligned(p, Bytes));
#pragma unroll
        for (int k = 0; k < n; ++k) a[k] = v[k];
    }
};

// ---- generated and remapped vectors ---------------------------------------------------------
// fn_vec(n, f): element i is f(i) (by value). mapped(x, f): f(x(i)). slice(x, s, n, step):
// x(s + i * step). gathered(x, p): x(p[i]).
template <class F> struct FnVec {
    int n;
    F f;
    BATCHLAS_DLA_INLINE auto operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DLA_INLINE FnVec<F> fn_vec(int n, F f) { return {n, f}; }
template <class X, class F> BATCHLAS_DLA_INLINE auto mapped(const X& x, F f) {
    return fn_vec(x.size(), [x, f](int i) { return f(x(i)); });
}
template <class X> struct Sliced {
    X x;
    int start, n, step;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return x(start + i * step); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class X> BATCHLAS_DLA_INLINE Sliced<X> slice(const X& x, int start, int n, int step = 1) { return {x, start, n, step}; }
template <class X, class P> struct Gathered {
    X x;
    P p;
    int n;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return x(static_cast<int>(p[i])); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class X, class P> BATCHLAS_DLA_INLINE Gathered<X, P> gathered(const X& x, P p, int n) { return {x, p, n}; }

}  // namespace batchlas::device
