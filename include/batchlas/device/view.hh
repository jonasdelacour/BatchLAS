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

namespace batchlas::device {

namespace detail {
template <class V> inline constexpr bool is_vec_like_v = requires(const V& v) { v.size(); v(0); };
template <class M> inline constexpr bool is_mat_like_v = requires(const M& a) { a.rows(); a.cols(); a(0, 0); };
template <class V> using view_elem_t = std::remove_cvref_t<decltype(std::declval<const V&>()(0))>;
template <class M> using view_elem2_t = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;
}  // namespace detail

// ---- sub-operands as free functions (the members of Mat / Vec, for any operand) ----------
template <class M> BATCHLAS_DEVICE_INLINE auto row(const M& a, int i) { return a.row(i); }
template <class M> BATCHLAS_DEVICE_INLINE auto col(const M& a, int j) { return a.col(j); }
template <class M> BATCHLAS_DEVICE_INLINE auto window(const M& a, int i0, int j0, int m, int n) { return a.sub(i0, j0, m, n); }

// ---- flip: reversed index order ---------------------------------------------------------
enum class Axis { Rows, Cols, Both };

// Vector: x(n - 1 - i) when on (a run-time switch, uniform), else x(i).
template <class V> struct Flipped {
    V v;
    bool on;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return v(on ? v.size() - 1 - i : i); }
    BATCHLAS_DEVICE_INLINE int size() const { return v.size(); }
};
// Matrix: rows (m - 1 - i), columns (n - 1 - j) or both reversed when on.
template <Axis AX, class M> struct FlippedMat {
    M a;
    bool on;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (AX != Axis::Cols) i = on ? a.rows() - 1 - i : i;
        if constexpr (AX != Axis::Rows) j = on ? a.cols() - 1 - j : j;
        return a(i, j);
    }
    BATCHLAS_DEVICE_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DEVICE_INLINE int cols() const { return a.cols(); }
};
template <Axis AX = Axis::Rows, class X> BATCHLAS_DEVICE_INLINE auto flip(const X& x, bool on = true) {
    static_assert(AX == Axis::Rows || AX == Axis::Cols || AX == Axis::Both,
                  "flip: Axis::Rows, Axis::Cols or Axis::Both");
    if constexpr (detail::is_mat_like_v<X>) return FlippedMat<AX, X>{x, on};
    else return Flipped<X>{x, on};
}

// ---- conj views ---------------------------------------------------------------------------
// conj(x(i)) (or conj(a(i, j))) when on, by value; the identity on real T. Always (conj_view,
// conj_if<true>): no select, the form of the legacy views (the select costs the sygst diag kernel).
template <class X, bool Always = false> struct ConjView {
    X x;
    bool on;
    template <class... I> BATCHLAS_DEVICE_INLINE auto operator()(I... i) const {
        using T = std::remove_cvref_t<decltype(x(i...))>;
        if constexpr (Always && Arith<T>::is_complex) {
            return Arith<T>::conj(x(i...));
        } else {
            const T v = x(i...);
            if constexpr (Arith<T>::is_complex) return on ? Arith<T>::conj(v) : v;
            else return v;
        }
    }
    BATCHLAS_DEVICE_INLINE int size() const requires detail::is_vec_like_v<X> { return x.size(); }
    BATCHLAS_DEVICE_INLINE int rows() const requires detail::is_mat_like_v<X> { return x.rows(); }
    BATCHLAS_DEVICE_INLINE int cols() const requires detail::is_mat_like_v<X> { return x.cols(); }
};
template <class X> BATCHLAS_DEVICE_INLINE ConjView<X, true> conj_view(const X& x) { return {x, true}; }
// Compile-time switch: the operand itself when !C (no view, no branch).
template <class X> inline constexpr bool is_operand_v = detail::is_vec_like_v<X> || detail::is_mat_like_v<X>;
template <bool C, class X>
    requires is_operand_v<X>
BATCHLAS_DEVICE_INLINE auto conj_if(const X& x) {
    if constexpr (C) return ConjView<X, true>{x, true};
    else return x;
}
// Run-time switch (uniform): one select per element.
template <class X>
    requires is_operand_v<X>
BATCHLAS_DEVICE_INLINE ConjView<X> conj_if(const X& x, bool on) { return {x, on}; }

// ---- op(A) --------------------------------------------------------------------------------
// op(A)(i, j) for a compile-time Transpose: NoTrans is A itself; Trans swaps indices (writable);
// ConjTrans is conj(A(j, i)) by value.
template <Transpose TR, class M> struct OpView {
    M a;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (TR == Transpose::Trans) {
            return a(j, i);
        } else {
            using T = detail::view_elem2_t<M>;
            return Arith<T>::conj(static_cast<T>(a(j, i)));
        }
    }
    BATCHLAS_DEVICE_INLINE int rows() const { return a.cols(); }
    BATCHLAS_DEVICE_INLINE int cols() const { return a.rows(); }
};
template <Transpose TR, class M> BATCHLAS_DEVICE_INLINE auto op_view(const M& a) {
    if constexpr (TR == Transpose::NoTrans) return a;
    else return OpView<TR, M>{a};
}
// The run-time form (uniform tr): by value, one branch per element.
template <class M> struct RtOpView {
    M a;
    Transpose tr;
    BATCHLAS_DEVICE_INLINE auto operator()(int i, int j) const {
        using T = detail::view_elem2_t<M>;
        if (tr == Transpose::NoTrans) return static_cast<T>(a(i, j));
        const T v = a(j, i);
        return tr == Transpose::ConjTrans ? Arith<T>::conj(v) : v;
    }
    BATCHLAS_DEVICE_INLINE int rows() const { return tr == Transpose::NoTrans ? a.rows() : a.cols(); }
    BATCHLAS_DEVICE_INLINE int cols() const { return tr == Transpose::NoTrans ? a.cols() : a.rows(); }
};
template <class M> BATCHLAS_DEVICE_INLINE RtOpView<M> op_view(const M& a, Transpose tr) { return {a, tr}; }

// ---- Hermitian / symmetric full view of one stored triangle -------------------------------
// herm_view<UL>(a)(i, j) = a(i, j) inside the stored UL triangle, conj(a(j, i)) outside (one
// load per element; the other triangle is never read). DiagReal = true returns (re(a(i, i)), 0)
// on the diagonal (LAPACK's assumption for a Hermitian matrix; it masks a poisoned Im(diag)).
// sym_view is the same without the conj (complex symmetric, and every real matrix).
template <Uplo UL, bool Conj, bool DiagReal, class M> struct HermView {
    M a;
    BATCHLAS_DEVICE_INLINE auto operator()(int i, int j) const {
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
    BATCHLAS_DEVICE_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DEVICE_INLINE int cols() const { return a.cols(); }
};
template <Uplo UL, bool DiagReal = false, class M> BATCHLAS_DEVICE_INLINE HermView<UL, true, DiagReal, M> herm_view(const M& a) { return {a}; }
template <Uplo UL, class M> BATCHLAS_DEVICE_INLINE HermView<UL, false, false, M> sym_view(const M& a) { return {a}; }

// The general Hermitian view (Im(diag) always dropped): herm_view<UL, Storage, Load>.
//   Storage::Full        a is the n x n matrix, its UL triangle stored
//   Storage::Band        a is LAPACK band storage of the UL band (kd + 1 rows, n columns):
//                        Lower AB(i - j, j) = A(i, j) for 0 <= i - j <= kd, Upper AB(kd + i - j, j);
//                        outside the band the view reads 0
//   alg::OneLoad           one load at the selected index; alg::TwoLoads: one load per branch
//                        (the stored and the mirrored element are separate loads, so a caller's
//                        guard folds into one CFG: syev's herm_load)
enum class Storage { Full, Band };
namespace alg {
struct OneLoad {};
struct TwoLoads {};
}  // namespace alg
template <Uplo UL, Storage ST, class Load, class M> struct HermViewOf {
    static_assert(std::is_same_v<Load, alg::OneLoad> || std::is_same_v<Load, alg::TwoLoads>, "herm_view: alg::OneLoad or alg::TwoLoads");
    M a;
    int n;
    int kd;
    BATCHLAS_DEVICE_INLINE auto operator()(int i, int j) const {
        using T = detail::view_elem2_t<M>;
        using A = Arith<T>;
        const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
        const int si = stored ? i : j, sj = stored ? j : i;  // the stored element's indices
        T v;
        if constexpr (ST == Storage::Band) {
            const int d = UL == Uplo::Lower ? si - sj : sj - si;
            if (d > kd) return A::zero();
            const int br = UL == Uplo::Lower ? d : kd - d;
            v = static_cast<T>(a(br, sj));
        } else {
            static_assert(ST == Storage::Full, "herm_view: Storage::Full or Storage::Band");
            if constexpr (std::is_same_v<Load, alg::TwoLoads>) {
                if (stored) v = static_cast<T>(a(i, j));
                else v = static_cast<T>(a(j, i));
            } else {
                v = static_cast<T>(a(si, sj));
            }
        }
        if constexpr (A::is_complex) {
            if (i == j) return A::make(A::re(v), typename A::real_t(0));
            return stored ? v : A::conj(v);
        } else {
            return v;
        }
    }
    BATCHLAS_DEVICE_INLINE int rows() const { return n; }
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
template <Uplo UL, Storage ST, class Load = alg::OneLoad, class M> BATCHLAS_DEVICE_INLINE HermViewOf<UL, ST, Load, M> herm_view(const M& a) {
    static_assert(ST == Storage::Full, "herm_view<UL, Storage::Band>: pass kd and n, herm_view<UL, Storage::Band>(ab, n, kd)");
    return {a, a.rows(), 0};
}
template <Uplo UL, Storage ST, class Load = alg::OneLoad, class M>
BATCHLAS_DEVICE_INLINE HermViewOf<UL, ST, Load, M> herm_view(const M& ab, int n, int kd) {
    return {ab, n, kd};
}

// x(i - k) for 0 <= i - k < x.size(), else 0, over n elements (k = 1: the down-shift that puts
// a zero first).
template <class X> struct Shifted {
    X x;
    int k, n;
    BATCHLAS_DEVICE_INLINE auto operator()(int i) const {
        using T = detail::view_elem_t<X>;
        const int s = i - k;
        return (s >= 0 && s < x.size()) ? static_cast<T>(x(s)) : Arith<T>::zero();
    }
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
template <class X> BATCHLAS_DEVICE_INLINE Shifted<X> shifted(const X& x, int k, int n) { return {x, k, n}; }

// N scalar references as a vector (element i is *p[i], writable): loose registers or fields as
// an operand of the Thread ops.
template <class T, int N> struct Refs {
    using value_type = std::remove_cv_t<T>;
    T* p[N];
    BATCHLAS_DEVICE_INLINE T& operator()(int i) const { return *p[i]; }
    BATCHLAS_DEVICE_INLINE constexpr int size() const { return N; }
};
template <class T, class... U> BATCHLAS_DEVICE_INLINE Refs<T, 1 + int(sizeof...(U))> refs(T& a, U&... b) {
    static_assert((std::is_same_v<T, U> && ...), "refs: every element has one type");
    return {{&a, &b...}};
}

// ---- packed triangles ----------------------------------------------------------------------
// ---- trsm's canonical-order views (one kernel serves all 24 trsm cases) --------------------
// Element (s, t) of the canonical lower factor Lc of a stored triangle: rho(s) = fwd ? s :
// n-1-s maps canonical to stored indices, Left keeps (row, col) and Right swaps them, trans
// reads A(c, r), conj conjugates (the diagonal too). Offsets are 64-bit; Ld is the ld's type.
template <class T, bool Left, class Ld = int> struct CanonTri {
    using value_type = std::remove_cv_t<T>;
    T* a;
    Ld ld;
    int n;
    bool fwd, trans, conj;
    BATCHLAS_DEVICE_INLINE value_type operator()(int s, int t) const {
        const int rs = fwd ? s : (n - 1 - s);
        const int rt = fwd ? t : (n - 1 - t);
        const int r = Left ? rs : rt;
        const int c = Left ? rt : rs;
        value_type v = trans ? a[c + static_cast<std::ptrdiff_t>(r) * ld] : a[r + static_cast<std::ptrdiff_t>(c) * ld];
        if (conj) v = Arith<value_type>::conj(v);
        return v;
    }
    // Lc(s, s), addressed as rho(s) * (ld + 1). A kernel body calls this, never operator():
    // see scripts/check_kernel_entry_calls.py (the SemaSYCL kernel-body quirk).
    BATCHLAS_DEVICE_INLINE value_type diag(int s) const {
        const std::ptrdiff_t rs = fwd ? s : (n - 1 - s);
        value_type v = a[rs + rs * static_cast<std::ptrdiff_t>(ld)];
        if (conj) v = Arith<value_type>::conj(v);
        return v;
    }
};

// Right-hand side u in canonical order: element s is p[b0 + s*ds + u*du]. Left: column u of
// B, rows flipped when !fwd; Right: row u of B, columns flipped.
template <class T, bool Left> struct CanonRhs {
    T* p;
    std::ptrdiff_t b0, ds, du;
    struct Col {
        T* p;
        std::ptrdiff_t b0, ds, udu;
        BATCHLAS_DEVICE_INLINE T& operator()(int s) const { return p[b0 + static_cast<std::ptrdiff_t>(s) * ds + udu]; }
    };
    BATCHLAS_DEVICE_INLINE CanonRhs(T* p_, int n, int ld, bool fwd) : p(p_) {
        const std::ptrdiff_t unit_s = Left ? 1 : static_cast<std::ptrdiff_t>(ld);
        du = Left ? static_cast<std::ptrdiff_t>(ld) : 1;
        b0 = fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * unit_s;
        ds = fwd ? unit_s : -unit_s;
    }
    BATCHLAS_DEVICE_INLINE Col rhs(int u) const { return {p, b0, ds, u * du}; }
};

// An m x q column-major matrix with its rows read in canonical order: (s, c) is
// p[rho(s) + c*ld], rho(s) = fwd ? s : m-1-s.
template <class T, class Ld = int> struct FlipRows {
    T* p;
    Ld ld;
    int m, q;
    bool fwd;
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    BATCHLAS_DEVICE_INLINE int cols() const { return q; }
    BATCHLAS_DEVICE_INLINE T& operator()(int s, int c) const {
        const int row = fwd ? s : (m - 1 - s);
        return p[static_cast<std::ptrdiff_t>(row) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

// Columns c0.. of v: (r, j) is v(r, c0 + j), a reference (the core store_row target).
template <class V> struct ColsFrom {
    V v;
    int c0;
    BATCHLAS_DEVICE_INLINE auto& operator()(int r, int j) const { return v(r, c0 + j); }
};
template <class V> ColsFrom(V, int) -> ColsFrom<V>;

// Columns c0.. of v read through f: (r, j) is f(v(r, c0 + j)), a value (the core load_row
// source; f scales by alpha or negates, so the product is formed only for a live element).
template <class V, class F> struct MapCols {
    V v;
    int c0;
    F f;
    BATCHLAS_DEVICE_INLINE auto operator()(int r, int j) const { return f(v(r, c0 + j)); }
};
template <class V, class F> MapCols(V, int, F) -> MapCols<V, F>;

// The element map of an n x n triangle stored packed (no padding), LAPACK's AP for ByCols:
// Upper (i <= j): i + j (j + 1) / 2; Lower (i >= j): i + j (2n - j - 1) / 2. ByRows is the
// transposed traversal: Lower row i holds (i, 0..i): i (i + 1) / 2 + j; Upper row i holds
// (i, i..n-1). decode(k) inverts index exactly (an integer square root, then a +-1 fix).
enum class PackOrder { ByCols, ByRows };

template <Uplo UL, PackOrder OR = PackOrder::ByCols> struct PackedTri {
    static_assert(OR == PackOrder::ByCols || OR == PackOrder::ByRows, "PackedTri: PackOrder::ByCols or ByRows");
    // Number of stored elements.
    static constexpr int size(int n) { return n * (n + 1) / 2; }
    static constexpr int index(int i, int j, int n) {
        // ByRows of Lower == ByCols of Upper with (i, j) swapped, and the same for Upper.
        constexpr bool cols = OR == PackOrder::ByCols;
        constexpr bool head = (UL == Uplo::Upper) == cols;  // a triangular-number prefix layout
        const int a = cols ? i : j, b = cols ? j : i;       // a: within the line, b: the line
        if constexpr (head) return a + b * (b + 1) / 2;
        else return a + b * (2 * n - b - 1) / 2;
    }
    struct RowCol {
        int i, j;
    };
    // Exact for k < 2^30: a float sqrt estimate, then at most a step each way.
    BATCHLAS_DEVICE_INLINE static RowCol decode(int k, [[maybe_unused]] int n) {
        constexpr bool cols = OR == PackOrder::ByCols;
        constexpr bool head = (UL == Uplo::Upper) == cols;
        int b, a;
        if constexpr (head) {
            // largest b with b (b + 1) / 2 <= k
            b = static_cast<int>((sycl::sqrt(8.0f * static_cast<float>(k) + 1.0f) - 1.0f) * 0.5f);
            while ((b + 1) * (b + 2) / 2 <= k) ++b;
            while (b * (b + 1) / 2 > k) --b;
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
template <Uplo UL, PackOrder OR, class P> struct PackedMat {
    P p;  // T* or any base with p[k]
    int n;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const { return p[PackedTri<UL, OR>::index(i, j, n)]; }
    BATCHLAS_DEVICE_INLINE int rows() const { return n; }
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
template <Uplo UL, PackOrder OR = PackOrder::ByCols, class P> BATCHLAS_DEVICE_INLINE PackedMat<UL, OR, P> packed(P p, int n) { return {p, n}; }

// v / L(s, s) for a packed L: v * rd(s), or a true division once any reciprocal was
// non-finite (divide is group-uniform); unit leaves v.
template <class T> struct TpDiag {
    const T* ap;
    const T* rd;
    bool unit, divide;
    template <class Ar> BATCHLAS_DEVICE_INLINE T apply(int s, T v) const {
        if (!unit) v = divide ? Ar::div(v, ap[PackedTri<Uplo::Lower, PackOrder::ByRows>::index(s, s, 0)]) : Ar::mul(v, rd[s]);
        return v;
    }
};

// ---- Packet: one vector-width unit as a plain struct ---------------------------------------
// Bytes / sizeof(T) elements loaded or stored through a pointer the caller guarantees is
// Bytes-aligned (the compiler is told so; one 8/16-byte access where the target has it). A
// struct, not sycl::vec: Metal has no vector types in this position (design C7).
template <class T, int Bytes> struct Packet {
    static_assert(Bytes % sizeof(T) == 0 && (Bytes & (Bytes - 1)) == 0, "Packet: Bytes is a power of two multiple of sizeof(T)");
    static constexpr int n = Bytes / static_cast<int>(sizeof(T));
    T v[n];
    BATCHLAS_DEVICE_INLINE T& operator[](int k) { return v[k]; }
    BATCHLAS_DEVICE_INLINE const T& operator[](int k) const { return v[k]; }
    BATCHLAS_DEVICE_INLINE static Packet load(const T* p) {
        const T* a = static_cast<const T*>(__builtin_assume_aligned(p, Bytes));
        Packet r;
#pragma unroll
        for (int k = 0; k < n; ++k) r.v[k] = a[k];
        return r;
    }
    BATCHLAS_DEVICE_INLINE void store(T* p) const {
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
    BATCHLAS_DEVICE_INLINE auto operator()(int i) const { return f(i); }
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DEVICE_INLINE FnVec<F> fn_vec(int n, F f) { return {n, f}; }
template <class X, class F> BATCHLAS_DEVICE_INLINE auto mapped(const X& x, F f) {
    return fn_vec(x.size(), [x, f](int i) { return f(x(i)); });
}
template <class X> struct Sliced {
    X x;
    int start, n, step;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return x(start + i * step); }
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
template <class X> BATCHLAS_DEVICE_INLINE Sliced<X> slice(const X& x, int start, int n, int step = 1) { return {x, start, n, step}; }
template <class X, class P> struct Gathered {
    X x;
    P p;
    int n;
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return x(static_cast<int>(p[i])); }
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
template <class X, class P> BATCHLAS_DEVICE_INLINE Gathered<X, P> gathered(const X& x, P p, int n) { return {x, p, n}; }

}  // namespace batchlas::device
