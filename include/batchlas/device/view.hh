// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Lazy views: index maps over any operand (flip, conj, op, Hermitian and symmetric,
/// shifted, sliced, gathered), packed triangles and the Packet load/store unit.
/// @ingroup api_device_operands
///
/// A view works over any operand: a Vec-like x(i) with size(), or a Mat-like a(i, j) with
/// rows() and cols(). It holds its base by value and computes the index; nothing is copied. A
/// view whose element is a function of the stored one (conj, op with ConjTrans, herm off its
/// stored triangle) is read-only and returns values.

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

/// @brief Row i of any matrix operand, `a.row(i)`.
/// @ingroup api_device_operands
/// @param a the matrix operand
/// @param i the row
/// @return the row view
template <class M> BATCHLAS_DEVICE_INLINE auto row(const M& a, int i) { return a.row(i); }
/// @brief Column j of any matrix operand, `a.col(j)`.
/// @ingroup api_device_operands
/// @param a the matrix operand
/// @param j the column
/// @return the column view
template <class M> BATCHLAS_DEVICE_INLINE auto col(const M& a, int j) { return a.col(j); }
/// @brief The m x n block at (i0, j0) of any matrix operand, `a.sub(i0, j0, m, n)`.
/// @ingroup api_device_operands
/// @param a the matrix operand
/// @param i0 the block's first row
/// @param j0 the block's first column
/// @param m the block's rows
/// @param n the block's columns
/// @return the block view
template <class M> BATCHLAS_DEVICE_INLINE auto window(const M& a, int i0, int j0, int m, int n) { return a.sub(i0, j0, m, n); }

/// @brief The index order flip reverses: Rows, Cols or Both.
/// @ingroup api_device_operands
enum class Axis { Rows, Cols, Both };

/// @brief A vector with its order reversed when on: x(n - 1 - i), else x(i).
/// @ingroup api_device_operands
/// @tparam V the vector operand type
template <class V> struct Flipped {
    V v;  ///< The vector.
    bool on;  ///< The run-time switch, uniform over the executor.
    /// @brief Element i of the (reversed) vector.
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return v(on ? v.size() - 1 - i : i); }
    /// @brief The vector's size().
    BATCHLAS_DEVICE_INLINE int size() const { return v.size(); }
};
/// @brief A matrix with its rows (m - 1 - i), columns (n - 1 - j) or both reversed when on.
/// @ingroup api_device_operands
/// @tparam AX the axes to reverse
/// @tparam M the matrix operand type
template <Axis AX, class M> struct FlippedMat {
    M a;  ///< The matrix.
    bool on;  ///< The run-time switch, uniform over the executor.
    /// @brief Element (i, j) of the (reversed) matrix.
    /// @param i the row
    /// @param j the column
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (AX != Axis::Cols) i = on ? a.rows() - 1 - i : i;
        if constexpr (AX != Axis::Rows) j = on ? a.cols() - 1 - j : j;
        return a(i, j);
    }
    /// @brief The matrix's rows().
    BATCHLAS_DEVICE_INLINE int rows() const { return a.rows(); }
    /// @brief The matrix's cols().
    BATCHLAS_DEVICE_INLINE int cols() const { return a.cols(); }
};
/// @brief The reversed view of a vector (Flipped) or a matrix (FlippedMat).
/// @ingroup api_device_operands
/// @tparam AX for a matrix, the axes to reverse (Rows by default)
/// @param x the operand
/// @param on the run-time switch, uniform over the executor
/// @return the view
template <Axis AX = Axis::Rows, class X> BATCHLAS_DEVICE_INLINE auto flip(const X& x, bool on = true) {
    static_assert(AX == Axis::Rows || AX == Axis::Cols || AX == Axis::Both,
                  "flip: Axis::Rows, Axis::Cols or Axis::Both");
    if constexpr (detail::is_mat_like_v<X>) return FlippedMat<AX, X>{x, on};
    else return Flipped<X>{x, on};
}

/// @brief conj(x(i)) or conj(a(i, j)) when on, by value; the identity on a real element type.
/// @ingroup api_device_operands
///
/// With Always (conj_view, `conj_if<true>`) the view conjugates with no select.
/// @tparam X the operand type
/// @tparam Always true to conjugate unconditionally
// Always is the form of the legacy views: the select costs the sygst diag kernel.
template <class X, bool Always = false> struct ConjView {
    X x;  ///< The operand.
    bool on;  ///< The run-time switch, uniform over the executor (ignored when Always).
    /// @brief The element at the indices, conjugated when on.
    /// @param i the index or indices
    /// @return the element, by value
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
    /// @brief The operand's size() (a vector operand).
    BATCHLAS_DEVICE_INLINE int size() const requires detail::is_vec_like_v<X> { return x.size(); }
    /// @brief The operand's rows() (a matrix operand).
    BATCHLAS_DEVICE_INLINE int rows() const requires detail::is_mat_like_v<X> { return x.rows(); }
    /// @brief The operand's cols() (a matrix operand).
    BATCHLAS_DEVICE_INLINE int cols() const requires detail::is_mat_like_v<X> { return x.cols(); }
};
/// @brief The conjugate view of x, unconditional.
/// @ingroup api_device_operands
/// @param x the operand
/// @return the view
template <class X> BATCHLAS_DEVICE_INLINE ConjView<X, true> conj_view(const X& x) { return {x, true}; }
/// @brief True when X is a vector-like or matrix-like operand.
/// @ingroup api_device_operands
/// @tparam X the type to test
template <class X> inline constexpr bool is_operand_v = detail::is_vec_like_v<X> || detail::is_mat_like_v<X>;
/// @brief Compile-time switch: the conjugate view when C, else x itself (no view, no branch).
/// @ingroup api_device_operands
/// @tparam C true to conjugate
/// @param x the operand
/// @return the view, or x
template <bool C, class X>
    requires is_operand_v<X>
BATCHLAS_DEVICE_INLINE auto conj_if(const X& x) {
    if constexpr (C) return ConjView<X, true>{x, true};
    else return x;
}
/// @brief Run-time switch (on uniform): the conjugate view with one select per element.
/// @ingroup api_device_operands
/// @param x the operand
/// @param on the run-time switch
/// @return the view
template <class X>
    requires is_operand_v<X>
BATCHLAS_DEVICE_INLINE ConjView<X> conj_if(const X& x, bool on) { return {x, on}; }

/// @brief op(A)(i, j) for a compile-time Transpose: Trans swaps the indices (writable), ConjTrans
/// is conj(A(j, i)) by value.
/// @ingroup api_device_operands
///
/// op_view returns A itself for NoTrans.
/// @tparam TR Transpose::Trans or Transpose::ConjTrans
/// @tparam M the matrix operand type
template <Transpose TR, class M> struct OpView {
    M a;  ///< The matrix.
    /// @brief Element (i, j) of op(A).
    /// @param i the row
    /// @param j the column
    /// @return the element (a reference for Trans, a value for ConjTrans)
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const {
        if constexpr (TR == Transpose::Trans) {
            return a(j, i);
        } else {
            using T = detail::view_elem2_t<M>;
            return Arith<T>::conj(static_cast<T>(a(j, i)));
        }
    }
    /// @brief The rows of op(A), A's cols().
    BATCHLAS_DEVICE_INLINE int rows() const { return a.cols(); }
    /// @brief The columns of op(A), A's rows().
    BATCHLAS_DEVICE_INLINE int cols() const { return a.rows(); }
};
/// @brief op(A) for a compile-time Transpose: A itself for NoTrans, else an OpView.
/// @ingroup api_device_operands
/// @tparam TR the transpose
/// @param a the matrix operand
/// @return the view, or a
template <Transpose TR, class M> BATCHLAS_DEVICE_INLINE auto op_view(const M& a) {
    if constexpr (TR == Transpose::NoTrans) return a;
    else return OpView<TR, M>{a};
}
/// @brief op(A) for a run-time Transpose (uniform): by value, one branch per element.
/// @ingroup api_device_operands
/// @tparam M the matrix operand type
template <class M> struct RtOpView {
    M a;  ///< The matrix.
    Transpose tr;  ///< The transpose, uniform over the executor.
    /// @brief Element (i, j) of op(A), by value.
    /// @param i the row
    /// @param j the column
    /// @return the element
    BATCHLAS_DEVICE_INLINE auto operator()(int i, int j) const {
        using T = detail::view_elem2_t<M>;
        if (tr == Transpose::NoTrans) return static_cast<T>(a(i, j));
        const T v = a(j, i);
        return tr == Transpose::ConjTrans ? Arith<T>::conj(v) : v;
    }
    /// @brief The rows of op(A).
    BATCHLAS_DEVICE_INLINE int rows() const { return tr == Transpose::NoTrans ? a.rows() : a.cols(); }
    /// @brief The columns of op(A).
    BATCHLAS_DEVICE_INLINE int cols() const { return tr == Transpose::NoTrans ? a.cols() : a.rows(); }
};
/// @brief op(A) for a run-time Transpose.
/// @ingroup api_device_operands
/// @param a the matrix operand
/// @param tr the transpose, uniform over the executor
/// @return the RtOpView
template <class M> BATCHLAS_DEVICE_INLINE RtOpView<M> op_view(const M& a, Transpose tr) { return {a, tr}; }

/// @brief Where a Hermitian or symmetric view's stored triangle lives: Full (the n x n matrix) or
/// Band (LAPACK band storage).
/// @ingroup api_device_operands
///
/// Storage::Band is the UL band in kd + 1 rows and n columns: Lower `AB(i - j, j) = A(i, j)` for
/// 0 <= i - j <= kd, Upper `AB(kd + i - j, j)`. Outside the band the view reads 0.
enum class Storage { Full, Band };
namespace alg {
/// @brief One load at the selected index.
/// @ingroup api_device_operands
struct OneLoad {};
/// @brief One load per branch: the stored and the mirrored element are separate loads, so a
/// caller's guard folds into one CFG.
/// @ingroup api_device_operands
struct TwoLoads {};
/// @brief The diagonal tested first, then the triangle, each branch its own load (Full, Conj and
/// DiagReal only).
/// @ingroup api_device_operands
// syev's CTA stage 0.
struct ThreeLoads {};
}  // namespace alg
/// @brief A stored triangle known at compile time, as the U parameter of HermView.
/// @ingroup api_device_operands
/// @tparam UL the triangle
template <Uplo UL> using UploC = std::integral_constant<Uplo, UL>;
/// @brief Column j of a HermView as a vector, HermView::col returns it.
/// @ingroup api_device_operands
/// @tparam H the HermView type
template <class H> struct HermCol {
    H h;  ///< The Hermitian view.
    int j;  ///< The column.
    using value_type = typename H::value_type;  ///< The view's element type.
    /// @brief Element (i, j) of the full matrix.
    /// @param i the row
    /// @return the element
    BATCHLAS_DEVICE_INLINE value_type operator()(int i) const { return h(i, j); }
    /// @brief The matrix order.
    BATCHLAS_DEVICE_INLINE int size() const { return h.rows(); }
};
/// @brief The full Hermitian or symmetric matrix from one stored triangle.
/// @ingroup api_device_operands
///
/// (i, j) reads the stored triangle's element at (i, j) or its mirror (j, i), conjugated outside
/// the triangle when Conj (complex T), and (re, 0) on the diagonal when DiagReal (LAPACK's
/// assumption for a Hermitian matrix; it masks a poisoned imaginary part on the diagonal). The
/// spellings:
/// - `herm_view<UL, DiagReal = false>(a)`: Full, alg::TwoLoads, Conj.
/// - `sym_view<UL>(a)`: the same without the conj (complex symmetric).
/// - `herm_view<UL, Storage, Load>(a)`: DiagReal, Conj; Storage::Band takes `(ab, n, kd)`.
/// - `herm_view<Load = OneLoad, Ops>(a, uplo)`: uplo at run time; Full, DiagReal, Conj.
///
/// `col(j)` is column j as a vector.
/// @tparam U the stored triangle: `UploC<UL>` (compile time) or Uplo (run time)
/// @tparam ST Storage::Full or Storage::Band
/// @tparam Load alg::OneLoad, alg::TwoLoads or alg::ThreeLoads
/// @tparam Conj conjugate outside the stored triangle (Hermitian); false is symmetric
/// @tparam DiagReal read the diagonal as (re, 0)
/// @tparam Ops the arithmetic of the conj and the real diagonal (arith::Native: T's own)
/// @tparam M the stored operand type
template <class U, Storage ST, class Load, bool Conj, bool DiagReal, class Ops, class M> struct HermView {
    static_assert(std::is_same_v<Load, alg::OneLoad> || std::is_same_v<Load, alg::TwoLoads> || std::is_same_v<Load, alg::ThreeLoads>,
                  "herm_view: alg::OneLoad, TwoLoads or ThreeLoads");
    using value_type = detail::view_elem2_t<M>;  ///< The stored operand's element type.
    M a;  ///< The stored operand (the band array for Storage::Band).
    int n;  ///< The matrix order.
    int kd;  ///< The bandwidth (Storage::Band), else 0.
    U ul;  ///< The stored triangle.
    /// @brief Element (i, j) of the full matrix.
    /// @param i the row
    /// @param j the column
    /// @return the element, by value
    BATCHLAS_DEVICE_INLINE value_type operator()(int i, int j) const {
        using T = value_type;
        using A = arith_t<Ops, T>;
        const bool lower = static_cast<Uplo>(ul) == Uplo::Lower;
        if constexpr (std::is_same_v<Load, alg::ThreeLoads>) {
            static_assert(ST == Storage::Full && Conj && DiagReal, "herm_view: alg::ThreeLoads is the Full Hermitian form");
            T v;
            if (i == j) v = A::make(A::re(a(i, j)), typename A::real_t(0));
            else if (!lower) v = (i < j) ? a(i, j) : A::conj(a(j, i));
            else v = (i > j) ? a(i, j) : A::conj(a(j, i));
            return v;
        }
        const bool stored = lower ? i >= j : i <= j;
        const int si = stored ? i : j, sj = stored ? j : i;  // the stored element's indices
        T v;
        if constexpr (ST == Storage::Band) {
            const int d = lower ? si - sj : sj - si;
            if (d > kd) return A::zero();
            const int br = lower ? d : kd - d;
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
            if constexpr (DiagReal) {
                if (i == j) return A::make(A::re(v), typename A::real_t(0));
            }
            if constexpr (Conj) return stored ? v : A::conj(v);
        }
        return v;
    }
    /// @brief The matrix order, n.
    BATCHLAS_DEVICE_INLINE int rows() const { return n; }
    /// @brief The matrix order, n.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
    /// @brief Column j as a vector.
    /// @param j the column
    /// @return the HermCol view
    BATCHLAS_DEVICE_INLINE HermCol<HermView> col(int j) const { return {*this, j}; }
};
/// @brief The Hermitian view of the UL triangle of a: Full, alg::TwoLoads, conjugated mirror.
/// @ingroup api_device_operands
/// @tparam UL the stored triangle
/// @tparam DiagReal read the diagonal as (re, 0)
/// @param a the n x n matrix
/// @return the HermView
template <Uplo UL, bool DiagReal = false, class M>
BATCHLAS_DEVICE_INLINE HermView<UploC<UL>, Storage::Full, alg::TwoLoads, true, DiagReal, arith::Explicit, M> herm_view(const M& a) {
    return {a, a.rows(), 0, {}};
}
/// @brief The symmetric view of the UL triangle of a: Full, alg::TwoLoads, no conjugation.
/// @ingroup api_device_operands
/// @tparam UL the stored triangle
/// @param a the n x n matrix
/// @return the HermView
template <Uplo UL, class M>
BATCHLAS_DEVICE_INLINE HermView<UploC<UL>, Storage::Full, alg::TwoLoads, false, false, arith::Explicit, M> sym_view(const M& a) {
    return {a, a.rows(), 0, {}};
}
/// @brief The Hermitian view of the UL triangle of a, with a chosen load shape: DiagReal, Conj.
/// @ingroup api_device_operands
///
/// ST must be Storage::Full here; Storage::Band takes `(ab, n, kd)`.
/// @tparam UL the stored triangle
/// @tparam ST Storage::Full
/// @tparam Load the load shape
/// @param a the n x n matrix
/// @return the HermView
template <Uplo UL, Storage ST, class Load = alg::OneLoad, class M>
BATCHLAS_DEVICE_INLINE HermView<UploC<UL>, ST, Load, true, true, arith::Explicit, M> herm_view(const M& a) {
    static_assert(ST == Storage::Full, "herm_view<UL, Storage::Band>: pass kd and n, herm_view<UL, Storage::Band>(ab, n, kd)");
    return {a, a.rows(), 0, {}};
}
/// @brief The Hermitian view of a matrix in LAPACK band storage, or of a full matrix of order n:
/// DiagReal, Conj.
/// @ingroup api_device_operands
/// @tparam UL the stored triangle
/// @tparam ST Storage::Band (or Storage::Full)
/// @tparam Load the load shape
/// @param ab the band array, kd + 1 rows and n columns (or the matrix)
/// @param n the matrix order
/// @param kd the bandwidth
/// @return the HermView
template <Uplo UL, Storage ST, class Load = alg::OneLoad, class M>
BATCHLAS_DEVICE_INLINE HermView<UploC<UL>, ST, Load, true, true, arith::Explicit, M> herm_view(const M& ab, int n, int kd) {
    return {ab, n, kd, {}};
}
/// @brief The Hermitian view of a with the stored triangle chosen at run time: Full, DiagReal, Conj.
/// @ingroup api_device_operands
/// @tparam Load the load shape (alg::OneLoad by default)
/// @tparam Ops the arithmetic of the conj and the real diagonal
/// @param a the n x n matrix
/// @param uplo the stored triangle, uniform over the executor
/// @return the HermView
template <class Load = alg::OneLoad, class Ops = arith::Explicit, class M>
BATCHLAS_DEVICE_INLINE HermView<Uplo, Storage::Full, Load, true, true, Ops, M> herm_view(const M& a, Uplo uplo) {
    return {a, a.rows(), 0, uplo};
}

/// @brief x(i - k) for 0 <= i - k < x.size(), else 0, over n elements.
/// @ingroup api_device_operands
///
/// k = 1 is the down-shift that puts a zero first.
/// @tparam X the vector operand type
template <class X> struct Shifted {
    X x;  ///< The vector.
    ///@{
    /// @brief The shift k and the size n.
    int k, n;
    ///@}
    /// @brief Element i, by value.
    /// @param i the index
    /// @return x(i - k), or 0 outside x
    BATCHLAS_DEVICE_INLINE auto operator()(int i) const {
        using T = detail::view_elem_t<X>;
        const int s = i - k;
        return (s >= 0 && s < x.size()) ? static_cast<T>(x(s)) : Arith<T>::zero();
    }
    /// @brief The size, n.
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
/// @brief The view x(i - k) over n elements, zero outside x.
/// @ingroup api_device_operands
/// @param x the vector
/// @param k the shift
/// @param n the size of the view
/// @return the Shifted view
template <class X> BATCHLAS_DEVICE_INLINE Shifted<X> shifted(const X& x, int k, int n) { return {x, k, n}; }

/// @brief N scalar references as a writable vector, element i is `*p[i]`: loose registers or
/// fields as an operand of the Thread ops.
/// @ingroup api_device_operands
/// @tparam Scalar the element type
/// @tparam N the number of elements
template <class Scalar, int N> struct Refs {
    using value_type = std::remove_cv_t<Scalar>;  ///< The element type without cv.
    Scalar* p[N];  ///< The element addresses.
    /// @brief Element i.
    /// @param i the index
    /// @return a reference to `*p[i]`
    BATCHLAS_DEVICE_INLINE Scalar& operator()(int i) const { return *p[i]; }
    /// @brief The number of elements, N.
    BATCHLAS_DEVICE_INLINE constexpr int size() const { return N; }
};
/// @brief The Refs vector of a and b..., all of one type.
/// @ingroup api_device_operands
/// @param a the first element
/// @param b the other elements
/// @return the Refs vector
template <class Scalar, class... U> BATCHLAS_DEVICE_INLINE Refs<Scalar, 1 + int(sizeof...(U))> refs(Scalar& a,
                                                                                                    U&... b) {
    static_assert((std::is_same_v<Scalar, U> && ...), "refs: every element has one type");
    return {{&a, &b...}};
}

/// @brief trsm's canonical lower factor of a stored triangle, one kernel for all 24 trsm cases.
/// @ingroup api_device_lowlevel
///
/// Element (s, t) of Lc: rho(s) = fwd ? s : n - 1 - s maps canonical to stored indices, Left keeps
/// (row, col) and Right swaps them, trans reads A(c, r), conj conjugates (the diagonal too).
/// Offsets are 64-bit.
/// @tparam Scalar the element type
/// @tparam Left true for a left-side solve
/// @tparam Ld the leading dimension's type
template <class Scalar, bool Left, class Ld = int> struct CanonTri {
    using value_type = std::remove_cv_t<Scalar>;  ///< The element type without cv.
    Scalar* a;  ///< The stored triangle.
    Ld ld;  ///< The leading dimension.
    int n;  ///< The order.
    ///@{
    /// @brief Forward order, transpose, conjugate.
    bool fwd, trans, conj;
    ///@}
    /// @brief Element (s, t) of the canonical factor.
    /// @param s the canonical row
    /// @param t the canonical column
    /// @return the element, by value
    BATCHLAS_DEVICE_INLINE value_type operator()(int s, int t) const {
        const int rs = fwd ? s : (n - 1 - s);
        const int rt = fwd ? t : (n - 1 - t);
        const int r = Left ? rs : rt;
        const int c = Left ? rt : rs;
        value_type v = trans ? a[c + static_cast<std::ptrdiff_t>(r) * ld] : a[r + static_cast<std::ptrdiff_t>(c) * ld];
        if (conj) v = Arith<value_type>::conj(v);
        return v;
    }
    /// @brief The diagonal Lc(s, s), addressed as rho(s) * (ld + 1).
    ///
    /// A kernel body calls this, never operator(): see scripts/check_kernel_entry_calls.py.
    /// @param s the canonical index
    /// @return the element, by value
// The SemaSYCL kernel-body quirk.
    BATCHLAS_DEVICE_INLINE value_type diag(int s) const {
        const std::ptrdiff_t rs = fwd ? s : (n - 1 - s);
        value_type v = a[rs + rs * static_cast<std::ptrdiff_t>(ld)];
        if (conj) v = Arith<value_type>::conj(v);
        return v;
    }
};

/// @brief A right-hand side in trsm's canonical order: element s of rhs(u) is `p[b0 + s * ds + u * du]`.
/// @ingroup api_device_lowlevel
///
/// Left: column u of B, its rows flipped when !fwd. Right: row u of B, its columns flipped.
/// @tparam Scalar the element type
/// @tparam Left true for a left-side solve
template <class Scalar, bool Left> struct CanonRhs {
    Scalar* p;  ///< The matrix B.
    ///@{
    /// @brief The canonical base offset, element stride and right-hand-side stride.
    std::ptrdiff_t b0, ds, du;
    ///@}
    /// @brief One right-hand side as a vector.
    struct Col {
        Scalar* p;  ///< The matrix B.
        ///@{
        /// @brief The base offset, element stride and this right-hand side's offset.
        std::ptrdiff_t b0, ds, udu;
        ///@}
        /// @brief Element s.
        /// @param s the canonical index
        /// @return a reference to the element
        BATCHLAS_DEVICE_INLINE Scalar& operator()(int s) const { return p[b0 + static_cast<std::ptrdiff_t>(s) * ds + udu]; }
    };
    /// @brief The canonical view of the n x nrhs (Left) or nrhs x n (Right) matrix B at p_.
    /// @param p_ the matrix B
    /// @param n the order of the triangle
    /// @param ld B's leading dimension
    /// @param fwd forward order
    BATCHLAS_DEVICE_INLINE CanonRhs(Scalar* p_, int n, int ld, bool fwd) : p(p_) {
        const std::ptrdiff_t unit_s = Left ? 1 : static_cast<std::ptrdiff_t>(ld);
        du = Left ? static_cast<std::ptrdiff_t>(ld) : 1;
        b0 = fwd ? 0 : static_cast<std::ptrdiff_t>(n - 1) * unit_s;
        ds = fwd ? unit_s : -unit_s;
    }
    /// @brief Right-hand side u.
    /// @param u the right-hand side index
    /// @return the vector view
    BATCHLAS_DEVICE_INLINE Col rhs(int u) const { return {p, b0, ds, u * du}; }
};

/// @brief An m x q column-major matrix with its rows read in canonical order: (s, c) is
/// `p[rho(s) + c * ld]`, rho(s) = fwd ? s : m - 1 - s.
/// @ingroup api_device_lowlevel
/// @tparam Scalar the element type
/// @tparam Ld the leading dimension's type
template <class Scalar, class Ld = int> struct FlipRows {
    Scalar* p;  ///< The matrix.
    Ld ld;  ///< The leading dimension.
    ///@{
    /// @brief The rows m and the columns q.
    int m, q;
    ///@}
    bool fwd;  ///< Forward order.
    /// @brief The rows, m.
    BATCHLAS_DEVICE_INLINE int rows() const { return m; }
    /// @brief The columns, q.
    BATCHLAS_DEVICE_INLINE int cols() const { return q; }
    /// @brief Element (s, c), by canonical row.
    /// @param s the canonical row
    /// @param c the column
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE Scalar& operator()(int s, int c) const {
        const int row = fwd ? s : (m - 1 - s);
        return p[static_cast<std::ptrdiff_t>(row) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

/// @brief Columns c0 onwards of v: (r, j) is v(r, c0 + j), a reference (the store_row target).
/// @ingroup api_device_lowlevel
/// @tparam V the matrix operand type
template <class V> struct ColsFrom {
    V v;  ///< The matrix.
    int c0;  ///< The first column.
    /// @brief Element (r, j), `v(r, c0 + j)`.
    /// @param r the row
    /// @param j the column offset
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE auto& operator()(int r, int j) const { return v(r, c0 + j); }
};
/// @cond
template <class V> ColsFrom(V, int) -> ColsFrom<V>;
/// @endcond

/// @brief Columns c0 onwards of v read through f: (r, j) is f(v(r, c0 + j)), a value (the load_row
/// source).
/// @ingroup api_device_lowlevel
///
/// f scales by alpha or negates, so the product is formed only for a live element.
/// @tparam V the matrix operand type
/// @tparam F the element map
template <class V, class F> struct MapCols {
    V v;  ///< The matrix.
    int c0;  ///< The first column.
    F f;  ///< The element map.
    /// @brief Element (r, j), `f(v(r, c0 + j))`.
    /// @param r the row
    /// @param j the column offset
    /// @return the element, by value
    BATCHLAS_DEVICE_INLINE auto operator()(int r, int j) const { return f(v(r, c0 + j)); }
};
/// @cond
template <class V, class F> MapCols(V, int, F) -> MapCols<V, F>;
/// @endcond

/// @brief The packing order of a triangle: ByCols (LAPACK's AP) or ByRows (the transposed
/// traversal).
/// @ingroup api_device_operands
enum class PackOrder { ByCols, ByRows };

/// @brief The element map of an n x n triangle stored packed, with no padding.
/// @ingroup api_device_operands
///
/// ByCols is LAPACK's AP: Upper (i <= j) at `i + j (j + 1) / 2`, Lower (i >= j) at
/// `i + j (2n - j - 1) / 2`. ByRows is the transposed traversal: Lower row i holds (i, 0..i) at
/// `i (i + 1) / 2 + j`, Upper row i holds (i, i..n-1). decode(k) inverts index exactly.
/// @tparam UL the stored triangle
/// @tparam OR the packing order
template <Uplo UL, PackOrder OR = PackOrder::ByCols> struct PackedTri {
    static_assert(OR == PackOrder::ByCols || OR == PackOrder::ByRows, "PackedTri: PackOrder::ByCols or ByRows");
    /// @brief The number of stored elements, n (n + 1) / 2.
    /// @param n the order
    /// @return the packed size
    static constexpr int size(int n) { return n * (n + 1) / 2; }
    /// @brief The packed index of element (i, j) inside the triangle.
    /// @param i the row
    /// @param j the column
    /// @param n the order
    /// @return the index into the packed array
    static constexpr int index(int i, int j, int n) {
        // ByRows of Lower == ByCols of Upper with (i, j) swapped, and the same for Upper.
        constexpr bool cols = OR == PackOrder::ByCols;
        constexpr bool head = (UL == Uplo::Upper) == cols;  // a triangular-number prefix layout
        const int a = cols ? i : j, b = cols ? j : i;       // a: within the line, b: the line
        if constexpr (head) return a + b * (b + 1) / 2;
        else return a + b * (2 * n - b - 1) / 2;
    }
    /// @brief A (row, column) pair, as decode returns it.
    struct RowCol {
        ///@{
        /// @brief The row and the column.
        int i, j;
        ///@}
    };
    /// @brief The (i, j) of packed index k: an integer square root, then a step of +-1.
    ///
    /// Exact for k < 2^30 (a float sqrt estimate, then at most a step each way).
    /// @param k the packed index
    /// @param n the order
    /// @return the RowCol
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

/// @brief A packed matrix view: (i, j) inside the UL triangle of a packed array; outside it is not
/// addressable.
/// @ingroup api_device_operands
/// @tparam UL the stored triangle
/// @tparam OR the packing order
/// @tparam P the base: `T*` or any base with `p[k]`
template <Uplo UL, PackOrder OR, class P> struct PackedMat {
    P p;  ///< The packed array.
    int n;  ///< The order.
    /// @brief Element (i, j), inside the triangle.
    /// @param i the row
    /// @param j the column
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i, int j) const { return p[PackedTri<UL, OR>::index(i, j, n)]; }
    /// @brief The order, n.
    BATCHLAS_DEVICE_INLINE int rows() const { return n; }
    /// @brief The order, n.
    BATCHLAS_DEVICE_INLINE int cols() const { return n; }
};
/// @brief The packed view of an order-n triangle at p.
/// @ingroup api_device_operands
/// @tparam UL the stored triangle
/// @tparam OR the packing order (ByCols by default)
/// @param p the packed array
/// @param n the order
/// @return the PackedMat
template <Uplo UL, PackOrder OR = PackOrder::ByCols, class P> BATCHLAS_DEVICE_INLINE PackedMat<UL, OR, P> packed(P p, int n) { return {p, n}; }

/// @brief The diagonal step of a packed lower solve: v / L(s, s), as v * rd(s), or a true
/// division once any reciprocal was non-finite.
/// @ingroup api_device_lowlevel
///
/// divide is uniform over the work-group; unit leaves v.
/// @tparam Scalar the element type
template <class Scalar> struct TpDiag {
    const Scalar* ap;  ///< The packed lower factor, ByRows.
    const Scalar* rd;  ///< The reciprocals of its diagonal.
    ///@{
    /// @brief Unit diagonal; divide instead of multiplying by rd.
    bool unit, divide;
    ///@}
    /// @brief The diagonal step for element s.
    /// @tparam Ar the arithmetic
    /// @param s the row
    /// @param v the value to divide
    /// @return v / L(s, s), or v when unit
    template <class Ar> BATCHLAS_DEVICE_INLINE Scalar apply(int s, Scalar v) const {
        if (!unit) v = divide ? Ar::div(v, ap[PackedTri<Uplo::Lower, PackOrder::ByRows>::index(s, s, 0)]) : Ar::mul(v, rd[s]);
        return v;
    }
};

/// @brief One vector-width unit as a plain struct: Bytes / sizeof(Scalar) elements.
/// @ingroup api_device_operands
///
/// load and store go through a pointer the caller guarantees is Bytes-aligned; the compiler is
/// told so, which gives one 8- or 16-byte access where the target has one. A struct, not
/// sycl::vec: Metal has no vector types in this position.
/// @tparam Scalar the element type
/// @tparam Bytes the width in bytes, a power of two multiple of sizeof(Scalar)
template <class Scalar, int Bytes> struct Packet {
    static_assert(Bytes % sizeof(Scalar) == 0 && (Bytes & (Bytes - 1)) == 0, "Packet: Bytes is a power of two multiple of sizeof(T)");
    static constexpr int n = Bytes / static_cast<int>(sizeof(Scalar));  ///< The number of elements.
    Scalar v[n];  ///< The elements.
    /// @brief Element k.
    /// @param k the index
    /// @return a reference to the element
    BATCHLAS_DEVICE_INLINE Scalar& operator[](int k) { return v[k]; }
    /// @brief Element k.
    /// @param k the index
    /// @return a const reference to the element
    BATCHLAS_DEVICE_INLINE const Scalar& operator[](int k) const { return v[k]; }
    /// @brief Loads n elements from p.
    /// @param p a Bytes-aligned pointer
    /// @return the Packet
    BATCHLAS_DEVICE_INLINE static Packet load(const Scalar* p) {
        const Scalar* a = static_cast<const Scalar*>(__builtin_assume_aligned(p, Bytes));
        Packet r;
#pragma unroll
        for (int k = 0; k < n; ++k) r.v[k] = a[k];
        return r;
    }
    /// @brief Stores the n elements to p.
    /// @param p a Bytes-aligned pointer
    BATCHLAS_DEVICE_INLINE void store(Scalar* p) const {
        Scalar* a = static_cast<Scalar*>(__builtin_assume_aligned(p, Bytes));
#pragma unroll
        for (int k = 0; k < n; ++k) a[k] = v[k];
    }
};

/// @brief A generated vector: element i is f(i), as f returns it (a reference f returns writes
/// through).
/// @ingroup api_device_operands
/// @tparam F the generator
template <class F> struct FnVec {
    int n;  ///< The number of elements.
    F f;  ///< The generator.
    /// @brief Element i, f(i).
    /// @param i the index
    /// @return f(i)
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return f(i); }
    /// @brief The number of elements, n.
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
/// @brief The generated vector of n elements f(i).
/// @ingroup api_device_operands
/// @param n the number of elements
/// @param f the generator, `f(int i)`
/// @return the FnVec
template <class F> BATCHLAS_DEVICE_INLINE FnVec<F> fn_vec(int n, F f) { return {n, f}; }
/// @brief The remapped vector f(x(i)), by value.
/// @ingroup api_device_operands
/// @param x the vector
/// @param f the element map
/// @return a FnVec of x.size() elements
template <class X, class F> BATCHLAS_DEVICE_INLINE auto mapped(const X& x, F f) {
    return fn_vec(x.size(), [x, f](int i) { return f(x(i)); });
}
/// @brief A strided slice of a vector: element i is x(start + i * step).
/// @ingroup api_device_operands
/// @tparam X the vector operand type
template <class X> struct Sliced {
    X x;  ///< The vector.
    ///@{
    /// @brief The first element, the size and the step.
    int start, n, step;
    ///@}
    /// @brief Element i, x(start + i * step).
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return x(start + i * step); }
    /// @brief The number of elements, n.
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
/// @brief The slice x(start + i * step) of n elements.
/// @ingroup api_device_operands
/// @param x the vector
/// @param start the first element
/// @param n the number of elements
/// @param step the step
/// @return the Sliced view
template <class X> BATCHLAS_DEVICE_INLINE Sliced<X> slice(const X& x, int start, int n, int step = 1) { return {x, start, n, step}; }
/// @brief A gathered vector: element i is x(p[i]).
/// @ingroup api_device_operands
/// @tparam X the vector operand type
/// @tparam P the index array type
template <class X, class P> struct Gathered {
    X x;  ///< The vector.
    P p;  ///< The indices.
    int n;  ///< The number of elements.
    /// @brief Element i, x(p[i]).
    /// @param i the index
    /// @return the element
    BATCHLAS_DEVICE_INLINE decltype(auto) operator()(int i) const { return x(static_cast<int>(p[i])); }
    /// @brief The number of elements, n.
    BATCHLAS_DEVICE_INLINE int size() const { return n; }
};
/// @brief The gathered vector x(p[i]) of n elements.
/// @ingroup api_device_operands
/// @param x the vector
/// @param p the indices
/// @param n the number of elements
/// @return the Gathered view
template <class X, class P> BATCHLAS_DEVICE_INLINE Gathered<X, P> gathered(const X& x, P p, int n) { return {x, p, n}; }

}  // namespace batchlas::device
