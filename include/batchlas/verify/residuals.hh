// SPDX-License-Identifier: MIT
// Residuals and backward errors of batchlas::verify (docs/design/verification.md). Every item is
// read through ld and stride, promoted to double / complex<double>, and the worst value over the
// checked items is a nanmax: a NaN anywhere is the result. The dense checks run as SYCL kernels in
// libbatchlas_verify (engine.hh); the O(input) scans below stay on the host.
#pragma once

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/verify/engine.hh>
#include <batchlas/verify/inputs.hh>
#include <batchlas/verify/items.hh>
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/scalar.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace batchlas::verify {
namespace detail {

inline constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

[[noreturn]] inline void bad(const char* who, const char* what) {
    throw std::invalid_argument(std::string("batchlas::verify::") + who + ": " + what);
}

inline std::vector<int> pick(std::span<const int> items, int batch) {
    return items.empty() ? default_items(batch) : std::vector<int>(items.begin(), items.end());
}

template <class T> promoted_t<T> get(const Item<T>& m, int r, int c) {
    return up(m.data[static_cast<long long>(c) * m.ld + r]);
}

// den > 0 is false for a NaN den, so test NaN first or a poisoned norm reads as "no scale".
inline double quot(double num, double den) {
    if (std::isnan(num) || std::isnan(den)) return kNaN;
    return den > 0 ? num / den : num;
}

template <class E> struct VecItem {
    const E* data;
    int size, inc;
    E operator[](int i) const { return data[static_cast<long long>(i) * inc]; }
};

template <class E> VecItem<E> vec_of(const VectorView<E>& v, int item) {
    if (item < 0 || item >= v.batch_size()) bad("vector", "batch item out of range");
    return {v.data_ptr() + static_cast<long long>(item) * v.stride(), v.size(), v.inc()};
}

template <class View> constexpr bool complex_v = is_complex<value_of_t<View>>::value;

// The engine's descriptor of a dense view; item_of rejects a heterogeneous one.
template <class View> Operand operand(const View& A) {
    using T = value_of_t<View>;
    if (A.batch_size() < 1) return {nullptr, dtype_of<T>(), A.rows(), A.cols(), A.ld(), 1, 0, 0};
    const auto m = item_of(A, 0);
    return {m.data, dtype_of<T>(), m.rows, m.cols, m.ld, 1, static_cast<long long>(A.stride()), A.batch_size()};
}

// The first @p size entries of each item of @p v, as a column.
template <class E> Operand operand(const VectorView<E>& v, int size) {
    if constexpr (std::is_same_v<std::remove_const_t<E>, std::int32_t>)
        return {v.data_ptr(), Dtype::i32, size, 1, 1, v.inc(), static_cast<long long>(v.stride()), v.batch_size()};
    else
        return {v.data_ptr(), dtype_of<std::remove_const_t<E>>(), size, 1, 1, v.inc(), static_cast<long long>(v.stride()), v.batch_size()};
}

inline bool square_shape(Shape s) { return s != Shape::general && s != Shape::lower && s != Shape::upper; }

template <class View> void check_shape(const View& A, Shape s) {
    if (square_shape(s) && A.rows() != A.cols()) bad("shape", "unit, Hermitian and symmetric shapes need a square operand");
}

template <class View> std::pair<int, int> op_dims(const View& A, Transpose t) {
    return t == Transpose::NoTrans ? std::pair{A.rows(), A.cols()} : std::pair{A.cols(), A.rows()};
}

template <class VA, class VX, class VB>
double solve_residual(const VA& A0, Shape sa, Transpose ta, const VX& X, const VB& B0, std::span<const int> items, double full_limit) {
    if (square_shape(sa) && A0.rows() != A0.cols()) bad("solve_residual", "unit, Hermitian and symmetric shapes need a square operand");
    const auto [m, k] = op_dims(A0, ta);
    const bool identity = B0.rows() == 0;
    if (X.rows() != k || (!identity && (B0.rows() != m || B0.cols() != X.cols()))) bad("solve_residual", "dimension mismatch");
    const auto it = pick(items, A0.batch_size());
    return solve(operand(A0), sa, ta, nullptr, operand(X), identity ? Operand{} : operand(B0), complex_v<VA>, it, full_limit);
}

template <class VA, class VX, class VB>
double solve_residual(const VA& A0, const VX& X, const VB& B0, std::span<const int> items, double full_limit) {
    return solve_residual(A0, Shape::general, Transpose::NoTrans, X, B0, items, full_limit);
}

}  // namespace detail

/// ‖A0 − LLᴴ‖_F / ‖A0‖_F (Lower) or ‖A0 − UᴴU‖_F / ‖A0‖_F (Upper), over the factor's triangle;
/// the other triangle of F is never read.
template <class VA, class VF>
double potrf_residual(const VA& A0, const VF& F, Uplo uplo, std::span<const int> items = {}) {
    const int n = F.rows();
    if (F.cols() != n || A0.rows() != n || A0.cols() != n) detail::bad("potrf_residual", "square matrices of equal order only");
    return detail::potrf(detail::operand(A0), detail::operand(F), uplo == Uplo::Upper, detail::complex_v<VA>,
                         detail::pick(items, A0.batch_size()));
}

/// ‖A0 − Q·B·Qᴴ‖_F / ‖A0‖_F for square A0, an explicit Q and a dense B (a tridiagonal or banded
/// reduction stored densely; entries outside the band must be stored zeros). Every element of each
/// operand is read. Judged as a factorization.
template <class VA, class VQ, class VB>
double similarity_residual(const VA& A0, const VQ& Q, const VB& B, std::span<const int> items = {}) {
    const int n = A0.rows();
    if (A0.cols() != n || Q.rows() != n || Q.cols() != n || B.rows() != n || B.cols() != n)
        detail::bad("similarity_residual", "square matrices of equal order only");
    return detail::similarity(detail::operand(A0), detail::operand(Q), detail::operand(B), detail::complex_v<VA>,
                              detail::pick(items, A0.batch_size()));
}

/// ‖PA0 − LU‖_F / ‖A0‖_F for an m x n factor; pivots packed 1-based, min(m, n) per item. An
/// out-of-range pivot makes that item NaN.
template <class VA, class VF>
double getrf_residual(const VA& A0, const VF& F, const VectorView<std::int32_t>& piv, std::span<const int> items = {}) {
    const int m = F.rows(), n = F.cols(), mn = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) detail::bad("getrf_residual", "A0 and F differ in shape");
    if (piv.size() < mn) detail::bad("getrf_residual", "fewer than min(m, n) pivots per item");
    return detail::getrf(detail::operand(A0), detail::operand(F), detail::operand(piv, mn), detail::complex_v<VA>,
                         detail::pick(items, A0.batch_size()));
}

/// ‖A0 − QR‖_F / ‖A0‖_F; F in LAPACK geqrf storage (R on and above the diagonal, reflector k below
/// it with an implicit unit), min(m, n) reflectors, H_k = I − tau_k v_k v_kᴴ.
template <class VA, class VF, class T>
double qr_residual(const VA& A0, const VF& F, const VectorView<T>& tau, std::span<const int> items = {}) {
    const int m = F.rows(), n = F.cols(), k = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) detail::bad("qr_residual", "A0 and F differ in shape");
    if (tau.size() < k) detail::bad("qr_residual", "fewer than min(m, n) tau entries per item");
    return detail::qr(detail::operand(A0), detail::operand(F), detail::operand(tau, k), is_complex<T>::value,
                      detail::pick(items, A0.batch_size()));
}

/// The first @p cols columns of Q = H_0 … H_{k−1} of item @p item (k = min(m, n) reflectors of the
/// m x n @p F in geqrf storage, as qr_residual): m x cols, column-major, ld = m. F's diagonal and
/// upper triangle are not read.
template <class VF, class T>
std::vector<promoted_t<T>> form_q(const VF& F, const VectorView<T>& tau, int item, int cols) {
    const int m = F.rows(), k = std::min(m, F.cols());
    if (cols < 0 || cols > m) detail::bad("form_q", "cols must lie in [0, rows]");
    if (tau.size() < k) detail::bad("form_q", "fewer than min(m, n) tau entries per item");
    const auto raw = detail::form_q(detail::operand(F), detail::operand(tau, k), is_complex<T>::value, item, cols);
    std::vector<promoted_t<T>> Q(static_cast<std::size_t>(m) * static_cast<std::size_t>(cols));
    for (std::size_t i = 0; i < Q.size(); ++i) {
        if constexpr (is_complex<T>::value) Q[i] = {raw[2 * i], raw[2 * i + 1]};
        else Q[i] = raw[i];
    }
    return Q;
}

/// ‖A0 − Q·triu(R)‖_F / ‖A0‖_F with an explicit m x k @p Q; R's first k rows are read on and above
/// the diagonal only, so a geqrf factor (m x n) and a compact k x n R are both accepted.
template <class VA, class VQ, class VR>
double qr_reconstruction(const VA& A0, const VQ& Q, const VR& R, std::span<const int> items = {}) {
    const int m = A0.rows(), n = A0.cols(), k = Q.cols();
    if (Q.rows() != m || R.cols() != n || R.rows() < k) detail::bad("qr_reconstruction", "need A0 m x n, Q m x k, R at least k x n");
    return detail::qr_reconstruction(detail::operand(A0), detail::operand(Q), detail::operand(R), detail::complex_v<VA>,
                                     detail::pick(items, A0.batch_size()));
}

/// ‖A0X − B0‖_F / (‖A0‖_F‖X‖_F); an empty B0 (rows() == 0) is the identity (getri). Above 2^28
/// multiply-adds of A0·X per item, 8 seeded probe columns of X stand in for all of them.
template <class VA, class VX, class VB>
double solve_residual(const VA& A0, const VX& X, const VB& B0, std::span<const int> items = {}) {
    return detail::solve_residual(A0, X, B0, items, double(1 << 28));
}

/// ‖op(A0)X − B0‖_F / (‖A0‖_F‖X‖_F) with A0 read through @p sa (as gemm_backward_error) and @p ta.
template <class VA, class VX, class VB>
double solve_residual(const VA& A0, Shape sa, Transpose ta, const VX& X, const VB& B0, std::span<const int> items = {}) {
    return detail::solve_residual(A0, sa, ta, X, B0, items, double(1 << 28));
}

/// solve_residual for X computed from getrf factors: A = P·L·U is formed per checked item from @p F
/// (unit L, U) and the packed 1-based @p piv, then ‖op(A)X − B0‖_F / (‖A‖_F‖X‖_F). An empty B0 is
/// the identity (getri from factors); an out-of-range pivot makes that item NaN.
template <class VF, class VX, class VB>
double lu_solve_residual(const VF& F, const VectorView<std::int32_t>& piv, Transpose trans, const VX& X, const VB& B0,
                         std::span<const int> items = {}) {
    const int n = F.rows();
    if (F.cols() != n) detail::bad("lu_solve_residual", "square factors only");
    if (piv.size() < n) detail::bad("lu_solve_residual", "fewer than n pivots per item");
    const bool identity = B0.rows() == 0;
    if (X.rows() != n || (!identity && (B0.rows() != n || B0.cols() != X.cols()))) detail::bad("solve_residual", "dimension mismatch");
    const auto p = detail::operand(piv, n);
    return detail::solve(detail::operand(F), Shape::general, trans, &p, detail::operand(X), identity ? detail::Operand{} : detail::operand(B0),
                         detail::complex_v<VF>, detail::pick(items, F.batch_size()), double(1 << 28));
}

/// Componentwise max |C − (α op(A) op(B) + βC0)| / (|α||op(A)||op(B)| + |β||C0|) over the elements
/// @p sc selects; A and B are read as their Shape says. A zero denominator uses the numerator.
template <class VA, class VB, class VC0, class VC>
double gemm_backward_error(const VA& A, Shape sa, Transpose ta, const VB& B, Shape sb, Transpose tb, const VC0& C0, const VC& C, Shape sc,
                           promoted_t<detail::value_of_t<VC>> alpha, promoted_t<detail::value_of_t<VC>> beta, std::span<const int> items = {}) {
    detail::check_shape(A, sa);
    detail::check_shape(B, sb);
    const auto [am, ak] = detail::op_dims(A, ta);
    const auto [bk, bn] = detail::op_dims(B, tb);
    if (am != C.rows() || bn != C.cols() || ak != bk) detail::bad("gemm_backward_error", "dimension mismatch");
    const bool use_c0 = beta != decltype(beta)(0);
    if (use_c0 && (C0.rows() != C.rows() || C0.cols() != C.cols())) detail::bad("gemm_backward_error", "C0 and C differ in shape");
    const auto c0 = use_c0 ? detail::operand(C0) : detail::Operand{};
    const auto c = detail::operand(C);
    return detail::componentwise(detail::operand(A), sa, ta, detail::operand(B), sb, tb, &c0, &c, sc, alpha, beta, detail::complex_v<VC>,
                                 detail::pick(items, C.batch_size()), false);
}

/// Largest denominator of gemm_backward_error over item @p item: max_ij |α|(|op(A)||op(B)|)_ij + |β||C0_ij|
/// (C0 not read when β == 0). An absolute tolerance divided by it is a componentwise one.
template <class VA, class VB, class VC0>
double gemm_max_denominator(const VA& A, Shape sa, Transpose ta, const VB& B, Shape sb, Transpose tb, const VC0& C0,
                            promoted_t<detail::value_of_t<VC0>> alpha, promoted_t<detail::value_of_t<VC0>> beta, int item = 0) {
    detail::check_shape(A, sa);
    detail::check_shape(B, sb);
    const auto [am, ak] = detail::op_dims(A, ta);
    const auto [bk, bn] = detail::op_dims(B, tb);
    if (ak != bk) detail::bad("gemm_max_denominator", "dimension mismatch");
    const bool use_c0 = beta != decltype(beta)(0);
    if (use_c0 && (C0.rows() != am || C0.cols() != bn)) detail::bad("gemm_max_denominator", "C0 and op(A) op(B) differ in shape");
    const auto c0 = use_c0 ? detail::operand(C0) : detail::Operand{};
    const int one[] = {item};
    return detail::componentwise(detail::operand(A), sa, ta, detail::operand(B), sb, tb, &c0, nullptr, Shape::general, alpha, beta,
                                 detail::complex_v<VC0>, one, true);
}

/// Componentwise backward error of y = α op(A) x + β y0, as gemm_backward_error.
template <class VA, class T>
double gemv_backward_error(const VA& A, Transpose ta, const VectorView<T>& x, const VectorView<T>& y0, const VectorView<T>& y,
                           promoted_t<T> alpha, promoted_t<T> beta, std::span<const int> items = {}) {
    const auto [am, ak] = detail::op_dims(A, ta);
    if (x.size() != ak || y.size() != am) detail::bad("gemv_backward_error", "dimension mismatch");
    const bool use_c0 = beta != promoted_t<T>(0);
    if (use_c0 && y0.size() != y.size()) detail::bad("gemv_backward_error", "y0 and y differ in size");
    const auto c0 = use_c0 ? detail::operand(y0, y.size()) : detail::Operand{};
    const auto c = detail::operand(y, y.size());
    return detail::componentwise(detail::operand(A), Shape::general, ta, detail::operand(x, x.size()), Shape::general, Transpose::NoTrans,
                                 &c0, &c, Shape::general, alpha, beta, is_complex<T>::value, detail::pick(items, y.batch_size()), false);
}

/// ‖op(A)X − αB0‖_F / (‖A‖_F‖X‖_F + |α|‖B0‖_F) (Side::Right: X op(A)); only the @p uplo triangle of
/// A is read, and its diagonal only for Diag::NonUnit.
template <class VA, class VX, class VB>
double trsm_residual(const VA& A, Side side, Uplo uplo, Transpose ta, Diag diag, const VX& X, const VB& B0,
                     promoted_t<detail::value_of_t<VX>> alpha, std::span<const int> items = {}) {
    const bool unit = diag == Diag::Unit;
    const Shape s = uplo == Uplo::Lower ? (unit ? Shape::unit_lower : Shape::lower) : (unit ? Shape::unit_upper : Shape::upper);
    const bool left = side == Side::Left;
    if (A.rows() != A.cols()) detail::bad("trsm_residual", "A must be square");
    if ((left ? X.rows() : X.cols()) != A.rows() || B0.rows() != X.rows() || B0.cols() != X.cols()) detail::bad("trsm_residual", "dimension mismatch");
    return detail::trsm(detail::operand(A), s, ta, left, detail::operand(X), detail::operand(B0), alpha, detail::complex_v<VX>,
                        detail::pick(items, X.batch_size()));
}

/// Componentwise backward error of C = α op(A) op(B) + βC0 with a CSR A; an out-of-range column
/// index makes the item NaN.
template <class T, class VB, class VC0, class VC>
double spmm_backward_error(const MatrixView<T, MatrixFormat::CSR>& A, Transpose ta, const VB& B, Transpose tb, const VC0& C0, const VC& C,
                           promoted_t<T> alpha, promoted_t<T> beta, std::span<const int> items = {}) {
    using D = promoted_t<T>;
    if (A.is_heterogeneous()) detail::bad("spmm_backward_error", "heterogeneous views are not supported");
    const auto [am, ak] = detail::op_dims(A, ta);
    const auto [bk, bn] = detail::op_dims(B, tb);
    if (am != C.rows() || bn != C.cols() || ak != bk) detail::bad("spmm_backward_error", "dimension mismatch");
    const bool use_c0 = beta != D(0);
    if (use_c0 && (C0.rows() != C.rows() || C0.cols() != C.cols())) detail::bad("spmm_backward_error", "C0 and C differ in shape");
    // The checked items packed back to back in the promoted type (A densified, so duplicate entries
    // sum as they always have), and checked in one engine call.
    const auto picked = detail::pick(items, C.batch_size());
    const int ar = A.rows(), ac = A.cols();
    const std::size_t per_a = std::size_t(ar) * std::size_t(ac);
    std::vector<D> dense(per_a * picked.size(), D(0));
    bool bad_index = false;
    for (std::size_t g = 0; g < picked.size(); ++g) {
        const int b = picked[g];
        if (b < 0 || b >= A.batch_size()) detail::bad("spmm_backward_error", "batch item out of range");
        const int* ro = A.row_offsets().data() + static_cast<long long>(b) * A.offset_stride();
        const int* ci = A.col_indices().data() + static_cast<long long>(b) * A.matrix_stride();
        const T* v = A.data_ptr() + static_cast<long long>(b) * A.matrix_stride();
        for (int r = 0; r < ar; ++r)
            for (int p = ro[r]; p < ro[r + 1]; ++p) {
                if (ci[p] < 0 || ci[p] >= ac) { bad_index = true; continue; }
                dense[g * per_a + static_cast<std::size_t>(ci[p]) * ar + r] += up(v[p]);
            }
    }
    const auto pack = [&](const auto& V, std::vector<D>& out) {
        out.assign(std::size_t(V.rows()) * std::size_t(V.cols()) * picked.size(), D(0));
        for (std::size_t g = 0; g < picked.size(); ++g) {
            const auto m = detail::item_of(V, picked[g]);
            for (int c = 0; c < m.cols; ++c)
                for (int r = 0; r < m.rows; ++r) out[(g * m.cols + c) * m.rows + r] = detail::get(m, r, c);
        }
        return detail::Operand{out.data(), detail::dtype_of<D>(), V.rows(), V.cols(), std::max(V.rows(), 1), 1,
                               static_cast<long long>(V.rows()) * V.cols(), int(picked.size())};
    };
    std::vector<D> bp, c0p, cp;
    const auto bo = pack(B, bp), co = pack(C, cp);
    const auto c0 = use_c0 ? pack(C0, c0p) : detail::Operand{};
    const detail::Operand a{dense.data(), detail::dtype_of<D>(), ar, ac, std::max(ar, 1), 1, static_cast<long long>(per_a), int(picked.size())};
    std::vector<int> pos(picked.size());
    for (std::size_t g = 0; g < pos.size(); ++g) pos[g] = int(g);
    const double e = pos.empty() ? 0.0
                                 : detail::componentwise(a, Shape::general, ta, bo, Shape::general, tb, &c0, &co, Shape::general, alpha, beta,
                                                         is_complex<T>::value, pos, false);
    return bad_index ? detail::kNaN : e;
}

/// Componentwise backward error of the rank-2k update over the @p uplo triangle of C:
/// her2k (@p hermitian) C = α op(A)op(B)ᴴ + conj(α) op(B)op(A)ᴴ + βC0, with β real and C0's diagonal
/// read as real; syr2k C = α op(A)op(B)ᵀ + α op(B)op(A)ᵀ + βC0. The denominator is
/// |α||op(A)||op(B)|ᵀ + |α||op(B)||op(A)|ᵀ + |β||C0|. For complex data her2k takes NoTrans or
/// ConjTrans and syr2k NoTrans or Trans (std::invalid_argument otherwise).
template <class VA, class VB, class VC0, class VC>
double rank2k_backward_error(const VA& A, const VB& B, Transpose trans, const VC0& C0, const VC& C, Uplo uplo,
                             promoted_t<detail::value_of_t<VC>> alpha, promoted_t<detail::value_of_t<VC>> beta, bool hermitian,
                             std::span<const int> items = {}) {
    using T = detail::value_of_t<VC>;
    using D = promoted_t<T>;
    if constexpr (is_complex<T>::value) {
        if (hermitian && trans == Transpose::Trans) detail::bad("rank2k_backward_error", "her2k takes NoTrans or ConjTrans");
        if (!hermitian && trans == Transpose::ConjTrans) detail::bad("rank2k_backward_error", "syr2k takes NoTrans or Trans");
        if (hermitian && std::imag(beta) != 0.0) detail::bad("rank2k_backward_error", "her2k takes a real beta");
    }
    const auto [an, ak] = detail::op_dims(A, trans);
    const auto [bn, bk] = detail::op_dims(B, trans);
    if (C.rows() != C.cols() || an != C.rows() || bn != C.rows() || ak != bk) detail::bad("rank2k_backward_error", "dimension mismatch");
    const bool use_c0 = beta != D(0);
    if (use_c0 && (C0.rows() != C.rows() || C0.cols() != C.cols())) detail::bad("rank2k_backward_error", "C0 and C differ in shape");
    const auto c0 = use_c0 ? detail::operand(C0) : detail::Operand{};
    return detail::rank2k(detail::operand(A), detail::operand(B), trans, &c0, detail::operand(C), uplo, alpha, beta, hermitian,
                          is_complex<T>::value, detail::pick(items, C.batch_size()));
}

/// max cabs1(L(i,k)·U(k,k)) / cabs1(U(k,k)) over a getrf factor (min(m, n) columns; a zero U(k,k)
/// skips its column). Partial pivoting on cabs1 keeps it at most 1 up to rounding
/// (pivot_ratio_bound); a modulus pivot rule or a wrong argmax does not. Residuals cannot see this:
/// any valid pivot order passes them.
template <class VF>
double pivot_ratio(const VF& F, std::span<const int> items = {}) {
    const int m = F.rows(), k = std::min(m, F.cols());
    double worst = 0;
    for (int b : detail::pick(items, F.batch_size())) {
        const auto f = detail::item_of(F, b);
        for (int j = 0; j < k; ++j) {
            const auto ukk = detail::get(f, j, j);
            const double den = cabs1(ukk);
            if (den == 0.0) continue;
            for (int i = j + 1; i < m; ++i) worst = nanmax(worst, cabs1(detail::get(f, i, j) * ukk) / den);
        }
    }
    return worst;
}

/// ‖QᴴQ − I‖_F over Q's columns.
template <class VQ>
double orthogonality(const VQ& Q, std::span<const int> items = {}) {
    return detail::orthogonality(detail::operand(Q), detail::complex_v<VQ>, detail::pick(items, Q.batch_size()));
}

/// ‖AV − V diag(w)‖_F / ‖A‖_F; A Hermitian, read from its lower triangle and mirrored.
template <class VA, class VV, class R>
double eigen_residual(const VA& A, const VV& V, const VectorView<R>& w, std::span<const int> items = {}) {
    detail::check_shape(A, Shape::hermitian_lower);
    if (V.rows() != A.rows() || w.size() < V.cols()) detail::bad("eigen_residual", "dimension mismatch");
    return detail::eigen(detail::operand(A), detail::operand(V), detail::operand(w, V.cols()), detail::complex_v<VV>,
                         detail::pick(items, V.batch_size()));
}

/// max |w − ref| / scale per item; ref[item] holds w.size() values in the caller's order.
template <class R>
double values_error(const VectorView<R>& w, const std::vector<std::vector<double>>& ref, double scale, std::span<const int> items = {}) {
    double worst = 0;
    for (int b : detail::pick(items, w.batch_size())) {
        const auto v = detail::vec_of(w, b);
        if (b >= int(ref.size()) || int(ref[static_cast<std::size_t>(b)].size()) != v.size) detail::bad("values_error", "ref has no matching item");
        for (int i = 0; i < v.size; ++i)
            worst = nanmax(worst, detail::quot(std::fabs(double(v[i]) - ref[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)]), scale));
    }
    return worst;
}

/// True when every pivot of every item lies in [1, n].
inline bool pivots_valid(const VectorView<std::int32_t>& piv, int n) {
    for (int b = 0; b < piv.batch_size(); ++b) {
        const auto p = detail::vec_of(piv, b);
        for (int k = 0; k < p.size; ++k)
            if (p[k] < 1 || p[k] > n) return false;
    }
    return true;
}

}  // namespace batchlas::verify
