// SPDX-License-Identifier: MIT
// Residuals and backward errors of batchlas::verify (docs/design/verification.md). Every item is
// read through item_of (ld and stride), promoted to double / complex<double>, and the worst value
// over the checked items is a nanmax: a NaN anywhere is the result.
#pragma once

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
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
#include <utility>
#include <vector>

namespace batchlas::verify {

/// How an operand is read: the stored triangle, an implicit unit diagonal, or a mirrored triangle.
enum class Shape { general, lower, upper, unit_lower, unit_upper, hermitian_lower, hermitian_upper, symmetric_lower, symmetric_upper };

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

template <class D> struct Dense {
    int rows = 0, cols = 0;
    std::vector<D> a;
    Dense(int r, int c) : rows(r), cols(c), a(static_cast<std::size_t>(r) * static_cast<std::size_t>(c), D(0)) {}
    D& operator()(int r, int c) { return a[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)]; }
    D operator()(int r, int c) const { return a[static_cast<std::size_t>(c) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(r)]; }
    double frobenius() const {
        double s = 0;
        for (const D& x : a) s += abs(x) * abs(x);
        return std::sqrt(s);
    }
};

inline bool lower_family(Shape s) {
    return s == Shape::lower || s == Shape::unit_lower || s == Shape::hermitian_lower || s == Shape::symmetric_lower;
}
inline bool upper_family(Shape s) {
    return s == Shape::upper || s == Shape::unit_upper || s == Shape::hermitian_upper || s == Shape::symmetric_upper;
}

// Element (i, j) of the matrix the shape describes. Only the named triangle (and the diagonal
// unless unit) is read; a Hermitian diagonal is taken real, as LAPACK does.
template <class T> promoted_t<T> shaped(const Item<T>& m, Shape s, int i, int j) {
    using D = promoted_t<T>;
    switch (s) {
        case Shape::general: return get(m, i, j);
        case Shape::lower: return i >= j ? get(m, i, j) : D(0);
        case Shape::upper: return i <= j ? get(m, i, j) : D(0);
        case Shape::unit_lower: return i == j ? D(1) : (i > j ? get(m, i, j) : D(0));
        case Shape::unit_upper: return i == j ? D(1) : (i < j ? get(m, i, j) : D(0));
        case Shape::hermitian_lower:
            if (i == j) return D(std::real(get(m, i, i)));
            return i > j ? get(m, i, j) : conj(get(m, j, i));
        case Shape::hermitian_upper:
            if (i == j) return D(std::real(get(m, i, i)));
            return i < j ? get(m, i, j) : conj(get(m, j, i));
        case Shape::symmetric_lower: return i >= j ? get(m, i, j) : get(m, j, i);
        case Shape::symmetric_upper: return i <= j ? get(m, i, j) : get(m, j, i);
    }
    return D(0);
}

template <class T> Dense<promoted_t<T>> shaped_dense(const Item<T>& m, Shape s) {
    if (s != Shape::general && s != Shape::lower && s != Shape::upper && m.rows != m.cols) bad("shape", "unit, Hermitian and symmetric shapes need a square operand");
    Dense<promoted_t<T>> out(m.rows, m.cols);
    for (int j = 0; j < m.cols; ++j)
        for (int i = 0; i < m.rows; ++i) out(i, j) = shaped(m, s, i, j);
    return out;
}

template <class D> Dense<D> apply_op(const Dense<D>& M, Transpose t) {
    if (t == Transpose::NoTrans) return M;
    Dense<D> out(M.cols, M.rows);
    for (int j = 0; j < M.cols; ++j)
        for (int i = 0; i < M.rows; ++i) out(j, i) = t == Transpose::ConjTrans ? conj(M(i, j)) : M(i, j);
    return out;
}

// max over the selected (i, j) of |C - (alpha opA opB + beta C0)| / (|alpha||opA||opB| + |beta||C0|).
// C0 is not read when beta == 0 (BLAS semantics: it may hold anything).
template <class D, class C0At, class CAt>
double componentwise(const Dense<D>& opA, const Dense<D>& opB, C0At c0, CAt c, int m, int n, Shape sc, D alpha, D beta) {
    double worst = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            if ((lower_family(sc) && i < j) || (upper_family(sc) && i > j)) continue;
            D acc = D(0);
            double mag = 0;
            for (int l = 0; l < opA.cols; ++l) {
                acc += opA(i, l) * opB(l, j);
                mag += abs(opA(i, l)) * abs(opB(l, j));
            }
            D want = alpha * acc;
            double den = abs(alpha) * mag;
            if (beta != D(0)) {
                const D z = c0(i, j);
                want += beta * z;
                den += abs(beta) * abs(z);
            }
            worst = nanmax(worst, quot(abs(c(i, j) - want), den));
        }
    return worst;
}

template <class T> Dense<promoted_t<T>> csr_dense(const MatrixView<T, MatrixFormat::CSR>& A, int item, bool& bad_index) {
    if (A.is_heterogeneous()) bad("spmm_backward_error", "heterogeneous views are not supported");
    if (item < 0 || item >= A.batch_size()) bad("spmm_backward_error", "batch item out of range");
    const int* ro = A.row_offsets().data() + static_cast<long long>(item) * A.offset_stride();
    const int* ci = A.col_indices().data() + static_cast<long long>(item) * A.matrix_stride();
    const T* v = A.data_ptr() + static_cast<long long>(item) * A.matrix_stride();
    Dense<promoted_t<T>> out(A.rows(), A.cols());
    for (int r = 0; r < A.rows(); ++r)
        for (int p = ro[r]; p < ro[r + 1]; ++p) {
            if (ci[p] < 0 || ci[p] >= A.cols()) { bad_index = true; continue; }
            out(r, ci[p]) += up(v[p]);
        }
    return out;
}

// ||A0 X V - B0 V||_F / (||A0||_F ||X V||_F). V is the identity while rows*cols*ncols of A0 X is at
// most full_limit, else 8 seeded dense probe columns: O(n^2) host work at n = 4096, and a dense probe
// still sees any wrong entry of X.
template <class VA, class VX, class VB>
double solve_residual(const VA& A0, const VX& X, const VB& B0, std::span<const int> items, double full_limit) {
    using T = value_of_t<VA>;
    using D = promoted_t<T>;
    const bool identity = B0.rows() == 0;
    const int m = A0.rows(), k = A0.cols(), nc = X.cols();
    if (X.rows() != k || (!identity && (B0.rows() != m || B0.cols() != nc))) bad("solve_residual", "dimension mismatch");
    const bool full = double(m) * double(k) * double(nc) <= full_limit;
    const int nv = full ? nc : 8;
    std::vector<D> V(full ? 0 : static_cast<std::size_t>(nc) * static_cast<std::size_t>(nv));
    Rng rng(4242);
    for (D& v : V) v = up(draw<T>(rng));
    auto vat = [&](int c, int j) { return full ? D(c == j ? 1 : 0) : V[static_cast<std::size_t>(j) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)]; };
    std::vector<D> xv(static_cast<std::size_t>(k)), bv(static_cast<std::size_t>(m));
    double worst = 0;
    for (int b : pick(items, A0.batch_size())) {
        const auto a = item_of(A0, b);
        const auto x = item_of(X, b);
        Item<T> rhs{};
        if (!identity) rhs = item_of(B0, b);
        double na = 0, nx = 0, num = 0;
        for (int c = 0; c < k; ++c)
            for (int r = 0; r < m; ++r) na += abs(get(a, r, c)) * abs(get(a, r, c));
        for (int j = 0; j < nv; ++j) {
            std::fill(xv.begin(), xv.end(), D(0));
            std::fill(bv.begin(), bv.end(), D(0));
            for (int c = 0; c < nc; ++c) {
                const D w = vat(c, j);
                if (w == D(0)) continue;
                for (int r = 0; r < k; ++r) xv[static_cast<std::size_t>(r)] += get(x, r, c) * w;
                for (int r = 0; r < m; ++r) bv[static_cast<std::size_t>(r)] += (identity ? D(r == c ? 1 : 0) : get(rhs, r, c)) * w;
            }
            for (int r = 0; r < k; ++r) nx += abs(xv[static_cast<std::size_t>(r)]) * abs(xv[static_cast<std::size_t>(r)]);
            for (int r = 0; r < m; ++r) {
                D acc = D(0);
                for (int l = 0; l < k; ++l) acc += get(a, r, l) * xv[static_cast<std::size_t>(l)];
                const double d = abs(acc - bv[static_cast<std::size_t>(r)]);
                num += d * d;
            }
        }
        worst = nanmax(worst, quot(std::sqrt(num), std::sqrt(na) * std::sqrt(nx)));
    }
    return worst;
}

}  // namespace detail

/// ‖A0 − LLᴴ‖_F / ‖A0‖_F (Lower) or ‖A0 − UᴴU‖_F / ‖A0‖_F (Upper), over the factor's triangle;
/// the other triangle of F is never read.
template <class VA, class VF>
double potrf_residual(const VA& A0, const VF& F, Uplo uplo, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VA>>;
    const int n = F.rows();
    if (F.cols() != n || A0.rows() != n || A0.cols() != n) detail::bad("potrf_residual", "square matrices of equal order only");
    const bool upper = uplo == Uplo::Upper;
    double worst = 0;
    for (int b : detail::pick(items, A0.batch_size())) {
        const auto a = detail::item_of(A0, b);
        const auto f = detail::item_of(F, b);
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = upper ? 0 : j; upper ? i <= j : i < n; ++i) {
                D acc = D(0);
                if (upper)
                    for (int k = 0; k <= i; ++k) acc += conj(detail::get(f, k, i)) * detail::get(f, k, j);
                else
                    for (int k = 0; k <= j; ++k) acc += detail::get(f, i, k) * conj(detail::get(f, j, k));
                const D x = detail::get(a, i, j);
                num += abs(acc - x) * abs(acc - x);
                den += abs(x) * abs(x);
            }
        worst = nanmax(worst, detail::quot(std::sqrt(num), std::sqrt(den)));
    }
    return worst;
}

/// ‖PA0 − LU‖_F / ‖A0‖_F for an m x n factor; pivots packed 1-based, min(m, n) per item. An
/// out-of-range pivot makes that item NaN.
template <class VA, class VF>
double getrf_residual(const VA& A0, const VF& F, const VectorView<std::int32_t>& piv, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VA>>;
    const int m = F.rows(), n = F.cols(), mn = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) detail::bad("getrf_residual", "A0 and F differ in shape");
    if (piv.size() < mn) detail::bad("getrf_residual", "fewer than min(m, n) pivots per item");
    detail::Dense<D> PA(m, n);
    double worst = 0;
    for (int b : detail::pick(items, A0.batch_size())) {
        const auto a = detail::item_of(A0, b);
        const auto f = detail::item_of(F, b);
        const auto p = detail::vec_of(piv, b);
        for (int c = 0; c < n; ++c)
            for (int r = 0; r < m; ++r) PA(r, c) = detail::get(a, r, c);
        bool in_range = true;
        for (int k = 0; k < mn && in_range; ++k) {
            const int ip = p[k] - 1;
            if (ip < 0 || ip >= m) in_range = false;
            else if (ip != k)
                for (int c = 0; c < n; ++c) std::swap(PA(k, c), PA(ip, c));
        }
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                D acc = D(0);
                for (int k = 0; k <= std::min({i, j, mn - 1}); ++k) acc += (k == i ? D(1) : detail::get(f, i, k)) * detail::get(f, k, j);
                num += abs(acc - PA(i, j)) * abs(acc - PA(i, j));
                den += abs(PA(i, j)) * abs(PA(i, j));
            }
        worst = nanmax(worst, in_range ? detail::quot(std::sqrt(num), std::sqrt(den)) : detail::kNaN);
    }
    return worst;
}

/// ‖A0 − QR‖_F / ‖A0‖_F; F in LAPACK geqrf storage (R on and above the diagonal, reflector k below
/// it with an implicit unit), min(m, n) reflectors, H_k = I − tau_k v_k v_kᴴ.
template <class VA, class VF, class T>
double qr_residual(const VA& A0, const VF& F, const VectorView<T>& tau, std::span<const int> items = {}) {
    using D = promoted_t<T>;
    const int m = F.rows(), n = F.cols(), k = std::min(m, n);
    if (A0.rows() != m || A0.cols() != n) detail::bad("qr_residual", "A0 and F differ in shape");
    if (tau.size() < k) detail::bad("qr_residual", "fewer than min(m, n) tau entries per item");
    std::vector<D> x(static_cast<std::size_t>(m));
    double worst = 0;
    for (int b : detail::pick(items, A0.batch_size())) {
        const auto a = detail::item_of(A0, b);
        const auto f = detail::item_of(F, b);
        const auto t = detail::vec_of(tau, b);
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < m; ++r) x[static_cast<std::size_t>(r)] = r <= j && r < k ? detail::get(f, r, j) : D(0);
            for (int i = k - 1; i >= 0; --i) {
                D s = x[static_cast<std::size_t>(i)];
                for (int r = i + 1; r < m; ++r) s += conj(detail::get(f, r, i)) * x[static_cast<std::size_t>(r)];
                s *= up(t[i]);
                x[static_cast<std::size_t>(i)] -= s;
                for (int r = i + 1; r < m; ++r) x[static_cast<std::size_t>(r)] -= s * detail::get(f, r, i);
            }
            for (int r = 0; r < m; ++r) {
                const D z = detail::get(a, r, j);
                num += abs(x[static_cast<std::size_t>(r)] - z) * abs(x[static_cast<std::size_t>(r)] - z);
                den += abs(z) * abs(z);
            }
        }
        worst = nanmax(worst, detail::quot(std::sqrt(num), std::sqrt(den)));
    }
    return worst;
}

/// ‖A0X − B0‖_F / (‖A0‖_F‖X‖_F); an empty B0 (rows() == 0) is the identity (getri). Above 2^28
/// multiply-adds of A0·X per item, 8 seeded probe columns of X stand in for all of them.
template <class VA, class VX, class VB>
double solve_residual(const VA& A0, const VX& X, const VB& B0, std::span<const int> items = {}) {
    return detail::solve_residual(A0, X, B0, items, double(1 << 28));
}

/// Componentwise max |C − (α op(A) op(B) + βC0)| / (|α||op(A)||op(B)| + |β||C0|) over the elements
/// @p sc selects; A and B are read as their Shape says. A zero denominator uses the numerator.
template <class VA, class VB, class VC0, class VC>
double gemm_backward_error(const VA& A, Shape sa, Transpose ta, const VB& B, Shape sb, Transpose tb, const VC0& C0, const VC& C, Shape sc,
                           promoted_t<detail::value_of_t<VC>> alpha, promoted_t<detail::value_of_t<VC>> beta, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VC>>;
    double worst = 0;
    for (int b : detail::pick(items, C.batch_size())) {
        const auto opA = detail::apply_op(detail::shaped_dense(detail::item_of(A, b), sa), ta);
        const auto opB = detail::apply_op(detail::shaped_dense(detail::item_of(B, b), sb), tb);
        const auto c = detail::item_of(C, b);
        if (opA.rows != c.rows || opB.cols != c.cols || opA.cols != opB.rows) detail::bad("gemm_backward_error", "dimension mismatch");
        detail::Item<detail::value_of_t<VC0>> c0{};
        if (beta != D(0)) {
            c0 = detail::item_of(C0, b);
            if (c0.rows != c.rows || c0.cols != c.cols) detail::bad("gemm_backward_error", "C0 and C differ in shape");
        }
        worst = nanmax(worst, detail::componentwise(
            opA, opB, [&](int i, int j) { return detail::get(c0, i, j); }, [&](int i, int j) { return detail::get(c, i, j); },
            c.rows, c.cols, sc, alpha, beta));
    }
    return worst;
}

/// Componentwise backward error of y = α op(A) x + β y0, as gemm_backward_error.
template <class VA, class T>
double gemv_backward_error(const VA& A, Transpose ta, const VectorView<T>& x, const VectorView<T>& y0, const VectorView<T>& y,
                           promoted_t<T> alpha, promoted_t<T> beta, std::span<const int> items = {}) {
    using D = promoted_t<T>;
    double worst = 0;
    for (int b : detail::pick(items, y.batch_size())) {
        const auto opA = detail::apply_op(detail::shaped_dense(detail::item_of(A, b), Shape::general), ta);
        const auto xv = detail::vec_of(x, b);
        const auto yv = detail::vec_of(y, b);
        if (xv.size != opA.cols || yv.size != opA.rows) detail::bad("gemv_backward_error", "dimension mismatch");
        detail::Dense<D> X(xv.size, 1);
        for (int i = 0; i < xv.size; ++i) X(i, 0) = up(xv[i]);
        detail::VecItem<T> y0v{};
        if (beta != D(0)) {
            y0v = detail::vec_of(y0, b);
            if (y0v.size != yv.size) detail::bad("gemv_backward_error", "y0 and y differ in size");
        }
        worst = nanmax(worst, detail::componentwise(
            opA, X, [&](int i, int) { return up(y0v[i]); }, [&](int i, int) { return up(yv[i]); }, yv.size, 1, Shape::general, alpha, beta));
    }
    return worst;
}

/// ‖op(A)X − αB0‖_F / (‖A‖_F‖X‖_F + |α|‖B0‖_F) (Side::Right: X op(A)); only the @p uplo triangle of
/// A is read, and its diagonal only for Diag::NonUnit.
template <class VA, class VX, class VB>
double trsm_residual(const VA& A, Side side, Uplo uplo, Transpose ta, Diag diag, const VX& X, const VB& B0,
                     promoted_t<detail::value_of_t<VX>> alpha, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VX>>;
    const bool unit = diag == Diag::Unit;
    const Shape s = uplo == Uplo::Lower ? (unit ? Shape::unit_lower : Shape::lower) : (unit ? Shape::unit_upper : Shape::upper);
    const bool left = side == Side::Left;
    double worst = 0;
    for (int b : detail::pick(items, X.batch_size())) {
        const auto ai = detail::item_of(A, b);
        if (ai.rows != ai.cols) detail::bad("trsm_residual", "A must be square");
        const auto opA = detail::apply_op(detail::shaped_dense(ai, s), ta);
        const auto x = detail::item_of(X, b);
        const auto b0 = detail::item_of(B0, b);
        if ((left ? x.rows : x.cols) != opA.rows || b0.rows != x.rows || b0.cols != x.cols) detail::bad("trsm_residual", "dimension mismatch");
        double num = 0;
        for (int j = 0; j < x.cols; ++j)
            for (int i = 0; i < x.rows; ++i) {
                D acc = D(0);
                if (left)
                    for (int l = 0; l < opA.cols; ++l) acc += opA(i, l) * detail::get(x, l, j);
                else
                    for (int l = 0; l < opA.rows; ++l) acc += detail::get(x, i, l) * opA(l, j);
                const double d = abs(acc - alpha * detail::get(b0, i, j));
                num += d * d;
            }
        worst = nanmax(worst, detail::quot(std::sqrt(num), opA.frobenius() * frobenius(X, b) + abs(alpha) * frobenius(B0, b)));
    }
    return worst;
}

/// Componentwise backward error of C = α op(A) op(B) + βC0 with a CSR A; an out-of-range column
/// index makes the item NaN.
template <class T, class VB, class VC0, class VC>
double spmm_backward_error(const MatrixView<T, MatrixFormat::CSR>& A, Transpose ta, const VB& B, Transpose tb, const VC0& C0, const VC& C,
                           promoted_t<T> alpha, promoted_t<T> beta, std::span<const int> items = {}) {
    using D = promoted_t<T>;
    double worst = 0;
    for (int b : detail::pick(items, C.batch_size())) {
        bool bad_index = false;
        const auto opA = detail::apply_op(detail::csr_dense(A, b, bad_index), ta);
        const auto opB = detail::apply_op(detail::shaped_dense(detail::item_of(B, b), Shape::general), tb);
        const auto c = detail::item_of(C, b);
        if (opA.rows != c.rows || opB.cols != c.cols || opA.cols != opB.rows) detail::bad("spmm_backward_error", "dimension mismatch");
        detail::Item<T> c0{};
        if (beta != D(0)) {
            c0 = detail::item_of(C0, b);
            if (c0.rows != c.rows || c0.cols != c.cols) detail::bad("spmm_backward_error", "C0 and C differ in shape");
        }
        const double e = detail::componentwise(
            opA, opB, [&](int i, int j) { return detail::get(c0, i, j); }, [&](int i, int j) { return detail::get(c, i, j); },
            c.rows, c.cols, Shape::general, alpha, beta);
        worst = nanmax(worst, bad_index ? detail::kNaN : e);
    }
    return worst;
}

/// ‖QᴴQ − I‖_F over Q's columns.
template <class VQ>
double orthogonality(const VQ& Q, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VQ>>;
    double worst = 0;
    for (int b : detail::pick(items, Q.batch_size())) {
        const auto q = detail::item_of(Q, b);
        double num = 0;
        for (int c = 0; c < q.cols; ++c)
            for (int a = 0; a < q.cols; ++a) {
                D acc = D(0);
                for (int r = 0; r < q.rows; ++r) acc += conj(detail::get(q, r, a)) * detail::get(q, r, c);
                const double d = abs(acc - D(a == c ? 1 : 0));
                num += d * d;
            }
        worst = nanmax(worst, std::sqrt(num));
    }
    return worst;
}

/// ‖AV − V diag(w)‖_F / ‖A‖_F; A Hermitian, read from its lower triangle and mirrored.
template <class VA, class VV, class R>
double eigen_residual(const VA& A, const VV& V, const VectorView<R>& w, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VV>>;
    double worst = 0;
    for (int b : detail::pick(items, V.batch_size())) {
        const auto full = detail::shaped_dense(detail::item_of(A, b), Shape::hermitian_lower);
        const auto v = detail::item_of(V, b);
        const auto lam = detail::vec_of(w, b);
        if (v.rows != full.rows || lam.size < v.cols) detail::bad("eigen_residual", "dimension mismatch");
        double num = 0;
        for (int j = 0; j < v.cols; ++j)
            for (int i = 0; i < v.rows; ++i) {
                D acc = D(0);
                for (int l = 0; l < full.cols; ++l) acc += full(i, l) * detail::get(v, l, j);
                const double d = abs(acc - detail::get(v, i, j) * double(lam[j]));
                num += d * d;
            }
        worst = nanmax(worst, detail::quot(std::sqrt(num), full.frobenius()));
    }
    return worst;
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
