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
    // op(A) transposed and both moduli formed once: the l loop then reads contiguous memory and
    // takes no modulus (the same terms in the same order, so the same value).
    const int kk = opA.cols;
    std::vector<D> at(static_cast<std::size_t>(kk) * static_cast<std::size_t>(opA.rows));
    std::vector<double> abs_at(at.size()), abs_b(opB.a.size());
    for (int i = 0; i < opA.rows; ++i)
        for (int l = 0; l < kk; ++l) {
            at[static_cast<std::size_t>(i) * kk + l] = opA(i, l);
            abs_at[static_cast<std::size_t>(i) * kk + l] = abs(opA(i, l));
        }
    for (std::size_t x = 0; x < opB.a.size(); ++x) abs_b[x] = abs(opB.a[x]);
    double worst = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            if ((lower_family(sc) && i < j) || (upper_family(sc) && i > j)) continue;
            const D* a_row = at.data() + static_cast<std::size_t>(i) * kk;
            const double* abs_a_row = abs_at.data() + static_cast<std::size_t>(i) * kk;
            const D* b_col = opB.a.data() + static_cast<std::size_t>(j) * opB.rows;
            const double* abs_b_col = abs_b.data() + static_cast<std::size_t>(j) * opB.rows;
            D acc = D(0);
            double mag = 0;
            for (int l = 0; l < kk; ++l) {
                acc += a_row[l] * b_col[l];
                mag += abs_a_row[l] * abs_b_col[l];
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

// Q = H_0 ... H_{k-1} (reflector i below f's diagonal, implicit unit, H_i = I - tau_i v_i v_i^H)
// applied to x in place.
template <class T, class E, class D>
void apply_q(const Item<T>& f, const VecItem<E>& t, int k, std::vector<D>& x) {
    const int m = f.rows;
    for (int i = k - 1; i >= 0; --i) {
        D s = x[static_cast<std::size_t>(i)];
        for (int r = i + 1; r < m; ++r) s += conj(get(f, r, i)) * x[static_cast<std::size_t>(r)];
        s *= up(t[i]);
        x[static_cast<std::size_t>(i)] -= s;
        for (int r = i + 1; r < m; ++r) x[static_cast<std::size_t>(r)] -= s * get(f, r, i);
    }
}

template <class T> promoted_t<T> op_shaped(const Item<T>& m, Shape s, Transpose t, int i, int j) {
    if (t == Transpose::NoTrans) return shaped(m, s, i, j);
    const auto x = shaped(m, s, j, i);
    return t == Transpose::ConjTrans ? conj(x) : x;
}

// ||op(A) X V - B0 V||_F / (||op(A)||_F ||X V||_F); op_of(b) returns item b's op(A)(r, l) (m x k).
// V is the identity while m*k*ncols is at most full_limit, else 8 seeded dense probe columns: O(n^2)
// host work at n = 4096, and a dense probe still sees any wrong entry of X.
template <class T, class OpOf, class VX, class VB>
double solve_core(OpOf op_of, int m, int k, int batch, const VX& X, const VB& B0, std::span<const int> items, double full_limit) {
    using D = promoted_t<T>;
    const bool identity = B0.rows() == 0;
    const int nc = X.cols();
    if (X.rows() != k || (!identity && (B0.rows() != m || B0.cols() != nc))) bad("solve_residual", "dimension mismatch");
    const bool full = double(m) * double(k) * double(nc) <= full_limit;
    const int nv = full ? nc : 8;
    std::vector<D> V(full ? 0 : static_cast<std::size_t>(nc) * static_cast<std::size_t>(nv));
    Rng rng(4242);
    for (D& v : V) v = up(draw<T>(rng));
    auto vat = [&](int c, int j) { return full ? D(c == j ? 1 : 0) : V[static_cast<std::size_t>(j) * static_cast<std::size_t>(nc) + static_cast<std::size_t>(c)]; };
    std::vector<D> xv(static_cast<std::size_t>(k)), bv(static_cast<std::size_t>(m));
    double worst = 0;
    for (int b : pick(items, batch)) {
        const auto a = op_of(b);
        const auto x = item_of(X, b);
        Item<value_of_t<VB>> rhs{};
        if (!identity) rhs = item_of(B0, b);
        double na = 0, nx = 0, num = 0;
        for (int c = 0; c < k; ++c)
            for (int r = 0; r < m; ++r) na += abs(a(r, c)) * abs(a(r, c));
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
                for (int l = 0; l < k; ++l) acc += a(r, l) * xv[static_cast<std::size_t>(l)];
                const double d = abs(acc - bv[static_cast<std::size_t>(r)]);
                num += d * d;
            }
        }
        worst = nanmax(worst, quot(std::sqrt(num), std::sqrt(na) * std::sqrt(nx)));
    }
    return worst;
}

template <class VA, class VX, class VB>
double solve_residual(const VA& A0, Shape sa, Transpose ta, const VX& X, const VB& B0, std::span<const int> items, double full_limit) {
    using T = value_of_t<VA>;
    if (sa != Shape::general && sa != Shape::lower && sa != Shape::upper && A0.rows() != A0.cols())
        bad("solve_residual", "unit, Hermitian and symmetric shapes need a square operand");
    const bool nt = ta == Transpose::NoTrans;
    const auto op_of = [&](int b) {
        const auto a = item_of(A0, b);
        return [a, sa, ta](int r, int l) { return op_shaped(a, sa, ta, r, l); };
    };
    return solve_core<T>(op_of, nt ? A0.rows() : A0.cols(), nt ? A0.cols() : A0.rows(), A0.batch_size(), X, B0, items, full_limit);
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

/// ‖A0 − Q·B·Qᴴ‖_F / ‖A0‖_F for square A0, an explicit Q and a dense B (a tridiagonal or banded
/// reduction stored densely; entries outside the band must be stored zeros). Every element of each
/// operand is read. Judged as a factorization.
template <class VA, class VQ, class VB>
double similarity_residual(const VA& A0, const VQ& Q, const VB& B, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VA>>;
    const int n = A0.rows();
    if (A0.cols() != n || Q.rows() != n || Q.cols() != n || B.rows() != n || B.cols() != n)
        detail::bad("similarity_residual", "square matrices of equal order only");
    detail::Dense<D> QB(n, n);
    double worst = 0;
    for (int b : detail::pick(items, A0.batch_size())) {
        const auto a = detail::item_of(A0, b);
        const auto q = detail::item_of(Q, b);
        const auto bb = detail::item_of(B, b);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int k = 0; k < n; ++k) acc += detail::get(q, i, k) * detail::get(bb, k, j);
                QB(i, j) = acc;
            }
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                D acc = D(0);
                for (int k = 0; k < n; ++k) acc += QB(i, k) * conj(detail::get(q, j, k));
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
            detail::apply_q(f, t, k, x);
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

/// The first @p cols columns of Q = H_0 … H_{k−1} of item @p item (k = min(m, n) reflectors of the
/// m x n @p F in geqrf storage, as qr_residual): m x cols, column-major, ld = m. F's diagonal and
/// upper triangle are not read.
template <class VF, class T>
std::vector<promoted_t<T>> form_q(const VF& F, const VectorView<T>& tau, int item, int cols) {
    using D = promoted_t<T>;
    const int m = F.rows(), k = std::min(m, F.cols());
    if (cols < 0 || cols > m) detail::bad("form_q", "cols must lie in [0, rows]");
    if (tau.size() < k) detail::bad("form_q", "fewer than min(m, n) tau entries per item");
    const auto f = detail::item_of(F, item);
    const auto t = detail::vec_of(tau, item);
    std::vector<D> Q(static_cast<std::size_t>(m) * static_cast<std::size_t>(cols)), x(static_cast<std::size_t>(m));
    for (int j = 0; j < cols; ++j) {
        std::fill(x.begin(), x.end(), D(0));
        x[static_cast<std::size_t>(j)] = D(1);
        detail::apply_q(f, t, k, x);
        std::copy(x.begin(), x.end(), Q.begin() + static_cast<std::ptrdiff_t>(j) * m);
    }
    return Q;
}

/// ‖A0 − Q·triu(R)‖_F / ‖A0‖_F with an explicit m x k @p Q; R's first k rows are read on and above
/// the diagonal only, so a geqrf factor (m x n) and a compact k x n R are both accepted.
template <class VA, class VQ, class VR>
double qr_reconstruction(const VA& A0, const VQ& Q, const VR& R, std::span<const int> items = {}) {
    using D = promoted_t<detail::value_of_t<VA>>;
    const int m = A0.rows(), n = A0.cols(), k = Q.cols();
    if (Q.rows() != m || R.cols() != n || R.rows() < k) detail::bad("qr_reconstruction", "need A0 m x n, Q m x k, R at least k x n");
    double worst = 0;
    for (int b : detail::pick(items, A0.batch_size())) {
        const auto a = detail::item_of(A0, b);
        const auto q = detail::item_of(Q, b);
        const auto r = detail::item_of(R, b);
        double num = 0, den = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                D acc = D(0);
                for (int l = 0; l <= std::min(j, k - 1); ++l) acc += detail::get(q, i, l) * detail::get(r, l, j);
                const D z = detail::get(a, i, j);
                num += abs(acc - z) * abs(acc - z);
                den += abs(z) * abs(z);
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

/// ‖op(A0)X − B0‖_F / (‖A0‖_F‖X‖_F) with A0 read through @p sa (as gemm_backward_error) and @p ta.
template <class VA, class VX, class VB>
double solve_residual(const VA& A0, Shape sa, Transpose ta, const VX& X, const VB& B0, std::span<const int> items = {}) {
    return detail::solve_residual(A0, sa, ta, X, B0, items, double(1 << 28));
}

/// solve_residual for X computed from getrf factors: A = P·L·U is formed on the host per checked
/// item from @p F (unit L, U) and the packed 1-based @p piv, then ‖op(A)X − B0‖_F / (‖A‖_F‖X‖_F). An
/// empty B0 is the identity (getri from factors); an out-of-range pivot makes that item NaN.
template <class VF, class VX, class VB>
double lu_solve_residual(const VF& F, const VectorView<std::int32_t>& piv, Transpose trans, const VX& X, const VB& B0,
                         std::span<const int> items = {}) {
    using T = detail::value_of_t<VF>;
    using D = promoted_t<T>;
    const int n = F.rows();
    if (F.cols() != n) detail::bad("lu_solve_residual", "square factors only");
    if (piv.size() < n) detail::bad("lu_solve_residual", "fewer than n pivots per item");
    const auto op_of = [&](int b) {
        const auto f = detail::item_of(F, b);
        const auto p = detail::vec_of(piv, b);
        detail::Dense<D> M(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                for (int l = 0; l <= std::min(i, j); ++l) M(i, j) += (l == i ? D(1) : detail::get(f, i, l)) * detail::get(f, l, j);
        for (int k = n - 1; k >= 0; --k) {
            const int ip = p[k] - 1;
            if (ip < 0 || ip >= n) {
                std::fill(M.a.begin(), M.a.end(), D(detail::kNaN));
                break;
            }
            for (int c = 0; c < n; ++c) std::swap(M(k, c), M(ip, c));
        }
        return [M = std::move(M), trans](int r, int l) {
            if (trans == Transpose::NoTrans) return M(r, l);
            return trans == Transpose::ConjTrans ? conj(M(l, r)) : M(l, r);
        };
    };
    return detail::solve_core<T>(op_of, n, n, F.batch_size(), X, B0, items, double(1 << 28));
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

/// Largest denominator of gemm_backward_error over item @p item: max_ij |α|(|op(A)||op(B)|)_ij + |β||C0_ij|
/// (C0 not read when β == 0). An absolute tolerance divided by it is a componentwise one.
template <class VA, class VB, class VC0>
double gemm_max_denominator(const VA& A, Shape sa, Transpose ta, const VB& B, Shape sb, Transpose tb, const VC0& C0,
                            promoted_t<detail::value_of_t<VC0>> alpha, promoted_t<detail::value_of_t<VC0>> beta, int item = 0) {
    using D = promoted_t<detail::value_of_t<VC0>>;
    const auto opA = detail::apply_op(detail::shaped_dense(detail::item_of(A, item), sa), ta);
    const auto opB = detail::apply_op(detail::shaped_dense(detail::item_of(B, item), sb), tb);
    if (opA.cols != opB.rows) detail::bad("gemm_max_denominator", "dimension mismatch");
    detail::Item<detail::value_of_t<VC0>> c0{};
    if (beta != D(0)) {
        c0 = detail::item_of(C0, item);
        if (c0.rows != opA.rows || c0.cols != opB.cols) detail::bad("gemm_max_denominator", "C0 and op(A) op(B) differ in shape");
    }
    double worst = 0;
    for (int j = 0; j < opB.cols; ++j)
        for (int i = 0; i < opA.rows; ++i) {
            double mag = 0;
            for (int l = 0; l < opA.cols; ++l) mag += abs(opA(i, l)) * abs(opB(l, j));
            worst = nanmax(worst, abs(alpha) * mag + (beta != D(0) ? abs(beta) * abs(detail::get(c0, i, j)) : 0.0));
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
    const D alpha2 = hermitian ? conj(alpha) : alpha;
    const auto back = [&](const D& x) { return hermitian ? conj(x) : x; };
    double worst = 0;
    for (int b : detail::pick(items, C.batch_size())) {
        const auto opA = detail::apply_op(detail::shaped_dense(detail::item_of(A, b), Shape::general), trans);
        const auto opB = detail::apply_op(detail::shaped_dense(detail::item_of(B, b), Shape::general), trans);
        const auto c = detail::item_of(C, b);
        if (c.rows != c.cols || opA.rows != c.rows || opB.rows != c.rows || opA.cols != opB.cols) detail::bad("rank2k_backward_error", "dimension mismatch");
        const int n = c.rows, k = opA.cols;
        // One product [op(A) op(B)] · [α back(op(B)); α2 back(op(A))] has exactly the rank-2k denominator.
        detail::Dense<D> P(n, 2 * k), Q(2 * k, n);
        for (int l = 0; l < k; ++l)
            for (int i = 0; i < n; ++i) {
                P(i, l) = opA(i, l);
                P(i, k + l) = opB(i, l);
                Q(l, i) = alpha * back(opB(i, l));
                Q(k + l, i) = alpha2 * back(opA(i, l));
            }
        detail::Item<detail::value_of_t<VC0>> c0{};
        if (beta != D(0)) {
            c0 = detail::item_of(C0, b);
            if (c0.rows != n || c0.cols != n) detail::bad("rank2k_backward_error", "C0 and C differ in shape");
        }
        const auto c0_at = [&](int i, int j) {
            const D z = detail::get(c0, i, j);
            return hermitian && i == j ? D(std::real(z)) : z;
        };
        worst = nanmax(worst, detail::componentwise(P, Q, c0_at, [&](int i, int j) { return detail::get(c, i, j); }, n, n,
                                                    uplo == Uplo::Lower ? Shape::lower : Shape::upper, D(1), beta));
    }
    return worst;
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
