// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 extension, tridiagonal eigen family (stebz, stein, the tridiagonal QR solver,
// syev_align's Procrustes step). Every op keeps the legacy kernels' expression shape, so a
// port onto it is bit-identical for real types:
//   DynWorkGroup        executor for launches whose work-group size is a host value
//   Refs<T, N>          N scattered scalars used as one vector (a stripe of a band matrix)
//   fill laset rscal_div sumsq amax_fmax            level-1 / elementwise extras
//   gram2 rot_phase                                  one-sided Jacobi pair (Gram entries, rotation)
//   house2 refl2                                     2-element Householder reflector, generate / apply
//   gershgorin lanst_inf                             symmetric tridiagonal reductions
//   sturm_count bisect_eig                           dstebz / dlaebz counting and bisection
//   gttrf gttrs                                      dgttrf / dgttrs (one right-hand side)

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::device::v2 {

// ---- executor ---------------------------------------------------------------------------
// A whole work-group whose size the host picks per call (min(256, k) and the like). Loops
// stride by the runtime local range and barriers are sycl::group_barrier, exactly as in the
// legacy kernels: the strict-port vehicle for them. A kernel whose size can be a type uses
// WorkGroup<WG, SG> (D1). There is no Scope::Sub here: sub-groups may be partial.
struct DynWorkGroup {
    static constexpr bool sub_whole = false;
    static constexpr Scope natural_scope = Scope::Group;
    sycl::nd_item<1> it;

    BATCHLAS_DLA_INLINE int lid() const { return static_cast<int>(it.get_local_linear_id()); }
    BATCHLAS_DLA_INLINE int range() const { return static_cast<int>(it.get_local_range(0)); }
    BATCHLAS_DLA_INLINE bool leader() const { return lid() == 0; }
    template <class T, class Op = sycl::plus<T>> BATCHLAS_DLA_INLINE T group_reduce(T v, Op op = {}) const {
        return detail::sycl_reduce(it.get_group(), v, op);
    }
    BATCHLAS_DLA_INLINE void barrier() const { sycl::group_barrier(it.get_group()); }
};

// Found by ADL from every v2 op (more specialised than the static-size primitive).
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const DynWorkGroup& ex, int n, F&& f) {
    const int s = ex.range();
    for (int i = ex.lid(); i < n; i += s) f(i);
}

// ---- operands ---------------------------------------------------------------------------
template <class T, int N> struct Refs {
    using value_type = std::remove_cv_t<T>;
    T* p[N];
    BATCHLAS_DLA_INLINE T& operator()(int i) const { return *p[i]; }
    BATCHLAS_DLA_INLINE constexpr int size() const { return N; }
};
template <class T, class... U> BATCHLAS_DLA_INLINE Refs<T, 1 + int(sizeof...(U))> refs(T& a, U&... b) {
    static_assert((std::is_same_v<T, U> && ...), "v2::refs: every element has one type");
    return {{&a, &b...}};
}

namespace detail {
template <class T, class S> BATCHLAS_DLA_INLINE T tse_tag_value(S s) {
    if constexpr (std::is_same_v<S, Zero>) return Arith<T>::zero();
    else if constexpr (std::is_same_v<S, One>) return Arith<T>::one();
    else if constexpr (std::is_same_v<S, NegOne>) return Arith<T>::neg(Arith<T>::one());
    else return static_cast<T>(s.v);
}
// |x|^2 as its own expression: a sum `s += abs2(x)` is FMUL + FADD, never one FMA.
template <class T> BATCHLAS_DLA_INLINE real_t<T> tse_abs2(const T& x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::re(x) * A::re(x) + A::im(x) * A::im(x);
    else return x * x;
}
}  // namespace detail

// ---- level 1 / elementwise ----------------------------------------------------------------
// x := alpha (every element).
template <class Ex, class AS, class X> BATCHLAS_DLA_INLINE void fill(const Ex& ex, AS alpha_, const X& x) {
    using T = elem_t<X>;
    const auto alpha = as_tag(alpha_);
    for_each_owned(ex, x.size(), [&](int i) { x(i) = detail::tse_tag_value<T>(alpha); });
}

// A := alpha off the diagonal, beta on it (dlaset 'A'). Columns outer, rows Cyclic: the item
// that owns row r writes all of it, so an op on the same rows that follows needs no barrier.
template <class Ex, class AS, class BS, class MA> BATCHLAS_DLA_INLINE void laset(const Ex& ex, AS alpha_, BS beta_, const MA& a) {
    using T = typename MA::value_type;
    const auto alpha = as_tag(alpha_);
    const auto beta = as_tag(beta_);
    const int m = a.rows(), n = a.cols();
    for (int c = 0; c < n; ++c)
        for_each_owned(ex, m, [&](int r) { a(r, c) = r == c ? detail::tse_tag_value<T>(beta) : detail::tse_tag_value<T>(alpha); });
}

// x := x / s for a real s, by division (a reciprocal multiply rounds differently).
template <class Ex, class R, class X> BATCHLAS_DLA_INLINE void rscal_div(const Ex& ex, R s, const X& x) {
    using T = elem_t<X>;
    using A = Arith<T>;
    for_each_owned(ex, x.size(), [&](int i) {
        const T v = x(i);
        if constexpr (A::is_complex) x(i) = A::make(A::re(v) / s, A::im(v) / s);
        else x(i) = v / s;
    });
}

// sum_i |x_i|^2 accumulated as s += |x_i|^2 (FMUL + FADD; nrm2<alg::Unscaled> fuses to FMA).
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto sumsq(const Ex& ex, const X& x) {
    using R = real_t<elem_t<X>>;
    R s = R(0);
    for_each_owned(ex, x.size(), [&](int i) { s += detail::tse_abs2(static_cast<elem_t<X>>(x(i))); });
    return detail::scope_reduce<S>(ex, s);
}

// max_i |x_i| (real x) folded with sycl::max: the rescaling idiom of the legacy kernels, with
// sycl::max's NaN behaviour instead of amax's NaN::Propagate (one compare, no isnan).
template <Scope S = Scope::Auto, class Ex, class X> BATCHLAS_DLA_INLINE auto amax_fmax(const Ex& ex, const X& x) {
    using R = elem_t<X>;
    static_assert(!Arith<R>::is_complex, "v2::amax_fmax: real operands only (amax keys complex by |re| + |im|)");
    R m = R(0);
    for_each_owned(ex, x.size(), [&](int i) { m = sycl::max(m, sycl::fabs(x(i))); });
    return detail::scope_reduce<S>(ex, m, sycl::maximum<R>());
}

// ---- one-sided Jacobi pair ------------------------------------------------------------------
template <class R, class T> struct Gram2 {
    R xx;  // ||x||^2
    R yy;  // ||y||^2
    T xy;  // x^H y
};
// The 2x2 Gram matrix of a column pair in one pass (one-sided Jacobi SVD / eigensolvers).
template <Scope S = Scope::Auto, class Ex, class X, class Y> BATCHLAS_DLA_INLINE auto gram2(const Ex& ex, const X& x, const Y& y) {
    using T = elem_t<X>;
    using A = Arith<T>;
    using R = real_t<T>;
    R a = R(0), b = R(0);
    T g = A::zero();
    for_each_owned(ex, x.size(), [&](int i) {
        const T xi = x(i), yi = y(i);
        a += detail::tse_abs2(xi);
        b += detail::tse_abs2(yi);
        A::mac_conj(g, xi, yi);
    });
    return Gram2<R, T>{detail::scope_reduce<S>(ex, a), detail::scope_reduce<S>(ex, b), detail::scope_reduce<S>(ex, g)};
}

// [x, y] := [x, ph y] [[c, s], [-s, c]] for real c, s: the Jacobi rotation after the phase ph
// that makes x^H (ph y) real (ph = 1 or -1 for real data).
template <class Ex, class R, class T, class X, class Y>
BATCHLAS_DLA_INLINE void rot_phase(const Ex& ex, R c, R s, T ph, const X& x, const Y& y) {
    using A = Arith<T>;
    for_each_owned(ex, x.size(), [&](int i) {
        const T xi = x(i), yi = A::mul(static_cast<T>(y(i)), ph);
        x(i) = c * xi - s * yi;
        y(i) = s * xi + c * yi;
    });
}

// ---- 2-element Householder reflector (tau = 2, unit-norm v) --------------------------------
// v with (I - 2 v v^T) (a0, a1)^T = (-sign(a0) ||a||, 0)^T. Unscaled norms (the QR shift step's
// inputs are O(||T||)).
template <class R> BATCHLAS_DLA_INLINE std::array<R, 2> house2(R a0, R a1) {
    const R anorm = sycl::sqrt(a0 * a0 + a1 * a1);
    const R alpha = -sycl::copysign(anorm, a0);
    const R v0 = a0 - alpha, v1 = a1;
    const R vnorm = sycl::sqrt(v0 * v0 + v1 * v1);
    const R inv = R(1) / vnorm;
    return {v0 * inv, v1 * inv};
}

namespace detail {
template <class R, class T> BATCHLAS_DLA_INLINE void tse_refl2_one(R v0, R v1, T& x, T& y) {
    const T t = x * v0 + y * v1;
    x -= R(2) * v0 * t;
    y -= R(2) * v1 * t;
}
}  // namespace detail

// [x_i, y_i] := [x_i, y_i] (I - 2 v v^T) for every i: H from the right on a column pair, or
// from the left on a row pair. v is taken by value (read once, not per element).
template <class Ex, class R, class X, class Y>
BATCHLAS_DLA_INLINE void refl2(const Ex& ex, const std::array<R, 2>& v, const X& x, const Y& y) {
    const R v0 = v[0], v1 = v[1];
    for_each_owned(ex, x.size(), [&](int i) { detail::tse_refl2_one(v0, v1, x(i), y(i)); });
}
// Scattered stripe: unrolled at compile time, so the pointer array never reaches memory.
template <class R, class T, int N>
BATCHLAS_DLA_INLINE void refl2(const Thread&, const std::array<R, 2>& v, const Refs<T, N>& x, const Refs<T, N>& y) {
    const R v0 = v[0], v1 = v[1];
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (detail::tse_refl2_one(v0, v1, *x.p[I], *y.p[I]), ...);
    }(std::make_integer_sequence<int, N>{});
}

// ---- symmetric tridiagonal (d, e): d has n entries, e(i) couples rows i and i+1 ------------
template <class R> struct Interval {
    R lo, hi;
    R e2max;  // max_i e_i^2 (dstebz's pivmin scale)
};
// Gershgorin interval [min_i d_i - r_i, max_i d_i + r_i], r_i = |e_{i-1}| + |e_i|, in three
// reductions (min, max, max) in that order.
template <Scope S = Scope::Auto, class Ex, class D, class E> BATCHLAS_DLA_INLINE auto gershgorin(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    const int n = d.size();
    R lo = std::numeric_limits<R>::max(), hi = -std::numeric_limits<R>::max(), e2 = R(0);
    for_each_owned(ex, n, [&](int i) {
        const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
        const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
        const R radius = left + right;
        lo = sycl::min(lo, d(i) - radius);
        hi = sycl::max(hi, d(i) + radius);
        e2 = sycl::max(e2, right * right);
    });
    const R glo = detail::scope_reduce<S>(ex, lo, sycl::minimum<R>());
    const R ghi = detail::scope_reduce<S>(ex, hi, sycl::maximum<R>());
    const R ge2 = detail::scope_reduce<S>(ex, e2, sycl::maximum<R>());
    return Interval<R>{glo, ghi, ge2};
}

// ||T||_inf = max_i |d_i| + |e_{i-1}| + |e_i| (dlanst 'I').
template <Scope S = Scope::Auto, class Ex, class D, class E> BATCHLAS_DLA_INLINE auto lanst_inf(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    const int n = d.size();
    R m = R(0);
    for_each_owned(ex, n, [&](int i) {
        const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
        const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
        m = sycl::max(m, sycl::fabs(d(i)) + left + right);
    });
    return detail::scope_reduce<S>(ex, m, sycl::maximum<R>());
}

// Number of eigenvalues <= x: sign count of the Sturm sequence q_i = d_i - x - e_{i-1}^2 / q_{i-1}
// with |q| < pivmin clamped to -pivmin (dlaebz), which keeps the count monotone in x.
template <class D, class E, class R> BATCHLAS_DLA_INLINE int sturm_count(const Thread&, const D& d, const E& e, R x, R pivmin) {
    const int n = d.size();
    int count = 0;
    R q = d(0) - x;
    if (sycl::fabs(q) < pivmin) q = -pivmin;
    if (q <= R(0)) ++count;
    for (int i = 1; i < n; ++i) {
        const R ei = e(i - 1);
        q = d(i) - x - (ei * ei) / q;
        if (sycl::fabs(q) < pivmin) q = -pivmin;
        if (q <= R(0)) ++count;
    }
    return count;
}

// Eigenvalue j (0-based, ascending) by bisection of (lo, hi] with count(lo) <= j < count(hi):
// stops at width <= tol, a midpoint that no longer moves, or max_iter halvings; returns the
// final midpoint.
template <class D, class E, class R>
BATCHLAS_DLA_INLINE R bisect_eig(const Thread& th, const D& d, const E& e, int j, R lo, R hi, R tol, R pivmin, int max_iter) {
    R left = lo, right = hi;
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((right - left) <= tol) break;
        const R mid = left + (right - left) * R(0.5);
        if (mid <= left || mid >= right) break;
        if (sturm_count(th, d, e, mid, pivmin) >= j + 1) right = mid;
        else left = mid;
    }
    return left + (right - left) * R(0.5);
}

// ---- tridiagonal LU (general tridiagonal: dl sub-, d main, du super-diagonal) ---------------
// dgttrf with a pivot floor: a pivot below tol becomes +-tol (inverse iteration factors a
// deliberately near-singular matrix). On exit dl holds the multipliers, d / du / du2 the three
// diagonals of U and piv(i) = 1 where rows i and i+1 were interchanged.
template <class DL, class DD, class DU, class DU2, class P, class R>
BATCHLAS_DLA_INLINE void gttrf(const Thread&, const DL& dl, const DD& dd, const DU& du, const DU2& du2, const P& piv, R tol) {
    using T = elem_t<DD>;
    const int n = dd.size();
    for (int i = 0; i < n - 1; ++i) {
        if (sycl::fabs(dd(i)) >= sycl::fabs(dl(i))) {
            piv(i) = 0;
            if (sycl::fabs(dd(i)) < tol) dd(i) = (dd(i) < T(0)) ? -tol : tol;
            const T fact = dl(i) / dd(i);
            dl(i) = fact;
            dd(i + 1) -= fact * du(i);
            if (i < n - 2) du2(i) = T(0);
        } else {
            piv(i) = 1;
            const T fact = dd(i) / dl(i);
            dd(i) = dl(i);
            dl(i) = fact;
            const T tmp = du(i);
            du(i) = dd(i + 1);
            dd(i + 1) = tmp - fact * dd(i + 1);
            if (i < n - 2) {
                du2(i) = du(i + 1);
                du(i + 1) = -fact * du(i + 1);
            }
        }
    }
    if (sycl::fabs(dd(n - 1)) < tol) dd(n - 1) = (dd(n - 1) < T(0)) ? -tol : tol;
}

// b := (L U)^{-1} b from gttrf's factors (dgttrs, no transpose, one right-hand side).
template <class DL, class DD, class DU, class DU2, class P, class B>
BATCHLAS_DLA_INLINE void gttrs(const Thread&, const DL& dl, const DD& dd, const DU& du, const DU2& du2, const P& piv, const B& b) {
    using T = elem_t<B>;
    const int n = dd.size();
    for (int i = 0; i < n - 1; ++i) {
        if (!piv(i)) {
            b(i + 1) -= dl(i) * b(i);
        } else {
            const T tmp = b(i);
            b(i) = b(i + 1);
            b(i + 1) = tmp - dl(i) * b(i);
        }
    }
    b(n - 1) = b(n - 1) / dd(n - 1);
    if (n > 1) b(n - 2) = (b(n - 2) - du(n - 2) * b(n - 1)) / dd(n - 2);
    for (int i = n - 3; i >= 0; --i) b(i) = (b(i) - du(i) * b(i + 1) - du2(i) * b(i + 2)) / dd(i);
}

}  // namespace batchlas::device::v2
