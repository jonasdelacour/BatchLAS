// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to the tridiagonal eigen kernels (stebz, stein, tridiagonal_solver).
// Each keeps the legacy kernel's expression shape, so the ports stay bit-identical:
//   Refs<T, N> / refs     N scattered scalars used as one vector (a stripe of a band matrix)
//   house2 refl2          2-element Householder reflector (unit v, tau = 2), generate / apply
//   gershgorin lanst_inf  symmetric tridiagonal reductions
//   sturm_count bisect_eig  dstebz / dlaebz counting and bisection
// Executors, loops, reductions and level-1 ops come from core (batchlas/device).

#include <array>
#include <limits>
#include <type_traits>
#include <utility>

#include <batchlas/device.hh>

namespace batchlas::device::inline v2::tridiag {

template <class T, int N> struct Refs {
    using value_type = std::remove_cv_t<T>;
    T* p[N];
    BATCHLAS_DLA_INLINE T& operator()(int i) const { return *p[i]; }
    BATCHLAS_DLA_INLINE constexpr int size() const { return N; }
};
template <class T, class... U> BATCHLAS_DLA_INLINE Refs<T, 1 + int(sizeof...(U))> refs(T& a, U&... b) {
    static_assert((std::is_same_v<T, U> && ...), "tridiag::refs: every element has one type");
    return {{&a, &b...}};
}

// v with (I - 2 v v^T) (a0, a1)^T = (-sign(a0) ||a||, 0)^T; unscaled norms (inputs are O(||T||)).
template <class R> BATCHLAS_DLA_INLINE std::array<R, 2> house2(R a0, R a1) {
    const R anorm = sycl::sqrt(a0 * a0 + a1 * a1);
    const R alpha = -sycl::copysign(anorm, a0);
    const R v0 = a0 - alpha, v1 = a1;
    const R vnorm = sycl::sqrt(v0 * v0 + v1 * v1);
    const R inv = R(1) / vnorm;
    return {v0 * inv, v1 * inv};
}

namespace detail {
template <class R, class T> BATCHLAS_DLA_INLINE void refl2_one(R v0, R v1, T& x, T& y) {
    const T t = x * v0 + y * v1;
    x -= R(2) * v0 * t;
    y -= R(2) * v1 * t;
}
}  // namespace detail

// [x_i, y_i] := [x_i, y_i] (I - 2 v v^T) for every i (v read once). Not core larf: its
// w = 0 + x v0 + y v1 contracts in the other order, so the bits differ.
template <class Ex, class R, class X, class Y>
BATCHLAS_DLA_INLINE void refl2(const Ex& ex, const std::array<R, 2>& v, const X& x, const Y& y) {
    const R v0 = v[0], v1 = v[1];
    for_each_owned(ex, x.size(), [&](int i) { detail::refl2_one(v0, v1, x(i), y(i)); });
}
// Scattered stripe: unrolled at compile time, so the pointer array never reaches memory.
template <class R, class T, int N>
BATCHLAS_DLA_INLINE void refl2(const Thread&, const std::array<R, 2>& v, const Refs<T, N>& x, const Refs<T, N>& y) {
    const R v0 = v[0], v1 = v[1];
    [&]<int... I>(std::integer_sequence<int, I...>) {
        (detail::refl2_one(v0, v1, *x.p[I], *y.p[I]), ...);
    }(std::make_integer_sequence<int, N>{});
}

// ---- symmetric tridiagonal (d, e): d has n entries, e(i) couples rows i and i+1 ------------
template <class R> struct Interval {
    R lo, hi;
    R e2max;  // max_i e_i^2 (dstebz's pivmin scale)
};
// Gershgorin interval [min_i d_i - r_i, max_i d_i + r_i], r_i = |e_{i-1}| + |e_i|: three
// reductions, min then max then max (the legacy order).
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
    const R glo = allreduce<S>(ex, lo, sycl::minimum<R>());
    const R ghi = allreduce<S>(ex, hi, sycl::maximum<R>());
    const R ge2 = allreduce<S>(ex, e2, sycl::maximum<R>());
    return Interval<R>{glo, ghi, ge2};
}

// ||T||_inf = max_i |d_i| + |e_{i-1}| + |e_i| (dlanst 'I'), folded with sycl::max.
template <Scope S = Scope::Auto, class Ex, class D, class E> BATCHLAS_DLA_INLINE auto lanst_inf(const Ex& ex, const D& d, const E& e) {
    using R = elem_t<D>;
    const int n = d.size();
    R m = R(0);
    for_each_owned(ex, n, [&](int i) {
        const R left = (i > 0) ? sycl::fabs(e(i - 1)) : R(0);
        const R right = (i < n - 1) ? sycl::fabs(e(i)) : R(0);
        m = sycl::max(m, sycl::fabs(d(i)) + left + right);
    });
    return allreduce<S>(ex, m, sycl::maximum<R>());
}

// Number of eigenvalues <= x: sign count of q_i = d_i - x - e_{i-1}^2 / q_{i-1}, |q| < pivmin
// clamped to -pivmin (dlaebz), which keeps the count monotone in x.
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

}  // namespace batchlas::device::tridiag
