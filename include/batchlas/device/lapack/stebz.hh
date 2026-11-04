// SPDX-License-Identifier: MIT
#pragma once
// xSTEBZ / xLAEBZ on a symmetric tridiagonal (d, e): d has n entries, e(i) couples rows i and
// i + 1. Each keeps the legacy stebz kernel's expression shape (bit-identical ports):
//   stebz_gershgorin   the Gershgorin interval and dstebz's pivmin scale (three reductions)
//   laebz_count        the Sturm sign count, number of eigenvalues <= x
//   laebz_bisect       one eigenvalue by bisection on that count

#include <limits>

#include <batchlas/device/exec.hh>
#include <batchlas/device/level1.hh>

namespace batchlas::device::inline v2 {

namespace stebz_detail {
template <class R> struct Interval {
    R lo, hi;
    R e2max;  // max_i e_i^2 (dstebz's pivmin scale)
};
}  // namespace stebz_detail

// [min_i d_i - r_i, max_i d_i + r_i], r_i = |e_{i-1}| + |e_i|: min, then max, then max.
template <Scope S = Scope::Auto, class Ex, class D, class E>
BATCHLAS_DLA_INLINE auto stebz_gershgorin(const Ex& ex, const D& d, const E& e) {
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
    return stebz_detail::Interval<R>{glo, ghi, ge2};
}

// Sign count of q_i = d_i - x - e_{i-1}^2 / q_{i-1}; |q| < pivmin is clamped to -pivmin
// (dlaebz), which keeps the count monotone in x.
template <class D, class E, class R> BATCHLAS_DLA_INLINE int laebz_count(const Thread&, const D& d, const E& e, R x, R pivmin) {
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

// Eigenvalue j (0-based, ascending) in (lo, hi] with count(lo) <= j < count(hi): halves until
// width <= tol, a midpoint that no longer moves, or max_iter; returns the final midpoint.
template <class D, class E, class R>
BATCHLAS_DLA_INLINE R laebz_bisect(const Thread& th, const D& d, const E& e, int j, R lo, R hi, R tol, R pivmin, int max_iter) {
    R left = lo, right = hi;
    for (int iter = 0; iter < max_iter; ++iter) {
        if ((right - left) <= tol) break;
        const R mid = left + (right - left) * R(0.5);
        if (mid <= left || mid >= right) break;
        if (laebz_count(th, d, e, mid, pivmin) >= j + 1) right = mid;
        else left = mid;
    }
    return left + (right - left) * R(0.5);
}

}  // namespace batchlas::device::inline v2
