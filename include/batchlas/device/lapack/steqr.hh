// SPDX-License-Identifier: MIT
#pragma once
// xSTEQR's own steps on a tridiagonal (d, e), e(i) coupling d(i) and d(i+1): split test and
// bulge-chase similarity, serial (Thread) and lane-register (LaneVec) forms. steqr_detail also
// holds forms core does not reproduce bit for bit yet (serial dlanst 'M', LaneVec scal, the
// two-window reversal).

#include <cstddef>
#include <type_traits>

#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>

namespace batchlas::device::inline v2 {

namespace steqr_detail {

// dsteqr's split tolerances: e(i) is negligible when |e(i)|^2 <= eps2 |d(i)| |d(i+1)| + safmin.
template <class R> struct SplitTol {
    R eps2;
    R safmin;
};

// max(|d(i)|, |e(i)|), dlanst 'M', serial on one item. fmax ignores a NaN entry (steqr's rule).
template <class D, class E> BATCHLAS_DLA_INLINE auto lanst_max(const Thread&, const D& d, const E& e) {
    using R = elem_t<D>;
    static_assert(!Arith<R>::is_complex, "steqr_detail::lanst_max: d and e are real");
    const int n = d.size();
    R acc = sycl::fabs(d(n - 1));
    for (int i = 0; i < n - 1; ++i) acc = sycl::fmax(acc, sycl::fmax(sycl::fabs(d(i)), sycl::fabs(e(i))));
    return acc;
}

// x := a x on the window [x.lo, x.hi) only.
template <class Ex, class T> BATCHLAS_DLA_INLINE void scal_lane(const Ex& ex, const T& a, const LaneVec<T>& x) {
    if (x.owns(ex)) x.v *= a;
}

// e(j) := 0 for each negligible j < e.size(), serial; true when one was (or already was) zero.
template <class D, class E, class R> BATCHLAS_DLA_INLINE bool tridiag_split(const Thread& th, const D& d, const E& e, SplitTol<R> tol) {
    bool any = false;
    for_each_owned(th, e.size(), [&](int j) {
        if (sycl::fabs(e(j)) * sycl::fabs(e(j)) <= tol.eps2 * sycl::fabs(d(j)) * sycl::fabs(d(j + 1)) + tol.safmin) {
            e(j) = R(0);
            any = true;
        }
    });
    return any;
}

// Lane-register form: d is this item's d(lid); e's window must end before the last valid d
// (d(lid + 1) is read from the next item). An e already zero is left untouched.
template <class Ex, class T> BATCHLAS_DLA_INLINE void tridiag_split(const Ex& ex, const T& d, const LaneVec<T>& e, SplitTol<T> tol) {
    const T d_next = ex.shift(d, 1);  // every item shuffles; the last one's value is unused
    if (e.owns(ex)) {
        if (e.v != T(0)) {
            const T rhs = tol.eps2 * sycl::fabs(d) * sycl::fabs(d_next) + tol.safmin;
            if (sycl::fabs(e.v) * sycl::fabs(e.v) <= rhs) e.v = T(0);
        }
    }
}

// Both windows reversed when `on`: the two sources first, then the two shuffles, so the
// shuffles share one convergence check (the T := J T J of a tridiagonal block, d and e).
template <class Ex, class T, class U> BATCHLAS_DLA_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, const LaneVec<U>& y, bool on) {
    const int l = ex.lid();
    const int src_x = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    const int src_y = (on && l >= y.lo && l < y.hi) ? (y.lo + y.hi - 1 - l) : l;
    x.v = ex.shuffle(static_cast<std::remove_cv_t<T>>(x.v), src_x);
    y.v = ex.shuffle(static_cast<std::remove_cv_t<U>>(y.v), src_y);
}

// Memory form, serial: x(lo..hi-1) := x(hi-1..lo).
template <class X> BATCHLAS_DLA_INLINE void reverse(const Thread&, int lo, int hi, const X& x) {
    for (int a = lo, b = hi - 1; a < b; ++a, --b) {
        const auto t = x(a);
        x(a) = x(b);
        x(b) = t;
    }
}

// v(idx) in a sweep's virtual indexing: v(n - Off - idx) when !forward (QL). Plain inline: an
// always_inline form moved the Steqr*Francis SASS; kernels call this, not a local lambda.
template <class T, int Off, class V, class N> T virt_get(const V& v, bool forward, N n, std::size_t idx) {
    return forward ? v(idx) : v(n - Off - idx);
}

// One bulge-chase step on (d, e): T := G^T T G on rows/cols (i, i+1), prev_bulge the fill at
// (i-1, i+1); returns the fill at (i, i+2). forward == false indexes bottom-up (QL).
template <class D, class E, class T, class I>
BATCHLAS_DLA_INLINE T tridiag_rot(const Thread&, const D& d, const E& e, const T& prev_bulge, I i, I j, Givens<T> g, bool forward) {
    const T c = g.c;
    const T s = g.s;
    const I n = d.size();
    const I ne = e.size();
    auto d_get = [&](I k) -> T { return forward ? d(k) : d(n - 1 - k); };
    auto d_set = [&](I k, T val) {
        if (forward) d(k) = val;
        else d(n - 1 - k) = val;
    };
    auto e_get = [&](I k) -> T { return forward ? e(k) : e(ne - 1 - k); };
    auto e_set = [&](I k, T val) {
        if (forward) e(k) = val;
        else e(ne - 1 - k) = val;
    };

    const T di = d_get(i);
    const T dj = d_get(j);
    const T ei = e_get(i);
    const T ej = (j < ne) ? e_get(j) : T(0);

    // Explicit fma in the shape -ffp-contract=on gives (left product fused), so the x86 SIMD
    // paths cannot contract differently from NVPTX: c(c di - ei s) - s(ei c - s dj), etc.
    d_set(i, sycl::fma(c, sycl::fma(c, di, -(ei * s)), -(s * sycl::fma(ei, c, -(s * dj)))));
    d_set(j, sycl::fma(c, sycl::fma(c, dj, ei * s), s * sycl::fma(ei, c, s * di)));
    if (i > 0) e_set(i - 1, sycl::fma(e_get(i - 1), c, -(prev_bulge * s)));
    e_set(i, sycl::fma(c, sycl::fma(c, ei, s * di), -(s * sycl::fma(c, dj, s * ei))));
    if (j < ne) e_set(j, c * ej);
    return -ej * s;
}

}  // namespace steqr_detail

}  // namespace batchlas::device
