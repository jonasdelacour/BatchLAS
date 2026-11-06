// SPDX-License-Identifier: MIT
#pragma once
// The gesvd driver's own steps on the real upper bidiagonal B = (d, e) of gebrd (gesvd_detail):
// bdgram forms B^T B or B B^T (the gesvd-via-tridiagonal route), bdmv_div recovers a left
// singular vector B x / sigma, tridiag_abs_offdiag makes a tridiagonal's off-diagonal
// non-negative by a diagonal sign similarity. Thread / Grid1D / WorkGroup executors through core
// for_each_owned.

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>

namespace batchlas::device::inline v2::gesvd_detail {

// The symmetric tridiagonal Gram matrix of the real upper bidiagonal B = (d, e), k = d.size():
// Side::Right is B^T B (order k), Side::Left is B B^T padded with zero rows to order td.size().
template <Side SD, class Ex, class D, class E, class TD, class TE>
BATCHLAS_DLA_INLINE void bdgram(const Ex& ex, const D& d, const E& e, const TD& td, const TE& te) {
    using T = elem_t<TD>;
    static_assert(!Arith<T>::is_complex, "gesvd_detail::bdgram: the bidiagonal of gebrd is real");
    const int k = d.size();
    if constexpr (SD == Side::Right) {
        for_each_owned(ex, td.size(), [&](int i) {
            const T di = d(i);
            T diag = di * di;
            if (i > 0) {
                const T e_prev = e(i - 1);
                diag += e_prev * e_prev;
            }
            td(i) = diag;
            if (i < k - 1) te(i) = di * e(i);
        });
    } else {
        const int order = td.size();
        for_each_owned(ex, order, [&](int i) {
            T diag = T(0);
            if (i < k) {
                const T di = d(i);
                diag += di * di;
                if (i < k - 1) {
                    const T ei = e(i);
                    diag += ei * ei;
                    te(i) = ei * d(i + 1);
                } else if (i < order - 1) {
                    te(i) = T(0);
                }
            } else if (i < order - 1) {
                te(i) = T(0);
            }
            td(i) = diag;
        });
    }
}

// y := B x / sigma for the real upper bidiagonal B = (d, e), k = d.size(), with y(i) = 0 for
// k <= i < y.size(); y := 0 when sigma <= tol (a zero singular value has no vector B x / sigma).
template <class Ex, class D, class E, class X, class R, class Y>
BATCHLAS_DLA_INLINE void bdmv_div(const Ex& ex, const D& d, const E& e, const X& x, R sigma, R tol, const Y& y) {
    using T = elem_t<Y>;
    const int k = d.size();
    for_each_owned(ex, y.size(), [&](int i) {
        T value = T(0);
        if (sigma > tol && i < k) {
            value = d(i) * x(i);
            if (i < k - 1) value += e(i) * x(i + 1);
            value /= sigma;
        }
        y(i) = value;
    });
}

// e := |e| and s := the signs with s(0) = 1 for which diag(s) T diag(s) has off-diagonal |e|
// (T real symmetric tridiagonal of order s.size()). A prefix product: Thread only.
template <class Ex, class E, class S> BATCHLAS_DLA_INLINE void tridiag_abs_offdiag(const Ex&, const E& e, const S& s) {
    static_assert(is_thread_v<Ex>, "gesvd_detail::tridiag_abs_offdiag: a sequential scan, Thread only");
    using T = elem_t<S>;
    const int n = s.size();
    s(0) = T(1);
    for (int i = 0; i < n - 1; ++i) {
        const T ei = e(i);
        const T sgn = (ei >= T(0)) ? T(1) : T(-1);
        s(i + 1) = s(i) * sgn;
        e(i) = sycl::fabs(ei);
    }
}

}  // namespace batchlas::device::gesvd_detail
