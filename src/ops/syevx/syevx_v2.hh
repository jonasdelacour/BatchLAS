// SPDX-License-Identifier: MIT
#pragma once
// syevx-private device LA v2 ops (no second family uses them); shared ops come from v2 core.

#include <batchlas/device.hh>

namespace batchlas::device::inline v2::syevx {

// x(rev ? n-1-i : i), the direction chosen at run time.
template <class V> struct Flipped {
    V v;
    bool rev;
    using value_type = elem_t<V>;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i) const { return v(rev ? v.size() - 1 - i : i); }
    BATCHLAS_DLA_INLINE int size() const { return v.size(); }
};
template <class V> BATCHLAS_DLA_INLINE Flipped<V> flipped(V v, bool rev) { return {v, rev}; }

// A(i, rev ? n-1-j : j): the column order reversed at run time.
template <class M> struct ColsFlipped {
    M a;
    bool rev;
    using value_type = typename M::value_type;
    BATCHLAS_DLA_INLINE decltype(auto) operator()(int i, int j) const { return a(i, rev ? a.cols() - 1 - j : j); }
    BATCHLAS_DLA_INLINE int rows() const { return a.rows(); }
    BATCHLAS_DLA_INLINE int cols() const { return a.cols(); }
};
template <class M> BATCHLAS_DLA_INLINE ColsFlipped<M> cols_flipped(M a, bool rev) { return {a, rev}; }

// a - s x for a real s, one contractible expression per component (no Annex-G cross terms).
template <class T> BATCHLAS_DLA_INLINE T sub_scaled(T a, real_t<T> s, T x) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a) - s * A::re(x), A::im(a) - s * A::im(x));
    else return a - s * x;
}

// #{i : x(i) <= v} for an ascending x (serial upper_bound; NaN v counts nothing). The `<=` is
// stebz's Sturm count, which makes Direct and DirectSubset agree on m.
// evidence: docs/design/syevx.md#syevx-range-direct-and-directsubset-may-disagree-on-m-by-one
template <class X, class R> BATCHLAS_DLA_INLINE int count_le(const X& x, R v) {
    int lo = 0, hi = x.size();
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (x(mid) <= v) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

}  // namespace batchlas::device::syevx
