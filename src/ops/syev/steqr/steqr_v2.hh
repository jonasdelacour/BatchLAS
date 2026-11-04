// SPDX-License-Identifier: MIT
#pragma once
// steqr-private device LA v2 ops; shared ones (Givens, rot<rot_conv::Transposed>, lasr, scal)
// are core's. LaneVec: element i in a register of the item with lid == i (n <= Ex::size).

#include <cstdint>
#include <type_traits>

#include <batchlas/blas/device/v2/dla.hh>
#include <src/kernels/common/dla_v2_partition.hh>

namespace batchlas::steqr_v2 {

namespace v2 = batchlas::device::v2;

struct OnePerItem {};  // layout of LaneVec

// dsteqr's split tolerances: e(i) is negligible when |e(i)|^2 <= eps2 |d(i)| |d(i+1)| + safmin.
template <class R> struct SplitTol {
    R eps2;
    R safmin;
};

// One element per item, held in a register: element i belongs to the item with lid == i.
template <class T> struct LaneVec {
    using value_type = std::remove_cv_t<T>;
    using layout = OnePerItem;
    T& v;
    int lo;
    int hi;

    BATCHLAS_DLA_INLINE T& operator()(int) const { return v; }
    BATCHLAS_DLA_INLINE int size() const { return hi; }
    template <class Ex> BATCHLAS_DLA_INLINE bool owns(const Ex& ex) const {
        const int l = ex.lid();
        return l >= lo && l < hi;
    }
};
// The window [lo, hi) of the lane-register vector whose own element is v.
template <class T> BATCHLAS_DLA_INLINE LaneVec<T> lane_vec(T& v, int lo, int hi) { return {v, lo, hi}; }

namespace detail {

struct FmaxOp {
    template <class R> BATCHLAS_DLA_INLINE R operator()(R a, R b) const { return sycl::fmax(a, b); }
};
struct MinOp {
    BATCHLAS_DLA_INLINE int operator()(int a, int b) const { return (b < a) ? b : a; }
};

// Butterfly with masks 1, 2, .., P/2 (core's Partition::team_reduce runs P/2 .. 1): the
// solvers' order; min and fmax are order-free, the codegen is not.
template <class Ex, class T, class Op> BATCHLAS_DLA_INLINE T butterfly_up(const Ex& ex, T v, Op op) {
#pragma unroll
    for (unsigned m = 1u; m < static_cast<unsigned>(Ex::size); m <<= 1) v = op(v, permute_group_by_xor(ex.part, v, m));
    return v;
}

// The value of item lid + 1 of the chunk (unspecified on its last item).
template <class Ex, class T> BATCHLAS_DLA_INLINE T lane_next(const Ex& ex, T v) {
    return shift_group_left(ex.part, v, 1u);  // ADL: SubGroupPartition (src/kernels/common)
}

// The value of chunk item `src` (a lid, 0 <= src < Ex::size; may differ per item).
template <class Ex, class T> BATCHLAS_DLA_INLINE T lane_select(const Ex& ex, T v, int src) {
    return select_from_group(ex.part, v, static_cast<std::uint32_t>(src));  // ADL
}

}  // namespace detail

// ---- symmetric tridiagonal (d, e), n = d.size() >= 1, e(i) couples d(i) and d(i+1) ----------

// max(|d(i)|, |e(i)|), dlanst 'M', serial on one item. fmax ignores a NaN entry (steqr's rule).
template <class D, class E> BATCHLAS_DLA_INLINE auto lanst_max(const v2::Thread&, const D& d, const E& e) {
    using R = v2::elem_t<D>;
    static_assert(!v2::Arith<R>::is_complex, "steqr_v2::lanst_max: d and e are real");
    const int n = d.size();
    R acc = sycl::fabs(d(n - 1));
    for (int i = 0; i < n - 1; ++i) acc = sycl::fmax(acc, sycl::fmax(sycl::fabs(d(i)), sycl::fabs(e(i))));
    return acc;
}

// Lane-register form over the windows of d and e on a chunk (SgpPartition); uniform result.
template <class Ex, class T> BATCHLAS_DLA_INLINE T lanst_max(const Ex& ex, const LaneVec<T>& d, const LaneVec<T>& e) {
    T acc = T(0);
    if (d.owns(ex)) acc = sycl::fabs(d.v);
    if (e.owns(ex)) acc = sycl::fmax(acc, sycl::fabs(e.v));
    return detail::butterfly_up(ex, acc, detail::FmaxOp{});
}

// (d, e) := a (d, e), serial (dlascl). Named, not a kernel-local lambda (SemaSYCL body quirk).
template <class T, class D, class E> BATCHLAS_DLA_INLINE void scal_tridiag(const v2::Thread& th, const T& a, const D& d, const E& e) {
    v2::scal(th, a, d);
    v2::scal(th, a, e);
}

// x := a x on the window.
template <class Ex, class T> BATCHLAS_DLA_INLINE void scal_lane(const Ex& ex, const T& a, const LaneVec<T>& x) {
    if (x.owns(ex)) x.v *= a;
}

// e(j) := 0 for every j < e.size() that is negligible, serial; true when one was zeroed (an
// e(j) already zero counts). d.size() > e.size().
template <class D, class E, class R> BATCHLAS_DLA_INLINE bool tridiag_split(const v2::Thread& th, const D& d, const E& e, SplitTol<R> tol) {
    bool any = false;
    v2::for_each_owned(th, e.size(), [&](int j) {
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
    const T d_next = detail::lane_next(ex, d);  // every item shuffles; the last one's value is unused
    if (e.owns(ex)) {
        if (e.v != T(0)) {
            const T rhs = tol.eps2 * sycl::fabs(d) * sycl::fabs(d_next) + tol.safmin;
            if (sycl::fabs(e.v) * sycl::fabs(e.v) <= rhs) e.v = T(0);
        }
    }
}

// The smallest i in x's window with pred(x(i)), else `none` (core find_first: -1, memory operand).
template <class Ex, class T, class Pred> BATCHLAS_DLA_INLINE int find_first_lane(const Ex& ex, const LaneVec<T>& x, int none, Pred&& pred) {
    const int l = ex.lid();
    const int cand = (l >= x.lo && l < x.hi && pred(static_cast<std::remove_cv_t<T>>(x.v))) ? l : none;
    return detail::butterfly_up(ex, cand, detail::MinOp{});
}

// Both windows reversed when `on`: the two sources first, then the two shuffles, so the
// shuffles share one convergence check (the T := J T J of a tridiagonal block, d and e).
template <class Ex, class T, class U> BATCHLAS_DLA_INLINE void reverse(const Ex& ex, const LaneVec<T>& x, const LaneVec<U>& y, bool on) {
    const int l = ex.lid();
    const int src_x = (on && l >= x.lo && l < x.hi) ? (x.lo + x.hi - 1 - l) : l;
    const int src_y = (on && l >= y.lo && l < y.hi) ? (y.lo + y.hi - 1 - l) : l;
    x.v = detail::lane_select(ex, static_cast<std::remove_cv_t<T>>(x.v), src_x);
    y.v = detail::lane_select(ex, static_cast<std::remove_cv_t<U>>(y.v), src_y);
}

// Memory form, serial: x(lo..hi-1) := x(hi-1..lo).
template <class X> BATCHLAS_DLA_INLINE void reverse(const v2::Thread&, int lo, int hi, const X& x) {
    for (int a = lo, b = hi - 1; a < b; ++a, --b) {
        const auto t = x(a);
        x(a) = x(b);
        x(b) = t;
    }
}

// v(idx) in a sweep's virtual indexing: v(n - Off - idx) when !forward (QL). Plain inline: an
// always_inline form moved the Steqr*Francis SASS; kernels call this, not a local lambda.
template <class T, int Off, class V, class N> T virt_get(const V& v, bool forward, N n, size_t idx) {
    return forward ? v(idx) : v(n - Off - idx);
}

// One bulge-chase step on (d, e): T := G^T T G on rows/cols (i, i+1), prev_bulge the fill at
// (i-1, i+1); returns the fill at (i, i+2). forward == false indexes bottom-up (QL). I is the
// caller's index type (its address arithmetic).
template <class D, class E, class T, class I>
BATCHLAS_DLA_INLINE T tridiag_rot(const v2::Thread&, const D& d, const E& e, const T& prev_bulge, I i, I j, v2::Givens<T> g,
                                  bool forward) {
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

}  // namespace batchlas::steqr_v2
