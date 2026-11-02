// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2, symmetric-indefinite family: the pieces of Bunch-Kaufman ?SYTF2 / ?LASYF (the
// pivot search and decision, the rank-1 and rank-2 trailing updates, the 2 x 2 pivot block),
// ?SYTRS / ?SYTRS2 (the 2 x 2 D solve, ?SYCONV per column, unit triangular sweeps on a
// register-distributed vector) and the index-range elementwise glue (fill, copy, swap, scal,
// axpy, gemv, ?LACPY). Additive: the core headers are unchanged.
//
// Conventions of this header:
//  - RtGroup is a run-time-width executor: the launch picks the width per order (32 .. 256).
//    Its argmax and sum are deterministic (a 32-lane xor butterfly, then the sub-group results
//    in sub-group order through caller scratch), so a pivot decision is the same on every lane.
//  - An IdxRange op visits absolute indices i in [lo, hi), owned from lo: i = lo + lid,
//    lo + lid + width, ... (LAPACK's reference loops, distributed). Two ops compose without a
//    barrier only when they share lo.
//  - Operands are anything indexable: v2 Vec / Mat, a lambda, or the views below (RevMat, the
//    index-reversed matrix that runs an Upper factorisation on the Lower loops; col_of, row_of,
//    cols_from, sym_row_lower).
//  - Scalar arithmetic is sfp::, the fma-written complex forms of src/sycl/device_scalar.hh
//    (dev_mul, dev_div, dev_recip), so ports stay bit-identical. A complex fma product is not
//    commutative in the last bit: each op states its operand order.
//  - No op contains a barrier unless it takes scratch; the caller owns every other barrier.
//  - Inlining follows the code each op replaced, because the attribute changes SASS (the
//    inliner's order moves unrolling, select-vs-branch and register allocation): an op that was
//    loop text in the kernel body is BATCHLAS_DLA_INLINE; one that was an `inline` helper
//    (scalar arithmetic, CABS1, the group reductions and pivot tests, the view accessors, the
//    register broadcast, the ?SYCONV column, the D-solve row) is BATCHLAS_DLA_SY_HELPER.

#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/blas/device/v2/level1.hh>
#include <batchlas/util/group-collectives.hh>

#ifndef BATCHLAS_DLA_UNROLL_FULL
#if BATCHLAS_SYCL_IMPL_ACPP
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
#define BATCHLAS_DLA_UNROLL_FULL _Pragma("unroll")
#endif
#endif

#ifndef BATCHLAS_DLA_SY_HELPER
#define BATCHLAS_DLA_SY_HELPER inline
#endif
// The per-item body of a loop op, spliced into the caller like the loop text it replaced.
#ifndef BATCHLAS_DLA_SY_BODY
#define BATCHLAS_DLA_SY_BODY __attribute__((always_inline))
#endif

namespace batchlas::device::v2 {

// ---- scalar arithmetic: the device_scalar.hh forms -----------------------------------------
namespace sfp {
template <class T> BATCHLAS_DLA_SY_HELPER T mul(T a, T b) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        return A::make(sycl::fma(A::re(a), A::re(b), -A::im(a) * A::im(b)),
                       sycl::fma(A::re(a), A::im(b), A::im(a) * A::re(b)));
    } else {
        return a * b;
    }
}
template <class T> BATCHLAS_DLA_SY_HELPER T add(T a, T b) { return Arith<T>::add(a, b); }
template <class T> BATCHLAS_DLA_SY_HELPER T sub(T a, T b) { return Arith<T>::sub(a, b); }
template <class T> BATCHLAS_DLA_SY_HELPER T neg(T a) { return Arith<T>::neg(a); }
template <class T> BATCHLAS_DLA_SY_HELPER T zero() { return Arith<T>::zero(); }
template <class T> BATCHLAS_DLA_SY_HELPER T one() { return Arith<T>::one(); }
// CABS1, |re| + |im|: LAPACK's Bunch-Kaufman pivot metric.
template <class T> BATCHLAS_DLA_SY_HELPER real_t<T> cabs1(T a) {
    using A = Arith<std::remove_cv_t<T>>;
    if constexpr (A::is_complex) return sycl::fabs(A::re(a)) + sycl::fabs(A::im(a));
    else return sycl::fabs(a);
}
template <class T> BATCHLAS_DLA_SY_HELPER T nan() {
    using R = real_t<T>;
    constexpr R q = std::numeric_limits<R>::quiet_NaN();
    if constexpr (Arith<T>::is_complex) return Arith<T>::make(q, q);
    else return q;
}
// Smith's reciprocal and quotient (never (c - di) / (c^2 + d^2), which overflows).
template <class T> BATCHLAS_DLA_SY_HELPER T recip(T d) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R dr = A::re(d), di = A::im(d);
        if (sycl::fabs(dr) >= sycl::fabs(di)) {
            const R r = di / dr;
            const R den = sycl::fma(di, r, dr);
            return A::make(R(1) / den, -r / den);
        }
        const R r = dr / di;
        const R den = sycl::fma(dr, r, di);
        return A::make(r / den, R(-1) / den);
    } else {
        return T(1) / d;
    }
}
template <class T> BATCHLAS_DLA_SY_HELPER T div(T a, T b) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R ar = A::re(a), ai = A::im(a), br = A::re(b), bi = A::im(b);
        if (sycl::fabs(br) >= sycl::fabs(bi)) {
            const R r = bi / br;
            const R den = sycl::fma(bi, r, br);
            return A::make(sycl::fma(ai, r, ar) / den, sycl::fma(-ar, r, ai) / den);
        }
        const R r = br / bi;
        const R den = sycl::fma(br, r, bi);
        return A::make(sycl::fma(ar, r, ai) / den, sycl::fma(ai, r, -ar) / den);
    } else {
        return a / b;
    }
}
// A NaN key ranks above every number (so a NaN column is found, and a lane's pick never
// depends on a NaN compare).
template <class R> BATCHLAS_DLA_SY_HELPER R nan_as_inf(R v) { return sycl::isnan(v) ? std::numeric_limits<R>::infinity() : v; }
}  // namespace sfp

template <class T> BATCHLAS_DLA_SY_HELPER void swap_elem(T& a, T& b) {
    const T t = a;
    a = b;
    b = t;
}

// ---- RtGroup: the run-time-width executor ------------------------------------------------
// `nl` items with ids `lid_` (a whole work-group, or with sg_only one sub-group of a wider
// work-group, nl == 32). Its barrier is sycl::group_barrier at that scope (all memory).
struct RtGroup {
    static constexpr Scope natural_scope = Scope::Group;
    static constexpr bool sub_whole = false;
    sycl::nd_item<1> it;
    int lid_;
    int nl;
    bool sg_only = false;

    BATCHLAS_DLA_SY_HELPER int lid() const { return lid_; }
    BATCHLAS_DLA_SY_HELPER int width() const { return nl; }
    BATCHLAS_DLA_SY_HELPER bool leader() const { return lid_ == 0; }
    BATCHLAS_DLA_SY_HELPER void barrier(Mem = Mem::Local) const {
        if (sg_only) sycl::group_barrier(it.get_sub_group());
        else sycl::group_barrier(it.get_group());
    }
    // Item 0's value, to every item.
    BATCHLAS_DLA_SY_HELPER int bcast(int v) const {
        if (sg_only) return batchlas::portable::group_broadcast(it.get_sub_group(), v, 0);
        return batchlas::portable::group_broadcast(it.get_group(), v, 0);
    }
};

// Found by ADL from the core level-1 ops (more specialised than the core template).
template <class F> BATCHLAS_DLA_INLINE void for_each_owned(const RtGroup& ex, int n, F&& f) {
    for (int i = ex.lid(); i < n; i += ex.width()) f(i);
}

namespace detail_sy {
template <class Ex> BATCHLAS_DLA_INLINE int width(const Ex& ex) {
    if constexpr (requires(const Ex& e) { e.width(); }) return ex.width();
    else return Ex::size;
}
}  // namespace detail_sy

// ---- index ranges and views ---------------------------------------------------------------
struct IdxRange {
    int lo;
    int hi;
};

template <class Ex, class F> BATCHLAS_DLA_INLINE void for_range(const Ex& ex, IdxRange r, F&& f) {
    const int w = detail_sy::width(ex);
    for (int i = r.lo + ex.lid(); i < r.hi; i += w) f(i);
}

// Logical (i, j) of an m x n matrix; with rev, physical (m - 1 - i, n - 1 - j). The Upper
// triangle of A is the Lower triangle of its reversal, so one Lower loop serves both uplo.
template <class T> struct RevMat {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::ptrdiff_t ld;
    int m;
    int n;
    bool rev;
    BATCHLAS_DLA_SY_HELPER T& operator()(int i, int j) const {
        return p[static_cast<std::ptrdiff_t>(rev ? m - 1 - i : i) + static_cast<std::ptrdiff_t>(rev ? n - 1 - j : j) * ld];
    }
};
template <class T> BATCHLAS_DLA_INLINE RevMat<T> rev_mat(T* p, std::ptrdiff_t ld, int n, bool rev) { return {p, ld, n, n, rev}; }
template <class T> BATCHLAS_DLA_INLINE RevMat<T> rev_mat(T* p, std::ptrdiff_t ld, int m, int n, bool rev) {
    return {p, ld, m, n, rev};
}

template <class T> struct RevVec {
    using value_type = std::remove_cv_t<T>;
    T* p;
    int n;
    bool rev;
    BATCHLAS_DLA_SY_HELPER T& operator()(int i) const { return p[rev ? n - 1 - i : i]; }
};

template <class M> struct ColOf {
    M a;
    int j;
    BATCHLAS_DLA_SY_HELPER decltype(auto) operator()(int i) const { return a(i, j); }
};
template <class M> struct RowOf {
    M a;
    int i;
    BATCHLAS_DLA_SY_HELPER decltype(auto) operator()(int j) const { return a(i, j); }
};
// Columns j0, j0 + 1, ... as columns 0, 1, ...
template <class M> struct ColsFrom {
    M a;
    int j0;
    BATCHLAS_DLA_SY_HELPER decltype(auto) operator()(int i, int l) const { return a(i, j0 + l); }
};
// Row r of a symmetric matrix, read from its lower triangle.
template <class M> struct SymRowLower {
    M a;
    int r;
    BATCHLAS_DLA_SY_HELPER decltype(auto) operator()(int j) const { return j > r ? a(j, r) : a(r, j); }
};
template <class M> BATCHLAS_DLA_INLINE ColOf<M> col_of(const M& a, int j) { return {a, j}; }
template <class M> BATCHLAS_DLA_INLINE RowOf<M> row_of(const M& a, int i) { return {a, i}; }
template <class M> BATCHLAS_DLA_INLINE ColsFrom<M> cols_from(const M& a, int j0) { return {a, j0}; }
template <class M> BATCHLAS_DLA_INLINE SymRowLower<M> sym_row_lower(const M& a, int r) { return {a, r}; }

// ---- level 1 over an index range ------------------------------------------------------------
// y := -x * v, written y - x v: the axpy alpha of ?SYTRS's forward sweep.
template <class T> struct Minus {
    T v;
};
template <class T> Minus(T) -> Minus<T>;

template <class Ex, class T, class Y> BATCHLAS_DLA_INLINE void fill(const Ex& ex, T v, const Y& y, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { y(i) = v; });
}
template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void copy(const Ex& ex, const X& x, const Y& y, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { y(i) = x(i); });
}
template <class Ex, class X, class Y> BATCHLAS_DLA_INLINE void swap(const Ex& ex, const X& x, const Y& y, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { swap_elem(x(i), y(i)); });
}
// x(i) := alpha * x(i) (product alpha first).
template <class Ex, class T, class X> BATCHLAS_DLA_INLINE void scal(const Ex& ex, T alpha, const X& x, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { x(i) = sfp::mul(alpha, x(i)); });
}
// y(i) := f(i)
template <class Ex, class Y, class F> BATCHLAS_DLA_INLINE void map_range(const Ex& ex, const Y& y, IdxRange r, F&& f) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { y(i) = f(i); });
}
// y(i) := y(i) - x(i) a (product x first).
template <class Ex, class T, class X, class Y>
BATCHLAS_DLA_INLINE void axpy(const Ex& ex, Minus<T> a, const X& x, const Y& y, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY { y(i) = sfp::sub(y(i), sfp::mul(static_cast<T>(x(i)), a.v)); });
}
// y(i) := (y(i) - x0(i) a0) - x1(i) a1, one pass.
template <class Ex, class T, class X0, class X1, class Y>
BATCHLAS_DLA_INLINE void axpy2(const Ex& ex, Minus<T> a0, const X0& x0, Minus<T> a1, const X1& x1, const Y& y, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        y(i) = sfp::sub(sfp::sub(y(i), sfp::mul(static_cast<T>(x0(i)), a0.v)), sfp::mul(static_cast<T>(x1(i)), a1.v));
    });
}

namespace detail_sy {
// acc - a(l0) b(l0) - a(l0 + 1) b(l0 + 1) - ..., one term at a time (LAPACK's reference order).
template <class T, class FA, class FB> BATCHLAS_DLA_INLINE T sub_dot(T acc, int l0, int l1, FA&& a, FB&& b) {
    for (int l = l0; l < l1; ++l) acc = sfp::sub(acc, sfp::mul(static_cast<T>(a(l)), static_cast<T>(b(l))));
    return acc;
}
}  // namespace detail_sy

// ---- level 2 / 3 over an index range ---------------------------------------------------------
// y(i) := b(i) - sum_{l0 <= l < l1} A(i, l) x(l) for i in r: gemv NoTrans, alpha -1, beta 1, the
// sum one term at a time from b(i). b may be y. Item per row, no reduction.
template <class Ex, class MA, class X, class B, class Y>
BATCHLAS_DLA_INLINE void gemv_sub(const Ex& ex, const MA& a, const X& x, const B& b, const Y& y, IdxRange r, int l0, int l1) {
    // The body must be spliced in (BATCHLAS_DLA_SY_BODY): simplified as its own function, the l
    // loop is IV-canonicalised (exit `==`) and the device link then runtime-unrolls it 8x.
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        using T = std::remove_cvref_t<decltype(b(i))>;
        T acc = b(i);
        for (int l = l0; l < l1; ++l) acc = sfp::sub(acc, sfp::mul(static_cast<T>(a(i, l)), static_cast<T>(x(l))));
        y(i) = acc;
    });
}

// A := A + alpha x x^T on the lower triangle of A(r, r): ?SYR's column loop, t = alpha x(j)
// (alpha first), then A(i, j) += x(i) t for i >= j. x must not be a column of A(r, r).
template <Uplo UL, class Ex, class T, class X, class MA>
BATCHLAS_DLA_INLINE void syr(const Ex& ex, T alpha, const X& x, const MA& a, IdxRange r) {
    static_assert(UL == Uplo::Lower, "v2::syr (range form): only Uplo::Lower is implemented");
    for (int j = r.lo; j < r.hi; ++j) {
        const T t = sfp::mul(alpha, static_cast<T>(x(j)));
        for_range(ex, IdxRange{j, r.hi}, [&](int i) BATCHLAS_DLA_SY_BODY { a(i, j) = sfp::add(static_cast<T>(a(i, j)), sfp::mul(static_cast<T>(x(i)), t)); });
    }
}

// A := A - x0 w0^T - x1 w1^T on the lower triangle of A(r, r) (?GEMMT, k = 2, alpha -1,
// beta 1): per column j, w0(j) then w1(j) once (they may read A outside A(r, r)), then
// A(i, j) := (A(i, j) - x0(i) w0(j)) - x1(i) w1(j) for i >= j.
template <Uplo UL, class Ex, class X0, class W0, class X1, class W1, class MA>
BATCHLAS_DLA_INLINE void gemmt2_sub(const Ex& ex, const X0& x0, W0&& w0, const X1& x1, W1&& w1, const MA& a, IdxRange r) {
    static_assert(UL == Uplo::Lower, "v2::gemmt2_sub: only Uplo::Lower is implemented");
    using T = typename MA::value_type;
    for (int j = r.lo; j < r.hi; ++j) {
        const T c0 = w0(j);
        const T c1 = w1(j);
        for_range(ex, IdxRange{j, r.hi}, [&](int i) BATCHLAS_DLA_SY_BODY {
            a(i, j) = sfp::sub(sfp::sub(static_cast<T>(a(i, j)), sfp::mul(static_cast<T>(x0(i)), c0)),
                               sfp::mul(static_cast<T>(x1(i)), c1));
        });
    }
}

// C(i, j) -= sum_{l < k} X(i, l) Y(j, l) for i >= j on the diagonal block [j0, j0 + jb)^2,
// flat over jb * jb (row fastest) so a block fills the executor (?GEMMT NoTrans/Trans on a
// block, alpha -1, beta 1; the sum one term at a time from C(i, j)).
template <Uplo UL, class Ex, class MX, class MY, class MC>
BATCHLAS_DLA_INLINE void gemmt_block_sub(const Ex& ex, const MX& x, const MY& y, const MC& c, int j0, int jb, int k) {
    static_assert(UL == Uplo::Lower, "v2::gemmt_block_sub: only Uplo::Lower is implemented");
    using T = typename MC::value_type;
    const int w = detail_sy::width(ex);
    for (int idx = ex.lid(); idx < jb * jb; idx += w) {
        const int i = j0 + idx % jb, j = j0 + idx / jb;
        if (i >= j) c(i, j) = detail_sy::sub_dot(static_cast<T>(c(i, j)), 0, k, [&](int l) -> T { return x(i, l); }, [&](int l) -> T { return y(j, l); });
    }
}

// B := A on the UL triangle of n x n (Diag::Unit: the strict triangle), flat over n * n with
// index type Idx (row fastest).
template <Uplo UL, Diag DG = Diag::NonUnit, class Idx = int, class Ex, class MA, class MB>
BATCHLAS_DLA_INLINE void lacpy_flat(const Ex& ex, const MA& a, const MB& b, int n) {
    static_assert(UL == Uplo::Lower, "v2::lacpy_flat: only Uplo::Lower is implemented");
    const Idx nn = static_cast<Idx>(n) * n;
    for (Idx idx = ex.lid(); idx < nn; idx += detail_sy::width(ex)) {
        const int i = static_cast<int>(idx % n), j = static_cast<int>(idx / n);
        if constexpr (DG == Diag::Unit) {
            if (i > j) b(i, j) = a(i, j);
        } else {
            if (i >= j) b(i, j) = a(i, j);
        }
    }
}

// Rows r of the first ncols columns := v.
template <class Ex, class T, class MA> BATCHLAS_DLA_INLINE void fill_rows(const Ex& ex, T v, const MA& a, IdxRange r, int ncols) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        for (int c = 0; c < ncols; ++c) a(i, c) = v;
    });
}

// ---- deterministic group reductions on RtGroup ---------------------------------------------
template <class R> struct ArgMax {
    R v;
    int i;
};
template <class R> struct ArgMaxScratch {
    R* v;    // width / 32 slots
    int* i;  // width / 32 slots
};
template <class R> struct SumScratch {
    R* re;  // width / 32 slots
    R* im;  // width / 32 slots (complex only)
};

// (ov, oi) beats (bv, bi): a larger value; on a tie the preferred index.
template <class R> BATCHLAS_DLA_SY_HELPER bool argmax_beats(R ov, int oi, R bv, int bi, bool prefer_low) {
    return ov > bv || (ov == bv && (prefer_low ? oi < bi : oi > bi));
}

// Uniform (v, idx): a 32-lane xor butterfly (offsets 16 .. 1), then with more than one
// sub-group the sub-group winners in sub-group order through scratch (two barriers).
template <class R> BATCHLAS_DLA_SY_HELPER void argmax_reduce(const RtGroup& ex, ArgMaxScratch<R> scr, R& v, int& idx, bool prefer_low) {
    auto sg = ex.it.get_sub_group();
    for (unsigned off = 16; off > 0; off >>= 1) {
        const R ov = sycl::permute_group_by_xor(sg, v, off);
        const int oi = sycl::permute_group_by_xor(sg, idx, off);
        if (argmax_beats(ov, oi, v, idx, prefer_low)) {
            v = ov;
            idx = oi;
        }
    }
    const int nsg = ex.nl / 32;
    if (nsg == 1) return;
    if (sg.leader()) {
        scr.v[sg.get_group_linear_id()] = v;
        scr.i[sg.get_group_linear_id()] = idx;
    }
    ex.barrier();
    v = scr.v[0];
    idx = scr.i[0];
    for (int s = 1; s < nsg; ++s)
        if (argmax_beats(scr.v[s], scr.i[s], v, idx, prefer_low)) {
            v = scr.v[s];
            idx = scr.i[s];
        }
    ex.barrier();
}

struct NoSkip {};
struct SkipAt {
    int i;
};

// ?SYTF2's pivot search: the arg-max over i in r of nan_as_inf(|re x(i)| + |im x(i)|) (CABS1),
// ties to the lower index (prefer_low) or the higher; SkipAt leaves one index out. An empty
// range gives {-1, -1}. Uniform; the barriers of argmax_reduce.
template <class Skip = NoSkip, class X, class R>
BATCHLAS_DLA_INLINE ArgMax<R> iamax_bk(const RtGroup& ex, ArgMaxScratch<R> scr, const X& x, IdxRange r, bool prefer_low, Skip skip = {}) {
    R cv = R(-1);
    int ci = -1;
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        if constexpr (std::is_same_v<Skip, SkipAt>) {
            if (i == skip.i) return;
        }
        const R v = sfp::nan_as_inf(sfp::cabs1(x(i)));
        if (argmax_beats(v, i, cv, ci, prefer_low)) {
            cv = v;
            ci = i;
        }
    });
    argmax_reduce(ex, scr, cv, ci, prefer_low);
    return {cv, ci};
}

// Uniform sum: the butterfly, then the sub-group sums in sub-group order through `slots`.
template <class R> BATCHLAS_DLA_SY_HELPER R sum_reduce_real(const RtGroup& ex, R v, R* slots) {
    auto sg = ex.it.get_sub_group();
    for (unsigned off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    const int nsg = ex.nl / 32;
    if (nsg == 1) return v;
    if (sg.leader()) slots[sg.get_group_linear_id()] = v;
    ex.barrier();
    v = slots[0];
    for (int s = 1; s < nsg; ++s) v += slots[s];
    ex.barrier();
    return v;
}
template <class T, class R> BATCHLAS_DLA_SY_HELPER T sum_reduce(const RtGroup& ex, SumScratch<R> scr, T v) {
    using A = Arith<T>;
    if constexpr (A::is_complex) {
        const R re = sum_reduce_real(ex, A::re(v), scr.re);
        const R im = sum_reduce_real(ex, A::im(v), scr.im);
        return A::make(re, im);
    } else {
        return sum_reduce_real(ex, v, scr.re);
    }
}

// {x0^T y, two ? x1^T y : 0} over r (products x first), each a sum_reduce. Every item must
// pass the same `two`.
template <class T, class R, class X0, class X1, class Y>
BATCHLAS_DLA_INLINE std::array<T, 2> dotu2_sum(const RtGroup& ex, SumScratch<R> scr, bool two, const X0& x0, const X1& x1, const Y& y, IdxRange r) {
    T p0 = sfp::zero<T>(), p1 = sfp::zero<T>();
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        p0 = sfp::add(p0, sfp::mul(static_cast<T>(x0(i)), static_cast<T>(y(i))));
        if (two) p1 = sfp::add(p1, sfp::mul(static_cast<T>(x1(i)), static_cast<T>(y(i))));
    });
    const T s0 = sum_reduce(ex, scr, p0);
    const T s1 = two ? sum_reduce(ex, scr, p1) : sfp::zero<T>();
    return {s0, s1};
}

// ---- Bunch-Kaufman ---------------------------------------------------------------------------
// ?SYTF2's first test, item 0's verdict to every item: a zero (or NaN) column, a 1 x 1 pivot
// without interchange, or the row search.
inline constexpr int kBkNeedRow = 0;
inline constexpr int kBkZero = 1;
inline constexpr int kBkKeep = 2;

template <class R> BATCHLAS_DLA_SY_HELPER R bk_alpha() { return (R(1) + sycl::sqrt(R(17))) / R(8); }

template <class R> BATCHLAS_DLA_SY_HELPER int bk_stage(const RtGroup& ex, R absakk, R colmax, R alpha) {
    return ex.bcast(sycl::fmax(absakk, colmax) == R(0) || sycl::isnan(absakk) ? kBkZero
                    : absakk >= alpha * colmax                                ? kBkKeep
                                                                              : kBkNeedRow);
}

// ?SYTF2's second test after the row search: kp (k or imax) and kstep (1 or 2), item 0's.
template <class R>
BATCHLAS_DLA_SY_HELPER void bk_choice(const RtGroup& ex, R absakk, R colmax, R rowmax, R absimax, R alpha, int k, int imax, int& kp, int& kstep) {
    kp = imax;
    kstep = 1;
    if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
    else if (!(absimax >= alpha * rowmax)) kstep = 2;
    const int code = ex.bcast(kp * 2 + kstep - 1);
    kp = code >> 1;
    kstep = (code & 1) + 1;
}

// The 2 x 2 pivot [[akk, ak1k], [ak1k, ak1k1]] of ?SYTF2 / ?LASYF, inverted LAPACK's way:
// d11 = ak1k1 / d21, d22 = akk / d21, e21 = (1 / (d11 d22 - 1)) / d21.
template <class T> struct BkBlock2 {
    T d11, d22, e21;
    // Row entries of [xk xk1] D^-1.
    BATCHLAS_DLA_INLINE T first(T xk, T xk1) const { return sfp::mul(e21, sfp::sub(sfp::mul(d11, xk), xk1)); }
    BATCHLAS_DLA_INLINE T second(T xk, T xk1) const { return sfp::mul(e21, sfp::sub(sfp::mul(d22, xk1), xk)); }
};
template <class T> BATCHLAS_DLA_INLINE BkBlock2<T> bk_block2(T akk, T ak1k, T ak1k1) {
    const T d21 = ak1k;
    const T d11 = sfp::div(ak1k1, d21);
    const T d22 = sfp::div(akk, d21);
    const T t = sfp::recip(sfp::sub(sfp::mul(d11, d22), sfp::one<T>()));
    const T e21 = sfp::div(t, d21);
    return {d11, d22, e21};
}
// [y0 y1](i) := [x0 x1](i) D^-1 for i in r (x may be y).
template <class Ex, class T, class X0, class X1, class Y0, class Y1>
BATCHLAS_DLA_INLINE void bk_apply2(const Ex& ex, const BkBlock2<T>& blk, const X0& x0, const X1& x1, const Y0& y0, const Y1& y1, IdxRange r) {
    for_range(ex, r, [&](int i) BATCHLAS_DLA_SY_BODY {
        const T u = x0(i), v = x1(i);
        const T a = blk.first(u, v);
        const T b = blk.second(u, v);
        y0(i) = a;
        y1(i) = b;
    });
}

// ?SYTRS's 2 x 2 D solve, D = [[dkk, off], [off, dk1k1]]: akm1 = dkk / off, ak = dk1k1 / off,
// denom = akm1 ak - 1; solve(bk, bk1) -> D^-1 [bk; bk1].
template <class T> struct LdlD2 {
    T off, akm1, ak, denom;
    BATCHLAS_DLA_INLINE void solve(T bk, T bk1, T& o0, T& o1) const {
        const T bkm1 = sfp::div(bk, off);
        const T bkk = sfp::div(bk1, off);
        o0 = sfp::div(sfp::sub(sfp::mul(ak, bkm1), bkk), denom);
        o1 = sfp::div(sfp::sub(sfp::mul(akm1, bkk), bkm1), denom);
    }
};
template <class T> BATCHLAS_DLA_INLINE LdlD2<T> ldl_d2(T off, T dkk, T dk1k1) {
    const T akm1 = sfp::div(dkk, off);
    const T ak = sfp::div(dk1k1, off);
    const T denom = sfp::sub(sfp::mul(akm1, ak), sfp::one<T>());
    return {off, akm1, ak, denom};
}

// Row t of D^-1 x for a block of kind kd (0: 1 x 1; 1 / 2: the first / second row of a 2 x 2):
// x row t's value, xp its partner's, dtt and dpp their diagonals, off the block's off-diagonal.
// ?SYTRS's arithmetic, so both rows of a block agree with LdlD2.
template <class T> BATCHLAS_DLA_SY_HELPER T ldl_dsolve_row(int kd, T x, T xp, T dtt, T dpp, T off) {
    if (kd == 0) return sfp::mul(x, sfp::recip(dtt));
    const T akm1 = sfp::div(kd == 1 ? dtt : dpp, off);
    const T ak = sfp::div(kd == 1 ? dpp : dtt, off);
    const T denom = sfp::sub(sfp::mul(akm1, ak), sfp::one<T>());
    const T bkm1 = sfp::div(kd == 1 ? x : xp, off);
    const T bk = sfp::div(kd == 1 ? xp : x, off);
    return kd == 1 ? sfp::div(sfp::sub(sfp::mul(ak, bkm1), bk), denom) : sfp::div(sfp::sub(sfp::mul(akm1, bk), bkm1), denom);
}

// ?SYCONV on column j of a staged lower factor (one item): the interchanges of the steps after j
// in step order (a 2 x 2 step at k swaps row k + 1), and the 2 x 2 off-diagonal of a block
// starting at j moved to e[j] (when e is set). ps is the logical ipiv (+(p + 1) / -(p + 1)).
template <class T, class ML> BATCHLAS_DLA_SY_HELPER void syconv_col(const ML& l, const int* ps, int n, int j, T* e) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v > 0) {
            if (k > j) swap_elem(l(k, j), l(v - 1, j));
            k += 1;
        } else {
            if (k == j) {
                if (e != nullptr) e[j] = l(j + 1, j);
                l(j + 1, j) = sfp::zero<T>();
            }
            if (k > j) swap_elem(l(k + 1, j), l(-v - 1, j));
            k += 2;
        }
    }
}

// ---- register-distributed vectors on one 32-lane sub-group ------------------------------------
// x[r] of lane l holds logical row l + 32 r.

// Row k, to every lane. Collective.
template <int R, class T, class SG> BATCHLAS_DLA_SY_HELPER T frag_get(const SG& sg, const T (&x)[R], int k) {
    T v = x[0];
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 1; r < R; ++r)
        if ((k >> 5) == r) v = x[r];
    return batchlas::portable::group_broadcast(sg, v, k & 31);
}

// x := op(L)^-1 x for L unit lower n x n (strict lower triangle read), column (NoTrans) or row
// (Trans) broadcast form: no reduction, no barrier. Collective.
template <Uplo UL, Transpose TR, Diag DG, int R, class SG, class ML, class T>
BATCHLAS_DLA_INLINE void trsv_frag(const SG& sg, int lane, const ML& l, int n, T (&x)[R]) {
    static_assert(UL == Uplo::Lower && DG == Diag::Unit, "v2::trsv_frag: only unit lower is implemented");
    if constexpr (TR == Transpose::NoTrans) {
        for (int k = 0; k + 1 < n; ++k) {
            const T xk = frag_get<R>(sg, x, k);
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t > k && t < n) x[r] = sfp::sub(x[r], sfp::mul(static_cast<T>(l(t, k)), xk));
            }
        }
    } else {
        static_assert(TR == Transpose::Trans, "v2::trsv_frag: NoTrans or Trans");
        for (int i = n - 1; i > 0; --i) {
            const T xi = frag_get<R>(sg, x, i);
            BATCHLAS_DLA_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t < i) x[r] = sfp::sub(x[r], sfp::mul(static_cast<T>(l(i, t)), xi));
            }
        }
    }
}

// x := D^-1 x for D block diagonal with 1 x 1 / 2 x 2 blocks: kind[t] per ldl_dsolve_row, the
// diagonal on d(t, t), e[k] the off-diagonal of the block starting at k. The partner rows come
// by one shift each way (no barrier). Collective.
template <int R, class SG, class MD, class T>
BATCHLAS_DLA_INLINE void ldl_dsolve_frag(const SG& sg, int lane, const MD& d, const int* kind, const T* e, int n, T (&x)[R]) {
    T up[R], dn[R];
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        up[r] = batchlas::portable::shift_group_left(sg, x[r], 1);
        dn[r] = batchlas::portable::shift_group_right(sg, x[r], 1);
        const T hi = batchlas::portable::group_broadcast(sg, x[r + 1 < R ? r + 1 : r], 0);
        const T lo = batchlas::portable::group_broadcast(sg, x[r > 0 ? r - 1 : 0], 31);
        if (lane == 31 && r + 1 < R) up[r] = hi;
        if (lane == 0 && r > 0) dn[r] = lo;
    }
    BATCHLAS_DLA_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        const int t = lane + 32 * r;
        if (t >= n) continue;
        const int kd = kind[t];
        const int p = kd == 1 ? t + 1 : kd == 2 ? t - 1 : t;
        const T off = kd == 1 ? e[t] : kd == 2 ? e[t - 1] : sfp::one<T>();
        x[r] = ldl_dsolve_row(kd, x[r], kd == 1 ? up[r] : dn[r], static_cast<T>(d(t, t)), static_cast<T>(d(p, p)), off);
    }
}

}  // namespace batchlas::device::v2
