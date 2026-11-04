// SPDX-License-Identifier: MIT
#pragma once
// sytrf / sytrs-private device LA v2 pieces (beyond include/batchlas/device): the
// index-reversed views, the deterministic Bunch-Kaufman pivot search, ?SYTF2's two pivot tests,
// the 2 x 2 pivot block, ?SYCONV on one column and the sub-group register sweeps of the fused
// ?SYTRS tiers. Executors are v2::RtGroup<Group> (one work-group per item) and v2::SubGroup<32>
// (one sub-group of a wider work-group). Arithmetic is v2 arith::Fma, which is
// src/sycl/device_scalar.hh's dev_mul / dev_div / dev_recip bit for bit; a complex fma product is
// not commutative in the last bit, so each op states its operand order.
// Tests: tests/device_blas_v2_sytrf-sytrs_tests.cc.
//
// Inlining follows the code each op replaced (core.hh 2.9): ops that were loop text are
// BATCHLAS_DLA_INLINE, ops that were plain-inline helpers are BATCHLAS_DLA_HELPER.

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/kernels/common/dla_v2_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::sytrf_v2 {

namespace v2 = batchlas::device;

template <class T> using Ar = v2::arith_t<v2::arith::Fma, T>;

template <class T> BATCHLAS_DLA_HELPER T nan_of() {
    using R = v2::real_t<T>;
    constexpr R q = std::numeric_limits<R>::quiet_NaN();
    return Ar<T>::make(q, q);
}

// ---- views ------------------------------------------------------------------------------------
// Logical (i, j) of an m x n matrix; with rev, physical (m - 1 - i, n - 1 - j). The Upper
// triangle of A is the Lower triangle of its reversal, so one Lower loop serves both uplo.
template <class T> struct RevMat {
    using value_type = std::remove_cv_t<T>;
    T* p;
    std::ptrdiff_t ld;
    int m;
    int n;
    bool rev;
    BATCHLAS_DLA_HELPER int rows() const { return m; }
    BATCHLAS_DLA_HELPER int cols() const { return n; }
    BATCHLAS_DLA_HELPER T& operator()(int i, int j) const {
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
    BATCHLAS_DLA_HELPER int size() const { return n; }
    BATCHLAS_DLA_HELPER T& operator()(int i) const { return p[rev ? n - 1 - i : i]; }
};

// ---- executor glue ----------------------------------------------------------------------------
// The legacy barrier: sycl::group_barrier over the work-group (RtGroup; the global-memory tiers
// factor in place, so it must order global memory too: v2::sync fences local memory only) or
// over the sub-group (SubGroup).
template <class Ex> BATCHLAS_DLA_HELPER void group_sync(const Ex& ex) {
    if constexpr (v2::is_runtime_v<Ex>) sycl::group_barrier(ex.it.get_group());
    else sycl::group_barrier(ex.sub());
}

// Item 0's v, to every item: every pivot decision goes through this, so no lane can diverge
// from the next barrier (docs/perf/sytrf.md#sytrf-termination).
template <class Ex> BATCHLAS_DLA_HELPER int uniform(const Ex& ex, int v) {
    if constexpr (v2::is_runtime_v<Ex>) return ex.bcast(v, 0);
    else return batchlas::portable::group_broadcast(ex.sub(), v, 0);
}

// ---- deterministic group reductions -----------------------------------------------------------
// A 32-lane xor butterfly (offsets 16 .. 1), then with more than one sub-group the sub-group
// results in sub-group order through caller scratch (width / 32 slots, two barriers): the same
// value on every item, whatever the width. v2::argmax / dot_many reduce in another order.
template <class R> struct ArgMaxScratch {
    R* v;
    int* i;
};
template <class R> struct SumScratch {
    R* re;
    R* im;  // complex only
};
template <class R> struct ArgMax {
    R v;
    int i;
};

// (ov, oi) beats (bv, bi): a larger value; on a tie the preferred index.
template <class R> BATCHLAS_DLA_HELPER bool argmax_beats(R ov, int oi, R bv, int bi, bool prefer_low) {
    return ov > bv || (ov == bv && (prefer_low ? oi < bi : oi > bi));
}

template <class Ex, class R> BATCHLAS_DLA_HELPER void argmax_reduce(const Ex& ex, ArgMaxScratch<R> scr, R& v, int& idx, bool prefer_low) {
    auto sg = ex.sub();
    for (unsigned off = 16; off > 0; off >>= 1) {
        const R ov = sycl::permute_group_by_xor(sg, v, off);
        const int oi = sycl::permute_group_by_xor(sg, idx, off);
        if (argmax_beats(ov, oi, v, idx, prefer_low)) {
            v = ov;
            idx = oi;
        }
    }
    const int nsg = v2::detail::size_of(ex) / 32;
    if (nsg == 1) return;
    if (sg.leader()) {
        scr.v[sg.get_group_linear_id()] = v;
        scr.i[sg.get_group_linear_id()] = idx;
    }
    group_sync(ex);
    v = scr.v[0];
    idx = scr.i[0];
    for (int s = 1; s < nsg; ++s)
        if (argmax_beats(scr.v[s], scr.i[s], v, idx, prefer_low)) {
            v = scr.v[s];
            idx = scr.i[s];
        }
    group_sync(ex);
}

template <class Ex, class R> BATCHLAS_DLA_HELPER R sum_reduce_real(const Ex& ex, R v, R* slots) {
    auto sg = ex.sub();
    for (unsigned off = 16; off > 0; off >>= 1) v += sycl::permute_group_by_xor(sg, v, off);
    const int nsg = v2::detail::size_of(ex) / 32;
    if (nsg == 1) return v;
    if (sg.leader()) slots[sg.get_group_linear_id()] = v;
    group_sync(ex);
    v = slots[0];
    for (int s = 1; s < nsg; ++s) v += slots[s];
    group_sync(ex);
    return v;
}
template <class Ex, class T, class R> BATCHLAS_DLA_HELPER T sum_reduce(const Ex& ex, SumScratch<R> scr, T v) {
    using A = Ar<T>;
    if constexpr (A::is_complex) {
        const R re = sum_reduce_real(ex, A::re(v), scr.re);
        const R im = sum_reduce_real(ex, A::im(v), scr.im);
        return A::make(re, im);
    } else {
        return sum_reduce_real(ex, v, scr.re);
    }
}

// A NaN key ranks above every number, so a NaN column is found and no lane's pick depends on a
// NaN compare.
template <class R> BATCHLAS_DLA_HELPER R nan_as_inf(R v) { return sycl::isnan(v) ? std::numeric_limits<R>::infinity() : v; }

// ?SYTF2's pivot search: the arg-max over i in r of nan_as_inf(abs1(x(i))) (LAPACK's CABS1),
// ties to the lower index (prefer_low) or the higher; with Skip, index `skip` is left out. An
// empty range gives {-1, -1}. Uniform; the barriers of argmax_reduce.
template <bool Skip = false, class Ex, class R, class X>
BATCHLAS_DLA_INLINE ArgMax<R> iamax_bk(const Ex& ex, ArgMaxScratch<R> scr, const X& x, v2::Range r, bool prefer_low, int skip = -1) {
    R cv = R(-1);
    int ci = -1;
    v2::for_each_owned(ex, r, [&](int i) BATCHLAS_DLA_BODY {
        if constexpr (Skip) {
            if (i == skip) return;
        }
        const R v = nan_as_inf(v2::abs1(x(i)));
        if (argmax_beats(v, i, cv, ci, prefer_low)) {
            cv = v;
            ci = i;
        }
    });
    argmax_reduce(ex, scr, cv, ci, prefer_low);
    return {cv, ci};
}

// ---- Bunch-Kaufman ----------------------------------------------------------------------------
// ?SYTF2's first test, item 0's verdict to every item: a zero (or NaN) column, a 1 x 1 pivot
// without interchange, or the row search. ?SYTF2 and ?LASYF share both tests.
inline constexpr int kBkNeedRow = 0;
inline constexpr int kBkZero = 1;
inline constexpr int kBkKeep = 2;

template <class R> BATCHLAS_DLA_HELPER R bk_alpha() { return (R(1) + sycl::sqrt(R(17))) / R(8); }

template <class Ex, class R> BATCHLAS_DLA_HELPER int bk_stage(const Ex& ex, R absakk, R colmax, R alpha) {
    return uniform(ex, sycl::fmax(absakk, colmax) == R(0) || sycl::isnan(absakk) ? kBkZero
                       : absakk >= alpha * colmax                                ? kBkKeep
                                                                                 : kBkNeedRow);
}

// ?SYTF2's second test after the row search: kp (k or imax) and kstep (1 or 2), item 0's.
template <class Ex, class R>
BATCHLAS_DLA_HELPER void bk_choice(const Ex& ex, R absakk, R colmax, R rowmax, R absimax, R alpha, int k, int imax, int& kp, int& kstep) {
    kp = imax;
    kstep = 1;
    if (absakk >= alpha * colmax * (colmax / rowmax)) kp = k;
    else if (!(absimax >= alpha * rowmax)) kstep = 2;
    const int code = uniform(ex, kp * 2 + kstep - 1);
    kp = code >> 1;
    kstep = (code & 1) + 1;
}

// The 2 x 2 pivot [[akk, ak1k], [ak1k, ak1k1]] of ?SYTF2 / ?LASYF, inverted LAPACK's way:
// d11 = ak1k1 / d21, d22 = akk / d21, e21 = (1 / (d11 d22 - 1)) / d21.
template <class T> struct BkBlock2 {
    T d11, d22, e21;
    // The two entries of the row [xk xk1] D^-1.
    BATCHLAS_DLA_INLINE T first(T xk, T xk1) const { return Ar<T>::mul(e21, Ar<T>::sub(Ar<T>::mul(d11, xk), xk1)); }
    BATCHLAS_DLA_INLINE T second(T xk, T xk1) const { return Ar<T>::mul(e21, Ar<T>::sub(Ar<T>::mul(d22, xk1), xk)); }
};
template <class T> BATCHLAS_DLA_INLINE BkBlock2<T> bk_block2(T akk, T ak1k, T ak1k1) {
    using A = Ar<T>;
    const T d21 = ak1k;
    const T d11 = A::div(ak1k1, d21);
    const T d22 = A::div(akk, d21);
    const T t = A::recip(A::sub(A::mul(d11, d22), A::one()));
    return {d11, d22, A::div(t, d21)};
}

// Row t of D^-1 x for a block of kind kd (0: 1 x 1; 1 / 2: the first / second row of a 2 x 2):
// x is row t's value, xp its partner's, dtt and dpp their diagonals, off the block's
// off-diagonal. ?SYTRS's arithmetic, so both rows of a block agree with ldl_solve's.
template <class T> BATCHLAS_DLA_HELPER T ldl_dsolve_row(int kd, T x, T xp, T dtt, T dpp, T off) {
    using A = Ar<T>;
    if (kd == 0) return A::mul(x, A::recip(dtt));
    const T akm1 = A::div(kd == 1 ? dtt : dpp, off);
    const T ak = A::div(kd == 1 ? dpp : dtt, off);
    const T denom = A::sub(A::mul(akm1, ak), A::one());
    const T bkm1 = A::div(kd == 1 ? x : xp, off);
    const T bk = A::div(kd == 1 ? xp : x, off);
    return kd == 1 ? A::div(A::sub(A::mul(ak, bkm1), bk), denom) : A::div(A::sub(A::mul(akm1, bk), bkm1), denom);
}

// ?SYCONV on column j of a staged lower factor (one item): the interchanges of the steps after j
// in step order (a 2 x 2 step at k swaps row k + 1), and the 2 x 2 off-diagonal of a block
// starting at j moved to e[j] (when e is set). ps is the logical ipiv (+(p + 1) / -(p + 1)).
template <class T, class ML> BATCHLAS_DLA_HELPER void syconv_col(const ML& l, const int* ps, int n, int j, T* e) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v > 0) {
            if (k > j) std::swap(l(k, j), l(v - 1, j));
            k += 1;
        } else {
            if (k == j) {
                if (e != nullptr) e[j] = l(j + 1, j);
                l(j + 1, j) = Ar<T>::zero();
            }
            if (k > j) std::swap(l(k + 1, j), l(-v - 1, j));
            k += 2;
        }
    }
}

// ---- register-distributed vectors on one 32-lane sub-group --------------------------------------
// x[r] of lane l holds logical row l + 32 r. v2::trsv is Thread-only; these are the distributed
// (broadcast-form) sweeps: no reduction, no barrier. Collective.

// Row k, to every lane.
template <int R, class T, class SG> BATCHLAS_DLA_HELPER T frag_get(const SG& sg, const T (&x)[R], int k) {
    T v = x[0];
    BATCHLAS_UNROLL_FULL
    for (int r = 1; r < R; ++r)
        if ((k >> 5) == r) v = x[r];
    return batchlas::portable::group_broadcast(sg, v, k & 31);
}

// x := op(L)^-1 x for L unit lower n x n (the strict lower triangle read), NoTrans by columns,
// Trans by rows.
template <Transpose TR, int R, class SG, class ML, class T>
BATCHLAS_DLA_INLINE void trsv_frag(const SG& sg, int lane, const ML& l, int n, T (&x)[R]) {
    using A = Ar<T>;
    if constexpr (TR == Transpose::NoTrans) {
        for (int k = 0; k + 1 < n; ++k) {
            const T xk = frag_get<R>(sg, x, k);
            BATCHLAS_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t > k && t < n) x[r] = A::sub(x[r], A::mul(l(t, k), xk));
            }
        }
    } else {
        static_assert(TR == Transpose::Trans, "sytrf_v2::trsv_frag: NoTrans or Trans");
        for (int i = n - 1; i > 0; --i) {
            const T xi = frag_get<R>(sg, x, i);
            BATCHLAS_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t < i) x[r] = A::sub(x[r], A::mul(l(i, t), xi));
            }
        }
    }
}

// x := D^-1 x for D block diagonal with 1 x 1 / 2 x 2 blocks: kind[t] as ldl_dsolve_row's, the
// diagonal on d(t, t), e[k] the off-diagonal of the block starting at k. The partner rows come by
// one shift each way.
template <int R, class SG, class MD, class T>
BATCHLAS_DLA_INLINE void ldl_dsolve_frag(const SG& sg, int lane, const MD& d, const int* kind, const T* e, int n, T (&x)[R]) {
    T up[R], dn[R];
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        up[r] = batchlas::portable::shift_group_left(sg, x[r], 1);
        dn[r] = batchlas::portable::shift_group_right(sg, x[r], 1);
        const T hi = batchlas::portable::group_broadcast(sg, x[r + 1 < R ? r + 1 : r], 0);
        const T lo = batchlas::portable::group_broadcast(sg, x[r > 0 ? r - 1 : 0], 31);
        if (lane == 31 && r + 1 < R) up[r] = hi;
        if (lane == 0 && r > 0) dn[r] = lo;
    }
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        const int t = lane + 32 * r;
        if (t >= n) continue;
        const int kd = kind[t];
        const int p = kd == 1 ? t + 1 : kd == 2 ? t - 1 : t;
        const T off = kd == 1 ? e[t] : kd == 2 ? e[t - 1] : Ar<T>::one();
        x[r] = ldl_dsolve_row(kd, x[r], kd == 1 ? up[r] : dn[r], d(t, t), d(p, p), off);
    }
}

}  // namespace batchlas::sytrf_v2
