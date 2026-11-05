// SPDX-License-Identifier: MIT
#pragma once
// LAPACK ?sytrf on device (v22 design A.3): Bunch-Kaufman L D L^T, written once for the lower
// form. sytf2 is the unblocked factor; lasyf / lasyf_undo are one panel of the blocked factor
// and its last loop. Uplo::Upper runs the same steps on the index-reversed matrix (logical i is
// physical n-1-i): the reversal maps ?sytf2's upper loop onto its lower loop step for step, so
// the stored factor, D and ipiv are LAPACK's for either uplo. evidence:
// docs/perf/sytrf.md#sytrf-the-reversal-map
//
// Every item of the executor calls these with the same arguments. Each pivot decision is item
// 0's, broadcast (ex.uniform): a decision that differed between items (a NaN compare, items
// aliased by a short stride) would deadlock the next barrier. evidence:
// docs/perf/sytrf.md#sytrf-termination
//
// The executor is a RtGroup<Group> (one work-group per item) or a SubGroup<32> (one sub-group of a
// wider work-group). Arithmetic is arith::Fma; a complex fma product is not commutative in the last
// bit, so each op states its operand order. Barriers order global memory too (the global-memory
// tiers factor in place). Tests: tests/device_blas_v2_sytrf-sytrs_tests.cc.

#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

namespace batchlas::device::inline v2 {

namespace sytrf_detail {

// Logical (i, j) of an m x n matrix; with rev, physical (m - 1 - i, n - 1 - j). The Upper
// triangle of A is the Lower triangle of its reversal, so one Lower loop serves both uplo.
// Pending core: flip<axis::Both> lacks value_type and a 64-bit ld.
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

// ---- deterministic group reductions -----------------------------------------------------------
// The 32-lane butterfly, then with more than one sub-group the sub-group results in sub-group
// order through caller scratch (width / 32 slots, two barriers): the same value on every item,
// whatever the width. Pending core: allreduce / argmax over RtGroup<Group> (run-time nteams), a
// run-time tie rule and nan::AsInf; until then the cross-sub-group fold stays here.
template <class R> struct ArgMaxScratch {
    R* v;
    int* i;
};
template <class R> struct SumScratch {
    R* re;
    R* im;  // complex only
};

// (ov, oi) beats (bv, bi): a larger value; on a tie the preferred index.
template <class R> BATCHLAS_DLA_HELPER bool argmax_beats(R ov, int oi, R bv, int bi, bool prefer_low) {
    return ov > bv || (ov == bv && (prefer_low ? oi < bi : oi > bi));
}

template <class Ex, class R> BATCHLAS_DLA_HELPER void argmax_reduce(const Ex& ex, ArgMaxScratch<R> scr, R& v, int& idx, bool prefer_low) {
    auto sg = ex.sub();
    const ArgMax<R> m = butterfly<1, 32>(sg, ArgMax<R>{idx, v}, [&](const ArgMax<R>& a, const ArgMax<R>& o) {
        return argmax_beats(o.value, o.index, a.value, a.index, prefer_low) ? o : a;
    });
    v = m.value;
    idx = m.index;
    const int nsg = detail::size_of(ex) / 32;
    if (nsg == 1) return;
    if (sg.leader()) {
        scr.v[sg.get_group_linear_id()] = v;
        scr.i[sg.get_group_linear_id()] = idx;
    }
    sync<scope::Group>(ex, Mem::Global);
    v = scr.v[0];
    idx = scr.i[0];
    for (int s = 1; s < nsg; ++s)
        if (argmax_beats(scr.v[s], scr.i[s], v, idx, prefer_low)) {
            v = scr.v[s];
            idx = scr.i[s];
        }
    sync<scope::Group>(ex, Mem::Global);
}

template <class Ex, class R> BATCHLAS_DLA_HELPER R sum_reduce_real(const Ex& ex, R v, R* slots) {
    auto sg = ex.sub();
    v = butterfly<1, 32>(sg, v, sycl::plus<R>());
    const int nsg = detail::size_of(ex) / 32;
    if (nsg == 1) return v;
    if (sg.leader()) slots[sg.get_group_linear_id()] = v;
    sync<scope::Group>(ex, Mem::Global);
    v = slots[0];
    for (int s = 1; s < nsg; ++s) v += slots[s];
    sync<scope::Group>(ex, Mem::Global);
    return v;
}
template <class Ex, class T, class R> BATCHLAS_DLA_HELPER T sum_reduce(const Ex& ex, SumScratch<R> scr, T v) {
    using A = arith_t<arith::Fma, T>;
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
BATCHLAS_DLA_INLINE ArgMax<R> iamax_bk(const Ex& ex, ArgMaxScratch<R> scr, const X& x, Range r, bool prefer_low, int skip = -1) {
    R cv = R(-1);
    int ci = -1;
    for_each_owned(ex, r, [&](int i) BATCHLAS_DLA_BODY {
        if constexpr (Skip) {
            if (i == skip) return;
        }
        const R v = nan_as_inf(abs1(x(i)));
        if (argmax_beats(v, i, cv, ci, prefer_low)) {
            cv = v;
            ci = i;
        }
    });
    argmax_reduce(ex, scr, cv, ci, prefer_low);
    return {ci, cv};
}

// ---- Bunch-Kaufman ----------------------------------------------------------------------------
// ?SYTF2's first test, item 0's verdict to every item: a zero (or NaN) column, a 1 x 1 pivot
// without interchange, or the row search. ?SYTF2 and ?LASYF share both tests.
inline constexpr int kBkNeedRow = 0;
inline constexpr int kBkZero = 1;
inline constexpr int kBkKeep = 2;

template <class R> BATCHLAS_DLA_HELPER R bk_alpha() { return (R(1) + sycl::sqrt(R(17))) / R(8); }

template <class Ex, class R> BATCHLAS_DLA_HELPER int bk_stage(const Ex& ex, R absakk, R colmax, R alpha) {
    return ex.uniform(sycl::fmax(absakk, colmax) == R(0) || sycl::isnan(absakk) ? kBkZero
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
    const int code = ex.uniform(kp * 2 + kstep - 1);
    kp = code >> 1;
    kstep = (code & 1) + 1;
}

// The 2 x 2 pivot [[akk, ak1k], [ak1k, ak1k1]] of ?SYTF2 / ?LASYF, inverted LAPACK's way:
// d11 = ak1k1 / d21, d22 = akk / d21, e21 = (1 / (d11 d22 - 1)) / d21.
template <class T> struct BkBlock2 {
    using A = arith_t<arith::Fma, T>;
    T d11, d22, e21;
    // The two entries of the row [xk xk1] D^-1.
    BATCHLAS_DLA_INLINE T first(T xk, T xk1) const { return A::mul(e21, A::sub(A::mul(d11, xk), xk1)); }
    BATCHLAS_DLA_INLINE T second(T xk, T xk1) const { return A::mul(e21, A::sub(A::mul(d22, xk1), xk)); }
};
template <class T> BATCHLAS_DLA_INLINE BkBlock2<T> bk_block2(T akk, T ak1k, T ak1k1) {
    using A = arith_t<arith::Fma, T>;
    const T d21 = ak1k;
    const T d11 = A::div(ak1k1, d21);
    const T d22 = A::div(akk, d21);
    const T t = A::recip(A::sub(A::mul(d11, d22), A::one()));
    return {d11, d22, A::div(t, d21)};
}

// ?lasyf's W, nb + 1 columns: logical row i at physical row map(i), panel column c at physical
// column c (Lower) or ncol-1-c (Upper), so W's physical column order is the order of the panel's
// physical columns in A and the trailing gemm takes both as plain sub-views.
template <class D> BATCHLAS_DLA_HELPER RevMat<D> w_tile(D* p, std::ptrdiff_t ld, int n, int ncol, bool rev) {
    return rev_mat(p, ld, n, ncol, rev);
}

}  // namespace sytrf_detail

// ?sytf2, lower form, on the logical tile. `piv_out(phys_k, value)` stores LAPACK's ipiv entry
// (1-based; a 2x2 pivot writes -(p+1) at both of its positions); item 0 calls it. Returns info.
template <typename D, typename R, typename Ex, typename PivOut>
inline int sytf2(const sytrf_detail::RevMat<D>& S, int n, bool rev, const Ex& g, sytrf_detail::ArgMaxScratch<R> am,
                 PivOut&& piv_out) {
    namespace d = sytrf_detail;
    using A = arith_t<arith::Fma, D>;
    const R alpha = d::bk_alpha<R>();
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    int info = 0;
    int k = 0;
    while (k < n) {
        int kstep = 1;
        int kp = k;
        const R absakk = abs1(S(k, k));
        // ?sytf2's I?AMAX takes the first maximum in storage order: the smallest physical row,
        // which for the reversed (upper) form is the largest logical one.
        const auto col = d::iamax_bk(g, am, [&](int i) -> D& { return S(i, k); }, Range{k + 1, n}, !rev);
        const R colmax = col.value < R(0) ? R(0) : col.value;
        const int imax = col.index;
        const int stage = d::bk_stage(g, absakk, colmax, alpha);
        if (stage == d::kBkNeedRow) {
            const auto row = d::iamax_bk<true>(
                g, am, [&](int j) -> D& { return j > imax ? S(j, imax) : S(imax, j); }, Range{k, n}, true, imax);
            d::bk_choice(g, absakk, colmax, row.value, abs1(S(imax, imax)), alpha, k, imax, kp, kstep);
        }
        if (stage == d::kBkZero) {
            if (info == 0) info = phys(k) + 1;
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {  // the symmetric interchange of kk and kp
                for_each_owned(g, Range{kp + 1, n}, [&](int i) BATCHLAS_DLA_BODY { std::swap(S(i, kk), S(i, kp)); });
                for_each_owned(g, Range{kk + 1, kp}, [&](int j) BATCHLAS_DLA_BODY { std::swap(S(j, kk), S(kp, j)); });
                if (g.leader()) {
                    std::swap(S(kk, kk), S(kp, kp));
                    if (kstep == 2) std::swap(S(k + 1, k), S(kp, k));
                }
            }
            sync<scope::Group>(g, Mem::Global);
            if (kstep == 1) {  // A22 -= A21 d^-1 A21^T (?syr), then L21 = A21 d^-1
                const D mr1 = A::neg(A::recip(S(k, k)));
                for (int j = k + 1; j < n; ++j) {
                    const D t = A::mul(mr1, S(j, k));
                    for_each_owned(g, Range{j, n}, [&](int i) BATCHLAS_DLA_BODY { S(i, j) = A::add(S(i, j), A::mul(S(i, k), t)); });
                }
                sync<scope::Group>(g, Mem::Global);
                const D r1 = A::neg(mr1);
                for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY { S(i, k) = A::mul(r1, S(i, k)); });
            } else if (k < n - 2) {  // A22 -= A21 D^-1 A21^T, then L21 = A21 D^-1
                const auto blk = d::bk_block2(S(k, k), S(k + 1, k), S(k + 1, k + 1));
                for (int j = k + 2; j < n; ++j) {
                    const D a = blk.first(S(j, k), S(j, k + 1)), b = blk.second(S(j, k), S(j, k + 1));
                    for_each_owned(g, Range{j, n}, [&](int i) BATCHLAS_DLA_BODY {
                        S(i, j) = A::sub(A::sub(S(i, j), A::mul(S(i, k), a)), A::mul(S(i, k + 1), b));
                    });
                }
                sync<scope::Group>(g, Mem::Global);
                for_each_owned(g, Range{k + 2, n}, [&](int i) BATCHLAS_DLA_BODY {
                    const D a = blk.first(S(i, k), S(i, k + 1)), b = blk.second(S(i, k), S(i, k + 1));
                    S(i, k) = a;
                    S(i, k + 1) = b;
                });
            }
        }
        if (g.leader()) {
            if (kstep == 1) {
                piv_out(phys(k), phys(kp) + 1);
            } else {
                piv_out(phys(k), -(phys(kp) + 1));
                piv_out(phys(k + 1), -(phys(kp) + 1));
            }
        }
        sync<scope::Group>(g, Mem::Global);
        k += kstep;
    }
    return info;
}

// ?lasyf, lower form, on the uniform panel [k0, k0 + nb) of the logical matrix, starting at
// `start` (k0, or k0 + 1 when the previous panel's last \f$2 \times 2\f$ block took column k0). The step at
// the panel's last column is always taken, so a \f$2 \times 2\f$ block may take column k0 + nb (W's column nb):
// the panel ends at k0 + nb or k0 + nb + 1, and the trailing update has one uniform rank, nb + 1.
// W's columns the item did not compute are zero. Returns the end; *info as sytf2's.
// evidence: docs/perf/sytrf.md#sytrf-the-blocked-route
// The loops stay hand-written: as v2 ops the W zero-fill unrolled fully at device link and the
// column update moved loads and FMAs (complex<float> registers 68 -> 128).
template <typename D, typename R, typename Ex, typename PivOut>
inline int lasyf(const sytrf_detail::RevMat<D>& S, const sytrf_detail::RevMat<D>& W, int n, int k0, int nb, int start, bool rev,
                 const Ex& g, sytrf_detail::ArgMaxScratch<R> am, int& info, PivOut&& piv_out) {
    namespace d = sytrf_detail;
    using A = arith_t<arith::Fma, D>;
    const R alpha = d::bk_alpha<R>();
    const int lid = g.lid(), nl = detail::size_of(g);
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    for (int i = k0 + lid; i < n; i += nl)
        for (int c = 0; c <= nb; ++c) W(i, c) = A::zero();
    sync<scope::Group>(g, Mem::Global);
    // W(k:n, dst) := source - A(k:n, start:k) W(r, start-k0:k-k0): the panel's finished columns only.
    auto update_col = [&](int k, int dst, int r, auto&& source) {
        for (int i = k + lid; i < n; i += nl) {
            D acc = source(i);
            for (int l = start - k0; l < k - k0; ++l) acc = A::sub(acc, A::mul(S(i, k0 + l), W(r, l)));
            W(i, dst) = acc;
        }
        sync<scope::Group>(g, Mem::Global);
    };
    int k = start;
    while (k < k0 + nb) {
        const int c = k - k0;
        update_col(k, c, k, [&](int i) { return S(i, k); });
        int kstep = 1;
        int kp = k;
        const R absakk = abs1(W(k, c));
        R cv = R(-1);
        int ci = -1;
        for (int i = k + 1 + lid; i < n; i += nl) {
            const R v = d::nan_as_inf(abs1(W(i, c)));
            if (d::argmax_beats(v, i, cv, ci, !rev)) { cv = v; ci = i; }
        }
        d::argmax_reduce(g, am, cv, ci, !rev);
        const R colmax = cv < R(0) ? R(0) : cv;
        const int imax = ci;
        const int stage = d::bk_stage(g, absakk, colmax, alpha);
        if (stage == d::kBkNeedRow) {
            update_col(k, c + 1, imax, [&](int i) { return i < imax ? S(imax, i) : S(i, imax); });
            R rv = R(-1);
            int ri = -1;
            for (int j = k + lid; j < n; j += nl) {
                if (j == imax) continue;
                const R v = d::nan_as_inf(abs1(W(j, c + 1)));
                if (d::argmax_beats(v, j, rv, ri, true)) { rv = v; ri = j; }
            }
            d::argmax_reduce(g, am, rv, ri, true);
            d::bk_choice(g, absakk, colmax, rv, abs1(W(imax, c + 1)), alpha, k, imax, kp, kstep);
            if (kstep == 1 && kp != k)
                for (int i = k + lid; i < n; i += nl) W(i, c) = W(i, c + 1);
            sync<scope::Group>(g, Mem::Global);
        }
        if (stage == d::kBkZero) {
            if (info == 0) info = phys(k) + 1;
            for (int i = k + lid; i < n; i += nl) S(i, k) = W(i, c);
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {
                // The non-updated column kk moves to kp; the panel's finished columns and W swap
                // rows kk and kp (lasyf_undo restores the columns' ?sytf2 form after the update).
                if (lid == 0) S(kp, kp) = S(kk, kk);
                for (int j = kk + 1 + lid; j < kp; j += nl) S(kp, j) = S(j, kk);
                for (int i = kp + 1 + lid; i < n; i += nl) S(i, kp) = S(i, kk);
                for (int l = start + lid; l < kk; l += nl) std::swap(S(kk, l), S(kp, l));
                for (int l = start - k0 + lid; l <= kk - k0; l += nl) std::swap(W(kk, l), W(kp, l));
            }
            sync<scope::Group>(g, Mem::Global);
            if (kstep == 1) {
                const D r1 = A::recip(W(k, c));
                for (int i = k + lid; i < n; i += nl) S(i, k) = i == k ? W(k, c) : A::mul(r1, W(i, c));
            } else {
                const auto blk = d::bk_block2(W(k, c), W(k + 1, c), W(k + 1, c + 1));
                for (int j = k + 2 + lid; j < n; j += nl) {
                    const D wk = W(j, c), wk1 = W(j, c + 1);
                    S(j, k) = blk.first(wk, wk1);
                    S(j, k + 1) = blk.second(wk, wk1);
                }
                if (lid == 0) {
                    S(k, k) = W(k, c);
                    S(k + 1, k) = W(k + 1, c);
                    S(k + 1, k + 1) = W(k + 1, c + 1);
                }
            }
        }
        if (lid == 0) {
            if (kstep == 1) {
                piv_out(phys(k), phys(kp) + 1);
            } else {
                piv_out(phys(k), -(phys(kp) + 1));
                piv_out(phys(k + 1), -(phys(kp) + 1));
            }
        }
        sync<scope::Group>(g, Mem::Global);
        k += kstep;
    }
    // The last step's row search may have left column imax in W(:, nb) though the block was \f$1 \times 1\f$:
    // the uniform-rank gemm reads that column, so it must be zero unless a \f$2 \times 2\f$ block owns it.
    if (k == k0 + nb)
        for (int i = k0 + lid; i < n; i += nl) W(i, nb) = A::zero();
    return k;
}

// ?lasyf's last loop: undo the panel's interchanges on its own finished columns [start, end), in
// reverse, so they hold ?sytf2's form. `logical_partner(j)` is ipiv's partner of logical column j
// (rows j / j-1 for a 2 x 2 block ending at j) and its sign. Runs after the trailing update.
template <typename D, typename Ex, typename PivIn>
inline void lasyf_undo(const sytrf_detail::RevMat<D>& S, int start, int end, const Ex& g, PivIn&& logical_partner) {
    const int lid = g.lid(), nl = detail::size_of(g);
    int j = end - 1;
    while (j >= start) {
        const int jj = j;
        bool two = false;
        const int jp = logical_partner(j, two);
        if (two) --j;
        --j;
        if (jp != jj && j >= start)
            for (int l = start + lid; l <= j; l += nl) std::swap(S(jp, l), S(jj, l));
        sync<scope::Group>(g, Mem::Global);
    }
}

}  // namespace batchlas::device
