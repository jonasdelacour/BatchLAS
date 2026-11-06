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
#include <batchlas/device/level3/group.hh>

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
    // Member by member: AdaptiveCpp's Metal backend miscompiles a select of the whole struct.
    const ArgMax<R> m = butterfly<1, 32>(sg, ArgMax<R>{idx, v}, [&](const ArgMax<R>& a, const ArgMax<R>& o) {
        const bool take = argmax_beats(o.value, o.index, a.value, a.index, prefer_low);
        return ArgMax<R>{take ? o.index : a.index, take ? o.value : a.value};
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

// ---- public device API (v22 design C): sytrf<UL, Tier>(ex, A, ipiv, ws) -------------------------
// Tiers (one per kernel; with_tier picks one on the host for a run-time n):
//   tier::Cta<N>, N <= 32   A's logical lower triangle in local memory; sub-group 0 of ex factors
//   tier::Blocked<1>        any n: ?sytf2 in place in global memory (LAPACK's NB = 1, unblocked);
//                           ex at most 256 items (the arg-max scratch's eight sub-groups)
//   tier::Blocked<NB>, NB>1 any n: ?sytrf's blocked loop in one work-group (the host `blocked`
//                           route's steps): ?lasyf panels of NB columns, W (n x (NB + 1)) in the
//                           caller's global workspace, the trailing A22 -= L21 W21^T on the
//                           in-group L3 (sytrf_l3_cfg), ?sytf2 on the last n - end <= NB columns;
//                           ex = WorkGroup<256, 32>. Not in tiers_t (no measured win yet, design
//                           F6): name it, or pass with_tier a tier::list holding it. NB = 31 is
//                           the host route's (the update's k = 32).
// The host routes `cta` and `wg` run the same bodies (sytrf_cta_factor, sytf2) with uplo a
// run-time flag: the reversal map makes Upper an index transform, so one kernel per T serves
// both and the kernel names stay.

namespace routine {
template <Uplo UL> struct sytrf {
    static_assert(UL == Uplo::Lower || UL == Uplo::Upper, "sytrf: Uplo::Lower or Uplo::Upper");
    static constexpr const char* name = "sytrf";
};
}  // namespace routine

namespace sytrf_detail {

inline constexpr int sytrf_wg_max = 256;

template <int N, class T> struct SytrfCtaPlan {
    ws::Slot<T> tile;
    WorkspaceSize size;
};
// The Cta tier's (N + 1) x N tile: the host launch's odd ld.
template <int N, class T> constexpr SytrfCtaPlan<N, T> sytrf_cta_plan() {
    Plan p;
    const auto tile = p.take<T>(ws::Space::Local, static_cast<std::size_t>(N + 1) * N);
    return {tile, p.size()};
}
template <class R> struct SytrfWgPlan {
    ws::Slot<R> v;
    ws::Slot<int> i;
    WorkspaceSize size;
};
template <class R> constexpr SytrfWgPlan<R> sytrf_wg_plan() {
    Plan p;
    const auto v = p.take<R>(ws::Space::Local, sytrf_wg_max / 32);
    const auto i = p.take<int>(ws::Space::Local, sytrf_wg_max / 32);
    return {v, i, p.size()};
}

// Blocked<NB>'s L3 configuration: a function of T only (one kernel per T, never of n), 256 items.
template <class T> using sytrf_l3_cfg = l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;
inline constexpr int sytrf_blocked_wg = 256;
static_assert(sytrf_l3_cfg<float>::threads == sytrf_blocked_wg);

template <int NB, class T> struct SytrfBlockedPlan {
    ws::Slot<real_t<T>> v;
    ws::Slot<int> i;
    ws::Slot<T> smem;
    ws::Slot<T> w;  // global: ?lasyf's W, n x (NB + 1) at ld n
    WorkspaceSize size;
};
template <int NB, class T> constexpr SytrfBlockedPlan<NB, T> sytrf_blocked_plan(int n) {
    Plan p;
    const auto v = p.take<real_t<T>>(ws::Space::Local, sytrf_wg_max / 32);
    const auto i = p.take<int>(ws::Space::Local, sytrf_wg_max / 32);
    const auto sm = p.take<T>(ws::Space::Local, l3::smem_elems<sytrf_l3_cfg<T>>());
    const auto w = p.take<T>(ws::Space::Global, static_cast<std::size_t>(n) * (NB + 1));
    return {v, i, sm, w, p.size()};
}

// Whole sub-groups of 32 (a compile-time width a multiple of 32): the butterflies are 32-lane.
template <class Ex> constexpr bool sytrf_exec_ok() {
    if constexpr (!requires { Ex::sg_size; }) return false;
    else if constexpr (is_runtime_v<Ex>) return Ex::sg_size == 32;
    else return Ex::sg_size == 32 && Ex::size % 32 == 0;
}
// -1 when a run-time width is not whole sub-groups or exceeds Blocked<1>'s scratch, else 0.
template <class Tier, class Ex> BATCHLAS_DLA_INLINE int sytrf_exec_check(const Ex& ex) {
    constexpr bool wg = std::is_same_v<Tier, tier::Blocked<1>>;
    if constexpr (is_runtime_v<Ex>) {
        const int w = detail::size_of(ex);
        return w % 32 != 0 || (wg && w > sytrf_wg_max) ? -1 : 0;
    } else {
        static_assert(!wg || Ex::size <= sytrf_wg_max, "tier::Blocked<1>: an executor of at most 256 items");
        return 0;
    }
}
// Sub-group index and count inside ex: a whole-sub-group executor (SubGroup, Teams, a 32-item
// group) is one sub-group whatever its team() says.
template <class Ex> BATCHLAS_DLA_INLINE int sytrf_team(const Ex& ex) {
    if constexpr (Ex::sub_whole) return 0;
    else return ex.team();
}
template <class Ex> BATCHLAS_DLA_INLINE int sytrf_nteams(const Ex& ex) {
    if constexpr (Ex::sub_whole) return 1;
    else return detail::nteams_of(ex);
}
template <class MA> BATCHLAS_DLA_INLINE std::ptrdiff_t sytrf_ld(const MA& a) { return static_cast<int>(a.ld); }
// A dense column-major operand (A(i, j) an lvalue, a member ld) as the routine's logical view.
template <class MA> BATCHLAS_DLA_INLINE auto sytrf_view(const MA& a, bool rev) {
    return rev_mat(&a(0, 0), sytrf_ld(a), a.rows(), rev);
}
// LAPACK's argument numbers: -2 (N: A not square or outside the tier), -4 (LDA < max(1, n)).
template <class Tier, class Ex, class MA> BATCHLAS_DLA_INLINE int sytrf_check(const Ex& ex, const MA& a) {
    if (const int e = sytrf_exec_check<Tier>(ex); e != 0) return e;
    const int n = a.rows();
    if (a.cols() != n || !detail::bound_admits<Tier>(n)) return -2;
    if (sytrf_ld(a) < (n > 1 ? n : 1)) return -4;
    return 0;
}

// The Cta tier's factor (sytrf `cta`; the first half of sysv `cta`): every item of `all` copies
// the logical lower triangle of G into the local tile S; then the factoring sub-group g (the
// items with `factors`) runs ?sytf2 on S, copies the factor back to G, and its leader calls
// info_out(info). Returns ?sytf2's info on g, 0 elsewhere. No trailing barrier.
template <class All, class Sub, typename D, typename R, typename PivOut, typename InfoOut>
BATCHLAS_DLA_INLINE int sytrf_cta_factor(const All& all, const Sub& g, bool factors, const RevMat<D>& G, const RevMat<D>& S, int n,
                                         bool rev, ArgMaxScratch<R> am, PivOut&& piv_out, InfoOut&& info_out) {
    lacpy<Part::Lower, Transpose::NoTrans, alg::DivMod>(all, G, S);
    sync<scope::Group>(all, Mem::Global);
    int st = 0;
    if (factors) {
        st = sytf2(S, n, rev, g, am, piv_out);
        lacpy<Part::Lower, Transpose::NoTrans, alg::DivMod>(g, S, G);
        if (g.leader()) info_out(st);
    }
    return st;
}

template <Uplo UL, int N, class Ex, class MA>
BATCHLAS_DLA_INLINE int sytrf_run(tier::Cta<N>, const Ex& ex, const MA& A, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using R = real_t<D>;
    constexpr bool rev = UL == Uplo::Upper;
    constexpr auto pl = sytrf_cta_plan<N, D>();
    const int n = A.rows();
    const RevMat<D> S = rev_mat(carve_local(ws, pl.tile).get_raw(), N + 1, n, false);
    const SubGroup<32> g{ex.sub()};
    // One sub-group factors: its arg-max never reaches the cross-sub-group scratch.
    const int st = sytrf_cta_factor(ex, g, sytrf_team(ex) == 0, sytrf_view(A, rev), S, n, rev, ArgMaxScratch<R>{nullptr, nullptr},
                                    [&](int k, int v) { ipiv[k] = v; }, NoOp{});
    sync<scope::Group>(ex, Mem::Global);
    if constexpr (Ex::sub_whole) return st;
    else return ex.uniform(st);
}

template <Uplo UL, class Ex, class MA>
BATCHLAS_DLA_INLINE int sytrf_run(tier::Blocked<1>, const Ex& ex, const MA& A, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using R = real_t<D>;
    constexpr bool rev = UL == Uplo::Upper;
    constexpr auto pl = sytrf_wg_plan<R>();
    const ArgMaxScratch<R> am{carve_local(ws, pl.v).get_raw(), carve_local(ws, pl.i).get_raw()};
    // sytf2's verdicts are uniform and each step ends on a barrier: info is uniform, A visible.
    return sytf2(sytrf_view(A, rev), A.rows(), rev, ex, am, [&](int k, int v) { ipiv[k] = v; });
}

// The host `blocked` route's step order in one work-group: each panel's ?lasyf, its trailing
// update, then (before the next panel) its ?lasyf_undo; [first, end) is the last panel's columns,
// the host's per-item state. The update has the panel's own rank, end - k0 (NB, or NB + 1 when a
// 2 x 2 block took column k0 + NB), on the logical trailing matrix [end, n): physically the lower
// triangle at (end, end), or for Upper the upper triangle at (0, 0) with the panel's columns and
// W's both reversed. evidence: docs/perf/sytrf.md#sytrf-the-blocked-route
template <Uplo UL, int NB, class Ex, class MA>
    requires(NB > 1)
BATCHLAS_DLA_INLINE int sytrf_run(tier::Blocked<NB>, const Ex& ex, const MA& A, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using R = real_t<D>;
    using Cfg = sytrf_l3_cfg<D>;
    using Ar = arith_t<typename Cfg::ops, D>;
    static_assert(!is_runtime_v<Ex> && Ex::size == Cfg::threads, "sytrf tier::Blocked<NB>: a WorkGroup<256, 32> executor");
    constexpr bool rev = UL == Uplo::Upper;
    constexpr Uplo tri = rev ? Uplo::Upper : Uplo::Lower;
    const int n = A.rows();
    const auto pl = sytrf_blocked_plan<NB, D>(n);
    const ArgMaxScratch<R> am{carve_local(ws, pl.v).get_raw(), carve_local(ws, pl.i).get_raw()};
    const l3::Smem<Cfg> sm{carve_local(ws, pl.smem).get_raw()};
    D* const w = carve_global(ws, pl.w);
    D* const a0 = &A(0, 0);
    const int lda = static_cast<int>(sytrf_ld(A));
    const RevMat<D> S = rev_mat(a0, lda, n, rev);
    const RevMat<D> W = w_tile(w, n, n, NB + 1, rev);
    auto piv_out = [&](int k, int v) { ipiv[k] = v; };
    auto partner = [&](int j, bool& two) {
        const int raw = ipiv[rev ? n - 1 - j : j];
        two = raw < 0;
        const int p = (raw < 0 ? -raw : raw) - 1;
        return rev ? n - 1 - p : p;
    };
    const D neg_one = Ar::neg(Ar::one()), one = Ar::one();
    int info = 0, first = 0, end = 0;
    for (int k0 = 0; n - k0 > NB; k0 += NB) {
        lasyf_undo(S, first, end, ex, partner);
        const int e = lasyf(S, W, n, k0, NB, end, rev, ex, am, info, piv_out);
        first = end;
        end = e;
        const int m = n - e, rank = e - k0;
        if (m == 0) continue;
        const l3::Global<D> c{rev ? a0 : a0 + e + e * lda, lda, m, m};
        const D* const lp = rev ? a0 + (n - k0 - rank) * lda : a0 + e + k0 * lda;
        const D* const wp = rev ? w + (NB + 1 - rank) * n : w + e;
        const auto pass = l3::make_pass<true, false>(rank, [=](int r, int kk) BATCHLAS_DLA_BODY { return lp[r + kk * lda]; },
                                                      [=](int kk, int col) BATCHLAS_DLA_BODY { return wp[col + kk * n]; });
        l3::rank_update<Cfg, tri, false>(ex, sm, c, neg_one, one, pass);
        sync<scope::Group>(ex, Mem::Global);
    }
    lasyf_undo(S, first, end, ex, partner);
    const int sub = n - end;
    if (sub > 0) {  // ?sytf2 on the trailing sub-matrix; its ipiv and info shifted to A's indices
        const int off = rev ? 0 : end;
        const int t = sytf2(rev_mat(a0 + off + off * lda, lda, sub, rev), sub, rev, ex, am,
                            [&](int k, int v) { ipiv[k + off] = v > 0 ? v + off : v - off; });
        if (info == 0 && t > 0) info = t + off;
    }
    return info;
}

}  // namespace sytrf_detail

template <Uplo UL, class T> struct tiers<routine::sytrf<UL>, T> {
    using type = tier::list<tier::Cta<32>, tier::Blocked<1>>;
};
template <Uplo UL, int N, class T> struct tier_info<routine::sytrf<UL>, tier::Cta<N>, T> {
    static_assert(N <= 32, "sytrf tier::Cta<N>: N <= 32 (one sub-group factors)");
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int) { return sytrf_detail::sytrf_cta_plan<N, T>().size; }
};
template <Uplo UL, class T> struct tier_info<routine::sytrf<UL>, tier::Blocked<1>, T> {
    static constexpr int wg_size = sytrf_detail::sytrf_wg_max, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int) { return sytrf_detail::sytrf_wg_plan<real_t<T>>().size; }
};
template <Uplo UL, int NB, class T> struct tier_info<routine::sytrf<UL>, tier::Blocked<NB>, T> {
    static constexpr int wg_size = sytrf_detail::sytrf_blocked_wg, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n) { return sytrf_detail::sytrf_blocked_plan<NB, T>(n).size; }
};

// The workspace of one matrix: local bytes for the tier's tile / scratch; a global part only for
// Blocked<NB> (W), which the caller offsets per matrix (global_slice).
template <Uplo UL, class Tier, class T> constexpr WorkspaceSize sytrf_buffer_size(int n) {
    using Rt = routine::sytrf<UL>;
    return tier_info<Rt, resolve_tier_t<Rt, T, Tier>, T>::buffer_size(n);
}

// ?sytrf of one matrix inside a kernel: A := its Bunch-Kaufman factor in LAPACK's storage for UL,
// ipiv[0, n) := LAPACK's ipiv (1-based int32; a 2 x 2 block negative at both of its rows).
// Collective: every item of ex calls it with the same arguments, A and ipiv's memory visible to
// all on entry; both are visible to every item on return. Returns LAPACK's info, the same on
// every item: 0; k > 0 (D(k, k) is exactly zero; the factor is complete); -2 (A not square, or n
// outside the tier); -4 (ld < max(1, n)); -1 (ex: a run-time width that is not whole 32-wide
// sub-groups, or wider than Blocked<1>'s 256). The executor needs sub-group size 32. ws holds
// sytrf_buffer_size<UL, Tier, T>(n): its local bytes, and its global bytes for this matrix alone.
template <Uplo UL, class Tier, class Ex, class MA> inline int sytrf(const Ex& ex, const MA& A, int* ipiv, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using Tr = resolve_tier_t<routine::sytrf<UL>, D, Tier>;
    static_assert(sytrf_detail::sytrf_exec_ok<Ex>(), "sytrf: an executor of whole 32-wide sub-groups");
    const int chk = sytrf_detail::sytrf_check<Tr>(ex, A);
    if (chk != 0 || A.rows() == 0) return chk;
    return sytrf_detail::sytrf_run<UL>(Tr{}, ex, A, ipiv, ws);
}

}  // namespace batchlas::device
