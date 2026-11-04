// SPDX-License-Identifier: MIT
#pragma once
// LAPACK ?sytrs / ?sytrs2 on device (v22 design A.3), from sytrf's Bunch-Kaufman factor in the
// logical lower form (lapack/sytrf.hh's reversal). sytrs_col is ?sytrs on one right-hand side.
// The converted form (?sytrs2) first runs ?syconv (each step's interchange applied to the earlier
// columns of L, the \f$2 \times 2\f$ off-diagonals moved out), after which \f$A = P L D L^{T} P^{T}\f$
// with L unit lower and P one permutation; sytrs_cols then solves one right-hand side per
// sub-group with lane i owning logical rows i, i + 32, ... in a lane-cyclic Frag: the gather,
// both sweeps (broadcast form, no reduction) and the scatter take no barrier.
// evidence: docs/perf/sytrf.md#sytrs-the-converted-solve

#include <cstddef>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/sytrf.hh>

namespace batchlas::device::inline v2 {

namespace sytrs_detail {

// Row t of D^-1 x for a block of kind kd (0: 1 x 1; 1 / 2: the first / second row of a 2 x 2):
// x is row t's value, xp its partner's, dtt and dpp their diagonals, off the block's
// off-diagonal. ?SYTRS's arithmetic, so both rows of a block agree with sytrs_col's.
template <class T> BATCHLAS_DLA_HELPER T ldl_dsolve_row(int kd, T x, T xp, T dtt, T dpp, T off) {
    using A = arith_t<arith::Fma, T>;
    if (kd == 0) return A::mul(x, A::recip(dtt));
    const T akm1 = A::div(kd == 1 ? dtt : dpp, off);
    const T ak = A::div(kd == 1 ? dpp : dtt, off);
    const T denom = A::sub(A::mul(akm1, ak), A::one());
    const T bkm1 = A::div(kd == 1 ? x : xp, off);
    const T bk = A::div(kd == 1 ? xp : x, off);
    return kd == 1 ? A::div(A::sub(A::mul(ak, bkm1), bk), denom) : A::div(A::sub(A::mul(akm1, bk), bkm1), denom);
}

// The staged factor in local memory, logical indices. `ps` is the logical ipiv: +(p + 1) a \f$1 \times 1\f$
// step with partner p, -(p + 1) both rows of a \f$2 \times 2\f$ step whose second row is swapped with p, 0
// for an entry sytrf cannot have written. `perm[t]`: the row of B that \f$P^{T}\f$ brings to row t.
// `kind[t]`: 0 a \f$1 \times 1\f$ block, 1 / 2 the first / second row of a \f$2 \times 2\f$ block. `e[k]`: the
// off-diagonal of the \f$2 \times 2\f$ block starting at k.
template <typename D>
struct Staged {
    using value_type = D;
    D* L;
    int ldt;
    int n;
    D* e;
    int* ps;
    int* perm;
    int* kind;
    int rows() const { return n; }
    int cols() const { return n; }
    D& operator()(int i, int j) const { return L[i + static_cast<std::ptrdiff_t>(j) * ldt]; }
};

inline int logical_piv(int raw, int n, bool rev) {
    if (raw == 0 || raw < -n || raw > n) return 0;
    const int p = (raw > 0 ? raw : -raw) - 1;
    const int lp = rev ? n - 1 - p : p;
    return raw > 0 ? lp + 1 : -(lp + 1);
}

// sytrs_col's check: an entry out of [1, n] or an unpaired \f$2 \times 2\f$ entry is a list sytrf cannot write.
inline bool piv_list_valid(const int* ps, int n) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v == 0 || (v < 0 && (k + 1 >= n || ps[k + 1] != v))) return false;
        k += v > 0 ? 1 : 2;
    }
    return true;
}

// The interchanges composed into one gather (y[t] = b[perm[t]]), and each row's block kind.
inline void build_perm(const int* ps, int n, int* perm, int* kind) {
    for (int t = 0; t < n; ++t) perm[t] = t;
    for (int k = 0; k < n;) {
        const int v = ps[k];
        const int s = v > 0 ? k : k + 1;
        const int p = (v > 0 ? v : -v) - 1;
        const int tmp = perm[s];
        perm[s] = perm[p];
        perm[p] = tmp;
        kind[k] = v > 0 ? 0 : 1;
        if (v < 0) kind[k + 1] = 2;
        k += v > 0 ? 1 : 2;
    }
}

// x := op(L)^-1 x for L unit lower n x n (the strict lower triangle read) on a 32-lane executor,
// x lane-cyclic (lane l holds row l + 32 r in x[r]); NoTrans by columns, Trans by rows, each step's
// element by frag_bcast: no reduction, no barrier. Collective. Pending core: trsv<alg::LanePerRow>
// over a multi-register lane-cyclic Frag with the sub(mul) update.
template <Transpose TR, class Ex, class ML, class T, int R>
BATCHLAS_DLA_INLINE void trsv_frag(const Ex& ex, const ML& l, int n, Frag<T, R, Cyclic>& x) {
    static_assert(Ex::size == 32, "sytrs trsv_frag: a 32-lane executor");
    using A = arith_t<arith::Fma, T>;
    const int lane = ex.lane();
    if constexpr (TR == Transpose::NoTrans) {
        for (int k = 0; k + 1 < n; ++k) {
            const T xk = frag_bcast(ex, x, k);
            BATCHLAS_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t > k && t < n) x[r] = A::sub(x[r], A::mul(l(t, k), xk));
            }
        }
    } else {
        static_assert(TR == Transpose::Trans, "sytrs trsv_frag: NoTrans or Trans");
        for (int i = n - 1; i > 0; --i) {
            const T xi = frag_bcast(ex, x, i);
            BATCHLAS_UNROLL_FULL
            for (int r = 0; r < R; ++r) {
                const int t = lane + 32 * r;
                if (t < i) x[r] = A::sub(x[r], A::mul(l(i, t), xi));
            }
        }
    }
}

// x := D^-1 x for D block diagonal with 1 x 1 / 2 x 2 blocks: kind[t] as ldl_dsolve_row's, the
// diagonal on d(t, t), e[k] the off-diagonal of the block starting at k. The partner rows come
// from neighbor_exchange.
template <class Ex, class MD, class T, int R>
BATCHLAS_DLA_INLINE void ldl_dsolve_frag(const Ex& ex, const MD& d, const int* kind, const T* e, int n, Frag<T, R, Cyclic>& x) {
    static_assert(Ex::size == 32, "sytrs ldl_dsolve_frag: a 32-lane executor");
    Frag<T, R, Cyclic> up, dn;
    neighbor_exchange(ex, x, up, dn);
    const int lane = ex.lane();
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        const int t = lane + 32 * r;
        if (t >= n) continue;
        const int kd = kind[t];
        const int p = kd == 1 ? t + 1 : kd == 2 ? t - 1 : t;
        const T off = kd == 1 ? e[t] : kd == 2 ? e[t - 1] : arith_t<arith::Fma, T>::one();
        x[r] = ldl_dsolve_row(kd, x[r], kd == 1 ? up[r] : dn[r], d(t, t), d(p, p), off);
    }
}

}  // namespace sytrs_detail

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
                l(j + 1, j) = arith_t<arith::Fma, T>::zero();
            }
            if (k > j) std::swap(l(k + 1, j), l(-v - 1, j));
            k += 2;
        }
    }
}

// ?sytrs for one right-hand side, lower form, on the unconverted factor. `piv(phys_k)` reads
// LAPACK's ipiv entry. A list sytrf cannot have written (|ipiv| outside [1, n], an unpaired
// \f$2 \times 2\f$ entry) would index b out of bounds: that item's X is NaN instead.
template <typename D, typename R, typename Ex, typename PivIn>
inline void sytrs_col(const sytrf_detail::RevMat<D>& S, int n, bool rev, const sytrf_detail::RevVec<D>& b, const Ex& g,
                      sytrf_detail::SumScratch<R> sum, PivIn&& piv) {
    using A = arith_t<arith::Fma, D>;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    // Every lane walks the list, lane 0 decides.
    bool ok = true;
    for (int k = 0; k < n && ok;) {
        const int raw = piv(phys(k));
        ok = raw != 0 && raw >= -n && raw <= n && (raw > 0 || (k + 1 < n && piv(phys(k + 1)) == raw));
        k += raw > 0 ? 1 : 2;
    }
    if (g.uniform(ok ? 1 : 0) == 0) {
        for_each_owned(g, n, [&](int i) BATCHLAS_DLA_BODY { b(i) = qnan<D>(); });
        sync<scope::Group>(g, Mem::Global);
        return;
    }
    // P L y = b, then D z = y
    int k = 0;
    while (k < n) {
        const int raw = piv(phys(k));
        if (raw > 0) {
            const int kp = phys(raw - 1);
            if (g.leader() && kp != k) std::swap(b(k), b(kp));
            sync<scope::Group>(g, Mem::Global);
            const D bk = b(k);
            sync<scope::Group>(g, Mem::Global);
            for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY { b(i) = A::sub(b(i), A::mul(S(i, k), bk)); });
            if (g.leader()) b(k) = sytrs_detail::ldl_dsolve_row(0, bk, bk, S(k, k), S(k, k), S(k, k));
            sync<scope::Group>(g, Mem::Global);
            k += 1;
        } else {
            const int kp = phys(-raw - 1);
            if (g.leader() && kp != k + 1) std::swap(b(k + 1), b(kp));
            sync<scope::Group>(g, Mem::Global);
            const D bk = b(k), bk1 = b(k + 1);
            sync<scope::Group>(g, Mem::Global);
            for_each_owned(g, Range{k + 2, n}, [&](int i) BATCHLAS_DLA_BODY {
                b(i) = A::sub(A::sub(b(i), A::mul(S(i, k), bk)), A::mul(S(i, k + 1), bk1));
            });
            if (g.leader()) {
                const D off = S(k + 1, k), d0 = S(k, k), d1 = S(k + 1, k + 1);
                b(k) = sytrs_detail::ldl_dsolve_row(1, bk, bk1, d0, d1, off);
                b(k + 1) = sytrs_detail::ldl_dsolve_row(2, bk1, bk, d1, d0, off);
            }
            sync<scope::Group>(g, Mem::Global);
            k += 2;
        }
    }
    // L^T P^T x = z
    k = n - 1;
    while (k >= 0) {
        const int raw = piv(phys(k));
        D p0 = A::zero(), p1 = A::zero();
        for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY {
            p0 = A::add(p0, A::mul(S(i, k), b(i)));
            if (raw < 0) p1 = A::add(p1, A::mul(S(i, k - 1), b(i)));
        });
        const D s0 = sytrf_detail::sum_reduce(g, sum, p0);
        const D s1 = raw < 0 ? sytrf_detail::sum_reduce(g, sum, p1) : A::zero();
        sync<scope::Group>(g, Mem::Global);
        if (g.leader()) {
            b(k) = A::sub(b(k), s0);
            if (raw < 0) b(k - 1) = A::sub(b(k - 1), s1);
            const int kp = phys((raw > 0 ? raw : -raw) - 1);
            if (kp != k) std::swap(b(k), b(kp));
        }
        sync<scope::Group>(g, Mem::Global);
        k -= raw > 0 ? 1 : 2;
    }
}

// Validation, ?syconv and the permutation for one item on the work-group `ex`.
// An invalid list leaves L as staged, perm the identity and every kind 1 x 1; the caller writes NaN.
template <typename D, typename Acc, typename Ex>
inline bool syconv(const Acc& L, int n, const int* ps, int* perm, int* kind, D* e, const Ex& ex, int* flag) {
    if (ex.leader()) *flag = sytrs_detail::piv_list_valid(ps, n) ? 1 : 0;
    sync<scope::Group>(ex, Mem::Global);
    const bool ok = *flag != 0;
    if (ok) {
        for_each_owned(ex, n, [&](int j) BATCHLAS_DLA_BODY { syconv_col<D>(L, ps, n, j, e); });
        if (ex.leader()) sytrs_detail::build_perm(ps, n, perm, kind);
    } else {
        for_each_owned(ex, n, [&](int t) BATCHLAS_DLA_BODY {
            perm[t] = t;
            kind[t] = 0;
        });
    }
    sync<scope::Group>(ex, Mem::Global);
    return ok;
}

// ?syconv as a gather, for a factor in global memory: column j of W is column j of G with its rows
// permuted by \f$\sigma_j\f$, the interchanges after step j composed (\f$W(t, j) = G(\sigma_j(t), j)\f$, \f$t > j\f$).
// Walking j down, \f$\sigma\f$ gains one interchange per block start, applied by lane 0 in \f$O(1)\f$ through
// its inverse; every read and write of a column is then one pass instead of one swap per step.
// kind must be build_perm's for a valid list.
template <typename D, typename Dst, typename Src, typename Ex>
inline void syconv_gather(const Dst& W, const Src& G, int n, const int* ps, const int* kind, int* sig, int* inv,
                          const Ex& ex) {
    for_each_owned(ex, n, [&](int t) BATCHLAS_DLA_BODY {
        sig[t] = t;
        inv[t] = t;
    });
    sync<scope::Group>(ex, Mem::Global);
    for (int j = n - 1; j >= 0; --j) {
        const bool two = kind[j] == 1;
        for_each_owned(ex, Range{j + 1, n}, [&](int t) BATCHLAS_DLA_BODY {
            W(t, j) = two && t == j + 1 ? arith_t<arith::Fma, D>::zero() : G(sig[t], j);
        });
        sync<scope::Group>(ex, Mem::Global);
        if (ex.leader() && kind[j] != 2) {
            const int a = two ? j + 1 : j, b = (ps[j] > 0 ? ps[j] : -ps[j]) - 1;
            if (a != b) {
                const int ia = inv[a], ib = inv[b];
                sig[ia] = b;
                sig[ib] = a;
                inv[a] = ib;
                inv[b] = ia;
            }
        }
        sync<scope::Group>(ex, Mem::Global);
    }
}

// L w = y, w := D^-1 w, L^T z = w on one 32-lane executor; x[r] holds logical row lane + 32 r.
// Collective.
template <class Ex, typename D, int R>
inline void sytrs_regs(const Ex& ex, const sytrs_detail::Staged<D>& S, Frag<D, R, Cyclic>& x) {
    sytrs_detail::trsv_frag<Transpose::NoTrans>(ex, S, S.n, x);
    sytrs_detail::ldl_dsolve_frag(ex, S, S.kind, S.e, S.n, x);
    sytrs_detail::trsv_frag<Transpose::Trans>(ex, S, S.n, x);
}

// Every right-hand side of one item, sub-group `sgid` of `nsg` taking columns sgid, sgid + nsg, ...
// Row t is gathered from and scattered back to the same physical row of B, map(perm[t]), so no
// lane reads what another writes. An invalid list (ok false) gives an all-NaN X.
template <int R, typename D>
inline void sytrs_cols(const sycl::sub_group& sg, int lane, int sgid, int nsg, const sytrs_detail::Staged<D>& S, bool ok, bool rev,
                       D* bb, std::ptrdiff_t ldb, int nrhs) {
    const SubGroup<32> ex{sg};
    const int n = S.n;
    for (int c = sgid; c < nrhs; c += nsg) {
        D* const col = bb + static_cast<std::ptrdiff_t>(c) * ldb;
        if (!ok) {
            for_each_owned(ex, n, [&](int t) BATCHLAS_DLA_BODY { col[t] = qnan<D>(); });
            continue;
        }
        Frag<D, R, Cyclic> x;
        int row[R];
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r) {
            const int t = lane + 32 * r;
            row[r] = t < n ? (rev ? n - 1 - S.perm[t] : S.perm[t]) : 0;
            x[r] = t < n ? col[row[r]] : arith_t<arith::Fma, D>::zero();
        }
        sytrs_regs(ex, S, x);
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r)
            if (lane + 32 * r < n) col[row[r]] = x[r];
    }
}

}  // namespace batchlas::device
