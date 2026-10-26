#pragma once

// LAPACK ?sytrs2 on device: the factor is first converted (?syconv: each step's interchange applied
// to the earlier columns of L, the 2 x 2 off-diagonals moved out), after which A = P L D L^T P^T with
// L unit lower and P one permutation. Logical lower form throughout (sytrf_device.hh's reversal).
// The fused tiers solve one right-hand side per sub-group with lane i owning logical rows i, i + 32,
// ... in registers: the gather, both sweeps (broadcast form, no reduction) and the scatter take no
// barrier. The blocked tier converts into workspace and hands the sweeps to the public trsm.
// evidence: docs/perf/sytrf.md#sytrs-the-converted-solve

#include <src/ops/sytrf/sytrf_device.hh>
#include <src/sycl/kernel_attrs.hh>

#include <cstddef>

namespace batchlas::sytrf_device {

// The staged factor in local memory, logical indices. `ps` is the logical ipiv: +(p + 1) a 1 x 1
// step with partner p, -(p + 1) both rows of a 2 x 2 step whose second row is swapped with p, 0
// for an entry sytrf cannot have written. `perm[t]`: the row of B that P^T brings to row t.
// `kind[t]`: 0 a 1 x 1 block, 1 / 2 the first / second row of a 2 x 2 block. `e[k]`: the
// off-diagonal of the 2 x 2 block starting at k.
template <typename D>
struct Staged {
    D* L;
    int ldt;
    int n;
    D* e;
    int* ps;
    int* perm;
    int* kind;
    D& operator()(int i, int j) const { return L[i + static_cast<std::ptrdiff_t>(j) * ldt]; }
};

inline int logical_piv(int raw, int n, bool rev) {
    if (raw == 0 || raw < -n || raw > n) return 0;
    const int p = (raw > 0 ? raw : -raw) - 1;
    const int lp = rev ? n - 1 - p : p;
    return raw > 0 ? lp + 1 : -(lp + 1);
}

// ldl_solve's check: an entry out of [1, n] or an unpaired 2 x 2 entry is a list sytrf cannot write.
inline bool piv_list_valid(const int* ps, int n) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v == 0 || (v < 0 && (k + 1 >= n || ps[k + 1] != v))) return false;
        k += v > 0 ? 1 : 2;
    }
    return true;
}

// ?syconv on column j: the interchanges of the steps after j, in step order (a 2 x 2 step at k
// swaps row k + 1). The 2 x 2 off-diagonal of a block starting at j moves to e[j] (when e is set).
template <typename D, typename Acc>
inline void convert_column(const Acc& L, const int* ps, int n, int j, D* e) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v > 0) {
            if (k > j) swap(L(k, j), L(v - 1, j));
            k += 1;
        } else {
            if (k == j) {
                if (e != nullptr) e[j] = L(j + 1, j);
                L(j + 1, j) = zero<D>();
            }
            if (k > j) swap(L(k + 1, j), L(-v - 1, j));
            k += 2;
        }
    }
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

// Validation, conversion and the permutation for one item, `nl` lanes; `bar` is the group barrier.
// An invalid list leaves L as staged, perm the identity and every kind 1 x 1; the caller writes NaN.
template <typename D, typename Acc, typename Bar>
inline bool ldl_convert(const Acc& L, int n, const int* ps, int* perm, int* kind, D* e, int lid, int nl, int* flag,
                        Bar&& bar) {
    if (lid == 0) *flag = piv_list_valid(ps, n) ? 1 : 0;
    bar();
    const bool ok = *flag != 0;
    if (ok) {
        for (int j = lid; j < n; j += nl) convert_column<D>(L, ps, n, j, e);
        if (lid == 0) build_perm(ps, n, perm, kind);
    } else {
        for (int t = lid; t < n; t += nl) {
            perm[t] = t;
            kind[t] = 0;
        }
    }
    bar();
    return ok;
}

// Row t of D^-1 x for a block of kind kd: x is row t's value, xp its 2 x 2 partner's, dtt and dpp
// their diagonals, off the block's off-diagonal. ?sytrs's arithmetic, so both rows of a block agree.
template <typename D>
inline D d_solve_row(int kd, D x, D xp, D dtt, D dpp, D off) {
    using namespace sycl_device;
    if (kd == 0) return dev_mul(x, dev_recip(dtt));
    const D akm1 = dev_div(kd == 1 ? dtt : dpp, off);
    const D ak = dev_div(kd == 1 ? dpp : dtt, off);
    const D denom = dev_sub(dev_mul(akm1, ak), dev_one<D>());
    const D bkm1 = dev_div(kd == 1 ? x : xp, off);
    const D bk = dev_div(kd == 1 ? xp : x, off);
    return kd == 1 ? dev_div(dev_sub(dev_mul(ak, bkm1), bk), denom) : dev_div(dev_sub(dev_mul(akm1, bk), bkm1), denom);
}

// Logical row k of a register-resident vector, to every lane. Collective.
template <int R, typename D, typename SG>
inline D row_bcast(const SG& sg, const D (&x)[R], int k) {
    D v = x[0];
    BATCHLAS_UNROLL_FULL
    for (int r = 1; r < R; ++r)
        if ((k >> 5) == r) v = x[r];
    return batchlas::portable::group_broadcast(sg, v, k & 31);
}

// L w = y, w := D^-1 w, L^T z = w on one sub-group; x[r] holds logical row lane + 32 r. Collective.
template <int R, typename D, typename SG>
inline void ldl_solve_regs(const SG& sg, int lane, const Staged<D>& S, D (&x)[R]) {
    using namespace sycl_device;
    const int n = S.n;
    for (int k = 0; k + 1 < n; ++k) {
        const D xk = row_bcast<R>(sg, x, k);
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r) {
            const int t = lane + 32 * r;
            if (t > k && t < n) x[r] = dev_sub(x[r], dev_mul(S(t, k), xk));
        }
    }
    D up[R], dn[R];
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        up[r] = batchlas::portable::shift_group_left(sg, x[r], 1);
        dn[r] = batchlas::portable::shift_group_right(sg, x[r], 1);
        const D hi = batchlas::portable::group_broadcast(sg, x[r + 1 < R ? r + 1 : r], 0);
        const D lo = batchlas::portable::group_broadcast(sg, x[r > 0 ? r - 1 : 0], 31);
        if (lane == 31 && r + 1 < R) up[r] = hi;
        if (lane == 0 && r > 0) dn[r] = lo;
    }
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        const int t = lane + 32 * r;
        if (t >= n) continue;
        const int kd = S.kind[t];
        const int p = kd == 1 ? t + 1 : kd == 2 ? t - 1 : t;
        const D off = kd == 1 ? S.e[t] : kd == 2 ? S.e[t - 1] : dev_one<D>();
        x[r] = d_solve_row(kd, x[r], kd == 1 ? up[r] : dn[r], S(t, t), S(p, p), off);
    }
    for (int i = n - 1; i > 0; --i) {
        const D xi = row_bcast<R>(sg, x, i);
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r) {
            const int t = lane + 32 * r;
            if (t < i) x[r] = dev_sub(x[r], dev_mul(S(i, t), xi));
        }
    }
}

// Every right-hand side of one item, sub-group `sgid` of `nsg` taking columns sgid, sgid + nsg, ...
// Row t is gathered from and scattered back to the same physical row of B, map(perm[t]), so no
// lane reads what another writes. An invalid list (ok false) gives an all-NaN X.
template <int R, typename D, typename SG>
inline void solve_columns(const SG& sg, int lane, int sgid, int nsg, const Staged<D>& S, bool ok, bool rev, D* bb,
                          std::ptrdiff_t ldb, int nrhs) {
    const int n = S.n;
    for (int c = sgid; c < nrhs; c += nsg) {
        D* const col = bb + static_cast<std::ptrdiff_t>(c) * ldb;
        if (!ok) {
            for (int t = lane; t < n; t += 32) col[t] = nan_of<D>();
            continue;
        }
        D x[R];
        int row[R];
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r) {
            const int t = lane + 32 * r;
            row[r] = t < n ? (rev ? n - 1 - S.perm[t] : S.perm[t]) : 0;
            x[r] = t < n ? col[row[r]] : zero<D>();
        }
        ldl_solve_regs<R>(sg, lane, S, x);
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r)
            if (lane + 32 * r < n) col[row[r]] = x[r];
    }
}

}  // namespace batchlas::sytrf_device
