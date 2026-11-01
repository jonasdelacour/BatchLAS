#pragma once

// LAPACK ?sytrs2 on device: the factor is first converted (?syconv: each step's interchange applied
// to the earlier columns of L, the 2 x 2 off-diagonals moved out), after which A = P L D L^T P^T with
// L unit lower and P one permutation. Logical lower form throughout (sytrf_device.hh's reversal).
// The fused tiers solve one right-hand side per sub-group with lane i owning logical rows i, i + 32,
// ... in registers: the gather, both sweeps (broadcast form, no reduction) and the scatter take no
// barrier. The blocked tier converts into workspace and hands the sweeps to the public trsm.
// The sweeps and ?syconv are device LA v2 ops (include/batchlas/blas/device/v2/ext/sytrf-sytrs.hh).
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

// Validation, ?syconv (v2::syconv_col per column) and the permutation for one item on `ex`.
// An invalid list leaves L as staged, perm the identity and every kind 1 x 1; the caller writes NaN.
template <typename D, typename Acc>
inline bool ldl_convert(const Acc& L, int n, const int* ps, int* perm, int* kind, D* e, const v2::RtGroup& ex, int* flag) {
    if (ex.leader()) *flag = piv_list_valid(ps, n) ? 1 : 0;
    ex.barrier();
    const bool ok = *flag != 0;
    if (ok) {
        v2::for_range(ex, {0, n}, [&](int j) { v2::syconv_col<D>(L, ps, n, j, e); });
        if (ex.leader()) build_perm(ps, n, perm, kind);
    } else {
        v2::for_range(ex, {0, n}, [&](int t) {
            perm[t] = t;
            kind[t] = 0;
        });
    }
    ex.barrier();
    return ok;
}

// ?syconv as a gather, for a factor in global memory: column j of W is column j of G with its rows
// permuted by sigma_j, the interchanges after step j composed (W(t, j) = G(sigma_j(t), j), t > j).
// Walking j down, sigma gains one interchange per block start, applied by lane 0 in O(1) through
// its inverse; every read and write of a column is then one pass instead of one swap per step.
// kind must be build_perm's for a valid list.
template <typename D, typename Dst, typename Src>
inline void convert_gather(const Dst& W, const Src& G, int n, const int* ps, const int* kind, int* sig, int* inv,
                           const v2::RtGroup& ex) {
    v2::for_range(ex, {0, n}, [&](int t) {
        sig[t] = t;
        inv[t] = t;
    });
    ex.barrier();
    for (int j = n - 1; j >= 0; --j) {
        const bool two = kind[j] == 1;
        v2::map_range(ex, v2::col_of(W, j), {j + 1, n},
                      [&](int t) { return two && t == j + 1 ? v2::sfp::zero<D>() : G(sig[t], j); });
        ex.barrier();
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
        ex.barrier();
    }
}

// Row t of D^-1 x for a block of kind kd (v2::ldl_dsolve_row): ?sytrs's arithmetic, so both rows
// of a block agree.
template <typename D>
inline D d_solve_row(int kd, D x, D xp, D dtt, D dpp, D off) {
    return v2::ldl_dsolve_row(kd, x, xp, dtt, dpp, off);
}

// L w = y, w := D^-1 w, L^T z = w on one sub-group; x[r] holds logical row lane + 32 r. Collective.
template <int R, typename D, typename SG>
inline void ldl_solve_regs(const SG& sg, int lane, const Staged<D>& S, D (&x)[R]) {
    v2::trsv_frag<Uplo::Lower, Transpose::NoTrans, Diag::Unit>(sg, lane, S, S.n, x);
    v2::ldl_dsolve_frag<R>(sg, lane, S, S.kind, S.e, S.n, x);
    v2::trsv_frag<Uplo::Lower, Transpose::Trans, Diag::Unit>(sg, lane, S, S.n, x);
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
            v2::fill(v2::SubGroup<32>{sg}, nan_of<D>(), v2::vec(col, n), {0, n});
            continue;
        }
        D x[R];
        int row[R];
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r) {
            const int t = lane + 32 * r;
            row[r] = t < n ? (rev ? n - 1 - S.perm[t] : S.perm[t]) : 0;
            x[r] = t < n ? col[row[r]] : v2::sfp::zero<D>();
        }
        ldl_solve_regs<R>(sg, lane, S, x);
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < R; ++r)
            if (lane + 32 * r < n) col[row[r]] = x[r];
    }
}

}  // namespace batchlas::sytrf_device
