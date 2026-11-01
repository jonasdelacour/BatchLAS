#pragma once

// Bunch-Kaufman LDL^T as group-cooperative device code: LAPACK ?sytf2 (factor) and ?sytrs (solve),
// written once for the lower form. Uplo::Upper runs the same steps on the index-reversed matrix
// (logical i is physical n-1-i, v2::RevMat): the reversal maps ?sytf2's upper loop onto its lower
// loop step for step, so the stored factor, D and ipiv are LAPACK's for either uplo, and only the
// referenced triangle is ever read or written. evidence: docs/perf/sytrf.md#sytrf-the-reversal-map
//
// Every lane of the group must call these with the same arguments. Each pivot decision is lane 0's,
// broadcast (RtGroup::bcast): a decision that differed between lanes (a NaN compare, items aliased
// by a short stride) would deadlock the next barrier. With it every loop below is bounded by n and
// every lane runs the same barriers, whatever the data. evidence: docs/perf/sytrf.md#sytrf-termination
//
// The bodies are sequences of device LA v2 ops (include/batchlas/blas/device/v2/ext/sytrf-sytrs.hh);
// each op is the loop it replaced, so the SASS is the hand-written kernel's.

#include <src/kernels/common/dla_v2_scalar.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/blas/device/v2/ext/sytrf-sytrs.hh>

#include <sycl/sycl.hpp>

namespace batchlas::sytrf_device {

namespace v2 = batchlas::device::v2;

template <typename D> using Tile = v2::RevMat<D>;
template <typename D> using Vec = v2::RevVec<D>;

template <typename D> inline D nan_of() { return v2::sfp::nan<D>(); }

// ?sytf2, lower form, on the logical tile. `piv_out(phys_k, value)` stores LAPACK's ipiv entry
// (1-based; a 2x2 pivot writes -(p+1) at both of its positions); lane 0 calls it. Returns info.
template <typename D, typename R, typename PivOut>
inline int ldl_factor(const Tile<D>& S, int n, bool rev, const v2::RtGroup& g, v2::ArgMaxScratch<R> am, PivOut&& piv_out) {
    using v2::sfp::neg;
    const R alpha = v2::bk_alpha<R>();
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    int info = 0;
    int k = 0;
    while (k < n) {
        int kstep = 1;
        int kp = k;
        const R absakk = v2::abs1(S(k, k));
        // ?sytf2's I?AMAX takes the first maximum in storage order: the smallest physical row,
        // which for the reversed (upper) form is the largest logical one.
        const v2::ArgMax<R> col = v2::iamax_bk(g, am, v2::col_of(S, k), {k + 1, n}, !rev);
        const R colmax = col.v < R(0) ? R(0) : col.v;
        const int imax = col.i;
        const int stage = v2::bk_stage(g, absakk, colmax, alpha);
        if (stage == v2::kBkNeedRow) {
            const v2::ArgMax<R> row = v2::iamax_bk(g, am, v2::sym_row_lower(S, imax), {k, n}, true, v2::SkipAt{imax});
            v2::bk_choice(g, absakk, colmax, row.v, v2::abs1(S(imax, imax)), alpha, k, imax, kp, kstep);
        }
        if (stage == v2::kBkZero) {
            if (info == 0) info = phys(k) + 1;
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {  // the symmetric interchange of kk and kp: ?sytf2's swaps
                v2::swap(g, v2::col_of(S, kk), v2::col_of(S, kp), {kp + 1, n});
                v2::swap(g, v2::col_of(S, kk), v2::row_of(S, kp), {kk + 1, kp});
                if (g.leader()) {
                    v2::swap_elem(S(kk, kk), S(kp, kp));
                    if (kstep == 2) v2::swap_elem(S(k + 1, k), S(kp, k));
                }
            }
            g.barrier();
            if (kstep == 1) {  // A22 -= A21 d^-1 A21^T, then L21 = A21 d^-1
                const D mr1 = neg(v2::sfp::recip(S(k, k)));
                v2::syr<Uplo::Lower>(g, mr1, v2::col_of(S, k), S, {k + 1, n});
                g.barrier();
                v2::scal(g, neg(mr1), v2::col_of(S, k), {k + 1, n});
            } else if (k < n - 2) {  // A22 -= A21 D^-1 A21^T, then L21 = A21 D^-1
                const auto blk = v2::bk_block2(S(k, k), S(k + 1, k), S(k + 1, k + 1));
                const auto c0 = v2::col_of(S, k), c1 = v2::col_of(S, k + 1);
                v2::gemmt2_sub<Uplo::Lower>(g, c0, [&](int j) { return blk.first(c0(j), c1(j)); }, c1,
                                            [&](int j) { return blk.second(c0(j), c1(j)); }, S, {k + 2, n});
                g.barrier();
                v2::bk_apply2(g, blk, c0, c1, c0, c1, {k + 2, n});
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
        g.barrier();
        k += kstep;
    }
    return info;
}

// ?lasyf's W, kRank = nb + 1 columns: logical row i at physical row map(i), panel column c at
// physical column c (Lower) or kRank-1-c (Upper), so W's physical column order is the order of the
// panel's physical columns in A and the trailing gemm takes both as plain sub-views.
template <typename D> inline v2::RevMat<D> w_tile(D* p, std::ptrdiff_t ld, int n, int ncol, bool rev) {
    return v2::rev_mat(p, ld, n, ncol, rev);
}

// ?lasyf, lower form, on the uniform panel [k0, k0 + nb) of the logical matrix, starting at
// `start` (k0, or k0 + 1 when the previous panel's last 2 x 2 block took column k0). The step at
// the panel's last column is always taken, so a 2 x 2 block may take column k0 + nb (W's column nb):
// the panel ends at k0 + nb or k0 + nb + 1, and the trailing update has one uniform rank, nb + 1.
// W's columns the item did not compute are zero. Returns the end; *info as ldl_factor's.
// evidence: docs/perf/sytrf.md#sytrf-the-blocked-route
template <typename D, typename R, typename PivOut>
inline int ldl_panel(const Tile<D>& S, const v2::RevMat<D>& W, int n, int k0, int nb, int start, bool rev,
                     const v2::RtGroup& g, v2::ArgMaxScratch<R> am, int& info, PivOut&& piv_out) {
    using v2::col_of;
    using v2::row_of;
    const R alpha = v2::bk_alpha<R>();
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    v2::fill_rows(g, v2::sfp::zero<D>(), W, {k0, n}, nb + 1);
    g.barrier();
    // W(k:n, c) := source - A(k:n, start:k) W(r, start-k0:k-k0): the panel's finished columns only.
    const auto L = v2::cols_from(S, k0);
    int k = start;
    while (k < k0 + nb) {
        const int c = k - k0;
        v2::gemv_sub(g, L, row_of(W, k), col_of(S, k), col_of(W, c), {k, n}, start - k0, k - k0);
        g.barrier();
        int kstep = 1;
        int kp = k;
        const R absakk = v2::abs1(W(k, c));
        const v2::ArgMax<R> col = v2::iamax_bk(g, am, col_of(W, c), {k + 1, n}, !rev);
        const R colmax = col.v < R(0) ? R(0) : col.v;
        const int imax = col.i;
        const int stage = v2::bk_stage(g, absakk, colmax, alpha);
        if (stage == v2::kBkNeedRow) {
            v2::gemv_sub(g, L, row_of(W, imax), [&](int i) { return i < imax ? S(imax, i) : S(i, imax); }, col_of(W, c + 1),
                         {k, n}, start - k0, k - k0);
            g.barrier();
            const v2::ArgMax<R> row = v2::iamax_bk(g, am, col_of(W, c + 1), {k, n}, true, v2::SkipAt{imax});
            v2::bk_choice(g, absakk, colmax, row.v, v2::abs1(W(imax, c + 1)), alpha, k, imax, kp, kstep);
            if (kstep == 1 && kp != k) v2::copy(g, col_of(W, c + 1), col_of(W, c), {k, n});
            g.barrier();
        }
        if (stage == v2::kBkZero) {
            if (info == 0) info = phys(k) + 1;
            v2::copy(g, col_of(W, c), col_of(S, k), {k, n});
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {
                // The non-updated column kk moves to kp; the panel's finished columns and W swap
                // rows kk and kp (ldl_undo restores the columns' ?sytf2 form after the update).
                if (g.leader()) S(kp, kp) = S(kk, kk);
                v2::copy(g, col_of(S, kk), row_of(S, kp), {kk + 1, kp});
                v2::copy(g, col_of(S, kk), col_of(S, kp), {kp + 1, n});
                v2::swap(g, row_of(S, kk), row_of(S, kp), {start, kk});
                v2::swap(g, row_of(W, kk), row_of(W, kp), {start - k0, kk - k0 + 1});
            }
            g.barrier();
            if (kstep == 1) {  // L(k:n, k) = W(k:n, c) / d, the pivot kept
                const D r1 = v2::sfp::recip(W(k, c));
                v2::map_range(g, col_of(S, k), {k, n}, [&](int i) { return i == k ? W(k, c) : v2::sfp::mul(r1, W(i, c)); });
            } else {  // L(k+2:n, k:k+1) = W D^-1, then D itself
                const auto blk = v2::bk_block2(W(k, c), W(k + 1, c), W(k + 1, c + 1));
                v2::bk_apply2(g, blk, col_of(W, c), col_of(W, c + 1), col_of(S, k), col_of(S, k + 1), {k + 2, n});
                if (g.leader()) {
                    S(k, k) = W(k, c);
                    S(k + 1, k) = W(k + 1, c);
                    S(k + 1, k + 1) = W(k + 1, c + 1);
                }
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
        g.barrier();
        k += kstep;
    }
    // The last step's row search may have left column imax in W(:, nb) though the block was 1 x 1:
    // the uniform-rank gemm reads that column, so it must be zero unless a 2 x 2 block owns it.
    if (k == k0 + nb) v2::fill(g, v2::sfp::zero<D>(), col_of(W, nb), {k0, n});
    return k;
}

// ?lasyf's last loop: undo the panel's interchanges on its own finished columns [start, end), in
// reverse, so they hold ?sytf2's form. `logical_partner(j)` is ipiv's partner of logical column j
// (rows j / j-1 for a 2 x 2 block ending at j) and its sign. Runs after the trailing update.
template <typename D, typename PivIn>
inline void ldl_undo(const Tile<D>& S, int start, int end, const v2::RtGroup& g, PivIn&& logical_partner) {
    int j = end - 1;
    while (j >= start) {
        const int jj = j;
        bool two = false;
        const int jp = logical_partner(j, two);
        if (two) --j;
        --j;
        if (jp != jj && j >= start) v2::swap(g, v2::row_of(S, jp), v2::row_of(S, jj), {start, j + 1});
        g.barrier();
    }
}

// ?sytrs for one right-hand side, lower form. `piv(phys_k)` reads LAPACK's ipiv entry.
template <typename D, typename R, typename PivIn>
inline void ldl_solve(const Tile<D>& S, int n, bool rev, const Vec<D>& b, const v2::RtGroup& g, v2::SumScratch<R> sum,
                      PivIn&& piv) {
    using v2::col_of;
    using v2::Minus;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    // A list sytrf cannot have written (|ipiv| outside [1, n], an unpaired 2 x 2 entry) would index
    // b out of bounds: that item's X is NaN instead. Every lane walks the list, lane 0 decides.
    bool ok = true;
    for (int k = 0; k < n && ok;) {
        const int raw = piv(phys(k));
        ok = raw != 0 && raw >= -n && raw <= n && (raw > 0 || (k + 1 < n && piv(phys(k + 1)) == raw));
        k += raw > 0 ? 1 : 2;
    }
    if (g.bcast(ok ? 1 : 0) == 0) {
        v2::fill(g, nan_of<D>(), b, {0, n});
        g.barrier();
        return;
    }
    // P L y = b, then D z = y
    int k = 0;
    while (k < n) {
        const int raw = piv(phys(k));
        if (raw > 0) {
            const int kp = phys(raw - 1);
            if (g.leader() && kp != k) v2::swap_elem(b(k), b(kp));
            g.barrier();
            const D bk = b(k);
            g.barrier();
            v2::axpy(g, Minus{bk}, col_of(S, k), b, {k + 1, n});
            if (g.leader()) b(k) = v2::sfp::mul(bk, v2::sfp::recip(S(k, k)));
            g.barrier();
            k += 1;
        } else {
            const int kp = phys(-raw - 1);
            if (g.leader() && kp != k + 1) v2::swap_elem(b(k + 1), b(kp));
            g.barrier();
            const D bk = b(k), bk1 = b(k + 1);
            g.barrier();
            v2::axpy2(g, Minus{bk}, col_of(S, k), Minus{bk1}, col_of(S, k + 1), b, {k + 2, n});
            if (g.leader()) v2::ldl_d2(S(k + 1, k), S(k, k), S(k + 1, k + 1)).solve(bk, bk1, b(k), b(k + 1));
            g.barrier();
            k += 2;
        }
    }
    // L^T P^T x = z
    k = n - 1;
    while (k >= 0) {
        const int raw = piv(phys(k));
        const auto s = v2::dotu2_sum<D>(g, sum, raw < 0, col_of(S, k), col_of(S, k - 1), b, {k + 1, n});
        g.barrier();
        if (g.leader()) {
            b(k) = v2::sfp::sub(b(k), s[0]);
            if (raw < 0) b(k - 1) = v2::sfp::sub(b(k - 1), s[1]);
            const int kp = phys((raw > 0 ? raw : -raw) - 1);
            if (kp != k) v2::swap_elem(b(k), b(kp));
        }
        g.barrier();
        k -= raw > 0 ? 1 : 2;
    }
}

}  // namespace batchlas::sytrf_device
