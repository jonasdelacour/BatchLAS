#pragma once

// Bunch-Kaufman LDL^T as group-cooperative device code: LAPACK ?sytf2 (factor) and ?sytrs (solve),
// written once for the lower form. Uplo::Upper runs the same steps on the index-reversed matrix
// (logical i is physical n-1-i): the reversal maps ?sytf2's upper loop onto its lower loop step for
// step, so the stored factor, D and ipiv are LAPACK's for either uplo, and only the referenced
// triangle is ever read or written. evidence: docs/perf/sytrf.md#sytrf-the-reversal-map
//
// Every lane of the group must call these with the same arguments. Each pivot decision is lane 0's,
// broadcast (sytrf_v2::uniform): a decision that differed between lanes (a NaN compare, items
// aliased by a short stride) would deadlock the next barrier. With it every loop below is bounded by
// n and every lane runs the same barriers, whatever the data. evidence: docs/perf/sytrf.md#sytrf-termination
//
// The executor is a v2::RtGroup<Group> (one work-group per item) or a v2::SubGroup<32> (the CTA
// tier's sub-group 0). ldl_factor and ldl_solve are device LA v2 loops (v2::for_each_owned over the
// hand loops' ranges) plus the family's private ops in sytrf_v2.hh; ldl_panel and ldl_undo, the
// blocked tier's step, keep their hand-written loops.

#include <src/ops/sytrf/sytrf_v2.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <utility>

namespace batchlas::sytrf_device {

namespace v2 = batchlas::device::v2;
using sytrf_v2::Ar;
using sytrf_v2::ArgMaxScratch;
using sytrf_v2::group_sync;
using sytrf_v2::nan_of;
using sytrf_v2::SumScratch;

template <typename D> using Tile = sytrf_v2::RevMat<D>;
template <typename D> using Vec = sytrf_v2::RevVec<D>;

// ?sytf2, lower form, on the logical tile. `piv_out(phys_k, value)` stores LAPACK's ipiv entry
// (1-based; a 2x2 pivot writes -(p+1) at both of its positions); lane 0 calls it. Returns info.
template <typename D, typename R, typename Ex, typename PivOut>
inline int ldl_factor(const Tile<D>& S, int n, bool rev, const Ex& g, ArgMaxScratch<R> am, PivOut&& piv_out) {
    using A = Ar<D>;
    using v2::Range;
    const R alpha = sytrf_v2::bk_alpha<R>();
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    int info = 0;
    int k = 0;
    while (k < n) {
        int kstep = 1;
        int kp = k;
        const R absakk = v2::abs1(S(k, k));
        // ?sytf2's I?AMAX takes the first maximum in storage order: the smallest physical row,
        // which for the reversed (upper) form is the largest logical one.
        const auto col = sytrf_v2::iamax_bk(g, am, [&](int i) -> D& { return S(i, k); }, Range{k + 1, n}, !rev);
        const R colmax = col.v < R(0) ? R(0) : col.v;
        const int imax = col.i;
        const int stage = sytrf_v2::bk_stage(g, absakk, colmax, alpha);
        if (stage == sytrf_v2::kBkNeedRow) {
            const auto row = sytrf_v2::iamax_bk<true>(
                g, am, [&](int j) -> D& { return j > imax ? S(j, imax) : S(imax, j); }, Range{k, n}, true, imax);
            sytrf_v2::bk_choice(g, absakk, colmax, row.v, v2::abs1(S(imax, imax)), alpha, k, imax, kp, kstep);
        }
        if (stage == sytrf_v2::kBkZero) {
            if (info == 0) info = phys(k) + 1;
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {  // the symmetric interchange of kk and kp
                v2::for_each_owned(g, Range{kp + 1, n}, [&](int i) BATCHLAS_DLA_BODY { std::swap(S(i, kk), S(i, kp)); });
                v2::for_each_owned(g, Range{kk + 1, kp}, [&](int j) BATCHLAS_DLA_BODY { std::swap(S(j, kk), S(kp, j)); });
                if (g.leader()) {
                    std::swap(S(kk, kk), S(kp, kp));
                    if (kstep == 2) std::swap(S(k + 1, k), S(kp, k));
                }
            }
            group_sync(g);
            if (kstep == 1) {  // A22 -= A21 d^-1 A21^T (?syr), then L21 = A21 d^-1
                const D mr1 = A::neg(A::recip(S(k, k)));
                for (int j = k + 1; j < n; ++j) {
                    const D t = A::mul(mr1, S(j, k));
                    v2::for_each_owned(g, Range{j, n}, [&](int i) BATCHLAS_DLA_BODY { S(i, j) = A::add(S(i, j), A::mul(S(i, k), t)); });
                }
                group_sync(g);
                const D r1 = A::neg(mr1);
                v2::for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY { S(i, k) = A::mul(r1, S(i, k)); });
            } else if (k < n - 2) {  // A22 -= A21 D^-1 A21^T, then L21 = A21 D^-1
                const auto blk = sytrf_v2::bk_block2(S(k, k), S(k + 1, k), S(k + 1, k + 1));
                for (int j = k + 2; j < n; ++j) {
                    const D a = blk.first(S(j, k), S(j, k + 1)), b = blk.second(S(j, k), S(j, k + 1));
                    v2::for_each_owned(g, Range{j, n}, [&](int i) BATCHLAS_DLA_BODY {
                        S(i, j) = A::sub(A::sub(S(i, j), A::mul(S(i, k), a)), A::mul(S(i, k + 1), b));
                    });
                }
                group_sync(g);
                v2::for_each_owned(g, Range{k + 2, n}, [&](int i) BATCHLAS_DLA_BODY {
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
        group_sync(g);
        k += kstep;
    }
    return info;
}

// ?lasyf's W, kRank = nb + 1 columns: logical row i at physical row map(i), panel column c at
// physical column c (Lower) or kRank-1-c (Upper), so W's physical column order is the order of the
// panel's physical columns in A and the trailing gemm takes both as plain sub-views.
template <typename D> inline Tile<D> w_tile(D* p, std::ptrdiff_t ld, int n, int ncol, bool rev) {
    return sytrf_v2::rev_mat(p, ld, n, ncol, rev);
}

// ?lasyf, lower form, on the uniform panel [k0, k0 + nb) of the logical matrix, starting at
// `start` (k0, or k0 + 1 when the previous panel's last 2 x 2 block took column k0). The step at
// the panel's last column is always taken, so a 2 x 2 block may take column k0 + nb (W's column nb):
// the panel ends at k0 + nb or k0 + nb + 1, and the trailing update has one uniform rank, nb + 1.
// W's columns the item did not compute are zero. Returns the end; *info as ldl_factor's.
// evidence: docs/perf/sytrf.md#sytrf-the-blocked-route
// The loops stay hand-written: as v2 ops the W zero-fill unrolled fully at device link and the
// column update moved loads and FMAs (complex<float> registers 68 -> 128).
template <typename D, typename R, typename Ex, typename PivOut>
inline int ldl_panel(const Tile<D>& S, const Tile<D>& W, int n, int k0, int nb, int start, bool rev, const Ex& g,
                     ArgMaxScratch<R> am, int& info, PivOut&& piv_out) {
    using A = Ar<D>;
    const R alpha = sytrf_v2::bk_alpha<R>();
    const int lid = g.lid(), nl = v2::detail::size_of(g);
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    for (int i = k0 + lid; i < n; i += nl)
        for (int c = 0; c <= nb; ++c) W(i, c) = A::zero();
    group_sync(g);
    // W(k:n, dst) := source - A(k:n, start:k) W(r, start-k0:k-k0): the panel's finished columns only.
    auto update_col = [&](int k, int dst, int r, auto&& source) {
        for (int i = k + lid; i < n; i += nl) {
            D acc = source(i);
            for (int l = start - k0; l < k - k0; ++l) acc = A::sub(acc, A::mul(S(i, k0 + l), W(r, l)));
            W(i, dst) = acc;
        }
        group_sync(g);
    };
    int k = start;
    while (k < k0 + nb) {
        const int c = k - k0;
        update_col(k, c, k, [&](int i) { return S(i, k); });
        int kstep = 1;
        int kp = k;
        const R absakk = v2::abs1(W(k, c));
        R cv = R(-1);
        int ci = -1;
        for (int i = k + 1 + lid; i < n; i += nl) {
            const R v = sytrf_v2::nan_as_inf(v2::abs1(W(i, c)));
            if (sytrf_v2::argmax_beats(v, i, cv, ci, !rev)) { cv = v; ci = i; }
        }
        sytrf_v2::argmax_reduce(g, am, cv, ci, !rev);
        const R colmax = cv < R(0) ? R(0) : cv;
        const int imax = ci;
        const int stage = sytrf_v2::bk_stage(g, absakk, colmax, alpha);
        if (stage == sytrf_v2::kBkNeedRow) {
            update_col(k, c + 1, imax, [&](int i) { return i < imax ? S(imax, i) : S(i, imax); });
            R rv = R(-1);
            int ri = -1;
            for (int j = k + lid; j < n; j += nl) {
                if (j == imax) continue;
                const R v = sytrf_v2::nan_as_inf(v2::abs1(W(j, c + 1)));
                if (sytrf_v2::argmax_beats(v, j, rv, ri, true)) { rv = v; ri = j; }
            }
            sytrf_v2::argmax_reduce(g, am, rv, ri, true);
            sytrf_v2::bk_choice(g, absakk, colmax, rv, v2::abs1(W(imax, c + 1)), alpha, k, imax, kp, kstep);
            if (kstep == 1 && kp != k)
                for (int i = k + lid; i < n; i += nl) W(i, c) = W(i, c + 1);
            group_sync(g);
        }
        if (stage == sytrf_v2::kBkZero) {
            if (info == 0) info = phys(k) + 1;
            for (int i = k + lid; i < n; i += nl) S(i, k) = W(i, c);
        } else {
            const int kk = k + kstep - 1;
            if (kp != kk) {
                // The non-updated column kk moves to kp; the panel's finished columns and W swap
                // rows kk and kp (ldl_undo restores the columns' ?sytf2 form after the update).
                if (lid == 0) S(kp, kp) = S(kk, kk);
                for (int j = kk + 1 + lid; j < kp; j += nl) S(kp, j) = S(j, kk);
                for (int i = kp + 1 + lid; i < n; i += nl) S(i, kp) = S(i, kk);
                for (int l = start + lid; l < kk; l += nl) std::swap(S(kk, l), S(kp, l));
                for (int l = start - k0 + lid; l <= kk - k0; l += nl) std::swap(W(kk, l), W(kp, l));
            }
            group_sync(g);
            if (kstep == 1) {
                const D r1 = A::recip(W(k, c));
                for (int i = k + lid; i < n; i += nl) S(i, k) = i == k ? W(k, c) : A::mul(r1, W(i, c));
            } else {
                const auto blk = sytrf_v2::bk_block2(W(k, c), W(k + 1, c), W(k + 1, c + 1));
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
        group_sync(g);
        k += kstep;
    }
    // The last step's row search may have left column imax in W(:, nb) though the block was 1 x 1:
    // the uniform-rank gemm reads that column, so it must be zero unless a 2 x 2 block owns it.
    if (k == k0 + nb)
        for (int i = k0 + lid; i < n; i += nl) W(i, nb) = A::zero();
    return k;
}

// ?lasyf's last loop: undo the panel's interchanges on its own finished columns [start, end), in
// reverse, so they hold ?sytf2's form. `logical_partner(j)` is ipiv's partner of logical column j
// (rows j / j-1 for a 2 x 2 block ending at j) and its sign. Runs after the trailing update.
template <typename D, typename Ex, typename PivIn>
inline void ldl_undo(const Tile<D>& S, int start, int end, const Ex& g, PivIn&& logical_partner) {
    const int lid = g.lid(), nl = v2::detail::size_of(g);
    int j = end - 1;
    while (j >= start) {
        const int jj = j;
        bool two = false;
        const int jp = logical_partner(j, two);
        if (two) --j;
        --j;
        if (jp != jj && j >= start)
            for (int l = start + lid; l <= j; l += nl) std::swap(S(jp, l), S(jj, l));
        group_sync(g);
    }
}

// ?sytrs for one right-hand side, lower form. `piv(phys_k)` reads LAPACK's ipiv entry.
template <typename D, typename R, typename Ex, typename PivIn>
inline void ldl_solve(const Tile<D>& S, int n, bool rev, const Vec<D>& b, const Ex& g, SumScratch<R> sum, PivIn&& piv) {
    using A = Ar<D>;
    using v2::Range;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    // A list sytrf cannot have written (|ipiv| outside [1, n], an unpaired 2 x 2 entry) would index
    // b out of bounds: that item's X is NaN instead. Every lane walks the list, lane 0 decides.
    bool ok = true;
    for (int k = 0; k < n && ok;) {
        const int raw = piv(phys(k));
        ok = raw != 0 && raw >= -n && raw <= n && (raw > 0 || (k + 1 < n && piv(phys(k + 1)) == raw));
        k += raw > 0 ? 1 : 2;
    }
    if (sytrf_v2::uniform(g, ok ? 1 : 0) == 0) {
        v2::for_each_owned(g, n, [&](int i) BATCHLAS_DLA_BODY { b(i) = nan_of<D>(); });
        group_sync(g);
        return;
    }
    // P L y = b, then D z = y
    int k = 0;
    while (k < n) {
        const int raw = piv(phys(k));
        if (raw > 0) {
            const int kp = phys(raw - 1);
            if (g.leader() && kp != k) std::swap(b(k), b(kp));
            group_sync(g);
            const D bk = b(k);
            group_sync(g);
            v2::for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY { b(i) = A::sub(b(i), A::mul(S(i, k), bk)); });
            if (g.leader()) b(k) = A::mul(bk, A::recip(S(k, k)));
            group_sync(g);
            k += 1;
        } else {
            const int kp = phys(-raw - 1);
            if (g.leader() && kp != k + 1) std::swap(b(k + 1), b(kp));
            group_sync(g);
            const D bk = b(k), bk1 = b(k + 1);
            group_sync(g);
            v2::for_each_owned(g, Range{k + 2, n}, [&](int i) BATCHLAS_DLA_BODY {
                b(i) = A::sub(A::sub(b(i), A::mul(S(i, k), bk)), A::mul(S(i, k + 1), bk1));
            });
            if (g.leader()) {
                const D akm1k = S(k + 1, k);
                const D akm1 = A::div(S(k, k), akm1k);
                const D ak = A::div(S(k + 1, k + 1), akm1k);
                const D denom = A::sub(A::mul(akm1, ak), A::one());
                const D bkm1 = A::div(bk, akm1k);
                const D bkk = A::div(bk1, akm1k);
                b(k) = A::div(A::sub(A::mul(ak, bkm1), bkk), denom);
                b(k + 1) = A::div(A::sub(A::mul(akm1, bkk), bkm1), denom);
            }
            group_sync(g);
            k += 2;
        }
    }
    // L^T P^T x = z
    k = n - 1;
    while (k >= 0) {
        const int raw = piv(phys(k));
        D p0 = A::zero(), p1 = A::zero();
        v2::for_each_owned(g, Range{k + 1, n}, [&](int i) BATCHLAS_DLA_BODY {
            p0 = A::add(p0, A::mul(S(i, k), b(i)));
            if (raw < 0) p1 = A::add(p1, A::mul(S(i, k - 1), b(i)));
        });
        const D s0 = sytrf_v2::sum_reduce(g, sum, p0);
        const D s1 = raw < 0 ? sytrf_v2::sum_reduce(g, sum, p1) : A::zero();
        group_sync(g);
        if (g.leader()) {
            b(k) = A::sub(b(k), s0);
            if (raw < 0) b(k - 1) = A::sub(b(k - 1), s1);
            const int kp = phys((raw > 0 ? raw : -raw) - 1);
            if (kp != k) std::swap(b(k), b(kp));
        }
        group_sync(g);
        k -= raw > 0 ? 1 : 2;
    }
}

}  // namespace batchlas::sytrf_device
