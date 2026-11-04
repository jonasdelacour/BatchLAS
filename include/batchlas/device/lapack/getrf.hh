// SPDX-License-Identifier: MIT
#pragma once
// ?GETF2, GETRF's own routine: by row interchanges (getf2) or lazy relabel (getf2_relabel), with a
// fused pivot search (one butterfly, one slot barrier); arith::Fma. getf2_detail: the policies and
// the steps core has no SASS-identical form of yet. evidence: docs/perf/lu.md#the-shape-of-the-getrf-device-code

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device::inline v2 {

namespace getf2_detail {

// ---- pivot search ---------------------------------------------------------------------------
template <class R> struct MaxLoc {
    R v;
    int i;
};
// Take (ov, oi) when larger, or equal with a lower index; a NaN never wins.
template <class R> BATCHLAS_DLA_INLINE void maxloc_merge(MaxLoc<R>& a, R ov, int oi) {
    if (ov > a.v || (ov == a.v && oi < a.i)) {
        a.v = ov;
        a.i = oi;
    }
}

// xor butterfly over `lanes` (a launch value) lanes of sg, masks lanes/2 .. 1.
template <class R> BATCHLAS_DLA_INLINE void maxloc_bfly(const sycl::sub_group& sg, MaxLoc<R>& a, int lanes) {
    for (std::uint32_t off = static_cast<std::uint32_t>(lanes) / 2u; off > 0u; off >>= 1) {
        const R ov = batchlas::portable::permute_group_by_xor(sg, a.v, off);
        const int oi = batchlas::portable::permute_group_by_xor(sg, a.i, off);
        maxloc_merge(a, ov, oi);
    }
}

// xor butterfly over a P-lane Partition, masks 1 .. P/2 (the tiny tiers' order). THE CALLER
// OWES A NaN-FREE SEED: a NaN loses every comparison and the lanes disagree on the winner.
template <class Px, class R> BATCHLAS_DLA_INLINE void maxloc_bfly_up(const Px& px, MaxLoc<R>& a) {
    constexpr int P = Px::team_size;
BATCHLAS_UNROLL_FULL
    for (std::uint32_t mk = 1u; mk < static_cast<std::uint32_t>(P); mk <<= 1) {
        const R ov = detail::xor_shuffle(px.part, a.v, mk);
        const int oi = detail::xor_shuffle(px.part, a.i, mk);
        maxloc_merge(a, ov, oi);
    }
}

// Lane 0 of team t writes slots[t], ONE barrier (B1), every item scans the slots in team order;
// no trailing barrier (the caller's B2 precedes the next write). Scope::Sub returns `a`.
template <class Ex, class R, class VS, class IS>
BATCHLAS_DLA_INLINE MaxLoc<R> maxloc_group(const Ex& ex, MaxLoc<R> a, const VS& vs, const IS& is, int none) {
    if constexpr (Ex::natural_scope == Scope::Sub) {
        return a;
    } else {
        if (ex.lane() == 0) {
            vs[static_cast<std::size_t>(ex.team())] = a.v;
            is[static_cast<std::size_t>(ex.team())] = a.i;
        }
        ex.barrier(Mem::Local);  // B1
        R fv = R(-1);
        int p = none;
        const int nt = detail::nteams_of(ex);
        for (int t = 0; t < nt; ++t) {  // the legacy statements: ptxas unrolls this scan by them
            const R v = vs[static_cast<std::size_t>(t)];
            const int ii = is[static_cast<std::size_t>(t)];
            if (v > fv || (v == fv && ii < p)) {
                fv = v;
                p = ii;
            }
        }
        return {fv, p};
    }
}

// ?IAMAX (cabs1) of A(r0:m-1, c): the absolute row of the first maximum, NaN rows skipped; m
// when none qualifies. vs/is: nteams slots of local memory (unread at Scope::Sub).
template <class Ex, class MA, class VS, class IS>
BATCHLAS_DLA_INLINE int iamax_col(const Ex& ex, const MA& a, int r0, int c, int lanes, const VS& vs, const IS& is) {
    using D = typename MA::value_type;
    using R = real_t<D>;
    const int m = a.rows();
    MaxLoc<R> best{R(-1), m};  // m: "no candidate", never wins a tie
    for_each_owned(ex, Range{r0, m}, [&](int i) BATCHLAS_DLA_BODY {
        const R v = abs1<D>(a(i, c));
        if (v > best.v) {
            best.v = v;
            best.i = i;
        }
    });
    maxloc_bfly(ex.sub(), best, lanes);
    return maxloc_group(ex, best, vs, is, m).i;
}

// The pivot key of the tiny tiers: ORDERING field high (lowest wins), winner's LANE low. Every
// ordering field must stay distinct in a partition. evidence: docs/perf/lu.md#the-pivot-key-encoding
inline constexpr int kPivotKeyLaneBits = 8;
inline constexpr int pivot_key(int order_field, int lane) { return (order_field << kPivotKeyLaneBits) | lane; }
inline constexpr int pivot_key_order(int key) { return key >> kPivotKeyLaneBits; }
inline constexpr std::uint32_t pivot_key_lane(int key) {
    return static_cast<std::uint32_t>(key) & ((1u << kPivotKeyLaneBits) - 1u);
}

// ---- level 1/2 steps of ?GETF2 ------------------------------------------------------------------
// A(r0:m-1, c) /= d with ?GETF2's arms: one reciprocal and a multiply when 1/d is finite and
// nonzero, else a division per element (the sfmin arm), the arm chosen OUTSIDE the loop (core
// rscl selects per element). d != 0 is the caller's test.
template <class Ex, class MA>
BATCHLAS_DLA_INLINE void rscl_col(const Ex& ex, typename MA::value_type d, const MA& a, int r0, int c) {
    using Ar = arith_t<arith::Fma, typename MA::value_type>;
    const auto r = Ar::recip(d);
    const bool use_mul = Ar::isfinite(r) && !Ar::is_zero(r);
    const Range rows{r0, a.rows()};
    if (use_mul) for_each_owned(ex, rows, [&](int i) BATCHLAS_DLA_BODY { a(i, c) = Ar::mul(a(i, c), r); });
    else for_each_owned(ex, rows, [&](int i) BATCHLAS_DLA_BODY { a(i, c) = Ar::div(a(i, c), d); });
}

// A(k+1:, k+1:) -= A(k+1:, k) A(k, k+1:), flattened (docs/perf/lu.md#lu-negative-results), two
// roundings (sub of mul); written out, not a for_each_owned_2d lambda: through the lambda the
// panels drift a VOTE, +4 REG (sm_89).
template <class Ex, class MA> BATCHLAS_DLA_INLINE void ger_trailing(const Ex& ex, const MA& a, int k) {
    using Ar = arith_t<arith::Fma, typename MA::value_type>;
    const int mm = a.rows() - k - 1;
    const int nn = a.cols() - k - 1;
    if (mm > 0 && nn > 0) {
        const int total = mm * nn;
        const int w = detail::size_of(ex);
        for (int e = ex.lid(); e < total; e += w) {
            const int i = k + 1 + (e % mm);
            const int j = k + 1 + (e / mm);
            a(i, j) = Ar::sub(a(i, j), Ar::mul(a(i, k), a(k, j)));
        }
    }
}

// ---- getf2_relabel policies -------------------------------------------------------------------
struct PivotWinner {
    int row;  // winning logical row
    int src;  // the lane that holds it (policies that broadcast by lane)
};

// Piv over a Scope::Group executor: candidates are rows >= j (a pad item carries rowid -1), a NaN
// magnitude maps to -1 or the lanes disagree on the winner, a non-candidate carries `m`; then
// maxloc_group (B1). The leader writes ipiv and keeps info (global, 1-based, sticky).
// evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
template <class Ex, class VS, class IS> struct PivotRowsGroup {
    const Ex& ex;
    VS vs;
    IS is;
    int m;
    int lanes;
    IpivSink ipiv;
    std::int32_t& info;
    template <class R> BATCHLAS_DLA_INLINE PivotWinner argmax(int j, R mag, int rowid) const {
        const bool cand = (rowid >= j);
        MaxLoc<R> a{(cand && (mag == mag)) ? mag : R(-1), cand ? rowid : m};
        maxloc_bfly(ex.sub(), a, lanes);
        a = maxloc_group(ex, a, vs, is, m);
        return {a.i >= m ? j : a.i, 0};  // >= m: unreachable, logical row j exists
    }
    BATCHLAS_DLA_INLINE void record(int j, PivotWinner w) const {
        if (ex.leader()) ipiv.set(j, w.row);
    }
    BATCHLAS_DLA_INLINE void on_zero(bool zero, int j) const {
        if (zero && ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + j + 1);
    }
};
template <class Ex, class VS, class IS>
BATCHLAS_DLA_INLINE PivotRowsGroup<Ex, VS, IS> pivot_rows_group(const Ex& ex, VS vs, IS is, int m, IpivSink ipiv,
                                                                std::int32_t& info) {
    return {ex, vs, is, m, detail::team_size_of(ex), ipiv, info};
}

// Bc through a local-memory row sx (>= NC elements): the item whose rowid is j writes the row,
// one barrier (B2); everyone reads it. No third barrier: docs/perf/lu.md#two-barriers-per-column
template <class Ex, class SX> struct BcastRowLocal {
    const Ex& ex;
    SX sx;
    template <int NC, class T> BATCHLAS_DLA_INLINE void publish(int j, const Frag<T, NC>& rA, int rowid, PivotWinner) const {
        if (rowid == j) {
BATCHLAS_UNROLL_FULL
            for (int k = j; k < NC; ++k) sx[static_cast<std::size_t>(k)] = rA[k];
        }
        ex.barrier(Mem::Local);  // B2
    }
    template <int NC, class T> BATCHLAS_DLA_INLINE T pivot(int j, const Frag<T, NC>&, PivotWinner) const {
        return sx[static_cast<std::size_t>(j)];
    }
    template <int NC, class T, class F>
    BATCHLAS_DLA_INLINE void for_right(int j, int ncols, const Frag<T, NC>&, PivotWinner, F&& f) const {
BATCHLAS_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, static_cast<T>(sx[static_cast<std::size_t>(k)]));
        }
    }
};
template <class Ex, class SX> BATCHLAS_DLA_INLINE BcastRowLocal<Ex, SX> bcast_row_local(const Ex& ex, SX sx) {
    return {ex, sx};
}

// Bc by shuffles from the winner's lane (w.src) of a Partition: no memory, no barrier, one
// select per element (two per complex).
template <class Px> struct BcastShuffle {
    const Px& px;
    template <int NC, class T> BATCHLAS_DLA_INLINE void publish(int, const Frag<T, NC>&, int, PivotWinner) const {}
    template <int NC, class T> BATCHLAS_DLA_INLINE T pivot(int j, const Frag<T, NC>& rA, PivotWinner w) const {
        return px.bcast(rA[j], static_cast<std::uint32_t>(w.src));
    }
    template <int NC, class T, class F>
    BATCHLAS_DLA_INLINE void for_right(int j, int ncols, const Frag<T, NC>& rA, PivotWinner w, F&& f) const {
BATCHLAS_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, px.bcast(rA[k], static_cast<std::uint32_t>(w.src)));
        }
    }
};
template <class Px> BATCHLAS_DLA_INLINE BcastShuffle<Px> bcast_shuffle(const Px& px) { return {px}; }

}  // namespace getf2_detail

// ---- ?GETF2, explicit row interchanges ----------------------------------------------------------
// In place on A (mat_wide: legacy 64-bit addressing) for kmax steps (NOT min(m, n)): iamax, ipiv,
// swap of the WHOLE rows (only columns >= k is the classic silently wrong LU), rscl, rank-1
// update; barriers B1 (Scope::Group) to B4. Ex: RtGroup<Scope::Sub | Scope::Group>. info: the
// leader's running first failure (global, 1-based); the first EXACT zero pivot, no epsilon,
// leaves its column unscaled. vs / is: nteams slots of local memory for the pivot exchange.
template <class Ex, class MA, class VS, class IS>
BATCHLAS_DLA_INLINE std::int32_t getf2(const Ex& ex, const MA& a, int kmax, IpivSink ipiv, std::int32_t info, const VS& vs,
                                       const IS& is) {
    using D = typename MA::value_type;
    constexpr Mem fence = MA::mem == Mem::Local ? Mem::Local : Mem::Global;
    const int m = a.rows();
    const int lanes = detail::team_size_of(ex);  // read once, as the hand kernel did
    for (int k = 0; k < kmax; ++k) {
        int p = getf2_detail::iamax_col(ex, a, k, k, lanes, vs, is);
        if (p >= m) p = k;  // unreachable: k < m, so column k always has a row
        if (ex.leader()) ipiv.set(k, p);
        if (p != k) swap(ex, a.row(k), a.row(p));
        ex.barrier(fence);  // B2
        const D d = a(k, k);
        if (Arith<D>::is_zero(d)) {
            if (ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + k + 1);
        } else {
            getf2_detail::rscl_col(ex, d, a, k + 1, k);
        }
        ex.barrier(fence);  // B3
        getf2_detail::ger_trailing(ex, a, k);
        ex.barrier(fence);  // B4
    }
    return info;
}

// Both residencies' entry; `info_item` is READ too (first failure wins): launchers zero it.
template <class Ex, class MA, class ValAcc, class IdxAcc>
inline void getf2_panel_device(const Ex& ex, const MA& A, int kmax, int* piv_item, int piv_base, std::int32_t* info_item,
                               ValAcc rval, IdxAcc ridx) {
    std::int32_t info = ex.leader() ? *info_item : 0;
    info = getf2(ex, A, kmax, IpivSink{piv_item, piv_base}, info, rval, ridx);
    if (ex.leader()) *info_item = info;
}

// ---- ?GETF2 on a RowFrag, pivoting by lazy relabel -------------------------------------------------
// Each item holds one matrix row in rA and its logical row index in `rowid`; an interchange swaps
// two labels and moves no data. After the relabel the winner of column j is the item with
// rowid == j, so `act = rowid > j` keeps it from writing rA while others read its row. Policies
// (getf2_detail has the stock ones):
//   Piv: argmax(j, mag, rowid) -> PivotWinner (NaN mapped by the policy), record(j, w) (ipiv),
//        on_zero(zero, j) (info)
//   Bc:  publish(j, rA, rowid, w) (make the pivot row readable), pivot(j, rA, w) -> T,
//        for_right(j, ncols, rA, w, f) calling f(k, u_k) for the pivot row's j < k < ncols
// THREE SPELLINGS ARE SILENT WRONG ANSWERS OR SILENT SPILLS: the j loop's COMPILE-TIME bound with
// `continue` (never `break`; kmax must be uniform, collectives are skipped), publishing the pivot
// row AFTER the relabel, and selecting a complex scalar by if/else, never `?:`.
// evidence: docs/perf/lu.md#the-register-panel-leaf
template <int NC, class T, class Piv, class Bc>
BATCHLAS_DLA_INLINE void getf2_relabel(Frag<T, NC>& rA, int& rowid, int kmax, int ncols, Piv& piv, Bc& bc) {
    using Ar = arith_t<arith::Fma, T>;
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < NC; ++j) {
        if (j >= kmax) continue;
        const getf2_detail::PivotWinner w = piv.argmax(j, abs1<T>(rA[j]), rowid);  // ?IAMAX
        piv.record(j, w);
        if (rowid == w.row) rowid = j;  // ?SWAP as two labels
        else if (rowid == j) rowid = w.row;
        bc.publish(j, rA, rowid, w);
        const T d = bc.pivot(j, rA, w);
        const bool zero = Ar::is_zero(d);  // EXACT zero, no epsilon
        piv.on_zero(zero, j);
        const T rc = Ar::recip(d);
        const bool use_mul = !zero && Ar::isfinite(rc) && !Ar::is_zero(rc);
        const bool act = (rowid > j);
        if (act) {
            if (use_mul) rA[j] = Ar::mul(rA[j], rc);
            else if (!zero) rA[j] = Ar::div(rA[j], d);  // ?GETF2's sfmin arm
        }
        // `k = j + 1` is a COMPILE-TIME lower bound once j is unrolled; unguarded by `zero`: a
        // zero pivot WAS the argmax, so every multiplier is 0 (?GER as a no-op).
        bc.for_right(j, ncols, rA, w, [&](int k, T u) {
            if (act) rA[k] = Ar::sub(rA[k], Ar::mul(rA[j], u));
        });
    }
}

}  // namespace batchlas::device
