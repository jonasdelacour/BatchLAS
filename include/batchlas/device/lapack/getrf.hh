// SPDX-License-Identifier: MIT
#pragma once
// ?GETF2, GETRF's own routine: by row interchanges (getf2) or lazy relabel (getf2_relabel), with a
// fused pivot search (one butterfly, one slot barrier); arith::Fma. detail: the policies and
// the steps core has no SASS-identical form of yet. evidence: docs/perf/lu.md#the-shape-of-the-getrf-device-code
// The public op getrf<Tier>(ex, A, ipiv, ws): Reg and Cta are the bodies the host tiny and cta
// routes launch; Blocked (any n) runs ?GETRF's panel loop on the in-group Level 3. detail
// holds the plans, sinks and tier bodies, and the LU family's Blocked configuration and plan,
// which getrs, gesv, getri and inv share (one executor and local layout for a fused kernel).

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>
#include <batchlas/util/group-collectives.hh>

namespace batchlas::device {

namespace detail {

// The pivot key of the tiny tiers: ORDERING field high (lowest wins), winner's LANE low. Every
// ordering field must stay distinct in a partition. evidence: docs/perf/lu.md#the-pivot-key-encoding
inline constexpr int kPivotKeyLaneBits = 8;
inline constexpr int pivot_key(int order_field, int lane) { return (order_field << kPivotKeyLaneBits) | lane; }
inline constexpr int pivot_key_order(int key) { return key >> kPivotKeyLaneBits; }
inline constexpr std::uint32_t pivot_key_lane(int key) {
    return static_cast<std::uint32_t>(key) & ((1u << kPivotKeyLaneBits) - 1u);
}

// ---- level 1/2 steps of ?GETF2 ------------------------------------------------------------------
// A(r0:m-1, c) /= d with ?GETF2's arms (SafeRecip: one reciprocal and a multiply when 1/d is
// finite and nonzero, else a division per element, the sfmin arm), the arm chosen OUTSIDE the
// loop (core rscl selects per element). d != 0 is the caller's test.
template <class Ex, class MA>
BATCHLAS_DEVICE_INLINE void rscl_col(const Ex& ex, typename MA::value_type d, const MA& a, int r0, int c) {
    using T = typename MA::value_type;
    using Ar = arith_t<arith::Fma, T>;
    const auto s = SafeRecip<T>::make(d);
    const Range rows{r0, a.rows()};
    if (s.use_mul) for_each_owned(ex, rows, [&](int i) BATCHLAS_DEVICE_BODY { a(i, c) = Ar::mul(a(i, c), s.r); });
    else for_each_owned(ex, rows, [&](int i) BATCHLAS_DEVICE_BODY { a(i, c) = Ar::div(a(i, c), s.d); });
}

// A(k+1:, k+1:) -= A(k+1:, k) A(k, k+1:), flattened (docs/perf/lu.md#lu-negative-results), two
// roundings (sub of mul); written out, not a for_each_owned_2d lambda: through the lambda the
// panels drift a VOTE, +4 REG (sm_89).
template <class Ex, class MA> BATCHLAS_DEVICE_INLINE void ger_trailing(const Ex& ex, const MA& a, int k) {
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
// argmax_allreduce (B1). The leader writes ipiv and keeps info (global, 1-based, sticky).
// evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
template <class Ex, class VS, class IS> struct PivotRowsGroup {
    const Ex& ex;
    VS vs;
    IS is;
    int m;
    int lanes;
    IpivSink ipiv;
    std::int32_t& info;
    template <class R> BATCHLAS_DEVICE_INLINE PivotWinner argmax(int j, R mag, int rowid) const {
        const bool cand = (rowid >= j);
        const ArgMax<R> a = argmax_allreduce(ex, ArgMax<R>{cand ? rowid : m, (cand && (mag == mag)) ? mag : R(-1)}, alg::TieLowest{},
                                             argmax_slots(vs, is), m, R(-1), lanes);
        return {a.index >= m ? j : a.index, 0};  // >= m: unreachable, logical row j exists
    }
    BATCHLAS_DEVICE_INLINE void record(int j, PivotWinner w) const {
        if (ex.leader()) ipiv.set(j, w.row);
    }
    BATCHLAS_DEVICE_INLINE void on_zero(bool zero, int j) const {
        if (zero && ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + j + 1);
    }
};
template <class Ex, class VS, class IS>
BATCHLAS_DEVICE_INLINE PivotRowsGroup<Ex, VS, IS> pivot_rows_group(const Ex& ex, VS vs, IS is, int m, IpivSink ipiv,
                                                                std::int32_t& info) {
    return {ex, vs, is, m, detail::team_size_of(ex), ipiv, info};
}

// Piv for one matrix per N-lane Partition (the Reg tier): the (magnitude, key) pair argmax,
// XorUp. A pad row is kept out by the tie-break DIRECTION (`rowid + N` sorts every
// non-candidate below every candidate), not by the `rowid < n` mask; the `mag == mag` map IS
// load-bearing. Lane j keeps ipiv[j]; info is partition-uniform (the pivot is broadcast).
// evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
template <int N, class Px> struct PivotPartition {
    const Px& px;
    int n;
    int lane;
    int& my_piv;
    std::int32_t& linfo;
    template <class R> BATCHLAS_DEVICE_INLINE PivotWinner argmax(int j, R mag, int rowid) const {
        const bool cand = (rowid >= j) && (rowid < n);
        const ArgMax<R> a = argmax_allreduce(px, ArgMax<R>{pivot_key(cand ? rowid : (rowid + N), lane), (cand && (mag == mag)) ? mag : R(-1)},
                                             alg::TieLowest{}, None{}, 0, R(-1));
        return {pivot_key_order(a.index), static_cast<int>(pivot_key_lane(a.index))};
    }
    BATCHLAS_DEVICE_INLINE void record(int j, PivotWinner w) const {
        if (lane == j) my_piv = w.row;
    }
    BATCHLAS_DEVICE_INLINE void on_zero(bool zero, int j) const {
        if (zero && linfo == 0) linfo = static_cast<std::int32_t>(j + 1);
    }
};

// Bc through a local-memory row sx (>= NC elements): the item whose rowid is j writes the row,
// one barrier (B2); everyone reads it. No third barrier: docs/perf/lu.md#two-barriers-per-column
template <class Ex, class SX> struct BcastRowLocal {
    const Ex& ex;
    SX sx;
    template <int NC, class T> BATCHLAS_DEVICE_INLINE void publish(int j, const Frag<T, NC>& rA, int rowid, PivotWinner) const {
        if (rowid == j) {
BATCHLAS_UNROLL_FULL
            for (int k = j; k < NC; ++k) sx[static_cast<std::size_t>(k)] = rA[k];
        }
        ex.barrier(Mem::Local);  // B2
    }
    template <int NC, class T> BATCHLAS_DEVICE_INLINE T pivot(int j, const Frag<T, NC>&, PivotWinner) const {
        return sx[static_cast<std::size_t>(j)];
    }
    template <int NC, class T, class F>
    BATCHLAS_DEVICE_INLINE void for_right(int j, int ncols, const Frag<T, NC>&, PivotWinner, F&& f) const {
BATCHLAS_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, static_cast<T>(sx[static_cast<std::size_t>(k)]));
        }
    }
};
template <class Ex, class SX> BATCHLAS_DEVICE_INLINE BcastRowLocal<Ex, SX> bcast_row_local(const Ex& ex, SX sx) {
    return {ex, sx};
}

// Bc by shuffles from the winner's lane (w.src) of a Partition: no memory, no barrier, one
// select per element (two per complex).
template <class Px> struct BcastShuffle {
    const Px& px;
    template <int NC, class T> BATCHLAS_DEVICE_INLINE void publish(int, const Frag<T, NC>&, int, PivotWinner) const {}
    template <int NC, class T> BATCHLAS_DEVICE_INLINE T pivot(int j, const Frag<T, NC>& rA, PivotWinner w) const {
        return px.bcast(rA[j], static_cast<std::uint32_t>(w.src));
    }
    template <int NC, class T, class F>
    BATCHLAS_DEVICE_INLINE void for_right(int j, int ncols, const Frag<T, NC>& rA, PivotWinner w, F&& f) const {
BATCHLAS_UNROLL_FULL
        for (int k = j + 1; k < NC; ++k) {
            if (k >= ncols) continue;
            f(k, px.bcast(rA[k], static_cast<std::uint32_t>(w.src)));
        }
    }
};
template <class Px> BATCHLAS_DEVICE_INLINE BcastShuffle<Px> bcast_shuffle(const Px& px) { return {px}; }

}  // namespace detail

// ---- ?GETF2, explicit row interchanges ----------------------------------------------------------
// In place on A (mat_wide: legacy 64-bit addressing) for kmax steps (NOT min(m, n)): iamax, ipiv,
// swap of the WHOLE rows (only columns >= k is the classic silently wrong LU), rscl, rank-1
// update; barriers B1 (Scope::Group) to B4. Ex: RtGroup<Scope::Sub | Scope::Group>. info: the
// leader's running first failure (global, 1-based); the first EXACT zero pivot, no epsilon,
// leaves its column unscaled. vs / is: nteams slots of local memory for the pivot exchange.
template <class Ex, class MA, class VS, class IS>
BATCHLAS_DEVICE_INLINE std::int32_t getf2(const Ex& ex, const MA& a, int kmax, IpivSink ipiv, std::int32_t info, const VS& vs,
                                       const IS& is) {
    using D = typename MA::value_type;
    constexpr Mem fence = MA::mem == Mem::Local ? Mem::Local : Mem::Global;
    const int m = a.rows();
    const int lanes = detail::team_size_of(ex);  // read once, as the hand kernel did
    for (int k = 0; k < kmax; ++k) {
        int p = iamax<alg::NanSkip>(ex, [&](int i) -> D { return a(i, k); }, Range{k, m}, alg::TieLowest{}, argmax_slots(vs, is), None{}, m,
                                    lanes)
                    .index;
        if (p >= m) p = k;  // unreachable: k < m, so column k always has a row
        if (ex.leader()) ipiv.set(k, p);
        if (p != k) swap(ex, a.row(k), a.row(p));
        ex.barrier(fence);  // B2
        const D d = a(k, k);
        if (Arith<D>::is_zero(d)) {
            if (ex.leader() && info == 0) info = static_cast<std::int32_t>(ipiv.base + k + 1);
        } else {
            detail::rscl_col(ex, d, a, k + 1, k);
        }
        ex.barrier(fence);  // B3
        detail::ger_trailing(ex, a, k);
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
// rowid == j, so `act = rowid > j` keeps it from writing rA while others read its row. Policies:
//   Piv: argmax(j, mag, rowid) -> PivotWinner (NaN mapped by the policy), record(j, w) (ipiv),
//        on_zero(zero, j) (info)
//   Bc:  publish(j, rA, rowid, w) (make the pivot row readable), pivot(j, rA, w) -> T,
//        for_right(j, ncols, rA, w, f) calling f(k, u_k) for the pivot row's j < k < ncols
// THREE SPELLINGS ARE SILENT WRONG ANSWERS OR SILENT SPILLS: the j loop's COMPILE-TIME bound with
// `continue` (never `break`; kmax must be uniform, collectives are skipped), publishing the pivot
// row AFTER the relabel, and selecting a complex scalar by if/else, never `?:`.
// evidence: docs/perf/lu.md#the-register-panel-leaf
template <int NC, class T, class Piv, class Bc>
BATCHLAS_DEVICE_INLINE void getf2_relabel(Frag<T, NC>& rA, int& rowid, int kmax, int ncols, Piv& piv, Bc& bc) {
    using Ar = arith_t<arith::Fma, T>;
BATCHLAS_UNROLL_FULL
    for (int j = 0; j < NC; ++j) {
        if (j >= kmax) continue;
        const detail::PivotWinner w = piv.argmax(j, abs1<T>(rA[j]), rowid);  // ?IAMAX
        piv.record(j, w);
        if (rowid == w.row) rowid = j;  // ?SWAP as two labels
        else if (rowid == j) rowid = w.row;
        bc.publish(j, rA, rowid, w);
        const auto sr = SafeRecip<T>::make(bc.pivot(j, rA, w));  // zero: EXACT, no epsilon
        piv.on_zero(sr.zero, j);
        const bool act = (rowid > j);
        if (act) {  // the arms written out: apply()'s value form adds a spill (cfloat n = 24)
            if (sr.use_mul) rA[j] = Ar::mul(rA[j], sr.r);
            else if (!sr.zero) rA[j] = Ar::div(rA[j], sr.d);  // ?GETF2's sfmin arm
        }
        // `k = j + 1` is a COMPILE-TIME lower bound once j is unrolled; unguarded by `zero`: a
        // zero pivot WAS the argmax, so every multiplier is 0 (?GER as a no-op).
        bc.for_right(j, ncols, rA, w, [&](int k, T u) {
            if (act) rA[k] = Ar::sub(rA[k], Ar::mul(rA[j], u));
        });
    }
}

// ==== the public op: getrf<Tier>(ex, A, ipiv, ws [, live]) (v22 design C.2-C.7) =================
namespace routine {
struct getrf {
    static constexpr const char* name = "getrf";
};
}  // namespace routine

namespace detail {

inline constexpr int kRedSlots = 32;  // pivot-search slots: one per sub-group, WG <= 1024 at SG 32

// The Reg tier's ceiling: complex<double> stops at 16 (32 rows of a 16-byte scalar is 128
// registers before anything else). evidence: docs/perf/lu.md#d3-why-cdouble-stops-at-n16
template <class T> constexpr int reg_cap() { return (Arith<T>::is_complex && std::is_same_v<real_t<T>, double>) ? 16 : 32; }
constexpr int reg_lanes(int n) { return n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32; }

// The Cta tier's work-group: about four columns' worth of rows in [64, 512] (getrf_cta.cc's
// leaf width at an unlimited device maximum). evidence: docs/perf/lu.md#lu-negative-results
constexpr int leaf_wg(int n) {
    const int cols = n >= 4 ? 4 : (n < 1 ? 1 : n);
    const long long target = static_cast<long long>(n) * cols;
    int wg = 32;
    while (wg < target && wg < 512) wg <<= 1;
    return wg < 64 ? 64 : wg;
}

// The Cta tier's local memory: the tile at an ODD ld (the row interchange walks a row at
// stride ld; an even ld puts the whole walk in one bank), then the pivot-search slots.
template <class T> struct GetrfCtaPlan {
    WsSlot<T> tile;
    WsSlot<real_t<T>> vs;
    WsSlot<int> is;
    int ld;
    WorkspaceSize size;
};
template <class T> constexpr GetrfCtaPlan<T> getrf_cta_plan(int m, int n) {
    Plan p;
    const int ld = m | 1;
    const auto tile = p.take<T>(WsSpace::Local, static_cast<std::size_t>(ld) * static_cast<std::size_t>(n));
    const auto vs = p.take<real_t<T>>(WsSpace::Local, kRedSlots);
    const auto is = p.take<int>(WsSpace::Local, kRedSlots);
    return {tile, vs, is, ld, p.size()};
}


// An operand's scalar: its value_type, or what its (i, j) names (l3::Global has no value_type).
template <class M> struct lu_value {
    using type = std::remove_cvref_t<decltype(std::declval<const M&>()(0, 0))>;
};
template <class M>
    requires requires { typename M::value_type; }
struct lu_value<M> {
    using type = typename M::value_type;
};
template <class M> using lu_value_t = typename lu_value<M>::type;

// The LU family's Blocked Level 3 configuration: the tier's Alg when it is an l3 preset, else
// 32 x 32 tiles of 2 x 2 per item (256 items, device_lapack_l3_tests' shape).
template <class T, class Tier>
using lu_blocked_cfg_t = std::conditional_t<l3::Config<typename Tier::alg_type>, typename Tier::alg_type,
                                            l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>>;

// The LU family's Blocked local memory: the L3 stages, then getf2's pivot-search slots. No
// global workspace (A is factored in place). lu_blocked_take extends a caller's plan (inv's).
template <class T> struct LuBlockedPlan {
    WsSlot<T> smem;
    WsSlot<real_t<T>> vs;
    WsSlot<int> is;
    WorkspaceSize size;
};
template <class T, class Cfg> constexpr LuBlockedPlan<T> lu_blocked_take(Plan& p) {
    const auto smem = p.take<T>(WsSpace::Local, static_cast<std::size_t>(Cfg::smem_elems));
    const auto vs = p.take<real_t<T>>(WsSpace::Local, kRedSlots);
    const auto is = p.take<int>(WsSpace::Local, kRedSlots);
    return {smem, vs, is, p.size()};
}
template <class T, class Cfg> constexpr LuBlockedPlan<T> lu_blocked_plan() {
    Plan p;
    return lu_blocked_take<T, Cfg>(p);
}

// Where the Reg tier's info goes besides the return value: nowhere (the public op), or the
// launcher's slot (host routes, inside the same `live` branch as the factor store).
struct NoInfo {
    template <class Px> BATCHLAS_DEVICE_INLINE void operator()(const Px&, std::int32_t) const {}
};
struct PivPtr {
    std::int32_t* p;
    BATCHLAS_DEVICE_INLINE std::int32_t& operator()(int i) const { return p[i]; }
};

// ONE ELEMENT of the padded tile, per element DELIBERATELY: the one spelling that cannot take
// the address of the caller's register array. Pad rows and columns carry the IDENTITY.
template <class D, class MA> inline D load_pad_identity(const MA& a, int r, int c, int n, bool live) {
    if (live && r < n && c < n) return a(r, c);
    return (r == c) ? Arith<D>::one() : D{};
}

// The Reg tier: lane r of an N-lane Partition owns row r of a compile-time Frag<D, NC>, ?GETF2
// by lazy relabel, the pivot row through Bc (detail::bcast_shuffle, or a launcher's local
// memory policy). THREE INVARIANTS, each breaking SILENTLY: rA is never dynamically indexed;
// N divides the sub-group; every lane of the SUB-GROUP reaches every collective (a dead
// partition passes live = false and factors an identity; n is sub-group uniform).
// The factor is stored at row `rowid` (rows are relabelled, never moved); ipiv 1-based.
// `lane` is px.lid(), read by the caller before it builds the operands (the host route's order).
// evidence: docs/perf/lu.md#lu-the-tiny-tier
template <int NC, class Px, class MA, class PV, class Bc, class Sink>
BATCHLAS_DEVICE_INLINE std::int32_t getrf_reg_run(const Px& px, int lane, const MA& a, const PV& ipiv, int n, bool live,
                                         const Bc& bc, const Sink& sink) {
    using D = typename MA::value_type;
    Frag<D, NC> rA;
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < NC; ++c) rA[c] = load_pad_identity<D>(a, lane, c, n, live);
    int rowid = lane;   // WHICH matrix row this lane currently owns
    int my_piv = lane;  // lane j accumulates ipiv[j]
    std::int32_t linfo = 0;
    const detail::PivotPartition<Px::size, Px> piv{px, n, lane, my_piv, linfo};
    getf2_relabel(rA, rowid, n, n, piv, bc);
    if (live) {
BATCHLAS_UNROLL_FULL
        for (int k = 0; k < NC; ++k) {
            if (k >= n) continue;
            if (lane < n) a(rowid, k) = rA[k];  // not `rowid < n`: equivalent, and loop-invariant
        }
        if (lane < n) ipiv(lane) = my_piv + 1;
        sink(px, linfo);
    }
    return linfo;
}

// The Cta tier: stage A into the padded local tile, ?GETF2, store back; info at the leader
// (getf2_panel_device's contract: *info_item is read, first failure wins).
template <class Ex, class MG, class MT, class VS, class IS>
BATCHLAS_DEVICE_INLINE void getrf_cta_run(const Ex& ex, const MG& ag, const MT& at, int kmax, int* piv_item, int piv_base,
                                 std::int32_t* info_item, const VS& vs, const IS& is) {
    lacpy<Part::Full, Transpose::NoTrans, alg::DivMod>(ex, ag, at);  // stage into the padded ld
    ex.barrier();                                                    // B0
    getf2_panel_device(ex, at, kmax, piv_item, piv_base, info_item, vs, is);
    ex.barrier();                                                    // B5
    lacpy<Part::Full, Transpose::NoTrans, alg::DivMod>(ex, at, ag);  // store
}

// The Blocked tier: ?GETRF's right-looking loop over NB-wide panels of A in global memory. The
// panel is ?GETF2 in place (the host blocked route's global leaf), its interchanges go to the
// columns left and right of it (laswp), then U12 := L11^-1 A12 (in-group trsm, Unit) and
// A22 -= L21 U12 (in-group gemm). getf2 ends on a global barrier (ipiv and the panel are
// visible), one global barrier follows the interchanges, trsm and gemm end on their own. info:
// the leader's first exact zero pivot (global, 1-based, sticky), made uniform at the end.
template <int NB, class Cfg, class Ex, class MA>
BATCHLAS_DEVICE_INLINE std::int32_t factor_blocked(const Ex& ex, const MA& a_in, std::int32_t* ipiv, const Workspace& w) {
    using D = typename Cfg::value_type;
    const auto g = l3::global_of(a_in);
    const int m = g.m, n = g.n, mn = m < n ? m : n;
    const auto a = mat_wide(g.p, m, n, g.ld);
    constexpr auto pl = lu_blocked_plan<D, Cfg>();
    const l3::Smem<Cfg> sm{carve_local(w, pl.smem).get_raw()};
    const auto vs = carve_local(w, pl.vs);
    const auto is = carve_local(w, pl.is);
    const PivPtr piv{ipiv};
    std::int32_t info = 0;
    for (int j = 0; j < mn; j += NB) {
        const int jb = mn - j < NB ? mn - j : NB;
        const int j2 = j + jb;
        info = getf2(ex, a.sub(j, j, m - j, jb), jb, IpivSink{ipiv + j, j}, info, vs, is);
        laswp(ex, a.sub(0, 0, m, j), piv, j, j2);
        laswp(ex, a.sub(0, j2, m, n - j2), piv, j, j2);
        ex.barrier(Mem::Global);
        if (j2 >= n) continue;
        const auto u12 = l3::global_of(a.sub(j, j2, jb, n - j2));
        trsm<Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::Unit, Cfg, NB>(ex, l3::global_of(a.sub(j, j, jb, jb)), u12, One{}, sm);
        if (j2 < m)
            gemm<Transpose::NoTrans, Transpose::NoTrans, Cfg>(ex, l3::global_of(a.sub(j2, j, m - j2, jb)), u12,
                                                              l3::global_of(a.sub(j2, j2, m - j2, n - j2)), NegOne{}, One{}, sm);
    }
    return ex.uniform(info);
}

}  // namespace detail

/// getrf's Reg<N>: an N-lane (or wider) Partition, n <= N (complex<double>: N <= 16), no workspace.
template <int N, bool Exact, class T>
    requires(N <= detail::reg_cap<T>())
struct tier_info<routine::getrf, tier::Reg<N, Exact>, T> {
    static constexpr int part_size = detail::reg_lanes(N);
    static constexpr int wg_size = 32, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && (Exact ? n == N : n <= N); }
    static constexpr WorkspaceSize buffer_size(int) { return {}; }
};
/// getrf's Cta<N>: one WorkGroup (or RtGroup) per matrix; the tile and 32 slots in local memory.
template <int N, class T> struct tier_info<routine::getrf, tier::Cta<N>, T> {
    static constexpr int wg_size = detail::leaf_wg(N), sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int n) { return detail::getrf_cta_plan<T>(n, n).size; }
};
/// getrf's Blocked<NB>: any m x n; one WorkGroup of the L3 configuration's items (256 by
/// default); A in global memory, the L3 stages and 32 slots in local memory, no global workspace.
template <int NB, class Alg, class T> struct tier_info<routine::getrf, tier::Blocked<NB, Alg>, T> {
    using cfg = detail::lu_blocked_cfg_t<T, tier::Blocked<NB, Alg>>;
    static_assert(std::is_same_v<typename cfg::value_type, T>, "getrf: tier::Blocked's l3 configuration is for another scalar type");
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int) { return detail::lu_blocked_plan<T, cfg>().size; }
};
template <class T> struct tiers<routine::getrf, T> {
    using type = std::conditional_t<
        (detail::reg_cap<T>() >= 32),
        tier::list<tier::Reg<8>, tier::Reg<16>, tier::Reg<32>, tier::Cta<64>, tier::Cta<96>, tier::Cta<128>, tier::Blocked<32>>,
        tier::list<tier::Reg<8>, tier::Reg<16>, tier::Cta<64>, tier::Cta<96>, tier::Cta<128>, tier::Blocked<32>>>;
};

/// The workspace of one n x n matrix for getrf<Tier> (host and device; constexpr).
template <class Tier, class T> constexpr WorkspaceSize getrf_buffer_size(int n) {
    return tier_info<routine::getrf, resolve_tier_t<routine::getrf, T, Tier>, T>::buffer_size(n);
}

/// @brief ?GETRF in a kernel: \f$A = PLU\f$ in place for one matrix, by the executor `ex`.
///
/// Collective: every item of `ex` calls it with the same arguments; A must be visible to all
/// items on entry and is visible to all on return. One tier per kernel (choose it on the host
/// with with_tier<routine::getrf, T>, never by a run-time branch between two tiers).
/// - `tier::Reg<N>`: `ex` a Partition<P> (P >= N, P divides the sub-group); A square, n <= N,
///   n uniform across the sub-group; a dead partition passes `live = false` (it factors an
///   identity and stores nothing; every lane still calls). `ws` unused.
/// - `tier::Cta<N>`: `ex` a WorkGroup<WG, SG> or RtGroup; A m x n with
///   `ws.local` >= getrf_buffer_size<Tier, T>(max(m, n)) bytes; `live` executor-uniform.
/// - `tier::Blocked<NB>`: `ex` a WorkGroup<tier_info::wg_size, 32>; A m x n of any size in
///   global memory (Mat over T*, ItemMat or l3::Global), factored in place by NB-wide panels;
///   ipiv in memory every item reads; `ws.local` >= getrf_buffer_size<Tier, T>(n) bytes, no
///   global workspace; `live` executor-uniform. Alg: an l3::cfg preset for the trailing updates.
/// - `tier::upto<N>`: the first tier of tiers_t<routine::getrf, T> that admits N.
/// @param[in]     ex    the executor
/// @param[in,out] a     this matrix (any operand: Mat, ItemMat, ...); overwritten with L and U
/// @param[out]    ipiv  this matrix's min(m, n) pivots, 1-based int32
/// @param[in]     ws    this matrix's workspace slice
/// @param[in]     live  false: a dead matrix of the launch (see the tiers)
/// @return LAPACK's info, uniform across `ex`: 0, or the first k with \f$U_{kk} = 0\f$ exactly
template <class Tier, class Ex, class MA>
BATCHLAS_DEVICE_INLINE std::int32_t getrf(const Ex& ex, const MA& a, std::int32_t* ipiv, const Workspace& ws = {},
                                       bool live = true) {
    using D = detail::lu_value_t<MA>;
    using Tr = resolve_tier_t<routine::getrf, D, Tier>;
    static_assert(HasTier<routine::getrf, Tr, D>, "getrf: this tier is not offered for this scalar type");
    if constexpr (tier::is_reg_v<Tr>) {
        static_assert(Tr::n_max <= Ex::size, "getrf: tier::Reg<N> needs a Partition of at least N lanes");
        return detail::getrf_reg_run<Tr::n_max>(ex, ex.lid(), a, detail::PivPtr{ipiv}, a.cols(), live,
                                                detail::bcast_shuffle(ex), detail::NoInfo{});
    } else if constexpr (tier::is_cta_v<Tr>) {
        if (!live) return 0;
        const int m = a.rows(), n = a.cols();
        const auto pl = detail::getrf_cta_plan<D>(m, n);
        const auto at = mat_wide<Mem::Local>(carve_local(ws, pl.tile).get_raw(), m, n, pl.ld);
        std::int32_t info = 0;
        detail::getrf_cta_run(ex, a, at, m < n ? m : n, ipiv, 0, &info, carve_local(ws, pl.vs), carve_local(ws, pl.is));
        return ex.uniform(info);
    } else {
        using Cfg = detail::lu_blocked_cfg_t<D, Tr>;
        static_assert(Ex::size == Cfg::threads, "getrf: tier::Blocked runs on a WorkGroup of tier_info::wg_size items");
        if (!live) return 0;
        return detail::factor_blocked<Tr::nb, Cfg>(ex, a, ipiv, ws);
    }
}

}  // namespace batchlas::device
