// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_lapack_ldl
/// @brief ?SYTRS in a kernel: the solve \f$A X = B\f$ for one symmetric indefinite matrix from
/// sytrf()'s Bunch-Kaufman factor and pivots, X overwriting B.
///
/// The routine is sytrs(); its routine tag is routine::sytrs. Its tiers are `tier::Cta<32>`,
/// `tier::Cta<64>`, `tier::Cta<128>` (the converted ?SYTRS2 form in local memory) and
/// `tier::Blocked<1>` (?SYTRS on the factor in global memory, any n); `tier::Blocked<NB>` with
/// NB > 1 (?SYTRS2 on the in-group trsm) is offered but not in `tiers_t`. The kernels
/// sytrs_col(), syconv() and syconv_gather() are in @ref api_device_lowlevel.
// From sytrf's factor in the logical lower form (lapack/sytrf.hh's reversal). sytrs_col is
// ?sytrs on one right-hand side. The converted form (?sytrs2) first runs ?syconv (each step's interchange applied to the earlier
// columns of L, the \f$2 \times 2\f$ off-diagonals moved out), after which \f$A = P L D L^{T} P^{T}\f$
// with L unit lower and P one permutation; detail::sytrs_cols then solves one right-hand side per
// sub-group with lane i owning logical rows i, i + 32, ... in a lane-cyclic Frag: the gather,
// both sweeps (broadcast form, no reduction) and the scatter take no barrier.
// evidence: docs/perf/sytrf.md#sytrs-the-converted-solve

#include <cstddef>
#include <utility>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/sytrf.hh>

namespace batchlas::device {

namespace detail {

// ?SYTRS's 2 x 2 solve: x0 / x1 and d0 / d1 the first / second row's value and diagonal, off the
// block's off-diagonal. ldl_block2 divides once per block; ldl_block2_row gives one row.
template <class Scalar> struct LdlBlock2 {
    Scalar akm1, ak, denom, b0, b1;
};
template <class Scalar> BATCHLAS_DEVICE_INLINE LdlBlock2<Scalar> ldl_block2(Scalar x0, Scalar x1, Scalar d0, Scalar d1,
                                                                            Scalar off) {
    using A = arith_t<arith::Fma, Scalar>;
    const Scalar akm1 = A::div(d0, off);
    const Scalar ak = A::div(d1, off);
    const Scalar denom = A::sub(A::mul(akm1, ak), A::one());
    return {akm1, ak, denom, A::div(x0, off), A::div(x1, off)};
}
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar ldl_block2_row(const LdlBlock2<Scalar>& b, bool second) {
    using A = arith_t<arith::Fma, Scalar>;
    return second ? A::div(A::sub(A::mul(b.akm1, b.b1), b.b0), b.denom) : A::div(A::sub(A::mul(b.ak, b.b0), b.b1), b.denom);
}

// Row t of D^-1 x for a block of kind kd (0: 1 x 1; 1 / 2: the first / second row of a 2 x 2):
// x is row t's value, xp its partner's, dtt and dpp their diagonals, off the block's
// off-diagonal. Bitwise sytrs_col's rows.
template <class Scalar> BATCHLAS_DEVICE_HELPER Scalar ldl_dsolve_row(int kd, Scalar x, Scalar xp, Scalar dtt,
                                                                     Scalar dpp, Scalar off) {
    using A = arith_t<arith::Fma, Scalar>;
    if (kd == 0) return A::mul(x, A::recip(dtt));
    const bool first = kd == 1;
    return ldl_block2_row(ldl_block2(first ? x : xp, first ? xp : x, first ? dtt : dpp, first ? dpp : dtt, off), !first);
}

// The staged factor in local memory, logical indices. `ps` is the logical ipiv: +(p + 1) a \f$1 \times 1\f$
// step with partner p, -(p + 1) both rows of a \f$2 \times 2\f$ step whose second row is swapped with p, 0
// for an entry sytrf cannot have written. `perm[t]`: the row of B that \f$P^{T}\f$ brings to row t.
// `kind[t]`: 0 a \f$1 \times 1\f$ block, 1 / 2 the first / second row of a \f$2 \times 2\f$ block. `e[k]`: the
// off-diagonal of the \f$2 \times 2\f$ block starting at k.
template <class D>
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

// x := D^-1 x for D block diagonal with 1 x 1 / 2 x 2 blocks: kind[t] as ldl_dsolve_row's, the
// diagonal on d(t, t), e[k] the off-diagonal of the block starting at k. The partner rows come
// from neighbor_exchange.
template <class Ex, class MD, class Scalar, int R>
BATCHLAS_DEVICE_INLINE void ldl_dsolve_frag(const Ex& ex, const MD& d, const int* kind, const Scalar* e, int n, Frag<Scalar, R, Cyclic>& x) {
    static_assert(Ex::size == 32, "sytrs ldl_dsolve_frag: a 32-lane executor");
    Frag<Scalar, R, Cyclic> up, dn;
    neighbor_exchange(ex, x, up, dn);
    const int lane = ex.lane();
    BATCHLAS_UNROLL_FULL
    for (int r = 0; r < R; ++r) {
        const int t = lane + 32 * r;
        if (t >= n) continue;
        const int kd = kind[t];
        const int p = kd == 1 ? t + 1 : kd == 2 ? t - 1 : t;
        // One load at a selected index: a select of two local pointers fails Metal compilation.
        const Scalar off = kd == 0 ? arith_t<arith::Fma, Scalar>::one() : e[kd == 1 ? t : t - 1];
        x[r] = ldl_dsolve_row(kd, x[r], select(kd == 1, up[r], dn[r]), d(t, t), d(p, p), off);
    }
}

// ?SYCONV on column j of a staged lower factor (one item): the interchanges of the steps after j
// in step order (a 2 x 2 step at k swaps row k + 1), and the 2 x 2 off-diagonal of a block
// starting at j moved to e[j]. ps is the logical ipiv (+(p + 1) / -(p + 1)). e is never null:
// a null test on a local e breaks AdaptiveCpp's Metal address-space inference.
template <class Scalar, class ML> BATCHLAS_DEVICE_HELPER void syconv_col(const ML& l, const int* ps, int n, int j,
                                                                         Scalar* e) {
    for (int k = 0; k < n;) {
        const int v = ps[k];
        if (v > 0) {
            if (k > j) std::swap(l(k, j), l(v - 1, j));
            k += 1;
        } else {
            if (k == j) {
                e[j] = l(j + 1, j);
                l(j + 1, j) = arith_t<arith::Fma, Scalar>::zero();
            }
            if (k > j) std::swap(l(k + 1, j), l(-v - 1, j));
            k += 2;
        }
    }
}

}  // namespace detail

/// @ingroup api_device_lowlevel
/// @brief ?SYTRS for one right-hand side, lower form, on the unconverted n x n factor `S`.
///
/// Not covered by the compatibility promise: this is the body of sytrs()'s `tier::Blocked<1>` and
/// of the host `wg` route. A pivot list sytrf() cannot have written (an entry outside [1, n] in
/// magnitude, an unpaired 2 x 2 entry) would index `b` out of bounds, so that item's X is NaN
/// instead. Every item of `ex` calls it with the same arguments.
/// @param[in]     ex    the executor
/// @param[in]     S     sytrf()'s factor, any matrix operand; Upper passes the reversed view
/// @param[in]     piv   `piv(phys_k)` reads LAPACK's ipiv entry (a `Vec<int>`, or any callable)
/// @param[in,out] b     the right-hand side, overwritten with x; Upper passes the reversed view
/// @param[in]     n     the order
/// @param[in]     rev   true: `S` and `b` are index-reversed views of an Upper factor
/// @param[in]     sum   the reduction scratch (a detail type)
template <class Ex, class MS, class PV, class VB, class Scr>
inline void sytrs_col(const Ex& ex, const MS& S, const PV& piv, const VB& b, int n, bool rev, Scr sum) {
    using D = typename MS::value_type;
    using A = arith_t<arith::Fma, D>;
    auto phys = [&](int i) { return rev ? n - 1 - i : i; };
    // Every lane walks the list, lane 0 decides.
    bool ok = true;
    for (int k = 0; k < n && ok;) {
        const int raw = piv(phys(k));
        ok = raw != 0 && raw >= -n && raw <= n && (raw > 0 || (k + 1 < n && piv(phys(k + 1)) == raw));
        k += raw > 0 ? 1 : 2;
    }
    if (ex.uniform(ok ? 1 : 0) == 0) {
        for_each_owned(ex, n, [&](int i) BATCHLAS_DEVICE_BODY { b(i) = qnan<D>(); });
        sync<Scope::Group>(ex, Mem::Global);
        return;
    }
    // P L y = b, then D z = y
    int k = 0;
    while (k < n) {
        const int raw = piv(phys(k));
        if (raw > 0) {
            const int kp = phys(raw - 1);
            if (ex.leader() && kp != k) std::swap(b(k), b(kp));
            sync<Scope::Group>(ex, Mem::Global);
            const D bk = b(k);
            sync<Scope::Group>(ex, Mem::Global);
            for_each_owned(ex, Range{k + 1, n}, [&](int i) BATCHLAS_DEVICE_BODY { b(i) = A::sub(b(i), A::mul(S(i, k), bk)); });
            if (ex.leader()) b(k) = detail::ldl_dsolve_row(0, bk, bk, S(k, k), S(k, k), S(k, k));
            sync<Scope::Group>(ex, Mem::Global);
            k += 1;
        } else {
            const int kp = phys(-raw - 1);
            if (ex.leader() && kp != k + 1) std::swap(b(k + 1), b(kp));
            sync<Scope::Group>(ex, Mem::Global);
            const D bk = b(k), bk1 = b(k + 1);
            sync<Scope::Group>(ex, Mem::Global);
            for_each_owned(ex, Range{k + 2, n}, [&](int i) BATCHLAS_DEVICE_BODY {
                b(i) = A::sub(A::sub(b(i), A::mul(S(i, k), bk)), A::mul(S(i, k + 1), bk1));
            });
            if (ex.leader()) {
                const auto blk = detail::ldl_block2(bk, bk1, S(k, k), S(k + 1, k + 1), S(k + 1, k));
                b(k) = detail::ldl_block2_row(blk, false);
                b(k + 1) = detail::ldl_block2_row(blk, true);
            }
            sync<Scope::Group>(ex, Mem::Global);
            k += 2;
        }
    }
    // L^T P^T x = z
    k = n - 1;
    while (k >= 0) {
        const int raw = piv(phys(k));
        D p0 = A::zero(), p1 = A::zero();
        for_each_owned(ex, Range{k + 1, n}, [&](int i) BATCHLAS_DEVICE_BODY {
            p0 = A::add(p0, A::mul(S(i, k), b(i)));
            if (raw < 0) p1 = A::add(p1, A::mul(S(i, k - 1), b(i)));
        });
        const D s0 = detail::sum_reduce(ex, sum, p0);
        const D s1 = raw < 0 ? detail::sum_reduce(ex, sum, p1) : A::zero();
        sync<Scope::Group>(ex, Mem::Global);
        if (ex.leader()) {
            b(k) = A::sub(b(k), s0);
            if (raw < 0) b(k - 1) = A::sub(b(k - 1), s1);
            const int kp = phys((raw > 0 ? raw : -raw) - 1);
            if (kp != k) std::swap(b(k), b(kp));
        }
        sync<Scope::Group>(ex, Mem::Global);
        k -= raw > 0 ? 1 : 2;
    }
}

/// @ingroup api_device_lowlevel
/// @brief ?SYCONV with validation and the composed permutation, for one matrix on the work-group
/// `ex`: the staged factor `L` is converted in place.
///
/// Not covered by the compatibility promise: this is a step of sytrs()'s `tier::Cta`. An invalid
/// list leaves `L` as staged, the permutation the identity and every block 1 x 1; the caller then
/// writes NaN. Every item of `ex` calls it with the same arguments.
/// @param[in]     ex      the work-group
/// @param[in,out] L       the staged n x n lower factor (any matrix operand)
/// @param[in]     ps_v    its logical ipiv
/// @param[out]    perm_v  the composed permutation (a `Vec` view of n)
/// @param[out]    kind_v  each row's block kind: 0 a 1 x 1 block, 1 or 2 the first or second row
///                        of a 2 x 2 block (a `Vec` view of n)
/// @param[out]    e_v     the off-diagonal of the 2 x 2 block starting at each row (a `Vec` view
///                        of n)
/// @param[out]    flag    one int every item reads
/// @return true when the pivot list is one sytrf() can have written
template <class Ex, class ML, class VP, class VI, class VK, class VE>
inline bool syconv(const Ex& ex, const ML& L, const VP& ps_v, const VI& perm_v, const VK& kind_v, const VE& e_v, int* flag) {
    using D = typename VE::value_type;
    const int n = ps_v.size();
    const int* const ps = &ps_v(0);
    int* const perm = &perm_v(0);
    int* const kind = &kind_v(0);
    D* const e = &e_v(0);
    if (ex.leader()) *flag = detail::piv_list_valid(ps, n) ? 1 : 0;
    sync<Scope::Group>(ex, Mem::Global);
    const bool ok = *flag != 0;
    if (ok) {
        for_each_owned(ex, n, [&](int j) BATCHLAS_DEVICE_BODY { detail::syconv_col<D>(L, ps, n, j, e); });
        if (ex.leader()) detail::build_perm(ps, n, perm, kind);
    } else {
        for_each_owned(ex, n, [&](int t) BATCHLAS_DEVICE_BODY {
            perm[t] = t;
            kind[t] = 0;
        });
    }
    sync<Scope::Group>(ex, Mem::Global);
    return ok;
}

/// @ingroup api_device_lowlevel
/// @brief ?SYCONV as a gather, for a factor in global memory: column j of `W` is column j of `G`
/// with its rows permuted by \f$\sigma_j\f$, the interchanges after step j composed
/// (\f$W(t, j) = G(\sigma_j(t), j)\f$, \f$t > j\f$).
///
/// Not covered by the compatibility promise: this is a step of sytrs()'s `tier::Blocked<NB>` and
/// of the host blocked route. `kind` must describe a valid pivot list. Every item of `ex` calls
/// it with the same arguments.
/// @tparam D  the scalar type
/// @param[out] W     the converted lower factor
/// @param[in]  G     sytrf()'s factor
/// @param[in]  n     the order
/// @param[in]  ps    the logical ipiv
/// @param[in]  kind  each row's block kind (0, 1 or 2)
/// @param[out] sig   n ints of scratch every item reads: the permutation
/// @param[out] inv   n ints of scratch: its inverse
/// @param[in]  ex    the work-group
// Walking j down, sigma gains one interchange per block start, applied by lane 0 in O(1) through
// its inverse; every read and write of a column is then one pass instead of one swap per step.
template <class D, class Dst, class Src, class Ex>
inline void syconv_gather(const Dst& W, const Src& G, int n, const int* ps, const int* kind, int* sig, int* inv,
                          const Ex& ex) {
    for_each_owned(ex, n, [&](int t) BATCHLAS_DEVICE_BODY {
        sig[t] = t;
        inv[t] = t;
    });
    sync<Scope::Group>(ex, Mem::Global);
    for (int j = n - 1; j >= 0; --j) {
        const bool two = kind[j] == 1;
        for_each_owned(ex, Range{j + 1, n}, [&](int t) BATCHLAS_DEVICE_BODY {
            W(t, j) = two && t == j + 1 ? arith_t<arith::Fma, D>::zero() : G(sig[t], j);
        });
        sync<Scope::Group>(ex, Mem::Global);
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
        sync<Scope::Group>(ex, Mem::Global);
    }
}

namespace detail {

// L w = y, w := D^-1 w, L^T z = w on one 32-lane executor; x[r] holds logical row lane + 32 r.
// The unit-lower solves are core trsv<alg::LanePerRow> on the lane-cyclic Frag, sub of mul
// (no reduction, no barrier). Collective.
template <class Ex, class D, int R>
inline void sytrs_regs(const Ex& ex, const Staged<D>& S, Frag<D, R, Cyclic>& x) {
    static_assert(Ex::size == 32, "sytrs_regs: a 32-lane executor");
    trsv<Uplo::Lower, Transpose::NoTrans, Diag::Unit, alg::Div, alg::LanePerRow, arith::Fma, alg::Unfused>(ex, S, x, S.n);
    detail::ldl_dsolve_frag(ex, S, S.kind, S.e, S.n, x);
    trsv<Uplo::Lower, Transpose::Trans, Diag::Unit, alg::Div, alg::LanePerRow, arith::Fma, alg::Unfused>(ex, S, x, S.n);
}

// Every right-hand side of one item, sub-group `sgid` of `nsg` taking columns sgid, sgid + nsg, ...
// Row t is gathered from and scattered back to the same physical row of B, map(perm[t]), so no
// lane reads what another writes. An invalid list (ok false) gives an all-NaN X.
template <int R, class D>
inline void sytrs_cols(const sycl::sub_group& sg, int lane, int sgid, int nsg, const Staged<D>& S, bool ok, bool rev,
                       D* bb, std::ptrdiff_t ldb, int nrhs) {
    const SubGroup<32> ex{sg};
    const int n = S.n;
    for (int c = sgid; c < nrhs; c += nsg) {
        D* const col = bb + static_cast<std::ptrdiff_t>(c) * ldb;
        if (!ok) {
            for_each_owned(ex, n, [&](int t) BATCHLAS_DEVICE_BODY { col[t] = qnan<D>(); });
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

}  // namespace detail

// ---- public device API: sytrs<UL, Tier>(ex, A, ipiv, B, ws) -------------------------------------
// Tiers (one per kernel; with_tier picks one on the host for a run-time n):
//   tier::Cta<N>, N <= 128  the converted factor in local memory, one sub-group per right-hand
//                           side, x in registers (R = ceil(N / 32) per lane); the host `cta`
//                           launch's buckets are Cta<32>, Cta<64>, Cta<128>
//   tier::Blocked<1>        any n: ?sytrs on the factor in global memory, one column at a time
//                           (host `wg`'s body, sytrs_col); ex at most 256 items. Its sums
//                           reduce in an order fixed by ex's width: bit-identical to `wg` at
//                           the host's width min(256, roundup32(n)).
//   tier::Blocked<NB>, NB>1 any n: ?sytrs2 in one work-group (the host `blocked` route's steps):
//                           ?syconv as a gather into the caller's global workspace, P^T B
//                           gathered there too, the in-group trsm (NB-wide diagonal blocks) twice
//                           around D^-1, then the scatter; ex = WorkGroup<256, 32>. Not in tiers_t
//                           (as sytrf's Blocked<NB>). Its global part grows with nrhs:
//                           buffer_size<routine::sytrs<UL>, Tier, T>(n, nrhs).

namespace routine {
/// @ingroup api_device_lapack_ldl
/// @brief The routine tag of sytrs() for the triangle `UL`: names the routine to `with_tier`,
/// `launch`, `tier_info`, `tiers_t` and `buffer_size`.
template <Uplo UL> struct sytrs {
    static_assert(UL == Uplo::Lower || UL == Uplo::Upper, "sytrs: Uplo::Lower or Uplo::Upper");
    static constexpr const char* name = "sytrs";  ///< The routine's name, "sytrs".
};
}  // namespace routine

namespace detail {

template <class Scalar> struct SytrsCtaPlan {
    WsSlot<Scalar> tile;   // the n x n factor at ld n | 1, then e (n)
    WsSlot<int> idx;  // ps, perm, kind (n each), the validity flag
    WorkspaceSize size;
};
template <class Scalar> constexpr SytrsCtaPlan<Scalar> sytrs_cta_plan(int n) {
    Plan p;
    const auto un = static_cast<std::size_t>(n);
    const auto tile = p.take<Scalar>(WsSpace::Local, un * static_cast<std::size_t>(n | 1) + un);
    const auto idx = p.take<int>(WsSpace::Local, 3 * un + 1);
    return {tile, idx, p.size()};
}
template <class Real> struct SytrsWgPlan {
    WsSlot<Real> red;  // SumScratch: re then im, one slot per sub-group
    WorkspaceSize size;
};
template <class Real> constexpr SytrsWgPlan<Real> sytrs_wg_plan() {
    Plan p;
    const auto red = p.take<Real>(WsSpace::Local, 2 * (detail::sytrf_wg_max / 32));
    return {red, p.size()};
}
template <class Scalar> struct SytrsBlockedPlan {
    WsSlot<int> flag;  // local: the pivot list's validity
    WsSlot<Scalar> smem;    // local: the in-group trsm's tiles
    WsSlot<Scalar> l;       // global: the converted factor, n x n at ld n (the strict triangle)
    WsSlot<Scalar> y;       // global: P^T B, n x nrhs at ld n
    WsSlot<int> idx;   // global: ps, perm, kind, sig, inv (n each)
    WorkspaceSize size;
};
template <class Scalar> constexpr SytrsBlockedPlan<Scalar> sytrs_blocked_plan(int n, int nrhs) {
    Plan p;
    const auto un = static_cast<std::size_t>(n);
    const auto flag = p.take<int>(WsSpace::Local, 1);
    const auto sm = p.take<Scalar>(WsSpace::Local, l3::smem_elems<detail::sytrf_l3_cfg<Scalar>>());
    const auto l = p.take<Scalar>(WsSpace::Global, un * un);
    const auto y = p.take<Scalar>(WsSpace::Global, un * static_cast<std::size_t>(nrhs));
    const auto idx = p.take<int>(WsSpace::Global, 5 * un);
    return {flag, sm, l, y, idx, p.size()};
}

// LAPACK's argument numbers (?sytrs and ?sysv): -2 (N: A not square or outside the tier),
// -5 (LDA), -7 (B: not n rows), -8 (LDB < max(1, n)); -1 the executor as sytrf's.
template <class Tier, class Ex, class MA, class MB> BATCHLAS_DEVICE_INLINE int sytrs_check(const Ex& ex, const MA& a, const MB& b) {
    if (const int e = detail::sytrf_exec_check<Tier>(ex); e != 0) return e;
    const int n = a.rows(), lo = n > 1 ? n : 1;
    if (a.cols() != n || !detail::bound_admits<Tier>(n)) return -2;
    if (detail::sytrf_ld(a) < lo) return -5;
    if (b.rows() != n) return -7;
    if (detail::sytrf_ld(b) < lo) return -8;
    return 0;
}

// sytrs `cta`'s solve on a factor staged in local memory (sytrs `cta`; the second half of sysv
// `cta`): S's logical ipiv from `raw` (LAPACK's, physical order), ?syconv, then every column of
// B (ld ldb) on sub-group sgid of nsg. `flag`: one int of local memory. No trailing barrier.
template <int R, class All, class D>
BATCHLAS_DEVICE_INLINE void sytrs_cta_run(const All& all, const sycl::sub_group& sg, int lane, int sgid, int nsg, const Staged<D>& S,
                                         const int* raw, bool rev, int* flag, D* bb, std::ptrdiff_t ldb, int nrhs) {
    const int n = S.n;
    for_each_owned(all, n, [&](int k) BATCHLAS_DEVICE_BODY { S.ps[k] = logical_piv(raw[rev ? n - 1 - k : k], n, rev); });
    sync<Scope::Group>(all, Mem::Global);
    const bool ok = syconv(all, S, vec(S.ps, n), vec(S.perm, n), vec(S.kind, n), vec(S.e, n), flag);
    sytrs_cols<R>(sg, lane, sgid, nsg, S, ok, rev, bb, ldb, nrhs);
}

template <Uplo UL, int N, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE void sytrs_run(tier::Cta<N>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    constexpr bool rev = UL == Uplo::Upper;
    const int n = A.rows(), ldt = n | 1;
    const auto pl = sytrs_cta_plan<D>(n);
    D* const tp = carve_local(ws, pl.tile).get_raw();
    int* const im = carve_local(ws, pl.idx).get_raw();
    const Staged<D> S{tp, ldt, n, tp + static_cast<std::size_t>(n) * ldt, im, im + n, im + 2 * n};
    lacpy<Part::Lower, Transpose::NoTrans, alg::DivMod>(ex, detail::sytrf_view(A, rev), S);
    sytrs_cta_run<(N + 31) / 32>(ex, ex.sub(), ex.lane(), detail::sytrf_team(ex), detail::sytrf_nteams(ex), S, &ipiv(0),
                                 rev, im + 3 * n, &B(0, 0), detail::sytrf_ld(B), B.cols());
    sync<Scope::Group>(ex, Mem::Global);
}

template <Uplo UL, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE void sytrs_run(tier::Blocked<1>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using Rl = real_t<D>;
    constexpr bool rev = UL == Uplo::Upper;
    constexpr auto pl = sytrs_wg_plan<Rl>();
    Rl* const red = carve_local(ws, pl.red).get_raw();
    const int n = A.rows(), nrhs = B.cols();
    const auto G = detail::sytrf_view(A, rev);
    D* const b0 = &B(0, 0);
    const std::ptrdiff_t ldb = detail::sytrf_ld(B);
    // sytrs_col ends on a barrier: the scratch is free for the next column, B visible at the end.
    for (int c = 0; c < nrhs; ++c)
        sytrs_col(ex, G, [&](int k) { return ipiv(k); }, RevVec<D>{b0 + c * ldb, n, rev}, n, rev,
                  SumScratch<Rl>{red, red + detail::sytrf_wg_max / 32});
}

// The host `blocked` route's ?sytrs2 in one work-group: the logical ipiv, its validity and the
// composed permutation; ?syconv as a gather (syconv_gather) into L; y := P^T B; L y = y; D y = y
// (the diagonal blocks read from A); L^T y = y; B := P y. Physical row t of y holds logical row
// t (Lower) or n - 1 - t (Upper), so L is stored as the physical triangle UL of the converted
// factor and each trsm takes UL. An invalid list gives an all-NaN X, as every sytrs route.
template <Uplo UL, int NB, class Ex, class MA, class PV, class MB>
    requires(NB > 1)
BATCHLAS_DEVICE_INLINE void sytrs_run(tier::Blocked<NB>, const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    using Cfg = detail::sytrf_l3_cfg<D>;
    static_assert(!is_runtime_v<Ex> && Ex::size == Cfg::threads, "sytrs tier::Blocked<NB>: a WorkGroup<256, 32> executor");
    constexpr bool rev = UL == Uplo::Upper;
    const int n = A.rows(), nrhs = B.cols();
    const auto pl = sytrs_blocked_plan<D>(n, nrhs);
    int* const flag = carve_local(ws, pl.flag).get_raw();
    const Workspace sm = sub_workspace(ws, pl.smem);
    D* const lw = carve_global(ws, pl.l);
    D* const y = carve_global(ws, pl.y);
    int* const ps = carve_global(ws, pl.idx);
    int* const perm = ps + n;
    int* const kind = ps + 2 * n;
    const auto G = detail::sytrf_view(A, rev);
    D* const bb = &B(0, 0);
    const int ldb = static_cast<int>(detail::sytrf_ld(B));
    const int total = n * nrhs;
    for_each_owned(ex, n, [&](int k) BATCHLAS_DEVICE_BODY { ps[k] = logical_piv(ipiv(rev ? n - 1 - k : k), n, rev); });
    sync<Scope::Group>(ex, Mem::Global);
    if (ex.leader()) {
        const bool valid = piv_list_valid(ps, n);
        *flag = valid ? 1 : 0;
        if (valid) build_perm(ps, n, perm, kind);
    }
    sync<Scope::Group>(ex, Mem::Global);
    if (*flag == 0) {
        for_each_owned(ex, total, [&](int e) BATCHLAS_DEVICE_BODY { bb[e % n + (e / n) * ldb] = qnan<D>(); });
        sync<Scope::Group>(ex, Mem::Global);
        return;
    }
    syconv_gather<D>(detail::rev_mat(lw, n, n, rev), G, n, ps, kind, ps + 3 * n, ps + 4 * n, ex);
    // The row of B that physical row t of y comes from and goes back to.
    auto src = [&](int t) {
        const int lt = rev ? n - 1 - t : t;
        return rev ? n - 1 - perm[lt] : perm[lt];
    };
    for_each_owned(ex, total, [&](int e) BATCHLAS_DEVICE_BODY { y[e] = bb[src(e % n) + (e / n) * ldb]; });
    sync<Scope::Group>(ex, Mem::Global);
    constexpr Uplo tri = rev ? Uplo::Upper : Uplo::Lower;
    const l3::Global<D> L{lw, n, n, n}, Y{y, n, n, nrhs};
    trsm<Side::Left, tri, Transpose::NoTrans, Diag::Unit, Cfg, NB>(ex, One{}, L, Y, sm);
    for_each_owned(ex, total, [&](int e) BATCHLAS_DEVICE_BODY {
        const int t = e % n;  // logical: x reads through the reversal
        const int kd = kind[t];
        if (kd == 2) return;  // the first row of its block writes both
        const detail::RevVec<D> x{y + (e / n) * n, n, rev};
        if (kd == 0) {
            x(t) = ldl_dsolve_row(0, x(t), x(t), G(t, t), G(t, t), G(t, t));
            return;
        }
        const D x0 = x(t), x1 = x(t + 1), d0 = G(t, t), d1 = G(t + 1, t + 1), off = G(t + 1, t);
        x(t) = ldl_dsolve_row(1, x0, x1, d0, d1, off);
        x(t + 1) = ldl_dsolve_row(2, x1, x0, d1, d0, off);
    });
    sync<Scope::Group>(ex, Mem::Global);
    trsm<Side::Left, tri, Transpose::Trans, Diag::Unit, Cfg, NB>(ex, One{}, L, Y, sm);
    for_each_owned(ex, total, [&](int e) BATCHLAS_DEVICE_BODY { bb[src(e % n) + (e / n) * ldb] = y[e]; });
    sync<Scope::Group>(ex, Mem::Global);
}

}  // namespace detail

/// @ingroup api_device_lapack_ldl
/// @brief sytrs()'s tiers for every `Scalar`, smallest first: `Cta<32>`, `Cta<64>`, `Cta<128>`,
/// `Blocked<1>`. `Blocked<NB>` with NB > 1 is offered but not listed.
template <Uplo UL, class Scalar> struct tiers<routine::sytrs<UL>, Scalar> {
    using type = tier::list<tier::Cta<32>, tier::Cta<64>, tier::Cta<128>, tier::Blocked<1>>;
};
/// @ingroup api_device_lapack_ldl
/// @brief sytrs()'s local-memory tier `tier::Cta<N>`, N <= 128: the converted factor in local
/// memory, one sub-group per right-hand side, x in registers.
///
/// The executor is `WorkGroup<32, 32>`. `admits(n)` is 0 <= n <= N. `buffer_size(n, nrhs)` is
/// local memory only and does not depend on nrhs.
template <Uplo UL, int N, class Scalar> struct tier_info<routine::sytrs<UL>, tier::Cta<N>, Scalar> {
    static_assert(N <= 128, "sytrs tier::Cta<N>: N <= 128 (at most four registers per lane)");
    static constexpr int wg_size = 32, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0 && n <= N; }
    static constexpr WorkspaceSize buffer_size(int n, int = 1) { return detail::sytrs_cta_plan<Scalar>(n).size; }
};
/// @ingroup api_device_lapack_ldl
/// @brief sytrs()'s unblocked tier `tier::Blocked<1>`: any n, ?SYTRS on the factor in global
/// memory, one column at a time.
///
/// The executor is `WorkGroup<256, 32>`; any executor of whole 32-wide sub-groups, at most 256
/// items, also works. Its sums reduce in an order fixed by the executor's width. `admits(n)` is
/// n >= 0. `buffer_size(n, nrhs)` is local memory only and depends on neither.
template <Uplo UL, class Scalar> struct tier_info<routine::sytrs<UL>, tier::Blocked<1>, Scalar> {
    static constexpr int wg_size = detail::sytrf_wg_max, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int, int = 1) { return detail::sytrs_wg_plan<real_t<Scalar>>().size; }
};
/// @ingroup api_device_lapack_ldl
/// @brief sytrs()'s blocked tier `tier::Blocked<NB>`, NB > 1: any n, ?SYTRS2 in one work-group on
/// the in-group trsm.
///
/// ?SYCONV runs as a gather into the global workspace, \f$P^T B\f$ is gathered there too, the
/// trsm runs twice around \f$D^{-1}\f$, then the result is scattered back. The executor is
/// `WorkGroup<256, 32>`. `admits(n)` is n >= 0. `buffer_size(n, nrhs)`: the global part grows
/// with nrhs; the local part does not, so `with_tier`'s `buffer_size(n)` filter is exact.
template <Uplo UL, int NB, class Scalar> struct tier_info<routine::sytrs<UL>, tier::Blocked<NB>, Scalar> {
    static constexpr int wg_size = detail::sytrf_blocked_wg, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int n, int nrhs = 1) { return detail::sytrs_blocked_plan<Scalar>(n, nrhs).size; }
};

/// @ingroup api_device_lapack_ldl
/// @brief ?SYTRS in a kernel: \f$B := A^{-1} B\f$ for one matrix, from sytrf()'s factor.
///
/// `A` and `ipiv` are as sytrf() left them for the same `UL`; neither is written. The n x nrhs
/// `B` is overwritten with the solution X. A pivot list sytrf() cannot have written leaves that
/// `B` all NaN (the host routes' contract). The executor needs sub-group size 32 and must be
/// whole 32-wide sub-groups.
/// - `tier::Cta<N>`, N <= 128: `ex` a `WorkGroup<32, 32>`; n <= N.
/// - `tier::Blocked<1>`: any n; `ex` at most 256 items.
/// - `tier::Blocked<NB>`, NB > 1: any n; `ex` a `WorkGroup<256, 32>`. Not in `tiers_t`.
/// - `tier::upto<N>`: the first tier of `tiers_t<routine::sytrs<UL>, T>` that admits N.
///
/// Workspace: `ws` holds `buffer_size<routine::sytrs<UL>, Tier, T>(n, nrhs)`; only
/// `tier::Blocked<NB>`'s global part uses nrhs.
///
/// @pre Collective as sytrf(): every item of `ex` calls with the same arguments; `A`, `ipiv` and
/// `B` visible to every item on entry. X is visible to every item on return.
/// @tparam UL    `Uplo::Lower` or `Uplo::Upper`: the triangle that holds the factor
/// @tparam Tier  `tier::Cta<N>`, `tier::Blocked<NB>` or `tier::upto<N>`
/// @param[in]     ex    the executor that owns this matrix
/// @param[in]     A     sytrf()'s factor of the n x n matrix
/// @param[in]     ipiv  sytrf()'s n pivots (a `Vec<int>`)
/// @param[in,out] B     the n x nrhs right-hand sides; overwritten with X
/// @param[in]     ws    this matrix's workspace slice
/// @return 0 or a negative LAPACK argument number, the same on every item of `ex`. The numbers
/// are ?SYTRS's (N is 2, LDA 5, B 7, LDB 8), not positions in this call:
/// @retval 0   success (an invalid pivot list gives an all-NaN X, not an error)
/// @retval -1  `ex` has a run-time width that is not whole 32-wide sub-groups, or is wider than
///             `tier::Blocked<1>`'s 256 items
/// @retval -2  `A` is not square, or n is outside the tier
/// @retval -5  `A`'s leading dimension is less than max(1, n)
/// @retval -7  `B` does not have n rows
/// @retval -8  `B`'s leading dimension is less than max(1, n)
template <Uplo UL, class Tier, class Ex, class MA, class PV, class MB>
BATCHLAS_DEVICE_INLINE int sytrs(const Ex& ex, const MA& A, const PV& ipiv, const MB& B, const Workspace& ws) {
    using D = std::remove_cv_t<typename MA::value_type>;
    static_assert(std::is_same_v<D, std::remove_cv_t<typename MB::value_type>>, "sytrs: A and B of one scalar type");
    using Tr = resolve_tier_t<routine::sytrs<UL>, D, Tier>;
    static_assert(detail::sytrf_exec_ok<Ex>(), "sytrs: an executor of whole 32-wide sub-groups");
    const int chk = detail::sytrs_check<Tr>(ex, A, B);
    if (chk != 0 || A.rows() == 0 || B.cols() == 0) return chk;
    detail::sytrs_run<UL>(Tr{}, ex, A, ipiv, B, ws);
    return 0;
}

}  // namespace batchlas::device
