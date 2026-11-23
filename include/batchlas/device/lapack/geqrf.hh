// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Householder QR inside a kernel: `geqrf`, LAPACK's xGEQRF on one matrix per work-group.
///
/// `geqrf<Tier>(ex, a, tau, ws)` factors one m x n matrix (any m, n) with `tier::Blocked<NB>`:
/// `geqr2` on each NB-wide panel, then the panel's block reflector (`larft`, `larfb`) on the
/// columns to its right, on the in-group Level 3. The panel leaves `geqr2` (a team executor, any
/// memory) and `geqr2_reg` (one row per work-item, the panel in registers) are in
/// @ref api_device_lowlevel. Reference: @ref api_device_lapack_qr.
/// @ingroup api_device_lapack_qr
// detail: pieces core has no strict spelling for yet. Statement boundaries are load-bearing
// (fp-contract=on fuses a*b+c only within one expression).

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/larfb.hh>

namespace batchlas::device {

namespace detail {

// A complex copied as two scalars, not as a struct: larfg_scalars takes alpha by const&, and
// a struct-copied Cx<float> alpha is re-read from A after the reductions (+2 loads a column).
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar copy_by_parts(const Scalar& a) {
    using A = Arith<Scalar>;
    if constexpr (A::is_complex) return A::make(A::re(a), A::im(a));
    else return a;
}

// ---- slots: s[off + ...] from one base, never a moved pointer (the index form decides how
// ptxas unrolls the run-time folds: +112 LD, +4 REG otherwise) --------------------------------
template <class Ex, class Scalar> BATCHLAS_DEVICE_INLINE void team_slot_put(const Ex& ex, Scalar* slots, int off,
                                                                            Scalar v) {
    if (ex.lane() == 0) slots[off + ex.team()] = v;
}
template <class Real> BATCHLAS_DEVICE_INLINE Real slot_fold_max(const Real* s, int off, int k) {
    Real v = s[off];
    for (int i = 1; i < k; ++i) v = sycl::fmax(v, s[off + i]);
    return v;
}
template <class Scalar> BATCHLAS_DEVICE_INLINE Scalar slot_fold_sum(const Scalar* s, int off, int k) {
    Scalar v = Arith<Scalar>::zero();
    for (int i = 0; i < k; ++i) v = Arith<Scalar>::add(v, s[off + i]);
    return v;
}

// ---- RowFrag applies: item / lane l holds row l, F[k] = column k --------------------------
// gemv^H partials: for lo <= k < hi, team sum of cv * F[k] (xor butterfly over the run-time
// sub-group range), stored by lane 0 at slots[off + team * ld + k]. N, lo and hi fold after the
// caller's unroll.
template <int N, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE void gemv_rowfrag_partials(const Ex& ex, Scalar cv, const Scalar* F, int lo, int hi, Scalar* slots, int off, int ld) {
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;  // uniform, or lanes drop out of the butterfly
        const Scalar pw = butterfly(ex.sub(), FmaArith<Scalar>::mul(cv, F[k]), [](const Scalar& p, const Scalar& q) { return Arith<Scalar>::add(p, q); }, 1,
                                    ex.team_size());
        if (ex.lane() == 0) slots[off + ex.team() * ld + k] = pw;
    }
}
// F[k] -= (tau * sum_t slots[off + t * ld + k]) * v for lo <= k < hi.
template <int N, class Scalar>
BATCHLAS_DEVICE_INLINE void ger_rowfrag_slots(Scalar tau, Scalar v, Scalar* F, int lo, int hi, const Scalar* slots, int off, int nslots, int ld) {
    using A = Arith<Scalar>;
    using Fm = FmaArith<Scalar>;
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;
        Scalar w = A::zero();
        for (int i = 0; i < nslots; ++i) w = A::add(w, slots[off + i * ld + k]);
        F[k] = A::sub(F[k], Fm::mul(Fm::mul(tau, w), v));
    }
}

// Three steps through a local tile, the caller's barrier between each: (1) tile[off + kk] =
// cv * F[k0 + kk] for k in [lo, hi), else 0; (2) lane < C: y[lane] = tau * sum_{r >= r0}
// tile(r, lane) in row order; (3) F[k0 + kk] -= v * y[kk].
template <int C, class Scalar> BATCHLAS_DEVICE_INLINE void frag_scale_to_row(Scalar cv, const Scalar* F, int k0, int lo, int hi, Scalar* tile, int off) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        const bool act = (k >= lo) && (k < hi);
        tile[off + kk] = act ? FmaArith<Scalar>::mul(cv, F[k]) : Scalar{};
    }
}
template <int C, int NR, class Scalar> BATCHLAS_DEVICE_INLINE void colsum_colperlane(int lane, Scalar tau, const Scalar* tile, int ldt, int r0, Scalar* y) {
    if (lane < C) {
        Scalar acc = Scalar{};
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < NR; ++r) {
            if (r >= r0) acc = Arith<Scalar>::add(acc, tile[r * ldt + lane]);
        }
        y[lane] = FmaArith<Scalar>::mul(tau, acc);
    }
}
template <int C, class Scalar> BATCHLAS_DEVICE_INLINE void frag_rank1(Scalar v, const Scalar* y, Scalar* F, int k0,
                                                                      int lo, int hi) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        if ((k >= lo) && (k < hi)) F[k] = Arith<Scalar>::sub(F[k], FmaArith<Scalar>::mul(v, y[kk]));
    }
}

}  // namespace detail

/// @brief xGEQR2 inside a kernel: unblocked Householder QR of the m x n `a` in place, the panel
/// leaf of `geqrf`.
///
/// The first `tau.size()` columns are reduced (`tau.size() <= min(m, n)`): R overwrites the upper
/// triangle of those columns, the reflectors' vectors the part below it (unit diagonal implied),
/// and `tau(j)` receives reflector j's scalar (xLARFG's convention; it is applied as `conj(tau)`,
/// as zgeqr2 does). Columns past `tau.size()` receive the reflectors' update. Collective over
/// `ex`: every item calls it with the same arguments, `a` visible to every item on entry. On
/// return `tau` is visible, and so is `a` except the last reduced diagonal entry, which the leader
/// stores after the final barrier: barrier before another item reads it.
/// @tparam Ex  `RtGroup<Scope::Sub>` (sub-group barriers and reductions) or `RtGroup<Scope::Group>`
/// @tparam MA  a matrix operand in any memory space
/// @tparam VT  a vector operand
/// @param[in]     ex   the executor
/// @param[in,out] a    the matrix; R and the reflectors on return
/// @param[out]    tau  the reflector scalars, one per reduced column
/// @ingroup api_device_lowlevel
// Scope::Sub: no static shared, docs/perf/qr.md#the-48-kib-launch-hole. Core larfg<
// alg::LarfgLapack> (alpha's phase kept; docs/perf/qr.md#a-residual-test-cannot-guard-a-convention), then
// core larf<Left, arith::Fma> with conj(tau) (zgeqr2's).
// ex, a and tau by value, and plain inline: the body is simplified before it is inlined, and
// through references it keeps another CFG (FP64 resident sub-group panel +2 REG, an occupancy step).
template <class Ex, class MA, class VT>
inline void geqr2(const Ex ex, const MA a, const VT tau) {
    using D = typename MA::value_type;
    constexpr Scope S = Ex::sub_whole ? Scope::Sub : Scope::Group;
    const int m = a.rows(), n = a.cols();
    const int kmax = tau.size();
    // zgeqr2's A(j, j) = 1 around the apply: v is read from memory, not a per-row select of a
    // unit_head. The leader keeps beta and stores it once every item is past the apply.
    D beta{};
    bool pending = false;

    for (int j = 0; j < kmax; ++j) {
        ex.barrier(Mem::Global);  // B1 -- the previous reflector's column writes are visible.
        if (pending) a(j - 1, j - 1) = beta;  // leader only; column j - 1 is not read again
        // Each item scales exactly the rows of v it read: no barrier inside larfg.
        D alpha = detail::copy_by_parts(a(j, j));
        const LarfgScalars<D> h = larfg<alg::LarfgLapack, S>(ex, alpha, a.col(j).tail(j + 1));

        ex.barrier(Mem::Global);  // B2 -- every item has read A(j,j) as `alpha`; item 0 overwrites it.
        if (ex.leader()) {
            if (!h.identity) a(j, j) = Arith<D>::one();
            tau(j) = h.tau;
            beta = alpha;
            pending = !h.identity;
        }
        ex.barrier(Mem::Global);  // B3 -- the unit head and v are published to the teams about to read them.

        // Uniform over the barriers' scope (two collectives there): cannot desync B1/B2/B3.
        if (h.identity) continue;
        larf<Side::Left, arith::Fma>(ex, Arith<D>::conj(h.tau), assume_published(a.col(j).tail(j)), a.sub(j, j + 1, m - j, n - j - 1));
    }
    if (kmax > 0) ex.barrier(Mem::Global);  // every item is past the last apply
    if (pending) a(kmax - 1, kmax - 1) = beta;
}

namespace detail {
// geqr2_reg's body, a call: ex is built here (an Ex argument: 24 B frame, FP64 N = 16 spills).
template <int N, int SG, class D, class Real>
inline void geqr2_reg_body(sycl::nd_item<1> it, D* a, int ld, int m, int n, int kmax, D* tau_ptr, Real* sred, D* sw) {
    const RtGroup<Scope::Group, SG> ex{it};
    using R = real_t<D>;
    using A = Arith<D>;
    using F = FmaArith<D>;  // arith::Fma: sycl_device::dev_mul / dev_div bit for bit
    static_assert(sizeof(R) == sizeof(Real), "the real slot array must match the scalar's real");

    const int r = ex.lid();
    const int nsg = ex.nteams();
    const bool live = (r < m);

    // unroll(full) on EVERY loop whose index reaches rA, or rA[j] is a DYNAMIC index and the
    // array leaves the register file. evidence: docs/perf/qr.md#the-stack-frame-is-the-gate-for-this-kernel-not-the-spill-counter
    D rA[N];
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        rA[k] = (live && k < n) ? a[static_cast<std::ptrdiff_t>(r) +
                                    static_cast<std::ptrdiff_t>(k) * ld]
                                : A::zero();
    }

#pragma clang loop unroll(full)
    for (int j = 0; j < N; ++j) {
        if (j >= kmax) continue;   // launch-uniform: cannot desync the barriers below

        // Same slots folded in the same order on every item: h is work-group-uniform.
        const R part = (live && r >= j) ? abs_inf(rA[j]) : R(0);
        detail::team_slot_put(ex, sred, 0, butterfly(ex.sub(), part, reduce_op::FMax{}, 1, ex.team_size()));
        if (r == j) sw[0] = rA[j];
        ex.barrier(Mem::Global);                                   // B1
        const R smax = detail::slot_fold_max(sred, 0, nsg);
        const D alpha = sw[0];

        R ssq_part = R(0);
        if (live && r > j && smax > R(0)) ssq_part = detail::abs2_scaled(rA[j], smax);
        detail::team_slot_put(ex, sred, nsg, butterfly(ex.sub(), ssq_part, sycl::plus<R>(), 1, ex.team_size()));
        ex.barrier(Mem::Global);                                   // B2
        const R ssq = detail::slot_fold_sum(sred, nsg, nsg);

        const LarfgScalars<D> h = larfg_scalars<alg::LarfgLapack>(alpha, ScaledSsq<R>{smax, ssq});

        // NO `if (h.identity) continue;`: inert arithmetically, and omitting it keeps the
        // barrier sequence uniform. tau is a defined output either way.
        if (r == j) {
            rA[j] = h.beta;
            tau_ptr[j] = h.tau;
        } else if (live && r > j) {
            rA[j] = h.use_mul ? F::mul(rA[j], h.scale) : F::div(rA[j], h.scale);
        }

        // One select: zeroes eliminated and dead rows, supplies the implicit v(j) = 1. Dropping
        // the `r < j` term lets an eliminated row re-enter the reflector.
        const D v = (!live || r < j) ? A::zero() : ((r == j) ? A::one() : rA[j]);
        const D cv = A::conj(v);
        // CONJ(tau), zgeqr2's convention. evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
        const D ctau = A::conj(h.tau);

        // larf: team partials of v^H A(:, j+1:n) into the slots, B3, then A -= (ctau w) v.
        detail::gemv_rowfrag_partials<N>(ex, cv, rA, j + 1, n, sw, 1, N);
        ex.barrier(Mem::Global);                                   // B3
        detail::ger_rowfrag_slots<N>(ctau, v, rA, j + 1, n, sw, 1, nsg, N);
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (live && k < n) {
            a[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k) * ld] = rA[k];
        }
    }
}
}  // namespace detail

/// @brief xGEQR2 on an m x n panel held in registers, one row per work-item of the work-group.
///
/// `n <= N`, m at most the work-group size, and `tau.size() <= min(m, n)` reflectors, the same on
/// every item. Every item of the work-group calls it: rows at or past m still run every barrier.
/// The results are stored as `geqr2` stores them; `a` and `tau` are written by their owners and a
/// barrier is needed before another item reads them.
/// @tparam N  the compile-time column bound
/// @param[in]     ex    an `RtGroup<Scope::Group>`
/// @param[in,out] a     the panel, a `Mat` over global memory
/// @param[out]    tau   the reflector scalars
/// @param         sred  local scratch: 2 reals per sub-group
/// @param         sw    local scratch of a's scalar type: N per sub-group, plus 1
/// @ingroup api_device_lowlevel
// Butterflies, never reduce_over_group: the 48 KB hole.
template <int N, class Ex, class MA, class VT, class Real, class D>
BATCHLAS_DEVICE_INLINE void geqr2_reg(const Ex& ex, const MA& a, const VT& tau, Real* sred, D* sw) {
    static_assert(Ex::runtime && !Ex::sub_whole, "geqr2_reg: ex is an RtGroup<Scope::Group>");
    static_assert(std::is_same_v<typename MA::value_type, D>, "geqr2_reg: sw holds a's scalar type");
    detail::geqr2_reg_body<N, Ex::sg_size>(ex.it, &a(0, 0), static_cast<int>(a.ld), a.rows(), a.cols(), tau.size(), &tau(0), sred, sw);
}

// ---- xGEQRF: the public in-kernel op (tier::Blocked) -----------------------------------------

namespace routine {
/// @brief The routine tag of `geqrf`, for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_qr
struct geqrf {
    static constexpr const char* name = "geqrf";
};
}  // namespace routine

namespace detail {
template <class Scalar> struct GeqrfSlots {
    WsSlot<Scalar> smem, v, t, work;
};
// The host blocked route's layout per matrix: V (m x NB), T (NB x NB), W1 / W2 (NB x (n - nb)).
template <int NB, class Cfg, class Scalar> constexpr GeqrfSlots<Scalar> geqrf_take(Plan& p, int m, int n) {
    const int k = m < n ? m : n;
    const int nb = k < NB ? (k > 0 ? k : 1) : NB;
    const int n2max = n - nb > 0 ? n - nb : 0;
    GeqrfSlots<Scalar> s{};
    s.smem = p.take<Scalar>(WsSpace::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
    s.v = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(m > 0 ? m : 0) * NB);
    s.t = p.take<Scalar>(WsSpace::Global, static_cast<std::size_t>(NB) * NB);
    s.work = p.take<Scalar>(WsSpace::Global, larfb_work_elems<Side::Left>(NB, n2max, NB));
    return s;
}
template <int NB, class Cfg, class Scalar> constexpr WorkspaceSize geqrf_size(int m, int n) {
    Plan p;
    (void)geqrf_take<NB, Cfg, Scalar>(p, m, n);
    return p.size();
}

// ?GEQRF's right-looking loop (the host geqrf_blocked_dispatch per matrix): per NB-wide panel,
// geqr2 on the panel in global memory (the host's global leaf, RtGroup<Scope::Group>), then
// larfb_from_panel<Left, ConjTrans> on A(j0 :, j0 + ib :). m < n: k = m panels, each update
// spanning every column to the right (no separate ormqr pass).
template <int NB, class Cfg, class Ex, class Scalar>
BATCHLAS_DEVICE_INLINE void geqrf_blocked_run(const Ex& ex, const l3::Global<Scalar>& a, Scalar* tau, const GeqrfSlots<Scalar>& s, const Workspace& w) {
    const Workspace sm = sub_workspace(w, s.smem, s.work);
    Scalar* const vb = carve_global(w, s.v);
    Scalar* const tb = carve_global(w, s.t);
    const RtGroup<Scope::Group, Ex::sg_size> pex{ex.it};
    const int m = a.m, n = a.n, k = m < n ? m : n;
    for (int j0 = 0; j0 < k; j0 += NB) {
        const int ib = k - j0 < NB ? k - j0 : NB;
        const int j2 = j0 + ib, mp = m - j0, n2 = n - j2;
        geqr2(pex, mat_wide(a.p + j0 + static_cast<std::ptrdiff_t>(j0) * a.ld, mp, ib, a.ld), vec(tau + j0, ib));
        ex.barrier(Mem::Global);  // the panel's last column update visible before V is packed
        if (n2 <= 0) break;       // uniform
        const l3::Global<Scalar> a22{a.p + j0 + static_cast<std::ptrdiff_t>(j2) * a.ld, a.ld, mp, n2};
        larfb_from_panel<Side::Left, Transpose::ConjTrans, Cfg>(ex, a, j0, ib, tau, a22, vb, tb, NB, sm);
    }
}

template <class Tier> inline constexpr bool geqrf_is_blocked_v = false;
template <int NB, class A> inline constexpr bool geqrf_is_blocked_v<tier::Blocked<NB, A>> = true;
}  // namespace detail

/// @brief `geqrf` in `tier::Blocked<NB, A>`: one m x n matrix per work-group, any m and n.
///
/// The executor is `WorkGroup<cfg::threads, 32>`, where `cfg` (`larfb_cfg_t<A, Scalar>`) is the
/// in-group Level 3 configuration: `A` itself when it is an `l3::cfg` preset, else a 256-item
/// default. The kernel must run at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(sg_size)`).
/// `buffer_size(m, n)` is one matrix's local and global bytes; `buffer_size(n)` is the square
/// case.
/// @ingroup api_device_lapack_qr
template <int NB, class A, class Scalar> struct tier_info<routine::geqrf, tier::Blocked<NB, A>, Scalar> {
    using cfg = larfb_cfg_t<A, Scalar>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return detail::geqrf_size<NB, cfg, Scalar>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
/// @brief The tiers `with_tier` tries for `geqrf`: `tier::Blocked<larfb_default_nb<Scalar>()>`.
/// @ingroup api_device_lapack_qr
template <class Scalar> struct tiers<routine::geqrf, Scalar> {
    using type = tier::list<tier::Blocked<larfb_default_nb<Scalar>()>>;
};

/// @brief xGEQRF inside a kernel: \f$A = QR\f$ in place for one m x n matrix (any m, n).
///
/// R overwrites the upper triangle of `a`, the reflectors' vectors (LAPACK storage, unit diagonal
/// implied) the part below it, and `tau` their k = min(m, n) scalars (xLARFG's convention; the
/// apply uses `conj(tau)`, as zgeqr2 does). `orgqr` forms Q from this storage and `ormqr`
/// applies it.
///
/// Collective over the tier's executor, `tier_info<routine::geqrf, Tier, T>::executor`: every
/// item calls it with the same arguments. `a` visible to every item on entry; `a` and `tau`
/// visible on return. One tier per kernel: the host picks it with `with_tier<routine::geqrf, T>`.
/// @pre The kernel runs at sub-group size 32 (`BATCHLAS_REQD_SG_SIZE(32)`).
/// @tparam Tier  `tier::Blocked<NB, Alg>`, the only tier; `Alg` is an `l3::cfg` preset or the
///               default
/// @tparam Ex    the executor type, `WorkGroup<cfg::threads, 32>`
/// @tparam MA    a global matrix operand: `l3::Global`, a `Mat` over global memory, or
///               `device::item`
/// @tparam Tau   a unit-stride vector operand (`device::vec`), not a pointer
/// @param[in]     ex   the executor
/// @param[in,out] a    this matrix; R and the reflectors on return
/// @param[out]    tau  at least k scalars
/// @param[in]     ws   `ws.local` at least `local_bytes` and `ws.global` this matrix's slice of
///                     `global_bytes`, both of `buffer_size<routine::geqrf, Tier, T>(m, n)`;
///                     contents undefined on return
/// @return `info`, the same on every item
/// @retval 0   success, including k = 0
/// @retval -2  `tau` is shorter than k (nothing written)
/// @retval -3  `ws.global` is null (nothing written)
/// @ingroup api_device_lapack_qr
template <class Tier, class Ex, class MA, class Tau>
BATCHLAS_DEVICE_INLINE int geqrf(const Ex& ex, const MA& a, const Tau& tau, const Workspace& ws) {
    using T = std::remove_cvref_t<decltype(a(0, 0))>;
    using Tr = resolve_tier_t<routine::geqrf, T, Tier>;
    static_assert(HasTier<routine::geqrf, Tr, T>, "geqrf: this tier is not offered for this scalar type");
    static_assert(detail::geqrf_is_blocked_v<Tr>, "geqrf: only tier::Blocked<NB> is offered");
    static_assert(!std::is_pointer_v<Tau>, "geqrf: tau is a vector operand (device::vec)");
    using Cfg = larfb_cfg_t<typename Tr::alg_type, T>;
    detail::larfb_check_ex<Cfg, Ex>();
    const auto ag = l3::global_of(a);
    const int k = ag.m < ag.n ? ag.m : ag.n;
    if (static_cast<int>(tau.size()) < k) return -2;
    if (ws.global == nullptr) return -3;
    if (k <= 0) return 0;
    Plan p;
    const auto s = detail::geqrf_take<Tr::nb, Cfg, T>(p, ag.m, ag.n);
    detail::geqrf_blocked_run<Tr::nb, Cfg>(ex, ag, &tau(0), s, ws);
    return 0;
}

}  // namespace batchlas::device
