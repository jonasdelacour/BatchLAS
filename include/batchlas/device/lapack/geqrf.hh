// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xGEQR2, geqrf's panel leaves: geqr2 (team executor, any memory) and geqr2_reg (a row per
// item, panel in registers); and xGEQRF in a kernel, geqrf<tier::Blocked<NB>> (geqr2 panels and
// larfb trailing updates on the in-group Level 3). geqrf_detail: pieces core has no strict
// spelling for yet. Statement boundaries are load-bearing (fp-contract=on fuses a*b+c only within
// one expression).

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <batchlas/device/lapack/larfb.hh>

namespace batchlas::device::inline v2 {

namespace geqrf_detail {

// A complex copied as two scalars, not as a struct: larfg_scalars takes alpha by const&, and
// a struct-copied Cx<float> alpha is re-read from A after the reductions (+2 loads a column).
template <class T> BATCHLAS_DLA_INLINE T copy_by_parts(const T& a) {
    using A = Arith<T>;
    if constexpr (A::is_complex) return A::make(A::re(a), A::im(a));
    else return a;
}
// |a / s|^2, dividing first; s > 0.
template <class T> BATCHLAS_DLA_INLINE real_t<T> abs2_scaled(T a, real_t<T> s) {
    using A = Arith<T>;
    using R = real_t<T>;
    if constexpr (A::is_complex) {
        const R x = A::re(a) / s;
        const R y = A::im(a) / s;
        return sycl::fma(x, x, y * y);
    } else {
        const R x = a / s;
        return x * x;
    }
}

// A(j:m, j+1:n) := (I - tau v v^H) A(j:m, j+1:n), v = [1; A(j+1:m, j)] in absolute rows and
// columns. Team t owns columns j+1+t mod nteams; lane 0 seeds w with the head row, the lanes
// stride rows j+1.. by `lanes` (the sub-group's run-time range, also the butterfly's width).
// Sub-group shuffles only: no work-group barrier inside.
template <class Ex, class T, class MA>
BATCHLAS_DLA_INLINE void larf_left_head(const Ex& ex, int lanes, T tau, const MA& a, int j, int m, int n) {
    using A = Arith<T>;
    using F = FmaArith<T>;
    const int lane = ex.lane();
    const int nt = detail::nteams_of(ex);
    for (int c = j + 1 + ex.team(); c < n; c += nt) {
        T acc = (lane == 0) ? static_cast<T>(a(j, c)) : A::zero();
        for (int r = j + 1 + lane; r < m; r += lanes) acc = A::add(acc, F::mul(A::conj(static_cast<T>(a(r, j))), static_cast<T>(a(r, c))));
        const T w = butterfly(ex.sub(), acc, [](const T& p, const T& q) { return A::add(p, q); }, 1, lanes);
        const T f = F::mul(tau, w);
        if (lane == 0) a(j, c) = A::sub(static_cast<T>(a(j, c)), f);
        for (int r = j + 1 + lane; r < m; r += lanes) a(r, c) = A::sub(static_cast<T>(a(r, c)), F::mul(f, static_cast<T>(a(r, j))));
    }
}

// ---- slots: s[off + ...] from one base, never a moved pointer (the index form decides how
// ptxas unrolls the run-time folds: +112 LD, +4 REG otherwise) --------------------------------
template <class Ex, class T> BATCHLAS_DLA_INLINE void team_slot_put(const Ex& ex, T* slots, int off, T v) {
    if (ex.lane() == 0) slots[off + ex.team()] = v;
}
template <class T> BATCHLAS_DLA_INLINE T slot_fold_max(const T* s, int off, int k) {
    T v = s[off];
    for (int i = 1; i < k; ++i) v = sycl::fmax(v, s[off + i]);
    return v;
}
template <class T> BATCHLAS_DLA_INLINE T slot_fold_sum(const T* s, int off, int k) {
    T v = Arith<T>::zero();
    for (int i = 0; i < k; ++i) v = Arith<T>::add(v, s[off + i]);
    return v;
}

// ---- RowFrag applies: item / lane l holds row l, F[k] = column k --------------------------
// gemv^H partials: for lo <= k < hi, team sum of cv * F[k] (xor butterfly over the run-time
// sub-group range), stored by lane 0 at slots[off + team * ld + k]. N, lo and hi fold after the
// caller's unroll.
template <int N, class Ex, class T>
BATCHLAS_DLA_INLINE void gemv_rowfrag_partials(const Ex& ex, T cv, const T* F, int lo, int hi, T* slots, int off, int ld) {
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;  // uniform, or lanes drop out of the butterfly
        const T pw = butterfly(ex.sub(), FmaArith<T>::mul(cv, F[k]), [](const T& p, const T& q) { return Arith<T>::add(p, q); }, 1,
                               ex.team_size());
        if (ex.lane() == 0) slots[off + ex.team() * ld + k] = pw;
    }
}
// F[k] -= (tau * sum_t slots[off + t * ld + k]) * v for lo <= k < hi.
template <int N, class T>
BATCHLAS_DLA_INLINE void ger_rowfrag_slots(T tau, T v, T* F, int lo, int hi, const T* slots, int off, int nslots, int ld) {
    using A = Arith<T>;
    using Fm = FmaArith<T>;
#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (k < lo || k >= hi) continue;
        T w = A::zero();
        for (int i = 0; i < nslots; ++i) w = A::add(w, slots[off + i * ld + k]);
        F[k] = A::sub(F[k], Fm::mul(Fm::mul(tau, w), v));
    }
}

// Three steps through a local tile, the caller's barrier between each: (1) tile[off + kk] =
// cv * F[k0 + kk] for k in [lo, hi), else 0; (2) lane < C: y[lane] = tau * sum_{r >= r0}
// tile(r, lane) in row order; (3) F[k0 + kk] -= v * y[kk].
template <int C, class T> BATCHLAS_DLA_INLINE void frag_scale_to_row(T cv, const T* F, int k0, int lo, int hi, T* tile, int off) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        const bool act = (k >= lo) && (k < hi);
        tile[off + kk] = act ? FmaArith<T>::mul(cv, F[k]) : T{};
    }
}
template <int C, int NR, class T> BATCHLAS_DLA_INLINE void colsum_colperlane(int lane, T tau, const T* tile, int ldt, int r0, T* y) {
    if (lane < C) {
        T acc = T{};
        BATCHLAS_UNROLL_FULL
        for (int r = 0; r < NR; ++r) {
            if (r >= r0) acc = Arith<T>::add(acc, tile[r * ldt + lane]);
        }
        y[lane] = FmaArith<T>::mul(tau, acc);
    }
}
template <int C, class T> BATCHLAS_DLA_INLINE void frag_rank1(T v, const T* y, T* F, int k0, int lo, int hi) {
#pragma clang loop unroll(full)
    for (int kk = 0; kk < C; ++kk) {
        const int k = k0 + kk;
        if ((k >= lo) && (k < hi)) F[k] = Arith<T>::sub(F[k], FmaArith<T>::mul(v, y[kk]));
    }
}

}  // namespace geqrf_detail

// xGEQR2 on a (m x n) in place, ex = RtGroup<Scope::Sub> (sub-group barriers and butterflies: no
// static shared, docs/perf/qr.md#the-48-kib-launch-hole) or <Scope::Group>. Scalars: larfg_scalars<
// conv::Lapack> (alpha's phase kept; docs/perf/qr.md#a-residual-test-cannot-guard-a-convention), the
// apply uses conj(tau) (zgeqr2's); tau[j] per reflector, kmax <= min(m, n). COLLECTIVE.
// ex and a by value: the body is simplified before it is inlined, and through references it
// keeps another CFG (FP64 resident sub-group panel +2 REG, an occupancy step).
template <class Ex, class MA, class D>
inline void geqr2(const Ex ex, const MA a, int kmax, D* tau) {
    using R = real_t<D>;
    using F = FmaArith<D>;  // arith::Fma: sycl_device::dev_mul / dev_div bit for bit
    constexpr bool sub = Ex::sub_whole;
    const int lanes = ex.team_size();
    const int m = a.rows(), n = a.cols();

    for (int j = 0; j < kmax; ++j) {
        ex.barrier(Mem::Global);  // B1 -- the previous reflector's column writes are visible.
        const D alpha = geqrf_detail::copy_by_parts(a(j, j));
        R smax = abs_inf(alpha);
        for_each_owned(ex, Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { smax = sycl::fmax(smax, abs_inf(a(r, j))); });
        // Scope::Sub: the run-time xor butterfly; Scope::Group: portable::reduce_over_group.
        if constexpr (sub) smax = butterfly(ex.sub(), smax, op::FMax{}, 1, lanes);
        else smax = ex.group_reduce(smax, sycl::maximum<R>());
        R ssq = R(0);  // the collective sits outside the `smax > 0` test: all items reach it
        if (smax > R(0)) {
            for_each_owned(ex, Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { ssq += geqrf_detail::abs2_scaled(a(r, j), smax); });
        }
        if constexpr (sub) ssq = butterfly(ex.sub(), ssq, sycl::plus<R>(), 1, lanes);
        else ssq = ex.group_reduce(ssq, sycl::plus<R>());
        const LarfgScalars<D> h = larfg_scalars<conv::Lapack>(alpha, ScaledSsq<R>{smax, ssq});

        ex.barrier(Mem::Global);  // B2 -- every item has read A(j,j) as `alpha`; item 0 overwrites it.
        if (!h.identity) {
            if (ex.leader()) a(j, j) = h.beta;
            // No barrier needed here: each item writes exactly the rows it read.
            if (h.use_mul) {
                for_each_owned(ex, Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { a(r, j) = F::mul(a(r, j), h.scale); });
            } else {
                for_each_owned(ex, Range{j + 1, m}, [&](int r) BATCHLAS_DLA_BODY { a(r, j) = F::div(a(r, j), h.scale); });
            }
        }
        if (ex.leader()) tau[j] = h.tau;
        ex.barrier(Mem::Global);  // B3 -- beta and v are published to the teams about to read them.

        // Uniform over the barriers' scope (two collectives there): cannot desync B1/B2/B3.
        if (h.identity) continue;
        geqrf_detail::larf_left_head(ex, lanes, Arith<D>::conj(h.tau), a, j, m, n);
    }
}

namespace geqrf_detail {
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
        geqrf_detail::team_slot_put(ex, sred, 0, butterfly(ex.sub(), part, op::FMax{}, 1, ex.team_size()));
        if (r == j) sw[0] = rA[j];
        ex.barrier(Mem::Global);                                   // B1
        const R smax = geqrf_detail::slot_fold_max(sred, 0, nsg);
        const D alpha = sw[0];

        R ssq_part = R(0);
        if (live && r > j && smax > R(0)) ssq_part = geqrf_detail::abs2_scaled(rA[j], smax);
        geqrf_detail::team_slot_put(ex, sred, nsg, butterfly(ex.sub(), ssq_part, sycl::plus<R>(), 1, ex.team_size()));
        ex.barrier(Mem::Global);                                   // B2
        const R ssq = geqrf_detail::slot_fold_sum(sred, nsg, nsg);

        const LarfgScalars<D> h = larfg_scalars<conv::Lapack>(alpha, ScaledSsq<R>{smax, ssq});

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
        geqrf_detail::gemv_rowfrag_partials<N>(ex, cv, rA, j + 1, n, sw, 1, N);
        ex.barrier(Mem::Global);                                   // B3
        geqrf_detail::ger_rowfrag_slots<N>(ctau, v, rA, j + 1, n, sw, 1, nsg, N);
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (live && k < n) {
            a[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k) * ld] = rA[k];
        }
    }
}
}  // namespace geqrf_detail

// xGEQR2 on an m x n panel (a, ld), n <= N, a row per item of the work-group ex (RtGroup<
// Scope::Group>); m <= its size, kmax <= min(m, n), launch-uniform. Rows >= m run every barrier
// (no early return). sred: 2 reals, sw: N scalars per sub-group, plus 1 (butterflies, never
// reduce_over_group: the 48 KB hole).
template <int N, class D, class Real, class Ex>
BATCHLAS_DLA_INLINE void geqr2_reg(const Ex& ex, D* a, int ld, int m, int n, int kmax, D* tau_ptr, Real* sred, D* sw) {
    static_assert(Ex::runtime && !Ex::sub_whole, "geqr2_reg: ex is an RtGroup<Scope::Group>");
    geqrf_detail::geqr2_reg_body<N, Ex::sg_size>(ex.it, a, ld, m, n, kmax, tau_ptr, sred, sw);
}

// ---- xGEQRF: the public in-kernel op (v22 design C.7 Blocked) --------------------------------

namespace routine {
struct geqrf {
    static constexpr const char* name = "geqrf";
};
}  // namespace routine

namespace geqrf_detail {
template <class T> struct GeqrfSlots {
    ws::Slot<T> smem, v, t, work;
};
// The host blocked route's layout per matrix: V (m x NB), T (NB x NB), W1 / W2 (NB x (n - nb)).
template <int NB, class Cfg, class T> constexpr GeqrfSlots<T> geqrf_take(Plan& p, int m, int n) {
    const int k = m < n ? m : n;
    const int nb = k < NB ? (k > 0 ? k : 1) : NB;
    const int n2max = n - nb > 0 ? n - nb : 0;
    GeqrfSlots<T> s{};
    s.smem = p.take<T>(ws::Space::Local, static_cast<std::size_t>(l3::smem_elems<Cfg>()));
    s.v = p.take<T>(ws::Space::Global, static_cast<std::size_t>(m > 0 ? m : 0) * NB);
    s.t = p.take<T>(ws::Space::Global, static_cast<std::size_t>(NB) * NB);
    s.work = p.take<T>(ws::Space::Global, larfb_work_elems<Side::Left>(NB, n2max, NB));
    return s;
}
template <int NB, class Cfg, class T> constexpr WorkspaceSize geqrf_size(int m, int n) {
    Plan p;
    (void)geqrf_take<NB, Cfg, T>(p, m, n);
    return p.size();
}

// ?GEQRF's right-looking loop (the host geqrf_blocked_dispatch per matrix): per NB-wide panel,
// geqr2 on the panel in global memory (the host's global leaf, RtGroup<Scope::Group>), then
// larfb_from_panel<Left, ConjTrans> on A(j0 :, j0 + ib :). m < n: k = m panels, each update
// spanning every column to the right (no separate ormqr pass).
template <int NB, class Cfg, class Ex, class T>
BATCHLAS_DLA_INLINE void geqrf_blocked_run(const Ex& ex, const l3::Global<T>& a, T* tau, const GeqrfSlots<T>& s, const Workspace& w) {
    const l3::Smem<Cfg> sm{carve_local(w, s.smem).get_raw()};
    T* const vb = carve_global(w, s.v);
    T* const tb = carve_global(w, s.t);
    T* const wb = carve_global(w, s.work);
    const RtGroup<Scope::Group, Ex::sg_size> pex{ex.it};
    const int m = a.m, n = a.n, k = m < n ? m : n;
    for (int j0 = 0; j0 < k; j0 += NB) {
        const int ib = k - j0 < NB ? k - j0 : NB;
        const int j2 = j0 + ib, mp = m - j0, n2 = n - j2;
        geqr2(pex, mat_wide(a.p + j0 + static_cast<std::ptrdiff_t>(j0) * a.ld, mp, ib, a.ld), ib, tau + j0);
        ex.barrier(Mem::Global);  // the panel's last column update visible before V is packed
        if (n2 <= 0) break;       // uniform
        const l3::Global<T> a22{a.p + j0 + static_cast<std::ptrdiff_t>(j2) * a.ld, a.ld, mp, n2};
        larfb_from_panel<Side::Left, Transpose::ConjTrans, Cfg>(ex, a, j0, ib, tau, a22, vb, tb, NB, wb, sm);
    }
}

template <class Tier> inline constexpr bool geqrf_is_blocked_v = false;
template <int NB, class A> inline constexpr bool geqrf_is_blocked_v<tier::Blocked<NB, A>> = true;
}  // namespace geqrf_detail

template <int NB, class A, class T> struct tier_info<routine::geqrf, tier::Blocked<NB, A>, T> {
    using cfg = larfb_cfg_t<A, T>;
    static constexpr int wg_size = cfg::threads, sg_size = 32;
    static constexpr bool admits(int n) { return n >= 0; }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return geqrf_detail::geqrf_size<NB, cfg, T>(m, n); }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};
template <class T> struct tiers<routine::geqrf, T> {
    using type = tier::list<tier::Blocked<larfb_default_nb<T>()>>;
};

/// The workspace of one m x n matrix for geqrf<Tier> (host and device; constexpr).
template <class Tier, class T> constexpr WorkspaceSize geqrf_buffer_size(int m, int n) {
    return tier_info<routine::geqrf, resolve_tier_t<routine::geqrf, T, Tier>, T>::buffer_size(m, n);
}

/// @brief ?GEQRF in a kernel: A = Q R in place for one m x n matrix (any m, n).
///
/// R overwrites A's upper triangle, the reflectors' vectors (LAPACK storage, unit diagonal
/// implied) the part below it, tau their k = min(m, n) scalars (xLARFG's convention; the apply
/// uses conj(tau), zgeqr2's). Collective over a WorkGroup<tier_info<...>::wg_size, 32>; A visible
/// on entry, A and tau visible on return. One tier per kernel (with_tier<routine::geqrf, T>).
/// @param ex   the executor
/// @param A    this matrix (l3::Global, a global Mat or device::item)
/// @param tau  k scalars: a T* or a unit-stride vector operand
/// @param ws   ws.local >= local_bytes, ws.global this matrix's slice of global_bytes of
///             geqrf_buffer_size<Tier, T>(m, n); contents undefined on return
/// @return 0, or -i for a bad i-th argument after the executor (2: tau shorter than k, 3: no
///         global workspace), uniform
template <class Tier, class Ex, class MA, class Tau>
BATCHLAS_DLA_INLINE int geqrf(const Ex& ex, const MA& A, const Tau& tau, const Workspace& ws) {
    using T = std::remove_cvref_t<decltype(A(0, 0))>;
    using Tr = resolve_tier_t<routine::geqrf, T, Tier>;
    static_assert(HasTier<routine::geqrf, Tr, T>, "geqrf: this tier is not offered for this scalar type");
    static_assert(geqrf_detail::geqrf_is_blocked_v<Tr>, "geqrf: only tier::Blocked<NB> is offered");
    using Cfg = larfb_cfg_t<typename Tr::alg_type, T>;
    larfb_check_ex<Cfg, Ex>();
    const auto a = l3::global_of(A);
    const int k = a.m < a.n ? a.m : a.n;
    if constexpr (requires { tau.size(); }) {
        if (static_cast<int>(tau.size()) < k) return -2;
    }
    if (ws.global == nullptr) return -3;
    if (k <= 0) return 0;
    Plan p;
    const auto s = geqrf_detail::geqrf_take<Tr::nb, Cfg, T>(p, a.m, a.n);
    geqrf_detail::geqrf_blocked_run<Tr::nb, Cfg>(ex, a, larfb_tau_base(tau), s, ws);
    return 0;
}

}  // namespace batchlas::device
