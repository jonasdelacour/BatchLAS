// SPDX-License-Identifier: MIT
#pragma once
// LAPACK xGEQR2, geqrf's panel leaves: geqr2 (team executor, any memory) and geqr2_reg (a row per
// item, panel in registers). geqrf_detail: pieces core has no strict spelling for yet. Statement
// boundaries are load-bearing (fp-contract=on fuses a*b+c only within one expression).

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>

#include <batchlas/device.hh>
#include <batchlas/device/arith.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

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

// ==== public geqrf (v22 design C.2-C.7) ======================================================
namespace routine {
struct geqrf {
    static constexpr const char* name = "geqrf";
};
}  // namespace routine

namespace geqrf_detail {

// Reg<N>, per matrix: an N x (C + 1) tile and a SEPARATE C-element y (why two barriers per chunk
// suffice). evidence: docs/perf/qr.md#the-two-partition-butterflies
template <class D, int N, int C> struct RegPlan {
    ws::Slot<D> tile{};
    ws::Slot<D> y{};
    WorkspaceSize size{};
    constexpr RegPlan() {
        Plan p;
        tile = p.take<D>(ws::Space::Local, static_cast<std::size_t>(N) * static_cast<std::size_t>(C + 1));
        y = p.take<D>(ws::Space::Local, static_cast<std::size_t>(C));
        size = p.size();
    }
};

// Cta<N>: m * n scalars unpadded, stepped over the CUDA launch hole (47104, 49664].
// evidence: docs/perf/qr.md#the-48-kib-launch-hole
inline constexpr std::size_t kCtaHoleLo = 47104;
inline constexpr std::size_t kCtaHoleHi = 49664;
inline constexpr std::size_t kCtaHolePadTo = 49920;
constexpr std::size_t cta_hole_padded(std::size_t bytes) {
    return (bytes > kCtaHoleLo && bytes <= kCtaHoleHi) ? kCtaHolePadTo : bytes;
}
template <class D> constexpr WorkspaceSize cta_size(int m, int n) {
    Plan p;
    p.take<D>(ws::Space::Local, static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    WorkspaceSize s = p.size();
    s.local_bytes = cta_hole_padded(s.local_bytes);
    return s;
}
// The Cta launch width: 32 * teams, teams = min(8, pow2 >= n) (teams track columns, lanes rows).
constexpr int cta_wg(int n) {
    int teams = 1;
    while (teams < 8 && teams < n) teams *= 2;
    return teams * 32;
}

// Row r := identity row r (so the unrolled body needs no lane guard), then A(r, 0:n) where
// row_live. Plain inline, as tiny_device.hh's helpers; `select`, never a complex `?:`.
template <class D, int N, class MA> BATCHLAS_DLA_HELPER void reg_load_rows(D* rA, const MA& a, int lane, int n, bool row_live) {
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) rA[c] = select(c == lane, Arith<D>::one(), D{});
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) rA[c] = a(lane, c);
    }
}
template <class D, int N, class MA> BATCHLAS_DLA_HELPER void reg_store_rows(const D* rA, const MA& a, int lane, int n, bool row_live) {
    BATCHLAS_UNROLL_FULL
    for (int c = 0; c < N; ++c) {
        if (row_live && c < n) a(lane, c) = rA[c];
    }
}

// The Reg<N> body (xGEQR2 in registers), the trailing update by C-column chunks with exactly two
// sub-group barriers each. NO EARLY RETURN: a dead partition carries the identity, stores nothing.
// evidence: docs/perf/qr.md#the-tiny-tier-wp6--p1-square-n--32-in-registers
template <int N, int C, class Ex, class MA, class VT, class D>
BATCHLAS_DLA_INLINE void geqrf_reg_body(const Ex& ex, const MA& a, const VT& tau, D* sP, D* sy, bool live) {
    using R = real_t<D>;
    using Ar = Arith<D>;
    using Fa = FmaArith<D>;  // arith::Fma: sycl_device::dev_mul / dev_div bit for bit
    constexpr int SLDA = C + 1;
    static_assert(N == 8 || N == 16 || N == 32, "geqrf Reg<N>: N divides the 32-lane sub-group");
    static_assert(C >= 1 && C <= N, "geqrf Reg<N>: the chunk width C is in [1, N]");
    const sycl::sub_group sg = ex.sub();
    const int lane = ex.lid();
    const int n = a.cols();
    const bool row_live = live && lane < n;

    D rA[N];  // top level, never a parameter: tiny_device.hh invariant 1
    reg_load_rows<D, N>(rA, a, lane, n, row_live);

    D my_tau = D{};  // one register, stored once N-wide rather than N times

    // unroll(FULL) on every loop indexing rA. evidence: docs/perf/qr.md#the-stack-frame-is-the-gate-for-this-kernel-not-the-spill-counter
#pragma clang loop unroll(full)
    for (int j = 0; j < N; ++j) {
        if (j >= n) continue;  // n is batch-uniform: the barrier sequence is uniform

        const D alpha = ex.bcast(rA[j], static_cast<uint32_t>(j));

        // Order-symmetric, so every lane leaves with a BIT-IDENTICAL smax and ssq. Widening
        // either leaks across partitions. evidence: docs/perf/qr.md#the-two-partition-butterflies
        R smax = (lane >= j) ? abs_inf(rA[j]) : R(0);
        smax = detail::butterfly<N, bfly::Asc>(ex.part, smax, op::FMax());

        R ssq = R(0);
        if (lane > j && smax > R(0)) ssq = abs2_scaled(rA[j], smax);
        ssq = detail::butterfly<N, bfly::Asc>(ex.part, ssq, sycl::plus<R>());

        const ScaledSsq<R> nrm{smax, ssq};
        const LarfgScalars<D> hh = larfg_scalars<conv::Lapack>(alpha, nrm);

        // NO `if (hh.identity) continue;`: inert arithmetically; keeps the barriers uniform.
        if (lane == j) {
            rA[j] = hh.beta;
            my_tau = hh.tau;
        } else if (lane > j) {
            rA[j] = hh.use_mul ? Fa::mul(rA[j], hh.scale) : Fa::div(rA[j], hh.scale);
        }

        // One select drives the apply: it zeroes eliminated rows (dropping `lane < j` lets them
        // re-enter the reflector) and supplies the implicit v(j) = 1.
        const D vr = (lane < j) ? D{} : ((lane == j) ? Ar::one() : rA[j]);
        const D cvr = Ar::conj(vr);
        // CONJ(tau), zgeqr2's convention. evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
        const D ctau = Ar::conj(hh.tau);

        // larf by C-column chunks; the column sums run in ROW order (LAPACK's).
#pragma clang loop unroll(full)
        for (int k0 = 0; k0 < N; k0 += C) {
            if (k0 + C <= j + 1) continue;  // wholly at or below the diagonal
            if (k0 >= n) continue;          // wholly past the runtime cap
            frag_scale_to_row<C>(cvr, rA, k0, j + 1, n, sP, lane * SLDA);
            sycl::group_barrier(sg);
            colsum_colperlane<C, N>(lane, ctau, sP, SLDA, j, sy);
            sycl::group_barrier(sg);
            // The two barriers cover all three hazards ONLY because sy is a separate array.
            // evidence: docs/perf/qr.md#break-sweeps-the-tiny-tier
            frag_rank1<C>(vr, sy, rA, k0, j + 1, n);
        }
        // No barrier between columns: rA[j+1] is register-private.
    }

    reg_store_rows<D, N>(rA, a, lane, n, row_live);
    if (row_live) tau(lane) = my_tau;
}

// The executor's own barrier at its scope, in the spelling the panel kernels use.
template <class Ex> BATCHLAS_DLA_INLINE void cta_sync(const Ex& ex) {
    if constexpr (Ex::sub_whole) sycl::group_barrier(ex.sub());
    else sycl::group_barrier(ex.it.get_group());
}

// The Cta<N> body: A (m x n, any memory) copied into `tile` (m * n scalars, ld = m), geqr2 there,
// copied back. kmax = min(m, n) and elems = m * n are the caller's (a host launcher folds them).
template <class Ex, class MA, class D>
BATCHLAS_DLA_INLINE void geqrf_cta_body(const Ex& ex, const MA& a, D* tau, D* tile, int kmax, std::size_t elems) {
    const int m = a.rows(), n = a.cols();
    const int tid = ex.lid();
    const int lwg = ex.width();
    for (std::size_t e = static_cast<std::size_t>(tid); e < elems; e += static_cast<std::size_t>(lwg)) {
        const int r = static_cast<int>(e % static_cast<std::size_t>(m));
        const int c = static_cast<int>(e / static_cast<std::size_t>(m));
        tile[e] = a(r, c);
    }
    cta_sync(ex);  // B0
    geqr2(ex, mat_wide(tile, m, n, m), kmax, tau);
    cta_sync(ex);  // B4
    for (std::size_t e = static_cast<std::size_t>(tid); e < elems; e += static_cast<std::size_t>(lwg)) {
        const int r = static_cast<int>(e % static_cast<std::size_t>(m));
        const int c = static_cast<int>(e / static_cast<std::size_t>(m));
        a(r, c) = tile[e];
    }
}

template <class Tier> inline constexpr bool is_reg_v = false;
template <int N, bool E> inline constexpr bool is_reg_v<tier::Reg<N, E>> = true;
template <class Tier> inline constexpr bool is_cta_v = false;
template <int N, class Alg> inline constexpr bool is_cta_v<tier::Cta<N, Alg>> = true;

template <class D> BATCHLAS_DLA_INLINE D* local_base(const Workspace& w, std::size_t offset) {
    return reinterpret_cast<D*>(w.local.get_raw() + offset);
}

}  // namespace geqrf_detail

/// @brief Local and global bytes one m x n matrix needs for geqrf() at this tier (host and device).
/// @tparam Tier tier::Reg<N> or tier::Cta<N>
/// @tparam T    the scalar the op runs on
/// @tparam C    Reg's chunk width (default N; the factors do not depend on it)
/// @param m, n  the matrix extents (Reg: square, unused)
/// @return the WorkspaceSize the op carves (the same plan the op uses)
template <class Tier, class T, int C = Tier::n_max> constexpr WorkspaceSize geqrf_buffer_size(int m, int n) {
    if constexpr (geqrf_detail::is_reg_v<Tier>) {
        static_cast<void>(m);
        static_cast<void>(n);
        return geqrf_detail::RegPlan<T, Tier::n_max, C>{}.size;
    } else {
        static_assert(geqrf_detail::is_cta_v<Tier>, "geqrf: tiers are Reg<N> and Cta<N>");
        return geqrf_detail::cta_size<T>(m, n);
    }
}
template <class Tier, class T, int C = Tier::n_max> constexpr WorkspaceSize geqrf_buffer_size(int n) {
    return geqrf_buffer_size<Tier, T, C>(n, n);
}

/// @brief xGEQRF inside a kernel: A = Q R, R in the upper triangle, Q's reflectors below it.
/// @tparam Tier tier::Reg<N> (N in {8, 16, 32}; square n <= N; ex = SgpPartition<N>, one matrix
///              per N-lane partition, lane r owns row r) or tier::Cta<N> (m, n <= N, A held in
///              local memory; ex = RtGroup<Scope::Group> or RtGroup<Scope::Sub>)
/// @param ex    the executor; COLLECTIVE: every item calls with the same arguments (Reg: every
///              lane of the sub-group)
/// @param A     the matrix operand (Mat, mat_wide, item(view, b)), overwritten
/// @param tau   tau(0:min(m, n)) receives the reflector scalars (zgeqr2's convention); Cta: a
///              unit-stride Vec
/// @param ws    this matrix's slice, geqrf_buffer_size<Tier, T>(m, n) bytes
/// @param live  Reg only: false for a partition with no matrix (it runs every collective on the
///              identity and touches no memory)
/// @return 0 (geqrf has no info; with_tier and tier_info::admits check the arguments on the host)
template <class Tier, int C = Tier::n_max, class Ex, class MA, class VT>
BATCHLAS_DLA_INLINE int geqrf(const Ex& ex, const MA& A, const VT& tau, const Workspace& ws, bool live = true) {
    using D = typename MA::value_type;
    if constexpr (geqrf_detail::is_reg_v<Tier>) {
        constexpr int N = Tier::n_max;
        static_assert(Ex::size == N && Ex::sub_whole, "geqrf Reg<N>: ex is an N-lane Partition (SgpPartition<N>)");
        constexpr geqrf_detail::RegPlan<D, N, C> plan{};
        geqrf_detail::geqrf_reg_body<N, C>(ex, A, tau, geqrf_detail::local_base<D>(ws, plan.tile.offset),
                                           geqrf_detail::local_base<D>(ws, plan.y.offset), live);
    } else {
        static_assert(geqrf_detail::is_cta_v<Tier>, "geqrf: tiers are Reg<N> and Cta<N>");
        static_assert(is_runtime_v<Ex>, "geqrf Cta<N>: ex is an RtGroup<Scope::Group> or RtGroup<Scope::Sub>");
        static_assert(std::is_same_v<std::remove_cv_t<decltype(VT::inc)>, Unit>, "geqrf Cta<N>: tau is a unit-stride Vec");
        static_cast<void>(live);
        const int m = A.rows(), n = A.cols();
        geqrf_detail::geqrf_cta_body(ex, A, &tau(0), geqrf_detail::local_base<D>(ws, 0), m < n ? m : n,
                                     static_cast<std::size_t>(m) * static_cast<std::size_t>(n));
    }
    return 0;
}

// No Reg<32> for complex<double>: evidence: docs/perf/qr.md#the-cdouble-n32-cell
template <class T> struct tiers<routine::geqrf, T> {
    using type = std::conditional_t<std::is_same_v<real_t<T>, double> && Arith<T>::is_complex,
                                    tier::list<tier::Reg<8>, tier::Reg<16>, tier::Cta<64>, tier::Cta<128>>,
                                    tier::list<tier::Reg<8>, tier::Reg<16>, tier::Reg<32>, tier::Cta<64>, tier::Cta<128>>>;
};
template <int N, class T> struct tier_info<routine::geqrf, tier::Reg<N>, T> {
    static constexpr int wg_size = 32, sg_size = 32;  // 32 / N matrices per sub-group
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr bool admits(int m, int n) { return m == n && admits(n); }
    static constexpr WorkspaceSize buffer_size(int n) { return geqrf_buffer_size<tier::Reg<N>, T>(n); }
};
template <int N, class Alg, class T> struct tier_info<routine::geqrf, tier::Cta<N, Alg>, T> {
    static constexpr int wg_size = geqrf_detail::cta_wg(N), sg_size = 32;
    static constexpr bool admits(int n) { return n >= 1 && n <= N; }
    static constexpr bool admits(int m, int n) { return m >= 1 && admits(n) && m <= N; }
    static constexpr WorkspaceSize buffer_size(int n) { return geqrf_buffer_size<tier::Cta<N, Alg>, T>(n); }
    static constexpr WorkspaceSize buffer_size(int m, int n) { return geqrf_buffer_size<tier::Cta<N, Alg>, T>(m, n); }
};

}  // namespace batchlas::device
