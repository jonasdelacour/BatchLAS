// SPDX-License-Identifier: MIT
#pragma once

// The REGISTER panel leaf: one panel ROW per work-item, the m x N panel in registers. xLARFG is
// v2::larfg_scalars<conv::Lapack> (tau's real-beta convention is read by ormqr/orgqr/ormbr/sy2sb).
// evidence: docs/perf/qr.md#the-register-panel-leaf-wp6--p5

#include <src/ops/geqrf/geqrf_cta_device.hh>

#include <src/sycl/device_scalar.hh>
#include <src/util/resident_capacity.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::geqrf_native {

// THE ONE source of this leaf's width and cost. cols 0 = ABSENT; else MUST equal geqrf_nb_for_type<T>().
// evidence: docs/perf/qr.md#the-register-panel-leaf-width-and-height-table
template <typename D> struct GeqrfPanelRegPlan {
    static constexpr int cols = 0;
    static constexpr int probed_cols = 0;
    static constexpr int probed_regs = 0;
    static constexpr int launch_max_rows = 0;
};
template <> struct GeqrfPanelRegPlan<float> {
    static constexpr int cols = 0;          // 32 costs 39.6 s of ptxas; set it to 32 to re-enable
    static constexpr int probed_cols = 32;
    static constexpr int probed_regs = 168;
    static constexpr int launch_max_rows = 0;   // must be MEASURED before cols goes non-zero
};
template <> struct GeqrfPanelRegPlan<double> {
    static constexpr int cols = 16;
    static constexpr int probed_cols = 16;
    static constexpr int probed_regs = 136;
    // MEASURED on the SHIPPED library, not what probed_regs predicts. evidence: below.
    static constexpr int launch_max_rows = 384;
};
template <> struct GeqrfPanelRegPlan<sycl_device::Cx<float>> {
    static constexpr int cols = 0;          // 32 costs 46.0 s of ptxas
    static constexpr int probed_cols = 32;
    static constexpr int probed_regs = 180;
    static constexpr int launch_max_rows = 0;
};
template <> struct GeqrfPanelRegPlan<sycl_device::Cx<double>> {
    // 60.5 s of ptxas; at the 255-reg ceiling: one block per SM.
    static constexpr int cols = 0;
    static constexpr int probed_cols = 32;
    static constexpr int probed_regs = 255;
    static constexpr int launch_max_rows = 0;
};

// A MODEL, for cells with no probe row only -- measured wrong in BOTH directions here.
// evidence: docs/perf/qr.md#the-register-panel-leaf-width-and-height-table
inline constexpr int kGeqrfPanelRegOverhead = 96;
inline constexpr int kGeqrfPanelRegMargin = 16;

// A row probed at a width the plan no longer uses describes another kernel.
template <typename D>
constexpr bool geqrf_panel_reg_row_is_current() {
    return GeqrfPanelRegPlan<D>::cols == 0 || GeqrfPanelRegPlan<D>::probed_regs == 0 ||
           GeqrfPanelRegPlan<D>::probed_cols == GeqrfPanelRegPlan<D>::cols;
}

constexpr int geqrf_panel_reg_words(std::size_t scalar_bytes) {
    return static_cast<int>(scalar_bytes / sizeof(float));
}

template <typename D>
constexpr int geqrf_panel_reg_regs() {
    constexpr int probed = GeqrfPanelRegPlan<D>::probed_regs;
    if constexpr (probed > 0) {
        return probed;
    } else {
        return GeqrfPanelRegPlan<D>::cols * geqrf_panel_reg_words(sizeof(D)) +
               kGeqrfPanelRegOverhead + kGeqrfPanelRegMargin;
    }
}

inline constexpr int kGeqrfPanelRegMaxWg = 1024;

// THE HARD GATE as a HEIGHT: warps overflowing a register SUB-PARTITION abort the launch. The
// optimistic PROBED arithmetic does not decide it: the MEASURED launch_max_rows wins the min().
// evidence: docs/perf/qr.md#the-height-ceiling-the-aot-probe-got-wrong
template <typename D>
constexpr int geqrf_panel_reg_max_rows() {
    if constexpr (GeqrfPanelRegPlan<D>::cols < 1) {
        return 0;
    } else {
        static_assert(GeqrfPanelRegPlan<D>::launch_max_rows > 0,
                      "a scalar with a register panel leaf must carry a MEASURED height ceiling");
        const int by_regs = resident::sm89_max_work_group(geqrf_panel_reg_regs<D>());
        const int capped = (by_regs < kGeqrfPanelRegMaxWg) ? by_regs : kGeqrfPanelRegMaxWg;
        const int measured = GeqrfPanelRegPlan<D>::launch_max_rows;
        const int lo = (measured < capped) ? measured : capped;
        return (lo / 32) * 32;
    }
}

// THE POLICY HEIGHT (not the hard ceiling): tallest panel where this leaf is the right CHOICE.
// evidence: docs/perf/qr.md#the-panel-height-window
inline constexpr int kGeqrfPanelRegPolicyRows = 128;

constexpr int geqrf_panel_reg_wg(int m) {
    const int w = ((m + 31) / 32) * 32;
    return (w < 32) ? 32 : w;
}

// Two real slots and N scalar slots per sub-group, plus alpha. Butterflies, NEVER reduce_over_group
// (its static shared opens the hole). evidence: docs/perf/qr.md#the-48-kib-launch-hole
constexpr std::size_t geqrf_panel_reg_real_slots(int wg) {
    return 2u * static_cast<std::size_t>(wg / 32);
}
constexpr std::size_t geqrf_panel_reg_scalar_slots(int wg, int cols) {
    return 1u + static_cast<std::size_t>(wg / 32) * static_cast<std::size_t>(cols);
}

// LAPACK ?GEQR2 on an \f$m \times n\f$ panel with \f$n \le N\f$, in place, one row per work-item. CONTRACT the
// launcher holds: \f$m \le\f$ work-group size, \f$n \le N\f$, \f$\mathrm{kmax} \le \min(m, n)\f$, all launch-uniform. Rows >= m
// carry zeros, skip the store but run every barrier (an early return is not work-group-uniform).
// Three barriers per column, none double-buffered.
template <typename D, int N, typename Real>
inline void geqr2_panel_reg_device(sycl::nd_item<1> it, D* a, int ld, int m, int n, int kmax,
                                   D* tau_ptr, Real* sred, D* sw) {
    namespace v2 = batchlas::device::v2;
    namespace q2 = batchlas::geqrf_v2;
    using R = real_of<D>;
    using A = v2::Arith<D>;
    using F = v2::FmaArith<D>;  // arith::Fma: sycl_device::dev_mul / dev_div bit for bit
    static_assert(sizeof(R) == sizeof(Real), "the real slot array must match the scalar's real");

    const v2::RtGroup<v2::Scope::Group> ex{it};
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
                                : dev_zero<D>();
    }

#pragma clang loop unroll(full)
    for (int j = 0; j < N; ++j) {
        if (j >= kmax) continue;   // launch-uniform: cannot desync the barriers below

        // Same slots folded in the same order on every item: h is work-group-uniform.
        const R part = (live && r >= j) ? q2::absmax_ri(rA[j]) : R(0);
        q2::team_slot_put(ex, sred, 0, q2::xor_down_rt(ex.sub(), part, q2::Fmax()));
        if (r == j) sw[0] = rA[j];
        ex.barrier(v2::Mem::Global);                                   // B1
        const R smax = q2::slot_fold_max(sred, 0, nsg);
        const D alpha = sw[0];

        R ssq_part = R(0);
        if (live && r > j && smax > R(0)) ssq_part = q2::abs2_scaled(rA[j], smax);
        q2::team_slot_put(ex, sred, nsg, q2::xor_down_rt(ex.sub(), ssq_part, sycl::plus<R>()));
        ex.barrier(v2::Mem::Global);                                   // B2
        const R ssq = q2::slot_fold_sum(sred, nsg, nsg);

        const v2::LarfgScalars<D> h = v2::larfg_scalars<v2::conv::Lapack>(alpha, v2::ScaledSsq<R>{smax, ssq});

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
        q2::gemv_rowfrag_partials<N>(ex, cv, rA, j + 1, n, sw, 1, N);
        ex.barrier(v2::Mem::Global);                                   // B3
        q2::ger_rowfrag_slots<N>(ctau, v, rA, j + 1, n, sw, 1, nsg, N);
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (live && k < n) {
            a[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k) * ld] = rA[k];
        }
    }
}

}  // namespace batchlas::geqrf_native
