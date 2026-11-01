// SPDX-License-Identifier: MIT
#pragma once

// The REGISTER panel leaf: one panel ROW per item, m x N in registers between one load and one
// store; v2::larfg_from_ssq is the xLARFG of all three leaves (tau is an ormqr/orgqr contract).
// evidence: docs/perf/qr.md#the-register-panel-leaf-wp6--p5

#include <src/ops/geqrf/geqrf_cta_device.hh>

#include <src/sycl/device_scalar.hh>
#include <src/util/resident_capacity.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::geqrf_native {

// THE ONE source of this leaf's width and register cost. cols 0 = ABSENT; a non-zero cols MUST
// equal geqrf_nb_for_type<T>(), which geqrf_blocked.cc static_asserts.
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
    // 60.5 s of ptxas, and clamped at the 255-register ceiling: one block per SM.
    static constexpr int cols = 0;
    static constexpr int probed_cols = 32;
    static constexpr int probed_regs = 255;
    static constexpr int launch_max_rows = 0;
};

// A MODEL, for cells with no probe row only -- measured wrong in BOTH directions here.
// evidence: docs/perf/qr.md#the-register-panel-leaf-width-and-height-table
inline constexpr int kGeqrfPanelRegOverhead = 96;
inline constexpr int kGeqrfPanelRegMargin = 16;

// A row probed at a width the plan no longer uses reads as measured but describes another kernel.
template <typename D>
constexpr bool geqrf_panel_reg_row_is_current() {
    return GeqrfPanelRegPlan<D>::cols == 0 || GeqrfPanelRegPlan<D>::probed_regs == 0 ||
           GeqrfPanelRegPlan<D>::probed_cols == GeqrfPanelRegPlan<D>::cols;
}

constexpr int geqrf_panel_reg_words(std::size_t scalar_bytes) {
    return static_cast<int>(scalar_bytes / sizeof(float));
}

// A PROBED row needs no margin; the margin is the model's, because the model is an estimate.
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

// THE HARD GATE as a HEIGHT: warps overflowing a register SUB-PARTITION abort the launch. THE
// ARITHMETIC DOES NOT DECIDE IT -- sm89_max_work_group divides a PROBED count, optimistic against
// what the library allocates, so the MEASURED launch_max_rows stays in the min() and wins.
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

// THE POLICY HEIGHT, a different fact from the hard ceiling: the tallest panel at which this
// leaf is the right CHOICE. evidence: docs/perf/qr.md#the-panel-height-window
inline constexpr int kGeqrfPanelRegPolicyRows = 128;

constexpr int geqrf_panel_reg_wg(int m) {
    const int w = ((m + 31) / 32) * 32;
    return (w < 32) ? 32 : w;
}

// Two real slots and N scalar slots per sub-group, plus alpha. The reductions are butterflies and
// NEVER reduce_over_group, whose static shared opens the hole. evidence: docs/perf/qr.md#the-48-kib-launch-hole
constexpr std::size_t geqrf_panel_reg_real_slots(int wg) {
    return 2u * static_cast<std::size_t>(wg / 32);
}
constexpr std::size_t geqrf_panel_reg_scalar_slots(int wg, int cols) {
    return 1u + static_cast<std::size_t>(wg / 32) * static_cast<std::size_t>(cols);
}

// ?GEQR2 in place, one row per item. CONTRACT: m <= WG, n <= N, kmax <= min(m, n), launch-uniform.
// Rows >= m carry zeros, skip their store and execute every barrier (an early return is not
// work-group-uniform). Three barriers per column; column j's writes follow column j-1's last.
template <typename D, int N, typename Real>
inline void geqr2_panel_reg_device(sycl::nd_item<1> it, D* a, int ld, int m, int n, int kmax,
                                   D* tau_ptr, Real* sred, D* sw) {
    namespace v2 = batchlas::device::v2;
    using R = real_of<D>;
    static_assert(sizeof(R) == sizeof(Real), "the real slot array must match the scalar's real");

    const v2::DynWorkGroup ex{it};
    const int r = ex.lid();
    const int nsg = ex.nteams_dyn();
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
        const R part = (live && r >= j) ? v2::absmax_ri(rA[j]) : R(0);
        v2::team_slot_put(ex, sred, 0, ex.team_reduce(part, v2::Fmax{}));
        if (r == j) sw[0] = rA[j];
        ex.barrier();                                                  // B1
        const R smax = v2::slot_fold_max(sred, 0, nsg);
        const D alpha = sw[0];

        R ssq_part = R(0);
        if (live && r > j && smax > R(0)) ssq_part = v2::abs2_scaled(rA[j], smax);
        v2::team_slot_put(ex, sred, nsg, ex.team_reduce(ssq_part));
        ex.barrier();                                                  // B2
        const R ssq = v2::slot_fold_sum(sred, nsg, nsg);

        const LarfgScalars<D> h = v2::larfg_from_ssq<v2::Conv::Lapack>(alpha, smax, ssq);

        // NO `if (h.identity) continue;`: inert, and the barrier sequence stays uniform.
        if (r == j) {
            rA[j] = h.beta;
            tau_ptr[j] = h.tau;
        } else if (live && r > j) {
            rA[j] = h.use_mul ? v2::mul_fma(rA[j], h.vfactor) : v2::div_smith(rA[j], h.vfactor);
        }

        // One select: zero eliminated (`r < j`, load-bearing) and dead rows, implicit v(j) = 1.
        const D v = (!live || r < j)
                        ? dev_zero<D>()
                        : ((r == j) ? sycl_device::dev_one<D>() : rA[j]);
        // larf: team partials of v^H A(:, j+1:n), B3, A -= (CONJ(tau) w) v (zgeqr2's convention).
        // evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
        v2::gemv_rowfrag_partials<N>(ex, v2::Arith<D>::conj(v), rA, j + 1, n, sw, 1, N);
        ex.barrier();                                                  // B3
        v2::ger_rowfrag_slots<N>(v2::Arith<D>::conj(h.tau), v, rA, j + 1, n, sw, 1, nsg, N);
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (live && k < n) {
            a[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k) * ld] = rA[k];
        }
    }
}

}  // namespace batchlas::geqrf_native
