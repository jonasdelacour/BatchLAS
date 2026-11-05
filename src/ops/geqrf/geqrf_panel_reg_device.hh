// SPDX-License-Identifier: MIT
#pragma once

// The REGISTER panel leaf's plan: one panel ROW per work-item, the m x N panel in registers. The
// body is the public device::geqr2_reg (include/batchlas/device/lapack/geqrf.hh); xLARFG is
// device::larfg_scalars<conv::Lapack> (tau's real-beta convention is read by ormqr/orgqr/ormbr/sy2sb).
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

}  // namespace batchlas::geqrf_native
