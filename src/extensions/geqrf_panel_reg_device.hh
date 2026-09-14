#pragma once

// The REGISTER panel leaf (WP6 / P5): one panel ROW per work-item, the m x N panel in registers
// between one coalesced load and one store. LarfgScalars/geqrf_larfg_scalars are reused VERBATIM,
// because tau's real-beta convention is a contract ormqr/orgqr/ormbr/sy2sb read.
// evidence: docs/perf/qr.md#the-register-panel-leaf-wp6--p5

#include "geqrf_cta_device.hh"

#include "../sycl/device_scalar.hh"

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

inline constexpr int kGeqrfPanelRegFilePerBlock = 65536;
inline constexpr int kGeqrfPanelRegMaxWg = 1024;

// Banks of eight: a rule on the raw count admits a work-group the hardware refuses at launch.
constexpr int geqrf_panel_reg_ceil8(int regs) { return (regs + 7) & ~7; }

// THE HARD GATE as a HEIGHT: regs x wg over the per-block register file aborts the launch. THE
// AOT PROBE DOES NOT DECIDE IT -- the arithmetic below is optimistic against the library's own
// allocation, so `launch_max_rows` wins.
// evidence: docs/perf/qr.md#the-height-ceiling-the-aot-probe-got-wrong
template <typename D>
constexpr int geqrf_panel_reg_max_rows() {
    if constexpr (GeqrfPanelRegPlan<D>::cols < 1) {
        return 0;
    } else {
        static_assert(GeqrfPanelRegPlan<D>::launch_max_rows > 0,
                      "a scalar with a register panel leaf must carry a MEASURED height ceiling");
        const int by_file = kGeqrfPanelRegFilePerBlock /
                            geqrf_panel_reg_ceil8(geqrf_panel_reg_regs<D>());
        const int capped = (by_file < kGeqrfPanelRegMaxWg) ? by_file : kGeqrfPanelRegMaxWg;
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

// LAPACK ?GEQR2 on an m x n panel with n <= N, in place, one row per work-item. CONTRACT the
// launcher holds: m <= work-group size, n <= N, kmax <= min(m, n), all launch-uniform. Rows at or
// above m carry zeros and suppress their store but still execute every barrier -- an early return
// is not work-group-uniform. Three barriers per column, none double-buffered: column j's writes
// all follow column j-1's last barrier.
template <typename D, int N, typename Real>
inline void geqr2_panel_reg_device(sycl::nd_item<1> it, D* a, int ld, int m, int n, int kmax,
                                   D* tau_ptr, Real* sred, D* sw) {
    using R = real_of<D>;
    static_assert(sizeof(R) == sizeof(Real), "the real slot array must match the scalar's real");

    const auto g = it.get_group();
    const auto sg = it.get_sub_group();
    const int r = static_cast<int>(it.get_local_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int nsg = static_cast<int>(sg.get_group_linear_range());
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

        R part = (live && r >= j) ? dev_absmax<D>(rA[j]) : R(0);
        part = geqrf_sg_max<R>(sg, part);
        if (lane == 0) sred[sgid] = part;
        if (r == j) sw[0] = rA[j];
        sycl::group_barrier(g);                                        // B1

        // Same slots in the same order on every item, so h below is work-group-uniform.
        R smax = sred[0];
        for (int i = 1; i < nsg; ++i) smax = sycl::fmax(smax, sred[i]);
        const D alpha = sw[0];

        R ssq_part = R(0);
        if (live && r > j && smax > R(0)) ssq_part = dev_abs2_scaled<D>(rA[j], smax);
        ssq_part = geqrf_sg_sum<R>(sg, ssq_part);
        if (lane == 0) sred[nsg + sgid] = ssq_part;
        sycl::group_barrier(g);                                        // B2

        R ssq = R(0);
        for (int i = 0; i < nsg; ++i) ssq += sred[nsg + i];

        const LarfgScalars<D> h = geqrf_larfg_scalars<D>(alpha, smax, ssq);

        // NO `if (h.identity) continue;`: inert arithmetically, and omitting it keeps the
        // barrier sequence uniform. tau is a defined output either way.
        if (r == j) {
            rA[j] = h.beta;
            tau_ptr[j] = h.tau;
        } else if (live && r > j) {
            rA[j] = h.use_mul ? sycl_device::dev_mul(rA[j], h.vfactor)
                              : sycl_device::dev_div(rA[j], h.vfactor);
        }

        // One select: zeroes eliminated and dead rows, supplies the implicit v(j) = 1. Dropping
        // the `r < j` term lets an eliminated row re-enter the reflector.
        const D v = (!live || r < j)
                        ? dev_zero<D>()
                        : ((r == j) ? sycl_device::dev_one<D>() : rA[j]);
        const D cv = sycl_device::dev_conj(v);
        // CONJ(tau), zgeqr2's convention. evidence: docs/perf/qr.md#a-residual-test-cannot-guard-a-convention
        const D ctau = sycl_device::dev_conj(h.tau);

#pragma clang loop unroll(full)
        for (int k = 0; k < N; ++k) {
            if (k <= j || k >= n) continue;   // uniform, or lanes drop out of the butterfly
            const D pw = geqrf_sg_sum<D>(sg, sycl_device::dev_mul(cv, rA[k]));
            if (lane == 0) sw[1 + sgid * N + k] = pw;
        }
        sycl::group_barrier(g);                                        // B3

#pragma clang loop unroll(full)
        for (int k = 0; k < N; ++k) {
            if (k <= j || k >= n) continue;
            D w = dev_zero<D>();
            for (int i = 0; i < nsg; ++i) w = dev_add<D>(w, sw[1 + i * N + k]);
            rA[k] = sycl_device::dev_sub(
                rA[k], sycl_device::dev_mul(sycl_device::dev_mul(ctau, w), v));
        }
    }

#pragma clang loop unroll(full)
    for (int k = 0; k < N; ++k) {
        if (live && k < n) {
            a[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(k) * ld] = rA[k];
        }
    }
}

}  // namespace batchlas::geqrf_native
