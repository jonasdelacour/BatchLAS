#pragma once

// P4's register-resident GETRF panel leaf: one work-group per m x ncols panel, item `tid`
// owning row `tid` of a compile-time `D rA[NB]`, pivoting by LAZY RELABEL. Same contract
// as getf2_panel_device. THREE SPELLINGS ARE SILENT WRONG ANSWERS OR SILENT SPILLS: the
// `j` loop's COMPILE-TIME bound with `continue` (never `break`); publishing the pivot row
// AFTER the relabel, by the item whose `act` is therefore false; and selecting a complex
// scalar by if/else and never `?:`. evidence: docs/perf/lu.md#the-register-panel-leaf

#include "getrf_cta_device.hh"

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

// `piv_item` carries the panel's piv_base; `kmax` = min(m, ncols) must be KERNEL-UNIFORM,
// because the `continue` guards below skip work-group barriers. `always_inline` is NOT a
// hint: without it the ABI call passes the accessors through a 200-byte stack depot.
// evidence: docs/perf/lu.md#the-register-panel-leaf-register-probe
template <typename D, int NB, typename SxAcc, typename ValAcc, typename IdxAcc>
[[gnu::always_inline]] inline void getf2_panel_reg_device(sycl::nd_item<1> it,
                                                          D* a, int ld,
                                                          int m, int ncols, int kmax,
                                                          int* piv_item, int piv_base,
                                                          int32_t* info_item,
                                                          SxAcc sx, ValAcc rval,
                                                          IdxAcc ridx) {
    using R = real_of<D>;
    namespace sd = ::batchlas::sycl_device;

    const auto g = it.get_group();
    const auto sg = it.get_sub_group();
    const int tid = static_cast<int>(it.get_local_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int nlanes = static_cast<int>(sg.get_local_linear_range());
    const int team = static_cast<int>(sg.get_group_linear_id());
    const int nteams = static_cast<int>(sg.get_group_linear_range());

    // CLAMP, never return: a pad item is still in every shuffle mask and every barrier.
    const bool live = (tid < m);
    const std::ptrdiff_t ldp = static_cast<std::ptrdiff_t>(ld);

    D rA[NB];  // top level, never a parameter: tiny_device.hh invariant 1
#pragma unroll
    for (int c = 0; c < NB; ++c) {
        D v = lu_zero<D>();   // a STATEMENT, not `?:`: see spelling 3
        if (live && c < ncols) v = a[tid + static_cast<std::ptrdiff_t>(c) * ldp];
        rA[c] = v;
    }

    int rowid = live ? tid : -1;   // -1: never a candidate, never == j, never > j
    int32_t info_local = (tid == 0) ? *info_item : 0;

#pragma unroll
    for (int j = 0; j < NB; ++j) {
        if (j >= kmax) continue;   // see spelling 1 at the top of this file

        // --- 1. argmax over logical rows >= j; a NaN magnitude is MAPPED to -1 or lanes
        // disagree about the winner. evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
        const R mag = lu_cabs1<D>(rA[j]);
        const bool cand = (rowid >= j);
        R bv = (cand && (mag == mag)) ? mag : R(-1);
        int bi = cand ? rowid : m;      // `m` loses every tie to a real row
        for (uint32_t off = static_cast<uint32_t>(nlanes) / 2u; off > 0u; off >>= 1) {
            const R ov = sycl::permute_group_by_xor(sg, bv, off);
            const int oi = sycl::permute_group_by_xor(sg, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            rval[static_cast<std::size_t>(team)] = bv;
            ridx[static_cast<std::size_t>(team)] = bi;
        }
        sycl::group_barrier(g);                                        // B1

        R fv = R(-1);
        int p = m;
        for (int t = 0; t < nteams; ++t) {
            const R v = rval[static_cast<std::size_t>(t)];
            const int ii = ridx[static_cast<std::size_t>(t)];
            if (v > fv || (v == fv && ii < p)) { fv = v; p = ii; }
        }
        static_cast<void>(fv);
        if (p >= m) p = j;   // unreachable: j < kmax <= m, so logical row j exists

        if (tid == 0) piv_item[j] = piv_base + p + 1;   // GLOBAL, 1-BASED, LAPACK ipiv

        // --- 2. the lazy swap: two labels change, no register moves work-item.
        if (rowid == p) rowid = j;
        else if (rowid == j) rowid = p;

        // --- 3. publish the pivot row; exactly one item now carries rowid == j.
        if (rowid == j) {
#pragma unroll
            for (int k = j; k < NB; ++k) sx[static_cast<std::size_t>(k)] = rA[k];
        }
        sycl::group_barrier(g);                                        // B2

        const D d = sx[static_cast<std::size_t>(j)];
        const bool zero = sd::dev_is_zero(d);      // EXACT zero, no epsilon
        if (zero && tid == 0 && info_local == 0) {
            info_local = static_cast<int32_t>(piv_base + j + 1);
        }
        const D rc = sd::dev_recip(d);
        const bool use_mul = !zero && sd::dev_isfinite(rc) && !sd::dev_is_zero(rc);
        const bool act = (rowid > j);
        if (act) {
            // --- 4. scale. An if/else STATEMENT, never `?:` -- see spelling 3.
            if (use_mul) rA[j] = sd::dev_mul(rA[j], rc);
            else if (!zero) rA[j] = sd::dev_div(rA[j], d);   // ?GETF2's sfmin arm
        }

        // `k = j + 1` is a COMPILE-TIME lower bound once j is unrolled.
        // evidence: docs/perf/lu.md#why-the-rank-1-update-starts-at-k--j--1
#pragma unroll
        for (int k = j + 1; k < NB; ++k) {
            if (k >= ncols) continue;
            const D u = sx[static_cast<std::size_t>(k)];
            // Unguarded by `zero`: a zero pivot WAS the argmax, so every multiplier is 0.
            if (act) rA[k] = sd::dev_sub(rA[k], sd::dev_mul(rA[j], u));
        }
        // No third barrier. evidence: docs/perf/lu.md#two-barriers-per-column
    }

    // --- 5. store to the RELABELLED row: rowid is where this data belongs in P A.
    if (live) {
#pragma unroll
        for (int k = 0; k < NB; ++k) {
            if (k >= ncols) continue;
            a[rowid + static_cast<std::ptrdiff_t>(k) * ldp] = rA[k];
        }
    }
    if (tid == 0) *info_item = info_local;
}

}  // namespace batchlas::getrf_native
