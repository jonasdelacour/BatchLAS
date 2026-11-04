// SPDX-License-Identifier: MIT
#pragma once

// The register-resident GETRF panel leaf: item `tid` owns row `tid` in registers, pivoting by
// LAZY RELABEL (getf2_relabel, getrf_v2.hh, keeps its three load-bearing spellings);
// getf2_panel_device's contract. evidence: docs/perf/lu.md#the-register-panel-leaf

#include <src/ops/getrf/getrf_v2.hh>

#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

// `kmax` must be KERNEL-UNIFORM (skipped steps skip barriers). `always_inline` is NOT a hint:
// evidence: docs/perf/lu.md#the-register-panel-leaf-register-probe
template <typename D, int NB, typename SxAcc, typename ValAcc, typename IdxAcc>
[[gnu::always_inline]] inline void getf2_panel_reg_device(sycl::nd_item<1> it,
                                                          D* a, int ld,
                                                          int m, int ncols, int kmax,
                                                          int* piv_item, int piv_base,
                                                          int32_t* info_item,
                                                          SxAcc sx, ValAcc rval,
                                                          IdxAcc ridx) {
    const LuExec<LuScope::WorkGroup> ex{it};
    const int tid = ex.lid();

    // CLAMP, never return: a pad item is still in every shuffle mask and every barrier.
    const bool live = (tid < m);
    const std::ptrdiff_t ldp = static_cast<std::ptrdiff_t>(ld);

    v2::Frag<D, NB> rA;  // top level, never a parameter: tiny_device.hh invariant 1
BATCHLAS_UNROLL_FULL
    for (int c = 0; c < NB; ++c) {
        D v = lu_zero<D>();   // a STATEMENT, not `?:`: see spelling 3
        if (live && c < ncols) v = a[tid + static_cast<std::ptrdiff_t>(c) * ldp];
        rA[c] = v;
    }

    int rowid = live ? tid : -1;   // -1: never a candidate, never == j, never > j
    int32_t info_local = (tid == 0) ? *info_item : 0;

    // iamax (B1), relabel, pivot row through sx (B2), rscl, ger: docs/perf/lu.md#two-barriers-per-column
    auto piv = pivot_rows_group(ex, rval, ridx, m, Ipiv{piv_item, piv_base}, info_local);
    auto bc = bcast_row_local(ex, sx);
    getf2_relabel(rA, rowid, kmax, ncols, piv, bc);

    if (live) {  // to the RELABELLED row: rowid is where this data belongs in P A
BATCHLAS_UNROLL_FULL
        for (int k = 0; k < NB; ++k) {
            if (k >= ncols) continue;
            a[rowid + static_cast<std::ptrdiff_t>(k) * ldp] = rA[k];
        }
    }
    if (tid == 0) *info_item = info_local;
}

}  // namespace batchlas::getrf_native
