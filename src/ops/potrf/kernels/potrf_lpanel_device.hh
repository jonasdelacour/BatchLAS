// SPDX-License-Identifier: MIT
#pragma once

// The native batched POTRF LPANEL kernel: LEFT-LOOKING, thread-per-row, on device LA ops. Local
// memory holds one `n x NB` panel; the columns to its left are re-read from global memory.
// Invariants: (1) EVERY barrier is a top-level WORK-GROUP barrier and failure is a predicated skip,
// so the schedule depends on (n, NB) alone -- what makes packing G matrices per work-group legal.
// (2) B6 is not optional: the next panel GLOBALLY reads the columns just stored, by other items.
// (3) Every global read and store is guarded by `row >= column` (ortho.cc depends on it).
// (4) rp, rS and rA stay in registers. evidence: docs/perf/potrf.md#the-lpanel-tier

#include <batchlas/device/lapack/potrf.hh>
#include <src/sycl/device_scalar.hh>
#include <batchlas/device/detail/kernel_attrs.hh>

#include <batchlas/device/lapack_aux.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::potrf_native {


// ONE matrix, Uplo::Lower. `L >= n`: lane `r` owns row `r` for the whole call. `store` is false for
// a packed group's dead slots, which run the WHOLE body (uniform barrier count), stores suppressed.
template <typename D, typename R, int NB, bool Tag = false>  // Tag: one inlined instance per kernel
inline void potrf_lpanel_body(const sycl::nd_item<1>& it,
                              int tid, int L, bool store,
                              D* __restrict sA, int slda,
                              D* __restrict sB,
                              int* __restrict fail,
                              D* __restrict Ag, int ldg,
                              int n) {
    using Ar = device::arith_t<device::arith::Fma, D>;
    const device::Slot<> ex{tid, L};  // this matrix's slot of the packed work-group
    const auto A = device::mat_wide(Ag, n, n, ldg);
    const auto P = device::detail::slm_mat(sA, n, NB, slda);  // local-space views: generic reads were LD.E
    const auto B = device::detail::slm_tile<NB, NB>(sB);
    auto bar = [&]() BATCHLAS_DEVICE_BODY { sycl::group_barrier(it.get_group()); };

    if (ex.leader()) *fail = 0;
    bar();  // B0

    const int row = tid;
    bool ok = true;  // matrix-uniform: every lane derives it from the same local pivot

    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const bool panel_ok = ok;
        const bool live = panel_ok && (row >= j) && (row < n);

        device::Frag<D, NB> rp, rS;  // (1) prefetch row `row` of the panel's lower triangle
        device::detail::frag_load_row_herm_lower<NB>(rp, A, row, j, ib, live);
        device::fill(rS, Ar::zero());

        // (2) rS(row, i) = sum_{k<j} L(row, k) conj(L(j+i, k)) against the ZERO-padded broadcast
        // block, so the FMA loop is a fully unrolled NB x NB with no inner bound test.
        for (int k = 0; k < j; k += NB) {
            const int kb = ((j - k) < NB) ? (j - k) : NB;
            device::lacpy<device::Part::Full, Transpose::NoTrans, device::alg::DivMod, device::alg::EdgeZeroPad>(ex, device::conj_view(A.sub(j, k, ib, kb)), B);
            bar();  // B1
            if (live) {
                device::Frag<D, NB> rA;
                device::load_row(rA, A.sub(0, k, n, kb), row, kb);
                device::detail::gemv_n_slm_bcast<NB>(rS, rA, sB);  // evidence: docs/perf/blackwell.md#potrf-lpanel-vector-sb
            }
            bar();  // B2 -- WAR on sB
        }

        if (live) {  // (3) publish rp - rS; the strictly upper diagonal block is ZEROED
            device::detail::frag_store_row_herm_lower<NB>(P, row, j, ib, [&](int i) BATCHLAS_DEVICE_BODY { return Ar::sub(rp[i], rS[i]); });
        }
        bar();  // B3

        // (4) potf2 with B3b / B4 / B5 per column. evidence: docs/perf/potrf.md#the-pivot-cell-war-race
        ok = device::potf2_panel_rows<device::detail::InvSqrt>(ex, P, j, ib, n, ok, fail, bar);

        if (store && panel_ok) device::lacpy<device::Part::Lower>(ex, P.sub(j, 0, n - j, ib), A.sub(j, j, n - j, ib));  // (5) ENTRY state: a failed panel is finite
        bar();  // B6 -- the next panel READS these columns
    }
}

}  // namespace batchlas::potrf_native
