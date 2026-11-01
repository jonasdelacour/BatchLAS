// SPDX-License-Identifier: MIT
#pragma once

// The native batched POTRF LPANEL kernel: LEFT-LOOKING, thread-per-row, on device LA v2 ops. Local
// memory holds one `n x NB` panel; the columns to its left are re-read from global memory.
// Invariants: (1) EVERY barrier is a top-level WORK-GROUP barrier and failure is a predicated skip,
// so the schedule depends on (n, NB) alone -- what makes packing G matrices per work-group legal.
// (2) B6 is not optional: the next panel GLOBALLY reads the columns just stored, by other items.
// (3) Every global read and store is guarded by `row >= column` (ortho.cc depends on it).
// (4) rp, rS and rA stay in registers. evidence: docs/perf/potrf.md#the-lpanel-tier

#include <src/sycl/device_scalar.hh>
#include <src/sycl/kernel_attrs.hh>

#include <batchlas/blas/device/v2/ext/potrf-cta-lpanel.hh>

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
    namespace v2 = batchlas::device::v2;
    using Ar = v2::ArithFma<D>;
    const v2::DynItems ex{tid, L};
    const auto A = v2::mat(Ag, n, n, ldg);
    const auto P = v2::slm_mat(sA, n, NB, slda);  // local-space views: generic reads were LD.E
    const auto B = v2::slm_mat<NB, NB>(sB);
    auto bar = [&] { sycl::group_barrier(it.get_group()); };

    if (ex.leader()) *fail = 0;
    bar();  // B0

    const int row = tid;
    bool ok = true;  // matrix-uniform: every lane derives it from the same local pivot

    for (int j = 0; j < n; j += NB) {
        const int ib = (n - j < NB) ? (n - j) : NB;
        const bool panel_ok = ok;
        const bool live = panel_ok && (row >= j) && (row < n);

        v2::Frag<D, NB> rp, rS;  // (1) prefetch row `row` of the panel's lower triangle
        v2::frag_load_row_herm_lower<NB, Ar>(rp, A, row, j, ib, live);
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < NB; ++i) rS[i] = D{};

        // (2) rS(row, i) = sum_{k<j} L(row, k) conj(L(j+i, k)) against the ZERO-padded broadcast
        // block, so the FMA loop is a fully unrolled NB x NB with no inner bound test.
        for (int k = 0; k < j; k += NB) {
            const int kb = ((j - k) < NB) ? (j - k) : NB;
            v2::lacpy_conj_pad<NB, Ar>(ex, A.sub(j, k, ib, kb), B);
            bar();  // B1
            if (live) {
                v2::Frag<D, NB> rA;
                v2::frag_load_row<NB>(rA, A, row, k, kb);
                v2::gemv_n_slm_bcast<NB, Ar>(rS, rA, sB);  // evidence: docs/perf/blackwell.md#potrf-lpanel-vector-sb
            }
            bar();  // B2 -- WAR on sB
        }

        if (live) {  // (3) publish rp - rS; the strictly upper diagonal block is ZEROED
            v2::frag_store_row_herm_lower<NB, Ar>(P, row, j, ib, [&](int i) { return Ar::sub(rp[i], rS[i]); });
        }
        bar();  // B3

        // (4) potf2 with B3b / B4 / B5 per column. evidence: docs/perf/potrf.md#the-pivot-cell-war-race
        ok = v2::potf2_panel_rows<Ar>(ex, P, j, ib, n, ok, fail, bar);

        if (store && panel_ok && row < n) v2::copy_row_lower(P, A, row, j, ib);  // (5) ENTRY state: a failed panel is finite
        bar();  // B6 -- the next panel READS these columns
    }
}

}  // namespace batchlas::potrf_native
