#pragma once

// LASWP -- apply a LAPACK interchange list to the rows of a column block.
//
// ONE implementation, shared by the three LU ops that need it, as a TEMPLATE ON
// A TAG rather than as a linked symbol. That is not style: getrf_blocked.cc sits
// in EXTENSIONS_CTA_SOURCES (its panel leaf IS the getrf CTA device function)
// while getrs_native.cc and getri_blocked.cc sit in
// EXTENSIONS_FACTORIZATION_SOURCES, and src/extensions/CMakeLists.txt:77-86's
// cluster rule says a source must sit with the sources whose device symbols it
// calls. A shared kernel would force all three into one cluster or produce
// `ptxas fatal: Unresolved extern function`. Templating the KERNEL NAME on a
// per-TU tag gives each cluster its own instantiation of the same source text,
// which is what geqrf/orgqr's wy:: helpers do with GeqrfWyTag.
//
// ===========================================================================
// THE SHAPE OF THE COST, MEASURED, AND WHY THIS IS THE FORM THAT SHIPPED.
//
// A LAPACK-faithful interchange walk is HALF of a composed solve. Measured by a
// deliberate break (BREAK=laswp, float n=128 nrhs=128 batch=256,
// experiments/wp6_lu/baseline/): getrs_trsm 0.4456 -> 0.2252 ms without the row
// exchange (49%), getri_trsm 0.4580 -> 0.2251 ms (51%).
//
// The cause is structural, not a bad kernel. ipiv is a SEQUENCE of transpositions
// that must be applied IN ORDER, so the only parallelism across the list is over
// the columns it is applied to -- and in COLUMN-MAJOR consecutive columns are
// `ld` apart, so a warp's 32 accesses land in 32 different cache lines. There is
// no mapping that fixes this while touching only the 2*len*ncols elements the
// interchanges actually move: a row is strided by construction.
//
// WHAT MAKES IT TOLERABLE ANYWAY, and it is why the walk is per-column rather
// than per-step: work-item (b, c) walks k = k0..k1 down ITS OWN column, and the
// "k side" of each exchange visits rows k0, k0+1, ... consecutively -- i.e. one
// 128 B line per column serves ~32 consecutive steps. Only the p side is
// scattered. A per-STEP kernel (one launch per k, all work-items on one row)
// would throw that reuse away AND pay len launches.
//
// WHAT IT COSTS THE SHIPPED getrf, MEASURED, AND IT IS THE SINGLE BIGGEST
// REMAINING LEVER IN WP6. Priced by disabling both of the blocked driver's
// interchange passes (a TIMING-ONLY break -- the answers are wrong by
// construction; experiments/wp6_lu/kernels/break.py's getrf_nolaswp_left /
// _right, laswp_cost.txt), vendor-free build, saturating batch:
//
//     type     n(batch)   with laswp   without   laswp share
//     float     512(512)    40.44 ms   10.53 ms      74%
//     float    2048(32)     70.86 ms   39.71 ms      44%
//     cdouble   512(512)   232.09 ms  181.97 ms      22%
//     cdouble  2048(32)    694.05 ms  645.02 ms       7%
//
// So for float this ONE kernel is most of the op, and the getrf A/B geomean of
// 0.886x is very largely its number rather than the factorisation's: without it
// the same cells run 4445 and 4652 GFLOP/s against cuBLAS's 918 and 354.
//
// THE MECHANISM, and it says which fixes can and cannot work. Per panel the
// interchange touches 2*ib rows across every column. The k SIDE is free: rows
// j0..j0+ib-1 are CONSECUTIVE, so one 128 B line per column serves all ib steps.
// The p SIDE is the whole cost: the ib selected rows are scattered over
// [j0, n), so each is its own line -- 4 B used of 128 B for float, a 32x
// inflation, ib lines per column per panel. That is ~2*n^2*(line/elem) of
// effective traffic over the factorisation, which is O(n^2) against the gemm's
// O(n^3/3) and therefore hurts MOST at moderate n, exactly as the table shows.
//
// TWO CANDIDATE FIXES, with their crossovers, neither implemented here:
//   (a) FULL-RANGE GATHER over [j0, n): collapse the panel's ib transpositions
//       into a row map and write dst[i] = src[perm[i]] for EVERY i, so
//       consecutive work-items land on consecutive rows and both sides are
//       coalesced. It moves 2*(n-j0) elements per column instead of 2*ib with a
//       32x line penalty, so it wins while (n - j0) < 16*ib -- i.e. below about
//       n = 512 at ib = 32, and it LOSES on the leading panels of a large
//       problem. It also needs an out-of-place buffer or an in-place cycle walk.
//   (b) A ROW-MAJOR staging of the trailing block, which is the only way to make
//       the p side contiguous at all. That is a different data layout, not a
//       different kernel.
// Neither is a WP6-local decision: (a) is shape-dependent and therefore belongs
// with the preferred() window, and (b) is a change to how the driver holds A.
//
// THE COST TO getrs AT nrhs = 1 IS SMALL, and that is the same measurement from
// the other side: 26% at float n=512, 11% at float n=2048, 2% and 1.4% at
// cdouble. It confirms the baseline's finding that at one right-hand side the
// permutation is a rounding error and the loss is entirely in the solves.
//
// THE nd_range IS BATCH-ONLY AT nrhs = 1, and that is stated rather than hidden:
// with one right-hand side the 2-D range degenerates to `batch` work-items, 32 of
// them at n=2048 b=32, each walking n dependent swaps. It is the recurring
// BatchLAS defect in miniature. It is not fixed here because the measurement
// above prices it at 1.4-11% of an op that ships route-neutral anyway, and the
// only fix is fix (a) -- whose workspace (an out-of-place RHS) is exactly what
// getrs_native.cc's strategy note declines to buy.
//
// THE COLLAPSED ALTERNATIVE IS REAL AND IS NOT WHAT SHIPPED HERE. Applying the
// interchanges to an identity index array once and then GATHERING puts
// consecutive work-items on consecutive ROWS, which is contiguous: measured, it
// turns the getri composition's geomean from 0.97x to 1.60x and getrs at
// nrhs=64 from 1.17x to 1.55x. getri takes exactly that route and needs NO
// permutation kernel at all (it writes P straight into C -- see
// getri_blocked.cc). getrs does NOT, because its gather needs an out-of-place
// RHS -- 67,371,008 B at n=2048, nrhs=64, batch=32, but only 262,144 B at the
// nrhs that decides; do not quote the nrhs=64 figure as the reason, because
// nrhs=64 is the case the gather WINS. At the nrhs where getrs is actually called
// (1, in linalg::solve) the permutation is a rounding error and the gather
// changes the geomean by nothing at all (0.36x either way). That trade is recorded in getrs_native.cc rather
// than made by accident here.
//
// ===========================================================================
// THE nd_range. A 2-D range of (batch, ncols) work-items -- NOT one work-item
// per batch item, which is this repository's recurring performance defect. SYCL
// linearises the LAST dimension fastest, so consecutive work-items take
// consecutive columns of ONE matrix, which is the best available locality for a
// strided access. At the smallest interesting cell (n=32, batch=8192, ncols=32)
// that is 262,144 work-items on 128 SMs; at the largest (n=2048, batch=32) it is
// 65,536. No barriers and no local memory: each work-item touches only its own
// column, so the ib transpositions of one panel are independent across columns.

#include "../sycl/device_scalar.hh"
// ../queue.hh, not just <batchlas/util/sycl-device-queue.hh>: the public header
// only FORWARD-DECLARES QueueImpl (sycl-device-queue.hh:219), so `ctx->submit`
// needs the definition. Including it here rather than relying on the includer's
// order is what keeps this header self-contained.
#include "../queue.hh"

#include <batchlas/util/sycl-device-queue.hh>

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::lu_native {

// The kernel name. Templated on the caller's TAG so each translation unit gets
// its own device symbol -- see the cluster note at the top of this file.
template <typename Tag, typename T> class LuLaswpKernel;

// Apply pivots k0 .. k1-1 (0-based positions in the interchange list) to the
// column block that `base` points at.
//
//   forward  : k = k0, k0+1, ..., k1-1     -- this is P B   (?LASWP incx = +1)
//   !forward : k = k1-1, ..., k0           -- this is P^T B (?LASWP incx = -1)
//
// The list values are GLOBAL 1-BASED row indices (LAPACK ipiv, and the packed
// int32 the CUDA/ROCm backends carry -- see getrf_native.hh's PIVOT CONTRACT),
// so `base` must point at row 0 of the matrix, offset only in the COLUMN
// direction. `piv_stride` is the caller's per-item pivot stride, which for every
// caller in this tree is the ORDER of A and not the length of the sub-list.
//
// A no-op (ncols <= 0, batch <= 0 or k1 <= k0) enqueues nothing and returns the
// queue's current event, so the callers do not each need the guard.
template <typename Tag, typename T>
Event lu_laswp_launch(Queue& ctx,
                      T* base, int ld, int stride, int ncols, int batch,
                      const int* piv, int piv_stride,
                      int k0, int k1, bool forward) {
    if (ncols <= 0 || batch <= 0 || k1 <= k0) return ctx.get_event();

    // std::complex is re-typed HERE, at the pointer boundary, and never crosses
    // into the kernel body -- device_scalar.hh's rule. It is a pure data move so
    // no arithmetic is involved, but the type must still be a POD aggregate for
    // the copy to compile without Annex-G machinery.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const bp = reinterpret_cast<D*>(base);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<LuLaswpKernel<Tag, T>>(
            sycl::range<2>(static_cast<std::size_t>(batch),
                           static_cast<std::size_t>(ncols)),
            [=](sycl::item<2> it) {
                const int b = static_cast<int>(it.get_id(0));
                const int c = static_cast<int>(it.get_id(1));

                D* const col = bp + static_cast<std::ptrdiff_t>(b) * stride +
                               static_cast<std::ptrdiff_t>(c) * ld;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                if (forward) {
                    for (int k = k0; k < k1; ++k) {
                        const int p = ip[k] - 1;      // 1-BASED on the wire
                        if (p != k) {
                            const D t = col[k];
                            col[k] = col[p];
                            col[p] = t;
                        }
                    }
                } else {
                    // REVERSE ORDER, and this is the classic silently-wrong
                    // answer in a transposed getrs: P^T = S_{k0} ... S_{k1-1}
                    // where P = S_{k1-1} ... S_{k0}, so the same list applied
                    // forwards computes P, not P^T. Every transposition is its
                    // own inverse, which is why only the ORDER changes.
                    for (int k = k1 - 1; k >= k0; --k) {
                        const int p = ip[k] - 1;
                        if (p != k) {
                            const D t = col[k];
                            col[k] = col[p];
                            col[p] = t;
                        }
                    }
                }
            });
    });
    return ctx.get_event();
}

}  // namespace batchlas::lu_native
