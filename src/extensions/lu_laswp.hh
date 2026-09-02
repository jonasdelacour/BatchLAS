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
// docs/perf/lu.md#the-vendor-baseline-and-saturation): getrs_trsm 0.4456 -> 0.2252 ms without the row
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
// construction; docs/perf/lu.md#the-laswp-gather's getrf_nolaswp_left /
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
// WP8-I1 UPDATE: HALF OF THAT COST IS NOW GONE, AND IT WAS NOT THE HALF THE
// LAUNCH-COUNT READING PREDICTED. The (S-left) half is deferred to one
// SLM-staged gather (lu_laswp_deferred_left_launch, below); the (S-right) half
// is untouched. Measured A/B against the schedule described above, interleaved
// inside one process, 11 reps, median, two passes, vendor-free, batch >= 128,
// 58 native:blocked cells: geomean 1.207x, minimum 1.018x, ZERO cells below
// 1.00, cross-pass median spread 1.0011 and worst 1.033. By type:
//     float 1.350x (1.263-1.456)   cfloat 1.305x (1.233-1.398)
//     double 1.138x (1.053-1.243)  cdouble 1.074x (1.018-1.110)
// The three cells that route native:cta rather than native:blocked measure
// 0.9995x, which is the anti-vacuity check: the change cannot reach them.
//
// AND THE RE-SCHEDULE ALONE IS NOT FREE, WHICH REFUTES THE MODEL THAT SAYS IT
// MUST BE. Deferring while KEEPING the walk moves byte-for-byte identical
// traffic -- the column-visit sums are the same arithmetic series read from
// either end -- so it was predicted at 1.00x and measured, over 11 cells, at
// geomean 1.055x with a spread from 0.707x to 1.315x: float n=512 batch 128 is
// 0.707x, float n=1024 batch 128 is 0.916x, float n=512 batch 1024 is 1.281x.
// The mechanism is the WORK-ITEM COUNT, not the traffic: (S-left) at step q runs
// batch*j0 work-items each walking ib = 32 steps, while the deferred form runs
// batch*ib work-items each walking n - j0 steps. Same product, 15x less
// parallelism at n = 512, and at batch 128 that is 4,096 work-items on 128 SMs.
// So the walk is NOT purely bandwidth-bound, the deferral is NOT a free
// re-schedule, and the gather is doing more than saving traffic -- it also puts
// the parallelism back (batch * nblk work-GROUPS of 256).
//
// TWO CANDIDATE FIXES, with their crossovers -- (a) is now IMPLEMENTED for the
// deferred left-hand pass and still open for the right-hand one:
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
// (b) is still a change to how the driver holds A and is not attempted.
//
// THE 32 B SECTOR IS THE UNIT, MEASURED, NOT THE 128 B LINE -- and that fixes
// every crossover below. ncu on the FIRST (S-right) launch at float n=512,
// ncols=480, ib=32, batch=8 (docs/perf/lu.md#the-laswp-gather, profiled
// through getrfab_v because ncu refuses the vendor-free binaries on this box):
//     element touches   480*8*32*2 = 245,760
//     ld sectors        249,600    = 1.016 per touch
//     st sectors        245,760    = 1.000 per touch
// One 32 B sector per 4 B element, not four. The same counters on the deferred
// gather, which performs the IDENTICAL composition over the whole factorisation:
//     grid 120 groups (batch 8 x nblk 15) x 256, ld 126,720 + st 122,880
//     = 249,600 sectors against 1,966,080 for the walk -- 7.9x fewer, and within
//     1.6% of the 245,760 that perfect coalescing predicts.
// So the walk's measured 303 GB/s is 3.3x above its SECTOR floor because of the
// kernel, not the traffic; the line model would have put it at 1,130 GB/s, above
// this device's ~1,008 GB/s DRAM peak, which is impossible on its face.
//
// WHAT WP8-I1 LEARNED ABOUT (a)'s CROSSOVER, and it is why (a) shipped for the
// LEFT pass unconditionally and NOT for the right one. The crossover is on
// (n - j0), the REMAINING ORDER AT A BLOCK STEP -- not on n, not on batch. It is
// the amortisation of ONE stage of R = n - j0 rows over L transpositions, so it
// is L/R that decides, and the two passes have completely different L:
//     (S-right)  L = ib = 32 against R = n - j0   -- pays only below
//                (n - j0) ~ 288 (float) / 160 (double, cfloat) / 96 (cdouble)
//     (S-left)   L = R = n - j0_{r+1} EXACTLY, once deferred -- 1:1 at every
//                order, so it pays everywhere and needs NO gate at all.
// Writing the gate on n instead of on (n - j0) inverts it. That is the
// "leg predicate as routing gate" defect in its getrf form, and it is the reason
// the deferred pass is the one that shipped: a lever with no gate cannot carry
// that defect.
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

// ===========================================================================
// THE DEFERRED LEFT-HAND INTERCHANGE, AS ONE SLM-STAGED GATHER.
//
// WHAT IT REPLACES. The blocked driver's (S-left) launch: at every block step q
// it applies panel q's ib transpositions to the ALREADY FINISHED columns
// [0, j0_q). Over a factorisation that is P-1 launches and sum_q q*nb
// column-visits.
//
// WHY IT MAY BE DEFERRED AT ALL -- the one-line composition identity, and it is
// the whole correctness argument. Column block r (columns [j0_r, j0_{r+1}))
// receives, under LAPACK's schedule, the transposition lists of panels
// r+1, r+2, ..., P-1 in that order, each in increasing k; concatenated that is
// exactly the list [j0_{r+1}, n) in INCREASING k. And NOTHING in the driver ever
// reads a column below j0 after that column's own panel step: the panel is at
// (j0, j0), L11 at (j0, ib, j0, ib), A12 at (j0, ib, j2, n2), L21 at
// (j2, m2, j0, ib) and A22 at (j2, m2, j2, n2) -- every one at column >= j0. So
// applying that same list, in that same order, ONCE at the end is bit-for-bit
// the same composition. The LAST block receives nothing.
//
// WHY DEFERRING ALONE BUYS NOTHING, and this is measured rather than argued:
// the total column-visit count is IDENTICAL (sum_r ib_r*(n - j0_{r+1}) equals
// sum_q j0_q*ib_q, the same arithmetic series read from the other end), so a
// re-schedule that keeps the per-column WALK moves exactly the same bytes. It is
// the control, not the change.
//
// WHAT DEFERRING MAKES POSSIBLE. The walk's cost is its p SIDE: the pivot row of
// each step is scattered over [k0, n), so every touch is its own DRAM sector --
// ~(32 + sizeof(T)) bytes of traffic per element actually moved. Staging the
// column in local memory and permuting there costs 2*sizeof(T) per element and
// nothing else, so the ratio is (32 + sizeof(T)) / sizeof(T): 9x for float, 5x
// for double and cfloat, 3x for cdouble. THAT staging only pays when the
// transposition list is LONG relative to the staged row extent -- one stage of
// R rows to serve L transpositions amortises as L/R. Under LAPACK's schedule
// (S-left) has L = ib = 32 against R = n - j0, so it does not pay above
// n - j0 ~ 288 (float). DEFERRED, block r has L = R = n - j0_{r+1} EXACTLY --
// the list runs to the end of the matrix -- so the amortisation is 1:1 at every
// order and the staging always pays. The deferral is what makes the kernel worth
// writing; the kernel is what makes the deferral worth doing.
//
// THE FORM. One work-group per (matrix, column block) -- so ONE launch replaces
// all P-1 of them -- and inside it:
//   (i)   stage the pivot sub-list and initialise an identity index array in SLM;
//   (ii)  ONE work-item walks the transpositions on that INT array (not on the
//         data), R serial SLM int swaps. That is the only serial phase and it is
//         paid once per (matrix, block) rather than once per column. After the
//         walk idxs[i] is the ORIGINAL row now sitting at position i, which is
//         precisely the gather map: A_new[i] = A_old[idxs[i]].
//   (iii) the whole group then streams the block's columns through a Cs x R local
//         tile -- coalesced in, permuted out, coalesced out.
//
// WHY THE INDEX ARRAY RATHER THAN WALKING THE DATA IN SLM. Walking the data
// needs one work-item per COLUMN and a full Cs x R tile resident, which caps the
// work-group at budget/(R*sizeof(T)) work-items -- 12 at float n=2048, i.e. 12
// threads per SM once the tile has taken the whole local-memory budget. The
// index array is 4*R bytes independent of the scalar width, so the tile can be
// small, the work-group full width, and the serial phase paid once instead of
// once per column tile.
//
// THE CAPACITY IS A FALLBACK, NEVER A THROW. If one column of the longest block
// will not fit local memory alongside the two int arrays this returns false and
// enqueues NOTHING, and the caller re-schedules the same composition with the
// ordinary walk. route_getrf.hh has no field to advertise a laswp capacity and
// route_potrf.hh:442-454 records what a capacity the table cannot see costs.
// ===========================================================================

template <typename Tag, typename T> class LuLaswpGatherKernel;

// The 48 KB LAUNCH HOLE. getrf_cta.cc:109-146's band and pad target, repeated
// here rather than shared because that file's copy is in an anonymous namespace
// in a .cc. Same box, same cause: the hole is a property of the STATIC shared
// memory the compiler emits, which no source controls, and WP5 walked into it
// while believing its kernel's shape could not. Padding costs at most 256 B and
// only inside the band.
constexpr std::size_t kLuLaswpHoleLo = 47104;
constexpr std::size_t kLuLaswpHoleHi = 49664;
constexpr std::size_t kLuLaswpHolePadTo = 49920;

constexpr std::size_t lu_laswp_hole_padded(std::size_t bytes) {
    return (bytes > kLuLaswpHoleLo && bytes <= kLuLaswpHoleHi) ? kLuLaswpHolePadTo : bytes;
}

// The DATA tile's share of local memory. NOT the whole budget: the tile is a
// pure streaming staging buffer, so every byte of it beyond what keeps the loads
// in flight buys nothing and costs work-group occupancy (97,280 B is ONE group
// per SM on this device). 24 KB leaves room for three.
constexpr std::size_t kLuLaswpTileCap = 24576;

// Apply, for every column block r of a blocked LU, the transposition suffix
// [j0_{r+1}, n) to that block's own columns -- the whole (S-left) schedule of a
// factorisation, in one launch, as a gather.
//
// `base` points at row 0 column 0 of item 0. `piv` is the FULL interchange list,
// GLOBAL 1-based, `piv_stride` per item. `nb` is the driver's block width; every
// per-block extent below is derived from it through the SAME std::min the driver
// uses, so the short final panel is carried identically (and, being the last
// block, receives nothing at all).
//
// Returns false having enqueued NOTHING when the staging tile does not fit.
template <typename Tag, typename T>
bool lu_laswp_deferred_left_launch(Queue& ctx,
                                   T* base, int ld, int stride, int batch,
                                   const int* piv, int piv_stride,
                                   int n, int nb,
                                   std::size_t slm_budget, int max_wg) {
    if (batch <= 0 || n <= 0 || nb <= 0) return true;

    const int P = (n + nb - 1) / nb;
    const int nblk = P - 1;                 // blocks that receive anything
    if (nblk <= 0) return true;             // a single panel defers nothing

    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    // r = 0 carries the longest suffix and sizes the allocation for every group.
    const int rmax = n - nb;
    const int ldt = rmax | 1;               // ODD: the permuted read is random in
                                            // the row index, so an even leading
                                            // dimension would put a whole column
                                            // in one bank (getrf_cta.cc:72-79).
    const std::size_t int_bytes =
        2u * static_cast<std::size_t>(rmax) * sizeof(int);
    if (slm_budget <= int_bytes) return false;

    const std::size_t col_bytes = static_cast<std::size_t>(ldt) * sizeof(D);
    std::size_t data_budget = slm_budget - int_bytes;
    if (data_budget > kLuLaswpTileCap) data_budget = kLuLaswpTileCap;
    std::size_t cs = data_budget / col_bytes;
    if (cs == 0) {
        // The CAP, not the device, is what refused. Retry against the whole
        // budget before giving up: a one-column tile is still a valid tile.
        cs = (slm_budget - int_bytes) / col_bytes;
        if (cs == 0) return false;
    }
    if (cs > static_cast<std::size_t>(nb)) cs = static_cast<std::size_t>(nb);
    const int Cs = static_cast<int>(cs);

    std::size_t tile_elems = static_cast<std::size_t>(Cs) * static_cast<std::size_t>(ldt);
    const std::size_t raw = int_bytes + tile_elems * sizeof(D);
    const std::size_t padded = lu_laswp_hole_padded(raw);
    if (padded > raw) {
        tile_elems = (padded - int_bytes + sizeof(D) - 1) / sizeof(D);
    }

    int wg = (max_wg < 256) ? max_wg : 256;
    if (wg < 32) wg = 32;

    D* const bp = reinterpret_cast<D*>(base);
    const int nb_k = nb;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> ints(
            sycl::range<1>(2u * static_cast<std::size_t>(rmax)), h);
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);

        h.parallel_for<LuLaswpGatherKernel<Tag, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(nblk) *
                                             static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const int gid = static_cast<int>(it.get_group(0));
                const int lid = static_cast<int>(it.get_local_id(0));
                const int r = gid / batch;
                const int b = gid - r * batch;

                const int c0 = r * nb_k;                  // first column of the block
                const int ib = (nb_k < n - c0) ? nb_k : (n - c0);   // THE std::min
                const int k0 = c0 + ib;                   // = j0_{r+1}
                const int R = n - k0;                     // rows AND transpositions
                if (R <= 0 || ib <= 0) return;

                int* const idxs = &ints[0];
                int* const ips = &ints[static_cast<std::size_t>(rmax)];

                D* const Ab = bp + static_cast<std::ptrdiff_t>(b) * stride;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                for (int i = lid; i < R; i += wg) {
                    int p = ip[k0 + i] - 1 - k0;
                    // The contract (getrf_cta_device.hh:300-306 always writes a
                    // row it searched, so p >= k) is p in [i, R). Clamped anyway:
                    // an out-of-range value here would corrupt the index array
                    // for the WHOLE block, where the global walk would corrupt
                    // one column.
                    if (p < 0 || p >= R) p = i;
                    ips[i] = p;
                    idxs[i] = i;
                }
                sycl::group_barrier(grp);

                // The ONLY serial phase, and it is on the INT array.
                if (lid == 0) {
                    for (int i = 0; i < R; ++i) {
                        const int p = ips[i];
                        if (p != i) {
                            const int t = idxs[i];
                            idxs[i] = idxs[p];
                            idxs[p] = t;
                        }
                    }
                }
                sycl::group_barrier(grp);

                for (int cb = 0; cb < ib; cb += Cs) {
                    const int cw = ((ib - cb) < Cs) ? (ib - cb) : Cs;

                    // Flat over (column, row) with the ROW fastest, so
                    // consecutive work-items take consecutive rows of one column
                    // -- the one contiguous direction in column-major. The
                    // running (col,row) update replaces a runtime division per
                    // element; the inner while runs at most ceil(wg/R) times.
                    int col = lid / R;
                    int row = lid - col * R;
                    while (col < cw) {
                        tile[static_cast<std::size_t>(col) * ldt + row] =
                            Ab[static_cast<std::ptrdiff_t>(c0 + cb + col) * ld + k0 + row];
                        row += wg;
                        while (row >= R) { row -= R; ++col; }
                    }
                    sycl::group_barrier(grp);

                    col = lid / R;
                    row = lid - col * R;
                    while (col < cw) {
                        Ab[static_cast<std::ptrdiff_t>(c0 + cb + col) * ld + k0 + row] =
                            tile[static_cast<std::size_t>(col) * ldt + idxs[row]];
                        row += wg;
                        while (row >= R) { row -= R; ++col; }
                    }
                    sycl::group_barrier(grp);
                }
            });
    });
    return true;
}

}  // namespace batchlas::lu_native
