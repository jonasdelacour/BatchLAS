// Native batched GETRS -- the row interchange plus two ROUTED triangular solves.
//
// This is the small file. Both solves are the routed trsm, injected from the
// facade; the only kernel getrs owns is the interchange, and even that is the
// shared lu_laswp.hh template.
//
// getrs_native.cc sits in EXTENSIONS_FACTORIZATION_SOURCES and NOT in
// EXTENSIONS_CTA_SOURCES: it shares no device symbol with the getrf pair
// (src/extensions/CMakeLists.txt:77-89's cluster rule, orgqr_blocked.cc's
// precedent). lu_laswp.hh is a TEMPLATE ON A TAG precisely so that this stays
// true -- see its opening note.
//
// ===========================================================================
// THE MEASUREMENT THAT GOVERNS THIS FILE IS A NEGATIVE RESULT, AND IT IS NOT
// PAPERED OVER.
//
// Composed "row permutation + two routed trsm" against cublas?getrsBatched, at
// saturating batch, in process, against a host oracle
// (docs/perf/lu.md#the-vendor-baseline-and-saturation, summary.txt):
//
//   nrhs = 1  : GEOMEAN 0.36x over 28 cells (4 types x 7 orders). 25 LOSSES.
//               n(batch)   float double cfloat cdouble
//                32(8192)   0.20   0.19   0.10   0.09
//               128(4096)   0.41   0.23   0.34   0.14
//               512(512)    0.66   0.32   0.59   0.26
//              2048(32)     0.94   1.14   0.87   1.07
//               The only wins are at n=2048, against a vendor which is NOT
//               SATURATED there (64x the work for 1.52x the time, batch 1 to 64).
//   nrhs = 64 : geomean 1.17x with this interchange walk, 1.55x with the
//               permutation collapsed to a gather. 20 and 25 wins of 28.
//
// THE nrhs=1 LOSS IS STRUCTURAL: at one right-hand side the permutation is a
// rounding error and the whole loss is in the triangular solves, because trsm's
// blocked driver amortises a panel over many columns and one column gives it
// nothing to amortise. The permutation strategy is irrelevant to it -- 0.36x
// either way. So this arm SHIPS ROUTE-NEUTRAL (preferred() is false for every
// shape), which is a legitimate outcome under the campaign's gate: "a native
// kernel that is correct but slower than the vendor ships route-neutral, it does
// not become the default". It exists so that a vendor-free build HAS a getrs,
// and getrs has no internal consumer at all -- src/extensions/inv.cc:48-49 calls
// getrf then getri, and the public layer reaches getrs only through
// linalg::solve (linalg-ops.hh:343-344), at nrhs = 1.
//
// WP8-I2 UPDATE: THE GATHER SHIPPED, AND THE COST THIS NOTE BUDGETED FOR IT --
// AN OUT-OF-PLACE RHS PLUS AN int32[n] PER ITEM -- TURNED OUT NOT TO EXIST.
//
// The note below was right that the gather pays only at wide nrhs and right that
// a workspace bought for it would be charged to every narrow call by the
// facade's tier-max. It was wrong about ONE thing, and that one thing was the
// whole objection: the collapse does not need a workspace. The prototype it was
// costed from (docs/perf/lu.md#getrs-collapsed-permutation:163-180) built the row
// map in a GLOBAL int32[n] per item and gathered into a SEPARATE global buffer S
// -- and then never copied the answer back, so its 1.55x also omitted a full
// extra pass. Doing the permutation in LOCAL memory, one work-group per matrix
// item, removes both buffers AND the copy-back: the tile is read coalesced from
// B and written permuted back to B's own addresses. getrs_blocked_buffer_size
// therefore still returns 0 at every shape and every width, and
// LuTest.GetrsPermGatherBuysNoWorkspace is what keeps that true.
//
// WHAT IT IS WORTH, MEASURED PER CELL against the arm it replaces (the walk),
// interleaved rep by rep INSIDE ONE PROCESS via BATCHLAS_GETRS_LASWP, 11 reps,
// median, warm JIT, CUDA_VISIBLE_DEVICES pinned, zero foreign compute processes
// on every row, TWO independent passes with the WORSE quoted, the two arms'
// solutions asserted BIT-IDENTICAL on every row, and the resolved spelling read
// back per arm so a silent fallback cannot report a flat 1.00x
// (docs/perf/lu.md#getrs-collapsed-permutation, ab_summary.txt). One saturating batch per
// order; 4 types x 5 orders x 6 widths:
//
//   nrhs      cells  geomean     min      max
//      1        20   0.9993   0.9953   1.0031     <- the walk's own noise
//      2        19   0.9996   0.9963   1.0026
//      4        20   0.9997   0.9964   1.0027
//      8        20   1.0011   0.9980   1.0256     <- LAST width that buys nothing
//     16        20   1.1182   1.0004   1.2941     <- FIRST width that pays
//     24        20   1.1511   1.0108   1.3414
//     32        20   1.2148   1.0141   1.4482
//     64        20   1.4171   1.0295   2.2392
//    128        20   1.5728   1.0411   2.7873
//
//   (nrhs 2, 8 and 24 are a SEPARATE sweep, ab_bnd_p{1,2}.csv, run for exactly
//   one reason: the main grid samples 4 and then 16, so it BRACKETS the boundary
//   without measuring either rung the boundary separates. GATE-C says transcribe
//   a boundary from a CSV rather than infer it from an inequality, and that
//   applies to this constant as much as to a preferred() clause. One row of that
//   sweep was refused for relsd > 0.10 -- float n=64 nrhs=2 batch=8192 -- and
//   is named rather than dropped.)
//
//   ADMITTED SET (nrhs >= kGetrsPermGatherMinNrhs = 16): 80 cells, geomean
//   1.3191, MIN 1.0004, ZERO cells below 1.00.
//     float 1.5799 (1.118-2.787)   cfloat  1.4766 (1.119-2.071)
//     double 1.2061 (1.040-1.696)  cdouble 1.0761 (1.0004-1.261)
//   Cross-pass median spread 1.0017, worst 1.0250, 0 of 240 arm-medians above
//   1.10.
//
// THE BOUNDARY IS TRANSCRIBED, NOT INFERRED. Every one of the 50 cells that
// measured below 1.00 across both sweeps is at nrhs <= 8 -- i.e. exactly the
// region the boundary keeps on the walk -- and none is below 0.995. The rung
// below the boundary (8) is 1.0011 and the rung at it (16) is 1.1182. Widening the default to nrhs = 1
// would ship a 1.00x, which this campaign calls a revert; narrowing it to 32
// would give up float's 1.12-1.27 and cfloat's 1.12-1.29 at nrhs = 16 for
// nothing. cdouble is the marginal type at the boundary (1.0004 at n=512) and is
// recorded as such rather than carved out: a per-type boundary would add a
// decision surface with no measured payoff, since cdouble at nrhs = 32 is only
// 1.01-1.09 either way.
//
// WHAT THE GATHER DOES *NOT* FIX, and this is the pass's negative result. It
// makes the composition faster; it does not make the composition WIN. See the
// ladder note under preferred() in route_getrs.hh -- at SATURATION the
// composition's advantage over cuBLAS falls monotonically with batch, and the
// recorded "nrhs=128 geomean 1.478x, 24 wins of 28" was measured at ONE batch
// per order, above which no ladder existed anywhere in this tree.
//
// AN UNCLAIMED LEVER IN THIS VERY KERNEL, named rather than left for someone to
// rediscover: THE GATHER IS PARALLEL OVER BATCH ONLY. Its launch is
// nd_range<1>(batch * wg, wg) with wg = 256, i.e. exactly `batch` work-groups.
// At the shipped clause's own batch floor of 128 that is 128 groups on a 128-SM
// RTX 4090 -- ONE WAVE, no cross-group latency hiding, 32,768 work-items on a
// part that holds 196,608. The getrf gather in lu_laswp.hh does NOT have this:
// it is nblk*batch groups (896 at n=256, batch=128). This is the campaign's
// signature defect -- "4 kernels parallel over batch ONLY" -- and it is present
// here in the arm this pass shipped.
//
// It is an UNCLAIMED LEVER AND NOT A DEFECT, and the distinction is measured:
// the gather never loses to the walk it replaces (A/B minimum 1.0004 over 80
// admitted cells, zero cells below 1.00). So nothing regressed; there is simply
// headroom that a (column-block x batch) decomposition, mirroring lu_laswp.hh,
// would collect. Not attempted here because the ladder that justifies the
// clause was measured against THIS geometry, and changing the geometry would
// invalidate it. Related and also unprofiled: the header prices the serial SLM
// index walk at "~3% at n=1024 batch=128"; an independent estimate from
// dependent-swap latency puts it nearer 10%. Neither figure comes from a
// profile, and both should, before anyone spends effort on either.
//
// ---- THE ORIGINAL NOTE, KEPT BECAUSE ITS REASONING STILL HOLDS AT nrhs = 1 ----
//
// WHY THE INTERCHANGE WALK AND NOT THE GATHER, stated as a decision rather than
// made by accident. The gather is worth +0.38 of geomean at nrhs = 64 and
// NOTHING at nrhs = 1 (MEASURED at 0.9993 above, so this half was exactly
// right), and the prototype it was costed from paid an OUT-OF-PLACE RHS plus an
// int32[n] per item for the collapsed permutation.
//
// THE BUFFER FIGURE MUST BE READ AT THE nrhs THAT DECIDES, and the one this note
// used to quote was not. 67,371,008 B is the buffer at n=2048, nrhs=64,
// batch=32 -- but nrhs=64 is the case where the gather WINS, so it never carried
// the decision. At nrhs=1, the only nrhs the library actually issues, the same
// buffer is n*batch*sizeof(T) = 262,144 B, i.e. 257x smaller, and the argument
// against buying it is NOT the memory. It is that at nrhs=1 the gather buys
// nothing measurable: the loss there is in the two triangular solves (0.36x
// either way), not in the interchange.

#include "getrs_native.hh"
#include "lu_laswp.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrs {

namespace {

// The per-TU tag that gives this cluster its own instantiation of the shared
// LASWP kernel. See lu_laswp.hh.
struct GetrsLaswpTag {};

// The kernel name for the collapsed permutation below. Local to this TU, so
// unlike lu_laswp.hh's templates it needs no tag: nothing else instantiates it.
template <typename T> class GetrsPermGatherKernel;

// ===========================================================================
// THE COLLAPSED PERMUTATION, AND THE REASON IT NEEDS NO WORKSPACE AT ALL.
//
// WHAT IT REPLACES. lu_laswp_launch's per-column WALK: work-item (b, c) walks
// k = 0..n-1 down its own column of B, swapping col[k] with col[ipiv[k]-1]. Its
// cost is the p SIDE -- the pivot row of each step is scattered over [0, n), so
// every touch is its own DRAM sector, ~(32 + sizeof(T)) bytes of traffic per
// element actually moved. That is 9x for float, 5x for double and cfloat, 3x for
// cdouble. lu_laswp.hh's header states the mechanism and the sector unit is not
// modelled: WP8-I1 settled it with ncu counters (one 32 B sector per 4 B float
// element, load AND store).
//
// THE COLLAPSE. Apply the transposition list ONCE to an identity INDEX array --
// after which idxs[i] is the original row now sitting at position i -- and then
// move data with dst[i] = src[idxs[i]], which is contiguous in i. Both sides of
// the data move are then coalesced and the traffic is 2*sizeof(T) per element.
//
// WHY THIS ONE IS IN PLACE AND THE PROTOTYPE'S WAS NOT, which is the whole
// difference between this and the "out-of-place RHS plus an int32[n] per item"
// the file header (and the WP6 plan) budget for. The prototype
// (docs/perf/lu.md#getrs-collapsed-permutation:163-180) built the map in a GLOBAL
// int32[n] per item with a batch-only-parallel kernel, then gathered B into a
// separate global buffer S and solved there -- 2 buffers, and it never copied
// the answer back (it probed the residual on S, :512-548), so a real driver
// would have owed one more full pass on top.
//
// Staging the column in LOCAL memory removes both. One work-group per matrix
// item holds the index array AND a Cs-column tile in SLM; it reads a tile of B
// coalesced, barriers, and writes B[i] = tile[idxs[i]] back to the SAME
// addresses. The permutation happens inside local memory, so:
//   * no out-of-place RHS               -> getrs_blocked_buffer_size stays 0
//   * no global int32[n] per item       -> and no second kernel to build it
//   * no copy-back pass                 -> the answer is already in the caller's B
// The facade's max over EVERY SUPPORTED NATIVE TIER (factorization.cc:846-866)
// therefore cannot bill a narrow caller for a wide one, because there is nothing
// to bill: both tiers still query 0 at every shape. That removes the tier-max
// hazard rather than gating around it.
//
// THE SERIAL PHASE is n SLM int swaps by one work-item, paid ONCE PER ITEM
// rather than once per column -- which is why the map is built on the index
// array and not on the data. Walking the data in SLM would need the whole
// n x nrhs block resident and one work-item per column.
//
// THE DIRECTION. `forward` walks k = 0..n-1 and builds P; !forward walks
// k = n-1..0 and builds P^{-1} = P^T. That is the SAME correspondence
// lu_laswp_launch's two branches carry, and it is what the transposed getrs
// needs: the permutation moves to the OUTPUT and is applied in reverse.
//
// CAPACITY IS A FALLBACK, NEVER A THROW: if one column plus the two int arrays
// will not fit local memory this enqueues nothing and returns false, and the
// caller re-schedules the identical composition with the ordinary walk.
// RouteTable<Op::getrs,T> has no field to advertise a laswp capacity, and
// route_potrf.hh:442-454 records what a capacity the table cannot see costs.
// ===========================================================================

// The DATA tile's share of local memory. lu_laswp.hh:340-347's constant and its
// reason, repeated rather than shared because that one is sized for a getrf
// block-suffix and this is a whole right-hand side: the tile is a pure streaming
// staging buffer, so every byte beyond what keeps the loads in flight buys
// nothing and costs work-group occupancy.
constexpr std::size_t kGetrsPermTileCap = 24576;

// The 48 KB LAUNCH HOLE (getrf_cta.cc:109-146, lu_laswp.hh:328-338). A property
// of the static shared memory the compiler emits, which no source controls.
constexpr std::size_t kGetrsPermHoleLo = 47104;
constexpr std::size_t kGetrsPermHoleHi = 49664;
constexpr std::size_t kGetrsPermHolePadTo = 49920;

constexpr std::size_t getrs_perm_hole_padded(std::size_t bytes) {
    return (bytes > kGetrsPermHoleLo && bytes <= kGetrsPermHoleHi) ? kGetrsPermHolePadTo
                                                                  : bytes;
}

// THE CAPACITY, in ONE place. The launcher and the debug query below both call
// it, so "would the gather run" and "does the gather run" cannot drift -- which
// is the failure mode a second copy of this arithmetic would create, and the one
// route_potrf.hh:442-454 records.
template <typename T>
bool getrs_perm_gather_fits(int n, std::size_t slm_budget) {
    if (n <= 0) return false;
    const std::size_t int_bytes = 2u * static_cast<std::size_t>(n) * sizeof(int);
    if (slm_budget <= int_bytes) return false;
    const std::size_t col_bytes =
        static_cast<std::size_t>(n | 1) * sizeof(typename sycl_device::DevMap<T>::type);
    return (slm_budget - int_bytes) >= col_bytes;
}

template <typename T>
bool getrs_perm_gather_launch(Queue& ctx,
                              T* base, int ld, int stride, int nrhs, int batch,
                              const int* piv, int piv_stride, int n,
                              bool forward,
                              std::size_t slm_budget, int max_wg) {
    if (nrhs <= 0 || batch <= 0 || n <= 0) return true;

    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    // ODD leading dimension: the permuted read tile[col*ldt + idxs[row]] is
    // random in the row index, so an even ldt would put a whole column in one
    // bank (getrf_cta.cc:72-79).
    const int ldt = n | 1;
    const std::size_t int_bytes = 2u * static_cast<std::size_t>(n) * sizeof(int);
    if (!getrs_perm_gather_fits<T>(n, slm_budget)) return false;

    const std::size_t col_bytes = static_cast<std::size_t>(ldt) * sizeof(D);
    std::size_t data_budget = slm_budget - int_bytes;
    if (data_budget > kGetrsPermTileCap) data_budget = kGetrsPermTileCap;
    std::size_t cs = data_budget / col_bytes;
    if (cs == 0) {
        // The CAP, not the device, is what refused. Retry against the whole
        // budget: a one-column tile is still a valid tile.
        cs = (slm_budget - int_bytes) / col_bytes;
        if (cs == 0) return false;
    }
    if (cs > static_cast<std::size_t>(nrhs)) cs = static_cast<std::size_t>(nrhs);
    const int Cs = static_cast<int>(cs);

    std::size_t tile_elems = static_cast<std::size_t>(Cs) * static_cast<std::size_t>(ldt);
    const std::size_t raw = int_bytes + tile_elems * sizeof(D);
    const std::size_t padded = getrs_perm_hole_padded(raw);
    // The pad target is an ABSOLUTE 49920 B, so on a device whose whole local
    // memory is 48 KB it would ask for more than exists and turn a slow launch
    // into a failed one. Padding is a performance fix, so it defers to the
    // budget: above it, the unpadded tile launches and simply sits in the hole.
    if (padded > raw && padded <= slm_budget) {
        tile_elems = (padded - int_bytes + sizeof(D) - 1) / sizeof(D);
    }

    int wg = (max_wg < 256) ? max_wg : 256;
    if (wg < 32) wg = 32;

    D* const bp = reinterpret_cast<D*>(base);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> ints(
            sycl::range<1>(2u * static_cast<std::size_t>(n)), h);
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);

        h.parallel_for<GetrsPermGatherKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const int b = static_cast<int>(it.get_group(0));
                const int lid = static_cast<int>(it.get_local_id(0));

                int* const idxs = &ints[0];
                int* const ips = &ints[static_cast<std::size_t>(n)];

                D* const Bb = bp + static_cast<std::ptrdiff_t>(b) * stride;
                const int* const ip = piv + static_cast<std::ptrdiff_t>(b) * piv_stride;

                for (int i = lid; i < n; i += wg) {
                    int p = ip[i] - 1;          // GLOBAL 1-BASED on the wire
                    // ?GETRF's contract is p >= k, so p is in [i, n). Clamped
                    // anyway: an out-of-range value here would corrupt the index
                    // array for the WHOLE item, where the global walk would
                    // corrupt one column.
                    if (p < 0 || p >= n) p = i;
                    ips[i] = p;
                    idxs[i] = i;
                }
                sycl::group_barrier(grp);

                // THE ONLY SERIAL PHASE, and it is on the INT array. Paid once
                // per item, not once per column.
                if (lid == 0) {
                    if (forward) {
                        for (int k = 0; k < n; ++k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    } else {
                        // REVERSE ORDER. P = S_{n-1}...S_0, so the same list
                        // applied forwards computes P, not P^T. Every
                        // transposition is its own inverse, which is why only
                        // the ORDER changes -- lu_laswp_launch:232-247's note.
                        for (int k = n - 1; k >= 0; --k) {
                            const int p = ips[k];
                            if (p != k) { const int t = idxs[k]; idxs[k] = idxs[p]; idxs[p] = t; }
                        }
                    }
                }
                sycl::group_barrier(grp);

                for (int cb = 0; cb < nrhs; cb += Cs) {
                    const int cw = ((nrhs - cb) < Cs) ? (nrhs - cb) : Cs;

                    // Flat over (column, row) with the ROW fastest, so
                    // consecutive work-items take consecutive rows of one column
                    // -- the one contiguous direction in column-major.
                    int col = lid / n;
                    int row = lid - col * n;
                    while (col < cw) {
                        tile[static_cast<std::size_t>(col) * ldt + row] =
                            Bb[static_cast<std::ptrdiff_t>(cb + col) * ld + row];
                        row += wg;
                        while (row >= n) { row -= n; ++col; }
                    }
                    sycl::group_barrier(grp);

                    col = lid / n;
                    row = lid - col * n;
                    while (col < cw) {
                        Bb[static_cast<std::ptrdiff_t>(cb + col) * ld + row] =
                            tile[static_cast<std::size_t>(col) * ldt + idxs[row]];
                        row += wg;
                        while (row >= n) { row -= n; ++col; }
                    }
                    // The tile is re-read on the next chunk, so the write-back
                    // must complete before it is overwritten.
                    sycl::group_barrier(grp);
                }
            });
    });
    return true;
}

// THE SPELLING KNOB, and it is load-bearing twice over: it is the only way two
// DRIVER SPELLINGS -- not two routes, not two builds -- can be interleaved
// inside ONE process for GATE-B, and it is the only way a test can reach the
// walk once the gather is the default. getrf's BATCHLAS_GETRF_LASWP is the
// precedent. The PRESENCE test is a function-local static (one getenv per
// process); the VALUE is re-read per call only when the variable is set at all.
enum class PermSpelling { kDefault, kWalk, kGather };

// NOTHING IS LATCHED HERE, and that is a DELIBERATE DEVIATION from
// getrf_left_laswp_mode() (getrf_blocked.cc:164-172), which latches PRESENCE in
// a function-local static and re-reads only the value.
//
// WHY. That split exists to keep a getenv off a hot path, and it costs a test
// hazard the campaign has now recorded eleven times: once presence has latched
// FALSE, a later setenv is invisible and the test runs the DEFAULT arm and
// passes green. getrf pre-empted it by exporting the resolved mode. getrs pays
// nothing to avoid it outright: one getenv is ~100 ns against an op whose
// SMALLEST measured cell in this pass is 0.31 ms, i.e. 3e-4 of one call, and it
// is one call per getrs and not one per block step. Re-reading unconditionally
// makes getrs_perm_spelling_debug BELOW truthful at every point in a process,
// including after unsetenv, which is what lets a test assert the DEFAULT
// boundary and the two overrides in the same binary.
PermSpelling perm_spelling() {
    const char* const s = std::getenv("BATCHLAS_GETRS_LASWP");
    if (s == nullptr) return PermSpelling::kDefault;
    if (std::strcmp(s, "walk") == 0) return PermSpelling::kWalk;
    if (std::strcmp(s, "gather") == 0) return PermSpelling::kGather;
    return PermSpelling::kDefault;
}

// THE SPELLING DECISION, IN **ONE** PLACE -- and this is a defect that was
// CAUGHT BY ITS OWN BREAK rather than avoided by design. The driver and the
// test-only query below each carried their own copy of the nrhs comparison, one
// spelled `nrhs >= kGetrsPermGatherMinNrhs` and the other `nrhs <
// kGetrsPermGatherMinNrhs`. Break `boundary_inverted` flipped the driver's copy
// and the whole suite stayed GREEN: the query, which is the only thing any test
// can observe, kept the old sense. That is the campaign's recurring
// two-copies-of-one-decision defect, in a function written to prevent exactly
// that for the CAPACITY and then not applied to the BOUNDARY. There is now one
// copy and both callers use it.
bool getrs_perm_use_gather(PermSpelling sp, int nrhs) {
    if (sp == PermSpelling::kWalk) return false;
    if (sp == PermSpelling::kGather) return true;
    return nrhs >= kGetrsPermGatherMinNrhs;
}

}  // namespace

// ---------------------------------------------------------------------------
// WHICH PERMUTATION SPELLING THIS CALL WOULD RESOLVE, from the SAME function the
// driver calls, and the SAME capacity arithmetic. Exported for tests only.
//
// Two things a test cannot otherwise see, and both have been blind guards in
// this campaign:
//   * the ENV: a test that sets BATCHLAS_GETRS_LASWP after presence has latched
//     runs the default arm and passes (getrf_blocked.cc:278-283's eleventh);
//   * the CAPACITY: the gather FALLS BACK to the walk rather than throwing, so a
//     test that believes it is exercising the gather at an order the tile cannot
//     hold is measuring the walk. `linked is not reachable` in miniature.
// Returns 1 for the gather and 0 for the walk.
// ---------------------------------------------------------------------------
template <typename T>
int getrs_perm_spelling_debug(Queue& ctx, int n, int nrhs) {
    if (!getrs_perm_use_gather(perm_spelling(), nrhs)) return 0;
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) return 0;
    const std::size_t lm = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (lm > 4096) ? (lm - 4096) : 0;
    return getrs_perm_gather_fits<T>(n, budget) ? 1 : 0;
}

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. TRUE for all four types.
//
// It moves no vendor-present traffic: RouteTable<Op::getrs,T>::preferred() is
// false everywhere, so a vendor-present build keeps taking cublas?getrsBatched
// and this driver is reachable only through BATCHLAS_GETRS_ROUTE, through the
// direct entry point, or in a vendor-free build (route_resolve.hh:60-63).
//
// DEFINED HERE, beside the driver, for potrf_native.hh:81-92's reason: these are
// full explicit specialisations and link from wherever they sit, so co-locating
// them is what makes "the flag is true" and "the file is compiled" the same fact.
// ---------------------------------------------------------------------------
template <> bool getrs_blocked_available<float>()                { return true; }
template <> bool getrs_blocked_available<double>()               { return true; }
template <> bool getrs_blocked_available<std::complex<float>>()  { return true; }
template <> bool getrs_blocked_available<std::complex<double>>() { return true; }

// ---------------------------------------------------------------------------
// WORKSPACE. ZERO, in every mode, and that is a consequence of the strategy
// decision above rather than a coincidence: the interchange is applied IN PLACE
// to the caller's B, and the routed trsm takes no workspace at all (verified
// against functions/trsm.hh -- there is no trsm_buffer_size anywhere in this
// tree, which is also why no buffer-size twin is injected alongside the solve
// seam). getrs has no `info` argument, so unlike getrf there is not even a
// fallback status span to reserve.
//
// It still takes B and transA, because the SIGNATURE is the contract: a later
// gather strategy scales with nrhs and applies its permutation to a different
// buffer depending on transA, and a query that could not see either would be
// right only for the strategy that shipped.
//
// Zero is returned directly rather than through a BumpAllocator::measuring()
// replay because there is no layout to replay; 0 is trivially an alignment
// multiple, which is the property the facade's max(native, vendor) needs
// (mempool.hh:45-58). Any term added later must arrive through workspace_bytes.
//
// It dereferences nothing: A and B arrive with null data pointers from a
// measuring pass.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      const MatrixView<T, MatrixFormat::Dense>&,
                                      Transpose) {
    return 0;
}

// ---------------------------------------------------------------------------
// THE DRIVER.
//
// A = F^{-1} L U, where F is the interchange sequence applied FORWARDS (F v
// permutes v by ipiv, in order) -- that is LAPACK's convention: ?GETRF returns
// ipiv such that applying it to A yields L U.
//
//   NoTrans   : A x = b  <=>  L U x = F b
//               apply F to B, solve L (unit lower), solve U (non-unit upper).
//
//   Trans     : A^T = (F^{-1} L U)^T = U^T L^T F^{-T} = U^T L^T F, because a
//               permutation matrix is orthogonal, so F^{-T} = F.
//               A^T x = b  <=>  U^T L^T (F x) = b
//               solve U^T, solve L^T, then x = F^{-1} w -- and F^{-1} is the
//               SAME list walked BACKWARDS, since each transposition is its own
//               inverse and only the order reverses.
//
//   ConjTrans : identical, with H for T; F is real so it is unchanged.
//
// THE TWO SOLVES SWAP ORDER **AND** THE PERMUTATION MOVES TO THE OUTPUT, IN
// REVERSE. Getting either half wrong is a silently wrong answer that no NoTrans
// test can see -- and the scaffolding's break B8 measured that no test in this
// suite issues a Trans getrs at all today, so the guard for this has to be
// written with the kernel.
// ---------------------------------------------------------------------------
template <typename T>
Event getrs_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             GetrsSolveTrsm<T> solve_trsm) {
    static_cast<void>(workspace);   // this arm needs none; see the query above

    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    // Every gate RouteTable<Op::getrs,T>::supports() applies, re-applied because
    // this entry point is reachable WITHOUT the table -- and it must be, for
    // potrf_native.hh:126-141's reason: route_resolve.hh:165 falls through to
    // automatic() when a forced route is unsupported, so a pinned-route test that
    // is wrong about one gate silently measures cuBLAS and passes green.
    if (n < 1 || nrhs < 1 || batch < 1) {
        throw std::invalid_argument("getrs_blocked: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getrs_blocked: A must be square");
    }
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("getrs_blocked: B must have A.rows() rows");
    }
    if (A.batch_size() != B.batch_size()) {
        throw std::invalid_argument("getrs_blocked: A and B must agree on batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        throw std::invalid_argument("getrs_blocked: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrs_blocked: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // Enumerated, never get_property(MAX_SUB_GROUP_SIZE) >= 32 -- that
        // property returns sub_group_sizes()[0] (queue-impl.cc:325), so the weak
        // test refuses a {8,16,32} device and ACCEPTS a {64} one. The gate is
        // transcribed from supports() and must match it exactly even though this
        // file's own kernel carries no sub-group requirement: the routed trsm it
        // calls does.
        throw std::runtime_error(
            "getrs_blocked: device does not offer sub-group size 32");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrs_blocked: pivot span is shorter than n * batch");
    }
    if (!solve_trsm) {
        // AN ABSENT INJECTION THROWS rather than silently reaching for a native
        // trsm entry point: the two solves ARE this op, and picking trsm's arm
        // here would be WP3 step 16's defect re-created (trsm_native.hh:82-104,
        // fix at level3.cc:186-231). A direct caller injects
        // trsm<Backend::CUDA, T> itself, which is still a call no vendor getrs
        // can serve.
        throw std::invalid_argument(
            "getrs_blocked: the solve seam is empty. Inject the ROUTED batchlas::trsm "
            "(the facade does; a direct caller must too) -- this driver deliberately has "
            "no native fallback, so that the router, and not this file, chooses the trsm "
            "arm.");
    }

    // PACKED 1-BASED int32 -- the format cublas.cc:1476 and rocsolver.cc:227 both
    // read through pivots.as_span<int>(), and the one a native getrf writes. See
    // getrf_native.hh's PIVOT CONTRACT: the two ops have independent env
    // variables and independent preferred() windows, so every mixture of native
    // and vendor arms is reachable and they must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    // THE PERMUTATION SEAM. Both spellings compute the SAME permutation of the
    // same buffer, in place; which one runs is a performance decision and never
    // a correctness one, which is what makes
    // GetrsTest.PermutationSpellingsAgreeBitForBit a meaningful assertion.
    const std::size_t local_mem_all = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t slm_budget = (local_mem_all > 4096) ? (local_mem_all - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const bool want_gather = getrs_perm_use_gather(perm_spelling(), nrhs);

    auto apply_perm = [&](bool forward) -> Event {
        if (want_gather &&
            getrs_perm_gather_launch<T>(ctx, B.data_ptr(), B.ld(), B.stride(), nrhs,
                                        batch, piv_i32.data(), /*piv_stride=*/n, n,
                                        forward, slm_budget, max_wg)) {
            return ctx.get_event();
        }
        // FALLBACK, not a throw: the gather refuses when one column of B plus the
        // two int arrays will not fit local memory, and the walk is the identical
        // composition.
        return lu_native::lu_laswp_launch<GetrsLaswpTag, T>(
            ctx, B.data_ptr(), B.ld(), B.stride(), nrhs, batch,
            piv_i32.data(), /*piv_stride=*/n, /*k0=*/0, /*k1=*/n, forward);
    };

    if (transA == Transpose::NoTrans) {
        (void)apply_perm(/*forward=*/true);
        // In-order queues give the ordering for free; an out-of-order one does
        // not, and every dependent boundary in this schedule carries its guard.
        if (!ctx.in_order()) ctx.wait();

        // alpha comes THIRD in the public trsm (functions/trsm.hh:100-108); the
        // old spelling is a DELETED overload at :121-138 so a stale call cannot
        // silently compile into a wrong answer.
        (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower,
                         Transpose::NoTrans, Diag::Unit);
        if (!ctx.in_order()) ctx.wait();
        return solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper,
                          Transpose::NoTrans, Diag::NonUnit);
    }

    // Trans and ConjTrans. The transpose flag is passed THROUGH to both solves --
    // ConjTrans on a real type is Trans, which the trsm layer already handles.
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Upper, transA, Diag::NonUnit);
    if (!ctx.in_order()) ctx.wait();
    (void)solve_trsm(ctx, A, B, T(1), Side::Left, Uplo::Lower, transA, Diag::Unit);
    if (!ctx.in_order()) ctx.wait();

    // F^{-1}: the SAME list, REVERSED. This is the half of the transposed case
    // that a NoTrans test cannot see -- and under the collapsed spelling it is
    // the reversed walk over the INDEX array, not over the data.
    return apply_perm(/*forward=*/false);
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. Everything that
// needs a Backend arrives injected.
// ---------------------------------------------------------------------------
#define BATCHLAS_GETRS_INSTANTIATE(T)                                                      \
    template std::size_t getrs_blocked_buffer_size<T>(                                     \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose);                             \
    template Event getrs_blocked_dispatch<T>(                                              \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose,                              \
        Span<int64_t>, Span<std::byte>, GetrsSolveTrsm<T>);                                \
    template int getrs_perm_spelling_debug<T>(Queue&, int, int);

BATCHLAS_GETRS_INSTANTIATE(float)
BATCHLAS_GETRS_INSTANTIATE(double)
BATCHLAS_GETRS_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
