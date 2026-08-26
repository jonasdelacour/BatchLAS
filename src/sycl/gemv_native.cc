// Native batched GEMV -- the kernel translation unit.
//
// Read src/sycl/gemv_native.hh first: it carries the layout premise (it is the
// TRANSPOSED direction that needs a reduction, not NoTrans), the zero-local-
// memory rule, the two silently-wrong contracts, and the reason this TU is
// compiled for native_cpu as well as for CUDA.
//
// This file follows src/sycl/trsm_native.cc and src/extensions/getrs_fused.cc:
// geometry constants live here beside the launchers, the capability flags are
// full explicit specialisations at the bottom of this same file, and one
// instantiation macro covers the four scalar types.

#include "gemv_native.hh"

#include "../queue.hh"
#include "device_scalar.hh"

#include <sycl/sycl.hpp>

#include <complex>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace batchlas::sycl_gemv {

using sycl_device::DevMap;
using sycl_device::dev_conj;
using sycl_device::dev_is_complex_v;
using sycl_device::fma_acc;

namespace {

// ---------------------------------------------------------------------------
// THE WORK-GROUP LADDER, AND THE DEFECT IT EXISTS TO PREVENT.
//
// THE BATCH-ONLY PARALLELISM DEFECT IS THIS CODEBASE'S SIGNATURE PERFORMANCE
// BUG (it has been found in at least four kernels here), and the obvious gemv
// decomposition walks straight into it. The first draft of body 1 used
//
//     nd_range<2>  global = {batch, ceil(m/wg)*wg}   local = {1, wg}
//
// and for m <= wg that grid is EXACTLY `batch` work-groups. At batch = 128 on a
// 128-SM RTX 4090 that is one work-group of at most 256 threads per SM -- about
// 16.7% occupancy -- and the ONLY parallel extent is the batch. Every shape the
// library actually issues a gemv on (ortho.cc, lanczos.cc) is small-m.
//
// THE FIX IS THE FLATTENING BELOW. One 1-D global range over
// (out_len * batch) work-items, with
//
//     b = gid / out_len       i = gid % out_len
//
// so ONE work-group can span several batch items and the work-group count no
// longer depends on the batch alone. Consecutive work-items still hold
// consecutive `i` WITHIN a matrix, so the coalescing the layout premise buys is
// preserved exactly; only the group boundaries move.
//
// The ladder itself is trsm_native.cc:266-274's, unchanged in shape: take the
// LARGEST candidate work-group that still leaves at least 4 work-groups per
// compute unit, and fall to the smallest if none does. Falling to 32 is the
// important half -- it is what keeps a small problem spread across the device
// instead of packed into a handful of fat groups.
//
// WORKED EXAMPLE, the one the design was required to prove. Body 1 at
// m = 64, batch = 128 on this box (CU = 128, so the target is 512 groups):
//
//     items = 64*128 = 8192
//     wg=256 -> 32 groups   (< 512)      wg=128 -> 64 groups   (< 512)
//     wg=64  -> 128 groups  (< 512)      wg=32  -> 256 groups  (< 512)
//
// none reaches 4*CU, so the ladder ends at wg = 32 and the launch is 256
// WORK-GROUPS -- twice the SM count, against the 128 groups of 64 threads the
// nd_range<2> draft would have produced, and with every group full.
// ---------------------------------------------------------------------------
inline int gemv_wg_ladder(int64_t work_units, int max_wg, int cu,
                          int units_per_wg_shift) {
    // units_per_wg_shift == 0 : one work-ITEM per unit of work (bodies 1, 2)
    // units_per_wg_shift == 5 : one 32-lane SUB-GROUP per unit (body 3)
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t per_wg = cand >> units_per_wg_shift;
        if (per_wg < 1) continue;
        const int64_t groups = (work_units + per_wg - 1) / per_wg;
        if (groups >= static_cast<int64_t>(4) * cu) break;
    }
    // If NO candidate was admissible the loop never assigned, and `wg` is still
    // the 32 it was initialised to -- which would be larger than the device
    // allows. Unreachable on both devices this was measured on (the CUDA GPU
    // reports MAX_WORK_GROUP_SIZE 1024 and the native_cpu device 2048), and
    // unreachable for the CTA body by construction, since a device that
    // ENUMERATES a sub-group size of 32 admits a work-group of 32. It is still
    // an invalid nd_range rather than a slow one, so it is clamped rather than
    // documented away.
    if (max_wg > 0 && wg > max_wg) wg = max_wg;
    return wg;
}

// A sub-group sum. HAND-ROLLED with shift_group_left rather than
// sycl::reduce_over_group, copied from getrs_fused.cc:347-366 for its two
// recorded reasons: a group reduction is what puts static shared into a kernel
// and reopens the 48 KB launch hole, and WP6 measured it 1.5-4.7x SLOWER than
// an explicit walk for double and complex<double>. After log2(32) shift-down
// steps LANE 0 holds the total; no other lane's value is used or valid.
template <typename SG, typename D>
inline D sg_sum(const SG& sg, D v) {
    if constexpr (dev_is_complex_v<D>) {
        auto re = v.re, im = v.im;
        for (int off = 16; off > 0; off >>= 1) {
            re += sycl::shift_group_left(sg, re, off);
            im += sycl::shift_group_left(sg, im, off);
        }
        return D{re, im};
    } else {
        for (int off = 16; off > 0; off >>= 1) v += sycl::shift_group_left(sg, v, off);
        return v;
    }
}

// The kernel names. THREE BODIES, NOT ONE WITH A RUNTIME MODE. NoTrans walks a
// row across `ld`-strided columns with no collective; Trans walks a contiguous
// column; the CTA arm additionally carries a shuffle ladder and a
// reqd_sub_group_size. One body would allocate registers for the union of three
// unrelated inner loops and would have to carry the sub-group attribute onto
// the two arms that must run where there is no sub-group of 32.
//
// CONJUGATION, by contrast, IS runtime: it is one branch on a value that is
// uniform across the entire launch, and making it a template parameter would
// buy a second set of instantiations for a sign flip.
template <typename T> class GemvDirectNKernel;
template <typename T> class GemvDirectTKernel;
template <typename T> class GemvCtaTKernel;
template <typename T, int W> class GemvSegNKernel;
template <typename T, int W> class GemvSegTKernel;

// ---------------------------------------------------------------------------
// THE SEGMENT WIDTH -- how many lanes body 4 puts on ONE output.
//
// Largest power of two W with W * out_len <= 32, so that W * out_len lanes of a
// single 32-lane sub-group cover the whole output vector of ONE batch item and
// the segmented fold below is a clean log2(W)-step shuffle at stride out_len.
//
//     out_len   1   2   4   8  10  16  17  24  32  64
//     W        32  16   8   4   2   2   1   1   1   1
//
// W == 1 means "no segmentation is available at this out_len", and that is the
// gate: body 4 is used only where W >= 2, i.e. out_len <= 16. Between 17 and 31
// the segmented body would be body 1 with extra shuffles for nothing.
//
// THIS FUNCTION IS THE DISPATCHER'S; THE KERNEL TAKES W AS A TEMPLATE
// PARAMETER, AND THAT IS NOT A STYLE CHOICE -- IT IS THE WHOLE EFFECT.
// The first version of body 4 carried W as a runtime `const int`, and it was
// MEASURED, at ~128 MB, float, NoTrans, native:direct vs cuBLAS:
//
//     out_len            1      2      4      8     12     16
//     body 1        235.1  335.4  517.3  730.5  692.9  827.2   GB/s
//     body 4, W runtime  906.5  707.9  576.1  607.5  373.8  461.1
//     body 4, W constexpr 934.9  921.4  913.2  903.0  624.7  861.1
//
// i.e. with a runtime W the segmented body FIXED out_len <= 4 and REGRESSED
// out_len >= 8 -- worse than the body it replaced. ncu says why it was not the
// memory system: at out_len = 16, float, the runtime-W version already had
// sectors-per-load 2.50 (ideal) and 8.27% occupancy against body 1's 3.00 and
// 4.12%, and still ran at 26% of DRAM where body 1 reached 69%. Better
// coalescing and better occupancy, slower kernel: the loop, not the traffic.
// With W a compile-time constant the trip count (red_len - jsub + W - 1)/W and
// the address stride W*lda are both known, the loop unrolls, and the same
// shapes run at 90-98% of the vendor. FIVE instantiations per scalar type is
// what that costs, and it is the reason for the switch in gemv_native_direct.
// ---------------------------------------------------------------------------
inline int gemv_seg_width(int out_len) {
    if (out_len <= 0) return 1;
    int w = 1;
    while (w * 2 * out_len <= 32) w *= 2;
    return w;
}

// ===========================================================================
// BODY 5's SEGMENT WIDTH -- how many OUTPUTS one sub-group serves under Trans,
// and the runtime gate that decides whether body 5 runs at all.
//
// ---- WHAT THE DEFECT ACTUALLY WAS ---------------------------------------
// route_gemv.hh recorded body 3's short-reduction collapse as "the shuffle
// ladder is a fixed cost per output (5 steps, doubled to 10 for a complex
// scalar)". THAT READING IS REFUTED BY THE COUNTERS. If the cost were the
// shuffle COUNT, `double` (5 shift_group_left on a 64-bit value = 10 hardware
// shuffles) and `complex<float>` (10 shift_group_left on a 32-bit value = 10
// hardware shuffles) would be hurt EQUALLY. Measured on body 3 at
// out_len = 2048, batch = 512, transA = Trans, DRAM-resident, GB/s:
//
//     red_len        32      64     128    2048(roof)
//     float       833.8   924.2   931.0    952.3
//     double      547.5   928.6   932.2    953.8
//     cfloat      921.2   925.4   932.7    954.0
//     cdouble     434.5   708.5   932.5    952.2
//
// double and cfloat are 1.68x apart at red_len = 32 at IDENTICAL bytes,
// IDENTICAL shape and IDENTICAL shuffle count. ncu on the same launches
// (experiments/wp8_gemv/ncu_precheck.csv) names the discriminator:
//
//     sm__pipe_fp64_cycles_active, % of peak     dram__throughput, % of peak
//     red_len      32      64     128              32      64     128
//     cdouble   85.55   86.11   84.69           40.61   66.09   95.08
//     double    85.01   82.74   58.33           50.61   95.78   95.39
//     cfloat     0.00    0.00    0.00           95.30   95.15   95.27
//     float      0.00    0.00    0.00           79.09   95.59   95.63
//
// THE FOLD IS FP64 WORK ON A 1/64-RATE GEFORCE PART. sg_sum runs log2(32) = 5
// add steps on ALL 32 lanes of every sub-group, i.e. 160 double-adds per output
// for `double` and 320 for `complex<double>` (the halves fold separately),
// against only red_len useful FMAs. At red_len = 32 that is five folds' worth
// of FP64 for one round of loads, and the FP64 pipe -- not DRAM, not occupancy
// (79-93% throughout), not coalescing (sectors/load 16.00 for cdouble, the
// ideal) -- is what runs out. The real types never touch that pipe at all.
//
// ---- THE FIX, AND WHY W IS A TEMPLATE PARAMETER --------------------------
// Put L = 32/W lanes on each output and serve W outputs per sub-group. The fold
// then costs L*log2(L) adds per output instead of 32*5: 160 -> 24 at L = 8,
// 8 at L = 4, 2 at L = 2. The loop's red_len FMAs per output are unchanged.
//
// W IS A TEMPLATE PARAMETER FOR BODY 4's RECORDED REASON, not for style. With
// W a runtime `const int` the NoTrans segmented body FIXED out_len <= 4 and
// REGRESSED out_len >= 8 below the body it replaced -- at BETTER
// sectors-per-load and BETTER occupancy -- because the trip count and the
// address stride stopped being compile-time constants and the loop stopped
// unrolling. See the table on gemv_seg_width above. Body 5 has the identical
// exposure: its trip count is (red_len - s + L - 1)/L and its stride is L.
//
// ---- THE SECTOR FLOOR IS WHAT BOUNDS W ----------------------------------
// Under Trans the sub-group reads W runs of L*sizeof(T) CONTIGUOUS bytes per
// round. A 32-byte sector is fully used only when L*sizeof(T) >= 32:
//
//     cdouble (16 B)  L >= 2  ->  W <= 16
//     double, cfloat  ( 8 B)  L >= 4  ->  W <=  8
//     float    ( 4 B)  L >= 8  ->  W <=  4
//
// It is also an ALIGNMENT argument against the extreme. A run starts at element
// (b*stride + j*ld + s0), so an ODD ld -- the suite uses ld = 79 -- misaligns
// every other column. A misaligned 32-B run (L = 2) costs 2 sectors instead of
// 1, +100%; a 64-B run (L = 4) costs 3 instead of 2, +50%; a 128-B run (L = 8)
// costs 5 instead of 4, +25%; body 3's 512-B runs cost +6%.
//
// AND THE MEASUREMENT ONLY HALF AGREES WITH IT, WHICH IS WHY THE W TABLE BELOW
// IS TRANSCRIBED AND NOT DERIVED. The floor predicts W <= 4 for float. Measured
// at out_len = 2048, batch = 512, Trans, body 5 against body 3:
//
//     red_len          1      2      4      8     16     24     32
//     float  W = 4  3.36x  3.34x  3.34x  3.40x  3.35x      -  1.069x
//     float  W = 8  5.89x  5.88x  5.69x  5.91x  5.16x  1.35x  1.064x
//
// W = 8 is L = 4, i.e. SIXTEEN-byte runs -- half a sector -- and it is 1.7x
// FASTER than the floor-respecting W = 4 across red_len <= 16. Below the warp
// width this kernel is not sector-bound at all: body 3 is idling 32 - red_len of
// its 32 lanes, and recovering them dominates a wasted half-sector. The floor
// re-asserts itself at the long end of the band, where the loop is long enough
// for traffic to matter again, and that is exactly where the W table turns to 4.
//
// The instantiated set is therefore W in {2, 4, 8}: 8 and 4 are what the table
// selects, and 2 exists so BATCHLAS_GEMV_SEGT=2 keeps meaning W = 2 rather than
// silently resolving to something else -- campaign trap 3, in miniature.
// ===========================================================================
inline constexpr int kGemvSegTransMaxW = 8;

// ===========================================================================
// THE TWO GATES, BOTH ON red_len() AND NEVER ON out_len(). Under Trans/ConjTrans
// red_len() == m == A.rows() and out_len() == n == A.cols(). The band is on the
// REDUCTION length: that is where body 3's per-output fixed cost stops being
// amortised. A predicate written on the other extent inverts the window, and
// route_gemv.hh:273-277 records that exact error being caught TWICE in WP7.
//
// BOTH TABLES ARE TRANSCRIBED FROM A CSV, CELL BY CELL, NOT INFERRED FROM AN
// INEQUALITY: experiments/wp8_gemv/wfine_p{1,2}.csv, body 5 against body 3
// interleaved rep by rep inside one process, 11 reps, median, two passes,
// red_len walked down to 1 at out_len 256 and 2048. The full table is in
// gemv_native.hh.
//
// ---- GATE 1: WHERE BODY 5 RUNS AT ALL -----------------------------------
// Body 3 reaches the ~950 GB/s DRAM roof at a DIFFERENT red_len for each scalar
// type, and above that point there is nothing to win and a whole unrolled loop
// to lose -- which is exactly where body 4's recorded history says a segmented
// body REGRESSES. Body 3's own GB/s at out_len = 2048, batch = 512, Trans:
//
//     red_len       1     8    16    24    32    48    64   128
//     float      59.3 261.5 485.5 655.9 832.6 912.9 922.7 931.0
//     cfloat    104.8 450.2 779.8 911.1 921.7 929.4 925.2 932.7
//     double     33.3 149.8 281.9 415.9 548.1 817.0 928.7 932.2
//     cdouble    26.6 118.8 224.2 329.2 434.3 456.2 707.9 932.5
//
// so the last red_len at which body 3 is still materially short of the roof is
// 32 for float, 16 for cfloat, 48 for double and 64 for cdouble. Those four
// numbers ARE the gate. Every cell admitted below is a measured win in both
// passes; every cell just above is measured 0.98-1.00x, i.e. a REVERT under
// GATE-B, and the gate is what keeps it on body 3.
//
// NOTE WHAT THE TWO COMPLEX TYPES DO NOT SHARE. cdouble's gate is 4x cfloat's,
// because their shortfalls have different causes: cdouble's is FP64 work in the
// fold (85% of the FP64 pipe at every red_len, see above) and cfloat's is
// nothing but idle lanes below the warp width. Same body, same shuffle count,
// different band -- which is why this is a per-type table and not one constant.
// ===========================================================================
template <typename T> inline constexpr int kGemvSegTransMaxRedLen = 0;
template <> inline constexpr int kGemvSegTransMaxRedLen<float> = 32;
template <> inline constexpr int kGemvSegTransMaxRedLen<std::complex<float>> = 16;
template <> inline constexpr int kGemvSegTransMaxRedLen<double> = 48;
template <> inline constexpr int kGemvSegTransMaxRedLen<std::complex<double>> = 64;

// ---- GATE 2: WHICH W ------------------------------------------------------
// W = 8 (four lanes per output) at the short end, W = 4 (eight lanes) at the
// long end of the admitted band, with the switch point per type. Transcribed
// from the same CSV by taking, per type, the W with the best WORST cell over
// both out_len levels -- not the best mean, because a rule fitted to the mean
// admits the cell that loses.
//
// THE SECTOR-FLOOR ARGUMENT IS REFUTED BELOW red_len ~ 32, AND THAT IS A
// MEASURED CORRECTION. The plan predicted W would be bounded by "L*sizeof(T)
// must reach a 32-byte sector", giving W <= 4 for float. Measured: float at
// out_len = 2048, red_len <= 16 runs 5.16x-5.91x at W = 8 (L = 4, i.e. 16-byte
// runs, HALF a sector) against 3.34x-3.40x at W = 4. Below the warp width the
// kernel is not sector-bound at all -- body 3 is leaving 32-L of 32 lanes idle,
// and recovering them is worth more than the wasted half-sector. The floor does
// re-assert itself at the long end, which is why the table turns to W = 4 there.
template <typename T> inline constexpr int kGemvSegTransW8MaxRedLen = 0;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<float> = 24;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<std::complex<float>> = 16;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<double> = 32;
template <> inline constexpr int kGemvSegTransW8MaxRedLen<std::complex<double>> = 32;

// ---- GATE 3: THE LAUNCH MUST BE BIG ENOUGH TO GIVE ONE AWAY ---------------
// BODY 5 LAUNCHES W TIMES FEWER SUB-GROUPS THAN BODY 3 -- (out_len*batch)/W
// against out_len*batch -- so a shape whose whole launch is already small is
// paying parallelism it cannot spare for a fold it barely runs.
//
// THIS GATE EXISTS BECAUSE MY OWN GRID COULD NOT SEE THE REGIME IT GUARDS, and
// that is campaign trap 8 committed and then caught. The (out_len, red_len)
// plane in experiments/wp8_gemv/plane_cells.txt starts at out_len = 64 and
// reported 83 admitted cells with ZERO below 1.00x. Re-running the WP7 audit's
// parity grid -- which reaches out_len = 1 -- then showed the native arm 3-6%
// SLOWER at out_len = 1 than it had been before the change. Walking the output
// axis down (experiments/wp8_gemv/skinny_p{1,2}.csv, out_len 1..64) found 16
// losing cells, worst 0.891x, every one of them at out_len*batch <= 4096:
//
//     out_len*batch   128    512   1024   2048   4096   >= 8192
//     losing cells      4      7      2      2      1         0
//     worst          0.891  0.957  0.978  0.983  0.998      n/a
//
// THE FLOOR IS A TABLE ON W, NOT ONE NUMBER, and the second row was bought with
// two more passes. A first cut at 8*MAX_COMPUTE_UNITS removed eleven of the
// sixteen; a re-run (skinny2_p{1,2}.csv) left five cells at 0.976x-0.998x, and a
// THIRD AND FOURTH PASS AT 31 REPS (resid_p{3,4}.csv) separated them: `double`
// recovered (0.986/1.002 -- noise) but THREE FLOAT CELLS REPRODUCED BELOW 1.00
// IN BOTH PASSES -- out_len 4 at batch 512 (0.976/0.986), out_len 16 at batch
// 128 (0.985/0.983), out_len 1 at batch 4096 (0.998/0.990). All three are float
// at red_len = 32, i.e. in the W = 4 band.
//
// THAT BAND NEEDS A BIGGER LAUNCH, and the reason is not mysterious: W = 4 is
// the LONG end of the admitted reduction, where body 3 is already near its roof
// and the margin body 5 is playing for is thin -- 1.07x-1.20x rather than the
// 3x-10x of the short end. A thin margin does not survive an 4x cut in the
// launch unless the launch was large to begin with. Transcribed:
//
//   W = 8 band, floor 16*CU (2048 here):
//     excluded, items 1024: double out_len 2 (0.986/1.002), cdouble out_len 2
//                           (1.046/1.049 -- a small win given up)
//     admitted, items 2048: cdouble 1.222/1.225, 1.228/1.220; double 1.043,
//                           1.031/1.056     -- every admitted cell a win
//   W = 4 band, floor 64*CU (8192 here):
//     excluded, items 2048-4096: float 0.976/0.986, 0.985/0.983, 0.998/0.990,
//                                1.003/1.005; cdouble 1.211, 1.361 (given up)
//     admitted, items >= 8192:   float 1.080..1.638; cdouble 1.523..1.875
//
// It is a PARALLELISM condition read off the device -- not a footprint condition
// and not a cache size. The quantity it bounds is the launch, which is exactly
// what body 5 trades away. (An L2-residency term would be the forbidden kind;
// route_gemv.hh:279-284 says why. This is not that.)
//
// EVERY CELL MEASURED ANYWHERE IN THIS PASS IS NOW AT OR ABOVE 1.00x INSIDE THE
// GATE. The cost is seven small wins (1.02x-1.36x) on shapes whose whole launch
// is a few thousand outputs; the alternative was shipping a reproduced 0.976x,
// which GATE-E forbids.
inline int gemv_seg_trans_min_items(int cu, int w) {
    const int c = (cu > 0) ? cu : 1;
    return (w >= 8 ? 16 : 64) * c;
}

// THE SPELLING KNOB. Body 5 and body 3 are two DRIVER SPELLINGS of one route,
// not two routes and not two builds, so BATCHLAS_GEMV_ROUTE cannot separate
// them; this is the only way to interleave them inside ONE process for GATE-B,
// and the only way a test can reach body 3 at a shape where body 5 is the
// default. src/extensions/getrs_native.cc:383-427's BATCHLAS_GETRS_LASWP is the
// precedent, including the deviation from getrf's latching form:
//
// NOTHING IS LATCHED. getrf_left_laswp_mode() latches PRESENCE in a
// function-local static to keep a getenv off a hot path, and that has been a
// blind guard eleven times in this campaign -- once presence has latched FALSE
// a later setenv is invisible, the test runs the DEFAULT arm and passes GREEN.
// One getenv is ~100 ns against a gemv whose smallest cell in this pass is
// 0.011 ms, and it is one getenv per LAUNCH, not one per work-item. Re-reading
// unconditionally is what makes gemv_seg_trans_width_debug truthful at every
// point in a process, including after unsetenv.
//
//   BATCHLAS_GEMV_SEGT = off   force body 3 (the pre-WP8 arm)
//                      = auto  the shipped decision (the default)
//                      = 2|4|8 force body 5 at that W, gate and all
enum class SegTMode { kAuto, kOff, kForce2, kForce4, kForce8 };

inline SegTMode gemv_segt_mode() {
    const char* const s = std::getenv("BATCHLAS_GEMV_SEGT");
    if (s == nullptr) return SegTMode::kAuto;
    if (std::strcmp(s, "off") == 0) return SegTMode::kOff;
    if (std::strcmp(s, "2") == 0) return SegTMode::kForce2;
    if (std::strcmp(s, "4") == 0) return SegTMode::kForce4;
    if (std::strcmp(s, "8") == 0) return SegTMode::kForce8;
    return SegTMode::kAuto;
}

// WHICH SCALAR TYPES EMIT BODY 5. ALL FOUR, and that is a measured correction
// to the plan, which budgeted for {double, complex<double>} on the strength of a
// grid whose minimum red_len was 64. Walking red_len down to 1 -- which nothing
// in WP7 had done; audit/parity.sh starts at 64 and the ncu ladder started at 32
// and ran cdouble only -- shows every type collapsing below the warp width,
// because body 3 puts 32 lanes on the reduction whatever its length. At
// out_len = 2048, batch = 512, red_len = 8, body 3 runs at 261 GB/s for float
// and 450 for cfloat against a 950 roof, and body 5 is worth 5.91x and 3.98x.
// "A GRID THAT CANNOT REACH A REGIME IS NOT EVIDENCE ABOUT IT" applied to the
// grid that was going to decide the instantiation set.
template <typename T>
inline constexpr bool kGemvSegTransEmit =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

// THE WIDTH DECISION, IN **ONE** PLACE. The driver and the test-only query
// below both call this; there is no second copy of the comparison. That is the
// campaign's recurring two-copies-of-one-decision defect, which
// getrs_native.cc:410-427 records being caught by its own break only because
// the second copy existed.
//
// Returns 1 for "body 3" and W >= 2 for "body 5 at that W".
template <typename T>
inline int gemv_seg_trans_width(SegTMode mode, int red_len, int64_t items, int cu) {
    if constexpr (!kGemvSegTransEmit<T>) {
        static_cast<void>(mode);
        static_cast<void>(red_len);
        static_cast<void>(items);
        static_cast<void>(cu);
        return 1;
    } else {
        // THE FORCED SPELLINGS BYPASS ALL THREE GATES ON PURPOSE. That is what
        // lets a measurement ask "what WOULD body 5 do here", which is the only
        // way to show a gate is load-bearing rather than decorative, and what
        // lets a test reach body 5 at a shape the gate declines.
        switch (mode) {
            case SegTMode::kOff:    return 1;
            case SegTMode::kForce2: return 2;
            case SegTMode::kForce4: return 4;
            case SegTMode::kForce8: return 8;
            case SegTMode::kAuto:   break;
        }
        static_assert(kGemvSegTransW8MaxRedLen<T> <= kGemvSegTransMaxRedLen<T>,
                      "the W = 8 band must lie inside the admitted band");
        static_assert(kGemvSegTransMaxW >= 8, "the W table selects 8 and 4");
        if (red_len <= 0) return 1;
        if (red_len > kGemvSegTransMaxRedLen<T>) return 1;
        const int w = (red_len <= kGemvSegTransW8MaxRedLen<T>) ? 8 : 4;
        if (items < gemv_seg_trans_min_items(cu, w)) return 1;
        return w;
    }
}

// ---------------------------------------------------------------------------
// THE QUICK RETURN -- reference BLAS ?GEMV, transcribed rather than paraphrased.
//
//     IF ((M.EQ.0) .OR. (N.EQ.0) .OR.
//    +    ((ALPHA.EQ.ZERO).AND.(BETA.EQ.ONE))) RETURN
//
// y is left COMPLETELY UNTOUCHED in all three cases -- in particular y is NOT
// scaled by beta when the reduction length is zero, which is the case a
// hand-written kernel gets wrong by writing the beta term unconditionally. Both
// vendors agree with the reference here, so a native path that differed would
// return a ROUTE-DEPENDENT wrong answer: correct under cuBLAS, wrong under the
// native kernel, on the same call.
// ---------------------------------------------------------------------------
template <typename T>
inline bool gemv_quick_return(int m, int n, T alpha, T beta) {
    return m == 0 || n == 0 || (alpha == T(0) && beta == T(1));
}

// ===========================================================================
// BODY 1 -- {Native, Direct}, transA == NoTrans.
//
//   y_i = alpha * sum_j A[i + j*ld] * x[j*xinc] + beta * y[i*yinc]
//
// ONE WORK-ITEM PER OUTPUT ROW, and no collective anywhere. At reduction step j
// work-items i and i+1 read A[i + j*ld] and A[i+1 + j*ld]: adjacent, so a
// 32-lane group covers 32 consecutive elements of one column and the access is
// fully coalesced. x[j] is broadcast -- every work-item in the launch reads the
// same element at the same step -- and y is written once, coalesced.
//
// *** THAT COALESCING PREMISE HOLDS ONLY FOR out_len >= 32, AND THE ORIGINAL
// *** VERSION OF THIS COMMENT STATED IT UNCONDITIONALLY. It is wrong below the
// warp width, in two ways that both stop exactly at 32 lanes, both measured
// with ncu (experiments/wp7_gemv/audit/mechanism.csv):
//
//   * COALESCING. With the flattening b = gid/out_len, i = gid%out_len a warp
//     of 32 work-items straddles 32/out_len BATCH ITEMS, whose rows are
//     stride_a apart. Sectors per global load is 32/out_len below the warp
//     width and floats at 8.5 above it: out_len 1, 2, 4, 8, 16, 24, 31, 32, 48,
//     64 measured 32.0, 16.0, 12.0, 10.0, 9.0, 9.0, 9.5, 8.5, 8.67, 8.5. The
//     transition is in LANES, not bytes -- float turns at the same out_len = 32
//     despite a 4x narrower scalar -- and it is not an alignment artefact:
//     padding ld moves nothing (at out_len 16, ld 16/17/24 gives 9.00/9.50/9.00).
//
//   * PARALLELISM. items == out_len * batch is the ONLY extent this body has.
//     At out_len = 1, batch = 512 that is 512 work-items, i.e. 16 work-groups
//     of 32 on a 128-SM box -- 2.08% achieved occupancy, 7.03% of DRAM. The
//     flattening cannot manufacture parallelism that the decomposition does not
//     contain; it can only stop the batch from being the sole extent.
//
// Together those cost 0.08x-0.38x of cuBLAS. BODY 4 BELOW EXISTS TO FIX BOTH,
// and gemv_native_direct routes out_len <= 16 to it wherever the device
// enumerates a sub-group size of 32. This body still serves every out_len on a
// device without one -- notably the native_cpu queue, where neither effect
// exists because there is no warp.
//
// ZERO BYTES OF LOCAL MEMORY: no local_accessor is created in this submit.
// ===========================================================================
template <typename T>
Event gemv_direct_notrans(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const VectorView<T>& X,
                          const VectorView<T>& Y,
                          T alpha, T beta) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(m) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__ ON ANY OF THESE. ortho.cc:227-232 passes views into
        // the same allocation; they are element-disjoint but alias at the
        // object level, and __restrict__ is a promise about the object.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        // Launch-uniform, so these are two predictable branches outside the
        // loop, never a per-element test. `alpha_zero` is what keeps A unread
        // when alpha == 0, matching the reference; `beta_zero` is what keeps y
        // unread when beta == 0, which matters because y may be uninitialised
        // (reference ?GEMV never reads y in that case either).
        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));

        const int out_len = m;
        const int red_len = n;
        const int64_t total = items;

        h.parallel_for<GemvDirectNKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                // b = gid / out_len, i = gid % out_len. Consecutive work-items
                // therefore hold consecutive ROWS of one matrix -- see the
                // flattening note above.
                const int64_t b = gid / out_len;
                const int i = static_cast<int>(gid - b * out_len);

                const D* Ab = a_ptr + b * stride_a;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int j = 0; j < red_len; ++j) {
                        fma_acc(acc, Ab[i + static_cast<int64_t>(j) * lda],
                                xb[static_cast<int64_t>(j) * xinc]);
                    }
                }

                // out = alpha*acc + beta*y0, built with fma_acc from an exact
                // zero rather than with a separate multiply-and-add: fma into a
                // zero addend is the product, rounded once.
                D out{};
                fma_acc(out, alpha_d, acc);
                if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);
                yb[static_cast<int64_t>(i) * yinc] = out;
            });
    });

    return ctx.get_event();
}

// ===========================================================================
// BODY 4 -- {Native, Direct}, transA == NoTrans, SHORT OUTPUT.
//
//   y_i = alpha * sum_j A[i + j*ld] * x[j*xinc] + beta * y[i*yinc]   (as body 1)
//
// SAME ROUTE, SAME ANSWER, DIFFERENT DECOMPOSITION. This is not a fourth route
// and it is not selectable from the environment: {Native, Direct} on a NoTrans
// shape already names two kernels the way it already names body 1 for NoTrans
// and body 2 for Trans, and the choice between this body and body 1 is a
// property of the DEVICE and the shape, not of the routing vocabulary. That
// keeps the speed cutoff out of supports(), where route_gemm.hh:25-28's rule
// forbids it, and keeps the Direct route free of the GPU gate that would
// forfeit the WP7 burn-down.
//
// W LANES PER OUTPUT, ONE SUB-GROUP PER BATCH ITEM. With W = gemv_seg_width(m),
// lane l of a 32-lane sub-group takes
//
//     i = l % m        (which output)        jsub = l / m   (which slice of j)
//
// and walks j = jsub, jsub+W, jsub+2W, .... Lanes l and l+m therefore hold
// partial sums of the SAME output and are folded by a log2(W)-step shuffle at
// stride m.
//
// WHY THIS FIXES BOTH OF BODY 1's SHORT-OUTPUT DEFECTS.
//
//   * COALESCING. At step k the sub-group reads A[i + (jsub + kW)*ld] over
//     l = i + m*jsub, i.e. W runs of m CONTIGUOUS elements, ld apart. When
//     ld == m -- the ordinary packed batched layout -- those W runs are
//     themselves contiguous and the warp reads 32 consecutive elements, which
//     is exactly what body 1 achieves only for out_len >= 32.
//
//   * PARALLELISM. The extent becomes 32 * batch work-items instead of
//     out_len * batch, i.e. 32/out_len times more. At out_len = 1, batch = 512
//     that is 16384 work-items and 512 work-groups on a 128-SM box, against the
//     16 work-groups body 1 launches there.
//
// THE FOLD IS CLOSED, WHICH IS WHY THE INACTIVE LANES ARE HARMLESS. Lane i's
// total draws only from lanes i + m*t for t in [0, W), and every shuffle it
// depends on is some lane p reading p + off with p + off also of that form,
// hence < m*W <= 32. Lanes at or above m*W may read past lane 31 -- an
// unspecified value, exactly as the last lanes of sg_sum's ladder already do in
// body 3 -- but nothing they produce is ever read by a lane below m.
//
// THAT CLOSURE IS MEASURED, NOT ONLY ARGUED. Break `segactive` drops the
// `jsub < W` half of the accumulate guard, so the lanes at or above m*W walk
// the reduction too and carry a nonzero partial into the fold. It turns NOTHING
// red -- 232 of 232 pass. So `jsub < W` below is a WORK SAVING, not a
// correctness condition, and the lanes it silences were already unread.
//
// W IS A TEMPLATE PARAMETER, NOT A RUNTIME VALUE. See the measured table on
// gemv_seg_width: with a runtime W this body is SLOWER than body 1 for
// out_len >= 8, at better sectors-per-load and better occupancy, because the
// inner loop's trip count and address stride both stop being compile-time
// constants and it stops unrolling.
//
// ZERO BYTES OF LOCAL MEMORY: the only collective is a sub-group shuffle.
// ===========================================================================
template <typename T, int W>
Event gemv_seg_notrans(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const VectorView<T>& X,
                       const VectorView<T>& Y,
                       T alpha, T beta) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int kSg = 32;

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    // ONE SUB-GROUP PER BATCH ITEM -- a whole output vector fits in a sub-group
    // by construction, so the unit of work here is the batch item, not the
    // output element. Shift 5 for the same reason as body 3.
    const int wg = gemv_wg_ladder(batch, max_wg, cu, /*units_per_wg_shift=*/5);
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (static_cast<int64_t>(batch) + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__, for ortho.cc:227-232's reason. See body 1.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));

        const int seg = m;                    // out_len
        constexpr int wlanes = W;
        const int red_len = n;
        const int64_t nbatch = batch;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvSegNKernel<T, W>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t b =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());

                // SUB-GROUP UNIFORM, as body 3's is and for the same reason:
                // the shuffle ladder below must never be reached by a partial
                // sub-group.
                if (b >= nbatch) return;

                const int i = lane % seg;
                const int jsub = lane / seg;

                const D* Ab = a_ptr + b * stride_a;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero && jsub < wlanes) {
                    for (int j = jsub; j < red_len; j += wlanes) {
                        fma_acc(acc, Ab[i + static_cast<int64_t>(j) * lda],
                                xb[static_cast<int64_t>(j) * xinc]);
                    }
                }

                // THE SEGMENTED FOLD: stride `seg`, log2(W) steps. Descending
                // offsets, so that at every step the lanes a surviving lane
                // depends on are still inside the sub-group.
                if constexpr (dev_is_complex_v<D>) {
                    auto re = acc.re, im = acc.im;
                    for (int w = wlanes >> 1; w >= 1; w >>= 1) {
                        const int off = seg * w;
                        re += sycl::shift_group_left(sg, re, off);
                        im += sycl::shift_group_left(sg, im, off);
                    }
                    acc = D{re, im};
                } else {
                    for (int w = wlanes >> 1; w >= 1; w >>= 1) {
                        acc += sycl::shift_group_left(sg, acc, seg * w);
                    }
                }

                if (lane < seg) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(i) * yinc]);
                    yb[static_cast<int64_t>(i) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

// ===========================================================================
// BODY 2 -- {Native, Direct}, transA != NoTrans.  THE PORTABLE ARM.
//
//   y_j = alpha * sum_i conj?(A[i + j*ld]) * x[i*xinc] + beta * y[j*yinc]
//
// ONE WORK-ITEM PER OUTPUT COLUMN, each walking a whole CONTIGUOUS column. On a
// GPU that means lanes read `ld` apart and the access is not coalesced -- which
// is exactly why body 3 exists and why the router prefers it wherever the
// device enumerates a sub-group size of 32.
//
// This body is not a fallback in the apologetic sense. IT IS THE ONE THAT
// CLOSES HALF THE BURN-DOWN: GemvMatrixViewTest/0..3 run on a native_cpu
// Device("cpu") queue, where a 32-lane sub-group does not exist and where one
// work-item walking a contiguous column is the RIGHT shape -- it is a serial
// dot product over a unit-stride array, which is what a vectorising host
// compiler wants.
//
// ZERO BYTES OF LOCAL MEMORY.
// ===========================================================================
template <typename T>
Event gemv_direct_trans(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const VectorView<T>& X,
                        const VectorView<T>& Y,
                        T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/0);
    const int64_t groups = (items + wg - 1) / wg;

    ctx->submit([&](sycl::handler& h) {
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;

        h.parallel_for<GemvDirectTKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int64_t gid = static_cast<int64_t>(it.get_global_linear_id());
                if (gid >= total) return;

                const int64_t b = gid / out_len;
                const int j = static_cast<int>(gid - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int i = 0; i < red_len; ++i) {
                        D av = Acol[i];
                        // ConjTrans. `if constexpr` so a real instantiation
                        // emits no branch at all, and a runtime bool inside it
                        // so complex gets ONE launch-uniform branch rather than
                        // a second set of instantiations.
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                D out{};
                fma_acc(out, alpha_d, acc);
                if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                yb[static_cast<int64_t>(j) * yinc] = out;
            });
    });

    return ctx.get_event();
}

// ===========================================================================
// BODY 3 -- {Native, CTA}, transA != NoTrans, GPU with an ENUMERATED sub-group
// size of 32.
//
// ONE 32-LANE SUB-GROUP PER OUTPUT ELEMENT. Lane l walks the reduction index as
// i = l, l+32, l+64, ..., so at every step the 32 lanes read
// A[l + i0 + j*ld] for consecutive l -- ADJACENT, i.e. fully coalesced, which
// is precisely what body 2 cannot do. The partial sums are then folded with the
// hand-rolled shift_group_left ladder and lane 0 alone writes y.
//
// THE EARLY EXIT IS SUB-GROUP UNIFORM, and it has to be: `sg_out` is derived
// from the sub-group's own id, so either every lane of a sub-group returns or
// none does. A shuffle reached by only some lanes of a sub-group is undefined
// behaviour, and the ladder below is reached by every lane that did not return.
//
// ZERO BYTES OF LOCAL MEMORY: the only collective here is a sub-group shuffle,
// never sycl::reduce_over_group.
// ===========================================================================
template <typename T>
Event gemv_cta_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    constexpr int kSg = 32;

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    // One 32-lane sub-group per output element, so a work-group of `wg` covers
    // wg/32 outputs -- hence the shift of 5. Same flattening as bodies 1 and 2:
    // the group count is driven by (out_len * batch), never by batch alone.
    const int wg = gemv_wg_ladder(items, max_wg, cu, /*units_per_wg_shift=*/5);
    // A work-group narrower than one sub-group would make this zero and the
    // group count a division by zero. supports(CTA) has already established
    // that the device enumerates a sub-group size of 32, so this cannot fire;
    // it is here because the alternative to a guard is a crash, not a fallback.
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (items + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvCtaTKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t sg_out =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());

                // SUB-GROUP UNIFORM -- see the note above. Every lane of this
                // sub-group computes the same sg_out, so this returns all 32
                // lanes or none, and the shuffle ladder below is never reached
                // by a partial sub-group.
                if (sg_out >= total) return;

                const int64_t b = sg_out / out_len;
                const int j = static_cast<int>(sg_out - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero) {
                    for (int i = lane; i < red_len; i += kSg) {
                        D av = Acol[i];
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                acc = sg_sum(sg, acc);

                if (lane == 0) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                    yb[static_cast<int64_t>(j) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

// ===========================================================================
// BODY 5 -- GemvSegTKernel<T, W>. {Native, CTA}, transA != NoTrans, GPU with an
// ENUMERATED sub-group size of 32, SHORT REDUCTION (red_len <= 64).
//
// W OUTPUTS PER SUB-GROUP, L = 32/W LANES PER OUTPUT. This is body 4's idea
// TRANSPOSED, and the important word is transposed rather than copied:
//
//     BODY 4 (NoTrans)      i = lane % out_len      jsub = lane / out_len
//     BODY 5 (Trans)        s = lane % L            o    = lane / L
//
// i.e. body 4 makes the OUTPUT index vary fastest across lanes and body 5 makes
// the REDUCTION index vary fastest. THAT INVERSION IS THE WHOLE POINT, and it
// follows from the layout premise in gemv_native.hh: A(i,j) lives at i + j*ld,
// so under NoTrans the contiguous memory index is the OUTPUT index i, while
// under Trans it is the REDUCTION index i (we walk down one column). Lanes must
// vary fastest along whichever index is contiguous. Writing body 5 with body
// 4's `o = lane % W` compiles, gives the right answer, and inverts the
// coalescing -- the way to reproduce body 4's "better sectors, slower kernel"
// result with the sign flipped. Break `segTmap` holds it.
//
// WHAT IT BUYS, AND WHY IT IS THE FOLD AND NOT THE TRAFFIC. See the counter
// table on kGemvSegTransMaxW above: body 3's fold is 32*5 = 160 FP64 adds per
// output for `double` and 320 for `complex<double>`, on a part whose FP64 rate
// is 1/64 of its FP32 rate, and ncu measures that pipe at 82-86% of peak while
// DRAM sits at 40-66%. Body 5's fold is L*log2(L) adds per output -- 8 at
// L = 4 -- a 20x cut on the term that is actually binding. The loop's red_len
// useful FMAs per output are UNCHANGED; only the lane count serving each output
// moves. Occupancy and sectors-per-load were never the defect and are not
// touched.
//
// THE EARLY EXIT IS **NOT** SUB-GROUP UNIFORM, AND THAT IS A REAL TRAP THAT
// BODY 3 DOES NOT HAVE. Body 3's `if (sg_out >= total) return;` is uniform BY
// CONSTRUCTION: sg_out comes from the sub-group's own id, so all 32 lanes agree.
// Body 5's sub-group covers W DIFFERENT outputs, so a tail sub-group can have
// some lane groups in range and others past the end. A shuffle reached by only
// part of a sub-group is undefined behaviour. So the sub-group returns as a
// whole only on `base >= total`, which IS uniform, and the partial tail is
// MASKED by `active`: out-of-range lane groups still execute the fold (with a
// zero accumulator, which is the fold's identity) and are silenced at the load
// and at the store. Their sg_out is also CLAMPED before any pointer arithmetic,
// so no out-of-range address is even formed. Break `segTtail`.
//
// THE FOLD IS CLOSED AT STRIDE 1. Lane o*L + 0 accumulates exactly lanes
// o*L + 0 .. o*L + L-1: the offsets descend L/2, L/4, ..., 1, and 32 % L == 0
// so every group of L lanes is L-aligned and no dependency crosses a group
// boundary. Lanes with s > 0 pick up their neighbours' partials and are never
// read; only the last group's high lanes can shift past lane 31, exactly as
// sg_sum's ladder already does in body 3. Break `segTfold` puts the stride at
// L instead of 1 and turns it red.
//
// W IS A TEMPLATE PARAMETER. Body 4's recorded history is that a runtime W
// fixed the short end and REGRESSED the long end below the body it replaced, at
// better sectors-per-load AND better occupancy -- the loop, not the traffic.
// See the measured table on gemv_seg_width.
//
// ZERO BYTES OF LOCAL MEMORY: the only collective is a sub-group shuffle, never
// sycl::reduce_over_group. That is what keeps the recorded 48 KB launch hole
// structurally unreachable in this TU.
// ===========================================================================
template <typename T, int W>
Event gemv_seg_trans(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& A,
                     const VectorView<T>& X,
                     const VectorView<T>& Y,
                     T alpha, T beta, bool conjugate) {
    using D = typename DevMap<T>::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");
    static_assert(W >= 2 && W <= 32 && (W & (W - 1)) == 0, "W must be a power of two in [2, 32]");

    constexpr int kSg = 32;
    constexpr int kLanes = kSg / W;          // L -- lanes per output

    const int m = A.rows();
    const int n = A.cols();
    const int batch = A.batch_size();

    const int64_t items = static_cast<int64_t>(n) * batch;
    // W FEWER SUB-GROUPS THAN BODY 3, and the unit handed to the ladder is the
    // SUB-GROUP COUNT, not the output count. Body 3 passes items with shift 5;
    // passing items here would over-launch by exactly W.
    const int64_t sub_groups = (items + W - 1) / W;

    const auto dev = ctx.device();
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cu = static_cast<int>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int wg = gemv_wg_ladder(sub_groups, max_wg, cu, /*units_per_wg_shift=*/5);
    const int sgs_per_wg = (wg / kSg) > 0 ? (wg / kSg) : 1;
    const int64_t groups = (sub_groups + sgs_per_wg - 1) / sgs_per_wg;

    ctx->submit([&](sycl::handler& h) {
        // NO __restrict__, for ortho.cc:227-232's reason. See body 1.
        const D* a_ptr = reinterpret_cast<const D*>(A.data_ptr());
        const D* x_ptr = reinterpret_cast<const D*>(X.data_ptr());
        D* y_ptr = reinterpret_cast<D*>(Y.data_ptr());

        const int64_t lda = A.ld();
        const int64_t stride_a = A.stride();
        const int64_t xinc = X.inc();
        const int64_t stride_x = X.stride();
        const int64_t yinc = Y.inc();
        const int64_t stride_y = Y.stride();

        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));

        const bool alpha_zero = (alpha == T(0));
        const bool beta_zero = (beta == T(0));
        const bool conj = conjugate;

        const int out_len = n;
        const int red_len = m;
        const int64_t total = items;
        const int sgs = sgs_per_wg;

        h.parallel_for<GemvSegTKernel<T, W>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(groups) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSg)]] {
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t sg_id =
                    static_cast<int64_t>(it.get_group_linear_id()) * sgs +
                    static_cast<int64_t>(sg.get_group_linear_id());
                const int64_t base = sg_id * W;

                // SUB-GROUP UNIFORM -- every lane computes the same `base`, so
                // this returns all 32 lanes or none, and the fold below is
                // never reached by a partial sub-group.
                if (base >= total) return;

                // THE MIRROR OF BODY 4: the lane index varies fastest in the
                // REDUCTION, because under Trans that is the contiguous one.
                const int s = lane % kLanes;         // slice of the reduction
                const int o = lane / kLanes;         // which of the W outputs

                const int64_t sg_out = base + o;
                // MASKED, NOT RETURNED. A tail sub-group is partially in range.
                const bool active = (sg_out < total);
                // CLAMPED before any pointer arithmetic, so an inactive lane
                // group never even forms an out-of-range address.
                const int64_t sg_out_c = active ? sg_out : (total - 1);

                const int64_t b = sg_out_c / out_len;
                const int j = static_cast<int>(sg_out_c - b * out_len);

                const D* Acol = a_ptr + b * stride_a + static_cast<int64_t>(j) * lda;
                const D* xb = x_ptr + b * stride_x;
                D* yb = y_ptr + b * stride_y;

                D acc{};
                if (!alpha_zero && active) {
                    // kLanes is a compile-time constant, so the trip count
                    // (red_len - s + L - 1)/L and the address stride L are both
                    // known and the loop unrolls. That is the whole reason W is
                    // a template parameter.
                    for (int i = s; i < red_len; i += kLanes) {
                        D av = Acol[i];
                        if constexpr (dev_is_complex_v<D>) {
                            if (conj) av = dev_conj(av);
                        }
                        fma_acc(acc, av, xb[static_cast<int64_t>(i) * xinc]);
                    }
                }

                // THE SEGMENTED FOLD: STRIDE 1, log2(L) steps, descending
                // offsets. Body 4's fold is at stride out_len; body 5's is at
                // stride 1 for the same reason its mapping is inverted.
                if constexpr (dev_is_complex_v<D>) {
                    auto re = acc.re, im = acc.im;
                    for (int off = kLanes >> 1; off >= 1; off >>= 1) {
                        re += sycl::shift_group_left(sg, re, off);
                        im += sycl::shift_group_left(sg, im, off);
                    }
                    acc = D{re, im};
                } else {
                    for (int off = kLanes >> 1; off >= 1; off >>= 1) {
                        acc += sycl::shift_group_left(sg, acc, off);
                    }
                }

                // W lanes write instead of body 3's one, and when the W outputs
                // fall inside one batch item they are contiguous in y.
                if (s == 0 && active) {
                    D out{};
                    fma_acc(out, alpha_d, acc);
                    if (!beta_zero) fma_acc(out, beta_d, yb[static_cast<int64_t>(j) * yinc]);
                    yb[static_cast<int64_t>(j) * yinc] = out;
                }
            });
    });

    return ctx.get_event();
}

}  // namespace

// ---------------------------------------------------------------------------
// WHICH CTA KERNEL A GIVEN (queue, red_len) WOULD RESOLVE TO: 1 = body 3, and
// W >= 2 = body 5 at that W. TEST-ONLY, and it exists because BOTH of the
// things it reports have been blind guards in this campaign.
//
//   * "LINKED IS NOT REACHABLE". {Native, CTA} now names two kernels, so the
//     resolved route column -- the campaign's usual instrument -- cannot tell
//     a body-5 launch from a body-3 launch. A GATE-D break against body 5 that
//     runs at a shape the gate sends to body 3 is VACUOUS and looks green.
//     This is the only thing a test can observe that separates them.
//
//   * THE ENV. BATCHLAS_GEMV_SEGT is re-read per call and NOT latched, for
//     getrs_native.cc:386-402's reason: a latched PRESENCE makes a later
//     setenv invisible, the test runs the DEFAULT arm, and it passes green.
//     That is the campaign's eleventh recorded blind guard.
//
// It resolves through the SAME function the launcher calls -- there is no second
// copy of the gate -- and it reports 1 for any type body 5 is not emitted for
// and for any device that does not enumerate a sub-group size of 32, which are
// the two ways the launcher can decline after the gate has said yes.
// ---------------------------------------------------------------------------
template <typename T>
int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch) {
    if constexpr (!kGemvSegTransEmit<T>) {
        static_cast<void>(ctx);
        static_cast<void>(red_len);
        static_cast<void>(out_len_times_batch);
        return 1;
    } else {
        if (!ctx.device().supports_sub_group_size(32)) return 1;
        return gemv_seg_trans_width<T>(
            gemv_segt_mode(), red_len, out_len_times_batch,
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
    }
}

// ---------------------------------------------------------------------------
// The two public entries.
// ---------------------------------------------------------------------------
template <typename T>
Event gemv_native_direct(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const VectorView<T>& X,
                         const VectorView<T>& Y,
                         T alpha, T beta, Transpose transA) {
    if (gemv_quick_return(A.rows(), A.cols(), alpha, beta)) return ctx.get_event();

    if (transA == Transpose::NoTrans) {
        // BODY 4 vs BODY 1 -- A DEVICE-AND-SHAPE CHOICE, NOT A ROUTE.
        //
        // Both compute the same y from the same inputs; body 4 only decomposes
        // the work differently, so this decision is deliberately invisible to
        // the routing vocabulary. Putting it in RouteTable<Op::gemv>::supports()
        // would be a speed cutoff in the one predicate that is documented to
        // carry correctness only; putting it in preferred() would need a second
        // native algorithm name for what is one route.
        //
        // THE GATE IS THE ENUMERATED SUB-GROUP SIZE, never
        // get_property(MAX_SUB_GROUP_SIZE) -- that returns sub_group_sizes()[0]
        // and is wrong in both directions, and body 4 carries
        // [[sycl::reqd_sub_group_size(32)]], for which the "accepted although it
        // has no 32" direction is a launch abort. This is the same query and the
        // same reasoning as GemvShape::has_sg32 (src/backends/gemv_route.hh).
        //
        // On the native_cpu queue supports_sub_group_size(32) is FALSE, so the
        // 20 Backend::NETLIB rows of tests/gemv_tests.cc keep taking body 1 and
        // the Direct route keeps its no-GPU-gate property intact.
        const int w = gemv_seg_width(A.rows());
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 32: return gemv_seg_notrans<T, 32>(ctx, A, X, Y, alpha, beta);
                case 16: return gemv_seg_notrans<T, 16>(ctx, A, X, Y, alpha, beta);
                case 8:  return gemv_seg_notrans<T, 8>(ctx, A, X, Y, alpha, beta);
                case 4:  return gemv_seg_notrans<T, 4>(ctx, A, X, Y, alpha, beta);
                default: return gemv_seg_notrans<T, 2>(ctx, A, X, Y, alpha, beta);
            }
        }
        return gemv_direct_notrans<T>(ctx, A, X, Y, alpha, beta);
    }
    return gemv_direct_trans<T>(ctx, A, X, Y, alpha, beta,
                                transA == Transpose::ConjTrans);
}

template <typename T>
Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha, T beta, Transpose transA) {
    if (transA == Transpose::NoTrans) {
        // ENFORCED, not assumed. supports() already gates the CTA route on
        // transA != NoTrans, so reaching here means a direct caller (or a
        // pinned BATCHLAS_GEMV_ROUTE that got past the table) violated the
        // contract -- and the alternative to throwing is quietly computing the
        // wrong product. trsm_native_v1_buckets sets the precedent.
        throw std::runtime_error(
            "BatchLAS: gemv_native_cta called with transA = NoTrans. The CTA body "
            "reduces down a column and serves only Trans/ConjTrans; NoTrans is "
            "already fully coalesced with one work-item per output row and is the "
            "Direct body's job (Algorithm::Direct).");
    }
    if (gemv_quick_return(A.rows(), A.cols(), alpha, beta)) return ctx.get_event();

    // BODY 5 vs BODY 3 -- A DEVICE-AND-SHAPE CHOICE, NOT A ROUTE, exactly as
    // body 4 vs body 1 is on the Direct side. Both compute the same y from the
    // same inputs; body 5 only decomposes the reduction differently, so this
    // decision is deliberately invisible to the routing vocabulary. It is NOT in
    // supports() (that predicate is documented to carry correctness only, and a
    // red_len cutoff there would be a speed cutoff in it) and NOT in preferred()
    // (that would need a second native algorithm name for what is one route).
    //
    // CONSEQUENCE FOR THE ROUTE COLUMN, recorded in route_gemv.hh: {Native, CTA}
    // now names TWO kernels, so a resolved route of `native:cta` no longer
    // identifies which kernel ran. gemv_seg_trans_width_debug is what does.
    //
    // THE GATE IS ON red_len(), never out_len() -- see kGemvSegTransMaxRedLen.
    // Under Trans/ConjTrans red_len() is A.rows().
    //
    // THE SUB-GROUP GATE IS THE ENUMERATED SIZE, never
    // get_property(MAX_SUB_GROUP_SIZE), which returns sub_group_sizes()[0] and
    // is wrong in both directions. Body 5 carries [[reqd_sub_group_size(32)]],
    // for which the "accepted although it has no 32" direction is a launch
    // abort. Same query and same reasoning as GemvShape::has_sg32.
    if constexpr (kGemvSegTransEmit<T>) {
        const int w = gemv_seg_trans_width<T>(
            gemv_segt_mode(), A.rows(),
            static_cast<int64_t>(A.cols()) * A.batch_size(),
            static_cast<int>(ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS)));
        if (w >= 2 && ctx.device().supports_sub_group_size(32)) {
            switch (w) {
                case 8: return gemv_seg_trans<T, 8>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans);
                case 4: return gemv_seg_trans<T, 4>(ctx, A, X, Y, alpha, beta,
                                                    transA == Transpose::ConjTrans);
                default: return gemv_seg_trans<T, 2>(ctx, A, X, Y, alpha, beta,
                                                     transA == Transpose::ConjTrans);
            }
        }
    }

    return gemv_cta_trans<T>(ctx, A, X, Y, alpha, beta,
                             transA == Transpose::ConjTrans);
}

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAGS, as full explicit specialisations in the same TU as the
// kernels (route_trsm.hh:62-84's rule). A build that drops this file drops
// these definitions too, so the route table can never advertise a kernel that
// is not linked -- the failure mode the campaign records as "LINKED is not
// REACHABLE", stated the other way round.
//
// ALL FOUR TYPES, BOTH TIERS. The instantiation macro below emits every body
// for every scalar type, so there is no type for which one of these is a lie.
//
// NOTE WHAT gemv_cta_available DOES NOT SAY. It is a statement about the BUILD,
// not about the device: `is_gpu` and the ENUMERATED sub-group size 32 are the
// shape builder's job (src/backends/gemv_route.hh) and the table's, and this
// flag deliberately carries neither. In particular gemv_direct_available must
// stay free of any device notion -- the Direct route has NO GPU GATE, and that
// is the WP7 deliverable: the 20 NETLIB rows of tests/gemv_tests.cc run on a
// native_cpu queue and a GPU-gated Direct route would leave the suite red and
// move the vendor-free burn-down by exactly zero.
// ---------------------------------------------------------------------------
template <> bool gemv_direct_available<float>()                { return true; }
template <> bool gemv_direct_available<double>()               { return true; }
template <> bool gemv_direct_available<std::complex<float>>()  { return true; }
template <> bool gemv_direct_available<std::complex<double>>() { return true; }

template <> bool gemv_cta_available<float>()                { return true; }
template <> bool gemv_cta_available<double>()               { return true; }
template <> bool gemv_cta_available<std::complex<float>>()  { return true; }
template <> bool gemv_cta_available<std::complex<double>>() { return true; }

#define BATCHLAS_GEMV_NATIVE_INSTANTIATE(fp)                                   \
    template Event gemv_native_direct<fp>(                                     \
        Queue&, const MatrixView<fp, MatrixFormat::Dense>&,                    \
        const VectorView<fp>&, const VectorView<fp>&, fp, fp, Transpose);      \
    template Event gemv_native_cta<fp>(                                        \
        Queue&, const MatrixView<fp, MatrixFormat::Dense>&,                    \
        const VectorView<fp>&, const VectorView<fp>&, fp, fp, Transpose);      \
    template int gemv_seg_trans_width_debug<fp>(Queue&, int, int64_t);

BATCHLAS_GEMV_NATIVE_INSTANTIATE(float)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(double)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(std::complex<float>)
BATCHLAS_GEMV_NATIVE_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEMV_NATIVE_INSTANTIATE

}  // namespace batchlas::sycl_gemv
