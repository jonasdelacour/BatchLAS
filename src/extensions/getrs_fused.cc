// Native batched GETRS -- the FUSED NARROW-RHS tier. ONE kernel per matrix:
// the row permutation, the forward substitution and the back substitution, with
// no GEMM launch and no separate laswp launch anywhere.
//
// ===========================================================================
// WHY THIS TIER EXISTS, AND IT IS A MEASUREMENT, NOT A PREFERENCE.
//
// The composed tier (getrs_native.cc: laswp + two ROUTED trsm) is 0.32x of
// cublas?getrsBatched at nrhs = 1 and only crosses 1.0x around nrhs = 64. nsys on
// the vendor-free build, float n=512 nrhs=1 batch=512, ONE public-API call:
//
//     29.3%  TrsmCtaKernel<float,32,Side=Left>          37,967 instances
//     26.4%  LuLaswpKernel<GetrsLaswpTag>                1,186 instances
//     39.7%  GemmTiledGeneralKernel<float,16> x3        ~26,000 instances
//      3.1%  GemmDirectKernel<float>                      9,489 instances
//
// The 39.7% is MATRIX-VECTOR products run through a TILE-16 GEMM kernel: trsm's
// blocked driver exists to amortise a panel over many columns and one column
// gives it nothing to amortise. BatchLAS has no native trsv, so a narrow solve
// was being served entirely by machinery built for matrices.
//
// A CORRECTION TO THE PREMISE THIS TIER WAS COMMISSIONED ON, and it is why the
// tier serves a WINDOW rather than nrhs = 1 alone. The brief asserted that
// "linalg::solve issues exactly nrhs = 1". It does not: linalg-ops.hh:336-344
// builds X with B.rows() x B.COLS() and passes it straight through, so the width
// is whatever the caller's right-hand side has. The same is true of the Python
// binding (python/batchlas/bindings/ops_factorization.cc:91). Those two ARE the
// only callers of getrs in the tree, so the "only internal caller" half of the
// claim holds and the "always nrhs = 1" half does not.
//
// THE CEILING IS MEMORY, AND THIS KERNEL REACHES IT IN A BAND -- NOT EVERYWHERE.
// The substitution is O(n^2) work over O(n^2) matrix reads, so at nrhs = 1 there
// is nothing to do but stream L and U once. Measured through this kernel at float
// n=512 nrhs=1 batch=512: 0.6506 ms for 512 MB of factored matrix = 825 GB/s, i.e.
// 82% of this device's 1008 GB/s DRAM peak, against cuBLAS's 366 GB/s and the
// composition's 211 GB/s.
//
// AN EARLIER VERSION OF THIS PARAGRAPH ENDED "there is no second factor of two
// available here", AND THAT IS TRUE ONLY IN THE n = 256..512 BAND. Achieved
// fraction of 1008 GB/s at nrhs = 1, recomputed per cell from
// experiments/wp6_perf/bench/grid_cta.csv as
// (sizeof(T)*n*n*batch + 2*sizeof(T)*n*nrhs*batch + 4*n*batch) / med_ms:
//
//              float   double   cfloat  cdouble
//     n=32      72%      38%      95%      24%
//     n=64      78%      61%      81%      42%
//     n=256     78%      79%      85%      75%
//     n=512     82%      86%      88%      83%      <- the band
//     n=1024    70%      74%      80%      71%
//     n=2048    41%      50%      60%      41%
//
// TWO NAMED MECHANISMS, both open work rather than a ceiling:
//   * LARGE n: one work-group per matrix means the CTA COUNT IS THE BATCH, and the
//     saturating batch at n=2048 is 32 -- 32 of this device's 128 SMs. This is the
//     batch-only-parallelism defect in its mildest form (the work-group is 1024
//     wide, so the device is not idle) and it is why the wg rule keeps growing
//     with n. It cannot be fixed by tuning; it needs work-groups to split a matrix.
//   * SMALL n: nb = 16 below n = 1024, so the diagonal-block recurrence runs on 16
//     lanes of one 32-lane sub-group while the rest of the work-group waits. It is
//     worst for cdouble, whose block solve is the most expensive per step.
//
// AND THE FOLDED PERMUTATION IS THE LARGEST NAMED RESIDUAL AT LARGE n. Priced by
// rebuilding with the interchange walk removed (experiments/wp6_getrs/proto/
// noperm.csv, residual column confirms the break took rather than being optimised
// away): float n=2048 b=32 1.2802 -> 1.1776 ms, i.e. 8.0% of the call; float n=512
// b=512 3.5%; cdouble n=2048 2.4%. It is the one fully SERIAL part of the kernel --
// `if (tid < nrhs)` over n dependent local-memory swaps -- so at nrhs = 1 exactly
// one work-item of up to 1024 does n round-trips while the rest wait at a barrier,
// and at n=2048 there is at most one work-group per SM to hide it behind. Folding
// it in was still right (26.4% as a separate launch), but 8.0% is not zero and it
// is the next lever.
//
// ===========================================================================
// THE SHAPE, AND THE THREE DESIGN CHOICES THAT ARE MEASURED RATHER THAN ASSUMED.
//
// ONE WORK-GROUP PER MATRIX, PARALLEL OVER ROWS. n = 512 float is 1 MB per
// matrix, so the MATRIX IS NOT RESIDENT and no CTA-resident-matrix design is
// possible at the sizes that matter. What is resident is the RHS VECTOR
// (n x nrhs) and one nb x nb diagonal block; L and U are streamed.
//
// (1) COLUMN-ORIENTED (axpy), NOT ROW-ORIENTED (dot), FOR NoTrans. BatchLAS is
//     column-major, so the axpy form reads a CONTIGUOUS column segment at every
//     step -- L[k+1..n-1, k] going forward, U[0..k-1, k] coming back. The dot
//     form would read a ROW, i.e. `ld` apart: 32 transactions per warp access,
//     which is the same defect lu_laswp.hh documents for the interchange walk.
//     The TRANSPOSED modes are the mirror image and use the DOT form for exactly
//     the same reason -- (U^T)[i][j] = U[j][i], so a dot against column i is the
//     contiguous access there.
//
// (2) A RESIDENT DIAGONAL BLOCK, NOT PURE STREAMING, AND THE MARGIN IS LARGE.
//     Both were built and measured (experiments/wp6_getrs/proto/grid_nv.csv).
//     Pure streaming pays a WORK-GROUP BARRIER PER COLUMN; the blocked form pays
//     one per BLOCK and runs the nb-step recurrence inside ONE SUB-GROUP with
//     shuffles. float n=512 nrhs=1 batch=512: streaming 1.0102 ms, blocked
//     0.6506 ms. The streaming arm is a REVERTED variant with its number.
//
// (3) THE PERMUTATION IS FOLDED IN, and it is the one part that cannot be
//     parallelised: LAPACK's ipiv is a SEQUENCE of transpositions, so it must be
//     walked IN ORDER and column c of the RHS is walked by a single work-item.
//     It is walked in LOCAL memory -- the RHS is loaded coalesced first -- and
//     never in global, where n dependent round-trips per matrix would each be a
//     few hundred cycles. As a separate launch it was 26.4% of the composed call.
//
// ===========================================================================
// TRANSPOSED MODES. transA = Trans / ConjTrans SWAP THE TWO SOLVES **AND** MOVE
// THE PERMUTATION TO THE OUTPUT, IN REVERSE. That is the classic silently-wrong
// answer in a getrs, and it is derived rather than copied:
//
//     A = F^{-1} L U, F the interchange sequence applied FORWARDS.
//     A^T = (F^{-1} L U)^T = U^T L^T F^{-T} = U^T L^T F   (F orthogonal)
//     A^T x = b  <=>  U^T L^T (F x) = b
//                 =>  solve U^T, solve L^T, then x = F^{-1} w
//     and F^{-1} is the SAME list walked BACKWARDS, because each transposition is
//     its own inverse and only the order reverses.
//
// ConjTrans is identical with H for T; F is real so it is unchanged.
//
// ===========================================================================
// THIS FILE SITS IN EXTENSIONS_FACTORIZATION_SOURCES, beside getrs_native.cc and
// NOT in EXTENSIONS_CTA_SOURCES: it calls no getrf CTA device function, so it
// shares no device symbol with that cluster (src/extensions/CMakeLists.txt:77-89's
// rule). If it ever does, it MOVES -- and getting that wrong is a hard
// `ptxas fatal: Unresolved extern function`, never a silent miscompile.

#include "getrs_native.hh"

#include "../queue.hh"
#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrs {

namespace {

using namespace batchlas::sycl_device;

// ---------------------------------------------------------------------------
// GEOMETRY. All of it is a measured rule, and all of it lives HERE so that the
// ceiling route_getrs.hh advertises and the allocation this launcher makes cannot
// disagree (route_trsm.hh:62-72's rule).
// ---------------------------------------------------------------------------

// The resident diagonal block. It must not exceed the sub-group size, because
// the block solve is a sub-group shuffle recurrence with lane i owning row i.
//
// Measured at float nrhs=1 (ms, lower is better), tuning sweep in
// experiments/wp6_getrs/proto/tune.sh:
//     n=64  b=8192 : nb 8 0.1825  nb 16 0.1786  nb 32 0.1947
//     n=128 b=4096 : nb 8 0.3491  nb 16 0.3404  nb 32 0.3431
//     n=512 b=512  : nb 8 0.6605  nb 16 0.6485  nb 32 0.6508
//     n=2048 b=32  : nb 8 1.5513  nb 16 1.3772  nb 32 1.2838
constexpr int kGetrsFusedNbSmall = 16;
constexpr int kGetrsFusedNbLarge = 32;
constexpr int kGetrsFusedNbMax   = 32;

inline int getrs_fused_nb(int n) {
    const int nb = (n >= 1024) ? kGetrsFusedNbLarge : kGetrsFusedNbSmall;
    return (nb > n) ? n : nb;
}

// The block's LEADING DIMENSION is nb + 1. The NoTrans block solve has lane i read
// blk[i + kk*ldb], stride 1 across lanes; the TRANSPOSED block solve has lane t
// read blk[s + t*ldb], stride ldb across lanes. At ldb = 16 or 32 that is a 16-
// or 32-way bank conflict on every step of the transposed recurrence; at 17 or 33
// the stride is coprime with 32 and the access is conflict-free.
//
// AND IT IS WORTH NOTHING MEASURABLE ON THIS DEVICE, which is stated rather than
// left as an unmeasured claim. Both spellings were timed ON THE TRANSPOSED PATH
// -- the only one whose recurrence can care; timing NoTrans would have reported
// 1.00x and proved nothing -- through the public API, vendor-free, pinned
// native:cta, at saturating batch (experiments/wp6_getrs/pad_ab.sh; ms, pad vs no
// pad):
//     float   n=512 nrhs=1  0.7245 / 0.7091     n=512 nrhs=8  1.6135 / 1.6200
//     float   n=2048 nrhs=1 1.2946 / 1.2904     n=2048 nrhs=8 3.5526 / 3.5636
//     cdouble n=512 nrhs=1  2.5899 / 2.5878     n=512 nrhs=8 23.4144 / 23.4122
//     cdouble n=2048 nrhs=1 6.4080 / 6.4068
// Eight cells, seven of them within 0.6% and split in both directions -- but the
// EIGHTH, float n=512 nrhs=1, is 2.17% and it is a LOSS for the spelling that was
// kept. Calling the table "a wash" was too kind to the decision: the cell with the
// largest signal in it points the other way. The honest reading is that the
// diagonal block solve is jb^2/2 flops against a kernel that streams n^2 elements,
// so nothing here can be worth more than noise on this device, and the one cell
// that exceeds noise says the pad costs rather than pays.
//
// IT STAYS ANYWAY, and the reason is portability rather than measurement -- for
// potrf_cta.cc:275-285's reason
// about its own measured-inert hole pad -- because it costs 32 elements of local
// memory out of thousands and this library is not compiled for one device;
// deleting it is a change with no upside on the only box that has measured it.
//
// Break B5 (pad removed) turned NOTHING red, correctly: this is a performance
// choice and not a correctness one.
inline int getrs_fused_blk_ld(int nb) { return nb + 1; }

// THE REGISTER GATE, AND IT IS NOT DEFENSIVE -- IT FIRED.
//
// registers-per-work-item x work-group-size must not exceed 65,536, the per-block
// register-file limit; over it the launch ABORTS ("Exceeded the number of
// registers available on the hardware"), it does not merely slow down.
//
// WITHOUT THIS CAP THE ABORT IS REACHABLE AND WAS REPRODUCED: float, n = 2048,
// nrhs = 8, transA = Trans picks wg = 1024 by the rule below and the float
// transposed NR = 8 kernel uses 68 registers, i.e. 69,632 -- the NoTrans arm of
// the very same call ran green first (48 registers) and then the Trans arm threw.
// tests/getrf_tests.cc FusedGetrsLaunchHoleAt48KiB reaches it at n = 1428,
// transA = Trans, nrhs = 8 (its top rung picks wg = 1024), which is the one shape
// in the whole binary that is both wide and deep enough; deleting the three cap
// lines turns that test RED as a hard abort, not as a wrong answer.
//
// THE TABLE IS PER (TYPE, BODY, WIDTH) AND NOT A MAX OVER THEM, and that is a
// repair. It used to be one number per width -- the max over both kernels and all
// four scalar types -- so GetrsFusedNKernel<float,8>, which ptxas puts at 48
// registers, was capped at wg = 672 by GetrsFusedTKernel<complex<double>,8>'s 86.
// Both coordinates are known at the call site (T is the entry point's template
// parameter, `trans` is computed there from transA), so charging a kernel for a
// different kernel's registers was never necessary. It cost real time in exactly
// the regime the wg rule exists for -- see the A/B in
// experiments/wp6_perf/regcap/README.md.
//
// MEASURED with scripts/register_probe.sh on batchlas_extensions_factorization,
// ZERO SPILL on all 528 entry functions of that library (32 of them are these
// kernels). `Used N registers`, max over <name> and <name>_with_offset:
//
//                NR=1        NR=2        NR=4        NR=8
//   type      NoTr  Tr    NoTr  Tr    NoTr  Tr    NoTr  Tr
//   float      39   39     48   40     48   48     48   68
//   double     39   46     52   44     44   51     61   72
//   cfloat     40   42     40   43     40   48     48   56
//   cdouble    54   56     56   58     56   58     72   86
//
// The widest are the TRANSPOSED bodies -- the dot form carries a sub-group
// reduction per right-hand side on top of the accumulator array -- and the spread
// at NR = 8 is 48 to 86, i.e. nearly 2x, which is why one number for all eight
// cells of that column was the wrong shape.
//
// THE MARGIN IS +8 REGISTERS on the measured figure, because these numbers are a
// property of one compiler version and the failure mode is a hard abort. Nothing
// here is load-bearing unless the uncapped wg would be 1024 (n >= 1026), so a
// drift of up to 8 registers cannot reach a launch that the previous build made.
// If ptxas ever moves a cell by more than that, RE-RUN THE PROBE -- the recipe is
// one line and it is in scripts/register_probe.sh's own header.
//
// The cap is rounded DOWN to a multiple of the sub-group size rather than to a
// power of two: every loop in these kernels is `for (i = base + tid; i < n; i +=
// wg)`, so nothing requires wg to be a power of two, and rounding 992 down to 512
// would cost the n >= 2048 cells 1.4x for nothing.
constexpr int kGetrsFusedRegMargin = 8;

// The accumulator-width bucket. It MUST agree with fused_dispatch_nr's ladder
// below (nrhs <= 1 -> NR 1, <= 2 -> 2, <= 4 -> 4, else 8): a cap computed for a
// bucket other than the one launched is a cap for a different kernel.
constexpr int getrs_fused_nr_bucket(int nrhs) {
    return (nrhs <= 1) ? 0 : (nrhs <= 2) ? 1 : (nrhs <= 4) ? 2 : 3;
}

template <typename S> struct GetrsFusedRegs;
template <> struct GetrsFusedRegs<float> {
    static constexpr int notrans[4] = {39, 48, 48, 48};
    static constexpr int trans[4]   = {39, 40, 48, 68};
};
template <> struct GetrsFusedRegs<double> {
    static constexpr int notrans[4] = {39, 52, 44, 61};
    static constexpr int trans[4]   = {46, 44, 51, 72};
};
template <> struct GetrsFusedRegs<std::complex<float>> {
    static constexpr int notrans[4] = {40, 40, 40, 48};
    static constexpr int trans[4]   = {42, 43, 48, 56};
};
template <> struct GetrsFusedRegs<std::complex<double>> {
    static constexpr int notrans[4] = {54, 56, 56, 72};
    static constexpr int trans[4]   = {56, 58, 58, 86};
};

template <typename T>
constexpr int getrs_fused_regs_for(int nrhs, bool trans) {
    const int i = getrs_fused_nr_bucket(nrhs);
    return trans ? GetrsFusedRegs<T>::trans[i] : GetrsFusedRegs<T>::notrans[i];
}

// The work-group width, ~ n/2 clamped to [64, 1024], then capped by the register
// gate above. Measured (float, nrhs=1, ms at the best nb of each row,
// experiments/wp6_getrs/proto/tune.sh):
//     n=64   b=8192 : wg 32 0.1916  wg 64 0.1786  wg 128 0.2032
//     n=128  b=4096 : wg 64 0.3457  wg 128 0.3404  wg 256 0.4012
//     n=512  b=512  : wg 128 0.6955 wg 256 0.6485  wg 512 0.7504
//     n=2048 b=32   : wg 256 2.8777 wg 512 1.7966  wg 1024 1.2838
// The n=2048 row is the batch-only-parallelism regime in miniature: batch = 32
// puts work-groups on only 32 of 128 SMs, and threads per work-group is then the
// ONLY remaining lever, which is why the rule keeps growing wg with n instead of
// settling at a constant -- and why the register cap being 350 threads too tight
// on the NoTrans arm was worth measuring rather than leaving.
template <typename T>
inline int getrs_fused_wg(int n, int nrhs, int max_wg, bool trans) {
    int wg = 32;
    while (wg < n / 2 && wg < 1024) wg *= 2;
    if (wg < 64) wg = 64;

    const int regs = getrs_fused_regs_for<T>(nrhs, trans) + kGetrsFusedRegMargin;
    int cap = (65536 / regs) & ~31;          // down to a multiple of the sub-group
    if (cap < 32) cap = 32;
    if (wg > cap) wg = cap;

    if (wg > max_wg) wg = max_wg;
    if (wg < 32) wg = 32;
    return wg;
}

// THE 48 KB LAUNCH HOLE, carried verbatim from potrf_cta.cc:259-296 because the
// two must agree: a dynamic local-memory request in (49152 - static_shared, 49152]
// fails with CUDA_ERROR_INVALID_VALUE at enqueue -- too big for CUDA's non-opt-in
// 48 KB limit once the kernel's static shared is added, not big enough for the UR
// adapter to raise MaxDynamicSharedMemorySize. It is STICKY PER CUfunction, so a
// larger earlier launch hides it BY EXECUTION ORDER and a warm test suite cannot
// see it.
//
// This kernel's collectives are SUB-GROUP shuffles (group_broadcast and
// shift_group_left over one sub-group), never a work-group reduce_over_group --
// which is the construct WP4 identified as what puts static shared into a kernel
// and reopens the hole. The band is carried anyway, for potrf_cta.cc's stated
// reason: static shared is not something this source controls.
constexpr std::size_t kGetrsHoleLo    = 47104;
constexpr std::size_t kGetrsHoleHi    = 49664;
constexpr std::size_t kGetrsHolePadTo = 49920;

constexpr std::size_t getrs_hole_padded(std::size_t bytes) {
    return (bytes > kGetrsHoleLo && bytes <= kGetrsHoleHi) ? kGetrsHolePadTo : bytes;
}

// The local-memory request, in bytes, for one work-group.
constexpr std::size_t getrs_fused_slm(std::size_t rhs_elems, int nb,
                                      std::size_t scalar_bytes) {
    return getrs_hole_padded(
        (rhs_elems + static_cast<std::size_t>(nb) *
                     static_cast<std::size_t>(getrs_fused_blk_ld(nb))) * scalar_bytes);
}

// ---------------------------------------------------------------------------
// A sub-group sum. HAND-ROLLED with shift_group_left rather than
// sycl::reduce_over_group, for getrf_cta_device.hh:20-31's reason: a group
// reduction is what reopens the 48 KB hole, and WP6 measured it slower than an
// explicit walk anyway. After log2(32) shift-down steps LANE 0 holds the total;
// no other lane's value is used.
// ---------------------------------------------------------------------------
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

template <typename D>
inline D dev_zero_of() {
    if constexpr (dev_is_complex_v<D>) return D{0, 0};
    else return D(0);
}

// The kernel names. Two bodies, not one with a runtime mode: the NoTrans path is
// an axpy recurrence over rows and the transposed path is a dot recurrence over
// sub-groups, so a single body would allocate registers for the union of two
// unrelated inner loops. CONJUGATION, by contrast, IS runtime -- it is one
// uniform branch on a value that is constant across the whole launch, and making
// it a template parameter would have bought a third set of instantiations for a
// sign flip.
template <typename T, int NR> class GetrsFusedNKernel;
template <typename T, int NR> class GetrsFusedTKernel;

// ===========================================================================
// NoTrans:  apply F to B, solve L y = F b (unit lower), solve U x = y.
// ===========================================================================
template <typename T, int NR>
Event fused_launch_notrans(Queue& ctx,
                           const T* A, int lda, int strideA,
                           T* B, int ldb, int strideB,
                           const int* piv, int pstride,
                           int n, int nrhs, int batch, int wg, int nb) {
    using DM = DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const Ap = reinterpret_cast<const D*>(A);
    D* const Bp = reinterpret_cast<D*>(B);

    const int bld = getrs_fused_blk_ld(nb);
    const std::size_t rhs_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    const std::size_t slm_elems =
        getrs_fused_slm(rhs_elems, nb, sizeof(D)) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(slm_elems), h);
        h.parallel_for<GetrsFusedNKernel<T, NR>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = static_cast<int>(it.get_local_id(0));
                const std::size_t b = it.get_group(0);
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());

                const D* const Ab = Ap + b * static_cast<std::size_t>(strideA);
                D* const Bb = Bp + b * static_cast<std::size_t>(strideB);
                const int* const pv = piv + b * static_cast<std::size_t>(pstride);
                D* const y = &slm[0];
                D* const blk = &slm[rhs_elems];

                // ---- the RHS, loaded coalesced -------------------------------
                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    y[e] = Bb[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // ---- F: the interchange list, walked FORWARDS, in LOCAL memory
                if (tid < nrhs) {
                    D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);
                    for (int k = 0; k < n; ++k) {
                        const int p = pv[k] - 1;          // 1-BASED on the wire
                        if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // ---- L y = F b, unit lower, forward -------------------------
                for (int j = 0; j < n; j += nb) {
                    const int jb = (n - j < nb) ? (n - j) : nb;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            Ab[static_cast<std::size_t>(j + i) +
                               static_cast<std::size_t>(j + c) * static_cast<std::size_t>(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // THE BLOCK SOLVE RUNS INSIDE ONE SUB-GROUP. Lane i owns row
                    // j+i and holds it in a REGISTER; the pivot value is a
                    // shuffle, and the jb-step recurrence contains NO work-group
                    // barrier. That is the whole margin over the streaming form.
                    //
                    // Every lane of the sub-group reaches group_broadcast, which
                    // is a collective and must not be called under divergence --
                    // hence the guards are INSIDE, after it, never around it.
                    if (sgid == 0 && jb > 1) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j + lane] : dev_zero_of<D>();
                            for (int kk = 0; kk < jb - 1; ++kk) {
                                const D pv2 = sycl::group_broadcast(sg, v, kk);
                                if (lane > kk && lane < jb)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(lane) +
                                                               static_cast<std::size_t>(kk) *
                                                               static_cast<std::size_t>(bld)], pv2));
                            }
                            if (lane < jb) yc[j + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- the trailing update, PARALLEL OVER ROWS ------------
                    // Consecutive work-items take consecutive rows, so the read
                    // of A[i, j+kk] is coalesced; the jb column values it
                    // multiplies come from local memory, and each A element is
                    // reused across all nrhs right-hand sides from a register --
                    // the only reuse a narrow solve has.
                    for (int i = j + jb + tid; i < n; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[static_cast<std::size_t>(i) +
                                           static_cast<std::size_t>(j + kk) *
                                           static_cast<std::size_t>(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(j + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + static_cast<std::size_t>(c) *
                                                  static_cast<std::size_t>(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // ---- U x = y, non-unit upper, backward ----------------------
                for (int jend = n; jend > 0; jend -= nb) {
                    const int j0 = (jend - nb > 0) ? (jend - nb) : 0;
                    const int jb = jend - j0;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            Ab[static_cast<std::size_t>(j0 + i) +
                               static_cast<std::size_t>(j0 + c) * static_cast<std::size_t>(lda)];
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (sgid == 0) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j0 + lane] : dev_zero_of<D>();
                            for (int kk = jb - 1; kk >= 0; --kk) {
                                if (lane == kk)
                                    v = dev_div(v, blk[static_cast<std::size_t>(kk) +
                                                       static_cast<std::size_t>(kk) *
                                                       static_cast<std::size_t>(bld)]);
                                const D pv2 = sycl::group_broadcast(sg, v, kk);
                                if (lane < kk)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(lane) +
                                                               static_cast<std::size_t>(kk) *
                                                               static_cast<std::size_t>(bld)], pv2));
                            }
                            if (lane < jb) yc[j0 + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    for (int i = tid; i < j0; i += wg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int kk = 0; kk < jb; ++kk) {
                            const D a = Ab[static_cast<std::size_t>(i) +
                                           static_cast<std::size_t>(j0 + kk) *
                                           static_cast<std::size_t>(lda)];
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(j0 + kk)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                D* const yc = y + static_cast<std::size_t>(c) *
                                                  static_cast<std::size_t>(n);
                                yc[i] = dev_sub(yc[i], acc[c]);
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)] = y[e];
                }
            });
    });
    return ctx.get_event();
}

// ===========================================================================
// Trans / ConjTrans: solve op(U) z = b, solve op(L) w = z, then w = F^{-1} w --
// the SAME interchange list walked BACKWARDS.
//
// Both solves are the DOT form, because op(U)[i][j] = op(U[j][i]): the reduction
// runs down a CONTIGUOUS COLUMN of A, and one sub-group owns one column of the
// current block so the 32 lanes read 32 consecutive elements.
// ===========================================================================
template <typename T, int NR>
Event fused_launch_trans(Queue& ctx,
                         const T* A, int lda, int strideA,
                         T* B, int ldb, int strideB,
                         const int* piv, int pstride,
                         int n, int nrhs, int batch, int wg, int nb, bool conj) {
    using DM = DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    const D* const Ap = reinterpret_cast<const D*>(A);
    D* const Bp = reinterpret_cast<D*>(B);

    const int bld = getrs_fused_blk_ld(nb);
    const std::size_t rhs_elems = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    const std::size_t slm_elems =
        getrs_fused_slm(rhs_elems, nb, sizeof(D)) / sizeof(D);
    const int nsg = wg / 32;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> slm(sycl::range<1>(slm_elems), h);
        h.parallel_for<GetrsFusedTKernel<T, NR>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = static_cast<int>(it.get_local_id(0));
                const std::size_t b = it.get_group(0);
                const auto sg = it.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sgid = static_cast<int>(sg.get_group_linear_id());

                const D* const Ab = Ap + b * static_cast<std::size_t>(strideA);
                D* const Bb = Bp + b * static_cast<std::size_t>(strideB);
                const int* const pv = piv + b * static_cast<std::size_t>(pstride);
                D* const y = &slm[0];
                D* const blk = &slm[rhs_elems];

                auto ld_a = [&](int i, int c) {
                    const D a = Ab[static_cast<std::size_t>(i) +
                                   static_cast<std::size_t>(c) * static_cast<std::size_t>(lda)];
                    return conj ? dev_conj(a) : a;
                };

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    y[e] = Bb[static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)];
                }
                it.barrier(sycl::access::fence_space::local_space);

                // ---- op(U) z = b : op(U) is LOWER, non-unit, forward --------
                for (int j = 0; j < n; j += nb) {
                    const int jb = (n - j < nb) ? (n - j) : nb;

                    // The block's own columns, staged. Read contiguously in i;
                    // the recurrence below indexes it as blk[s + t*bld].
                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            ld_a(j + i, j + c);
                    }

                    // The PAST contribution: y[j+t] -= sum_{i<j} op(A[i, j+t]) y[i],
                    // jb dot products of length j. ONE SUB-GROUP PER COLUMN, so
                    // its 32 lanes stride over i and read 32 consecutive elements.
                    for (int t = sgid; t < jb; t += nsg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int i = lane; i < j; i += 32) {
                            const D a = ld_a(i, j + t);
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(i)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                const D s = sg_sum(sg, acc[c]);
                                if (lane == 0) {
                                    D* const yc = y + static_cast<std::size_t>(c) *
                                                      static_cast<std::size_t>(n);
                                    yc[j + t] = dev_sub(yc[j + t], s);
                                }
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // The diagonal block, by ONE sub-group. Lane t owns row t of
                    // op(U)_block; it reads blk[s + t*bld], stride bld = nb+1
                    // across lanes, which is why the pad exists.
                    if (sgid == 0) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j + lane] : dev_zero_of<D>();
                            for (int s = 0; s < jb; ++s) {
                                if (lane == s)
                                    v = dev_div(v, blk[static_cast<std::size_t>(s) +
                                                       static_cast<std::size_t>(s) *
                                                       static_cast<std::size_t>(bld)]);
                                const D vs = sycl::group_broadcast(sg, v, s);
                                if (lane > s && lane < jb)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(s) +
                                                               static_cast<std::size_t>(lane) *
                                                               static_cast<std::size_t>(bld)], vs));
                            }
                            if (lane < jb) yc[j + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // ---- op(L) w = z : op(L) is UPPER, UNIT, backward -----------
                for (int jend = n; jend > 0; jend -= nb) {
                    const int j0 = (jend - nb > 0) ? (jend - nb) : 0;
                    const int jb = jend - j0;

                    for (int e = tid; e < jb * jb; e += wg) {
                        const int i = e % jb, c = e / jb;
                        blk[static_cast<std::size_t>(i) + static_cast<std::size_t>(c) *
                                                          static_cast<std::size_t>(bld)] =
                            ld_a(j0 + i, j0 + c);
                    }

                    for (int t = sgid; t < jb; t += nsg) {
                        D acc[NR];
                        #pragma unroll
                        for (int c = 0; c < NR; ++c) acc[c] = dev_zero_of<D>();
                        for (int i = jend + lane; i < n; i += 32) {
                            const D a = ld_a(i, j0 + t);
                            #pragma unroll
                            for (int c = 0; c < NR; ++c)
                                if (c < nrhs)
                                    fma_acc(acc[c], a, y[static_cast<std::size_t>(c) *
                                                         static_cast<std::size_t>(n) +
                                                         static_cast<std::size_t>(i)]);
                        }
                        #pragma unroll
                        for (int c = 0; c < NR; ++c)
                            if (c < nrhs) {
                                const D s = sg_sum(sg, acc[c]);
                                if (lane == 0) {
                                    D* const yc = y + static_cast<std::size_t>(c) *
                                                      static_cast<std::size_t>(n);
                                    yc[j0 + t] = dev_sub(yc[j0 + t], s);
                                }
                            }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // UNIT diagonal: no division, and the recurrence runs
                    // BACKWARDS because op(L) is upper.
                    if (sgid == 0 && jb > 1) {
                        for (int c = 0; c < nrhs; ++c) {
                            D* const yc = y + static_cast<std::size_t>(c) * static_cast<std::size_t>(n);
                            D v = (lane < jb) ? yc[j0 + lane] : dev_zero_of<D>();
                            for (int s = jb - 1; s > 0; --s) {
                                const D vs = sycl::group_broadcast(sg, v, s);
                                if (lane < s)
                                    v = dev_sub(v, dev_mul(blk[static_cast<std::size_t>(s) +
                                                               static_cast<std::size_t>(lane) *
                                                               static_cast<std::size_t>(bld)], vs));
                            }
                            if (lane < jb) yc[j0 + lane] = v;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                // ---- F^{-1}: the SAME list, walked BACKWARDS, on the OUTPUT --
                // This is the half of the transposed case that no NoTrans test
                // can see. P^T = S_{k0} ... S_{k1-1} where P = S_{k1-1} ... S_{k0},
                // so the same list applied forwards computes P, not P^T; every
                // transposition is its own inverse, which is why only the ORDER
                // changes.
                if (tid < nrhs) {
                    D* const yc = y + static_cast<std::size_t>(tid) * static_cast<std::size_t>(n);
                    for (int k = n - 1; k >= 0; --k) {
                        const int p = pv[k] - 1;
                        if (p != k) { const D t = yc[k]; yc[k] = yc[p]; yc[p] = t; }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int e = tid; e < n * nrhs; e += wg) {
                    const int i = e % n, c = e / n;
                    Bb[static_cast<std::size_t>(i) +
                       static_cast<std::size_t>(c) * static_cast<std::size_t>(ldb)] = y[e];
                }
            });
    });
    return ctx.get_event();
}

// Runtime nrhs -> the compile-time accumulator width. The ladder stops at 8
// because the tier's measured window does: above nrhs = 8 the composed arm and
// the vendor are both ahead (see getrs_fused_max_rhs and route_getrs.hh), so a
// wider instantiation would be device code nothing selects.
template <typename T>
Event fused_dispatch_nr(Queue& ctx, bool trans, bool conj,
                        const T* A, int lda, int sA, T* B, int ldb, int sB,
                        const int* piv, int pstride,
                        int n, int nrhs, int batch, int wg, int nb) {
    #define BATCHLAS_GETRS_FUSED_ARM(NRV)                                            \
        if (trans) return fused_launch_trans<T, NRV>(ctx, A, lda, sA, B, ldb, sB,    \
                                                     piv, pstride, n, nrhs, batch,   \
                                                     wg, nb, conj);                  \
        return fused_launch_notrans<T, NRV>(ctx, A, lda, sA, B, ldb, sB,             \
                                            piv, pstride, n, nrhs, batch, wg, nb);
    if (nrhs <= 1) { BATCHLAS_GETRS_FUSED_ARM(1) }
    if (nrhs <= 2) { BATCHLAS_GETRS_FUSED_ARM(2) }
    if (nrhs <= 4) { BATCHLAS_GETRS_FUSED_ARM(4) }
    BATCHLAS_GETRS_FUSED_ARM(8)
    #undef BATCHLAS_GETRS_FUSED_ARM
}

}  // namespace

// ---------------------------------------------------------------------------
// THE CAPABILITY FLAG. TRUE for all four types.
//
// DEFINED HERE, beside the kernel, for potrf_native.hh:81-92's reason: these are
// full explicit specialisations and link from wherever they sit, so co-locating
// them is what makes "the flag is true" and "the file is compiled" the same fact.
// ---------------------------------------------------------------------------
template <> bool getrs_fused_available<float>()                { return true; }
template <> bool getrs_fused_available<double>()               { return true; }
template <> bool getrs_fused_available<std::complex<float>>()  { return true; }
template <> bool getrs_fused_available<std::complex<double>>() { return true; }

// ---------------------------------------------------------------------------
// THE CAPACITY, IN RHS ELEMENTS (n * nrhs).
//
// The RHS VECTOR IS RESIDENT, so n * nrhs * sizeof(T) plus the diagonal block is
// a HARD CEILING: above it the kernel does not launch, which makes this a
// supports() question and not a preferred() one.
//
// ASKED OF THE DEVICE, never of a constant, for route_potrf.hh:114-127's reason.
// In particular it must NOT come from build/include/batchlas/device_limits.hh,
// whose 49152 is hardcoded by cmake/BatchLASDetectSYCL.cmake:44-45 for any
// nvidia_gpu_sm_* pattern and is 2.06x wrong on this box (local_mem_size is
// 101,376 B here).
//
// The largest nb the tier ever uses is charged, not the nb this call would pick:
// a capacity that depended on n would have to be re-derived by every caller that
// wants to compare against it, and charging the maximum can only be conservative.
//
// THE HOLE PAD IS APPLIED HERE TOO, and the inversion is not the obvious one.
// getrs_hole_padded is NOT monotone, so the largest admissible request is
//   budget                       when budget >  kGetrsHoleHi
//   min(budget, kGetrsHoleLo)    otherwise
// -- because a request landing inside the band is raised to kGetrsHolePadTo and
// then fails. Getting this backwards would advertise a capacity whose launch
// aborts, which is exactly what potrf_cta.cc:445-470's `break` exists to prevent.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrs_fused_max_rhs_elems(std::size_t slm_budget_bytes) {
    using D = typename sycl_device::DevMap<T>::type;
    const std::size_t admissible =
        (slm_budget_bytes > kGetrsHoleHi) ? slm_budget_bytes
                                          : std::min(slm_budget_bytes, kGetrsHoleLo);
    const std::size_t blk_bytes =
        static_cast<std::size_t>(kGetrsFusedNbMax) *
        static_cast<std::size_t>(getrs_fused_blk_ld(kGetrsFusedNbMax)) * sizeof(D);
    if (admissible <= blk_bytes) return 0;
    const std::size_t elems = (admissible - blk_bytes) / sizeof(D);

    // AND THE FLOOR DIVISION IS NOT THE END OF IT, because getrs_hole_padded is
    // not monotone in the other direction either. When the budget sits just above
    // kGetrsHoleHi the division can round the implied request BACK DOWN INTO the
    // band -- a cdouble budget of 49,665 B admits 2048 elements, whose raw request
    // is exactly 49,664 B, which is then RAISED to 49,920 and no longer fits. The
    // window is only sizeof(D) bytes wide and needs a device whose local memory is
    // 53,761-53,776 B, so nothing reachable on this box lands in it; it is closed
    // anyway because the failure mode is a capacity supports() promises and the
    // launch then refuses, which is exactly what potrf_cta.cc:445-470's `break`
    // exists to prevent -- and because a TEST can ask this pure function about any
    // budget it likes. tests/getrf_tests.cc's FusedGetrsLaunchHoleAt48KiB sweeps
    // the whole band one byte at a time, and that sweep is what found this.
    //
    // The repair is exact rather than a decrement loop: everything below the band
    // is admissible, so the largest safe request is the one that ends AT
    // kGetrsHoleLo.
    if (getrs_fused_slm(elems, kGetrsFusedNbMax, sizeof(D)) > slm_budget_bytes) {
        if (kGetrsHoleLo <= blk_bytes) return 0;
        return (kGetrsHoleLo - blk_bytes) / sizeof(D);
    }
    return elems;
}

// ---------------------------------------------------------------------------
// WORKSPACE. ZERO, and it is a consequence of the design rather than a
// coincidence: the RHS is permuted and solved IN LOCAL MEMORY and written back
// in place, so there is no out-of-place buffer, no collapsed permutation array,
// and no scratch of any kind. Nothing is dereferenced -- a measuring pass hands
// this null data pointers.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrs_fused_buffer_size(Queue&,
                                    const MatrixView<T, MatrixFormat::Dense>&,
                                    const MatrixView<T, MatrixFormat::Dense>&,
                                    Transpose) {
    return 0;
}

// ---------------------------------------------------------------------------
// THE DRIVER.
//
// Every gate RouteTable<Op::getrs,T>::supports() applies is RE-APPLIED here,
// because this entry point is reachable WITHOUT the table -- and it must be, for
// potrf_native.hh:126-141's reason: route_resolve.hh:165 falls through to
// automatic() when a forced route is unsupported, so a pinned-route test that is
// wrong about one gate silently measures cuBLAS and passes green over a kernel
// nothing executed.
// ---------------------------------------------------------------------------
template <typename T>
Event getrs_fused_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           Transpose transA,
                           Span<int64_t> pivots,
                           Span<std::byte> workspace) {
    static_cast<void>(workspace);   // this tier needs none; see the query above

    const int n = static_cast<int>(A.rows());
    const int nrhs = static_cast<int>(B.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (n < 1 || nrhs < 1 || batch < 1) {
        throw std::invalid_argument("getrs_fused: degenerate extents");
    }
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("getrs_fused: A must be square");
    }
    if (A.rows() != B.rows()) {
        throw std::invalid_argument("getrs_fused: B must have A.rows() rows");
    }
    if (A.batch_size() != B.batch_size()) {
        throw std::invalid_argument("getrs_fused: A and B must agree on batch size");
    }
    if (A.is_heterogeneous() || B.is_heterogeneous()) {
        throw std::invalid_argument("getrs_fused: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrs_fused: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32 -- that
        // property returns sub_group_sizes()[0] (queue-impl.cc:325), so the weak
        // test refuses a {8,16,32} device and ACCEPTS a {64} one. This kernel
        // carries [[sycl::reqd_sub_group_size(32)]] and its block solve is a
        // 32-lane shuffle recurrence, so a {64} device is a launch abort.
        throw std::runtime_error(
            "getrs_fused: device does not offer sub-group size 32");
    }
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrs_fused: pivot span is shorter than n * batch");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const std::size_t need = static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs);
    if (need > getrs_fused_max_rhs_elems<T>(budget)) {
        throw std::invalid_argument(
            "getrs_fused: n * nrhs = " + std::to_string(need) +
            " exceeds this device's resident-RHS capacity (" +
            std::to_string(getrs_fused_max_rhs_elems<T>(budget)) +
            " elements). This is a CAPACITY ceiling, not a speed one: route the "
            "call to Algorithm::Blocked instead.");
    }
    if (nrhs > kGetrsFusedMaxRhs) {
        throw std::invalid_argument(
            "getrs_fused: nrhs = " + std::to_string(nrhs) + " is above the widest "
            "instantiated accumulator (" + std::to_string(kGetrsFusedMaxRhs) +
            "). Route to Algorithm::Blocked.");
    }

    // PACKED 1-BASED int32 -- the format cublas.cc:1476 and rocsolver.cc:227 both
    // read through pivots.as_span<int>(), and the one a native getrf writes. See
    // getrf_native.hh's PIVOT CONTRACT: the ops have independent env variables and
    // independent preferred() windows, so every mixture of native and vendor arms
    // is reachable and they must agree bit for bit.
    auto piv_i32 = pivots.as_span<int>();

    // `trans` is hoisted rather than spelled inline at the call below because the
    // REGISTER CAP needs it: the transposed body carries a sub-group reduction per
    // right-hand side and is up to 20 registers wider than the NoTrans one at the
    // same width, so a cap that could not see which body it was sizing had to
    // charge every launch the widest kernel in the file.
    const bool trans = (transA != Transpose::NoTrans);

    const int nb = getrs_fused_nb(n);
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int wg = getrs_fused_wg<T>(n, nrhs, max_wg, trans);

    return fused_dispatch_nr<T>(
        ctx,
        trans,
        /*conj=*/transA == Transpose::ConjTrans,
        A.data_ptr(), A.ld(), A.stride(),
        B.data_ptr(), B.ld(), B.stride(),
        piv_i32.data(), /*pstride=*/n,
        n, nrhs, batch, wg, nb);
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY. This tier injects nothing -- unlike the
// composed one, which takes the routed trsm -- so there is no Backend to cross.
// ---------------------------------------------------------------------------
#define BATCHLAS_GETRS_FUSED_INSTANTIATE(T)                                                \
    template std::size_t getrs_fused_max_rhs_elems<T>(std::size_t);                        \
    template std::size_t getrs_fused_buffer_size<T>(                                       \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose);                             \
    template Event getrs_fused_dispatch<T>(                                                \
        Queue&, const MatrixView<T, MatrixFormat::Dense>&,                                 \
        const MatrixView<T, MatrixFormat::Dense>&, Transpose,                              \
        Span<int64_t>, Span<std::byte>);

BATCHLAS_GETRS_FUSED_INSTANTIATE(float)
BATCHLAS_GETRS_FUSED_INSTANTIATE(double)
BATCHLAS_GETRS_FUSED_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRS_FUSED_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRS_FUSED_INSTANTIATE

}  // namespace sycl_getrs
}  // namespace batchlas
