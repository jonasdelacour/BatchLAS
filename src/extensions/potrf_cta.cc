// Native batched POTRF, Phase 1: the CTA kernel's LAUNCHER and capability
// surface. All of the device code is in potrf_cta_device.hh.
//
// This TU sits in EXTENSIONS_CTA_SOURCES (src/extensions/CMakeLists.txt), which
// is W12: a source must sit with the sources whose device symbols it calls, and
// the blocked driver's diagonal leaf IS potrf_cta_body -- so potrf_blocked.cc
// sits in that same list. Splitting a device-code
// cluster across libraries is a hard `ptxas fatal: Unresolved extern function`
// today and, while every helper stays an inline template, a latent one: the link
// succeeds by duplicating the body into both TUs, and breaks the first time
// anyone marks one SYCL_EXTERNAL to stop duplicating it.
//
// ---------------------------------------------------------------------------
// W10 RESOLVED -- Scope is DERIVED, never asserted
// ---------------------------------------------------------------------------
// docs/perf/potrf.md#what-the-spec-got-wrong:225 says the blocked leaf runs "at Scope::SubGroup with G
// matrices per work-group". Its own L ladder at :189-195 contradicts that for
// float: at the spec's outer width NB_o = 64 the first trailing update has
// Ntiles_0 = 78 > 64, so the ladder returns L = 64, hence G = 1, hence
// Scope::WorkGroup. Obeying :225 there would make the four phase barriers
// sub-group barriers across a 64-work-item matrix -- two sub-groups on one tile
// with nothing between them, i.e. exactly the race docs/perf/potrf.md#what-the-spec-got-wrong:210 exists
// to close, producing a plausible wrong factor with no crash.
//
// The resolution is structural: potrf_cta_launch_params below COMPUTES the
// scope from L and the invariant is a static/runtime assertion, so no caller can
// assert a scope that the ladder disagrees with. Phase 2's leaf will call this
// same function.
//
// ---------------------------------------------------------------------------
// WHAT IS DELIBERATELY NOT HERE
// ---------------------------------------------------------------------------
// * No runtime `nb` ladder. docs/perf/potrf.md#what-the-spec-got-wrong:238 has one; NB is a compile-time
//   constant per scalar type here, so nb == NB always and `ib = min(NB, n-j)`
//   carries the ragged last panel. A runtime nb is a TUNING knob and nothing
//   about potrf has been measured; adding a knob whose settings are unmeasured
//   multiplies the instantiation count of a device-link-bound build for a
//   hypothesis. The falsification set for step 1.7 is (float NB = 32) and
//   (complex<double> TS = 4), both one-line changes here.
// * No `BATCHLAS_POTRF_UPDATE=herk` oracle swap (docs/perf/potrf.md#what-the-spec-got-wrong:2.5). The
//   tests' oracle is a host multiply-back residual, which is independent of
//   every other implementation in this tree; a device::herk A/B would compare
//   the kernel against another BatchLAS path.
// * No blocked driver IN THIS FILE. It is WP4 Phase 2 and lives in
//   src/extensions/potrf_blocked.cc, which calls potrf_cta_dispatch below as
//   its diagonal-block leaf, handed a sub-view.

#include "potrf_native.hh"
#include "potrf_cta_device.hh"

#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {

// The kernel name tag. Outside the anonymous namespace so it does not depend on
// an internal-linkage entity.
template <typename T, int NB, int TS, potrf_native::PotrfScope SC>
class PotrfCtaKernel;

namespace sycl_potrf {

namespace {

using potrf_native::PotrfScope;

// ---------------------------------------------------------------------------
// The (NB, TS) ladder, per scalar type. MEASURED, and it moved twice.
//
// TS is the (P3) thread tile; NB is the panel width and the length of the two
// register arrays d[NB] (P1) and x[NB] (P2). The predicted array cost, in 32-bit
// units, is max(NB*sizeof(T)/4, TS^2*sizeof(T)/4 + 2*TS*sizeof(T)/4) -- a max
// and not a sum, because (P2)'s x[] and (P3)'s acc[]/va[]/vb[] live ranges are
// disjoint. That prediction did not set this table; the register probe did
// (scripts/register_probe.sh's mechanism replayed on THIS library by
// docs/perf/potrf.md#register-gate -- the stock script
// hardcodes batchlas_sycl.dir/link.txt, which does not contain this TU and would
// have reported a clean result for code it never compiled).
//
// THE MEASUREMENTS, registers as SubGroup/WorkGroup, then the stack frame, with
// ZERO spill stores and ZERO spill loads throughout:
//
//   T                 NB  TS   (P1) loop   regs        frame
//   float             16   4   not unrolled  110 / 106     0
//   double            16   4   not unrolled  144 / 128   128   <- d[16], in .local
//   complex<float>    16   4   not unrolled  201 / 188   128   <- d[16], in .local
//   double             8   4   not unrolled   94 /  80     0
//   complex<float>     8   4   not unrolled   99 /  92     0
//   float             16   4   UNROLLED      156 / 154     0
//   double            16   4   UNROLLED      172 / 150     0
//   complex<float>    16   4   UNROLLED      206 / 167     0
//   float              8   4   UNROLLED       64 /  56     0   <- SHIPPED
//   double             8   4   UNROLLED       94 /  80     0   <- SHIPPED
//   complex<float>     8   4   UNROLLED      102 /  92     0   <- SHIPPED
//   complex<double>    8   2   UNROLLED      128 / 109     0   <- SHIPPED
//
// Two things that table says, neither of which was predicted:
//
//  1. THE 128-BYTE FRAME WAS d[16], NOT acc[TS][TS], and its cause was the
//     `break` in (P1)'s loop. A break makes the trip count data-dependent,
//     `#pragma unroll` fails with -Wpass-failed=transform-warning, and d[k] /
//     d[c] acquire a dynamic index -- an array in .local with ZERO reported
//     spill, which is exactly why the gate for this kernel is the
//     three-condition one (frame == 0 AND 0 spill AND regs*WG <= 65536) and not
//     register_probe.sh's stale two-condition header
//     (docs/perf/trsm.md#what-the-spec-got-wrong:136-160). Two other explanations were tested
//     and REFUTED first: routing fma_acc's by-reference accumulator through a
//     scalar temporary (this tree's recorded 43% out-parameter spill) gave a
//     byte-identical report, and dropping double to TS = 2 (acc[2][2] = 32 B)
//     left the frame at exactly 128. potrf_cta_device.hh predicates instead of
//     breaking; the warnings are gone.
//
//  2. UNROLLING IS NOT FREE, AND NB = 8 IS THE CHEAPER HALF OF THE TRADE.
//     Unrolled NB = 16 clears the gate for every type but costs 156-206
//     registers; unrolled NB = 8 costs 64-128. At the work-group ceiling of 128
//     that is floor(65536/(regs*128)) = 3 resident blocks against 8 for float --
//     and occupancy, not the local-memory cap, is what binds these CTA kernels
//     (the fit ceiling itself is 1 block/SM). float at 64 registers also lands
//     exactly on an occupancy step edge rather than one past it.
//
// The fit ceilings are UNCHANGED by any of this (155/109/109/77): NB enters
// slm_per_matrix only through the small diag[] and off[] terms, which move in
// opposite directions. tests/potrf_tests.cc's MeasuredFitCeilings pins them.
//
// NB = 16 WAS MEASURED AND IS WORSE. THIS IS SETTLED, NOT OPEN.
//
// The open question used to read: NB = 8 doubles the panel count and with it
// (P1)'s serial critical path and the barrier count, and nothing measured which
// way that trade fell. The adversarial review then argued from ncu that the
// register cost of NB = 16 is FREE above n ~ 32 -- correctly, as far as it goes:
// launch__occupancy_limit_shared_mem vs launch__occupancy_limit_registers is
// 2 vs 18 blocks/SM at float n = 96 and 1 vs 9 at n = 155, so shared memory
// binds and the extra registers buy nothing back. It predicted a time win.
//
// THE PREDICTION IS REFUTED. Built with NB = 16 for float, double and
// complex<float> (complex<double> held at 8), tests green, timed at batch = 4096
// under gpu_guard against the identical NB = 8 build. NB = 16 is slower in 18 of
// 20 cells -- native microseconds, NB=8 -> NB=16:
//
//   float   n=  8    7.7 ->   20.8   (2.70x WORSE)
//   float   n= 16   16.6 ->   30.7   (1.85x)
//   float   n= 32   43.8 ->   67.9   (1.55x)
//   float   n= 48  117.1 ->  153.3   (1.31x)
//   float   n= 96   1162 ->   2374   (2.04x WORSE)
//   float   n=155   4046 ->   4258   (1.05x)
//   double  n= 96   4938 ->   5400   (1.09x)
//   cfloat  n=  8   12.0 ->   26.4   (2.20x)
//
// The only cell that improved is double n = 16 (142.7 -> 131.6, 1.08x). The
// halved panel count does not pay for the registers, and at small n -- where
// registers DO bind, and where this kernel is the only thing that beats
// cuSOLVER -- it is catastrophic. The register gate and the clock agree, which
// is the outcome that needed checking because they so often do not.
//
// So NB = 8 is now a MEASURED choice rather than a gate-driven one, and the
// review's proposal to instantiate both NB and pick at launch is declined: it
// would double potrf's device instantiations from 8 to 16 in a build this
// repository documents as device-link-bound, to select between a winner and a
// loser.
template <typename T>
struct PotrfCtaConst;
template <> struct PotrfCtaConst<float>                { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<double>               { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<std::complex<float>>  { static constexpr int NB = 8;  static constexpr int TS = 4; };
template <> struct PotrfCtaConst<std::complex<double>> { static constexpr int NB = 8;  static constexpr int TS = 2; };

// The standard per-work-group local-memory budget: the RUNTIME local_mem_size
// minus the 4096 B reserve that cmake/BatchLASDetectSYCL.cmake:57-67 applies to
// every other device-BLAS sizing decision in this library.
//
// 97,280 on this box. NOT build/include/batchlas/device_limits.hh's 49152: that
// number is hardcoded by cmake/BatchLASDetectSYCL.cmake:44-45 for any
// nvidia_gpu_sm_* pattern and is wrong here by 2.06x (W1).
constexpr std::size_t kPotrfReferenceSlmBudget = 97280;

// The soft occupancy target that decides how many matrices share a work-group.
// A work-group spending this much is 4 resident blocks per SM on sm_89
// (102,400 B of shared memory per SM, reservedSharedMemPerBlock = 1024).
constexpr std::size_t kPotrfSlmSoftTarget = 24576;

// The L ladder's two knobs. See the measured grid in potrf_cta_launch_params.
//
// kPotrfMaxL is 256 and not 128 because the register gate has room for it:
// the widest instantiation is complex<double> at 109 registers under
// Scope::WorkGroup, and 109 * 256 = 27,904, which is 2.3x under the 65,536
// ceiling. It is shared memory, not registers or threads, that limits these
// launches (measured: 9-18x more restrictive in ncu's occupancy limiters), so
// widening the work-group buys warps at no cost in resident blocks.
constexpr int kPotrfMaxL = 256;
constexpr int kPotrfElemsPerItem = 24;

// ---------------------------------------------------------------------------
// THE SLM FORMULA. One function, called by BOTH the capability query and the
// launcher, so the ceiling supports() advertises and the allocation the kernel
// makes cannot disagree.
//
//   tile   lda * n * sizeof(D)   with lda = n | 1.  Odd, so a stride-lda row
//                                read is conflict-free.
//   diag   NB * sizeof(R)        the real pivot diagonal, consumed by (P2)
//   256                          *fail plus the alignment slack the runtime
//                                inserts between four separate local_accessors.
//                                A deliberate over-estimate: this figure is
//                                what the fit ceiling is computed from, so it
//                                must never be smaller than what the launch
//                                actually asks for.
//
//                                IT WAS 64, AND 64 WAS TOO SMALL -- MEASURED,
//                                not argued. At the advertised float ceiling
//                                (n = 155, G = 1) the raw accessor sum is
//                                96,100 + 32 + 4 + 152 = 96,288 B, the formula
//                                with the old term said 96,348, and ncu reports
//                                launch__shared_mem_per_block_dynamic = 96,408
//                                (static = 0). So the launch asked for 60 B MORE
//                                than the number potrf_cta_max_n_for_slm(),
//                                RouteTable::supports() and p.fits are all
//                                computed from -- the invariant three lines up,
//                                inverted. The measured runtime overhead is
//                                120 B for these four accessors; 256 covers it
//                                with margin and is ceiling-neutral for all four
//                                types (155/109/109/77 are unchanged, which
//                                MeasuredFitCeilings pins).
//
//                                On this box the unrelated 4,096 B reserve left
//                                872 B of headroom, so nothing failed here. On a
//                                device whose budget lands within ~120 B above
//                                the formula value at its ceiling, supports()
//                                would advertise an order whose launch requests
//                                more local memory than the device allows, and
//                                it would arrive as a CUDA_ERROR_INVALID_VALUE
//                                at enqueue rather than as the documented throw
//                                -- the exact false-positive class supports()
//                                exists to exclude.
//   off    4 * (Rt0 + 1)         the (P3) tile-index prefix table, per matrix.
//                                W9's missing term. Rt0 is the tile-row count of
//                                the FIRST trailing update -- the largest over
//                                all panels -- and the +1 is the sentinel.
// ---------------------------------------------------------------------------
constexpr std::size_t potrf_slm_per_matrix(int n, int NB, int TS,
                                           std::size_t sz_d, std::size_t sz_r) {
    const std::size_t lda = static_cast<std::size_t>(n | 1);
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;
    return lda * static_cast<std::size_t>(n) * sz_d
         + static_cast<std::size_t>(NB) * sz_r
         + 256
         + 4 * static_cast<std::size_t>(Rt0 + 1);
}

// ---------------------------------------------------------------------------
// THE 48 KB LAUNCH HOLE, and the pad that steps over it.
//
// Measured cold on this box (docs/perf/potrf.md#the-48-kb-launch-hole):
// a dynamic local-memory request in (49152 - static_shared, 49152] fails with
// CUDA_ERROR_INVALID_VALUE at enqueueKernelLaunch -- too big for CUDA's
// non-opt-in 48 KB limit once the kernel's static shared is added, not big
// enough for the UR adapter to raise MaxDynamicSharedMemorySize. It is
// ORDER-DEPENDENT: the attribute is sticky per CUfunction and one n serves every
// n here, so it reproduces only on the first launch of a process, which is
// exactly how it would escape a test suite that runs several sizes in one
// process. Verified cold at 49,064 (fail) and 49,408 (pass).
//
// The band below is deliberately about +-2 KB rather than the 256 B the probe
// kernel measured, and padding UP costs occupancy only at float n in ~108..111,
// which are already at 1-2 resident blocks per SM.
//
// IT IS INERT TODAY, AND THAT WAS MEASURED RATHER THAN ASSUMED. The hole's width
// EQUALS the kernel's static shared memory, and ptxas reports NO `smem` field at
// all for any of the eight potrf instantiations -- they have zero static shared,
// so (49152 - 0, 49152] is the empty interval. Disabling this function and
// re-running the residual sweep, which now includes float n = 108..111 (n = 110
// asks for 49,044 B, squarely inside the measured hole) after nothing but
// sub-48 KB launches so the sticky per-CUfunction attribute is still low, passed
// green. That is consistent with step 0.2's own zero-static control, which had
// no hole either.
//
// It stays because static shared is not something this source controls: one
// group algorithm added anywhere in the body -- a reduce_over_group, which is
// exactly what the probe kernel used -- reintroduces the hole, and the failure
// mode is a cold-start CUDA_ERROR_INVALID_VALUE that a warm test suite cannot
// see.
constexpr std::size_t kPotrfHoleLo = 47104;
constexpr std::size_t kPotrfHoleHi = 49664;
constexpr std::size_t kPotrfHolePadTo = 49920;

constexpr std::size_t potrf_hole_padded(std::size_t bytes) {
    return (bytes > kPotrfHoleLo && bytes <= kPotrfHoleHi) ? kPotrfHolePadTo : bytes;
}

inline int prev_pow2(int v) {
    int r = 1;
    while ((r << 1) <= v) r <<= 1;
    return r;
}

// ---------------------------------------------------------------------------
// Everything the launch needs, computed once. See the W10 note at the top: this
// is where Scope comes from, and it is the only place.
// ---------------------------------------------------------------------------
struct PotrfCtaLaunch {
    int L = 32;               // work-items per matrix
    int G = 1;                // matrices per work-group; > 1 only when L == 32
    int wg_size = 32;
    int num_wg = 0;
    int lda = 1;
    int Rt0 = 0;
    std::size_t slm_per_matrix = 0;
    std::size_t slm_total = 0;   // G * slm_per_matrix, after the hole pad
    PotrfScope scope = PotrfScope::SubGroup;
    bool fits = false;
};

template <int NB, int TS>
PotrfCtaLaunch potrf_cta_launch_params(int n, int batch, std::size_t sz_d, std::size_t sz_r,
                                       std::size_t slm_budget, int max_wg) {
    PotrfCtaLaunch p;
    p.lda = n | 1;
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    p.Rt0 = (m2_0 + TS - 1) / TS;
    const long long Ntiles_0 = static_cast<long long>(p.Rt0) * (p.Rt0 + 1) / 2;

    // [FIX-A2.1] L is derived from m2_0 = n - NB, the FIRST TRAILING UPDATE --
    // not from n. Sizing it from ceil(n/TS) counts a triangle that is never
    // updated and guarantees Ntiles < L in every panel, which makes the
    // anti-starvation argument arithmetically backwards.
    //
    // -----------------------------------------------------------------------
    // THE LADDER IS ELEMENTS PER WORK-ITEM, NOT TILES PER WORK-ITEM, AND THE
    // DIFFERENCE WAS MEASURED.
    // -----------------------------------------------------------------------
    // This was `(Ntiles_0 <= 64) ? 32 : (Ntiles_0 <= 256) ? 64 : 128`, justified
    // by "L = 256 costs 1536/256 = 6 blocks/SM for tile counts that do not
    // materialise". That is a THREAD-limit argument, and ncu says the thread
    // limit is never what binds: launch__occupancy_limit_shared_mem vs
    // launch__occupancy_limit_registers is 2 vs 18 blocks/SM at float n = 96 and
    // 1 vs 9 at n = 128 and n = 155. Shared memory binds by 9-18x, so the
    // work-group could be twice as wide at no cost in resident blocks -- and at
    // n >= 96 the kernel was running 2 blocks x 2 warps = 4 warps/SM, an ncu
    // sm__warps_active of 8.31%, where the hardware had room for 8-16.
    //
    // A TILE COUNT IS ALSO THE WRONG UNIT, because a tile is TS x TS and TS is
    // not constant across the type ladder. complex<double> runs TS = 2, so for
    // the same order it has 4x the tiles of float at 1/4 the work each; a
    // tile-count rule over-shoots it by two rungs. Counting the ELEMENTS the
    // trailing update touches, Ntiles_0 * TS * TS, is TS-independent and fixes
    // that by construction rather than by a per-type exception.
    //
    // MEASURED, batch = 4096, gpu_guard, JIT-warmed, native microseconds
    // (docs/perf/potrf.md#cta-kernel-measured-against-cusolver carries the full grid; * marks the pick):
    //
    //   float    n= 48  L32 131.9  L64 117.2*  L128 120.8   L256 240.7
    //   float    n= 64  L32 318.2  L64 274.5   L128 242.6*  L256 355.5
    //   float    n= 96  L32 2128   L64 1482    L128 1171    L256 1161*
    //   float    n=128  L32 7329   L64 4577    L128 3224    L256 3053*
    //   float    n=155  L32 11301  L64 6955    L128 4838    L256 4049*
    //   double   n= 96  L32 7481   L64 5976    L128 5020    L256 4812*
    //   cdouble  n= 64  L32 6826   L64 5086    L128 4711*   L256 4859
    //
    // Across 21 measured cells (4 types x n in {8,16,32,48,64,96,128,155} where
    // the type reaches it) this rule picks the measured best or a cell within
    // 1% of it in 19, and is within 5.5% in the other two (complex<float> at
    // n = 64 and complex<double> at n = 32). The old ladder cost up to 1.27x
    // (float n = 96) and 1.19x (float n = 155).
    //
    // 24 is elements per work-item, and it is a FITTED CONSTANT -- the one thing
    // on this line that is not derived. It is pinned by the grid above and by
    // nothing else; re-measure it if NB, TS or the (P3) inner loop changes.
    //
    // THIS MOVES NO USER-VISIBLE TRAFFIC. preferred() is false everywhere
    // (route_potrf.hh), so in a vendor-present build nothing routes here at all.
    // It is the vendor-free build -- the build this work package exists for --
    // that gets the 1.2-1.3x, and that build is still LOSING to cuSOLVER above
    // n ~ 64 by 2-3x. See the honest table in docs/perf/potrf.md#cta-kernel-measured-against-cusolver:
    // this is a real improvement to a kernel that is still far from good.
    {
        const long long work_elems = Ntiles_0 * static_cast<long long>(TS) * TS;
        int want = 32;
        while (want < kPotrfMaxL &&
               static_cast<long long>(want) * kPotrfElemsPerItem < work_elems) {
            want <<= 1;
        }
        p.L = want;
    }
    while (p.L > 32 && p.L > max_wg) p.L >>= 1;

    p.slm_per_matrix = potrf_slm_per_matrix(n, NB, TS, sz_d, sz_r);

    // G > 1 only at L == 32, and capped so wg_size <= 128 rather than the
    // spec:193 ladder's 256. Arithmetic, at an ASSUMED 96 registers/thread:
    // float n = 16 with G = 8 (wg 256) is register-limited to 65536/(96*256) = 2
    // blocks/SM = 16 matrices/SM, against G = 4 (wg 128) at 5 blocks/SM = 20
    // matrices/SM. Reopen 256 only if the register probe reports <= 64.
    if (p.L == 32 && p.slm_per_matrix > 0) {
        const std::size_t target = std::min(kPotrfSlmSoftTarget, slm_budget);
        const int by_slm = static_cast<int>(target / p.slm_per_matrix);
        p.G = std::clamp(prev_pow2(std::max(1, by_slm)), 1, 4);
        while (p.G > 1 && (p.G * p.L > max_wg ||
                           static_cast<std::size_t>(p.G) * p.slm_per_matrix > slm_budget)) {
            p.G >>= 1;
        }
    } else {
        p.G = 1;
    }

    p.wg_size = p.G * p.L;
    p.num_wg = (batch + p.G - 1) / p.G;
    p.scope = (p.L == 32) ? PotrfScope::SubGroup : PotrfScope::WorkGroup;
    p.slm_total = potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix);
    p.fits = (p.slm_total <= slm_budget) && (p.wg_size <= max_wg);

    // The invariant W10's contradiction violates, in code rather than in prose.
    // Under Scope::WorkGroup the phase barriers are work-group barriers, which
    // is only correct when the work-group holds exactly one matrix.
    if (p.scope == PotrfScope::WorkGroup && p.G != 1) {
        throw std::logic_error("potrf_cta: Scope::WorkGroup with G != 1 is a race by construction");
    }
    return p;
}

}  // namespace

// ---------------------------------------------------------------------------
// The capability queries.
// ---------------------------------------------------------------------------
template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    constexpr std::size_t sz_d = sizeof(typename DM::type);
    constexpr std::size_t sz_r = sizeof(typename DM::real);

    // Monotone in n, so a linear walk is exact and needs no closed form to be
    // audited. 4096 is a bound, not a capability: no budget on any device makes
    // an n^2 tile of that order resident.
    //
    // THE PAD IS APPLIED HERE TOO, AND THAT IS THE POINT. The launcher gates on
    // p.fits = potrf_hole_padded(G * slm_per_matrix) <= budget (see
    // potrf_cta_launch_params). Walking the RAW figure here made the ceiling
    // supports() advertises and the fit test the launch actually applies two
    // different predicates for the same question. Arithmetic, float, NB = 8,
    // TS = 4: at n = 111 slm_per_matrix is 49,680 B, which the pad raises to
    // 49,920. On a device whose budget (local_mem_size - 4096) landed in
    // [49680, 49920) the unpadded walk returned 111, supports() said true,
    // resolve_route handed the caller {Native, CTA}, and the facade's CTA arm
    // then threw std::invalid_argument from potrf_cta_dispatch -- an unhandled
    // throw in a vendor-free build on a call the route table had promised.
    // NOT REACHABLE ON THIS BOX (budget 97,280, far above the band), which is
    // exactly why it had to be closed by construction rather than by testing.
    //
    // THE `break` IS LOAD-BEARING AND MUST STAY, because potrf_hole_padded is
    // NOT MONOTONE: a raw figure just inside the band is padded ABOVE a larger
    // raw figure just outside it (47,200 -> 49,920, while 49,700 stays 49,700).
    // supports() spells the capacity `order <= cta_max_n`, i.e. a CONTIGUOUS
    // range, so the ceiling must be the largest n for which EVERY order up to n
    // launches -- which is what stopping at the first miss returns. Scanning for
    // the largest fitting n instead would advertise a ceiling with a hole under
    // it and hand a caller an order that throws. The cost is under-advertising
    // by a few orders on a device whose budget falls inside the band, and
    // under-advertising only ever costs a route, never an answer.
    int best = 0;
    for (int n = 1; n <= 4096; ++n) {
        if (potrf_hole_padded(potrf_slm_per_matrix(n, C::NB, C::TS, sz_d, sz_r))
            > slm_budget_bytes) break;
        best = n;
    }
    return best;
}

template <typename T>
int potrf_cta_max_n() {
    return potrf_cta_max_n_for_slm<T>(kPotrfReferenceSlmBudget);
}

// potrf_blocked_available<T>() IS NOT HERE. It moved to
// src/extensions/potrf_blocked.cc with WP4 Phase 2, so that "the flag is true"
// and "the driver is compiled into this build" are the same fact; see
// potrf_native.hh for why that placement is load-bearing.

// ---------------------------------------------------------------------------
// Workspace.
// ---------------------------------------------------------------------------
namespace {

template <typename T>
Span<int32_t> potrf_cta_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
std::size_t potrf_cta_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = A.batch_size();
    return workspace_bytes([&](BumpAllocator& p) {
        return potrf_cta_layout<T>(ctx, p, batch);
    });
}

// The launch geometry, for tests. See potrf_native.hh for why this exists.
template <typename T>
unsigned potrf_cta_debug_launch(Queue& ctx, int n, int batch) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    const auto dev = ctx.device();
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const auto p = potrf_cta_launch_params<C::NB, C::TS>(
        n, batch, sizeof(typename DM::type), sizeof(typename DM::real), budget, max_wg);
    if (!p.fits) return 0u;
    return (static_cast<unsigned>(p.L) << 16) | static_cast<unsigned>(p.G);
}

// ---------------------------------------------------------------------------
// The launch.
// ---------------------------------------------------------------------------
namespace {

template <typename T, int NB, int TS, PotrfScope SC>
Event potrf_cta_launch(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       bool upper,
                       Span<int32_t> info,
                       const PotrfCtaLaunch& p,
                       int n, int batch) {
    // The whole kernel runs on the POD device scalar. std::complex is re-typed
    // HERE, at the pointer boundary, and never crosses into the kernel body:
    // its operator* is Annex-G conformant, which means an isnan branch and a
    // __mulsc3 / __muldc3 CALL in the inner loop.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* a_ptr = reinterpret_cast<D*>(A.data_ptr());
    const int ldg = A.ld();
    const int stride_a = A.stride();
    const int lda = p.lda;
    const int Rt0 = p.Rt0;
    const int G = p.G;
    const int L = p.L;
    const int wg_size = p.wg_size;
    const int num_wg = p.num_wg;
    int32_t* info_ptr = info.data();

    // Pad the TILE accessor rather than adding a fifth, unused one: an unused
    // local_accessor is a plausible dead-code elimination and the pad has to
    // reach the launch to do its job.
    const std::size_t tile_elems_used = static_cast<std::size_t>(G) *
                                        static_cast<std::size_t>(lda) * static_cast<std::size_t>(n);
    const std::size_t natural = static_cast<std::size_t>(G) * p.slm_per_matrix;
    const std::size_t pad_bytes = (p.slm_total > natural) ? (p.slm_total - natural) : 0;
    const std::size_t tile_elems = tile_elems_used + (pad_bytes + sizeof(D) - 1) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_elems), h);
        sycl::local_accessor<R, 1> diag(sycl::range<1>(static_cast<std::size_t>(G) * NB), h);
        sycl::local_accessor<int, 1> fail(sycl::range<1>(static_cast<std::size_t>(G)), h);
        sycl::local_accessor<int, 1> off(
            sycl::range<1>(static_cast<std::size_t>(G) * static_cast<std::size_t>(Rt0 + 1)), h);

        h.parallel_for<PotrfCtaKernel<T, NB, TS, SC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) * wg_size),
                              sycl::range<1>(wg_size)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int wg_id = static_cast<int>(it.get_group_linear_id());

                int matrix_id;
                int slot;
                int tid;
                bool p1_active;
                if constexpr (SC == PotrfScope::SubGroup) {
                    const int sg_id = static_cast<int>(sg.get_group_linear_id());
                    matrix_id = wg_id * G + sg_id;
                    slot = sg_id;
                    tid = static_cast<int>(sg.get_local_linear_id());
                    p1_active = true;
                    // Site E of the barrier audit. Sub-group-uniform, and the
                    // sub-group's own barriers are the only ones this scope
                    // uses, so returning here strands nobody.
                    if (matrix_id >= batch) return;
                } else {
                    matrix_id = wg_id;   // G == 1 => num_wg == batch, cannot exceed
                    slot = 0;
                    tid = static_cast<int>(it.get_local_linear_id());
                    p1_active = (sg.get_group_linear_id() == 0);
                }

                D* S = &tile[0] + static_cast<std::ptrdiff_t>(slot) * lda * n;
                R* dg = &diag[0] + static_cast<std::ptrdiff_t>(slot) * NB;
                int* fl = &fail[0] + slot;
                int* of = &off[0] + static_cast<std::ptrdiff_t>(slot) * (Rt0 + 1);

                // The global base is built EXPLICITLY from data_ptr() + b*stride.
                // Never MatrixView::operator()(Slice,Slice): matrix.hh:1129-1141
                // carries a comment saying it must not propagate the parent
                // pointer array and the very next line does, and the 6-arg
                // constructor DEFAULTS stride to ld*cols when 0 is passed, after
                // which every batch item but the first reads the wrong matrix.
                D* Ag = a_ptr + static_cast<std::ptrdiff_t>(matrix_id) * stride_a;

                potrf_native::potrf_cta_body<D, R, NB, TS, SC>(
                    it, sg, tid, L, p1_active, S, lda, dg, fl, of, Ag, ldg, n, upper);

                // One writer per matrix. `fail` is published by B3 of the last
                // panel, which every work-item of the matrix has passed.
                if (tid == 0) info_ptr[matrix_id] = *fl;
            });
    });

    return ctx.get_event();
}

}  // namespace

template <typename T>
Event potrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Uplo uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info_out) {
    using C = PotrfCtaConst<T>;
    using DM = sycl_device::DevMap<T>;
    constexpr std::size_t sz_d = sizeof(typename DM::type);
    constexpr std::size_t sz_r = sizeof(typename DM::real);

    const int n = static_cast<int>(A.rows());
    const int batch = static_cast<int>(A.batch_size());

    // Every gate RouteTable<Op::potrf,T>::supports() applies, re-applied here
    // because this entry point is reachable without the table (the tests call it
    // directly, and they must: a forced route the table rejects silently falls
    // back to the vendor).
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("potrf_cta: A must be square");
    }
    if (n < 1 || batch < 1) {
        throw std::invalid_argument("potrf_cta: degenerate extents");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with a single (order, ld, stride) tuple and
        // reads at data_ptr() + b*stride with the CAPACITY extents, so a view
        // with per-item active dims would factorise the wrong order in place for
        // every item after the first. netlib's batched path honours the per-item
        // extents, so this is a disagreement with a path in this tree that gets
        // it right -- not a hypothetical.
        throw std::invalid_argument("potrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("potrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that
        // property returns sub_group_sizes()[0] (src/util/queue-impl.cc:325),
        // the FIRST supported size, so the weak test refuses a {8,16,32} device
        // and ACCEPTS a {64} one -- and the kernel below carries
        // [[sycl::reqd_sub_group_size(32)]], for which the second is a launch
        // abort.
        throw std::runtime_error(
            "potrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    const auto p = potrf_cta_launch_params<C::NB, C::TS>(n, batch, sz_d, sz_r, budget, max_wg);
    if (!p.fits) {
        throw std::invalid_argument(
            "potrf_cta: order " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(p.slm_total) + " B of " + std::to_string(budget) +
            " B); the ceiling for this type is " +
            std::to_string(potrf_cta_max_n_for_slm<T>(budget)));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined so this TU does not have to include
    // src/linalg-impl.hh: an empty or SHORT caller span means "not requested"
    // and falls back to pool scratch. Note the direction -- supplying a span
    // only ever removes a pool draw -- which is what keeps potrf_cta_buffer_size
    // correct in both modes.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : potrf_cta_layout<T>(ctx, pool, batch);

    const bool upper = (uplo == Uplo::Upper);

    if (p.scope == PotrfScope::SubGroup) {
        return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::SubGroup>(
            ctx, A, upper, info, p, n, batch);
    }
    return potrf_cta_launch<T, C::NB, C::TS, PotrfScope::WorkGroup>(
        ctx, A, upper, info, p, n, batch);
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. The kernel has
// no vendor dependency and no Backend parameter, so a 3x multiplication of a
// device-compiled family in a build that is device-link-bound is pure cost.
// trsm_native.cc:820-838 is the same shape.
// ---------------------------------------------------------------------------
#define BATCHLAS_POTRF_CTA_INSTANTIATE(T)                                                   \
    template int potrf_cta_max_n_for_slm<T>(std::size_t);                                   \
    template int potrf_cta_max_n<T>();                                                      \
    template unsigned potrf_cta_debug_launch<T>(Queue&, int, int);                          \
    template std::size_t potrf_cta_buffer_size<T>(Queue&,                                   \
                                                  const MatrixView<T, MatrixFormat::Dense>&); \
    template Event potrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&, \
                                         Uplo, Span<std::byte>, Span<int32_t>);

BATCHLAS_POTRF_CTA_INSTANTIATE(float)
BATCHLAS_POTRF_CTA_INSTANTIATE(double)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_POTRF_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_POTRF_CTA_INSTANTIATE

}  // namespace sycl_potrf
}  // namespace batchlas
