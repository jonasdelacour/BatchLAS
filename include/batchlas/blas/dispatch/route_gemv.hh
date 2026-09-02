#pragma once

// GEMV's routing table.
//
// PURE, in route_resolve.hh:19-21's sense: everything here reads ONLY its
// arguments -- no getenv, no SYCL query. Anything that has to ask the device or
// the environment lives in src/backends/gemv_route.hh instead.
//
// THE SPLIT RULE, restated from route_gemm.hh:25-28 and route_trsm.hh:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the alias table (route_env.hh), not here.
//
// For gemv the rule has teeth in one specific place, and it is the whole work
// package -- see the note on the Direct arm of supports() below.
//
// THE ENV VARIABLE IS BATCHLAS_GEMV_ROUTE. parse_route_env (route_env.hh:214)
// synthesises it from op_env_stem(Op::gemv), and legacy_variable_for has no
// Op::gemv case (route_env.hh:119's `default: return {}`), which is correct --
// no legacy gemv variable ever shipped and adding one would INVENT a spelling.
// Values that reach this table: "direct" / "cta" (a bare algorithm implies
// Origin::Native), "native", "vendor". Unset means {Auto, Auto}.
//
// AND A TRAP THE CAMPAIGN HAS PAID FOR TWICE: a bare BATCHLAS_GEMV_ROUTE=native
// resolves to the FIRST **SUPPORTED** native route in kGemvOrder -- which is
// CTA only on a GPU transposed shape that enumerates a sub-group size of 32,
// and DIRECT everywhere else. (An earlier version of this note said "which is
// CTA", full stop. Measured: with a bare `native`, 76 of 104 decisions in
// gemv_tests land on Direct, because CTA is unsupported on NoTrans, on the CPU
// device, and without an enumerated 32.) Pin "native:cta" or "native:direct"
// explicitly, always.
//
// AND THE TRAP ON THE OTHER SIDE, WHICH IS SILENT. Pinning a route the shape
// cannot take does NOT fail and does NOT warn -- resolve_route falls through to
// automatic(), and what automatic() then does DEPENDS ON THE BUILD:
//
//   BATCHLAS_GEMV_ROUTE=native:cta on a NoTrans shape, or on any CPU shape, or
//   on a GPU without an enumerated sub-group 32:
//       vendor-present build -> vendor:auto   (preferred() is all-false, so
//                                              automatic() IS the vendor)
//       vendor-free build    -> native:direct
//
// Measured in gemv_tests: that pin sends 76 of 136 decisions to cuBLAS/OpenBLAS
// while the operator believes CTA is pinned, and prints nothing. A MISSPELLED
// value behaves the same way and is worse: parse_route_value fails, the
// resulting ParsedRouteEnv::unparsed is discarded at gemv_route.hh's
// `parsed.found ? parsed.route : legacy_unset_default`, and all 136 decisions
// go to the vendor with no message. (trsm_route.hh, getrs_route.hh and
// potrf_route.hh discard it identically -- this is campaign-wide, not a gemv
// invention -- but gemv is the op whose README tells users to set the variable.)
//
// THE RESOLVED-ROUTE COLUMN IS THE ONLY WAY TO KNOW WHICH ARM RAN. Use
// BATCHLAS_COVERAGE_OUT, or a harness that prints the route it resolved. A
// kernel being linked is not evidence it ran.
//
// AND ONE THING THE ROUTE COLUMN CANNOT TELL YOU. {Native, Direct} on a NoTrans
// shape names TWO kernels: src/sycl/gemv_native.cc's body 1, and body 4 (the
// segmented short-output body) whenever out_len <= 16 and the device enumerates
// a sub-group size of 32. That choice is deliberately below the routing
// vocabulary -- it is a decomposition, not an algorithm, and putting it in
// supports() would be a speed cutoff in the predicate that carries correctness
// only. To establish which body ran, use a break that is red for one and green
// for the other; tests/gemv_tests.cc's `segld` / `segxinc` table is exactly that.
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE.
//
//     s.m = A.rows()      s.n = A.cols()      s.k = A.rows()
//     s.transA           NoTrans | Trans | ConjTrans
//
// and the two derived lengths, which are the ONLY spellings a predicate should
// use, because which of m and n is which SWAPS with transA:
//
//     out_len()   NoTrans -> m    Trans/ConjTrans -> n     (length of y)
//     red_len()   NoTrans -> n    Trans/ConjTrans -> m     (length of x)
//
// THIS IS NOT PEDANTRY. The one measured cuBLAS slow region in the whole gemv
// baseline (docs/perf/gemv.md#the-vendor-baseline) is
// complex<double> + Trans, 64 <= m <= 320, n >= 128 -- a band on **m**, which
// under Trans is red_len(), NOT out_len(). A predicate written on out_len()
// would test n, never touch m, and INVERT the window. Any preferred() clause
// added here must name its axis explicitly and cite the CSV it came from.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// GEMV reads three things OpShape has no field for. It reads NOTHING ELSE that
// OpShape already carries.
//
// DO NOT RE-DECLARE transA OR is_gpu HERE. OpShape already provides both
// (route.hh:230-240), along with max_sub_group and heterogeneous_batch.
// resolve_route SLICES this struct to OpShape on the way into the coverage
// table (route_resolve.hh:190-192), so a shadowing member would be written by
// the shape builder and then NOT copied: every gemv coverage row and every
// route_diff row would report transA = NoTrans, and gemv's two genuinely
// different access patterns would collapse into ONE first-writer-wins row.
// That is coverage.cc:40-58's stated failure mode, and for gemv the two arms
// are different KERNELS, not just different flags.
// ---------------------------------------------------------------------------
struct GemvShape : OpShape {
    // Is the Direct tier (src/sycl/gemv_native.cc bodies 1 and 2) linked in
    // this build? FALSE means "no native gemv kernel here" and correctly makes
    // the native routes unsupported rather than selectable-but-unimplemented --
    // TrsmShape::cta_max_n == 0's convention.
    bool direct_available = false;

    // Is the CTA tier (body 3) linked? Separate from the above because they are
    // independent capabilities, and because the CTA tier serves a strictly
    // narrower set of shapes.
    bool cta_available = false;

    // ENUMERATED from sycl::info::device::sub_group_sizes, never
    // get_property(MAX_SUB_GROUP_SIZE) -- that returns sub_group_sizes()[0],
    // the FIRST supported size, so `>= 32` is wrong in both directions: a
    // {8,16,32} device reads 8 and is refused although it supports 32, and a
    // {64} device reads 64 and is ACCEPTED although it has no 32 at all. Body 3
    // carries [[sycl::reqd_sub_group_size(32)]], for which the second is a
    // launch abort. See Device::supports_sub_group_size.
    bool has_sg32 = false;

    // The length of y and the length of x, in that order. See the field-mapping
    // note above: these SWAP with transA and a predicate that spells m or n
    // directly is testing a different axis depending on the transpose.
    int64_t out_len() const { return transA == Transpose::NoTrans ? m : n; }
    int64_t red_len() const { return transA == Transpose::NoTrans ? n : m; }
};

// CTA first, then Direct, then the vendor. A CAPABILITY LADDER, tighter first,
// not a preference: CTA serves only transposed shapes on a GPU that enumerates
// a sub-group size of 32, and Direct serves everything else -- including the
// native_cpu queue that half of tests/gemv_tests.cc runs on. With preferred()
// all-false the order matters only in the vendor-off walk at
// route_resolve.hh:60-63, where trying the tighter route first is what makes
// the transposed GPU case take the coalesced kernel.
//
// NO native_tier_preferred() HOOK, deliberately. That hook exists to arbitrate
// between two native routes that can BOTH serve a shape (geqrf's CTA vs
// Blocked). Here supports() already makes CTA and Direct mutually exclusive on
// every shape it admits -- CTA requires transA != NoTrans && is_gpu && has_sg32,
// and on any shape satisfying all three the ladder reaches CTA first anyway --
// so there is nothing for it to arbitrate and declaring it would add a
// predicate with no decision behind it.
inline constexpr Route kGemvOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::gemv, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a wrong answer or fail to
    // launch", never "the kernel would be slow".
    static bool supports(Route r, const GemvShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. HETEROGENEOUS BATCH. One launch covers the whole batch with a
        //    single (m, n, ld, stride) tuple, so per-item extents would be read
        //    at the wrong addresses. gemv has no analogue of gemm's
        //    heterogeneous walker (gemm_heterogeneous.hh) and cannot get one
        //    cheaply: VectorView has no active-size concept at all, so there is
        //    nothing to walk on the x and y side. For gemv this is therefore a
        //    CORRECTNESS gate, as it is for trsm, not merely un-preferred.
        if (s.heterogeneous_batch) return false;

        // 2. DEGENERATE EXTENTS. m == 0 or n == 0 is NOT here: that is a legal
        //    call and the kernel handles it exactly as reference ?GEMV does, by
        //    quick-returning WITHOUT touching y. A negative extent or an empty
        //    batch, on the other hand, has no launch geometry -- the flattened
        //    global range would be zero or negative -- so it goes to the vendor
        //    and gets whatever the vendor does with it.
        if (s.m < 0 || s.n < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // THE CAPABILITY FLAG ONLY, AND ***NO GPU GATE***. This one
                // line is the WP7 deliverable and it is deliberately unlike
                // every other native tier in this campaign.
                //
                // tests/gemv_tests.cc instantiates EIGHT suites:
                // GemvMatrixViewTest/0..3 are Backend::NETLIB on a native_cpu
                // Device("cpu") queue and /4..7 are Backend::CUDA. All eight
                // fail in a vendor-free build -- 40 failures, not 20. Add
                // `if (!s.is_gpu) return false;` here and the vendor-free walk
                // (route_resolve.hh:60-63) finds NO route for the four NETLIB
                // suites, the facade throws for them exactly as it does today,
                // gemv_tests stays RED, and the vendor-free burn-down moves by
                // ZERO. The kernel TU is compiled for the native_cpu target for
                // the same reason (src/sycl/CMakeLists.txt).
                //
                // It is also correct on the merits, which is what makes it
                // admissible in supports() at all: body 2 is a serial dot
                // product over a unit-stride column and body 1 is a serial dot
                // product over an ld-strided row. Neither uses a work-group
                // collective, neither allocates local memory, and neither
                // requires any sub-group size. There is nothing in either body
                // that a CPU device cannot execute.
                return s.direct_available;

            case Algorithm::CTA:
                // Body 3 is a 32-lane sub-group reduction down a COLUMN, so all
                // four of these are correctness or launch conditions:
                //   * the kernel must be linked;
                //   * a sub-group is a device notion -- the CPU device does not
                //     have one to reduce over;
                //   * it carries [[sycl::reqd_sub_group_size(32)]], and a
                //     device that does not ENUMERATE 32 aborts the launch;
                //   * there is no NoTrans body here at all. NoTrans is already
                //     fully coalesced with one work-item per output row, and
                //     gemv_native_cta throws rather than compute the wrong
                //     product if it is ever handed one.
                return s.cta_available && s.is_gpu && s.has_sg32 &&
                       s.transA != Transpose::NoTrans;

            default:
                // Including Algorithm::Auto. gemv has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:89-99).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // ALL-FALSE, AND THAT IS A RESULT, NOT AN OMISSION.
    //
    // The recon phase measured cuBLAS gemvStridedBatched at 94-105% of the
    // ~950 GB/s achievable DRAM roof on 90 of 92 reproducing cells, over all
    // four scalar types, both transA values, square and non-square, n from 32
    // to 2048 and batch from 64 to 65536
    // (docs/perf/gemv.md#the-vendor-baseline, two independent passes agreeing
    // to spread <= 1.01 on 98 of 104 cells). A batched gemv reads A once and
    // does two flops per element: there is no arithmetic to hide behind and no
    // reuse to exploit, so on a DRAM-resident cell the vendor is AT THE ROOF
    // and a win is arithmetically impossible. WP7's honest headline is
    // vendor-freedom at parity, which is exactly the headline WP6 shipped.
    //
    // A CLAUSE MAY BE ADDED HERE ONLY under the lead's acceptance gate: a
    // >= 1.15x median win reproducing across TWO independent passes, with the
    // axis named explicitly and the CSV cited. The one candidate region is
    // complex<double> + Trans at 64 <= m <= 320, n >= 128, where cuBLAS
    // measures 310-380 GB/s (33-40% of roof) while float, double and
    // complex<float> run 936-967 GB/s at IDENTICAL bytes and IDENTICAL (m, n).
    // That region is REAL -- 68 of 241 measured cdouble transposed cells win by
    // >= 1.15x, the bulk at 2.5-3.08x, cross-pass spread median 1.0054, and at
    // matched bytes the native CTA body reads 936-941 GB/s for ALL FOUR scalar
    // types while cuBLAS reads 323 for complex<double> and 940-946 for the other
    // three. The dip is cuBLAS's, and it is type-exclusive.
    //
    // AND IT STILL DOES NOT SHIP, FOR A THIRD-AXIS REASON THAT WAS MEASURED
    // TWICE INDEPENDENTLY. The rectangle "64 <= m <= 320 and n >= 256" was
    // fitted at a FIXED ~1 GB footprint, where batch moves inversely with shape
    // and the batch axis is therefore invisible. Resolved on a (m, n, batch)
    // grid it admits 17 cells BELOW 1.00x, worst 0.36x: at m=128, n=256,
    // transA=Trans the ratio is 0.52 at batch 128, 0.99 at batch 256 and 2.35 at
    // batch 512. The obvious repair -- an n*batch >= 131072 threshold, which all
    // four observed transitions straddle -- was PREDICTED AND THEN TESTED
    // OUT-OF-SAMPLE and REFUTED: at m=96, n=192, batch=1024 it predicts a win
    // and measures 0.97x/0.97x, with cuBLAS at 925 GB/s, i.e. at the roof.
    // Every clause that passes the >= 1.15x gate strictly (n >= 768, or
    // A >= 1024 MB) sits on the edge of the sampled range and captures at most
    // 22 of the 68 measured wins. See docs/perf/gemv.md#routing-hypotheses-refuted.
    //
    // SO THE 3x IS ONE ENVIRONMENT VARIABLE AWAY, NOT ZERO AWAY:
    // BATCHLAS_GEMV_ROUTE=native:cta, on a transposed GPU shape. On any other
    // shape that same pin measures the vendor -- see the env note at the top.
    //
    // TWO THINGS THAT MUST NOT GO IN A CLAUSE HERE.
    //
    //   * NOT out_len(). Under Trans, out_len() == n and red_len() == m. The
    //     measured band is on **m**. A predicate on out_len() tests the wrong
    //     axis and inverts the window.
    //
    //   * NOT AN L2-RESIDENCY GATE. The tempting story -- "the collapse
    //     switches on when A leaves the 72 MB L2" -- is contradicted by the
    //     measurements themselves: such a gate admits cells where cuBLAS runs
    //     at 92-96% of roof and the native kernel cannot win. If a residency
    //     notion is ever wanted here it has to come from a measured CSV, not
    //     from a cache size.
    //
    // ---- WHERE THE NATIVE KERNEL IS SLOWER THAN THE VENDOR ----------------
    // B6 requires every cell below 0.50x to be fixed or STATED. One family was
    // found, fixed and re-measured; one remains and is stated here. None of
    // these is reached by DEFAULT in a vendor-present build -- preferred() is
    // all-false, so the vendor takes every shape -- and in a vendor-free build
    // the alternative is not a slower kernel, it is no kernel at all.
    //
    //   FIXED. Algorithm::Direct, transA = NoTrans, out_len < 32 measured
    //   0.08x-0.38x of cuBLAS on 13 cells: below the 32-lane warp width the
    //   flattening b = gid/out_len makes a warp straddle batch items (32/out_len
    //   sectors per load) and out_len*batch is the whole launch (2.08% occupancy
    //   at out_len <= 8). src/sycl/gemv_native.cc body 4 puts W = 32/out_len
    //   lanes on each output; re-measured, the family is now 0.93x-1.44x with a
    //   worst cell of 0.54x and NOTHING below 0.50x.
    //
    //   STATED, NOT FIXED. Algorithm::CTA, complex<double>, transposed, with a
    //   SHORT REDUCTION. red_len() == 64 at batch 512 measures 0.43x-0.46x
    //   (vendor 1419.7 GB/s vs native 639.1); at batch 128 the whole
    //   red_len() <= 128 column loses, 0.27x-0.60x across n in {128, 256, 512}.
    //   MECHANISM, measured rather than modelled: the shuffle ladder is a fixed
    //   cost per output (5 steps, doubled to 10 for a complex scalar because the
    //   halves fold separately) against ceil(red_len/32) rounds of useful loads,
    //   and ncu shows occupancy steady at 82-95% and grid steady while DRAM
    //   throughput climbs 38.5% (red_len 32) -> 64.8% (64) -> 93.4% (128) ->
    //   95.6% (512). Fully amortised by red_len = 128. It is NOT occupancy and
    //   NOT coalescing. THE NAMED FIX, not attempted: serve W outputs per
    //   sub-group when red_len is small, cutting the ladder to log2(W) steps --
    //   the same idea body 4 applies on the NoTrans side, transposed.
    //
    //   ALSO STATED: L2-RESIDENT complex<double> transposed is 0.45x-0.56x, but
    //   for a different reason and it is not a defect in the kernel. The vendor's
    //   figures there are ABOVE the ~1008 GB/s DRAM peak (1398 and 1721 GB/s):
    //   it converts L2 residency into bandwidth. The native kernel streams -- one
    //   pass over A, no blocking that would give a second pass anything to hit --
    //   and produces the same 635-961 GB/s it produces from DRAM. It is not
    //   slower here; it is unchanged here while the vendor gets faster.
    //
    //   AND WHAT THE LIBRARY ITSELF ISSUES IS FINE. Over the 56 cells ortho.cc
    //   actually calls gemv on, the worst is 0.75x, the median 1.14x, and 49 of
    //   56 are at or above cuBLAS. The 0.08x family needed a short OUTPUT and
    //   ortho only ever gives the NoTrans body a short REDUCTION, so it was never
    //   reachable from inside the library.
    // =====================================================================
    // WP8 ROUTING PASS: THE PRIZE ROUTES. The all-false verdict above stands
    // for three of the four scalar types and for every shape outside the band
    // below; what changed is that the grid which produced "no predicate exists"
    // could not see the axis the effect lives on, and a finer one can.
    //
    // TWO THINGS HAD TO HAPPEN FIRST, AND BOTH ARE MEASUREMENTS, NOT OPINIONS.
    //
    // (1) THE CLAUSE FAMILY THAT WAS SEARCHED DID NOT CONTAIN `batch`.
    //     docs/perf/gemv.md#routing-hypotheses-refuted enumerates
    //     (m band) x (n threshold) x (A threshold) and nothing else, so every
    //     REFUTED verdict in clause_report.txt is a verdict about clauses that
    //     cannot express the boundary. Re-searched with batch as a first-class
    //     term (docs/perf/gemv.md#the-cdouble-window-boundaries), a clause survives.
    //
    // (2) THE BAND'S LOWER EDGE WAS OUR KERNEL'S LIMIT, NOT cuBLAS'S. Before
    //     WP8-I3's body 5, red_len 48 measured 0.67-0.79x -- the vendor was
    //     ALREADY dipped there (561-648 GB/s) and the native CTA arm was stuck
    //     at 442-449 GB/s by the short-reduction defect. Fitting the band before
    //     that fix would have fitted it one grid step too high.
    //
    // WHAT cuBLAS ACTUALLY DOES, AND IT IS A DISCRETE SWITCH RATHER THAN A
    // GRADIENT. At out_len 512, red_len 128, complex<double>, Trans, its
    // throughput goes 894.9 / 919.4 / 930.1 GB/s at batch 128 / 192 / 256 --
    // i.e. AT THE ROOF -- and then 360.4 / 359.7 / 363.1 / 358.4 at batch
    // 320 / 384 / 448 / 512. One batch rung, a 2.6x fall, and it stays fallen.
    // That is a kernel-selection threshold in the vendor, which is why no
    // function of n*batch and no power law n^a*batch can describe it (both
    // a > 1 and a < 1 are required simultaneously -- see the two refuting pairs
    // in docs/perf/gemv.md#routing-hypotheses-refuted section 5), and why the honest
    // predicate names batch outright.
    //
    // THE BAND IN red_len IS SHARP AT BOTH ENDS. At out_len 256/512, batch 512
    // (docs/perf/gemv.md#the-cdouble-window-boundaries, grid B, red_len walked 8..512):
    //     red_len   32     40     48     56     64    ...   320    352    384
    //     ratio   0.95   1.10   1.31   1.16   2.41    ...  3.01   2.84   1.03
    //     vendor   909    793    677    771    373    ...   310    329    906
    // The vendor is back AT THE ROOF at red_len 384 (901-906 GB/s) and only
    // partially dipped below 64 (677-909). 64 and 352 are therefore the rungs
    // where the dip is SATURATED, and the cells that bracket them are measured
    // non-winners: 0.9515 at red_len 32, 1.0304/1.0314 at red_len 384.
    //
    // THE out_len BOUNDARY IS BRACKETED TOO. At red_len 128, batch 512, walking
    // out_len 32..2048: 0.559 (32), 0.546 (64), 0.989 (96), 0.994 (128), 0.999
    // (192), then 2.32 (256). The cells below 256 are not losses we could fix --
    // at out_len 32 and 64 cuBLAS reads 1597 and 1777 GB/s, ABOVE this device's
    // ~1008 GB/s DRAM peak, because it is converting L2 residency into
    // bandwidth. That is the family route_gemv.hh already declines to chase.
    //
    // ---- THE CLAUSE, AND EVERY BOUNDARY IT NAMES IS BRACKETED BY A CELL ----
    //   scalar == complex<double>          (float, double and cfloat REFUTED
    //                                       below, each with its cell)
    //   route  == {Native, CTA}
    //   transA != NoTrans                  (CTA has no NoTrans body at all)
    //   64 <= red_len() <= 352             red_len(), NEVER out_len(): under
    //                                      Trans red_len() is A.rows(). A
    //                                      predicate on the wrong extent
    //                                      inverts this window and that error
    //                                      was caught twice in WP7.
    //   out_len() >= 256
    //   batch    >= 320
    //
    // WHERE THE CLAUSE WAS FITTED, AND WHAT ITS WEAKEST CELL ACTUALLY READS.
    // The g6_fit grids all carry gpu=0 on every row: the clause was fitted on
    // the DISPLAY GPU. The cross-device control taken at the time was valid but
    // was taken at DRAM-RESIDENT footprint, and the clause's lowest-footprint
    // admitted cell -- cdouble, out_len 256, red_len 64, batch 320 -- is 84 MB
    // against a 72 MB L2, i.e. exactly on the boundary where the control does
    // not hold. That cell is the noisiest in the clause and it is the only one
    // where the fitting device flatters us:
    //     device 0 (fit)            2.313
    //     device 1, two passes      1.867 / 2.065   (adversarial review)
    //     device 1, two more        2.394 / 2.091   (lead, idle box, GPU 1)
    // The NATIVE arm agrees across devices to 3.5%; only the VENDOR arm moves,
    // by ~20%, and only here. Read the floor as ~1.87, not as the fitted number.
    // It clears the >= 1.15x bar by a wide margin either way, and every other
    // admitted cell re-measured on the idle card sits at 2.25-2.49 -- so this is
    // ONE L2-boundary corner, not the band. Recorded because a fitted minimum
    // taken on a contended device is exactly the reading this pass otherwise
    // spent its measurement budget learning not to trust.
    //
    // THE THREE TYPES THIS CLAUSE EXCLUDES, WITH THE CELL THAT EXCLUDES EACH --
    // measured INSIDE the band, so this is a refutation and not an omission:
    //   float    out=256 red=128 batch=512   0.9340
    //   double   out=512 red=128 batch=1024  0.9722  (and 0.9746, 0.9749 beside)
    //   cfloat   out=256 red=48  batch=512   0.6644  (cuBLAS 2637 GB/s: L2)
    //
    // AND THE WIDER CANDIDATES, EACH WITH ITS REFUTING CELL:
    //   batch >= 256      0.9628 at out=512 red=128 batch=256 (cuBLAS 930.1)
    //   batch >= 192      0.9616 at out=256 red=128 batch=192 (cuBLAS  873.7)
    //   batch >= 128      0.9562 at out=512 red=128 batch=128 (cuBLAS  894.9)
    //   no batch term     the same 0.9562 -- and BELOW 128 the region is not
    //                     marginal, it is refuted outright: 46 cells inside the
    //                     (red_len, out_len) band at batch 1..96 hold 27 LOSSES,
    //                     worst 0.5417 at out=512 red=128 batch=64. A clause
    //                     without a batch floor routes those.
    //   A >= 256 MB instead of a batch term
    //                     0.9628 at out=512 red=128 batch=256, 256 MB -- the
    //                     footprint substitution is REFUTED by a cell, which is
    //                     the answer to "isn't batch just a proxy for size".
    //   red_len >= 48     THE ONE CANDIDATE THAT PASSES THE LETTER OF THE GATE
    //                     AND IS STILL DECLINED, so the reasoning is spelled
    //                     out. Over two passes it scores 88 cells, geomean
    //                     2.135, MIN 1.1605, zero losses -- 0.9% above the bar.
    //                     It is declined on a MECHANISM, not on a margin: at
    //                     red_len 64..352 the vendor sits on the FLAT FLOOR of
    //                     its dip, 304-386 GB/s at every out_len and every
    //                     batch, so extrapolating the clause to the (out_len,
    //                     batch) corners the grid did not reach is safe. At
    //                     red_len 48 it is on the SLOPE -- 456 GB/s at batch
    //                     128 climbing to 765 at batch 1024, still rising at the
    //                     top of the ladder -- and the clause admits batches
    //                     above 1024 and out_len above 2048 where the slope
    //                     continues. Below it, red_len 40 is 1.1028 and red_len
    //                     32 is 0.9515, so the region is genuinely ragged.
    //   red_len unbounded above
    //                     1.0304 at out=256 red=384 batch=512.
    //
    // THE UPPER EDGE OF 352 IS THE UNIVERSAL ONE, NOT THE ONLY ONE, because the
    // dip is a staircase in all three axes and not a box. At out_len 2048 the
    // vendor is STILL dipped at red_len 384 and 448 (313.6 and 317.2 GB/s,
    // ratios 2.998 and 2.973) where at out_len 768 and 1024 it is back at the
    // roof at the same red_len and batch (900.9 and 899.8 GB/s, 1.0406 and
    // 1.0426). So the true boundary moves outward with out_len and 352 is where
    // it sits for the SMALLEST admitted out_len. It closes for good at red_len
    // 512 even at out_len 2048 (927.7 GB/s, 1.0166). Encoding the movement needs
    // a two-variable boundary fitted on four out_len levels; the cells are here
    // and the fit is not attempted.
    //
    // WHAT THIS CLAUSE GIVES UP, STATED RATHER THAN BURIED. At out_len >= 768
    // the vendor is dipped at EVERY batch measured, down to 128, so a second
    // disjunct `out_len() >= 768 && batch >= 128` would capture roughly 18 more
    // measured cells at 2.26x-2.91x with no loss anywhere in the grid. It is NOT
    // shipped, for one reason: 128 is the LOWEST batch that grid ever reached at
    // those out_len, so the floor is the edge of the sampled range rather than a
    // bracketed boundary -- which is precisely the objection WP7's own audit
    // raised against its `A >= 1024 MB` candidate, and this pass is not going to
    // commit the error it is here to avoid. The bracketing sweep is one cell
    // list (docs/perf/gemv.md#open-debts, grids H and I) and is named as
    // open work.
    // =====================================================================
    static bool preferred(Route r, const GemvShape& s) {
        // Only the CTA tier. Direct is the NoTrans/CPU arm and nothing in this
        // window is about it; a true there would let the walk stop on Direct for
        // a shape CTA cannot serve, on the strength of a CTA measurement.
        if (!is_native(r) || r.algo != Algorithm::CTA) return false;

        if constexpr (std::is_same_v<T, std::complex<double>>) {
            // Redundant with supports(), and deliberately so: this is a SPEED
            // predicate about the transposed kernel, and a reader has to be able
            // to see that without holding supports() in their head.
            if (s.transA == Transpose::NoTrans) return false;

            const int64_t red = s.red_len();   // == A.rows() under Trans
            const int64_t out = s.out_len();   // == A.cols() under Trans
            return red >= 64 && red <= 352 && out >= 256 && s.batch >= 320;
        }
        return false;
    }

    static constexpr const Route* order_begin() { return kGemvOrder; }
    static constexpr const Route* order_end() {
        return kGemvOrder + (sizeof(kGemvOrder) / sizeof(kGemvOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// gemv into the coverage table: resolve_route records every op that goes
// through it (route_resolve.hh:139-150), slicing GemvShape to OpShape. No
// record_level3_route call is needed, and adding one would double-count.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_gemv_route(Route forced, const GemvShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::gemv, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
