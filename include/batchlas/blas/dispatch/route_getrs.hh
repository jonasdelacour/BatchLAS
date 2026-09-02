#pragma once

// GETRS's routing table (WP6 scaffolding).
//
// Same three rules as route_getrf.hh, route_geqrf.hh and route_potrf.hh, and
// they are not repeated here in full:
//
//     supports()   == correctness only. Never a speed cutoff, because
//                     route_resolve.hh:113-127 implements the vendor-free
//                     fallback by re-walking the order testing supports() ALONE,
//                     and :165 says a forced route bypasses preferred() but never
//                     supports().
//     preferred()  == the measured window. NO LONGER ALL-FALSE: it carries the
//                     nrhs window measured in docs/perf/lu.md#getrs-fused-window-evidence, so
//                     this table now moves the DEFAULT in a vendor-present build
//                     as well as in a vendor-free one. See preferred() below.
//     the env read == src/backends/getrs_route.hh, not here. PURE header.
//
// TWO NATIVE ARMS. getrs against a factored A is a row permutation plus two
// triangular solves, and there are two genuinely different ways to spend it:
//
//   {Native, CTA}     the FUSED narrow-RHS kernel, src/extensions/getrs_fused.cc.
//                     ONE launch per call: one work-group per matrix, the
//                     interchange walk and both substitutions inside it, no GEMM
//                     and no separate laswp. Serves only the (n, nrhs) pairs
//                     whose right-hand side fits local memory.
//   {Native, Blocked} the COMPOSITION, src/extensions/getrs_native.cc: a laswp
//                     launch plus two ROUTED trsm calls. `Blocked` names "a
//                     host-driven composition over routed BLAS-3", which is what
//                     orgqr's arm is too (ormqr on an identity) -- not a claim
//                     that this op has a panel schedule.
//
// The split is a MEASUREMENT, not a taxonomy. At nrhs = 1 the composition is
// 0.32x of cublas?getrsBatched and the fused kernel is 2.10x of it, a factor of
// 7.9 between the two native arms, because trsm's blocked driver amortises a
// panel over many columns and one column gives it nothing to amortise.
//
// nrhs = 1 IS NOT THE ONLY WIDTH THE LIBRARY ISSUES, and an earlier reading of
// this file that said so was wrong: linalg::solve (linalg-ops.hh:336-344) and the
// Python binding (ops_factorization.cc:91) are the only callers of getrs in the
// tree and BOTH pass the caller's own B.cols() through unchanged. The fused tier
// therefore serves a WINDOW, and the width at which it stops is a real routing
// question rather than a formality.
//
// FIELD MAPPING -- IT IS ITS OWN, AND transA IS A LIVE ROUTING INPUT. This is the
// only op in the LU family with a variant, and it is the only good news in
// WP6's instrument story:
//
//     s.m = A.rows()   == THE ORDER of the factored matrix
//     s.n = B.cols()   == nrhs
//     s.k = A.rows()   == the order again (so max_dim/min_dim behave)
//     s.transA         == THE TRANSPOSE MODE, SET
//
// SET transA OR THROW AWAY THE ONLY SEPARABLE COVERAGE ROWS THIS FAMILY HAS.
// coverage.cc:52-58's variant_key carries uplo/side/diag/transA/transB. getrf and
// getri set none of them, so their rows collapse to shape_class alone
// (route.hh:249-259, first-writer-wins at coverage.cc:284-292) and
// scripts/route_diff.sh cannot tell one LU call from another. getrs's transA is
// the one field that separates NoTrans from Trans from ConjTrans. The builder
// sets it (src/backends/getrs_route.hh); dropping that line would be silent.
//
// AND transA IS ALSO A REAL ALGORITHM FORK, not just a label. Verified against a
// host LAPACKE oracle through the public API at n=6, batch=3, nrhs=2, residual
// max|op(A)X - B|: NoTrans 1.19e-07 / 6.66e-16 / 2.39e-07 / 8.88e-16, Trans
// 2.38e-07 / 3.33e-16 / 3.58e-07 / 8.88e-16, ConjTrans (complex) 3.58e-07 /
// 8.88e-16. NoTrans is "apply P, solve L, solve U"; Trans/ConjTrans is
// "solve U^T/U^H, solve L^T/L^H, then apply P^T LAST, on the OUTPUT, in
// REVERSE". The two triangular solves swap order AND the permutation moves to
// the other end. Nothing in BatchLAS interprets transA today -- enum_convert
// hands it straight to the vendor (cublas.cc:1478, rocsolver.cc:274-287,
// netlib_lapack.cc:1247-1249) -- so the reverse-permutation trap is entirely on
// the native side, and it is the reason the transA field is worth carrying into
// the coverage table.
//
// STATUS: BOTH NATIVE ARMS ARE LIVE for all four scalar types. The composition
// (getrs_blocked_available<T>()) serves every square GPU shape; the fused tier
// (getrs_fused_available<T>() plus its two capacity numbers) serves the shapes
// whose right-hand side is resident and whose nrhs is instantiated. A VENDOR-FREE
// BUILD TAKES THE FUSED TIER WHEREVER IT FITS and the composition elsewhere,
// which is what native_tier_preferred below decides.
//
// THE VENDOR-PRESENT BUILD CHANGED TOO, and this paragraph used to say the
// opposite. preferred() below is the measured nrhs window -- nrhs <= 2 for every
// type, plus nrhs <= 4 for float -- so an Origin::Auto getrs inside it now
// resolves to {Native, CTA} even where cuBLAS exists. Outside it, Origin::Auto
// still returns {Vendor, Auto} (route_resolve.hh:110-112, :129), which is the
// state this file shipped in and the state every wider width stays in.
//
// AND `BATCHLAS_GETRS_ROUTE=native` CHANGED MEANING when CTA joined kGetrsOrder
// ahead of Blocked. A bare origin resolves to the FIRST supported route of that
// origin (route_resolve.hh:146-163), which is now the fused tier wherever the
// right-hand side is resident; it used to be the composition, because that was
// the only native route. Any baseline recorded with a bare `native` pin --
// docs/perf/lu.md#negative-results:37 and kernels/run_grid.sh:39 export one
// value into all three LU variables at once -- is measuring a different getrs
// today than when it was recorded. Pin `native:blocked` to mean what `native`
// used to mean. This is a measurement-comparability trap, not a correctness one,
// and tests/route_vocabulary_tests.cc's BareOriginResolvesToASpecificAlgorithm
// is where it is asserted.
//
// THE TWO MEASUREMENTS, both against cublas?getrsBatched at saturating batch, in
// process, against a host oracle:
//
//   THE COMPOSITION (docs/perf/lu.md#the-vendor-baseline-and-saturation):
//     nrhs = 1  : GEOMEAN 0.36x over 28 cells, 25 LOSSES, worst 0.09x (cdouble
//                 n=32). Only n=2048 wins (1.07-1.15x) and that is against an
//                 UNSATURATED vendor.
//     nrhs = 64 : geomean 1.17x (interchange list) / 1.55x (collapsed to a
//                 gather), 20 and 25 wins of 28.
//     The nrhs=1 loss is STRUCTURAL, not a bad kernel: trsm's blocked driver
//     amortises a panel over many columns and one column gives it nothing to
//     amortise; the permutation is a rounding error there (the gather strategy
//     changes the geomean by 0.00x).
//
//   THE FUSED TIER (docs/perf/lu.md#the-fused-narrow-rhs-getrs, grid_big.csv):
//     nrhs = 1  : GEOMEAN 2.10x over cuBLAS across 15 cells (4 types x n in
//                 {64,128,512,2048}), NO LOSSES, worst 1.24x, best 3.62x; and
//                 7.86x over the composition. It runs at 82% of this device's
//                 DRAM peak at float n=512 -- but ONLY in an n = 256..512 band,
//                 and the original form of this sentence ("the ceiling is
//                 reached") was too strong. Achieved fraction of 1008 GB/s at
//                 nrhs=1, recomputed per cell from grid_cta.csv:
//                     n=32   72% float, 38% double, 95% cfloat, 24% cdouble
//                     n=512  82% / 86% / 88% / 83%      <- the band
//                     n=2048 41% / 50% / 60% / 41%
//                 The large-n shortfall has a named mechanism: one work-group per
//                 matrix means the CTA COUNT IS THE BATCH, so n=2048 batch=32
//                 occupies 32 of this device's 128 SMs. The small-n shortfall is
//                 that nb=16 leaves the block solve to 16 lanes of one sub-group.
//                 Both are open work, not a ceiling.
//     nrhs<= 8  : ahead of the composition at every cell inside its own
//                 capability (worst 1.11x). Against the VENDOR it crosses over
//                 per type and n -- double and cdouble at n=64 are already below
//                 1.0x by nrhs = 4 -- and THAT crossover is what preferred()
//                 below now encodes, from 488 cells rather than from these 15.
//
// nrhs is s.n and IS available to preferred() -- that is why the field mapping
// puts it there rather than folding it into max_dim. inv.cc, the only internal
// consumer of the LU family, does not call getrs at all.
//
// A LATENT VENDOR DEFECT UNDER THIS OP, recorded rather than fixed inside WP6
// (the rule factorization.cc:69-87 sets for geqrf's twin): cublas.cc's getrs has
// TWO arms -- batch >= 2 takes cublas?getrsBatched (:1477-1480), batch <= 1 takes
// cusolverDnXgetrs (:1466-1473), a DIFFERENT LIBRARY and the 64-bit non-batched
// API, from inside a TU gated on BATCHLAS_HAS_CUBLAS. A cuBLAS-present /
// cuSOLVER-absent configure claims a vendor it cannot link. The fix is to the
// gate in vendor_available.hh, in its own change.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

struct GetrsShape : OpShape {
    // Whether the native getrs driver exists in this build. FALSE today.
    //
    // It is NOT "is trsm compiled" -- that is true already and would make this
    // table advertise a route the facade cannot service. It is "is the driver
    // that applies the pivots and issues the two solves compiled", which is the
    // question route_trsm.hh:99-110 insists on: the table must describe the
    // BUILD, not the design. Same distinction as OrgqrShape::blocked_available
    // (route_orgqr.hh:89-96), which is deliberately not "is ormqr_blocked
    // compiled".
    bool blocked_available = false;

    // ENUMERATED from sub_group_sizes, never `max_sub_group >= 32`. See
    // GetrfShape::has_sg32 for why the MAX_SUB_GROUP_SIZE property is wrong in
    // both directions.
    //
    // IT IS LIVE, AND THE FUSED TIER IS WHAT MAKES IT SO. The note here used to
    // end "if the permutation kernel ships without a required sub-group size,
    // DELETE this field rather than leaving it to read as live" -- the state
    // route_potrf.hh:83-96 criticises trsm for. That debt is now paid the other
    // way: GetrsFusedNKernel / GetrsFusedTKernel carry
    // [[sycl::reqd_sub_group_size(32)]] and their diagonal-block solve IS a
    // 32-lane shuffle recurrence, so a device offering only {64} cannot launch
    // them at all. The composition's routed trsm needs the same thing, which is
    // why the gate is applied to both algorithms rather than only to CTA.
    bool has_sg32 = false;

    // ---- THE FUSED NARROW-RHS TIER'S TWO CAPACITY NUMBERS -----------------
    //
    // Both are CAPABILITIES, not speed windows, and both belong in supports()
    // for the same reason GetrfShape::cta_max_n does: above either of them the
    // kernel DOES NOT LAUNCH, so reporting the route supported would hand a
    // vendor-free caller a route the facade cannot service.

    // The largest n * nrhs the fused kernel's RESIDENT RHS can hold on this
    // device, or 0 when the kernel is not in this build. It is n*nrhs and not n
    // alone because the fused tier holds the whole right-hand-side BLOCK in local
    // memory, so the two extents trade against each other -- which is exactly why
    // GetrfShape's equivalent is a single order and this one is not.
    //
    // ASKED OF THE DEVICE (src/backends/getrs_route.hh reads LOCAL_MEM_SIZE and
    // calls sycl_getrs::getrs_fused_max_rhs_elems<T>), never a constant, for
    // route_potrf.hh:114-127's reason.
    int64_t fused_max_elems = 0;

    // The widest nrhs the fused kernel is INSTANTIATED for, or 0 when it is
    // absent. Separate from the element capacity because it is a different kind
    // of ceiling: the element capacity is this DEVICE's local memory, this one is
    // what the BUILD compiled. A device with 4x the local memory does not gain an
    // nrhs = 16 instantiation.
    int64_t fused_max_nrhs = 0;

    // The order of the factored matrix.
    int64_t order() const { return m; }
    // The number of right-hand sides. Named, so a later predicate cannot reach
    // for m when it meant n -- the PotrfShape::order() discipline. It is also
    // the variable the nrhs=1 measurement above is about.
    int64_t nrhs() const { return n; }
};

// One native route, then the vendor. Same file-scope-array-with-sizeof-bounds
// form as kPotrfOrder, kGeqrfOrder and kOrgqrOrder, not a hand-counted
// std::array<Provider,6> (route_gemm.hh:43-46).
// TWO NATIVE ROUTES NOW, AND THE ORDER IS A CAPABILITY LADDER RATHER THAN A
// PREFERENCE -- kGetrfOrder's shape exactly. CTA is the FUSED narrow-RHS kernel
// (src/extensions/getrs_fused.cc): one work-group per matrix, permutation and
// both substitutions in ONE launch, serving only the (n, nrhs) pairs whose
// right-hand side fits local memory. Blocked is the composition
// (src/extensions/getrs_native.cc) and serves everything else.
//
// `CTA` is the right spelling: the kernel is one work-group per matrix with a
// resident working set and a sub-group recurrence, which is what CTA names for
// getrf, potrf and geqrf. It is NOT a claim that the matrix is resident -- it is
// not, and cannot be: n = 512 float is 1 MB per item.
//
// With preferred() all-false the order matters only in the vendor-free walk at
// route_resolve.hh:113-127, where native_tier_preferred runs FIRST and the raw
// order is the fallback -- so listing the tighter route first is right for the
// same reason it is right for getrf.
inline constexpr Route kGetrsOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getrs, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // The structural gates below are RouteTable<Op::trsm,T>::supports()'
    // (route_trsm.hh:132-160) TRANSCRIBED, plus getrs's own, because the routed
    // trsm is what will actually serve the two solves and silently omitting an
    // inherited gate is the wrong-answer class route_orgqr.hh:41-49 records.
    //
    // WHAT IS DELIBERATELY *NOT* TRANSCRIBED: trsm's CAPACITY gates
    // (TrsmShape::cta_max_n and blocked_available). Those choose between trsm's
    // own two tiers and carry NO upper bound on the order -- trsm's blocked
    // driver serves everything above the CTA cap -- so there is no order this
    // table could refuse on their account. Transcribing a capacity that is not a
    // ceiling would be a gate that cannot fire, which is worse than no gate: it
    // reads as live.
    static bool supports(Route r, const GetrsShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. THE ARM MUST EXIST IN THIS BUILD. Checked PER ALGORITHM, because
        //    the two native routes are independent capabilities: the fused tier
        //    advertises itself through fused_max_elems / fused_max_nrhs and the
        //    composition through blocked_available. An absent capability must
        //    never select a launch that is not there.
        if (r.algo == Algorithm::CTA) {
            if (s.fused_max_elems <= 0 || s.fused_max_nrhs <= 0) return false;
        } else if (r.algo == Algorithm::Blocked) {
            if (!s.blocked_available) return false;
        }

        // 2. GPU ONLY -- INHERITED from route_trsm.hh:138-142 and true of the
        //    permutation kernel in its own right. There is no host
        //    implementation of either to fall back on, so a CPU queue has to
        //    reach netlib.
        if (!s.is_gpu) return false;

        // 3. SUB-GROUP SIZE 32. NO LONGER DECORATIVE, and the note that used to
        //    say "delete both if the kernel that lands carries no
        //    reqd_sub_group_size" is now discharged: the fused tier's kernels
        //    carry [[sycl::reqd_sub_group_size(32)]] and their diagonal-block
        //    solve is a 32-lane shuffle recurrence, so a device offering only
        //    {64} is a launch abort and not a slowdown. It stays applied to BOTH
        //    algorithms because the composition's routed trsm needs it too.
        if (!s.has_sg32) return false;

        // 4. HETEROGENEOUS BATCH -- INHERITED from route_trsm.hh:151-154, where
        //    it is a hard correctness gate for exactly the same reason: one
        //    launch, one (order, nrhs, ld, stride) tuple, no batch walker. It is
        //    ALSO getrs's own: the pivot list is read at
        //    pivots[b*order + k] with a single order, so per-item extents
        //    scatter it.
        if (s.heterogeneous_batch) return false;

        // 5. DEGENERATE EXTENTS. The solve loops and the interchange walk are
        //    undefined for an empty system, no right-hand side, or an empty
        //    batch. Disagreement BETWEEN A and B -- non-square A, mismatched
        //    rows, mismatched batch -- is not testable here, because OpShape
        //    holds ONE shape; the builder reports it by returning no shape at all
        //    (the gemm_op_shape pattern, src/backends/gemm_variant.hh:189-197).
        if (s.order() < 1 || s.nrhs() < 1 || s.batch < 1) return false;

        // 6. THE PIVOT FORMAT MUST AGREE WITH WHATEVER ELSE TOUCHES THE SPAN.
        //    The pivot buffer is `Span<int64_t>` on the wire and its PHYSICAL
        //    layout is backend-dependent. The CUDA and ROCm vendor arms pack
        //    1-based int32 into the first half of it (cublas.cc:1509,
        //    rocsolver.cc:227, both `pivots.as_span<int>()`) and the native
        //    kernels write and read exactly that. NETLIB does NOT: it widens an
        //    int scratch into GENUINE int64 (netlib_lapack.cc:1312-1320) and reads
        //    genuine int64 back (:1235, :1361).
        //
        //    The three LU ops have three independent env variables and three
        //    independent preferred() windows, so every mixture of native and
        //    vendor arms is reachable -- and on a GPU queue constructed with
        //    Backend::NETLIB (a public constructor, and is_gpu reads the QUEUE,
        //    not the backend) a native getrf feeding netlib's getri reads the
        //    permutation out of the wrong four bytes of every eight. Measured:
        //    ||A*C - I||_F / n = 5.32e-01 with info == 0, against 5.15e-07 when
        //    both arms agree. No throw, no flag, no test in the suite can see it
        //    (tests/getrf_tests.cc skips every NETLIB row because its queue is a
        //    CPU queue).
        //
        //    This is a CORRECTNESS gate, not a speed one: the native arm computes
        //    a wrong answer for the CALL CHAIN it would sit in. The shape carries
        //    the field -- src/backends/getrs_route.hh sets `s.backend = B` -- so the
        //    gate is one predicate, and it is enumerated by the backend whose
        //    format disagrees rather than by an allow-list, so a new GPU backend
        //    that packs int32 like the other two needs no edit here.
        if (s.backend == Backend::NETLIB) return false;

        // 7. ALL THREE TRANSPOSE MODES ARE SUPPORTED, and saying so explicitly is
        //    the point of this clause. transA is a live routing input for this
        //    op (see the FIELD MAPPING note), and the natural wrong edit is to
        //    refuse Trans/ConjTrans "until the reversed path is written" -- which
        //    would be a gate that goes stale silently the moment it IS written.
        //    The vendor serves all three correctly (measured, see the header
        //    note), so a native arm that cannot must say so HERE, with a test
        //    that fails without it, at the moment it lands. Not before.
        switch (r.algo) {
            case Algorithm::CTA:
                // THE FUSED TIER'S TWO CAPACITY CEILINGS. Both are "the kernel
                // cannot run", never "the kernel would be slow":
                //
                //   * the RESIDENT RHS. n * nrhs elements of local memory plus one
                //     diagonal block; above it the launch is refused by the
                //     runtime, not merely slow.
                //   * the WIDEST INSTANTIATED nrhs. The trailing update carries a
                //     compile-time accumulator array so each A element is reused
                //     across the right-hand sides from a register, and above
                //     kGetrsFusedMaxRhs no instantiation exists.
                //
                // A SPEED WINDOW ON nrhs DOES **NOT** GO HERE, and that is the
                // whole point of the split: route_resolve.hh:165 says a forced
                // route bypasses preferred() but NEVER supports(), so a speed
                // threshold here would make a pinned `native:cta` fall through to
                // automatic() and the test that pinned it would measure cuBLAS
                // and pass green. The measured nrhs window lives in
                // native_tier_preferred (native-vs-native) and, when it is
                // written, in preferred() (native-vs-vendor).
                if (s.order() * s.nrhs() > s.fused_max_elems) return false;
                if (s.nrhs() > s.fused_max_nrhs) return false;
                return true;
            case Algorithm::Blocked:
                return true;
            default:
                // Including Algorithm::Auto: with one native route the
                // distinction is invisible in practice, but resolve_route walks
                // the order restricted to the requested origin
                // (route_resolve.hh:146-163) and expects a SPECIFIC algorithm
                // back, so {Native, Auto} must not be reported supported.
                return false;
        }
    }

    // ---- THE MEASURED WINDOW, AND IT IS A WINDOW ON nrhs ------------------
    //
    // THIS MOVES THE DEFAULT IN EVERY BUILD. preferred() is the native-vs-VENDOR
    // question (route_resolve.hh:110-112), so every clause below is a claim that
    // the fused kernel beats cublas?getrsBatched at that shape, measured, at
    // saturating batch AND across the batch ladder. It was all-false at merge --
    // deliberately, because the kernel had not been measured against the vendor
    // then -- and it is not all-false now.
    //
    // THE WINDOW:
    //     nrhs <= 2    every type, every order        (clause A)
    //   + nrhs <= 4    float only                     (clause B)
    // and nothing wider. The COMPOSITION is never preferred at any width.
    //
    // ---- THE EVIDENCE, cell by cell ---------------------------------------
    //
    // docs/perf/lu.md#getrs-fused-window-evidence -- WP6's own harness (wp6_lu/bench/lubench6.cpp),
    // its own build scripts, its own cell format, both arms verified from the
    // printed route column on every row, medians of 5-7 reps, in-process host
    // oracle on every timed row, nothing under a kernel trace. Both arms reproduce
    // WP6's published BEFORE column first: cuBLAS to 2.90% over 60 shared cells,
    // the composition to 0.76% over 42.
    //
    // POOLED OVER ALL SEVEN SWEEPS, deduplicated on (type, n, nrhs, batch), both
    // arms' routes verified from the printed route column on every row:
    //
    //   CLAUSE A   286 cells   geomean 2.261x   MIN 1.116x   ZERO LOSSES
    //     nrhs = 1   142 cells   geomean 2.290x   min 1.242x
    //     nrhs = 2   144 cells   geomean 2.232x   min 1.116x  max 5.470x
    //   CLAUSE B    36 cells   geomean 1.611x   MIN 1.133x   ZERO LOSSES
    //   BOTH       322 cells   geomean 2.177x   MIN 1.116x   ZERO LOSSES
    //
    //   On the saturating grid alone (grid_*.csv, 4 types x 7 orders) that is
    //       nrhs = 1  geomean 2.117x, 28 wins of 28    (BEFORE: 0.256x, 0 of 28)
    //       nrhs = 2  geomean 2.173x, 28 wins of 28    (BEFORE: 0.331x, 2 of 28)
    //
    // AND FLAT IN BATCH, which is the part that took FOUR passes to establish:
    //       nrhs = 1  full ladders at n = 32,64,128,256,512,1024,2048, all four
    //                 types, every one FLAT-WIN          (flat_*, flat2_*, flat4_*)
    //       nrhs = 2  full ladders at the same seven orders, all four types,
    //                 every one FLAT-WIN                 (flat2_*, flat4_*)
    //       float 4   full ladders at the same seven orders, every one FLAT-WIN
    //   ZERO rows cross 1.0 anywhere inside this window, at any batch, in 133 + 115
    //   + 159 laddered cells.
    //
    // THE n = 32 AND n = 256 LADDERS EXIST ONLY BECAUSE A REVIEW CAUGHT THEIR
    // ABSENCE. Before flat4 there was NO order-32 ladder in this directory at any
    // width, so the whole small-n end of clause A -- including the window's own
    // stated minimum -- rested on the saturating batch point alone. That is the
    // one-cell over-fit this campaign keeps paying for, and it was one review away
    // from shipping again.
    //
    // THE THINNEST MARGIN IN THE WINDOW, named so a re-measurement knows where to
    // look first: cdouble n = 32 nrhs = 2, whose ladder runs 1.257 / 1.162 / 1.132
    // / 1.120 / 1.116 at batch 1024 -> 16384. It DECLINES with batch and flattens
    // rather than falling, so it is a flat win by the rule -- but it is the only
    // cell in 322 under 1.12x, and if any clause here goes red on another box it
    // is that one.
    //
    // CLAUSE B's own minimum is float n = 2048 nrhs = 4 at batch 4 (1.133x), the
    // most UNSATURATED cell in the sweep, i.e. the one where the vendor is most
    // flattered.
    //
    // WHY CLAUSE B IS FLOAT-ONLY, and it is the whole reason this window is not
    // simply "nrhs <= 4". The other three types CROSS 1.0 MID-LADDER at nrhs = 4:
    //       double  n=128   0.940x at batch 2048   (1.363x at 256, 1.111x at 8192)
    //       cfloat  n=1024  0.976x at batch 16
    //       cdouble n=128   0.980x at batch 1024
    //       cdouble n=1024  0.987x at batch 16
    //   -- and cdouble is 0.577x at n = 32 outright. A dip in the MIDDLE of a
    //   ladder cannot be closed by any boundary in n or in batch, which is what
    //   killed the two wider candidates (C5/C8/C9 in bench/README.md, 4 losses
    //   each at 0.940-0.987x). They were the leading proposal until the third
    //   flatness pass measured the interior orders; recording them as refuted is
    //   the result, not the window.
    //
    // WHAT THIS WINDOW COSTS, stated rather than buried: 84 measured cells that
    // the fused tier WINS go to the vendor, the largest at 3.944x (double n=1024
    // nrhs=4 batch 256), 3.144x, 3.097x, 2.880x, 2.745x behind it. They are given up because the clause that would capture
    // them dips below 1.0 elsewhere on its own ladder, and shipping a measured
    // loss to collect a measured win is not a trade this campaign makes. Widening
    // this window is real work -- a per-(type, order) predicate measured at more
    // orders, or a kernel fix for the dip -- and not a constant.
    //
    // NO CAPACITY TERM APPEARS HERE. supports() already refuses the shapes the
    // kernel cannot launch, resolve_route requires supports() AND preferred(), and
    // repeating a capability test in preferred() would be the geqrf defect
    // route_resolve.hh:60-70 records. The converse matters more: a SPEED term must
    // never migrate into supports(), because a forced route bypasses preferred()
    // and never supports() (route_resolve.hh:8-10, :101), so `nrhs > 4` in
    // supports() would make a pinned `native:cta` at nrhs = 8 fall through to
    // automatic(), reach cuBLAS, and pass green while measuring the wrong thing.
    //
    // THE COMPOSITION IS NEVER PREFERRED, and that is measured too, not an
    // oversight: at every width the fused tier serves it is 2.7x-24.6x behind the
    // fused tier and 0.26-0.46x of the vendor.
    //
    // =====================================================================
    // WP8-I2: THE WIDE-nrhs LADDER EXISTS NOW, AND IT SAYS TWO THINGS. THE
    // CLAUSE IS *NOT* SHIPPED HERE -- this is the recommendation and its CSV.
    //
    // (1) THE RECORDED HEADLINE WAS AN ARTEFACT OF READING ONE BATCH PER ORDER.
    //     "nrhs = 64 geomean 1.09x, nrhs = 128 geomean 1.48x, 9 and 4 losses of
    //     28" came from grid_*.csv, which carries exactly one saturating batch
    //     per order and NO ladder on the batch axis at any width >= 16. A full
    //     ladder -- 4 types x 5 orders x 4 widths x 7 batches (32 .. 8192),
    //     464 paired cells, docs/perf/lu.md#getrs-collapsed-permutation -- shows
    //     the composition's advantage FALLING MONOTONICALLY WITH BATCH at every
    //     type and every order, because below saturation neither arm is measuring
    //     its own speed. At float n=128 nrhs=128 the composition costs
    //     9.96 / 2.80 / 1.90 / 1.76 us per item at batch 32 / 128 / 256 / 512 and
    //     cuBLAS 38.2 / 10.2 / 5.59 / 3.31: an 8x batch for a 1.06x time on one
    //     arm and a 3.7x on the other. Read at saturation, the WALK's best
    //     candidate (float nrhs >= 128, 11 cells, geomean 1.761, zero losses) has
    //     MINIMUM 1.0436 and FAILS GATE-C. So on the arm that shipped in WP6
    //     there is no window at any width, for any type.
    //
    // (2) THE GATHER CHANGES THE ANSWER. With the permutation collapsed
    //     (src/extensions/getrs_native.cc, default at nrhs >= 16), re-measured on
    //     the saturated rungs -- 5 orders x 3 batches per type, TWO native passes
    //     and TWO vendor passes, medians of 11 reps, warm JIT,
    //     CUDA_VISIBLE_DEVICES pinned, zero foreign compute processes on every
    //     row, quoted at the WORSE pass, cross-pass median spread 1.0022 and
    //     worst 1.1208 over 270 arm-medians (docs/perf/lu.md#getrs-collapsed-permutation
    //     gap_*.csv, scored by clause.py into clause_summary.txt):
    //
    //       CANDIDATE                       cells  geomean    min  loss  <1.15  GATE-C
    //       float   nrhs >= 64                 30    2.837  1.765     0      0  PASS
    //       float   nrhs >= 128                15    3.444  1.850     0      0  PASS
    //       double  nrhs >= 128                15    1.865  1.279     0      0  PASS
    //       float + double nrhs >= 128         30    2.535  1.279     0      0  PASS
    //       RECOMMENDED (the union below)      45    2.467  1.279     0      0  PASS
    //       ---- and every wider candidate FAILS, with its refuting cell ----
    //       float   nrhs >= 32                 45    2.147  0.907     1      4  float   n=64  nrhs=32  b=4096
    //       double  nrhs >= 64                 30    1.563  0.998     1      1  double  n=64  nrhs=64  b=2048
    //       cfloat  nrhs >= 128                20    1.801  0.994     1      0  cfloat  n=64  nrhs=128 b=1024
    //       cfloat  nrhs >= 64                 34    1.657  0.787     2      0  cfloat  n=64  nrhs=64  b=2048
    //       cdouble nrhs >= 128                13    1.131  0.924     2      8  cdouble n=128 nrhs=128 b=1024
    //       cdouble nrhs >= 64                 26    1.002  0.693    12     11  cdouble n=128 nrhs=64  b=1024
    //       all types nrhs >= 128              63    1.925  0.924     3      8  cdouble n=128 nrhs=128 b=1024
    //
    //     THE RECOMMENDED CLAUSE, on GetrsShape::nrhs() -- B.cols(), NEVER
    //     order(), because a predicate written on the wrong extent inverts the
    //     window and that error was caught twice in WP7:
    //
    //       if (r.algo == Algorithm::Blocked) {
    //           if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;
    //           if constexpr (std::is_same_v<T, double>) return s.nrhs() >= 128;
    //           return false;                  // cfloat and cdouble earn nothing
    //       }
    //
    //     IT WOULD BE THE FIRST CLAUSE IN THIS TABLE NAMING Algorithm::Blocked,
    //     so the `if (r.algo != Algorithm::CTA) return false` early return below
    //     is load-bearing for clauses A and B and must be RELAXED rather than
    //     deleted: CTA must stay unpreferred above nrhs = 4 and Blocked must stay
    //     unpreferred below the per-type boundary, where it is 0.09-0.36x.
    //     native_tier_preferred is untouched -- CTA cannot reach nrhs >= 64 at
    //     all (kGetrsFusedMaxRhs = 8).
    //
    //     BATCH COVERAGE OF THE ADMITTED SET, stated exactly, because this is
    //     where every previous getrs window died. 45 cells measured DIRECTLY on
    //     three saturated rungs of each of five orders, two passes each side. A
    //     further 58 admitted cells at the ladder's other rungs (batch 32..8192)
    //     are covered by a BOUND rather than a measurement: the vendor arm did
    //     not move in this pass, and the gather's own A/B measured MINIMUM 1.0004
    //     with ZERO cells below 1.00 over 80 cells and two passes, so
    //     post_ratio >= walk_ratio at every admitted cell -- and all 58 already
    //     clear 1.15 on the WALK ladder (min 1.1933, geomean 2.1616). ZERO
    //     admitted cells are left uncovered by measurement or bound.
    //
    // (2b) cfloat WAS IN THIS CLAUSE UNTIL THE BOUND'S OWN GAP WAS MEASURED, and
    //     that is the methodological result of this pass. `cfloat nrhs >= 128`
    //     scored 15 cells, geomean 1.974, MIN 1.482, zero losses -- a clean PASS
    //     -- on the directly measured rungs. The coverage bound then named FIVE
    //     admitted cfloat cells it could not cover, because the WALK ladder was
    //     itself below 1.15 there. Measuring those five
    //     (docs/perf/lu.md#getrs-collapsed-permutation) produced
    //         cfloat n=64 nrhs=128 batch=1024 = 0.9944  (0.9969 / 0.9944, two passes)
    //     with 1.2901 at batch 512 and 1.4824 at batch 2048 ON EITHER SIDE OF IT.
    //     A dip in the MIDDLE of a ladder cannot be closed by any boundary in
    //     batch, in order or in nrhs -- the C5/C8/C9 failure mode, found again.
    //     It was invisible to every candidate scored before the gap sweep
    //     existed, and it is the reason the clause is float+double and not
    //     non-cdouble. Widening to include cfloat costs 20 cells of geomean 1.80
    //     and buys one measured loss.
    //
    // (3) cdouble IS REFUTED AT EVERY WIDTH, and that is this pass's negative
    //     result rather than a gap in it. At nrhs = 128 it is 13 cells, geomean
    //     1.131, minimum 0.9238, with TWO outright losses and EIGHT more between
    //     1.00 and 1.15; at nrhs = 64 it is 12 losses of 13. The losses are
    //     CLUSTERED ON THE TYPE, not mid-ladder -- cdouble n=128 nrhs=64 loses at
    //     batch 1024, 2048 AND 4096, i.e. the whole ladder -- so a (type, nrhs)
    //     predicate CAN exclude them, which is why the recommendation is per-type
    //     rather than a single scalar. That is the answer to the question this
    //     pass was set: the recorded 9 and 4 losses are BOTH. On the type axis
    //     they cluster (cdouble is a whole-type loss at every width, and its
    //     n=128 nrhs=64 ladder loses at every rung); on the ORDER axis inside
    //     the remaining types they are interior, which is why no boundary in n
    //     appears anywhere in the recommendation. What is left losing for cdouble is the
    //     trsm/GEMM arm and not the permutation: the gather is worth only
    //     1.04-1.26x there against 1.12-2.79x for float, exactly as the
    //     (32 + sizeof(T)) / sizeof(T) sector inflation predicts.
    // =====================================================================
    // =====================================================================
    // WP8 ROUTING PASS: CLAUSE C LANDS. The recommendation above is applied
    // VERBATIM and its CSV is docs/perf/lu.md#getrs-composition-window-evidence, scored
    // in this pass by docs/perf/lu.md#getrs-composition-window-evidence's sibling reader.
    //
    // WHAT CHANGED HERE STRUCTURALLY. `if (r.algo != Algorithm::CTA) return
    // false` was RELAXED, not deleted -- clauses A and B depend on it to keep
    // the COMPOSITION unpreferred at the narrow widths, where it is 0.09-0.36x
    // of the vendor. The composition is now preferred at exactly the widths
    // where it was measured ahead, and nowhere else.
    //
    // THE AXIS IS GetrsShape::nrhs(), WHICH IS B.cols() AND NEVER order().
    // Spelled on the wrong extent this clause inverts: it would admit every
    // wide-order narrow-RHS call, which is the regime where the composition is
    // 0.09x. That error was caught twice in WP7 and once here in review.
    //
    // THE 45 ADMITTED CELLS, TRANSCRIBED (ratio = vendor_med / native_med,
    // QUOTED = the worse of two passes each side; zero losses, zero cells below
    // 1.15, geomean 2.467, min 1.2791):
    //
    //   float nrhs=64   n=64  b2048 2.076  b4096 2.819  b8192 4.074
    //                   n=128 b1024 1.871  b2048 1.775  b4096 3.167
    //                   n=256 b1024 1.849  b2048 2.426  b4096 3.861
    //                   n=512 b512  1.952  b1024 2.067  b2048 2.620
    //                   n=1024 b128 2.175  b256  1.914  b512  1.765
    //   float nrhs=128  n=64  b2048 3.683  b4096 5.074  b8192 6.335
    //                   n=128 b1024 2.609  b2048 3.931  b4096 5.007
    //                   n=256 b1024 3.121  b2048 4.463  b4096 4.888
    //                   n=512 b512  2.501  b1024 2.839  b2048 3.689
    //                   n=1024 b128 2.769  b256  2.049  b512  1.850
    //   double nrhs=128 n=64  b2048 2.100  b4096 2.519  b8192 2.766
    //                   n=128 b1024 1.803  b2048 2.422  b4096 2.809
    //                   n=256 b1024 1.666  b2048 2.044  b4096 2.244
    //                   n=512 b512  1.408  b1024 1.461  b2048 1.655
    //                   n=1024 b128 1.438  b256  1.341  b512  1.279
    //
    // AND THE CELL THAT REFUSES EACH WIDER CLAUSE, so none is rediscovered:
    //   float  nrhs >= 32   0.9069 at n=64  nrhs=32  b=4096
    //   double nrhs >= 64   0.9984 at n=64  nrhs=64  b=2048
    //   cfloat nrhs >= 128  0.9944 at n=64  nrhs=128 b=1024 -- and that cell is
    //     a dip in the MIDDLE of its own ladder (1.2901 at b=512, 1.4824 at
    //     b=2048 on either side of it), so no boundary in batch, order or nrhs
    //     reaches it. cfloat scored 15 cells / geomean 1.974 / min 1.482 on the
    //     directly measured rungs and was in this clause until the gap sweep
    //     existed.
    //   cdouble nrhs >= 128 0.9238 at n=128 nrhs=128 b=1024, with 2 losses and
    //     8 more cells between 1.00 and 1.15. cdouble is refuted at every width.
    //
    // ---- WHAT THIS PASS ADDED: A CLEAN RE-MEASURE, THE BATCH AXIS, AND A
    // ---- CORRECTION THAT NEARLY COST HALF THE CLAUSE
    //
    // THE RE-MEASURE FIRST. Everything above was re-run on device 1 with nothing
    // else on the box (docs/perf/lu.md#getrs-composition-window-evidence, pair_cells.sh: the
    // two arms are two BUILDS run back to back on each cell, 11 reps, median,
    // host oracle per row, resolved route checked per arm, foreign count 0). It
    // reproduces I2's figures cell for cell -- float n=512 nrhs=64 b=512 reads
    // 1.9703 against 1.9517, n=512 nrhs=128 b=512 reads 2.5344 against 2.5011,
    // cdouble n=128 nrhs=128 b=1024 reads 0.9196 against 0.9238 -- and scores:
    //     float  nrhs >= 64,  batch >= 128   22 cells, geomean 3.138, MIN 1.7695
    //     double nrhs >= 128, batch >= 128   15 cells, geomean 1.979, MIN 1.2858
    //     union                              37 cells, MIN 1.2858, ZERO losses,
    //                                        ZERO cells below 1.15
    // I2's own named risk -- "double n=1024 nrhs=128 batch=512 at 1.2791 is the
    // LAST rung measured at that order and the ladder is falling; it costs
    // 8.6 GB to measure batch 1024 and it was not measured" -- is closed: that
    // cell measures 1.3070 at batch 1024, i.e. the ladder turns back up.
    //
    // (a) THE CLAUSE GAINS A BATCH FLOOR THAT I2 DID NOT HAVE, AND IT IS A
    //     CONSERVATIVE ONE. I2's own note recorded the gap: "the recommended
    //     clause carries no batch floor -- so if it lands, it routes small-batch
    //     wide-nrhs shapes on the strength of a bound rather than a
    //     measurement". Every one of its 45 cells, and every one of the 37 this
    //     pass re-measured, is at batch >= 128.
    //
    //     WHAT THE LOW END ACTUALLY DOES: at nrhs = 128 the composition still
    //     WINS at batch 64 and 32 -- float 5.93 / 5.96 (n=64 / n=128), 5.60 at
    //     n=256, 4.71 at n=512, 3.87 at n=1024; double 4.31 / 4.05 / 3.56 -- so
    //     the floor at 128 GIVES UP measured wins rather than excluding measured
    //     losses. It is set there anyway, for two reasons that are stated rather
    //     than glossed: (i) below 32 the region is genuinely ragged, and the
    //     only readings there come from a CONTAMINATED sweep (0.055x-0.33x at
    //     batch 1-2), so the rung immediately under any lower floor is not
    //     bracketed by a trustworthy non-winner; (ii) at nrhs = 64 -- the other
    //     half of the clause -- the low end is not measured at all. A floor that
    //     is only conservative cannot admit a loss, which is the property
    //     GATE-C actually needs. Moving it down is one cheap sweep
    //     (docs/perf/lu.md#open-debts) and is named as open work.
    //
    // (b) A CORRECTION THAT NEARLY COST THE float nrhs >= 64 HALF OF THIS
    //     CLAUSE, recorded because the same trap will be there for the next
    //     pass. WP8's first sweep ran an LU harness on device 1 while a gemv
    //     harness ran on device 0, on the reasoning that two cards are two
    //     machines. They are not -- same NUMA node, same CPU affinity mask, one
    //     UVM driver, and lubench6 runs on managed memory -- and the per-row
    //     foreign() guard cannot see it, because --query-compute-apps is PER
    //     DEVICE and neither process is on the other's card. rel_sd cannot see
    //     it either: the contaminated rows read 0.012-0.017. Under that
    //     contention getrs float n=1024 nrhs=64 batch=1024 measured 0.8859 --
    //     an apparent outright LOSS inside the admitted set, exactly the
    //     mid-ladder failure this clause family has died of twice before. Run
    //     ALONE the same cell measures 1.9563. The narrowing it would have
    //     forced (float nrhs >= 128, giving up 15 cells at 1.77x-4.07x) was
    //     written and then reverted. SERIALISE THE BOX; a second card is not a
    //     second machine.
    // =====================================================================
    static bool preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return false;

        // The COMPOSITION -- clause C. Preferred only at the widths where it was
        // measured ahead of the vendor, per type. Everywhere else it stays the
        // arm the fused tier replaces, reached in a vendor-free build through
        // native_tier_preferred below and never through here.
        if (r.algo == Algorithm::Blocked) {
            // THE BATCH FLOOR IS CONSERVATIVE, AND KNOWN TO BE. See the WP8 note
            // above: at nrhs = 128 the composition still wins at batch 32 and 64
            // (3.56x-5.96x), so this line gives up measured wins. It is here
            // because below 32 the region is ragged and the only readings are
            // from a contaminated sweep, and because nrhs = 64 -- the other half
            // of the clause -- has no low-batch ladder at all.
            if (s.batch < 128) return false;
            if constexpr (std::is_same_v<T, float>)  return s.nrhs() >= 64;
            if constexpr (std::is_same_v<T, double>) return s.nrhs() >= 128;
            return false;   // cfloat and cdouble earn nothing at any width
        }

        if (r.algo != Algorithm::CTA) return false;

        if (s.nrhs() <= 2) return true;                  // clause A

        if constexpr (std::is_same_v<T, float>) {        // clause B
            if (s.nrhs() <= 4) return true;
        }
        return false;
    }

    // ---- THE NATIVE-VS-NATIVE TIE-BREAK, AND IT IS MEASURED ---------------
    //
    // Consulted ONLY in the vendor-free walk (route_resolve.hh:113-127), so
    // declaring it moves NOTHING in a vendor-present build -- which is exactly
    // why it is the right instrument for this question and preferred() is not.
    // WITHOUT it the vendor-free choice is decided entirely by kGetrsOrder,
    // which is static and cannot follow a crossover.
    //
    // THE MEASUREMENT. Prototype, vendor-free build, saturating batch, the fused
    // kernel and the composition INTERLEAVED IN ONE PROCESS against a host oracle
    // (docs/perf/lu.md#the-fused-narrow-rhs-getrs and grid_big.csv). Ratio is
    // composed_ms / fused_ms, so > 1 means the FUSED tier is ahead:
    //
    //   nrhs      1      2      4      8       16
    //   float   3.6-8.1 3.8-7.1 2.6-5.9 2.7-3.8  1.06-2.48
    //   double  4.0-17.7 2.7-10.5 1.4-5.4 1.7-2.8 0.55-1.28
    //   cfloat  3.8-8.1 3.7-7.7 3.3-6.0 1.8-3.8  0.58-1.26
    //   cdouble 4.1-24.6 12.8-18.0 6.4-8.1 3.2-3.5 1.33-1.75
    //
    // THE FUSED TIER IS AHEAD AT EVERY CELL WITHIN ITS OWN CAPABILITY (nrhs <= 8,
    // 51 cells, worst 1.11x at float n=2048 nrhs=8), so there is no crossover to
    // encode and this predicate is "CTA wherever supports() admits it". The
    // nrhs = 16 column is where it would turn -- 0.55x for double and 0.58x for
    // cfloat at n = 512, because the resident RHS has grown large enough to halve
    // the resident blocks per SM -- and that column is OUTSIDE supports() by
    // kGetrsFusedMaxRhs. If that constant is ever raised, THIS predicate is what
    // has to gain a window; raising it without one would re-create the geqrf
    // defect route_resolve.hh:60-70 records.
    //
    // NOT A CORRECTNESS GATE. Both arms stay fully supports()-able wherever they
    // can run, which is what keeps a pinned `native:blocked` actually running the
    // composition instead of falling through to automatic() and measuring the
    // thing it was pinned away from (route_resolve.hh:165 -> :175).
    static bool native_tier_preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return true;
        static_cast<void>(s);
        switch (r.algo) {
            case Algorithm::CTA:     return true;
            case Algorithm::Blocked: return false;
            default:                 return true;
        }
    }

    static constexpr const Route* order_begin() { return kGetrsOrder; }
    static constexpr const Route* order_end() {
        return kGetrsOrder + (sizeof(kGetrsOrder) / sizeof(kGetrsOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure. See resolve_getrf_route's note for why
// `vendor_available` is passed EXPLICITLY at both call sites and why this --
// rather than resolve_route_uninstrumented -- is what puts getrs in the coverage
// table.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_getrs_route(Route forced, const GetrsShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getrs, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
