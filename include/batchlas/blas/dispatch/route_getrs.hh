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
//                     nrhs window measured in experiments/wp6_perf/bench/, so
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
// experiments/wp6_lu/bench/run_cells.sh:37 and kernels/run_grid.sh:39 export one
// value into all three LU variables at once -- is measuring a different getrs
// today than when it was recorded. Pin `native:blocked` to mean what `native`
// used to mean. This is a measurement-comparability trap, not a correctness one,
// and tests/route_vocabulary_tests.cc's BareOriginResolvesToASpecificAlgorithm
// is where it is asserted.
//
// THE TWO MEASUREMENTS, both against cublas?getrsBatched at saturating batch, in
// process, against a host oracle:
//
//   THE COMPOSITION (experiments/wp6_lu/baseline/):
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
//   THE FUSED TIER (experiments/wp6_getrs/proto/grid_nv.csv, grid_big.csv):
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
    // experiments/wp6_perf/bench/ -- WP6's own harness (wp6_lu/bench/lubench6.cpp),
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
    // fused tier and 0.26-0.46x of the vendor. It DOES beat the vendor at nrhs =
    // 64 (geomean 1.09x) and nrhs = 128 (1.48x) -- but with 9 and 4 losses of 28
    // respectively and NO batch ladder anywhere on that axis, so it is not a
    // window yet. That is left open in experiments/wp6_perf/README.md rather than
    // shipped.
    static bool preferred(Route r, const GetrsShape& s) {
        if (!is_native(r)) return false;

        // The composition is the arm the fused tier replaces. It is never the
        // default anywhere; a vendor-free build reaches it through
        // native_tier_preferred below, not through here.
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
    // (experiments/wp6_getrs/proto/grid_nv.csv and grid_big.csv). Ratio is
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
