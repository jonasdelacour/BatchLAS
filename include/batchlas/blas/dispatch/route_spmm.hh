#pragma once

// SPMM's routing table.
//
// PURE, in route_resolve.hh:19-21's sense: everything here reads ONLY its
// arguments -- no getenv, no SYCL query, and no read of any operand's data.
// Anything that has to ask the device, the environment or a kernel translation
// unit lives in src/backends/spmm_route.hh instead.
//
// THE SPLIT RULE, restated from route_gemm.hh:25-28, route_trsm.hh and
// route_gemv.hh:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the alias table (route_env.hh), not here.
//
// For spmm the rule has teeth in two specific places, and between them they are
// the whole work package -- see the Direct arm of supports() and the note above
// preferred().
//
// THE ENV VARIABLE IS BATCHLAS_SPMM_ROUTE, AND NO route_env.hh EDIT CREATES IT.
// parse_route_env synthesises the name from op_env_stem(Op::spmm)
// (route_env.hh:214-217), whose stem comes from op_name (route.hh:188) and
// already spells "spmm". legacy_variable_for has no Op::spmm case and correctly
// falls to `default: return {}` (route_env.hh:109-121) -- DO NOT ADD ONE, it
// would invent a legacy spelling that never shipped. Values that reach this
// table: "direct" (a bare algorithm implies Origin::Native), "native", "vendor".
// Unset means {Auto, Auto}.
//
// AND THE TRAP ON THE OTHER SIDE, WHICH IS SILENT AND WHICH THIS CAMPAIGN HAS
// PAID FOR REPEATEDLY. Pinning a route the shape cannot take does NOT fail and
// does NOT warn: resolve_route falls through to automatic() (route_resolve.hh:
// 165-176), which in a vendor-present build IS the vendor. So
// BATCHLAS_SPMM_ROUTE=cta resolves to {Native, CTA}, supports() rejects it --
// there is no CTA body -- and the run silently measures cuSPARSE. A MISSPELLED
// value behaves the same way and is worse: parse_route_value fails, the
// resulting ParsedRouteEnv::unparsed is discarded at spmm_route.hh's
// `parsed.found ? parsed.route : legacy_unset_default`, and every decision goes
// to the vendor with no message. That discard is campaign-wide
// (gemv_route.hh:149-151, and trsm/getrs/potrf identically), not an spmm
// invention.
//
// THE RESOLVED-ROUTE COLUMN IS THE ONLY WAY TO KNOW WHICH ARM RAN. Use
// BATCHLAS_COVERAGE_OUT and read the `reached` rows. A kernel being linked is
// not evidence that it ran.
//
// AND ONE THING THE ROUTE COLUMN CANNOT TELL YOU. {Native, Direct} names THREE
// kernel bodies in src/sycl/spmm_native.cc -- the NoTrans gather, and the
// scale+scatter pair that together serve Trans and ConjTrans -- and the launcher
// picks between them on transA. That choice is deliberately below the routing
// vocabulary: it is a decomposition, not an algorithm, exactly as
// {Native, Direct} names both of gemv's Direct bodies. transA IS in coverage's
// variant_key (coverage.cc:52-58), so gather-vs-scatter stays separable in
// scripts/route_diff.sh; which of the two scatter kernels ran does not, and the
// way to establish that is a deliberate break that is red only for that body.
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE.
//
//     s.m     = A.rows()        rows of A AS STORED
//     s.k     = A.cols()        cols of A AS STORED
//     s.n     = C.cols()        the dense width, i.e. nrhs
//     s.batch = A.batch_size()
//     s.transA, s.transB        BOTH set, both meaningful
//
// and the three derived extents, which are the ONLY spellings a predicate should
// use, because which of m and k is the output extent SWAPS with transA:
//
//     nrhs()       == n                          (columns of B and of C)
//     out_rows()   NoTrans -> m   Trans -> k     (rows of C)
//     red_rows()   NoTrans -> k   Trans -> m     (rows of B, the reduction)
//
// THIS IS NOT PEDANTRY. It is the exact error the design review caught in a
// rejected staged tier: sizing a staged B slab over `m` when B has red_rows()
// rows reads past the operand for every non-square A, and it is invisible in a
// worked example because every worked example was square.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>
#include <batchlas/blas/enums.hh>

#include <complex>
#include <type_traits>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// SPMM reads three things OpShape has no field for. It reads NOTHING ELSE that
// OpShape already carries.
//
// MatrixFormat IS A RUNTIME FIELD HERE, NOT A TEMPLATE PARAMETER. RouteTable's
// primary is `template <Op O, typename T> struct RouteTable;`
// (route_resolve.hh:29-30) and resolve_route deduces Shape as a third FUNCTION
// template parameter (route_resolve.hh:196-197), so an op-specific shape struct
// is the supported extension point -- the same one GesvdShape and TrsmShape use.
// Adding a defaulted third parameter to RouteTable would silently narrow every
// existing two-argument specialisation to F == Dense.
//
// DO NOT RE-DECLARE ANY OpShape FIELD HERE. OpShape (route.hh:222-262) already
// carries op/scalar/backend, m/n/k/batch, transA/transB/uplo/side/diag/precision,
// heterogeneous_batch, is_gpu, max_sub_group and compute_units. resolve_route
// SLICES this struct to OpShape on the way into the coverage table
// (route_resolve.hh:212), so a shadowing member would be written by the shape
// builder and then NOT copied: every spmm coverage row and every route_diff row
// would report the default, and spmm's genuinely different access patterns would
// collapse into ONE first-writer-wins row. That is coverage.cc:40-58's stated
// failure mode.
// ---------------------------------------------------------------------------
struct SpmmShape : OpShape {
    // The storage format of A. Only CSR has native bodies; see supports().
    MatrixFormat format = MatrixFormat::Dense;

    // Build capabilities, asked of the kernel TU by the shape builder so the
    // table describes the BUILD and not the design (route_trsm.hh:62-97's
    // reasoning for trsm_cta_max_n). FALSE means "no such native body in this
    // build" and correctly makes the native route unsupported rather than
    // selectable-but-unimplemented -- the TrsmShape::cta_max_n == 0 convention.
    //
    // Two flags rather than one because they are independent capabilities and
    // because they serve disjoint halves of the transA axis.
    bool gather_available = false;   // the transA == NoTrans body
    bool scatter_available = false;  // the transA != NoTrans bodies (scale + scatter)

    // Derived, non-shadowing. See the field-mapping note above: out_rows() and
    // red_rows() SWAP with transA, and a predicate that spells m or k directly
    // is testing a different axis depending on the transpose.
    int64_t nrhs() const { return n; }
    int64_t out_rows() const { return transA == Transpose::NoTrans ? m : k; }
    int64_t red_rows() const { return transA == Transpose::NoTrans ? k : m; }
};

// NO nnz FIELD, DELIBERATELY, AND IT IS NOT AN OVERSIGHT.
//
// MatrixView<T, CSR>::nnz() is the per-item CAPACITY, equal to the batch
// MAXIMUM rather than to any one item's count (matrix.hh:1071-1074), so nnz/rows
// is several times wrong on a heterogeneous batch and is a misleading routing
// axis. The honest per-item spelling, nnz(b), READS row_offsets, and
// matrix.hh:1081-1086 states that a MatrixView over sycl::malloc_device memory
// is not host-reachable -- and the same shape builder runs in spmm_buffer_size,
// where a data read is an immediate segfault rather than a wrong route. No
// predicate below needs it, so the shape does not carry it.

// TWO ENTRIES, AND ONE NATIVE ROUTE NAMING THREE BODIES.
//
// The body is chosen in the LAUNCHER on transA -- gemv's precedent exactly,
// where {Native, Direct} names both GemvDirectNKernel and GemvDirectTKernel.
// Body selection is a decomposition, not an algorithm.
//
// This is why route.hh's Algorithm enum (:67-87), to_string(Algorithm)
// (:159-176) and route_env.hh's parse_algorithm_word (:55-71) need ZERO edits.
// A new sparse algorithm name would need all three in lockstep, and skipping any
// one makes BATCHLAS_SPMM_ROUTE=<newname> fail to parse SILENTLY, because
// ParsedRouteEnv::unparsed is discarded by every adapter in the tree.
//
// SINCE THE preferred() CLAUSE LANDED, the order is also walked in the
// vendor-PRESENT build: automatic()'s first walk (route_resolve.hh:109-112)
// takes the first route that is both supported AND preferred.
//
// AND A CLAIM THAT WAS WRITTEN HERE FIRST AND THEN MEASURED FALSE, RECORDED
// BECAUSE IT IS THE KIND OF THING THAT GETS ASSERTED AND BELIEVED. Reversing
// this array does NOT send the admitted shapes back to cuSPARSE. preferred()
// returns false for {Vendor, Auto} (its first line is `if (!is_native(r) ...)`),
// so the first walk SKIPS the vendor entry wherever it sits and still lands on
// {Native, Direct}; the vendor-free walk is likewise unaffected, because it
// filters on is_native() and there is exactly one native entry. The reversal was
// applied, rebuilt and run: exactly ONE case goes red,
// RouteSpmm.OrderIsExactlyTwoEntries, and it goes red structurally rather than
// through any decision.
//
// What WOULD invert the default is a preferred() that answered true for the
// vendor entry -- pinning cuSPARSE as "preferred" and making the native route
// unreachable no matter what the order says. That is the property
// RouteSpmm.PreferredIsFalseForEveryOtherRouteAndFormat pins, and it is a
// different mistake from the one this array can make.
inline constexpr Route kSpmmOrder[] = {
    {Origin::Native, Algorithm::Direct},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::spmm, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a wrong answer or fail to
    // launch", never "the kernel would be slow". A forced route bypasses
    // preferred() but NEVER supports() (route_resolve.hh:165), and a forced
    // route that supports() rejects falls through to automatic() SILENTLY, with
    // no warning and no diagnostic (:167-176) -- so a shape or speed cutoff here
    // makes the benchmark that pinned a route measure a different arm and print
    // nothing about it.
    static bool supports(Route r, const SpmmShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. FORMAT. Only CSR bodies exist. No other format is instantiated in
        //    entry_points/sparse.cc today, but this is a correctness gate rather
        //    than a bet on that staying true: a Dense or COO view reaching a CSR
        //    kernel is a wrong answer, not a slow one.
        if (s.format != MatrixFormat::CSR) return false;

        // 2. HETEROGENEOUS BATCH. One launch covers the batch with a single
        //    (ld, stride) tuple per dense operand, so per-item extents would be
        //    read at the wrong addresses. spmm has no analogue of gemm's
        //    heterogeneous walker (gemm_heterogeneous.hh).
        //
        //    NOTE THAT THIS GATE IS ABOUT B AND C, NOT ABOUT A. active_rows_ and
        //    active_cols_ are Dense-only (matrix.hh:1036-1042), so a CSR view is
        //    never heterogeneous in that sense and per-item variation is
        //    expressible ONLY as nnz(b) -- which every body handles exactly,
        //    through the row-offset array, and which therefore needs no routing
        //    axis.
        if (s.heterogeneous_batch) return false;

        // 3. DEGENERATE GEOMETRY. m == 0 or n == 0 is NOT here: those are legal
        //    calls that the launcher quick-returns on before any submit, without
        //    touching C. A NEGATIVE extent or an empty batch, on the other hand,
        //    has no launch geometry at all.
        if (s.m < 0 || s.n < 0 || s.k < 0 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Direct:
                // THE CAPABILITY FLAG FOR THE BODY THAT WOULD ACTUALLY RUN, AND
                // NOTHING ELSE.
                //
                // *** NO GPU GATE. *** This one line is a WP8 deliverable and it
                // is deliberately unlike every native tier in this campaign
                // except gemv's. Every body is a plain loop: zero local memory,
                // no work-group or sub-group collective, no required sub-group
                // size -- nothing a CPU device cannot execute. And the rows it
                // buys are real, not hypothetical: build-novendor has
                // BATCHLAS_HAS_HOST_BACKEND 1 while BATCHLAS_HAS_LAPACKE and
                // BATCHLAS_HAS_CBLAS are both 0 (the split comes from
                // cmake/BatchLASDependencies.cmake:310-318, which sets the host
                // backend from the LIBRARIES being found while the other two
                // additionally require their ENABLE options), so the
                // Backend::NETLIB spmm symbol EXISTS and throws NoRouteError
                // today. Add `if (!s.is_gpu) return false;` here and the
                // vendor-free walk finds no route for those rows, the facade
                // throws for them exactly as it does now, and half the burn-down
                // this work package exists for moves by ZERO. s.is_gpu IS
                // recorded by the shape builder, for the coverage row, and is
                // deliberately never read here.
                //
                // *** NO TRANSPOSE REFUSAL. *** All nine (transA, transB)
                // combinations are served, which is what keeps the transB=Trans
                // layout lever available to callers.
                //
                // *** NO nnz, DENSITY OR LOCAL-MEMORY CLAUSE. *** There is no
                // local memory in this design and no density axis in this shape;
                // see the note on the absent nnz field above.
                return (s.transA == Transpose::NoTrans) ? s.gather_available
                                                        : s.scatter_available;

            default:
                // Including Algorithm::Auto: a bare "native" names no body, and
                // resolve_route walks the order restricted to the requested
                // origin to pick one (route_resolve.hh:153-163).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // NO LONGER ALL-FALSE. One clause ships, and it moves the default in EVERY
    // build.
    //
    // READ THIS FIRST. preferred() is consulted by automatic()'s FIRST walk
    // (route_resolve.hh:109-112), which runs REGARDLESS of vendor_available. So
    // the clause below is not a vendor-free tier choice: it takes spmm away from
    // cuSPARSE in the vendor-present build, for the shapes it admits. That is
    // the intent, and the evidence for it is docs/perf/spmm.md#raw-evidence -- 7,536
    // timed rows over 9 sweeps, every sweep run as two fully independent passes,
    // one route per process, route proved from BATCHLAS_COVERAGE_OUT rather than
    // from the environment variable.
    //
    // THE ACCEPTANCE GATE THIS CLAUSE WAS HELD TO, restated from the campaign:
    // a clause may move a cell only if worst-of-two-passes t_native/t_vendor
    // <= 1.10 on EVERY cell it moves, at saturation, batch >= 128, fitted on the
    // (m, nnz/row, nrhs, batch, pattern, beta, transB) grid rather than at a
    // fixed footprint, with every boundary bracketed by measured rows on BOTH
    // sides and never sitting at the edge of the sampled range.
    //
    // WHAT THE CLAUSE MOVES: 176 saturated batch >= 128 cells, worst-of-two
    // 0.968, median 0.445, best 0.032
    // (docs/perf/spmm.md#the-gather-window, built from
    //  docs/perf/spmm.md#the-gather-window and scl{1,2}/joined.csv).
    // At unambiguous DRAM residency the gather reads 906-931 GB/s -- 90-92% of
    // this part's 1008 GB/s roof -- for all four scalar types, against
    // cuSPARSE's 120-366 GB/s (docs/perf/spmm.md#the-dram-roof). nsys says both
    // arms are > 93% GPU-kernel time, so this is a kernel result and not a
    // host-overhead one: cuSPARSE launches THREE kernels per spmm call and
    // re-runs csr_partition_kernel, 36% of its GPU time, on every call
    // (docs/perf/spmm.md#the-nsys-split).
    static bool preferred(Route r, const SpmmShape& s) {
        // ROUTE AND FORMAT: the clause speaks only for the route that has the
        // measured bodies. {Vendor, Auto} and any other native algorithm fall
        // through to the mere-support walk exactly as before.
        if (!is_native(r) || r.algo != Algorithm::Direct) return false;
        if (s.format != MatrixFormat::CSR) return false;

        // NO BATCH TERM, AND THAT IS MEASURED RATHER THAN ASSUMED.
        //
        // preferred() is consulted on EVERY call, while the acceptance gate is
        // stated at batch >= 128. That mismatch was a real outstanding caveat,
        // not a formality, so the batch 1..64 corner was swept separately --
        // 5 shape families x 4 scalar types x 2 column patterns x 2 betas x both
        // transB x batch {1,2,4,8,16,32,64,128}, twice, one route per process
        // (docs/perf/spmm.md#the-batch-axis-has-no-floor, sb1/joined.csv,
        //  sb2/joined.csv, smallbatch.csv, smallbatch.txt).
        //
        // Under THIS clause, 0 of 174 admitted rows at batch <= 64 exceed the
        // 1.10 gate in both passes, and exactly 1 of 174 costs the caller any
        // time at all. Per rung (worst-of-two): b=1 0.992, b=2 0.981, b=4 1.078,
        // b=8 0.956, b=16 0.966, b=32 0.978, b=64 0.967, b=128 0.964.
        //
        // A BATCH FLOOR NEEDS A MEASURED NON-WINNER OUTSIDE THE GATE TO BRACKET
        // IT AND THERE IS NONE. The worst cell anywhere in the region a floor
        // would cut off is 1.078 -- complex<float>, m=4096, nnz/row=16, nrhs=50,
        // scattered, transB=NoTrans, beta=0, BATCH=4, reproduced 1.078 / 1.078
        // across two processes, +13.46 us/call -- and it is INSIDE the gate. It
        // is also non-monotonic in batch on its own cell (b=2 0.977, b=4 1.078,
        // b=8 0.956), i.e. an unsaturated launch/occupancy artefact rather than
        // a structural loss. Adding `s.batch >= N` would forfeit the 0.099-0.5
        // region below batch 64 for nothing -- the getri mistake, made in the
        // opposite direction.
        //
        // HONESTY LABEL ON THOSE NUMBERS: below batch ~64 the timed region is
        // launch latency plus, on the vendor arm, spmm_vendor's unhoistable
        // per-call host chain (setStream, the SpmmCsrBatchPlan host walk, the
        // cusparseSpMM_bufferSize re-query, the BumpAllocator carve). The
        // small-batch ratios are admissible as evidence of NO HARM and as
        // nothing else; they are not kernel results and must not be quoted as
        // any. The 0.968/0.445 headline comes from the saturated grid alone.

        // THE GATHER ONLY -- AND THE TRANSPOSED REFUSAL BELOW IS A MEASURED
        // REFUSAL, NOT AN OMISSION AND NOT AN UNMEASURED CORNER.
        //
        // transA == NoTrans selects spmm_gather (src/sycl/spmm_native.cc body
        // 1); transA != NoTrans selects the scale+scatter pair (bodies 2 and 3).
        // The scatter arm was measured on the SAME grid, at the same saturation,
        // and it LOSES: 169 of 458 saturated cells above the 1.10 gate, median
        // 1.030, worst 3.011 (cdouble tA=1 m=4096 nnz/row=16 nrhs=50 b=512 tB=0
        // beta=0 banded, p1=3.000 p2=3.011 -- verdict.txt).
        //
        // REJECTED WIDER CANDIDATES ON THE SCATTER ARM, each with the cell that
        // refutes it (verdict.txt, from pass{1,2} and scl{1,2}):
        //   transA != NoTrans, unconditional  FAILS 169/458, worst 3.011
        //   ... AND nrhs <= 4                 FAILS  11/204, worst 1.208
        //                 refuted by cdouble m=2048 nnz/row=16 nrhs=2 b=512
        //                 tB=0 beta=0 banded, p1=1.208 p2=1.207
        //   ... AND nrhs <= 2                 FAILS   5/151, worst 1.208, same cell
        //   ... AND nrhs <= 1                 FAILS   2/60,  worst 1.132
        //                 refuted by cdouble m=2048 nnz/row=16 nrhs=1 b=1024
        //                 tB=0 beta=0 banded, p1=1.130 p2=1.132
        //   ... AND nrhs <= 2 AND type != cdouble   PASSES (worst 1.023) and is
        //                 STILL REJECTED: the real boundary is nnz/row, not
        //                 nrhs -- cdouble at 3 nnz/row is 0.390 at nrhs=1 while
        //                 cdouble at 16 nnz/row is 1.043 at the same nrhs
        //                 (bnd_scatter_a, README table) -- and SpmmShape carries
        //                 no nnz field, deliberately and unfixably (nnz() is a
        //                 per-item CAPACITY and the honest per-item spelling
        //                 reads device memory in spmm_buffer_size, where that is
        //                 a segfault). A clause whose true axis the shape cannot
        //                 see is fitted, not measured. It would also move 111
        //                 cells of a decomposition that has ZERO in-tree C++
        //                 callers today.
        // So: no shippable transposed window exists. This line is that result.
        if (s.transA != Transpose::NoTrans) return false;

        // THE ONE MEASURED NON-WINNER ON THE GATHER ARM, EXCLUDED BY TYPE AND
        // transB TOGETHER.
        //
        // complex<float> with a transposed dense operand runs 1.71-1.73x SLOWER
        // than cuSPARSE on a strongly banded column pattern at nrhs >= 17
        // (docs/perf/spmm.md#the-cfloat-transb-exclusion, both passes, m=2048,
        // nnz/row=16, batch=512):
        //     nrhs        8     9    12    16    17    20    25    32    50
        //     banded p1 0.630 0.713 0.689 1.087 1.315 1.218 1.731 1.159 1.695
        //     banded p2 0.663 0.737 0.761 1.040 1.274 1.162 1.714 1.157 1.703
        // and it is what refutes the UNCONDITIONAL gather clause: transA ==
        // NoTrans with no exclusion FAILS 1 of 186 at worst 1.934 -- cfloat
        // tA=0 m=2048 nnz/row=16 nrhs=25 b=128 tB=1 beta=0 banded, p1=1.934
        // p2=1.872 (verdict.txt).
        //
        // REJECTED NARROWER CANDIDATE, and this is the one worth arguing about:
        // `... && nrhs >= 16` (or >= 13) PASSES on verdict.txt's grid, at worst
        // 0.968, and would move 183 cells instead of 176. It is rejected because
        // its boundary rides on an axis the SHAPE CANNOT SEE. The same family on
        // the SCATTERED pattern is 0.79-1.02 at every nrhs, so the nrhs
        // threshold is a property of the BANDED column pattern, and SpmmShape
        // has no column-pattern field and cannot acquire one (it would have to
        // read col_indices on the device). Fitting a threshold to one pattern is
        // the "boundary at the edge of the sampled range" objection WP7's own
        // audit raised, in a new dress. Refusing the family whole costs at most
        // 2% on the scattered pattern, and 7 of the 186 gather cells.
        //
        // BRACKETED ON THREE AXES: nrhs 12 (0.69-0.76) vs 17 (1.27-1.32);
        // type -- float, double and complex<double> on the identical cells run
        // 0.22-0.69 and never lose, so the type conditional is exactly as narrow
        // as the data; and BATCH (docs/perf/spmm.md#the-batch-axis-has-no-floor) -- the same
        // cell runs 0.581 at batch 4, 1.447 at batch 8, peaks at 2.18 at batch
        // 32, 1.94 at 128 and 1.71-1.73 at 512, so the loss is NOT a saturation
        // artefact and no batch-conditional narrows it usefully.
        //
        // THE COST OF THE SIMPLE EXCLUSION, RECORDED RATHER THAN HIDDEN: at
        // batch 1-4 the excluded cells run 0.520 / 0.524 / 0.581, a 1.7-1.9x
        // native win this clause now declines.
        //
        // MECHANISM, HYPOTHESIS ONLY, NOT CONFIRMED BY PROFILE:
        // kNCmax<Cx<float>> is 8 (spmm_native.cc:88), so nrhs=25 needs
        // ceil(25/8)=4 passes over A with 7 idle accumulator lanes while
        // nrhs=32 needs 4 with none -- and nrhs=32 measures 1.16 against
        // nrhs=25's 1.73.
        if constexpr (std::is_same_v<T, std::complex<float>>) {
            if (s.transB != Transpose::NoTrans) return false;
        }

        // WHAT IS DELIBERATELY *NOT* HERE:
        //   * no is_gpu term. The gather body has zero local memory and no group
        //     collective, and the vendor-free burn-down needs the
        //     Backend::NETLIB rows; supports() has no GPU gate either, for the
        //     same reason.
        //   * no m / k / out_rows / red_rows term. The gather wins from m=1024
        //     to m=4096 and from 3 to 16 nnz/row on both column patterns and
        //     both betas; no extent boundary is bracketed by a measured
        //     non-winner.
        //   * no nnz or density term. SpmmShape has no honest nnz to read.
        return true;
    }

    // ---- NO native_tier_preferred() HOOK, DELIBERATELY --------------------
    // That hook exists to arbitrate between two native routes that can BOTH
    // serve one shape (route_geqrf.hh's CTA vs Blocked). There is ONE native
    // route here, so it would be a predicate with no decision behind it. It is
    // optional, detected by a `requires` expression, and defaults to true
    // (route_resolve.hh:76-83), which makes the two vendor-free passes identical
    // -- so declining to declare it is a no-op, not a gap. This is
    // route_gemv.hh's reasoning verbatim.

    static constexpr const Route* order_begin() { return kSpmmOrder; }
    static constexpr const Route* order_end() {
        return kSpmmOrder + (sizeof(kSpmmOrder) / sizeof(kSpmmOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// spmm into the coverage table: resolve_route records every op that goes through
// it (route_resolve.hh:196-214), slicing SpmmShape to OpShape.
//
// KNOWN AND ACCEPTED COVERAGE BLINDNESS, recorded here with its trigger
// condition. variant_key packs only uplo/side/diag/transA/transB
// (coverage.cc:52-58) and shape_class buckets max(m,n,k) and batch by power of
// two (route.hh:254-262). So a CSR spmm and a Dense spmm at the same extents
// would collapse into ONE first-writer-wins row. That is UNOBSERVABLE today
// because only CSR is instantiated. DO NOT add a format bit to variant_key to
// fix it: renumbering the key invalidates every stored .routes baseline.
// Revisit on the day a native Dense spmm arm exists, and not before.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_spmm_route(Route forced, const SpmmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::spmm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
