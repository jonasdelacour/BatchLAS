#pragma once

// GETRF's routing table (WP6 scaffolding -- the WP5 / WP4-Phase-0 equivalent).
//
// MODELLED ON route_potrf.hh FOR THE FIELD MAPPING AND ON route_geqrf.hh FOR THE
// HOUSE STYLE, and the three rules both files obey are restated here because
// getting any of them wrong has already shipped a defect in this tree:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the shape builder (src/backends/getrf_route.hh),
//                     not here. This header is PURE (route_resolve.hh:19-21).
//
// WHY THE SPLIT MATTERS FOR getrf SPECIFICALLY. route_resolve.hh:113-127 re-walks
// the candidate order testing ONLY `is_native(*r) && Table::supports(*r, s)`, so
// any speed threshold that lands in supports() removes getrf's vendor-free route
// entirely. And route_resolve.hh:8-10 / :165 say a forced route bypasses
// preferred() but NEVER supports(), so a test that pins BATCHLAS_GETRF_ROUTE and
// hits a wrongly-placed gate silently runs cuBLAS and passes GREEN over a kernel
// nothing executed.
//
// WHAT THIS FILE REPLACES. Before WP6, getrf/getrs/getri had NO route resolution
// whatsoever -- src/dispatch/entry_points/factorization.cc:464-541 was six bodies
// of `if constexpr (!vendor_available) throw; else vendor;`. That is exactly
// where geqrf/orgqr stood before WP5, so WP5 is the template being copied.
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE, BECAUSE IT IS WHERE
// getrf DEPARTS FROM geqrf AND AGREES WITH potrf.
//
//     s.m = s.n = s.k = THE ORDER
//
// getrf's operand is SQUARE. options.hh:615 calls detail::require_square before
// every arena-spelled call; cublas.cc:1501 and netlib_lapack.cc:1291 both read
// only A.rows() and pass it as both extents. So route_potrf.hh:213's
// `if (s.m != s.n) return false;` DOES belong in this file -- unlike in geqrf,
// where copying it was the recorded wrong edit (route_geqrf.hh:55-64), because
// rectangular A is the entire point of THAT op.
//
// (LAPACK's xGETRF is defined for rectangular A and cuBLAS's getrfBatched is
// not -- it takes one `n`. BatchLAS's public API followed cuBLAS. A native
// rectangular getrf would be a WIDENING of the public contract and needs the
// validator, the option-struct checks and the vendor arms to move with it; it is
// not something a routing table may quietly assume.)
//
// STATUS: BOTH NATIVE ARMS ARE LIVE for all four scalar types.
// src/extensions/getrf_cta.cc's capacity function measures 155/109/109/77 for
// float/double/cfloat/cdouble on an RTX 4090, and getrf_blocked.cc reports
// available for every type, so supports() admits the CTA arm up to the capacity
// and the Blocked arm at every order. A VENDOR-FREE BUILD RESOLVES NATIVE FOR
// EVERY SQUARE SHAPE (96 of 96 measured route cells).
//
// THE VENDOR-PRESENT BUILD IS UNCHANGED, and the reason is preferred() below,
// not a missing kernel: it is still false everywhere, so Origin::Auto returns
// {Vendor, Auto} wherever a vendor exists (route_resolve.hh:110-112, :129), also
// 96 of 96. `native_tier_preferred` chooses BETWEEN the two native arms and is
// consulted only in the vendor-free walk, which is why it can carry a real window
// while preferred() carries none.
//
// THE PIVOT CONTRACT IS NOT EXPRESSIBLE HERE, AND IT IS WP6's MOST LIKELY SILENT
// WRONG ANSWER. Measured: `Span<int64_t> pivots` is REINTERPRETED, not converted.
// cublas.cc:1508 does `pivots.as_span<int>()` (sycl-span.hh:45-47 is a
// reinterpret_cast with the size rescaled) and hands cuBLAS PACKED INT32 in the
// first half of the caller's int64 buffer; rocsolver.cc:227 does the same;
// netlib_lapack.cc:1312-1320 widens an int scratch into GENUINE int64. Both are
// 1-based and both are an INTERCHANGE LIST, not a permutation vector (verified
// element-by-element against LAPACKE_?getrf: 0/18 mismatches read as packed
// int32, 18/18 read as int64).
//
// So the physical pivot format is BACKEND-DEPENDENT, and a native getrf must
// pick one and agree with WHATEVER SERVES getrs/getri ON THAT CALL. That
// combination is entirely reachable -- three independent env variables and three
// independent preferred() windows -- so a native getrf feeding a vendor getri can
// read garbage pivots with no gate anywhere able to see it. There is no
// OpShape field that can express "the op downstream of me resolved differently",
// which is precisely why this note is here rather than in a predicate: the
// contract must be fixed in the KERNEL step, in writing, with a cross-op test
// (native getrf -> vendor getri and the reverse). See getrf_native.hh.
//
// WARNING FOR ANYONE USING scripts/route_diff.sh AS A GATE ON THIS OP, and it is
// as bad as geqrf's (route_geqrf.hh:82-90). getrf sets NONE of
// uplo/side/diag/transA/transB, so coverage.cc:52-58's variant_key is CONSTANT
// for it and rows collapse to shape_class alone -- a power-of-two bucket on
// max(m,n,k) and on batch (route.hh:249-259), first-writer-wins
// (coverage.cc:284-292). inverse_tests' n=40/batch=2 and ANY n in 33..64 at batch
// 2..3 are ONE row. Write the facade-routed test FIRST, capture SECOND, and pick
// extents in a max_dim bucket no other LU call in the suite touches.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// GETRF's routing reads three things OpShape has no field for, so the op extends
// it, exactly as PotrfShape (route_potrf.hh:105-186), GeqrfShape
// (route_geqrf.hh:104-180) and TrsmShape (route_trsm.hh:92-114) do.
// ---------------------------------------------------------------------------
struct GetrfShape : OpShape {
    // The largest ORDER the CTA tier can hold, or 0 when the CTA kernel is not
    // in this build. A SINGLE number and not a (height, area) pair as geqrf
    // needs, because getrf's operand is square: order is the only extent.
    //
    // ASKED OF THE DEVICE, never of a constant, for route_potrf.hh:114-127's
    // reason: the ceiling is a pure function of the local-memory budget, so a
    // build-time number makes supports() claim an unlaunchable route on a device
    // with less of it. In particular it must NOT come from
    // build/include/batchlas/device_limits.hh, whose 49152 is HARDCODED by
    // cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern and is
    // 2.06x wrong on this box (sycl::info::device::local_mem_size == 101,376 B
    // here -- WP4's finding W1, measured).
    //
    // AND IT MUST COUNT THE PIVOT-SEARCH SCRATCH, which is a WP6-only hazard with
    // no potrf or geqrf analogue. Measured (docs/perf/lu.md#the-vendor-baseline-and-saturation,
    // pivotcost.cpp): an explicit SLM tree argmax needs
    // wg*(sizeof(real)+sizeof(int)) ON TOP OF the tile -- 2040 B at wg 256 for
    // float, 3060 B for cdouble -- and that scratch alone
    //   (a) cost a launch OUTRIGHT at cdouble n=78, taking the request from
    //       98,608 B to 101,668 B past this device's 101,376 B hard cap
    //       ("Excessive allocation of local memory on the device"), and
    //   (b) MOVES THE BLOCKS-PER-SM CLIFF DOWN BY TWO ORDERS OF n: with
    //       slm+1024 crossing 102400/2 = 50,688 B the occupancy halves, which
    //       lands at n=112 without the scratch and n=110 with it, a 1.73x step.
    // (b) is a speed fact and belongs in preferred(); (a) is a LAUNCH FAILURE and
    // belongs in this number. The formula lives next to the kernel, in
    // src/extensions/getrf_cta.cc, so the ceiling this table advertises and the
    // allocation the launcher makes cannot disagree (route_trsm.hh:62-72).
    //
    // ZERO MEANS THE CTA KERNEL IS ABSENT FROM THIS BUILD, which is the state
    // today, and it correctly makes BOTH native routes unsupported rather than
    // selectable-but-unimplemented. Same convention as TrsmShape::cta_max_n,
    // pinned by RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable, and as
    // PotrfShape::cta_max_n and GeqrfShape::cta_max_elems.
    int cta_max_n = 0;

    // Whether the BLOCKED driver exists in this build. Separate from the CTA
    // capacity because the two are independent capabilities: blocked is what
    // serves orders the CTA tile cannot hold, and until it is written those
    // orders have no native route at all.
    //
    // Not belt-and-braces (route_trsm.hh:99-110): reporting Blocked as supported
    // while it does not exist makes resolve_route hand a VENDOR-FREE caller a
    // route the facade cannot service, i.e. a std::logic_error instead of a
    // factorisation. The table must describe the BUILD, not the design.
    bool blocked_available = false;

    // Does this device offer sub-group size 32 -- ENUMERATED from
    // sycl::info::device::sub_group_sizes (Device::supports_sub_group_size,
    // sycl-device-queue.hh:180-190), never inferred from OpShape::max_sub_group.
    //
    // WHY NOT max_sub_group: Device::get_property(MAX_SUB_GROUP_SIZE) returns
    // sub_group_sizes()[0] (src/util/queue-impl.cc:325) -- the FIRST entry of the
    // supported list, not the maximum -- so syev's `s.max_sub_group >= 32`
    // (syev.hh:837) is wrong in both directions: a device reporting {8,16,32}
    // reads 8 and is refused although it does support 32, and a device reporting
    // {64} reads 64 and is ACCEPTED although it has no 32 at all. The second is a
    // launch abort for a kernel carrying [[sycl::reqd_sub_group_size(32)]], i.e.
    // exactly what supports() exists to exclude.
    bool has_sg32 = false;

    // The order. NAMED, and the only spelling this file uses, so a later
    // predicate cannot reach for m when it meant k -- the PotrfShape::order()
    // discipline, and GeqrfShape::reflectors()' (route_geqrf.hh:177-182).
    int64_t order() const { return k; }
};

// CTA first, then the blocked driver whose diagonal-panel leaf will BE the CTA
// device function, then the vendor. The order is a CAPABILITY LADDER, not a
// preference: CTA serves only orders the resident tile can hold and blocked
// serves the rest. With preferred() all-false today the order matters only in the
// vendor-free walk at route_resolve.hh:113-127, where the tighter route is the
// right one to try first.
//
// A file-scope array of natural length with sizeof-computed bounds, NOT a
// std::array<Provider,6>: route_gemm.hh:43-46 records that this "removes the
// truncation hazard of the four hand-counted std::array<Provider,6> sites".
inline constexpr Route kGetrfOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getrf, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a WRONG ANSWER or could not
    // run at all", never "the kernel would be slow". Nothing here is
    // type-dependent: the whole per-type difference is SLM capacity, and that
    // arrives as cta_max_n.
    static bool supports(Route r, const GetrfShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. SQUARE ONLY, and here that IS route_potrf.hh:213's line -- see the
        //    FIELD MAPPING note at the top of this file. It is not a restriction
        //    the native kernel invents: the whole public API is square
        //    (options.hh:615 require_square) and cuBLAS's getrfBatched takes one
        //    `n`. Widening getrf to rectangular A is an API change, not a
        //    routing decision.
        if (s.m != s.n) return false;

        // 2. GPU ONLY. Not a speed judgement: the native path will be a SYCL
        //    nd_range kernel with a work-group-collective pivot search and a
        //    local_accessor tile, and there is no host implementation of it to
        //    fall back on, so a CPU queue has to reach netlib. Same wording and
        //    same reason as route_trsm.hh:138-142, route_potrf.hh:222 and
        //    route_geqrf.hh:236-243.
        //
        //    It is also why the getrf kernel TUs sit in EXTENSIONS_CTA_SOURCES,
        //    the only object library configured NO_CPU_TARGETS
        //    (src/CMakeLists.txt:66-72).
        if (!s.is_gpu) return false;

        // 3. SUB-GROUP SIZE 32. The kernels will carry
        //    [[sycl::reqd_sub_group_size(32)]] (the precedent set at
        //    syev_cta_fused.cc:185, gesvdj_cta.cc:297, sytrd_sb2st_cta.cc:403).
        //    On a device whose sub_group_sizes do not contain 32 the launch is
        //    REJECTED. That is "cannot run", not "runs slowly". It gates BOTH
        //    native arms, because the blocked driver's diagonal-panel leaf IS
        //    that same device function.
        if (!s.has_sg32) return false;

        // 4. HETEROGENEOUS BATCH. One launch will cover the whole batch with a
        //    single (order, ld, stride) tuple and read at data_ptr() + b*stride
        //    with the CAPACITY extents, so a view with per-item active dims
        //    (matrix.hh:1034; publicly constructible via Matrix::set_active_dims
        //    and MatrixView::with_active_dims) would silently factorise the wrong
        //    extents in place for every item after the first. getrf has no batch
        //    walker -- unlike gemm, where WP2 C2 made this merely un-preferred
        //    because the facade walks the batch (route_gemm.hh:70-80) -- so for
        //    getrf it is a correctness gate.
        //
        //    THE JUSTIFICATION IS geqrf's, NOT potrf's, and the difference is
        //    worth keeping straight. route_potrf.hh:255-257 argues the gate is
        //    needed because netlib's potrf already honours per-item extents
        //    (netlib_lapack.cc:1029 reads A_view[i].rows()), so a native route
        //    would disagree with a path in this tree that gets it right. For getrf
        //    netlib does NOT: netlib_lapack.cc:1291 hoists n from A.rows() OUTSIDE
        //    the loop and only indexes A_view[i].data_ptr(). Nothing in this tree
        //    serves a heterogeneous LU. The gate is here because the native kernel
        //    cannot, full stop.
        if (s.heterogeneous_batch) return false;

        // 5. DEGENERATE EXTENTS. The column loop, the pivot search and the tile
        //    index map are undefined for an empty matrix or an empty batch.
        //    Disagreement BETWEEN views is not tested here -- OpShape holds one
        //    batch, so the shape builder reports that by returning no shape at
        //    all (the gemm_op_shape pattern, src/backends/gemm_variant.hh:189-197).
        if (s.order() < 1 || s.batch < 1) return false;

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
        //    the field -- src/backends/getrf_route.hh sets `s.backend = B` -- so the
        //    gate is one predicate, and it is enumerated by the backend whose
        //    format disagrees rather than by an allow-list, so a new GPU backend
        //    that packs int32 like the other two needs no edit here.
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                // The whole matrix plus the pivot-search scratch is resident in a
                // local_accessor whose extent is chosen at launch, so this is a
                // HARD CAPACITY, not a tuning knob: above it there is no
                // launchable configuration. Zero means the kernel is not in this
                // build at all -- which is NOT the state any more; on this box it
                // is 155/109/109/77 for float/double/cfloat/cdouble.
                if (s.cta_max_n < 1) return false;
                return s.order() <= static_cast<int64_t>(s.cta_max_n);

            case Algorithm::Blocked:
                // The blocked driver's diagonal-panel factorisation IS the CTA
                // kernel, so it inherits the PRESENCE gate but NOT the capacity --
                // it splits the matrix into panels the leaf can hold itself. It
                // also has to exist, which is a separate question
                // (route_trsm.hh:172-177 is the identical pair).
                //
                // NO LOWER BOUND ON THE ORDER. "order <= the CTA capacity so
                // blocked should be false" is a FIT judgement between two native
                // routes, not a correctness claim, and route_potrf.hh:284-296
                // records what putting it here costs: per route_resolve.hh:165 a
                // forced `blocked` at a small order then falls through to
                // automatic() at :175, which at merge returns {Vendor, Auto} --
                // so the test that pinned the blocked driver measures cuBLAS and
                // passes green. Pinning all three routes at overlapping orders
                // only works if this arm carries no lower bound.
                //
                // THIS IS NOT HYPOTHETICAL FOR WP6. inverse_tests -- the one suite
                // WP6 can close outright -- is a single float case at n=40,
                // batch=2 (tests/inverse_tests.cc:10-39). Any batch floor or order
                // floor in this function keeps that suite red however good the
                // kernel is.
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                // Including Algorithm::Auto. getrf has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:146-163).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, DELIBERATELY, AND THAT IS THE MERGE STATE OF THIS FILE.
    //
    // Nothing native about getrf has been measured because nothing native about
    // getrf exists yet. route_trsm.hh:53-55 records the precedent in as many
    // words: "preferred() was all-false until WP3 step 9 measured the grid".
    // All-false means Origin::Auto takes the vendor wherever one exists
    // (route_resolve.hh:110-112 finds no supported AND preferred route, :129
    // returns {Vendor, Auto}), so merging this table moves ZERO traffic.
    //
    // It is ALSO what keeps the vendor-free build honest. Un-preferred is not
    // unroutable: route_resolve.hh:113-127 still hands a vendor-free caller any
    // SUPPORTED native route. So the day getrf_cta_max_n_for_slm<T>() comes off
    // zero, a vendor-free build starts using the native kernel and a
    // vendor-present build does not -- which is the correct order of events.
    //
    // WHAT GOES HERE WHEN THE GRID IS MEASURED, and nowhere else: the per-cell
    // window, with each clause citing the CSV rows it comes from as
    // route_trsm.hh:188-325 does. NOT tuning_params.hh -- that header is for
    // nb/L/G, and putting a routing threshold there mixes a route decision into a
    // workspace-sizing input.
    //
    // FALSE EVERYWHERE IS NOW A DELIBERATE HOLD, NOT AN ABSENCE OF KERNEL. Both
    // arms exist and both are measured (docs/perf/lu.md#the-vendor-baseline-and-saturation, 982
    // timed rows); the window is withheld because the measurement says the
    // crossover moves with BATCH as much as with order -- the same order sweep is
    // geomean 1.44x/3.57x at batch 32 and 0.67x/0.91x at batch 1024 -- so a window
    // fitted to an order sweep alone would be wrong at both ends.
    //
    // FOUR MEASURED FACTS THAT WILL SHAPE THIS FUNCTION, recorded so they are not
    // re-derived (docs/perf/lu.md#the-vendor-baseline-and-saturation and bench/README.md):
    //
    //   * THE VENDOR HERE IS GENUINELY BATCHED and is strong at small n.
    //     cublas{S,D,C,Z}getrfBatched (cublas.cc:1509) is a real batched routine,
    //     not the per-batch-item loop that made WP5's orgqr ratios enormous.
    //     Through the public API at saturating batch it reaches 1129 GFLOP/s
    //     (float n=32) and 992 (float n=128). A modest win or parity is the
    //     realistic good outcome for WP6; a loss is a real possibility.
    //   * ITS GFLOP/s IS NOT MONOTONE IN n -- float 1129 (n=32) -> 545 (n=64) ->
    //     992 (n=128) -> 353 (n=2048) -- the shape of a routine with a small-n
    //     special case and no large-n blocking. The soft targets are the large-n
    //     cells, and cdouble n=2048 at 70.6 GFLOP/s is the softest in the table.
    //   * EVERY LARGE-n RATIO WILL FLATTER THE NATIVE SIDE, because the vendor is
    //     NOT SATURATED there: getrf float n=1024 is still falling 10% from batch
    //     128 to 256, and cdouble n=2048 does 64x the work for 1.03x the time
    //     from batch 1 to 64. Any window written off an unsaturated cell is
    //     written off an artefact. State the saturation level beside every ratio.
    //     MEASURED, and it is large: read at each arm's OWN best batch instead of
    //     the grid's, getrf's geomean goes 0.885x -> 0.805x and the n=2048 row
    //     collapses from 7.10x to 2.95x. cublasZgetrfBatched at n=2048 takes
    //     2587 ms at batch 4 and 2595 ms at batch 32 -- 8x the work for 0.3% more
    //     time, i.e. a latency chain that is not using the GPU at all.
    //   * THE FP64 CASE IS CLOSED AND THE FP32 CASE IS NOT. cuBLAS runs cdouble
    //     getrf at 90-91% of this device's FP64 peak at n=512-1024, so there is no
    //     2x to find there; for FP32 BOTH arms sit at 1-10% of peak, and the
    //     decomposition -- not the kernels -- is why: at getrf double n=128 the
    //     native arm spends 48.5% panel + 33.6% laswp + 9.2% gemm + 8.8% trsm
    //     across four kernels per block step where cuBLAS does the whole
    //     factorisation in ONE fused kernel at 99.9%.
    //
    // ---- WP8-I1: THE LASWP HALF OF THAT LAST BULLET IS NOW CLOSED, AND THE
    //      "FOUR KERNELS VS ONE FUSED KERNEL" READING OF IT DID NOT SURVIVE ----
    //
    // THE FUSION READING IS REFUTED BY ARITHMETIC THIS TREE ALREADY HELD. The
    // blocked arm launches 5P-4 kernels per call (16 at n=128, confirmed
    // launch-for-launch by nsys). At 5 us a launch that is 80 us at n=128, which
    // is 0.12% of the 67.1 ms native call at batch 8192 and 8.7% of the
    // native-minus-vendor gap at the smallest saturating batch -- falling
    // monotonically with batch and never explaining the gap. And the fused arm
    // ALREADY EXISTS: float n <= 155 resolves native:cta, ONE kernel with no
    // laswp and no decomposition, and it measures 0.77-1.00x. The decomposition
    // costs DATA MOVEMENT, not launches.
    //
    // WHAT SHIPPED INSTEAD. (S-left) -- the interchange applied to the finished
    // columns -- is deferred to one SLM-staged permutation gather after the block
    // loop (lu_laswp.hh's deferral identity and lu_laswp_deferred_left_launch).
    // Unconditional, no shape gate, no crossover, no extra workspace, and the
    // three spellings are asserted BIT-IDENTICAL by
    // LuTest.LeftInterchangeSpellingsAgreeBitForBit.
    //
    // MEASURED AGAINST THE ARM IT REPLACES, vendor-free, interleaved inside one
    // process, 11 reps, median, two passes, batch >= 128, 58 native:blocked
    // cells: geomean 1.207x, min 1.018x, ZERO cells below 1.00, cross-pass median
    // spread 1.0011 / worst 1.033. float 1.350x, cfloat 1.305x, double 1.138x,
    // cdouble 1.074x. The native:cta rows measure 0.9995x -- the change cannot
    // reach them.
    //
    // AGAINST cuBLAS ON THE SAME 62-CELL GRID (batch 128..1024, order 128..2048,
    // all four types) the getrf geomean moves 0.839x -> 1.002x, 20 wins -> 28:
    //     float   1.273x -> 1.594x      cfloat  0.974x -> 1.271x
    //     double  0.629x -> 0.716x      cdouble 0.610x -> 0.659x
    // The float/cfloat families are now the story; double and cdouble are NOT
    // closed by this and will not be closed by anything short of a
    // register-resident fused panel, which is a work package and not a lever.
    //
    // ---- THE WINDOW THIS FUNCTION SHOULD CARRY, RECOMMENDED NOT APPLIED ------
    // The routing pass owns preferred(); WP8-I1 owned the kernel. Transcribed
    // from docs/perf/lu.md#getrf-window-evidence against base_v_p{1,2}.csv,
    // ratio = vendor_med / native_med, every cell reproduced in two passes:
    //
    //   float, order >= 256, batch >= 128   -- 12 cells, min 1.254, no loss
    //     n=256  b128 1.254  b256 1.279  b512 1.567  b1024 1.675
    //     n=512  b128 2.350  b256 1.988  b512 1.737  b1024 2.183
    //     n=1024 b128 2.773  b256 2.186  b512 2.237
    //     n=2048 b128 3.091
    //   cfloat, order >= 512, batch >= 128  --  8 cells, min 1.528, no loss
    //     n=512  b128 1.811  b256 1.528  b512 1.682  b1024 1.609
    //     n=1024 b128 2.124  b256 1.829  b512 2.088
    //     n=2048 b128 2.754
    //
    // AND THE CELLS THAT REFUSE THE OBVIOUS WIDER CLAUSES, so they are not
    // rediscovered: `float order >= 128` steals the native:cta rows at
    // 0.825/0.773/0.872 (batch 256/512/1024); `cfloat order >= 256` admits
    // b=128 at 0.884; `double order >= 512` admits b=256/512/1024 at
    // 0.933/0.813/0.748 and its best cell anywhere is 1.067; cdouble's best cell
    // anywhere is 1.012. Neither double family earns a window at any order.
    //
    // THE BATCH FLOOR IS A POLICY CHOICE, NOT A MEASURED ONE: this grid starts at
    // batch 128 and nothing below it was re-measured after the change.
    //
    // =====================================================================
    // WP8 ROUTING PASS: THE WINDOW LANDS, AND THE GRID ABOVE HAD TO BE
    // RE-MEASURED BEFORE IT COULD.
    //
    // WHY RE-MEASURED. WP8-I1's sweep ran with GPU=0, the default of its own
    // runner. Device 0 on this box drives the display, and this pass then found
    // a second and larger effect on top of that: a getrf sweep on device 1 run
    // CONCURRENTLY with a gemv sweep on device 0 reads getrf float n=256
    // batch=128 at 3.31-5.51 ms against 1.006 ms on an idle box, and the RATIO
    // moves 1.254 -> 1.764. Two RTX 4090s in one chassis are not two independent
    // machines: same NUMA node, same CPU affinity mask, one UVM driver, and
    // lubench6 runs on managed memory. The per-row foreign() guard reports 0
    // (--query-compute-apps is PER DEVICE) and rel_sd stays at 0.0004-0.017, so
    // neither instrument sees it. getri and gemv were unaffected -- their timed
    // regions are long and device-resident -- and getrf, whose timed region is
    // 1-8 ms of launch-bound work, was affected by up to 5x.
    //
    // So the numbers below come from docs/perf/lu.md#measured-boundaries, taken on
    // device 1 with NOTHING ELSE ON THE BOX. The clause is I1's recommendation,
    // tested rather than inherited.
    //
    // AND BE PRECISE ABOUT WHAT "REPRODUCED" MEANS HERE, because the gate says
    // two passes and this is not two passes. lu_c2.csv holds 45 rows and ZERO
    // getrf rows -- it is a getri pass -- so an earlier draft of this comment
    // cited a file that could not support it. The second source for getrf is
    // WP8-I1's own record, taken on the OTHER DEVICE, in a different session,
    // from a different binary. That is arguably stronger evidence than a repeat
    // of the same run, since it varies the device as well as the session; it is
    // not the same claim, and the header should not say it is. 26 cells are
    // common to the two sources: median spread 1.0053, worst 1.0311, none above
    // 1.10.
    //
    // AND THE RE-MEASURE VINDICATES IT. Run alone, device 1 reproduces I1's
    // device-0 figures to within 1% on every cell of the clause and both its
    // boundaries -- so the display GPU was NOT the problem and the concurrent
    // sweep was:
    //   float  n=128 b128  I1 1.0037  clean 1.0037     n=128 b512  0.7727 / 0.7757
    //   float  n=256 b128     1.2541        1.2626     n=256 b512  1.5665 / 1.5658
    //   float  n=256 b1024    1.6239        1.5750     n=512 b128  2.3504 / 2.3455
    //   float  n=512 b512     1.7369        1.7400
    //   cfloat n=128 b128     0.4891        0.4920     n=256 b128  0.8844 / 0.8851
    //   cfloat n=256 b512     1.4070        1.4326     n=256 b1024 1.1921 / 1.1880
    //   cfloat n=512 b128     1.8113        1.7748     n=512 b512  1.6823 / 1.6718
    //   cfloat n=512 b1024    1.6087        1.6065
    // The two BOUNDARIES are bracketed from below by measured non-winners in
    // both records: float n=128 (the native:cta rows) at 0.776-1.004, and cfloat
    // n=256 at 0.885 (batch 128) with 1.188 at batch 1024 still under the 1.15
    // bar. That is why the two thresholds differ by one grid step.
    //
    // NO BATCH TERM. The grid runs batch 128..1024 at every admitted order and
    // the window is flat across it; below 128 nothing was measured after the
    // kernel change, and that is stated as a bound rather than fitted away.
    // =====================================================================
    static bool preferred(Route r, const GetrfShape& s) {
        if (!is_native(r)) return false;

        // BLOCKED ONLY. The CTA arm is a different kernel with its own
        // measurement, and it LOSES: float n=128 (which is where CTA serves,
        // cta_max_n being 155 for float on this device) reads 0.825 / 0.773 /
        // 0.872 of cuBLAS at batch 256 / 512 / 1024. Writing the clause on the
        // order alone and letting the tier ladder sort it out would admit those.
        if (r.algo != Algorithm::Blocked) return false;

        if constexpr (std::is_same_v<T, float>)               return s.order() >= 256;
        if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 512;
        return false;   // double and cdouble earn nothing at any order
    }

    // ---- native_tier_preferred IS DECLARED, AND IT IS MEASURED ------------
    // The scaffolding left this hook deliberately ABSENT, on the ground that
    // "declaring an unmeasured window publishes a claim nothing measured".
    // The tier sweep has now run (docs/perf/lu.md#native_tier_preferred), so the
    // debt is paid rather than carried; the predicate and its numbers are at the
    // bottom of this table. The standard route_geqrf.hh:385-425 sets for
    // "measured" is met: both arms PINNED, EVERY PIN VERIFIED TO HAVE TAKEN by
    // reading the resolved route off each row (four rows had a `cta` pin fall
    // through above the capacity ceiling and are excluded), and the one type
    // whose answer differs from the order array re-run across four batches.
    //
    // The cost of NOT declaring it was known in advance and turned out to be
    // real here too: for geqrf it was 1.37x at double n=96; for getrf it is
    // 1.18-1.29x at double n=76..96, in the one build this campaign exists for.

    // ---------------------------------------------------------------------
    // THE NATIVE-VS-NATIVE TIE-BREAK. Consulted ONLY in the vendor-free walk
    // (route_resolve.hh:119-127), so declaring it moves NOTHING in a
    // vendor-present build -- which is exactly why it is the right instrument
    // and preferred() is not. preferred() runs above that walk regardless of
    // vendor_available, so a window written to fix the tier choice would also
    // drag vendor-present traffic onto that tier at shapes where cuBLAS beats
    // both natives (route_resolve.hh:40-63 states the distinction).
    //
    // WITHOUT IT the vendor-free choice is decided entirely by kGetrfOrder,
    // which lists CTA first and therefore cannot follow a crossover.
    //
    // THE MEASUREMENT. Both arms PINNED, in the vendor-free build, with the
    // resolved route read off every row -- which matters, because the CTA pin is
    // refused above the per-type ceiling and then falls through to automatic(),
    // and four of the sweep's rows did exactly that and are excluded.
    // docs/perf/lu.md#native_tier_preferred and run_tier.sh. Ratio is
    // blocked_ms / cta_ms, so > 1 means CTA is ahead:
    //
    //   float    n=64(8192) 1.74  n=76(8192) 1.48  n=96(8192) 1.49
    //            n=100(4096) 1.68  n=128(4096) 1.13
    //   cfloat   n=64(8192) 1.39  n=76(8192) 1.59  n=96(8192) 1.30
    //            n=100(4096) 1.33
    //   cdouble  n=64(8192) 1.37  n=76(8192) 1.09
    //   double   n=64(8192) 0.98  n=76(8192) 0.85  n=96(8192) 0.77
    //            n=100(4096) 1.00
    //
    // and double re-run across four batches at its worst order, because a
    // one-cell window is exactly the over-fit this campaign keeps warning about:
    //
    //   double n=76: b=2048 0.78   b=4096 0.84   b=8192 0.85   b=16384 0.85
    //   float  n=76: b=2048 1.19   b=4096 1.44   b=8192 1.48   b=16384 1.48
    //   cfloat n=76: b=2048 1.31   b=4096 1.55   b=8192 1.59   b=16384 1.60
    //   cdouble n=76: b=2048 1.04  b=4096 1.08   b=8192 1.09   b=16384 1.10
    //
    // One-directional, flat in batch, and per type. All relative sds < 0.2%.
    //
    // SO: DOUBLE PREFERS THE BLOCKED DRIVER BELOW ITS OWN CTA CEILING, and the
    // other three prefer CTA. That is the whole window.
    //
    // WHY n <= 32 GOES TO CTA FOR DOUBLE TOO, and it is not a hedge: at n <= 32
    // the blocked driver's nb is min(32, n) = n, so it runs exactly ONE panel
    // whose leaf IS the CTA device function. The two arms are the same code
    // there, measured identical (1.8126 vs 1.8113 ms at n=32, batch 8192), and
    // CTA is the better spelling of the same thing -- one launch instead of
    // three, no pointer arrays, no workspace draw.
    //
    // WHAT IS EXTRAPOLATED RATHER THAN MEASURED, named so it is the first place
    // to look if this is re-measured: the band between the last measured order
    // and each type's capacity ceiling -- float 129..155, cfloat 101..109,
    // cdouble 77 -- is left on CTA. cdouble's advantage is the one that is
    // clearly collapsing (1.37 at n=64 -> 1.09 at n=76 against a ceiling of 77),
    // so cdouble is where a re-measurement would find a crossover first.
    //
    // NOT A CORRECTNESS GATE. Everything here is "the other native route is
    // faster". Both arms stay fully supports()-able at every shape, which is
    // what keeps a pinned `cta` at double n=96 actually running CTA instead of
    // falling through to automatic() and measuring the thing it was pinned away
    // from (route_resolve.hh:165 -> :175).
    static bool native_tier_preferred(Route r, const GetrfShape& s) {
        if (!is_native(r)) return true;

        // The largest order at which CTA is the better native route, per type.
        const int64_t cta_max_order = [] () -> int64_t {
            if constexpr (std::is_same_v<T, double>) {
                // 0.98 at n=64, 0.85 at n=76, 0.77 at n=96 -- blocked ahead
                // everywhere the two arms are actually different code.
                return 32;
            } else {
                // float, cfloat and cdouble are ahead at EVERY cell their
                // capacity lets the sweep reach, so there is no measured
                // crossover to encode and supports()' fit gate is the only
                // ceiling.
                return 1 << 30;
            }
        }();

        switch (r.algo) {
            case Algorithm::CTA:
                return s.order() <= cta_max_order;
            case Algorithm::Blocked:
                return s.order() > cta_max_order;
            default:
                return true;
        }
    }

    static constexpr const Route* order_begin() { return kGetrfOrder; }
    static constexpr const Route* order_end() {
        return kGetrfOrder + (sizeof(kGetrfOrder) / sizeof(kGetrfOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default comes from
// legacy_unset_default(Op::getrf), which is {Auto, Auto} for every op since
// WP2 E6 (route_env.hh:145-148).
//
// `vendor_available` is the vendor-free switch, and the facade knows it as a
// compile-time fact: dispatch::factorization_vendor_available<B>
// (vendor_available.hh:41-45) -- NOT solver_vendor_available, which is cuSOLVER
// and differs from this on CUDA. All three LU ops come from cuBLAS on NVIDIA
// (cublas.cc:1493, :1453, :1521), like geqrf/orgqr and unlike potrf. See the
// note in src/dispatch/entry_points/factorization.cc, and the latent gate defect
// it records. It used to apply to getrs's batch <= 1 arm -- a cuBLAS-gated TU
// calling cusolverDnXgetrs -- but that arm was deleted in WP6's repair pass,
// because it also read the packed int32 pivots as genuine int64 and aborted.
//
// PASS THE ARGUMENT EXPLICITLY. resolve_ormqr_route has the same defaulted
// parameter and ormqr.hh:209 calls it with two arguments, taking the `= true`
// default -- so ormqr never reaches the vendor-free fallback at
// route_resolve.hh:113-127 at all. syev omits it too (syev.hh:948). Do not copy
// either.
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// getrf into the coverage table: resolve_route records every op that goes through
// it (route_resolve.hh:178-217), slicing GetrfShape to OpShape.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_getrf_route(Route forced, const GetrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
