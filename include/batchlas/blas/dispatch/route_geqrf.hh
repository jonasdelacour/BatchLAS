#pragma once

// GEQRF's routing table (WP5 scaffolding -- the WP4 Phase 0 equivalent).
//
// MODELLED LINE FOR LINE ON route_potrf.hh, and the three rules that file obeys
// are restated here because geqrf breaks the FIRST of them in a way potrf does
// not:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the shape builder (src/backends/geqrf_route.hh),
//                     not here. This header is PURE.
//
// WHY THE SPLIT MATTERS FOR geqrf SPECIFICALLY. route_resolve.hh:60-63 re-walks
// the candidate order testing ONLY `is_native(*r) && Table::supports(*r, s)`, so
// any speed threshold that lands in supports() removes geqrf's vendor-free route
// entirely -- and geqrf is the LINCHPIN of the vendor-free burn-down: four
// suites (ormqr_tests, ormqr_cta_tests, ormqr_blocked_tests, orgqr_tests) fail
// today with "no route for geqrf<T> ... built without cuBLAS" thrown from their
// SETUP, which calls geqrf to make the reflectors they then feed to the op under
// test. And route_resolve.hh:8-10/:101 say a forced route bypasses preferred()
// but NEVER supports(), so a test that pins BATCHLAS_GEQRF_ROUTE and hits a
// wrongly-placed gate silently runs cuSOLVER and passes GREEN over a kernel
// nothing executed.
//
// (Correction to the WP5 brief, measured: a native geqrf does NOT by itself
// close all four of those suites. Only orgqr_tests' GPU rows are closable by
// WP5 alone. ormqr_blocked_tests calls ormqr_vendor_or_throw DIRECTLY as its
// reference in all five tests; ormqr_cta_tests builds its reference on a host
// Queue("cpu") with netlib geqrf AND netlib ormqr; ormqr_tests has one case that
// pins BATCHLAS_ORMQR_PROVIDER=vendor. Those are test-code dependencies on the
// vendor, not kernel gaps. Recorded so the burn-down claim is not repeated.)
//
// THE ENV VARIABLE IS BATCHLAS_GEQRF_ROUTE, and it needs no registration:
// parse_route_env (route_env.hh:214-217) builds the canonical name from
// op_env_stem(Op::geqrf) == "GEQRF". legacy_variable_for (route_env.hh:109-121)
// has NO Op::geqrf case and must not gain one -- no BATCHLAS_GEQRF_PROVIDER ever
// shipped, so inventing a legacy spelling would create the hazard
// parse_legacy_route_value documents. (Op::ormqr at :118 is the one QR-family op
// that does have a legacy variable; that is not a precedent for this one.)
// Values that reach this table: "cta" / "blocked" (a bare algorithm implies
// Origin::Native), "native", "vendor", "native:cta", "native:blocked". Unset
// means {Auto, Auto} (route_env.hh:145-148). An UNRECOGNISED value is also
// {Auto, Auto}, which with preferred() all-false is the VENDOR -- so a "native"
// run that looks identical to the vendor probably IS the vendor;
// tests/route_vocabulary_tests.cc pins that the variable is actually READ.
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE, BECAUSE IT IS WHERE
// geqrf DEPARTS FROM potrf.
//
//     s.m = A.rows()   s.n = A.cols()   s.k = min(rows, cols) == THE REFLECTOR
//                                                                COUNT
//
// potrf sets m = n = k = the order (route_potrf.hh:87-95) because its operand is
// square by definition. geqrf's operand is RECTANGULAR AND THAT IS THE POINT --
// options.hh:727-730 says so in as many words ("No squareness check: rectangular
// A is the entire point of geqrf"), and the library's own callers pass tall
// panels (src/extensions/band_reduction.cc:595, sytrd_sy2sb.cc:504). So COPYING
// route_potrf.hh:213's `if (s.m != s.n) return false;` into this file would be a
// wrong edit that strips geqrf of its main use. The convention here is
// ormqr_op_shape's (ormqr.hh:182-192). reflectors() below is the only spelling
// this file uses for k, so a later predicate cannot pick the wrong field -- the
// same discipline as PotrfShape::order().
//
// STATUS: BOTH NATIVE KERNELS ARE LINKED as of WP5 --
// src/extensions/geqrf_cta.cc (one m x n matrix resident in local memory) and
// src/extensions/geqrf_blocked.cc (panel factorisation plus a WY trailing update
// through the routed gemm), sharing ONE panel device body in
// geqrf_cta_device.hh. So on a GPU offering sub-group size 32,
// geqrf_cta_max_m_for_slm<T>() and geqrf_cta_max_elems_for_slm<T>() answer from
// the device's real local_mem_size and geqrf_blocked_available<T>() is true.
//
// THAT MOVES NO VENDOR-PRESENT TRAFFIC. preferred() below is still false for
// both arms, so Origin::Auto still takes the vendor wherever one exists
// (route_resolve.hh:57-58, :65). What HAS changed is the vendor-free build: the
// hand-over at route_resolve.hh:60-63 now finds a supported native route instead
// of throwing NoRouteError, which is exactly the event this work package exists
// to cause. tests/route_vocabulary_tests.cc's RouteGeqrf block builds its own
// GeqrfShape values and is unaffected either way.
//
// WARNING FOR ANYONE USING scripts/route_diff.sh AS A GATE ON THIS OP, AND IT IS
// WORSE THAN potrf's (route_potrf.hh:79-92). coverage.cc:52-58's variant_key
// carries only uplo/side/diag/transA/transB, and OpShape::shape_class()
// (route.hh:255-261) buckets max(m,n,k) by power of two. So a tall panel
// (m=1024, n=32) and a square (m=1024, n=1024) collapse into ONE coverage row,
// and coverage.cc:277 is first-writer-wins. A route_diff capture cannot
// distinguish geqrf's two most important shape classes AT ALL. Write the
// facade-routed panel-shaped test FIRST, capture SECOND, and pick extents in a
// max_dim bucket no other geqrf call in the suite touches.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>
#include <type_traits>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// GEQRF's routing reads four things OpShape has no field for, so the op extends
// it, exactly as PotrfShape (route_potrf.hh:105-186) and TrsmShape
// (route_trsm.hh:92-114) do.
// ---------------------------------------------------------------------------
struct GeqrfShape : OpShape {
    // THE CTA CAPACITY IS A PRODUCT, NOT TWO INDEPENDENT BOUNDS, and getting
    // that wrong is the first way this table could claim an unlaunchable route.
    // The CTA kernel holds the whole m x n panel resident in a local_accessor,
    // so what fits is governed by m*n; a pair of per-extent ceilings would
    // accept a 155 x 155 float panel because each extent is "within range" while
    // the tile it needs is many times the budget.
    //
    // cta_max_m IS NOT INDEPENDENTLY BINDING WITH THE SHIPPED LAYOUT, and the
    // honest thing is to say so here rather than let the field read as a live
    // second constraint. The scaffolding predicted one -- "one Householder vector
    // spans a whole column, so the reduction that forms it carries its own
    // ceiling" -- and the implemented kernel does not have it: the reduction is a
    // work-group collective over a strided row loop and needs no per-row storage,
    // so the tile is exactly m*n scalars and the largest admissible m at n = 1 IS
    // the area bound (src/extensions/geqrf_cta.cc says the same at the
    // definitions). The field is kept, and supports() keeps testing both, for
    // three reasons: RouteGeqrf.CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds
    // pins that the pair is tested as an area AND a height; it is the number that
    // MOVES the moment a per-row resident array is added, at which point the area
    // bound stops describing the height; and the alternative -- inventing a
    // tighter height so the field looks load-bearing -- would be a SPEED
    // threshold in supports(), which deletes the vendor-free route above it
    // (route_resolve.hh:60-63).
    //
    // ZERO ON EITHER STILL MEANS THE CTA KERNEL IS ABSENT FROM THIS BUILD, and it
    // correctly makes BOTH native routes unsupported rather than
    // selectable-but-unimplemented. Same convention as TrsmShape::cta_max_n,
    // pinned by RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable, and as
    // PotrfShape::cta_max_n. It is no longer the state on a real GPU.
    //
    // ASKED OF THE DEVICE, never of a constant, for route_potrf.hh:114-127's
    // reason: the ceiling is a pure function of the local-memory budget, so a
    // build-time number makes supports() claim an unlaunchable route on a device
    // with less of it. In particular it must NOT come from
    // build/include/batchlas/device_limits.hh, whose 49152 is HARDCODED by
    // cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern and is
    // 2.06x wrong on this box -- WP4's finding W1, measured
    // (sycl::info::device::local_mem_size == 101,376 B here).
    //
    // The formulae live next to the kernel, in src/extensions/geqrf_cta.cc,
    // where the SAME functions will size the launch's local_accessor -- so the
    // ceiling this table advertises and the allocation the kernel makes cannot
    // disagree (route_trsm.hh:62-72).
    int cta_max_m = 0;
    int64_t cta_max_elems = 0;

    // Whether the BLOCKED driver exists in this build. Separate from the CTA
    // capacity because the two are independent capabilities: blocked is what
    // serves panels the CTA tile cannot hold, and until it is written those
    // shapes have no native route.
    //
    // Not belt-and-braces (route_trsm.hh:99-110): reporting Blocked as supported
    // while it does not exist makes resolve_route hand a VENDOR-FREE caller a
    // route the facade cannot service, i.e. a std::logic_error instead of a
    // factorisation. The table must describe the BUILD, not the design.
    bool blocked_available = false;

    // Does this device offer sub-group size 32 -- ENUMERATED from
    // sycl::info::device::sub_group_sizes (Device::supports_sub_group_size,
    // sycl-device-queue.hh:190), never inferred from OpShape::max_sub_group.
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

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    // The number of Householder reflectors, min(rows, cols). NOT an "order":
    // geqrf has no order. Named so that a later predicate cannot reach for m or
    // n when it meant k.
    int64_t reflectors() const { return k; }
};

// CTA first, then the blocked driver whose panel leaf will BE the CTA device
// function, then the vendor. The order is a CAPABILITY LADDER, not a preference:
// CTA serves only panels the resident tile can hold and blocked serves the rest.
// With preferred() all-false today the order matters only in the vendor-off walk
// at route_resolve.hh:60-63, where the tighter route is the right one to try
// first.
//
// A file-scope array of natural length with sizeof-computed bounds, NOT a
// std::array<Provider,6>: route_gemm.hh:43-46 records that this "removes the
// truncation hazard of the four hand-counted std::array<Provider,6> sites".
inline constexpr Route kGeqrfOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::geqrf, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a WRONG ANSWER or could not
    // run at all", never "the kernel would be slow". Nothing here is
    // type-dependent: the whole per-type difference is SLM capacity, and that
    // arrives as cta_max_m / cta_max_elems.
    static bool supports(Route r, const GeqrfShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. NO SQUARENESS GATE, DELIBERATELY. See the FIELD MAPPING note at the
        //    top of this file: rectangular A is the entire point of geqrf
        //    (options.hh:727-730), and both in-tree callers pass tall panels.
        //    route_potrf.hh:213's `if (s.m != s.n) return false;` does not belong
        //    here and copying it is a silent loss of the op's main use.

        // 2. TALL OR SQUARE ONLY (m >= n). This IS a correctness gate for the
        //    drivers WP5 will write: both are panel-oriented right-looking
        //    schedules over min(m,n) reflectors laid out down columns, and handed
        //    a WIDE view (m < n) the trailing update walks past the bottom of the
        //    panel. It is the omission-is-a-silent-wrong-answer class that
        //    route_potrf.hh:243-252 documents for Uplo.
        //
        //    It is also the CONSERVATIVE direction, which is why it lands with
        //    the table rather than with the kernel: a superfluous gate sends a
        //    wide geqrf to the vendor (loud, and in a vendor-free build a
        //    NoRouteError naming the op), while a missing one returns wrong
        //    numbers quietly. Nothing measured is lost -- every geqrf shape in
        //    the WP5 baseline is square, and both in-tree callers are tall.
        //    DELETE THIS LINE only when a driver actually handles m < n, and
        //    with a test that fails without it.
        if (s.m < s.n) return false;

        // 3. GPU ONLY. Not a speed judgement: the native path will be a SYCL
        //    nd_range kernel with a work-group-collective staging phase and a
        //    local_accessor tile, and there is no host implementation of it to
        //    fall back on, so a CPU queue has to reach netlib. Same wording and
        //    same reason as route_trsm.hh:138-142 and route_potrf.hh:222.
        //
        //    It is also why the geqrf kernel TUs sit in EXTENSIONS_CTA_SOURCES,
        //    the only object library configured NO_CPU_TARGETS
        //    (src/CMakeLists.txt:67-71).
        if (!s.is_gpu) return false;

        // 4. SUB-GROUP SIZE 32. The kernels will carry
        //    [[sycl::reqd_sub_group_size(32)]] (the precedent set at
        //    syev_cta_fused.cc:185, gesvdj_cta.cc:297, sytrd_sb2st_cta.cc:403).
        //    On a device whose sub_group_sizes do not contain 32 the launch is
        //    REJECTED. That is "cannot run", not "runs slowly". It gates BOTH
        //    native arms, because the blocked driver's panel leaf IS that same
        //    device function.
        if (!s.has_sg32) return false;

        // 5. HETEROGENEOUS BATCH. One launch will cover the whole batch with a
        //    single (m, n, ld, stride) tuple and read at data_ptr() + b*stride
        //    with the CAPACITY extents, so a view with per-item active dims
        //    (matrix.hh:1034; publicly constructible via Matrix::set_active_dims
        //    and MatrixView::with_active_dims) would silently factorise the
        //    wrong extents in place for every item after the first. geqrf has no
        //    batch walker -- unlike gemm, where WP2 C2 made this merely
        //    un-preferred because the facade walks the batch
        //    (route_gemm.hh:70-80) -- so for geqrf it is a correctness gate.
        //
        //    THE JUSTIFICATION IS THE OPPOSITE OF potrf's AND MUST NOT BE COPIED
        //    FROM IT. route_potrf.hh:255-257 argues the gate is needed because
        //    netlib's potrf already honours per-item extents
        //    (netlib_lapack.cc:1029 calls A_view[i].rows()), so a native route
        //    would disagree with a path in this tree that gets it right. For
        //    geqrf netlib does NOT: netlib_lapack.cc:1406-1417 hoists m and n
        //    from A_view.rows()/cols() OUTSIDE the loop and only indexes
        //    A_view[i].data_ptr(). There is no path in this tree that gets
        //    heterogeneous-batch QR right. The gate is here because the native
        //    kernel cannot serve it, full stop.
        if (s.heterogeneous_batch) return false;

        // 6. DEGENERATE EXTENTS. The panel loop and the tile index map are
        //    undefined for an empty panel or an empty batch. Disagreement
        //    BETWEEN views is not tested here -- OpShape holds one batch, so the
        //    shape builder reports that by returning no shape at all (the
        //    gemm_op_shape pattern, src/backends/gemm_variant.hh:189-197).
        if (s.m < 1 || s.n < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                // The whole panel is resident in a local_accessor tile whose
                // extent is chosen at launch, so this is a HARD CAPACITY, not a
                // tuning knob: above it there is no launchable configuration.
                // Zero on either field means the kernel is not in this build at
                // all -- which is the state today.
                //
                // BOTH tests, and the area one written as a product: see the
                // GeqrfShape::cta_max_elems note. int64_t arithmetic throughout,
                // because m*n overflows int at extents this API accepts.
                if (s.cta_max_m < 1 || s.cta_max_elems < 1) return false;
                return s.m <= static_cast<int64_t>(s.cta_max_m) &&
                       s.m * s.n <= s.cta_max_elems;

            case Algorithm::Blocked:
                // The blocked driver's panel factorisation IS the CTA kernel, so
                // it inherits the PRESENCE gate but NOT the capacity -- it splits
                // the panel into blocks the leaf can hold itself. It also has to
                // exist, which is a separate question (route_trsm.hh:172-177 is
                // the identical pair).
                //
                // NO LOWER BOUND ON THE EXTENTS. "n <= the CTA capacity so
                // blocked should be false" is a FIT judgement between two native
                // routes, not a correctness claim, and route_potrf.hh:284-296
                // records what putting it here costs: per route_resolve.hh:101 a
                // forced `blocked` at a small shape then falls through to
                // automatic() at :111, which at merge returns {Vendor, Auto} --
                // so the test that pinned the blocked driver measures cuSOLVER
                // and passes green. Pinning all three routes at overlapping
                // extents only works if this arm carries no lower bound.
                return s.blocked_available && s.cta_max_m >= 1 && s.cta_max_elems >= 1;

            default:
                // Including Algorithm::Auto. geqrf has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:87-98).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, DELIBERATELY, AND THAT IS THE MERGE STATE OF THIS FILE.
    //
    // Nothing native about geqrf has been measured because nothing native about
    // geqrf exists yet. route_trsm.hh:53-55 records the precedent in as many
    // words: "preferred() was all-false until WP3 step 9 measured the grid".
    // All-false means Origin::Auto takes the vendor wherever one exists
    // (route_resolve.hh:57-58 finds no supported AND preferred route, :65
    // returns {Vendor, Auto}), so merging this table moves ZERO traffic.
    //
    // It is ALSO what keeps the vendor-free build honest. Un-preferred is not
    // unroutable: route_resolve.hh:60-63 still hands a vendor-free caller any
    // SUPPORTED native route. So the day geqrf_cta_max_elems_for_slm<T>() comes
    // off zero, a vendor-free build starts using the native kernel and a
    // vendor-present build does not -- which is the correct order of events, and
    // the one that turns orgqr_tests' GPU rows on the burn-down without moving a
    // single vendor-present decision.
    //
    // WHAT GOES HERE WHEN THE GRID IS MEASURED, and nowhere else: the per-cell
    // window, with each clause citing the CSV rows it comes from as
    // route_trsm.hh:188-325 does. NOT tuning_params.hh -- that header is for
    // nb/L/G, and putting a routing threshold there mixes a route decision into a
    // workspace-sizing input.
    //
    // TWO MEASURED FACTS THAT WILL SHAPE THIS FUNCTION, recorded so they are not
    // re-derived, and NOT encoded as predicates because there is no kernel to
    // predicate on yet (experiments/wp5_qr/baseline/README.md):
    //   * The absolute target is the ms column of the baseline table, not its
    //     GFLOP/s column: cuBLAS geqrfBatched is latency-bound and nowhere near
    //     saturated at n >= 512 (float n=2048: 21361 ms at b=32, 23151 ms at
    //     b=256), so its GFLOP/s at those cells is not a statement about its
    //     ceiling.
    //   * The trailing update is NOT where this is decided. Summed over all 18
    //     panels of a real N=1024, nb=56, batch=128 factorisation, both WY GEMMs
    //     cost 33.40 ms vendor-free (float) against cuSOLVER's 2109.8 ms for the
    //     whole call -- 63.2x headroom, and 4.3x in the worst type (cdouble).
    //     The panel factorisation is the budget.
    static bool preferred(Route r, const GeqrfShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // ---- NATIVE-VS-NATIVE TIE-BREAK ---------------------------------------
    // "Among the native routes that CAN serve this shape, which is faster?"
    //
    // READ route_resolve.hh's note on why this is a third predicate before
    // editing it. In one line: it is consulted ONLY on the vendor-free walk, so
    // it moves nothing in a vendor-present build, whereas preferred() is
    // consulted regardless of vendor_available and would move both.
    //
    // WHY THIS EXISTS AT ALL. kGeqrfOrder lists CTA first, and with preferred()
    // all-false the vendor-free walk used to return the first SUPPORTED native
    // route unconditionally -- i.e. CTA everywhere the tile fits SLM. supports()
    // admits CTA to m*n <= cta_max_elems, which on this box is square n <= 155
    // for float and n <= 110 for double, and the tier sweep shows CTA is beaten
    // by the blocked driver well below both ceilings. That is a pure loss: the
    // better route is already linked into the same build.
    //
    // THE MEASUREMENT. Two independent sweeps agree, and the second is the one
    // that matters because it is an A/B of the SHIPPED DEFAULT.
    //
    // (a) experiments/wp5_qr/bench/tier_summary.txt -- square m == n, batch
    //     4096-8192, BOTH arms pinned and every pin verified to have taken (20 of
    //     44 cells in that sweep had a `cta` pin silently resolve to blocked and
    //     are excluded). Column is blocked_ms/cta_ms, so > 1 means CTA ahead:
    //
    //       float    n=48 2.686  n=64 2.034  n=80 2.037  n=96 1.294 | n=112 0.821  n=128 0.699
    //       double   n=32 1.049 | n=48 0.983  n=64 0.922  n=80 0.772  n=96 0.731
    //       cfloat   n=48 3.171  n=64 2.093  n=80 1.253  n=96 1.079  (ceiling n=110)
    //       cdouble  n=48 2.589  n=64 1.929                          (ceiling n=77)
    //
    // (b) The shipped default against a forced `cta` -- SAME BINARY, same
    //     session, interleaved, three reps, vendor-free build, and the forced
    //     arm's resolved route printed and checked to read native:cta on every
    //     row. This is what the window below actually buys, in ms:
    //
    //       float  n=112  9.15 vs 11.98 (1.31x)   double n=64   27.42 vs 29.72 (1.08x)
    //       float  n=128  9.97 vs 15.39 (1.54x)   double n=80   18.49 vs 24.07 (1.30x)
    //       float  n=155 16.39 vs 22.73 (1.39x)   double n=96   23.86 vs 32.75 (1.37x)
    //                                             double n=110  30.55 vs 42.52 (1.39x)
    //
    //     and, below the crossover, against a forced `blocked` -- the check that
    //     the window did not OVERSHOOT:
    //
    //       float  n=64   2.95 vs  5.70 (CTA 1.93x ahead)   cfloat  n=96 10.64 vs 10.72
    //       float  n=96   5.00 vs  6.02 (CTA 1.20x ahead)   cdouble n=64 59.18 vs 113.79
    //       double n=32  10.87 vs 11.42 (CTA 1.05x ahead)
    //       double n=48  19.34 vs 19.02 (blocked 1.7% ahead -- A TIE, see below)
    //
    // TWO CELLS ARE HONEST TIES AND ARE RESOLVED IN CTA'S FAVOUR ON PURPOSE.
    // double n=48 (blocked ahead by 1.7%) and cfloat n=96 (blocked ahead by 0.8%)
    // are both inside the run-to-run resolution of this harness. They go to CTA
    // because at a tie CTA is the better route for a reason that is not timing:
    // its workspace is ZERO -- the tile is local memory and tau is the caller's
    // span -- while the blocked driver allocates m*nb*batch of V plus T plus the
    // WY scratch. cfloat's advantage is collapsing as n approaches its ceiling
    // (3.171 -> 2.093 -> 1.253 -> 1.079 -> ~1.008 at n=96), so cfloat 97..110 is
    // the one band where this window is extrapolated rather than measured, and it
    // is the first place to look if this is ever re-measured.
    //
    // THE VARIABLE IS n, NOT m*n, AND THAT IS A MECHANISM ARGUMENT, NOT A
    // MEASURED ONE. CTA's serial cost is its per-reflector chain -- two
    // work-group reductions and three barriers each, k = min(m,n) times -- and
    // geqrf_panel_wg derives the work-group from n alone. m enters only through
    // supports()' m*n <= cta_max_elems fit gate, which already keeps genuinely
    // large panels off this arm. The sweep behind the numbers above is SQUARE
    // ONLY; a tall skinny panel (m=512, n=32, float) is CTA-eligible and is NOT
    // covered by a measured cell. It is left on CTA deliberately -- fewer
    // reflectors, one kernel, no trailing GEMM -- and that is the open question
    // recorded in experiments/wp5_qr/README.md rather than a claim.
    //
    // NOT A CORRECTNESS GATE. Everything here is "the other native route is
    // faster". Both arms remain fully supported() at every shape below, which is
    // what keeps a pinned `cta` at n=128 actually running CTA instead of falling
    // through to automatic() (route_resolve.hh:101) and measuring the thing it
    // was pinned away from.
    static bool native_tier_preferred(Route r, const GeqrfShape& s) {
        if (!is_native(r)) return true;

        // The crossover in n, per scalar type. Below it CTA is the better native
        // route; at or above it the blocked driver is.
        const int64_t cta_max_cols = [] () -> int64_t {
            if constexpr (std::is_same_v<T, float>) {
                return 96;      // 1.294 at 96, 0.821 at 112
            } else if constexpr (std::is_same_v<T, double>) {
                return 48;      // 1.049 at 32, 0.983 at 48 (a tie), 0.922 at 64
            } else {
                // Both complex types are ahead at every cell their capacity lets
                // the sweep reach -- cfloat 1.079 at its last measured n=96
                // against a ceiling of 110, cdouble 1.929 at n=64 against 77 --
                // so there is no measured crossover to encode and the fit gate in
                // supports() is the only ceiling. The trailing cfloat cells are
                // heading toward 1.0, so if this is ever re-measured, cfloat
                // 97..110 is where to look.
                return 1 << 30;
            }
        }();

        switch (r.algo) {
            case Algorithm::CTA:
                return s.cols() <= cta_max_cols;
            case Algorithm::Blocked:
                return s.cols() > cta_max_cols;
            default:
                return true;
        }
    }

    static constexpr const Route* order_begin() { return kGeqrfOrder; }
    static constexpr const Route* order_end() {
        return kGeqrfOrder + (sizeof(kGeqrfOrder) / sizeof(kGeqrfOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default comes from
// legacy_unset_default(Op::geqrf), which is {Auto, Auto} for every op since
// WP2 E6 (route_env.hh:145-148).
//
// `vendor_available` is the vendor-free switch, and the facade knows it as a
// compile-time fact: dispatch::factorization_vendor_available<B>
// (vendor_available.hh:41-45) -- NOT solver_vendor_available, which is cuSOLVER
// and differs from this on CUDA. See the note in
// src/dispatch/entry_points/factorization.cc on geqrf's vendor arm, and the
// latent gate defect it records.
//
// PASS THE ARGUMENT EXPLICITLY. resolve_ormqr_route has the same defaulted
// parameter and ormqr.hh:209 calls it with two arguments, taking the `= true`
// default -- so ormqr never reaches the vendor-free fallback at
// route_resolve.hh:60-63 at all. It happens not to matter for ormqr only because
// RouteTable<Op::ormqr,T>::preferred() returns native-first
// (route_ormqr.hh:78-80). syev omits it too (syev.hh:948). Do not copy either.
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// geqrf into the coverage table: resolve_route records every op that goes
// through it (route_resolve.hh:130-152), slicing GeqrfShape to OpShape.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_geqrf_route(Route forced, const GeqrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::geqrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
