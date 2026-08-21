#pragma once

// POTRF's routing table.
//
// WHY THE SPLIT EXISTS, RESTATED FOR AN OP WHOSE NATIVE KERNEL ARRIVED LAST
//
// route_gemm.hh:25-28 states the rule this file obeys:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the alias table (route_env.hh), not here.
//
// For POTRF that rule is the whole work package, and the WP4 spec breaks it in
// both directions at once. WP4_POTRF_SPEC.md:559/:567 put
// `batch >= kPotrfCtaMinBatch` and `batch >= kPotrfBlockedMinBatch` inside
// supports(), under the heading "Hard gate (correctness/fit)", and :574 sets
// both to INT_MAX at merge "i.e. both native providers are reachable only by
// force". Two failures follow:
//
//   1. THE VENDOR-FREE FALLBACK DIES. route_resolve.hh:60-63 re-walks the
//      candidate order testing ONLY `is_native(*r) && Table::supports(*r, s)`.
//      With a batch threshold in supports(), potrf keeps throwing NoRouteError
//      in a -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF build AFTER the native kernel is
//      written and linked -- which is the exact failure WP4 exists to remove.
//
//   2. "REACHABLE ONLY BY FORCE" IS FALSE. route_resolve.hh:8-10 says a forced
//      route bypasses preferred() but NEVER supports(), implemented at :101
//      (`if (Table::supports(forced, s)) return forced;`) falling through to
//      automatic() at :111, and the bare-origin branch at :87-98 gates on
//      supports() in both of its walks. So every test that pinned the native
//      route would silently run cuSOLVER and pass GREEN over an untested
//      kernel.
//
// So: every number in spec 10.2's table, both batch thresholds, the
// "n > 1024 blocked loses" prior and spec 10.3's 0.90x gate belong in
// preferred(), and the merge state is preferred() == false everywhere. That is
// literally how trsm shipped (route_trsm.hh:53-55).
//
// THE ENV VARIABLE IS BATCHLAS_POTRF_ROUTE, and it needs no registration:
// parse_route_env (route_env.hh:214-217) builds the canonical name from
// op_env_stem(Op::potrf) == "POTRF". legacy_variable_for (route_env.hh:109-121)
// has no Op::potrf case and must not gain one -- BATCHLAS_POTRF_PROVIDER never
// shipped and is read by nothing, so inventing a legacy spelling for it would
// create the very hazard parse_legacy_route_value documents. Values that reach
// this table: "cta" / "blocked" (a bare algorithm implies Origin::Native),
// "native", "vendor", "native:cta". Unset means {Auto, Auto}
// (route_env.hh:145-148). tests/route_vocabulary_tests.cc pins that the
// variable is actually READ, because the spec's BATCHLAS_POTRF_PROVIDER was
// read by nothing at all.
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE. The convention is
// syev's (syev.hh:774-776), because potrf is the same shape of op: a single
// square in-place operand.
//
//     s.m = A.rows()      s.n = A.cols()      s.k = A.rows() == THE ORDER
//
// m and n are the two extents SEPARATELY so that the `m == n` gate is
// representable at all -- spec 10.1's predicate signature takes only `int n`
// and cannot express non-squareness. order() below is the only spelling this
// file uses for the order, so a later predicate cannot pick the wrong field.
//
// STATUS. BOTH native tiers are LINKED. The CTA kernel is
// src/extensions/potrf_cta.cc (both Uplo, all four scalar types, order <=
// potrf_cta_max_n_for_slm<T>(runtime local_mem - 4096) == 155/109/109/77 on this
// box); the BLOCKED driver is src/extensions/potrf_blocked.cc (WP4 Phase 2,
// Uplo::Lower only, any order, all four types), so potrf_blocked_available<T>()
// is now true and the Blocked arm below is supported wherever its gates pass.
//
// preferred() is still FALSE EVERYWHERE, so in a vendor-present build
// Origin::Auto keeps taking cuSOLVER for every shape and no existing decision
// moves: scripts/route_diff.sh across either landing shows ZERO changed
// non-potrf rows, and the only new native decisions are the ones
// tests/potrf_tests.cc FORCES with BATCHLAS_POTRF_ROUTE=cta|blocked. What did
// change is the native_route_supported column on potrf's rows, 0 -> 1, which is
// the point.
//
// WARNING FOR ANYONE USING route_diff.sh AS THE MERGE GATE ON THE PHASE 2 FLIP:
// no test in the suite issues a potrf ABOVE the CTA ceiling through the facade
// (ortho_tests runs at k=5 and dim=12; potrf_tests' order sweep tops out at the
// ceiling and its one over-ceiling case calls supports() and the direct entry
// point, neither of which records a coverage row). So the capture will report
// IDENTICAL across this change, and that is not evidence of anything -- it is
// the exact failure mode that script's own header warns about. Write the
// facade-routed over-ceiling test first, capture second, and give it an `n` in a
// shape_class bucket no CTA-sized call touches (route.hh:254-261 buckets by
// power of two and coverage.cc:277 is first-writer-wins).
//
// In a VENDOR-FREE build the same table now hands a caller the CTA route at
// order <= the ceiling and the Blocked route above it (route_resolve.hh:60-63)
// instead of throwing NoRouteError. That is the whole of WP4 Phases 1 and 2.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// POTRF's routing reads three things OpShape has no field for, so the op
// extends it, exactly as TrsmShape does (route_trsm.hh:92-114) and GesvdShape
// does (route_gesvd.hh:31-42). uplo is NOT among the extras: OpShape already
// carries it (route.hh:233).
// ---------------------------------------------------------------------------
struct PotrfShape : OpShape {
    // sycl_potrf::potrf_cta_max_n_for_slm<T>(this device's budget), copied in by
    // the shape builder. ZERO MEANS THE NATIVE CTA KERNEL IS ABSENT FROM THIS
    // BUILD, and it correctly makes both native routes unsupported rather than
    // selectable-but-unimplemented. Same convention as TrsmShape::cta_max_n,
    // pinned by RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable.
    //
    // It is asked of the DEVICE and not of a constant: the ceiling is a pure
    // function of the local-memory budget, so a build-time number would make
    // supports() claim an unlaunchable route on a device with less of it.
    //
    // NOT A constexpr LITERAL HERE, for route_trsm.hh:62-72's reason. The
    // formula lives next to the kernel, in src/extensions/potrf_cta.cc, where
    // the SAME function sizes the launch's local_accessor -- so the ceiling this
    // table advertises and the allocation the kernel makes cannot disagree.
    //
    // What the measurement says, recorded here so the small numbers cannot be
    // shipped by accident: WP4_POTRF_SPEC.md:273's {float 105, double 74,
    // cfloat 74, cdouble 52} are derived from `slm_budget = 45056`, and that
    // budget is refuted. build/include/batchlas/device_limits.hh's 49152 is
    // HARDCODED by cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_*
    // pattern and is not a detected property at all -- the detection routine
    // never queries local_mem_size. WP4 step 0.2 queried it: this box reports
    // sycl::info::device::local_mem_size == 101,376 B, and a kernel with 0 B
    // static shared launches at exactly that
    // (experiments/wp4_potrf/slm/README.md, slm_probe_gpu0.log; cudaDeviceProp
    // agrees, sharedMemPerBlockOptin == 101,376). Re-derived at a 97,280 B
    // budget (runtime - 4096 B reserve) the ceilings are {float 155, double
    // 109, cfloat 109, cdouble 77}, and all four were launched cold and
    // computed the right answer (maxn_fitcheck.csv). Shipping 105 would leave
    // float n in 106..155 with NO ROUTE AT ALL in a vendor-free build
    // (route_resolve.hh:60-63).
    int cta_max_n = 0;

    // Whether the BLOCKED driver exists in this build. Separate from cta_max_n
    // because the two are independent capabilities: blocked is what serves
    // orders ABOVE cta_max_n, and until it is written those orders have no
    // native route.
    //
    // This is not belt-and-braces (route_trsm.hh:99-110): reporting Blocked as
    // supported while it does not exist makes resolve_route hand a vendor-free
    // caller a route the facade cannot service. The table must describe the
    // BUILD, not the design.
    bool blocked_available = false;

    // Does this device offer sub-group size 32 -- ENUMERATED from
    // sycl::info::device::sub_group_sizes, never inferred from
    // OpShape::max_sub_group.
    //
    // WHY NOT max_sub_group. Device::get_property(MAX_SUB_GROUP_SIZE) returns
    // `sub_group_sizes()[0]` (src/util/queue-impl.cc:325) -- the FIRST entry of
    // the supported list, not the maximum -- so syev's `s.max_sub_group >= 32`
    // (syev.hh:837) is wrong in both directions: a device reporting {8,16,32}
    // reads 8 and is refused although it does support 32, and a device
    // reporting {64} reads 64 and is ACCEPTED although it does not have 32 at
    // all. The second is a launch abort for a kernel carrying
    // [[sycl::reqd_sub_group_size(32)]], i.e. exactly what supports() exists to
    // exclude. Device::supports_sub_group_size does the enumeration.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

// CTA first, then the blocked driver that calls CTA as its diagonal leaf, then
// the vendor. The order is a CAPABILITY LADDER, not a preference: CTA serves
// only order <= cta_max_n and blocked serves the rest. With preferred()
// all-false today the order matters only in the vendor-off walk at
// route_resolve.hh:60-63, where the tighter route is the right one to try
// first.
//
// A file-scope array of natural length with sizeof-computed bounds, NOT
// spec:574's std::array<Provider,6>: route_gemm.hh:43-46 records that this
// "removes the truncation hazard of the four hand-counted std::array<Provider,6>
// sites".
inline constexpr Route kPotrfOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::potrf, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a WRONG ANSWER or could not
    // run at all", never "the kernel would be slow". Nothing here is
    // type-dependent: the whole per-type difference is SLM capacity, and that
    // arrives as s.cta_max_n.
    static bool supports(Route r, const PotrfShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. SQUARE. A = L L^H is defined only for a square operand; there is
        //    no Cholesky factor of a non-square view to compute. The kernel
        //    also drives both extents from ONE order, so a non-square view is
        //    read or written past one of its bounds. WRONG ANSWER / OOB, not
        //    speed. syev's table opens with the same gate (syev.hh:828).
        if (s.m != s.n) return false;

        // 2. GPU ONLY. Not a speed judgement: the native path is a SYCL
        //    nd_range kernel with a work-group-collective staging phase and a
        //    local_accessor tile. There is no host implementation of it to fall
        //    back on, so a CPU queue has to reach netlib. Same wording and same
        //    reason as route_trsm.hh:138-142.
        if (!s.is_gpu) return false;

        // 3. SUB-GROUP SIZE 32. The kernel will carry
        //    [[sycl::reqd_sub_group_size(32)]] (the precedent set at
        //    syev_cta_fused.cc:185, gesvdj_cta.cc:297, sytrd_sb2st_cta.cc:403).
        //    On a device whose sub_group_sizes do not contain 32 the launch is
        //    REJECTED. That is "cannot run", not "runs slowly". It gates BOTH
        //    native arms, because the blocked driver's diagonal leaf IS that
        //    same device function.
        if (!s.has_sg32) return false;

        // 4. HETEROGENEOUS BATCH. One launch covers the whole batch with a
        //    single (order, ld, stride) tuple and reads at
        //    data_ptr() + b*stride with the CAPACITY extents, so a view with
        //    per-item active dims (matrix.hh:1034; publicly constructible via
        //    Matrix::set_active_dims and MatrixView::with_active_dims) would
        //    silently factorise the wrong order in place for every item after
        //    the first. potrf has no batch walker -- unlike gemm, where WP2 C2
        //    made this merely un-preferred because the facade walks the batch
        //    (route_gemm.hh:70-80) -- so for potrf it is a correctness gate. If
        //    a walker is ever written, this line moves to preferred(), and not
        //    before.
        //
        //    Not merely defensive: netlib's BATCHED path already honours the
        //    per-item extents (netlib_lapack.cc:1029 calls A_view[i].rows()),
        //    so routing such a view natively would disagree with a path in this
        //    tree that gets it right.
        if (s.heterogeneous_batch) return false;

        // 5. DEGENERATE EXTENTS. The panel loop and the tile index map are
        //    undefined for an empty triangle or an empty batch. Disagreement
        //    BETWEEN views is not tested here -- OpShape holds one batch, so
        //    the shape builder reports that by returning no shape at all (the
        //    gemm_op_shape pattern, src/backends/gemm_variant.hh:189-197).
        if (s.order() < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                // The whole matrix is resident in a local_accessor tile whose
                // extent is chosen at launch, so the largest order the kernel
                // was sized for is a HARD CAPACITY, not a tuning knob: above it
                // there is no launchable configuration. Zero means the kernel
                // is not in this build at all.
                //
                // NO uplo GATE, AND THAT WAS VERIFIED AGAINST THE KERNEL
                // RATHER THAN ASSUMED. Uplo::Upper is not a second algorithm in
                // the CTA path: A = U^H U with U upper is the same recurrence on
                // the transformed tile S(i,c) = conj(A(c,i)), so it is a
                // load/store transform and the factorisation compiles once per
                // (T, NB, TS, Scope). tests/potrf_tests.cc's
                // ResidualBothTriangles and OtherTriangleIsNeitherReadNorWritten
                // both run the whole order sweep under BOTH Uplo values against
                // a host multiply-back residual. Had the kernel shipped
                // Lower-only, this arm would need
                // `if (s.uplo != Uplo::Lower) return false;` -- omitting it then
                // is a SILENT WRONG ANSWER, not a slowdown. The Blocked arm
                // below still carries exactly that line.
                if (s.cta_max_n < 1) return false;
                return s.order() <= s.cta_max_n;

            case Algorithm::Blocked:
                // The blocked driver's diagonal-block factorisation IS the CTA
                // kernel, so it inherits the presence gate but NOT the cap --
                // it splits the order into blocks of at most cta_max_n itself.
                // It also has to exist, which is a separate question
                // (route_trsm.hh:172-177 is the identical pair).
                //
                // NO LOWER BOUND ON ORDER. spec:567's
                // `if (n <= potrf_cta_max_n<T>(...)) return false;` is a FIT
                // judgement between two native routes, not a correctness claim;
                // the tree expresses that with the order ladder above. With it
                // here, a forced `blocked` at small n does not measure the
                // blocked kernel: per route_resolve.hh:101-111 it falls back to
                // automatic(), which at merge returns {Vendor, Auto}. spec 10.3
                // asks for all three routes pinned at overlapping n; that
                // overlap only exists if this arm carries no lower bound.
                //
                // Uplo::Upper is a CORRECTNESS gate until the driver mirrors:
                // it implements the Lower recurrence only, and handed an Upper
                // view it would read and overwrite the wrong triangle. NOT the
                // "merely slower" kind of false. Contrast syev, whose
                // blocked/two-stage arms accept Upper because both MIRROR the
                // upper triangle first (syev.hh:840-847, uplo_mirror.hh) -- if
                // potrf's driver adopts that trick, this line is what it
                // deletes.
                if (s.uplo != Uplo::Lower) return false;
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                // Including Algorithm::Auto. potrf has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:87-98).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, DELIBERATELY, AND THAT IS THE MERGE STATE OF THIS FILE.
    //
    // Nothing about potrf has been measured. route_trsm.hh:53-55 records the
    // precedent in as many words: "preferred() was all-false until WP3 step 9
    // measured the grid". All-false means Origin::Auto takes the vendor
    // wherever one exists (route_resolve.hh:57-58 finds no supported AND
    // preferred route, :65 returns {Vendor, Auto}), so merging this table moves
    // ZERO traffic -- scripts/route_diff.sh against the parent capture must
    // show no existing decision changing.
    //
    // It is ALSO what keeps the vendor-free build honest. Un-preferred is not
    // unroutable: route_resolve.hh:60-63 still hands a vendor-free caller any
    // SUPPORTED native route. So the day potrf_cta_max_n<T>() comes off zero, a
    // vendor-free build starts using the native kernel and a vendor-present
    // build does not -- which is the correct order of events.
    //
    // WHAT GOES HERE WHEN THE SPEC 10.3 GRID IS MEASURED, and nowhere else:
    // spec 10.2's cell table, both batch thresholds, and the per-clause cell
    // citations. NOT tuning_params.hh -- that header is for nb/L/G, and putting
    // a routing threshold there mixes a route decision into a workspace-sizing
    // input. Each clause cites the CSV rows it comes from, as
    // route_trsm.hh:188-325 does. A cell is flipped only where spec:600's
    // three-part gate fires: t_native <= 0.90 * t_vendor at saturation, AND the
    // section-7 accuracy harnesses show no regression, AND ortho_benchmark
    // shows the win end to end. A 2.16x kernel win in this repo once turned
    // into an 11% gesvd loss; a predicate justified only at kernel level does
    // not clear this gate.
    //
    // Starting hypothesis for that grid, from step 0.2's occupancy sweep and
    // NOT a routing claim: at the fit ceiling the kernel is 1 block/SM and 8.3%
    // warp occupancy at wg=128, so the preferred window is expected to be well
    // inside supports()' ceiling -- around the >= 4 blocks/SM line
    // (24,320 B -> 77/54/54/38). Measure it; do not encode it here.
    static bool preferred(Route r, const PotrfShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    static constexpr const Route* order_begin() { return kPotrfOrder; }
    static constexpr const Route* order_end() {
        return kPotrfOrder + (sizeof(kPotrfOrder) / sizeof(kPotrfOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default comes from
// legacy_unset_default(Op::potrf), which is {Auto, Auto} for every op since
// WP2 E6 (route_env.hh:145-148).
//
// `vendor_available` is the vendor-free switch, and for potrf the facade knows
// it as a compile-time fact: dispatch::solver_vendor_available<B>
// (vendor_available.hh:47-52) -- NOT factorization_vendor_available, which is
// cuBLAS and differs from this on CUDA. syev omits the argument entirely
// (syev.hh:948 takes the default `true`) and therefore never reaches the
// vendor-free fallback at all; do not copy that.
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// potrf into the coverage table: resolve_route records every op that goes
// through it (route_resolve.hh:130-152), slicing PotrfShape to OpShape.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_potrf_route(Route forced, const PotrfShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::potrf, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
