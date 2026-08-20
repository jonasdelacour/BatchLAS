#pragma once

// TRSM's routing table.
//
// WHY THE SPLIT EXISTS, RESTATED FOR AN OP THAT HAS NO NATIVE KERNEL YET
//
// route_gemm.hh:25-28 states the rule this file obeys:
//
//     supports()   == correctness only, and nothing else. Never a speed cutoff.
//     preferred()  == the measured window. Returning false never makes a route
//                     ineligible, only un-preferred.
//     the env read == lives in the alias table (route_env.hh), not here.
//
// For TRSM that rule is not stylistic, it is the whole work package. The WP3
// spec's S10 proposed ONE predicate, `trsm_use_native()`, mixing the env read,
// the structural checks and a starvation threshold (`batch*q >= 8*CU*32`) plus
// a real-vs-complex speed judgement. Put either of the last two in supports()
// and route_resolve.hh:60-63 -- the vendor-off fallback, which re-walks the
// order testing ONLY `is_native(*r) && Table::supports(*r, s)` -- finds no route
// at all for every real-typed call and for everything below the starvation cut.
// The facade at src/dispatch/entry_points/level3.cc:165-167 then throws for
// shapes a correct native kernel could serve, which is the exact failure this
// work package exists to remove. A speed number in supports() does not make
// trsm slower on a vendor-free box; it makes trsm THROW.
//
// So: everything about registers, occupancy and traffic goes in preferred();
// the only things in supports() are the ones where the kernel would compute a
// WRONG ANSWER.
//
// THE ENV VARIABLE IS BATCHLAS_TRSM_ROUTE. parse_route_env (route_env.hh:214)
// builds the canonical name from op_env_stem(Op::trsm), and legacy_variable_for
// (route_env.hh:109-121) has no Op::trsm case -- so BATCHLAS_TRSM_VARIANT is
// read by nothing, and the spec's instruction to pin the native path with it
// would silently pin nothing. Values that reach this table: "cta" / "blocked"
// (route_env.hh:58-59; a bare algorithm implies Origin::Native), "native",
// "vendor". Unset means {Auto, Auto} (route_env.hh:145-148).
//
// FIELD MAPPING -- READ THIS BEFORE ADDING A PREDICATE.
// The spec's notation is `n` = the triangular order and `q` = the number of
// independent solves. OpShape's m/n/k do NOT spell it that way, and the
// convention used here is trmm's, so the coverage rows of the two triangular
// level-3 ops stay comparable (trmm_custom_dispatch.cc:186-189 passes
// C.rows(), C.cols(), A.rows()):
//
//     s.m  = B.rows()                   s.n  = B.cols()
//     s.k  = A.rows() == A.cols()       == the TRIANGULAR ORDER (spec's `n`)
//     q    = (side == Left) ? s.n : s.m == the INDEPENDENT EXTENT (spec's `q`)
//
// tri_order() and rhs_count() below are the only spellings this file uses, so a
// later predicate cannot pick the wrong one by writing `s.n` and meaning the
// order.
//
// STATUS: live. preferred() was all-false until WP3 step 9 measured the grid;
// it now moves trsm traffic to the native kernels on a vendor-present box for
// every cell the measurement backs. See the note on preferred() for which.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

// ---------------------------------------------------------------------------
// The largest triangular order the CTA kernel can hold in registers, per scalar
// type. DECLARED HERE, DEFINED IN THE KERNEL TU (src/sycl/trsm_native.cc, as
// four explicit specialisations) -- and deliberately NOT called from this file.
//
// WHY IT IS NOT A constexpr LITERAL HERE, unlike gesvd_jacobi_max_dim
// (route_gesvd.hh:64-71), which is the obvious precedent. gesvd's four numbers
// are derived from a measured local-memory limit; TRSM's are not measured yet.
// The spec's {float 64, double 32, cfloat 32, cdouble 16} come from a
// "256 B/thread register cliff" that WP3_TRSM_SPEC_CORRECTIONS.md reports as
// contradicted at gemm_kernels.cc:725-735 (an 8x8 double tile compiles to 208
// registers and complex<float> to 247, both spill-free). Transcribing them into
// a header would launder four hypotheses into a compile-time constant.
//
// WHY THE TABLE DOES NOT CALL IT. A header that calls it acquires a link
// dependency on a TU that does not exist yet, so this table could not land
// before the kernel does. Instead the SHAPE BUILDER -- in src/, next to the
// kernel -- calls it once and puts the answer in TrsmShape::cta_max_n, and the
// table reads only that field. The table therefore stays pure in the sense
// route_resolve.hh:19-20 requires ("reads only its arguments -- no getenv, no
// SYCL query") and stays linkable on its own.
// ---------------------------------------------------------------------------
template <typename T>
int trsm_cta_max_n();

// ---------------------------------------------------------------------------
// TRSM's routing reads one thing OpShape has no field for, so the op extends
// it, exactly as GesvdShape does (route_gesvd.hh:31-42). side/uplo/diag/transA
// are NOT among the extras: OpShape already carries all four (route.hh:230-234).
// ---------------------------------------------------------------------------
struct TrsmShape : OpShape {
    // trsm_cta_max_n<T>(), copied in by the shape builder. ZERO MEANS THE
    // NATIVE KERNEL IS ABSENT FROM THIS BUILD -- which is the state until WP3
    // step 2 lands -- and it correctly makes both native routes unsupported
    // rather than selectable-but-unimplemented.
    int cta_max_n = 0;

    // Whether the BLOCKED driver (V2) exists in this build. Separate from
    // cta_max_n because the two are independent capabilities: V2 is what serves
    // orders ABOVE cta_max_n, and until it is written those orders have no
    // native route at all.
    //
    // This is not belt-and-braces. Reporting Blocked as supported while it does
    // not exist makes resolve_route hand a vendor-free caller a route the facade
    // cannot service, and the call falls through to a NoRouteError whose message
    // says "no native kernel for it yet" -- true, but only after the resolver
    // claimed otherwise. Same class of defect as a kernel being LINKED but not
    // REACHABLE: the table must describe the build, not the design.
    bool blocked_available = false;

    int64_t tri_order() const { return k; }
    int64_t rhs_count() const { return side == Side::Left ? n : m; }
};

// CTA first, then the blocked driver that calls CTA as its diagonal solver,
// then the vendor. The order is a capability ladder, not a preference: CTA
// serves only order <= cta_max_n and blocked serves the rest. With preferred()
// all-false today the order matters only in the vendor-off walk at
// route_resolve.hh:60-63, where the tighter route is the right one to try first.
inline constexpr Route kTrsmOrder[] = {
    {Origin::Native, Algorithm::CTA},
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::trsm, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // Every gate below is "the kernel would compute a wrong answer", never
    // "the kernel would be slow". Nothing here is type-dependent: the whole
    // per-type difference is the register capacity, and that arrives as
    // s.cta_max_n.
    static bool supports(Route r, const TrsmShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. GPU ONLY. Not a speed judgement: the native path is a SYCL
        //    nd_range kernel with a work-group-collective staging phase and a
        //    local_accessor triangle. There is no host implementation of it to
        //    fall back on, so a CPU queue has to reach netlib.
        if (!s.is_gpu) return false;

        // 2. HETEROGENEOUS BATCH. One launch covers the whole batch with a
        //    single (order, q, ld, stride) tuple, so per-item extents would be
        //    read at the wrong addresses. Unlike gemm -- where WP2 C2 made this
        //    merely un-preferred, because the facade walks the batch into
        //    homogeneous members (route_gemm.hh:70-80) -- trsm has no such
        //    walker, so for trsm this IS a correctness gate. If one is ever
        //    written, this line moves to preferred(), and not before.
        if (s.heterogeneous_batch) return false;

        // 3. DEGENERATE EXTENTS. The canonical index map rho(s) = fwd ? s :
        //    order-1-s and the per-thread solve are undefined for an empty
        //    triangle or an empty solve set. Batch DISAGREEMENT between A and B
        //    is not tested here -- OpShape holds one batch, so the shape builder
        //    reports that by returning no shape at all (the gemm_op_shape
        //    pattern, src/backends/gemm_variant.hh:189-191).
        const int64_t order = s.tri_order();
        const int64_t q     = s.rhs_count();
        if (order < 1 || q < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::CTA:
                // The solution vector lives in the thread's registers as a
                // compile-time-sized array, so the largest order the kernel was
                // instantiated for is a hard capacity, not a tuning knob: above
                // it there is no kernel object to launch.
                if (s.cta_max_n < 1) return false;
                return order <= s.cta_max_n;

            case Algorithm::Blocked:
                // The blocked driver's diagonal-block solver IS the CTA kernel,
                // so it inherits the presence gate but not the cap -- it splits
                // the order into blocks of at most cta_max_n itself. It also
                // needs to exist, which is a separate question.
                return s.blocked_available && s.cta_max_n >= 1;

            default:
                // Including Algorithm::Auto. trsm has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:89-99).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // WP3 step 9 measured it. Every clause below cites the cells it comes from;
    // the raw CSVs are in experiments/wp3_s9/ and the grid that produced them
    // is benchmarks/trsm_benchmark.cc's TrsmOrthoSizes.
    //
    // THE GRID. Not the square-RHS one the old trsm_benchmark swept -- the
    // library never issues a square RHS. The two real call sites (ortho.cc:202
    // and :289) pass a k x k Cholesky factor as A and an m x k basis as B, so
    // the triangular order is SMALL and the other extent is LARGE. Measured
    // n in {8..256} x q in {256,1024,4096} x batch in {128,512,2048}, all four
    // types, both sides, RTX 4090, one card guarded exclusive, vendor and
    // native runs differing ONLY in BATCHLAS_TRSM_ROUTE. Ratios are
    // vendor_ms / native_ms, so >1 means native is faster, and are quoted only
    // where the (type, side, n, q) family had stopped scaling with batch.
    //
    // THE RESULT, worst cell per type over the whole saturated grid:
    //
    //   double            1.39x   (32/32 cells win, best 9.62x)
    //   complex<double>   1.20x   (30/30 cells win, best 4.66x)
    //   complex<float>    1.01x   (30/30 cells win, best 21.91x)
    //   float, Right      1.54x   (every n wins once batch >= 512)
    //   float, Left       1.21x   at EVERY order 8..512 and every size, since
    //                             WP3 step 16 routed the trailing-update GEMM
    //
    // After step 16, 167 of the 168 measured cells win. The one that does not
    // is float / Side::Right / order 512 / q=256 / batch=128, reproducibly
    // 0.978-0.983x over three repeats -- the smallest-work cell at that order
    // (1.0 ms total; its neighbours win 1.30-1.38x). A 2% deficit on one cell
    // is not worth a fitted special case here: the clause would be narrower
    // than the noise floor of most of this table.
    //
    // and end-to-end through the actual caller, ortho at m in {1024,4096},
    // k in {16..256}, batch in {128,512}, Chol2 and ShiftChol3: 80 of 80 cells
    // at or above parity, 1.15x to 2.69x. That check is not decoration. A
    // 2.16x kernel win in this repo once turned into an 11% gesvd loss, and a
    // predicate justified only at kernel level would not have caught it.
    static bool preferred(Route r, const TrsmShape& s) {
        // The vendor is never "preferred"; it is where the walk ends when
        // nothing native is. Saying so explicitly matters because the vendor
        // route is LAST in kTrsmOrder and returning true here would be
        // indistinguishable from falling through -- until someone reorders the
        // list.
        if (!is_native(r)) return false;

        const int64_t order = s.tri_order();

        // BATCH FLOOR. At batch = 1 the native kernel loses at every order at
        // or above 32 -- float 0.40-0.46x, double 0.80-0.86x -- because one
        // work-item solves one system and there is nothing else on the device.
        // At batch = 8 it already wins (double 1.08-2.93x, float n<=32
        // 1.09-2.44x), so the boundary sits at the first measured win rather
        // than at a rounder number. See experiments/wp3_s9/starved-*.csv.
        //
        // NOTE WHAT THIS IS NOT. Spec S10 proposed a starvation guard
        // `batch*q < 8*CU*32 -> vendor`, and the measurement REFUTES it: at
        // batch=8, q=32 that product is 256 against a threshold of 32,768, and
        // native wins those cells 2.2-2.4x. The guard would have handed back
        // every one of them. It is also unimplementable as written --
        // OpShape::compute_units still has no writer and reads 0 -- but it is
        // rejected here on the measurement, not on the plumbing.
        if (s.batch < 8) return false;

        if constexpr (std::is_same_v<T, float>) {
            // FLOAT IS THE ONLY TYPE WHERE THE VENDOR IS STILL COMPETITIVE,
            // and it splits by SIDE, which no other type does.
            if (s.side == Side::Left) {
                // WP3 step 12 BUILT the S3.4 staging tile, and this clause
                // moved from `order <= 16` to `order <= 128` because of it.
                // Worst cell per order, vendor/native, before -> after:
                //
                //   order      8      16      32      64     128     256
                //   before  1.61x   1.34x   0.70x   0.79x   0.71x   0.57x
                //   after   1.60x   1.73x   1.79x   1.49x   1.19x   0.76x
                //
                // The mechanism, measured with ncu rather than assumed: for
                // Side::Left the q independent solves run down B's COLUMNS, so
                // consecutive work-items read addresses ldb apart -- 31.4 load
                // sectors per request against a coalesced floor of 4. The tile
                // transposes through SLM and brings that to 5.13 on the load
                // and exactly 4.00 on the store, which is the whole 3.5x.
                //
                // WP3 step 16 REMOVED THE WORK THRESHOLD THAT USED TO LIVE
                // HERE. Side::Left is now preferred at every order and every
                // size, because the thing that lost the large cells was never
                // the triangular solve -- it was the trailing-update GEMM
                // taking the native kernel unconditionally.
                //
                // V2 called sycl_gemm::gemm_custom directly, bypassing
                // RouteTable<Op::gemm>. It now calls the ROUTED gemm (injected
                // by the facade; see TrsmTrailingGemm in src/sycl/trsm_native.hh),
                // so each trailing update goes wherever it is actually fastest.
                // Measured effect on the cells that used to lose:
                //
                //   order 256, q*batch 524288    0.92x -> 1.32x
                //   order 256, q*batch 2097152   0.87x -> 1.28x
                //   order 512, q*batch 524288    0.76x -> 1.28x
                //
                // and worst clean cell per order, Side::Left, after:
                //   8    16    32    64   128   256   512
                // 1.60  1.72  1.69  1.65  1.42  1.25  1.21
                //
                // WHY THE GEMM WAS THE PROBLEM, since it is not obvious: every
                // operand V2 hands gemm is a SUB-VIEW carrying its parent's
                // leading dimension -- a 128-row C with ld=512. On those shapes
                // with ld == rows the native kernel is at parity (0.86-0.98x);
                // with the real ld it measures 0.43-0.62x, while cuBLAS barely
                // moves. Strided is the only case trsm ever issues, so a
                // square-matrix GEMM benchmark could not have shown it.
                return true;
            }
            // Side::Right: native wins at every order once there is batch to
            // work with -- 1.54-4.59x at batch >= 512, all q. The two sub-unit
            // cells in the whole Right table are n=128 and n=256 at batch=128,
            // q=256 (0.97x, 1.02x), and the starved profile agrees that large
            // orders need batch: n=128 at batch 8-32 measures 0.74-0.81x.
            // Below batch 128, keep only the orders measured to win there.
            return s.batch >= 128 || order <= 32;
        } else {
            // double, complex<float>, complex<double>: NO CELL LOSES anywhere
            // in the grid, either side, any order, any q. The worst is
            // complex<float> at 1.01x and it climbs monotonically with order
            // from there; double's worst is 1.39x.
            //
            // Complex is the least surprising of the three. The incumbent
            // (cublas.cc:1122-1225) is a serial per-column substitution in
            // which every work-item re-reads the whole triangle from global
            // memory, so a CTA-resident kernel has no mechanism by which to
            // lose -- and at order 256, Side::Left it wins 21.9x.
            //
            // NO UPPER BOUND ON ORDER, deliberately. Above cta_max_n the
            // blocked driver takes over and supports() has already routed it;
            // its ratio at order 256 (double 1.45x, complex<float> 21.1x) is
            // measured, not extrapolated. The largest measured order is 256
            // because beyond it the grid does not fit in 24 GB, not because
            // anything changes in kind.
            return true;
        }
    }

    static constexpr const Route* order_begin() { return kTrsmOrder; }
    static constexpr const Route* order_end() {
        return kTrsmOrder + (sizeof(kTrsmOrder) / sizeof(kTrsmOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// `forced` is what the environment (or an explicit policy) asked for; pass a
// default-constructed Route for "no opinion". The unset default comes from
// legacy_unset_default(Op::trsm), which is {Auto, Auto} for every op since
// WP2 E6 (route_env.hh:145-148).
//
// `vendor_available` is the vendor-free switch, and for trsm the facade already
// knows it as a compile-time fact: dispatch::level3_vendor_available<Back>
// (src/dispatch/entry_points/level3.cc:165).
//
// Calling THIS -- rather than resolve_route_uninstrumented -- is also what gets
// trsm into the coverage table: resolve_route records every op that goes
// through it (route_resolve.hh:139-150), slicing TrsmShape to OpShape. No
// record_level3_route call is needed for trsm, and adding one would double-count.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_trsm_route(Route forced, const TrsmShape& s,
                                bool vendor_available = true) {
    return resolve_route<Op::trsm, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
