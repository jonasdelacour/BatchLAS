#pragma once

// GETRI's routing table (WP6 scaffolding).
//
// Same three rules as route_getrf.hh / route_getrs.hh / route_geqrf.hh:
// supports() is correctness only, preferred() is the measured window and is
// all-false at merge, and the env read lives in src/backends/getri_route.hh so
// this header stays PURE (route_resolve.hh:19-21).
//
// ONE NATIVE ARM, and it is a COMPOSITION over the routed trsm -- the
// `orgqr = ormqr on an identity` precedent (route_orgqr.hh:41-49). getri against
// a factored A is: write the permutation P into C, then solve L and U against
// it. The order array carries a single {Native, Blocked}; `Blocked` names "a
// host-driven composition over routed BLAS-3", not a panel schedule.
//
// CONSEQUENCE FOR THIS FILE, AND IT IS THE ONE route_orgqr.hh:41-49 SPELLS OUT:
// supports() below must TRANSCRIBE the gates of the table that will actually
// serve the call. Silently omitting an inherited gate is the wrong-answer class.
// Where an inherited gate cannot fire under getri's fixed arguments it is still
// written, with the reason it is inert, rather than dropped -- a dropped gate is
// invisible to the next person to widen the op.
//
// FIELD MAPPING -- potrf's, because getri's operands are square:
//
//     s.m = s.n = s.k = THE ORDER
//
// options.hh:687-690 requires A square, C square, same rows, same batch; the
// vendor arms read only A.rows() (cublas.cc:1538). No transpose, no uplo, no
// side: getri sets NONE of variant_key's fields (coverage.cc:52-58), so its
// coverage rows collapse to shape_class alone, first-writer-wins. Same
// route_diff blindness as getrf; see the warning in route_getrf.hh.
//
// STATUS: NO NATIVE DRIVER IS LINKED. src/extensions/getri_blocked.cc returns
// false from getri_blocked_available<T>() for every type, so the native arm is
// unsupported for EVERY shape and resolve_getri_route always returns
// {Vendor, Auto}. Merging this table moves ZERO decisions.
//
// TWO CONTRACT FACTS A NATIVE getri MUST HONOUR, both measured, neither
// expressible as a predicate:
//
//   (a) A IS NOT WRITTEN. cuBLAS's prototype takes `const T* const A[]`
//       (cublas_api.h:5568-5576) and measured max|A_after - A_factored| == 0 for
//       all four types. The other two backends synthesise the same out-of-place
//       contract with a copy -- rocsolver.cc:330-332 memcpy's A into C then
//       inverts C in place (silently assuming C.stride() == A.stride() and a
//       contiguous batch), netlib_lapack.cc:1362-1365 does a host std::copy of
//       n*n per item and IGNORES ld. A native arm may pick either mechanism but
//       must not overwrite A.
//
//   (b) info IS EXACT-ZERO SEMANTICS, NOT A TOLERANCE. Measured on a matrix whose
//       second elimination cancels to a true binary zero: device info == 2 and
//       host LAPACK info == 2, 1-based, per item, identical. On a float matrix
//       with a duplicated column the device produced U(3,3) = -1.375e-08 and
//       reported info = 0 while the host got a true 0.0 and reported 3. A native
//       kernel that flags |pivot| < eps reports non-zero where the vendor reports
//       zero -- a contract divergence invisible to any native-vs-native test.
//       Both implementations also CONTINUE past a zero pivot and leave the rest
//       finite; a native kernel that divides unconditionally produces Inf/NaN
//       where the vendor gives finite garbage. And unlike potrf
//       (potrf_blocked.cc:436-440) LAPACK's LU does NOT quench a failed item, so
//       getri must keep info-only semantics.
//
// THE MEASURED CASE FOR THE COMPOSITION, and its crossover -- this is the one LU
// op where the native side has a clear win to go and get. Composed
// "P into C, then two routed trsm" against cublas<t>getriBatched, saturating
// batch, in process, host oracle (experiments/wp6_lu/baseline/):
//
//     n(batch)    float   double  cfloat  cdouble
//      32(8192)    0.54    0.23    0.23    0.23
//      64(8192)    0.83    0.53    0.35    0.54
//     128(4096)    1.32    0.90    1.06    0.89
//     256(2048)    3.89    1.16    2.05    1.04
//     512(512)     5.75    1.28    3.01    1.02
//    1024(128)    15.66    1.16    6.05    1.11
//    2048(32)     74.87    3.93   25.88    4.30
//
// Geomean 1.60x over the 28 cells, 18 wins. Crossover n ~ 128 for float/cfloat,
// n ~ 256 for double/cdouble; BELOW it cuBLAS's small-n getriBatched path wins by
// up to 4.3x. EVERY n >= 512 NUMBER IS AGAINST AN UNSATURATED VENDOR and must not
// be quoted alone -- getri float n=256 is best at batch 256 (13.85 us/item) and
// DEGRADES to 20.38 at batch 2048, so the grid's own schedule is 1.47x pessimistic
// to the vendor at that cell. That is stated rather than silently corrected.
//
// AND THE SINGLE BIGGEST LEVER IS FREE FOR THIS OP. A LAPACK-faithful laswp is
// 49-51% of the composed call at n=128 (BREAK=laswp: 0.4580 -> 0.2251 ms) and is
// structurally slow -- the interchange list is sequential in k so it cannot be
// parallelised, and in column-major consecutive work-items land ldb apart, i.e.
// 32 transactions per warp access. It CAN be collapsed: apply the interchanges to
// an identity index array once, then GATHER, which puts consecutive work-items on
// consecutive rows. getri gets that for nothing -- write P straight into C instead
// of writing I and permuting it: same store count, one kernel, zero workspace.
// That is what turns the geomean from 0.97x into 1.60x above. It belongs in the
// kernel step, recorded here so it is not re-derived.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

#include <cstdint>

namespace batchlas::dispatch {

struct GetriShape : OpShape {
    // Whether the native getri driver exists in this build. FALSE today.
    //
    // NOT "is trsm compiled" -- that is true already and would make this table
    // advertise a route the facade cannot service. It is "is the driver that
    // writes P into C and issues the solves compiled" (route_trsm.hh:99-110: the
    // table must describe the BUILD, not the design).
    bool blocked_available = false;

    // ENUMERATED from sub_group_sizes; see GetrfShape::has_sg32. Carried for the
    // permutation kernel, exactly as in GetrsShape -- and, exactly as there, to
    // be DELETED rather than left decorative if the kernel that lands carries no
    // reqd_sub_group_size.
    bool has_sg32 = false;

    int64_t order() const { return k; }
};

// One native route, then the vendor. Same file-scope-array-with-sizeof-bounds
// form as kOrgqrOrder and kGetrsOrder (route_gemm.hh:43-46).
inline constexpr Route kGetriOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::getri, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    // The structural gates are RouteTable<Op::trsm,T>::supports()'
    // (route_trsm.hh:132-160) TRANSCRIBED plus getri's own, for
    // route_orgqr.hh:41-49's reason: the routed trsm is what serves the two
    // solves, and an inherited gate omitted here is a wrong answer, not a slow
    // one.
    //
    // NOT TRANSCRIBED, DELIBERATELY: trsm's CAPACITY gates. They choose between
    // trsm's own tiers and impose no ceiling on the order (trsm's blocked driver
    // serves everything above the CTA cap), so there is no order this table could
    // refuse on their account. A transcribed gate that cannot fire reads as live
    // and is worse than no gate at all.
    static bool supports(Route r, const GetriShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. THE DRIVER MUST EXIST IN THIS BUILD. TRUE for all four scalar
        //    types now (src/extensions/getri_blocked.cc); the gate stays because
        //    a build that omits that TU must not select a launch that is absent.
        if (!s.blocked_available) return false;

        // 2. SQUARE ONLY. getri's own, and the same line as getrf's: the public
        //    API requires it (options.hh:687-690 checks A, C and their agreement)
        //    and every vendor arm reads only A.rows(). Not a native restriction.
        if (s.m != s.n) return false;

        // 3. GPU ONLY -- INHERITED from route_trsm.hh:138-142 and true of the
        //    permutation kernel in its own right. A CPU queue has to reach netlib.
        if (!s.is_gpu) return false;

        // 4. SUB-GROUP SIZE 32, for the permutation kernel. See
        //    GetriShape::has_sg32.
        if (!s.has_sg32) return false;

        // 5. HETEROGENEOUS BATCH -- INHERITED from route_trsm.hh:151-154, and
        //    getri's own besides: the pivot list is read at pivots[b*order + k]
        //    with a single order, so per-item extents scatter it.
        if (s.heterogeneous_batch) return false;

        // 6. DEGENERATE EXTENTS. The solve loops and the interchange walk are
        //    undefined for an empty matrix or an empty batch. Disagreement
        //    BETWEEN A and C is not testable here (OpShape holds ONE shape); the
        //    builder reports it by returning no shape at all.
        //
        //    NO LOWER BOUND BEYOND 1, AND NOT A BATCH FLOOR EITHER. inverse_tests
        //    -- the one suite WP6 can close outright -- is a single float case at
        //    n=40, BATCH=2 (tests/inverse_tests.cc:10-39). It closes if and only
        //    if a native getrf and getri serve batch 2, so any batch floor here
        //    keeps it red however good the kernel is. The acceptance MEASUREMENT
        //    is still done at batch >= 128 per standing policy; that is
        //    preferred()'s business, not this function's.
        if (s.order() < 1 || s.batch < 1) return false;

        // 7. THE PIVOT FORMAT MUST AGREE WITH WHATEVER ELSE TOUCHES THE SPAN.
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
        //    the field -- src/backends/getri_route.hh sets `s.backend = B` -- so the
        //    gate is one predicate, and it is enumerated by the backend whose
        //    format disagrees rather than by an allow-list, so a new GPU backend
        //    that packs int32 like the other two needs no edit here.
        if (s.backend == Backend::NETLIB) return false;

        switch (r.algo) {
            case Algorithm::Blocked:
                return true;
            default:
                // Including Algorithm::Auto: resolve_route walks the order
                // restricted to the requested origin (route_resolve.hh:146-163)
                // and expects a SPECIFIC algorithm back, so {Native, Auto} must
                // not be reported supported.
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, AND NOW A DELIBERATE HOLD: the kernel exists and is
    // measured (28 cells, geomean 1.463x at the grid's batch but 1.284x when each
    // arm is read at its OWN best batch). All-false means Origin::Auto takes the
    // vendor wherever one exists, while route_resolve.hh:113-127 hands a
    // vendor-free caller the native arm -- which it now does for every shape.
    //
    // =====================================================================
    // WP8 ROUTING PASS: THE WINDOW LANDS, AND IT IS NOT THE ONE THE NOTE ABOVE
    // PREDICTED. Two of its four types are wrong and the saturation caveat is
    // wrong in both directions.
    //
    // THE MEASUREMENT. experiments/wp8_getri/lu_c1.csv and lu_p1.csv, produced by
    // pair_cells.sh: the two arms are two BUILDS (lubench6_nv has no cuBLAS
    // linked, so the BINARY selects the arm and no pin can silently fall
    // through), run BACK TO BACK ON EACH CELL so clock and thermal state drift
    // together, 11 reps, median, WARM_S = 1.0, host LAPACKE oracle in process on
    // every timed row, resolved route checked against the arm on every row, the
    // number of FOREIGN compute processes sampled before and after every cell
    // and required to be 0. TWO INDEPENDENT PASSES; every ratio below is the
    // WORSE of the two.
    //
    // A THIRD PASS, lu_p2.csv, EXISTS IN THE TREE AND WAS DISCARDED. Say so
    // rather than leave a reader to find it: quoting the worse of lu_p1/lu_p2
    // would put getri cfloat n=256 batch=512 at 0.7391 -- an outright IN-CLAUSE
    // LOSS -- and n=256 batch=4096 at 1.0023, below the bar. Neither survives
    // re-measurement: that first cell reads 1.6794 and 1.6879 on two fresh
    // passes taken on an idle box, against the 1.682 printed below.
    // lu_p2 is contaminated, not merely noisy: 26 of its 55 comparable cells are
    // 1.2x-5.8x slower than lu_c1 on BOTH arms at once, with foreign == 0 and
    // rel_sd as low as 0.0012, and its runner (queue_p2.sh) gated on
    // `pgrep -f gemvab_v` and then ran on the DISPLAY GPU while a gemv sweep held
    // the other card. See the cross-device note below: on this box the two 4090s
    // share a NUMA node, a CPU affinity mask and one UVM driver, so a per-device
    // foreign-process count reports 0 and is CORRECT while the number is still
    // wrong by up to 5x. Discarding a whole pass is legitimate; discarding one
    // silently, and leaving the header pointing at the file that contains the
    // loss, is not.
    //
    // ON DEVICE 1, NOT DEVICE 0, AND THAT IS A CORRECTION RATHER THAN A
    // PREFERENCE. Device 0 on this box drives the display (Xorg, gnome-shell,
    // firefox: 1.1 GB resident and 6 graphics contexts), which WP8-I3 measured
    // as depressing a VENDOR arm by up to 1.8x on L2-resident cells while
    // leaving a DRAM-streaming kernel untouched. getri at n = 256, batch = 128,
    // float is a 33 MB working set -- inside this device's 72 MB L2 -- so the
    // small-order cells are exactly the ones that reading would corrupt.
    //
    // ---- THE CLAUSE -------------------------------------------------------
    //     float,  order >= 128     cfloat, order >= 256
    // AND NO BATCH TERM, which is the finding that most changed the answer.
    //
    // THE SHIPPED SET: 30 cells, GEOMEAN 4.181, MIN 1.2028, ZERO losses, ZERO
    // cells below 1.15. Cross-pass agreement on the whole getri population is
    // 0.5% or better on every cell. Ratio = vendor_ms / native_ms, quoted at the
    // WORSE pass, batch across five decades:
    //   float  n=128   b1     2.169   b2     2.163   b32    2.089   b256  1.819
    //                  b1024  1.556   b4096  1.262   b16384 1.203
    //   float  n=256   b128   4.047   b4096  2.641
    //   float  n=512   b1    18.565   b32   15.103   b128   7.600   b1024 3.991
    //                  b2048  4.466
    //   float  n=1024  b128   9.033   b512   5.068   b1024  5.910
    //   float  n=2048  b128   9.331   b256   7.095
    //   cfloat n=256   b128   1.977   b512   1.682   b2048  2.050   b4096 2.195
    //   cfloat n=512   b1    24.276   b32    8.861   b128   4.138   b1024 3.807
    //   cfloat n=1024  b128   5.970   b512   5.475
    //   cfloat n=2048  b128   8.600
    //
    // ---- WHAT REFUTES EVERY WIDER CLAUSE, WITH ITS CELL AND BOTH PASSES ----
    //   float order >= 64    n=64 LOSES: 0.8572 / 0.8561 at batch 8192 and
    //                        0.8499 / 0.8530 at batch 16384. The boundary is
    //                        bracketed from below by measured losses, not by the
    //                        edge of the grid.
    //   cfloat order >= 128  n=128 is 0.7109 / 0.7070 at batch 512 -- an
    //                        OUTRIGHT LOSS in the middle of its own ladder
    //                        (2.63 at batch 1..4, 1.92 at 32, 0.968 at 2048).
    //                        The old note's "crossover n ~ 128 for float/cfloat"
    //                        read that cell at ONE batch and at no other.
    //   double, ANY order    REFUTED, and the refuting cell is the same for every
    //                        threshold: n=2048 batch=128 is 1.0848 / 1.0823. So
    //                        `order >= 512`, `>= 1024` and `>= 256` all fail on
    //                        it. Below n=2048 the picture is a near-miss rather
    //                        than a loss -- n=1024 is 1.1602 / 1.1551 at batch
    //                        128 / 512, above the bar by 0.5% and FALLING with
    //                        the last rung measured being the lowest, and n=256
    //                        is 1.124-1.129, below the bar without losing. Only
    //                        n=512 is clean (1.297-1.312 over four batches), and
    //                        a clause admitting exactly one order is the leg
    //                        predicate this campaign has already found twice.
    //   cdouble, ANY order   n=512 LOSES at 0.9542 / 0.9525 (batch 1024) and is
    //                        1.045 at batch 128; n=1024 tops out at 1.135.
    //                        Nothing at or above 1.15 anywhere at batch >= 128.
    //
    // ---- THE BATCH AXIS: WHY THERE IS NO FLOOR, AND WHAT IS LEFT UNROUTED --
    // D4 predicted a batch floor of 128 on the ground that batch 1-3 was
    // measured NOWHERE in the campaign and that inverse_tests and
    // getrf_tests both run getri at batch 2. It is measured now, and it is the
    // OPPOSITE of a risk: at batch 1..32 the native driver beats cuBLAS by
    // 1.7x-28x for EVERY type at EVERY order measured, because cuBLAS's batched
    // getri is a per-item loop there. float n=512 is 18.6x at batch 1; cfloat
    // n=512 is 24.3x; double n=512 is 27.9x; cdouble n=128 is 3.99x. A floor
    // would have given those away for nothing.
    //
    // SO THE HONEST NEGATIVE IS THE OTHER HALF: the low-batch regime is a large
    // measured win for double AND cdouble too, and it is NOT routed here.
    // Routing it needs a clause naming batch, and a batch clause has to be
    // bracketed at every (type, order) it admits -- this pass measured batch <= 32
    // only at orders 128 and 512, so four of the six orders and both wide types
    // have no ladder to bracket it with. Naming the missing cells rather than
    // fitting the boundary: double and cdouble at orders 32, 64, 256, 1024 and
    // 2048, batch 1..64.
    //
    // ---- THE SATURATION CAVEAT, CORRECTED --------------------------------
    // The note above says "every n >= 512 ratio is against an unsaturated
    // vendor". It is too broad at n = 512, where the vendor IS saturated for all
    // four types (its us/item moves under 0.2% over the last doubling), and too
    // narrow at n = 2048, where it is unsaturated for all four by 19-50% per
    // doubling. That is why this clause's n = 2048 entries are quoted at batch
    // 128 and 256 -- the largest batches that fit -- and never at the batch-32
    // grid schedule, where float n = 2048 reads 33.9x and means nothing.
    //
    // ---- NOT A CAPACITY TERM ---------------------------------------------
    // No order ceiling appears here. supports() carries none either (the routed
    // trsm's blocked tier has no upper bound), and repeating a capability test
    // in preferred() is the defect route_resolve.hh:60-70 records.
    // =====================================================================
    static bool preferred(Route r, const GetriShape& s) {
        if (!is_native(r)) return false;
        if (r.algo != Algorithm::Blocked) return false;   // the only native arm

        if constexpr (std::is_same_v<T, float>)               return s.order() >= 128;
        if constexpr (std::is_same_v<T, std::complex<float>>) return s.order() >= 256;
        return false;   // double and cdouble earn nothing; see the cells above
    }

    // NO native_tier_preferred: one native arm, no native-vs-native question. The
    // hook defaults to `true` through a requires-expression
    // (route_resolve.hh:76-83), so its absence is neutral by construction.

    static constexpr const Route* order_begin() { return kGetriOrder; }
    static constexpr const Route* order_end() {
        return kGetriOrder + (sizeof(kGetriOrder) / sizeof(kGetriOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure. See resolve_getrf_route's note for why
// `vendor_available` is passed EXPLICITLY at both call sites, and why this --
// rather than resolve_route_uninstrumented -- is what puts getri in the coverage
// table.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_getri_route(Route forced, const GetriShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::getri, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
