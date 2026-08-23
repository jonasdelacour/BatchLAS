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
//     preferred()  == the measured window; all-false at merge.
//     the env read == src/backends/getrs_route.hh, not here. PURE header.
//
// ONE NATIVE ARM, NOT TWO, AND THAT IS THE SHAPE OF THE OP. getrs against a
// factored A is a row permutation plus two triangular solves; there is no
// second algorithm to choose between, so the order array carries a single
// {Native, Blocked} the way route_orgqr.hh:108-114 does. `Blocked` names "a
// host-driven composition over routed BLAS-3", which is what orgqr's arm is too
// (ormqr on an identity) -- not a claim that this op has a panel schedule.
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
// STATUS: NO NATIVE DRIVER IS LINKED. src/extensions/getrs_native.cc returns
// false from getrs_blocked_available<T>() for every type, so the native arm is
// unsupported for EVERY shape and resolve_getrs_route always returns
// {Vendor, Auto}. Merging this table moves ZERO decisions.
//
// THE MEASUREMENT THAT MUST BE ON RECORD BEFORE ANY KERNEL IS WRITTEN, because
// it argues against the obvious implementation. Composed "laswp + two routed
// trsm" against cublas?getrsBatched, at saturating batch, in process, against a
// host oracle (experiments/wp6_lu/baseline/):
//
//   nrhs = 1  : GEOMEAN 0.36x over 28 cells, 25 LOSSES, worst 0.09x (cdouble
//               n=32). Only n=2048 wins (1.07-1.15x) and that is against an
//               UNSATURATED vendor.
//   nrhs = 64 : geomean 1.17x (interchange list) / 1.55x (collapsed to a gather),
//               20 and 25 wins of 28.
//
// The nrhs=1 loss is STRUCTURAL, not a bad kernel: trsm's blocked driver
// amortises a panel over many columns and one column gives it nothing to
// amortise; the permutation is a rounding error there (the gather strategy
// changes the geomean by 0.00x). So a native getrs needs a separate narrow-RHS
// path, or it ships route-neutral at small nrhs. nrhs is s.n and IS available to
// preferred() -- that is why the field mapping puts it there rather than folding
// it into max_dim. inv.cc, the only internal consumer of the LU family, does not
// call getrs at all.
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
    // IT IS CARRIED EVEN THOUGH THE ARM IS A COMPOSITION, because the row
    // permutation is a kernel of this op's own -- a laswp walking the interchange
    // list, or the gather the baseline measured -- and its shape is chosen with
    // the same sub-group assumptions as the rest of this family. Note the
    // contrast with route_orgqr.hh:47-53, which deliberately has NO such field
    // because ormqr_blocked carries no reqd_sub_group_size: a gate whose kernel
    // does not need it is DECORATIVE, which is the state route_potrf.hh:83-96
    // criticises trsm for. If the permutation kernel ships without a required
    // sub-group size, DELETE this field rather than leaving it to read as live.
    bool has_sg32 = false;

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
inline constexpr Route kGetrsOrder[] = {
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

        // 1. THE DRIVER MUST EXIST IN THIS BUILD. TRUE for all four scalar
        //    types now (src/extensions/getrs_native.cc); the gate stays because an
        //    absent capability must never select a launch that is not there.
        if (!s.blocked_available) return false;

        // 2. GPU ONLY -- INHERITED from route_trsm.hh:138-142 and true of the
        //    permutation kernel in its own right. There is no host
        //    implementation of either to fall back on, so a CPU queue has to
        //    reach netlib.
        if (!s.is_gpu) return false;

        // 3. SUB-GROUP SIZE 32, for the permutation kernel. See
        //    GetrsShape::has_sg32 -- and delete both if the kernel that lands
        //    carries no reqd_sub_group_size.
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

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, AND NOW A DELIBERATE HOLD: the kernel exists and is
    // measured. All-false means Origin::Auto takes the vendor wherever one exists
    // (route_resolve.hh:110-112 finds nothing preferred, :129 returns
    // {Vendor, Auto}), so this table still moves ZERO vendor-present traffic,
    // while route_resolve.hh:113-127 hands a vendor-free caller the native arm --
    // which it now does for every shape.
    //
    // WHAT WILL GO HERE, and it is unusually well determined already: a window on
    // s.nrhs(), NOT on s.order(). The composed arm is 0.36x geomean at nrhs=1
    // (25 losses of 28, worst 0.09x) and 1.17-1.55x at nrhs=64 (20-25 wins of
    // 28) -- so this is the one op in WP6 whose preferred() is expected to be
    // false over a whole, common regime rather than below a size threshold.
    // Shipping route-neutral at small nrhs is a legitimate outcome under the
    // campaign's gate; engineering around the number is not.
    static bool preferred(Route r, const GetrsShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    // NO native_tier_preferred. One native arm means there is no native-vs-native
    // question to answer, and the hook is detected by a requires-expression
    // defaulting to `true` (route_resolve.hh:76-83) -- so a single-tier table is
    // neutral by construction, exactly as route_orgqr.hh is. Declaring it would
    // be decorative.

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
