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
// STATUS: table only. preferred() is all-false by construction -- see the note
// on it -- so this file changes no routing decision on a vendor-present box.

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
                // the order into blocks of at most cta_max_n itself.
                return s.cta_max_n >= 1;

            default:
                // Including Algorithm::Auto. trsm has two native routes, so a
                // bare "native" names neither; resolve_route walks the order
                // restricted to the requested origin to pick one
                // (route_resolve.hh:89-99).
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE FOR EVERY CELL, ON PURPOSE, AND NOT AN OVERSIGHT.
    //
    // Nothing about the native TRSM has been measured yet. Returning true
    // anywhere here would MOVE LIVE TRAFFIC off the vendor on the strength of
    // the spec's S10 table, whose own confidence column reads "low" for three
    // of its five rows and whose two numeric constants -- the starvation cut
    // `batch*q >= 8*CU*32` and "complex flips native" -- are both recorded as
    // hypotheses in WP3_TRSM_SPEC_CORRECTIONS.md.
    //
    // The starvation guard in particular COULD NOT BE WRITTEN YET even if it
    // were trusted: it needs the SM count, and OpShape::compute_units
    // (route.hh:240) has zero writers in include/, src/ and tests/ today -- it
    // reads 0, so any predicate comparing against it is comparing against zero.
    // The shape builder is what has to populate it; the table may not, because
    // a SYCL query in here breaks the purity route_resolve.hh:19-20 depends on.
    //
    // Consequence, and it is the one WP3 step 6 wants: with a vendor present
    // every trsm call keeps going to the vendor, so every existing test stays
    // green unchanged; with no vendor present route_resolve.hh:60-63 still finds
    // the supported native route, which is the gap being closed. On a
    // vendor-present box the native path is reachable only by naming it --
    // BATCHLAS_TRSM_ROUTE=cta / =blocked / =native -- which is a FORCED request
    // and so never consults this function (route_resolve.hh:89-101).
    //
    // Cells get flipped here one at a time, each with its measurement quoted in
    // place, in the style of route_gemm.hh:124-206. Do not flip one without one.
    static bool preferred(Route, const TrsmShape&) { return false; }

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
