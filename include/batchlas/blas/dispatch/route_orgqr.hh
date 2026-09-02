#pragma once

// ORGQR's routing table (WP5 scaffolding -- the WP4 Phase 0 equivalent).
//
// Same three rules as route_geqrf.hh and route_potrf.hh:
//
//     supports()   == correctness only. Never a speed cutoff.
//     preferred()  == the measured window; false means slower, never ineligible.
//     the env read == lives in src/backends/orgqr_route.hh. This header is PURE.
//
// THE DESIGN DECISION, RECORDED EXPLICITLY BECAUSE THE TABLE ONLY MAKES SENSE
// WITH IT: orgqr ships as ORMQR APPLIED TO AN IDENTITY, which is what
// docs/design/vendor-independence.md's WP5 section suggests and what WP5's baseline
// experiment MEASURED rather than assumed
// (docs/perf/qr.md#the-vendor-baseline):
//
//   * CORRECT. Q from ormqr(F, I, Side::Left, Transpose::NoTrans) is elementwise
//     identical to cuSOLVER orgqr's Q to 6.9e-07..3.2e-06 (float/cfloat) and
//     1.4e-15..6.2e-15 (double/cdouble) across all 24 (type, n) cells, checked in
//     ONE process with independent orthonormality and QR-reconstruction probes on
//     both sides.
//   * ALREADY VENDOR-FREE. ormqr resolves to Native:Blocked and runs correctly
//     for all four scalar types in build-novendor/ -- measured with synthetic
//     reflectors, so the check does not itself need geqrf.
//   * FAST, for a reason that is NOT a statement about cuSOLVER's kernel. The
//     vendor orgqr IS NOT BATCHED: cublas.cc:1413-1420 opens an out-of-order
//     sub-queue and calls cusolverDnXorgqr ONCE PER BATCH ITEM, and
//     cublas.cc:1447 sizes its workspace as single_ws * batch (1164 MB for float
//     n=64 b8192, 4644 MB for cdouble n=64 b8192, against 104 MB / 416 MB for
//     routed ormqr-on-identity). rocsolver.cc:189-194 does the same, and calls
//     the PUBLIC orgqr<B> per item rather than orgqr_vendor<B>, so once orgqr has
//     route resolution that loop re-enters this facade per item. Measured ratios
//     (vendor build, cuSOLVER orgqr / ormqr-on-identity, saturating batch):
//     111.8x / 15.8x / 2.3x / 0.67x for float at n = 64 / 256 / 1024 / 2048, and
//     the one losing cell is partly a batch artefact (float n=2048: 0.67x at
//     b=32, 0.95x at b=64, 1.12x -- a WIN -- at b=128). A native orgqr closes a
//     MEMORY hazard as well as a speed gap. Report any such win as "beats the
//     per-item loop", never as "beats cuSOLVER", exactly as WP3's complex trsm
//     column had to be re-labelled.
//
// CONSEQUENCE FOR THIS FILE: kOrgqrOrder carries {Native, Blocked} and nothing
// else, and supports() below is RouteTable<Op::ormqr,T>::supports()' gates
// TRANSCRIBED, because that table is what will actually serve the call. Silently
// omitting an inherited gate is the wrong-answer class. Where a gate cannot fire
// under orgqr's fixed (Side::Left, Transpose::NoTrans) it is still written, with
// the reason it is inert, rather than dropped -- a dropped gate is invisible to
// the next person to widen the op.
//
// TWO DEPARTURES FROM route_geqrf.hh, both deliberate:
//
//   * NO has_sg32 FIELD. RouteTable<Op::ormqr,T>::supports (route_ormqr.hh:54-67)
//     has no sub-group gate, ormqr_blocked carries no
//     [[sycl::reqd_sub_group_size(32)]], and inventing one here would be a
//     DECORATIVE gate -- the state route_potrf.hh:83-96 criticises trsm for.
//     Add the field with the arm that needs it, not before.
//   * NO CTA ARM. There is no orgqr CTA kernel planned; the identity apply is
//     one ormqr call plus an identity fill.
//
// THE ENV VARIABLE IS BATCHLAS_ORGQR_ROUTE, synthesised by parse_route_env from
// op_env_stem(Op::orgqr) == "ORGQR" (route_env.hh:214-217). legacy_variable_for
// (route_env.hh:109-121) has NO Op::orgqr case and must not gain one -- no
// BATCHLAS_ORGQR_PROVIDER ever shipped. Note that Op::ormqr DOES have one
// (:118), so orgqr and the op it delegates to are pinned by DIFFERENT variables:
// BATCHLAS_ORGQR_ROUTE chooses whether orgqr is native at all, and the native arm
// then re-enters routed ormqr, which reads BATCHLAS_ORMQR_ROUTE /
// BATCHLAS_ORMQR_PROVIDER for itself. Both must be considered when pinning.
//
// FIELD MAPPING. s.m = A.rows(), s.n = A.cols(), s.k = min(m, n) == the number
// of reflectors consumed. A is the geqrf output in place, so its trailing n
// columns are what Q overwrites.
//
// STATUS: THE NATIVE DRIVER IS LINKED as of WP5.
// src/extensions/orgqr_blocked.cc reports orgqr_blocked_available<T>() == true
// for all four scalar types, so on a GPU the native arm is SUPPORTED.
//
// THAT MOVES NO VENDOR-PRESENT TRAFFIC. preferred() below is still false, so
// Origin::Auto keeps taking cuSOLVER's per-batch-item loop wherever a vendor
// exists. What changed is the vendor-free build: route_resolve.hh:60-63 now
// hands it the native arm instead of throwing NoRouteError, which is what turns
// orgqr_tests' GPU rows on the burn-down.

#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/route_resolve.hh>

namespace batchlas::dispatch {

struct OrgqrShape : OpShape {
    // Whether the native orgqr driver exists in this build. TRUE for all four
    // scalar types as of WP5.
    //
    // It is NOT "is ormqr_blocked compiled" -- that is true already and would
    // make this table advertise a route the facade cannot service. It is "is the
    // orgqr driver that arranges the identity and calls ormqr compiled", which
    // is the question route_trsm.hh:99-110 insists on: the table must describe
    // the BUILD, not the design.
    bool blocked_available = false;

    int64_t rows() const { return m; }
    int64_t cols() const { return n; }
    // Reflectors consumed, min(rows, cols). Named so a later predicate cannot
    // reach for m or n when it meant k -- the PotrfShape::order() discipline.
    int64_t reflectors() const { return k; }
};

// One native route, then the vendor. Same file-scope-array-with-sizeof-bounds
// form as kPotrfOrder and kGeqrfOrder, not a hand-counted std::array<Provider,6>
// (route_gemm.hh:43-46).
inline constexpr Route kOrgqrOrder[] = {
    {Origin::Native, Algorithm::Blocked},
    {Origin::Vendor, Algorithm::Auto},
};

template <typename T>
struct RouteTable<Op::orgqr, T> {
    // ---- CORRECTNESS ------------------------------------------------------
    static bool supports(Route r, const OrgqrShape& s) {
        if (is_vendor(r)) return true;   // the vendor serves everything it is given
        if (!is_native(r)) return false;

        // 1. GPU ONLY -- INHERITED VERBATIM from route_ormqr.hh:59. The driver
        //    is an identity fill plus a routed ormqr, and ormqr's own native arm
        //    refuses a non-GPU queue, so a CPU queue has to reach netlib. Note
        //    that this is also why the four ormqr/orgqr suites' Backend::NETLIB
        //    rows are NOT closable by WP5: test_utils::backend_types instantiates
        //    them against Device("cpu"), and a GPU-only native route cannot serve
        //    them. That is WP9's territory.
        if (!s.is_gpu) return false;

        // 2. COMPLEX WITH Transpose::Trans -- INHERITED FROM route_ormqr.hh:63-66,
        //    AND IT CANNOT FIRE TODAY. The identity apply is fixed at
        //    (Side::Left, Transpose::NoTrans), and orgqr_op_shape writes exactly
        //    that into s.side / s.transA, so this branch is unreachable for every
        //    shape the builder produces.
        //
        //    IT IS WRITTEN ANYWAY, because the alternative is to drop an
        //    inherited correctness gate silently and rely on a distant invariant
        //    (`the builder always sets NoTrans`) that nothing checks. WP5's
        //    baseline also gave that ormqr exclusion the reason route_ormqr.hh:64
        //    admits it never had: cuSOLVER rejects complex + Trans with
        //    "CUSOLVER error: 3", so native and vendor agree in refusing it. If a
        //    future orgqr grows a Q^H spelling, this line is already correct;
        //    deleting it would be the wrong-answer class.
        if constexpr (is_std_complex_v<T>) {
            if (s.transA == Transpose::Trans) return false;
        }

        // 3. AT MOST m ORTHONORMAL COLUMNS. Q's columns live in R^m (C^m), so a
        //    view asking for n > m columns of an orthonormal basis is asking for
        //    something that does not exist -- the driver would run off the end of
        //    the identity it fills and of the reflector set. A WRONG ANSWER / OOB
        //    gate, not a speed one.
        //
        //    Deliberately NOT also placed in orgqr_validate_params: turning a
        //    currently-tolerated shape into a user-visible throw is a behaviour
        //    change that belongs in its own commit with its own test
        //    (potrf.hh:59-65 states the rule). Here it only routes such a view to
        //    the vendor, which is what happens today.
        if (s.n > s.m) return false;

        // 4. HETEROGENEOUS BATCH. The driver fills one identity and issues one
        //    ormqr over the whole batch with a single (m, n, ld, stride) tuple,
        //    so a view with per-item active dims (matrix.hh:1034) would build the
        //    wrong Q for every item after the first. netlib's orgqr does not get
        //    this right either (netlib_lapack.cc:1449-1451 hoists m, n and k out
        //    of the loop, exactly as its geqrf does), so nothing in this tree
        //    serves it and no path disagrees with the gate.
        //
        //    THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh:83-96):
        //    orgqr_op_shape sets s.heterogeneous_batch in the same change. Note
        //    that RouteTable<Op::ormqr,T> has NO such gate and its builder
        //    (ormqr.hh:182-192) never sets the field -- so ormqr's own routing is
        //    blind to this today. Do not inherit that.
        if (s.heterogeneous_batch) return false;

        // 5. DEGENERATE EXTENTS.
        if (s.m < 1 || s.n < 1 || s.batch < 1) return false;

        switch (r.algo) {
            case Algorithm::Blocked:
                // The driver has to exist. TRUE as of WP5; route-neutrality now
                // comes from preferred() alone, which is the correct place for
                // it -- a capability is not a policy.
                //
                // NO LOWER OR UPPER BOUND ON THE EXTENTS: there is no second
                // native tier to be traded against, so any extent gate here
                // would be a speed cutoff wearing a correctness gate's clothes,
                // and route_potrf.hh:284-296 records what that costs a forced
                // route (route_resolve.hh:101 falls through to automatic(), i.e.
                // to the vendor, and the test passes green over it).
                return s.blocked_available;

            default:
                // Including Algorithm::Auto and Algorithm::CTA. orgqr has exactly
                // one native route; a bare "native" is resolved by
                // resolve_route's origin-restricted walk (route_resolve.hh:87-98)
                // rather than by accepting Auto here.
                //
                // NOTE THE DEPARTURE FROM route_ormqr.hh:57, which accepts
                // `Algorithm::Auto` as if it were Blocked. That makes {Native,
                // Auto} a route a dispatch tail has to interpret; potrf's shape
                // (route_potrf.hh:298-305) is the one copied here.
                return false;
        }
    }

    // ---- MEASURED WINDOW --------------------------------------------------
    // FALSE EVERYWHERE, DELIBERATELY. Same merge state, and same reason, as
    // route_geqrf.hh and route_potrf.hh: all-false means Origin::Auto takes the
    // vendor wherever one exists, so this table moves ZERO traffic, while
    // route_resolve.hh:60-63 still hands a VENDOR-FREE caller the native route
    // the day orgqr_blocked_available<T>() comes off false.
    //
    // DO NOT COPY route_ormqr.hh:77-79's `return is_native(r) && supports(r, s);`.
    // That spelling makes native the default on every supported shape with no
    // measured window at all, which is precisely the collapse of the two
    // predicates this split exists to prevent -- and for orgqr there IS a
    // measured losing cell to respect (cfloat n=2048 loses at every batch that
    // fits in 24 GB: 0.64x at nb=56, 0.59x at nb=32).
    //
    // WHAT GOES HERE WHEN THE GRID IS RE-RUN AGAINST A REAL DRIVER, with cell
    // citations, and nowhere else. Two measured facts to carry into it:
    //   * The ratios above are from the VENDOR build. Vendor-free ormqr is
    //     0.88-2.39x slower on the same Native:Blocked route, so the vendor-free
    //     margin at n >= 1024 is thinner or negative. The fix for that is the
    //     Tiled16 transposed gemm (WP2 territory), not orgqr.
    //   * Applying Q to an identity does 1.5x the nominal flops of a specialised
    //     orgqr at m = n = k (2n^3 against 4n^3/3). That is the entire
    //     theoretical prize of specialising, against a 2.3-111x margin over the
    //     vendor across most of the range -- so specialise only if a cell
    //     measures it necessary.
    static bool preferred(Route r, const OrgqrShape& s) {
        static_cast<void>(r);
        static_cast<void>(s);
        return false;
    }

    static constexpr const Route* order_begin() { return kOrgqrOrder; }
    static constexpr const Route* order_end() {
        return kOrgqrOrder + (sizeof(kOrgqrOrder) / sizeof(kOrgqrOrder[0]));
    }
};

// ---------------------------------------------------------------------------
// Resolution for one call. Pure.
//
// `vendor_available` is PASSED EXPLICITLY by the facade
// (dispatch::factorization_vendor_available<B>), never left to the default.
// route_potrf.hh:361-366 warns about the two sites that do leave it: syev.hh:948
// and ormqr.hh:209. ormqr's omission is the one that matters here, because orgqr
// delegates to it -- ormqr never reaches route_resolve.hh:60-63's vendor-free
// fallback at all, and only gets away with it because its preferred() returns
// native-first. Do not inherit that.
//
// Calling resolve_route rather than resolve_route_uninstrumented is what puts
// orgqr in the coverage table (route_resolve.hh:130-152), slicing OrgqrShape to
// OpShape.
// ---------------------------------------------------------------------------
template <typename T>
inline Route resolve_orgqr_route(Route forced, const OrgqrShape& s,
                                 bool vendor_available = true) {
    return resolve_route<Op::orgqr, T>(forced, s, vendor_available);
}

} // namespace batchlas::dispatch
