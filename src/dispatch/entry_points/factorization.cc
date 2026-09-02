// The public factorization entry points, defined once, outside every vendor TU.
//
// Same move, and same reason, as entry_points/level3.cc: geqrf/orgqr/getrf/
// getrs/getri were DEFINED in cublas.cc, netlib_lapack.cc and rocsolver.cc, and
// potrf in cusolver.cc, netlib_lapack.cc and rocsolver.cc -- so dropping a
// vendor library dropped the public entry point along with the vendor path.
//
// Each op moves TOGETHER WITH ITS BUFFER-SIZE QUERY. Splitting them would let
// the two resolve differently, which is the defect class S4d found in ormqr
// (buffer size 2560 bytes, call demanded 276480).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/potrf.hh>

// The two operations the BLOCKED potrf driver INJECTS. It is instantiated per
// scalar type with no Backend parameter (potrf_cta.cc:706-726), so it cannot
// name a routed entry point itself; the facade is the only layer that can. Same
// include, for the same reason, as level3.cc:30/:51 for trsm's trailing gemm.
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// The POTRF shape builder + route resolution. Placed here for the same reason
// level3.cc:34 places "../../backends/trsm_route.hh": the routing adapter is a
// src/ header of public includes only, so the facade can include it in a
// vendor-free build.
#include "../../backends/potrf_route.hh"
#include "../../extensions/potrf_native.hh"

// WP5: the same pair for geqrf and orgqr. Both adapters are src/ headers of
// public includes only (plus one private kernel header each), so the facade can
// include them in a vendor-free build.
#include "../../backends/geqrf_route.hh"
#include "../../backends/orgqr_route.hh"
#include "../../extensions/geqrf_native.hh"
#include "../../extensions/orgqr_native.hh"

// WP6: the same pair for the LU family. All three adapters are src/ headers of
// public includes only (plus one private kernel header each), so the facade can
// include them in a vendor-free build -- the gemm_variant.hh:1-9 rule.
//
// getrf's trailing update and panel solve, and getrs's / getri's triangular
// solves, all reuse the gemm.hh / trsm.hh includes already present above for
// potrf's seams. Nothing new is needed for them.
#include "../../backends/getrf_route.hh"
#include "../../backends/getrs_route.hh"
#include "../../backends/getri_route.hh"
#include "../../extensions/getrf_native.hh"
#include "../../extensions/getrs_native.hh"
#include "../../extensions/getri_native.hh"

// orgqr's native arm is ormqr applied to an identity, and the apply must go
// through the ROUTER. Only this layer can name the routed entry point: the driver
// is instantiated per scalar type with no Backend parameter, and ormqr<B,T> needs
// one. Same include, for the same reason, as gemm.hh/trsm.hh above for potrf's
// injected seams.
#include <batchlas/blas/functions/ormqr.hh>

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

// ---------------------------------------------------------------------------
// WP5 -- the QR pair. Both are hooked exactly as potrf is (further down this
// file), and the three "native route with no kernel" diagnostics are written the
// same way; potrf's is at its own definition below because it names potrf's
// capability functions.
//
// A NOTE ON WHICH vendor_available THESE USE, BECAUSE IT IS A REAL FORK AND
// BECAUSE THERE IS A LATENT DEFECT UNDER IT.
//
// potrf passes dispatch::solver_vendor_available<B> (cuSOLVER). geqrf and orgqr
// pass dispatch::factorization_vendor_available<B> (cuBLAS), which is what the
// bodies below them have always tested and what their explicit instantiations are
// keyed on. The two DIFFER on CUDA (vendor_available.hh:41-45 vs :47-52), so this
// is not interchangeable.
//
// THE LATENT DEFECT, RECORDED RATHER THAN PAPERED OVER: vendor_available.hh:11-13
// asserts that "on NVIDIA, geqrf/getrf/ormqr and friends come from cublas.cc
// while potrf and syev come from cusolver.cc". That is true of the FILE and false
// of the SYMBOLS -- cublas.cc:1240-1253 calls cusolverDnXgeqrf and
// cublas.cc:1394-1411 calls cusolverDn{S,D}orgqr / {C,Z}ungqr, both from inside a
// TU gated on BATCHLAS_HAS_CUBLAS (src/backends/CMakeLists.txt:68-70). A
// cuBLAS-present / cuSOLVER-absent configure therefore claims a vendor it cannot
// link. Switching these two calls to solver_vendor_available would paper over it
// AND change which builds get the entry point at all, so it is deliberately not
// done here; the fix is to the gate, in its own change.
// ---------------------------------------------------------------------------

// The diagnostic for a native route the build cannot service. Written as a
// function per op so the call and the buffer-size query cannot drift into two
// different messages.
//
// IT IS A THROW RATHER THAN A FALL-THROUGH TO THE VENDOR, and that is the whole
// point of it existing before any kernel does. Omitting the native branch would
// silently take the vendor at the exact moment a capability first comes off zero
// -- a kernel LINKED but never REACHED, and a test suite passing green over it.
// route_compiled.hh:1-24 names that defect class.
template <typename T>
[[noreturn]] inline void geqrf_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native geqrf kernel is linked into this build. "
        "sycl_geqrf::geqrf_cta_max_m_for_slm / geqrf_cta_max_elems_for_slm / "
        "geqrf_blocked_available reported a capability the facade cannot "
        "service.");
}

template <typename T>
[[noreturn]] inline void orgqr_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native orgqr driver is linked into this build. "
        "sycl_orgqr::orgqr_blocked_available reported a capability the facade "
        "cannot service.");
}

template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> work_space) {
    // VALIDATION FIRST, hoisted to here on purpose -- the trsm precedent at
    // level3.cc:167-174 and potrf's at :203-206. The positional geqrf validated
    // nothing anywhere, and this must precede the shape builder, which reads
    // A.rows()/A.cols(). It is deliberately one negative-extent test: see
    // geqrf.hh's note on why squareness, m >= n and tau's length are NOT checked
    // here.
    geqrf_validate_params<T>(A);

    // THE ROUTE IS RESOLVED BEFORE THE VENDOR-AVAILABLE TEST. Everything below
    // the `if constexpr` at the bottom of this body is UNREACHABLE in the
    // vendor-free build, which is the build this campaign exists for -- and geqrf
    // is the op that build fails on hardest today, because four ormqr/orgqr
    // suites call it from their SETUP to make the reflectors they then test with.
    const dispatch::Route route = backend::geqrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // BOTH NATIVE ARMS ARE LIVE as of WP5's kernels, and this block is the reason
    // landing them was a kernel landing rather than also a facade change --
    // route_compiled.hh:1-24 names the defect class where a kernel is compiled
    // into the library and no call ever arrives at it. In a vendor-PRESENT build
    // nothing reaches here, because preferred() is still false for both arms and
    // Origin::Auto therefore resolves to the vendor; a vendor-free build reaches
    // it through route_resolve.hh:60-63, and so does an explicit
    // BATCHLAS_GEQRF_ROUTE. The throw at the end of the block is what makes a
    // THIRD tier impossible to ship silently.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_geqrf::geqrf_cta_dispatch<T>(ctx, A, tau, work_space);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // THE TRAILING UPDATE GOES THROUGH THE ROUTER, not straight to the
            // native kernel. Calling sycl_gemm::gemm_custom from the driver is a
            // RECORDED DEFECT (WP3 step 16, trsm_native.hh:82-104): it bypasses
            // RouteTable<Op::gemm> entirely, so the driver would get the native
            // kernel even on the shapes WP2 measured it losing. Injection rather
            // than an include also keeps the kernel TU free of the dispatch layer:
            // a test calling geqrf_blocked_dispatch directly gets the native gemm,
            // and a vendor-free build is unaffected because the resolver falls
            // back to native there anyway (route_resolve.hh:60-63). The signature
            // is identical to both sycl_gemm::gemm_custom and batchlas::gemm, so
            // nothing adapts.
            //
            // Measured stake, for whoever writes the driver: the two WY trailing
            // GEMMs are only 33.40 ms of a 2109.8 ms float N=1024 b=128 vendor
            // call, so this seam is NOT where WP5 is won -- but its complex arm is
            // where a vendor-free build loses 2.0-2.6x, all of it in the
            // transposed W = V^H A22 gemm, and only the router knows which
            // implementation to hand it.
            return sycl_geqrf::geqrf_blocked_dispatch<T>(
                ctx, A, tau, work_space,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ga,
                   const MatrixView<T, MatrixFormat::Dense>& gb,
                   const MatrixView<T, MatrixFormat::Dense>& gc,
                   T galpha, T gbeta, Transpose gta, Transpose gtb,
                   ComputePrecision gp) {
                    return gemm<B, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                });
        }
        // Kept as the trailing default: Algorithm::Auto, and any future native
        // tier, must not fall through to the vendor.
        geqrf_throw_native_unimplemented<T>(route, "geqrf");
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::geqrf_vendor<B, T>(ctx, A, tau, work_space);
    }
}

template <Backend B, typename T>
size_t geqrf_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         Span<T> tau) {
    // THE QUERY MOVES WITH THE CALL: same validator, same builder, same
    // *_route function, same arguments, therefore the same GeqrfShape and the
    // same Route BY CONSTRUCTION rather than by a comment asking for it. The note
    // at the top of this file records why -- "Splitting them would let the two
    // resolve differently, which is the defect class S4d found in ormqr (buffer
    // size 2560 bytes, call demanded 276480)".
    //
    // The double resolution is real for geqrf: options.hh:720 calls
    // ctx.workspace(geqrf_buffer_size<B,T>(...)) and :721 calls geqrf<B,T>(...),
    // two separate getenv reads inside one API call.
    geqrf_validate_params<T>(A);

    const dispatch::Route route = backend::geqrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor), NOT "whatever the chosen route needs", and with two
    // native tiers it is max OVER EVERY SUPPORTED ONE rather than over the one
    // THIS resolution chose. Both halves of the argument are potrf's
    // (:315-345 below) and transfer verbatim: a chosen-only size turns a
    // query/call disagreement into an UNDER-allocation, which is the ormqr
    // failure mode, while max() turns it into a harmless over-allocation. It is
    // safe ONLY because every term is an alignment multiple -- the native queries
    // replay their layouts through BumpAllocator::measuring() (mempool.hh:185-190)
    // rather than hand-summing, and every vendor query sums allocation_size terms
    // (cublas.cc:1284-1289, :1444-1448; rocsolver.cc:96-102 and
    // netlib_lapack.cc:1427-1434 return 0).
    //
    // A HAZARD max() DOES NOT COVER, AND IT IS geqrf-ONLY: THE QUERY AND THE CALL
    // ARE MADE AGAINST DIFFERENT SHAPES. band_reduction.cc:1041-1044 (duplicated
    // at :1185-1187) sizes sytrd's band reduction with
    //     MatrixView<T,Dense> dummyB(nullptr, m_max, nb_max, ...);
    //     bytes += geqrf_buffer_size<B,T>(ctx, dummyB, dummyTau);
    // while the actual call at band_reduction.cc:595 passes `Bsub`, an m x r
    // sub-view. max() over ROUTES at one shape says nothing about that.
    // Therefore any native geqrf_*_buffer_size must be MONOTONE NON-DECREASING in
    // (rows, cols, batch), with its own test over a grid -- and must never
    // dereference A.data_ptr() or tau.data(), both nullptr there.
    //
    // WHAT IS LEFT OPEN, deliberately, exactly as for potrf: the native terms are
    // computed only inside `if (is_native(route))`, so the VENDOR -> NATIVE
    // direction (the environment changing between options.hh:720 and :721) still
    // ends in a BumpAllocator::allocate throw. Closing it means computing the
    // blocked layout unconditionally on every vendor-present geqrf that will never
    // touch it, to defend against a getenv that changes inside one API call; a
    // throw is also the benign end of this defect class.
    // `native_fired` IS NOT `native_need != 0`, AND THE DIFFERENCE IS REAL FOR
    // geqrf WHERE IT WAS NOT FOR potrf. The CTA tier's workspace is legitimately
    // ZERO -- its tile is local memory and tau is the caller's span -- so a
    // size of 0 is a valid answer, not evidence that no tier fired. Reading the
    // internal-consistency check off the size would make a CTA-only build
    // (blocked absent, capacities positive) throw on every call the route table
    // had just promised. The flag says which question is being asked.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::geqrf_op_shape<B, T>(ctx, A);
        using Tbl = dispatch::RouteTable<dispatch::Op::geqrf, T>;
        if (shape) {
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA}, *shape)) {
                native_need = std::max(native_need,
                                       sycl_geqrf::geqrf_cta_buffer_size<T>(ctx, A));
                native_fired = true;
            }
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                              *shape)) {
                native_need = std::max(native_need,
                                       sycl_geqrf::geqrf_blocked_buffer_size<T>(ctx, A));
                native_fired = true;
            }
        }
        if (!native_fired) {
            // is_native(route) says supports() accepted SOMETHING; if neither
            // query above fired, the two disagree and that is a bug in this file
            // rather than a shape the caller can fix.
            geqrf_throw_native_unimplemented<T>(route, "geqrf_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        // A vendor-free build with no native route left is the NoRouteError this
        // work package exists to remove; with one, the native term is the whole
        // answer.
        //
        // GATED ON native_fired, NOT ON native_need != 0, for the reason above:
        // the CTA tier's workspace is legitimately zero, and reading "no route"
        // off a zero size would throw NoRouteError on a shape the table just
        // routed to a working kernel.
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::geqrf_vendor_buffer_size<B, T>(ctx, A, tau));
    }
}

template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace) {
    orgqr_validate_params<T>(A);

    const dispatch::Route route = backend::orgqr_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // LIVE as of WP5's kernels. In a vendor-PRESENT build nothing reaches here --
    // preferred() is still false, so Origin::Auto resolves to cuSOLVER's
    // per-batch-item loop -- while a vendor-free build reaches it through
    // route_resolve.hh:60-63, as does an explicit BATCHLAS_ORGQR_ROUTE. Written
    // ahead of the driver for route_compiled.hh:1-24's reason: without it, the
    // day the flag came off false the facade would have silently kept taking the
    // vendor and the driver would have been LINKED but never REACHED.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::Blocked) {
            // THE APPLY GOES THROUGH THE ROUTER. orgqr is ormqr against an
            // identity (route_orgqr.hh carries the measurement), and the apply is
            // injected rather than called directly for two reasons, the second
            // stronger than the first:
            //
            //   1. Calling ormqr_blocked from the driver TU would bypass
            //      RouteTable<Op::ormqr> and ignore BATCHLAS_ORMQR_ROUTE -- WP3
            //      step 16's recorded defect, one level up.
            //   2. batchlas::ormqr<B,T> needs a Backend and the driver is
            //      instantiated per scalar type with none, so injection is the
            //      ONLY way to reach the routed ormqr from that TU at all. The
            //      alternative is ormqr_blocked's 4x4 Backend x type
            //      cross-product (internal/ormqr_blocked.hh:23-39) in a build
            //      that is device-link-bound.
            //
            // The apply's SIZE is injected alongside it, from the same routed
            // query, so the two cannot resolve differently -- the anti-pattern
            // being avoided is ormqr_buffer_size_dispatch (ormqr.hh:281-303),
            // which returns only the chosen route's size and agrees with its call
            // only because getenv happened to return the same thing twice.
            //
            // Argument order is the POSITIONAL entry point's (ormqr.hh:311-320),
            // not an option struct's -- W13 records the compile error from
            // confusing the two.
            return sycl_orgqr::orgqr_blocked_dispatch<T>(
                ctx, A, tau, workspace,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& oa,
                   const MatrixView<T, MatrixFormat::Dense>& oc,
                   Side oside, Transpose otrans, Span<T> otau,
                   Span<std::byte> ows, int32_t obs) {
                    return ormqr<B, T>(c, oa, oc, oside, otrans, otau, ows, obs);
                },
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& oa,
                   const MatrixView<T, MatrixFormat::Dense>& oc,
                   Side oside, Transpose otrans, Span<T> otau, int32_t obs) {
                    return ormqr_buffer_size<B, T>(c, oa, oc, oside, otrans, otau, obs);
                });
        }
        orgqr_throw_native_unimplemented<T>(route, "orgqr");
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::orgqr_vendor<B, T>(ctx, A, tau, workspace);
    }
}

template <Backend B, typename T>
size_t orgqr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau) {
    // Same validator, same builder, same route function, same arguments as the
    // call above -- see geqrf_buffer_size. options.hh:741/:742 is orgqr's double
    // resolution: one ctx.workspace(orgqr_buffer_size<B,T>(...)) then one
    // orgqr<B,T>(...), two getenv reads inside one API call.
    orgqr_validate_params<T>(A);

    const dispatch::Route route = backend::orgqr_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor) with ONE native tier. The max is kept anyway, for the
    // reason potrf's comment gives: it costs one shape build and turns a
    // query/call disagreement into an over-allocation instead of the ormqr
    // under-allocation. Do not "simplify" it back to the chosen route.
    //
    // Worth knowing when the native term stops being 0: the VENDOR term here
    // scales linearly with the batch. orgqr_vendor_buffer_size returns
    // single * batch_size (cublas.cc:1446-1448) because the vendor path is a
    // per-item loop, which is 1164 MB for float n=64 b=8192 and 4644 MB for
    // cdouble at the same shape. A max() against that is not a tight bound in a
    // vendor-present build -- it is the vendor's own requirement, unchanged.
    // `native_fired` IS NOT `native_need != 0` -- THE SAME DISTINCTION
    // geqrf_buffer_size makes above, and it is made here for the same reason
    // even though today only one orgqr tier exists and its layout is never
    // empty. A zero workspace is a LEGITIMATE ANSWER, not evidence that no tier
    // fired: orgqr_native.hh and orgqr_blocked.cc both contemplate a specialised
    // orgqr that writes Q in place into A and therefore needs no m x n identity
    // scratch. On the day that tier lands, `native_need == 0` would make
    // supports() return true, orgqr() dispatch to the new kernel and work, while
    // orgqr_buffer_size() threw orgqr_throw_native_unimplemented -- and then, in
    // a vendor-free build, NoRouteError -- on a shape the table had just
    // promised. Every caller that sizes then calls (options.hh:741-742,
    // ortho.cc:90) would fail before reaching the working kernel. The flag says
    // which question is being asked; the size says how many bytes.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::orgqr_op_shape<B, T>(ctx, A);
        using Tbl = dispatch::RouteTable<dispatch::Op::orgqr, T>;
        if (shape &&
            Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked}, *shape)) {
            // THE APPLY'S SIZE IS INJECTED FROM THE SAME ROUTED QUERY THE CALL
            // USES. The driver's workspace is the m x n identity plus whatever
            // the routed ormqr demands, and only this layer can resolve
            // RouteTable<Op::ormqr>. Handing the driver a two-argument query
            // instead would force it to hand-roll ormqr_blocked's formula --
            // which is drift, and which is simply WRONG whenever the apply
            // resolves to a vendor ormqr, whose workspace has a different shape
            // entirely. `tau` travels with it because ormqr_buffer_size
            // validates tau.size() >= k * batch; it is read for its SIZE only,
            // and the C view the driver builds for this query is a NULL one.
            native_need = std::max(
                native_need,
                sycl_orgqr::orgqr_blocked_buffer_size<T>(
                    ctx, A, tau,
                    [](Queue& c,
                       const MatrixView<T, MatrixFormat::Dense>& oa,
                       const MatrixView<T, MatrixFormat::Dense>& oc,
                       Side oside, Transpose otrans, Span<T> otau, int32_t obs) {
                        return ormqr_buffer_size<B, T>(c, oa, oc, oside, otrans, otau, obs);
                    }));
            native_fired = true;
        }
        if (!native_fired) {
            // is_native(route) says supports() accepted SOMETHING; if the query
            // above did not fire, the two disagree and that is a bug in this
            // file rather than a shape the caller can fix.
            orgqr_throw_native_unimplemented<T>(route, "orgqr_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        // GATED ON native_fired, NOT ON native_need != 0 -- see above.
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::orgqr_vendor_buffer_size<B, T>(ctx, A, tau));
    }
}

// ---------------------------------------------------------------------------
// WP6 -- the LU family. All three are hooked exactly as geqrf/orgqr are above,
// and the three "native route with no kernel" diagnostics are written the same
// way.
//
// THEY USE factorization_vendor_available<B>, NOT solver_vendor_available<B>.
// getrf/getrs/getri all come from cuBLAS on NVIDIA (cublas.cc:1493, :1453, :1521),
// like geqrf/orgqr and unlike potrf. See the note at :69-87 for the LATENT GATE
// DEFECT underneath that predicate -- vendor_available.hh:11-13 asserts a
// file-level mapping that is false of the SYMBOLS. The extra place it used to
// bite the LU family is GONE: cublas.cc's getrs had TWO arms and the batch <= 1
// one called cusolverDnXgetrs -- a different library and the 64-bit non-batched
// API -- from inside a TU gated on BATCHLAS_HAS_CUBLAS, so a cuBLAS-present /
// cuSOLVER-absent configure claimed a vendor it could not link. That arm was
// deleted in WP6's repair pass, because it was ALSO a crash: it read the packed
// int32 pivots every getrf in this tree writes as genuine int64 and indexed out
// of bounds (see the note at cublas.cc:1465). The LATENT GATE DEFECT itself
// remains for geqrf and is still a change of its own.
//
// STATUS: ALL THREE OPS HAVE A LIVE NATIVE ARM, for all four scalar types.
// getrf has two tiers -- getrf_cta_max_n_for_slm<T>() measures 155/109/109/77 for
// float/double/cfloat/cdouble on an RTX 4090, and getrf_blocked_available<T>() is
// true for every type -- and getrs_blocked_available<T>() / getri_blocked_
// available<T>() are true as well. A VENDOR-FREE BUILD TAKES THE NATIVE ARM FOR
// EVERY SQUARE SHAPE; the six bodies below are reached, not scaffolding.
//
// The vendor-PRESENT build is unchanged, and that is a routing fact rather than a
// missing kernel: preferred() is still false everywhere in all three tables, so
// Origin::Auto keeps returning {Vendor, Auto} wherever a vendor exists
// (route_resolve.hh:110-112, :129). Measured at 96 of 96 route cells. The
// benchmark that would replace those all-false windows with a measured one is
// docs/perf/lu.md#getrf-window-evidence; read it before writing one, because the
// crossover moves with BATCH and not only with order.
// ---------------------------------------------------------------------------

// The diagnostics for a native route the build cannot service. One function per
// op so the call and the buffer-size query cannot drift into two different
// messages.
//
// THEY ARE THROWS RATHER THAN FALL-THROUGHS TO THE VENDOR, and that is the whole
// point of writing them before any kernel exists. Omitting the native branch would
// silently take the vendor at the exact moment a capability first comes off zero
// -- a kernel LINKED but never REACHED, and a test suite passing green over it.
// route_compiled.hh:1-24 names that defect class, and WP5's break B9 measured it:
// deleting geqrf's native arm turned NOTHING red anywhere, in either build.
template <typename T>
[[noreturn]] inline void getrf_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native getrf kernel is linked into this build. "
        "sycl_getrf::getrf_cta_max_n_for_slm / getrf_blocked_available reported a "
        "capability the facade cannot service.");
}

template <typename T>
[[noreturn]] inline void getrs_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native getrs driver is linked into this build. "
        "sycl_getrs::getrs_blocked_available reported a capability the facade "
        "cannot service.");
}

template <typename T>
[[noreturn]] inline void getri_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native getri driver is linked into this build. "
        "sycl_getri::getri_blocked_available reported a capability the facade "
        "cannot service.");
}

template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
    // VALIDATION FIRST, hoisted here on purpose -- the trsm precedent at
    // level3.cc:167-174 and potrf's at :203-206 -- because it must precede the
    // shape builder, which reads A.rows()/A.cols(). It is deliberately one
    // negative-extent test: see getrf.hh's note on why squareness and the pivot
    // span's length are NOT checked here.
    getrf_validate_params<T>(A);

    // THE ROUTE IS RESOLVED BEFORE THE VENDOR-AVAILABLE TEST. Everything below the
    // `if constexpr` at the bottom of this body is UNREACHABLE in the vendor-free
    // build, which is the build this campaign exists for.
    const dispatch::Route route = backend::getrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // BOTH NATIVE ARMS ARE LIVE, and this block serves EVERY getrf in the
    // vendor-free build (verified reached, not merely linked: the coverage capture
    // shows getrf resolving native for all four types and both tiers, and nsys
    // shows GetrfPanelResidentKernel / GetrfPanelGlobalKernel executing). In the
    // vendor-present build it is unreachable only because preferred() is all-false,
    // which is a MEASURED WINDOW that has not been written yet -- not a missing
    // kernel. The throw at the end is still what makes a THIRD tier impossible to
    // ship silently.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_getrf::getrf_cta_dispatch<T>(ctx, A, pivots, work_space, info);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // THE TRAILING UPDATE AND THE PANEL SOLVE GO THROUGH THE ROUTER, not
            // straight to a native kernel. Calling sycl_gemm::gemm_custom from the
            // driver is a RECORDED DEFECT (WP3 step 16, trsm_native.hh:82-104,
            // fix at level3.cc:186-231): it bypasses RouteTable<Op::gemm>
            // entirely, so the driver would get the native kernel even on the
            // shapes WP2 measured it losing.
            //
            // Injection is also the ONLY way to reach gemm/trsm from that TU at
            // all: the driver is instantiated per scalar type with no Backend
            // parameter, and gemm<B,T> / trsm<B,T> need one. Only this layer can
            // name a routed entry point.
            //
            // Both signatures are the routed entry points' verbatim -- note that
            // trsm's alpha comes THIRD (functions/trsm.hh:100-108); the old
            // spelling is a deleted overload at :121-138 so a stale call cannot
            // silently compile.
            //
            // MEASURED STAKE FOR WHOEVER WRITES THE DRIVER, and it INVERTS the
            // prediction WP6 inherited: at the real batch and stride, the LU
            // trailing update (NN, k = nb) reaches Tiled128x128RegisterK8 for
            // float and Tiled64x64RegisterK16Wide for BOTH complex types, while
            // DOUBLE lands on Tiled16 at all 13 shapes. That is structural -- the
            // wide-scalar CTA-count relaxation is complex-only and the other
            // wide-scalar door needs min_dim >= 256, which k = nb can never
            // satisfy -- and it belongs to GEMM, not to WP6.
            return sycl_getrf::getrf_blocked_dispatch<T>(
                ctx, A, pivots, work_space, info,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ga,
                   const MatrixView<T, MatrixFormat::Dense>& gb,
                   const MatrixView<T, MatrixFormat::Dense>& gc,
                   T galpha, T gbeta, Transpose gta, Transpose gtb,
                   ComputePrecision gp) {
                    return gemm<B, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                },
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ta,
                   const MatrixView<T, MatrixFormat::Dense>& tb,
                   T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
                    return trsm<B, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
                });
        }
        // Kept as the trailing default: Algorithm::Auto, and any future native
        // tier, must not fall through to the vendor.
        getrf_throw_native_unimplemented<T>(route, "getrf");
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getrf_vendor<B, T>(ctx, A, pivots, work_space, info);
    }
}

template <Backend B, typename T>
size_t getrf_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A) {
    // THE QUERY MOVES WITH THE CALL: same validator, same builder, same *_route
    // function, SAME ARGUMENTS, therefore the same GetrfShape and the same Route
    // BY CONSTRUCTION rather than by a comment asking for it. The note at the top
    // of this file records why -- "Splitting them would let the two resolve
    // differently, which is the defect class S4d found in ormqr (buffer size 2560
    // bytes, call demanded 276480)".
    //
    // The double resolution is real for getrf and there are TWO instances of it:
    // options.hh:619 calls ctx.workspace(getrf_buffer_size<B,T>(...)) and :620
    // calls getrf<B,T>(...), and src/extensions/inv.cc:36 sizes while :48 calls --
    // each pair being two separate getenv reads inside one API call.
    getrf_validate_params<T>(A);

    const dispatch::Route route = backend::getrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor), NOT "whatever the chosen route needs", and with two
    // native tiers it is max OVER EVERY SUPPORTED ONE rather than over the one
    // THIS resolution chose. Both halves of the argument are potrf's and geqrf's
    // and transfer verbatim: a chosen-only size turns a query/call disagreement
    // into an UNDER-allocation, which is the ormqr failure mode, while max() turns
    // it into a harmless over-allocation. It is safe ONLY because every term is an
    // alignment multiple -- the native queries must replay their layouts through
    // BumpAllocator::measuring() (mempool.hh:186-190) rather than hand-summing,
    // and every vendor query sums allocation_size terms (cublas.cc:1518, :1490;
    // netlib_lapack.cc:1331-1336).
    //
    // NO MONOTONICITY REQUIREMENT HERE, unlike geqrf. That hazard exists because
    // band_reduction.cc sizes at (m_max x nb_max) and calls at a smaller sub-view;
    // nothing in the LU consumers does that. inv_layout sizes against the CALLER's
    // A (inv.cc:34-36) and calls against the shape-identical Acopy (:17-23). Do
    // not import a constraint with no caller behind it.
    //
    // WHAT IS LEFT OPEN, deliberately, exactly as for potrf and geqrf: the native
    // terms are computed only inside `if (is_native(route))`, so the VENDOR ->
    // NATIVE direction (the environment changing between options.hh:619 and :620)
    // still ends in a BumpAllocator::allocate throw. Closing it means computing the
    // blocked layout unconditionally on every vendor-present getrf that will never
    // touch it, to defend against a getenv that changes inside one API call; a
    // throw is also the benign end of this defect class.
    //
    // `native_fired` IS NOT `native_need != 0`, and the difference is expected to
    // be real for getrf as it is for geqrf: the CTA tier's workspace is plausibly
    // ZERO -- its tile is local memory and the pivots are the caller's span -- so a
    // size of 0 is a valid answer, not evidence that no tier fired. Reading the
    // internal-consistency check off the size would make a CTA-only build (blocked
    // absent, capacity positive) throw on every call the route table had just
    // promised. orgqr_buffer_size shipped with precisely that latent defect and it
    // was fixed in the WP5 repair pass.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::getrf_op_shape<B, T>(ctx, A);
        using Tbl = dispatch::RouteTable<dispatch::Op::getrf, T>;
        if (shape) {
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA}, *shape)) {
                native_need = std::max(native_need,
                                       sycl_getrf::getrf_cta_buffer_size<T>(ctx, A));
                native_fired = true;
            }
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                              *shape)) {
                native_need = std::max(native_need,
                                       sycl_getrf::getrf_blocked_buffer_size<T>(ctx, A));
                native_fired = true;
            }
        }
        if (!native_fired) {
            // is_native(route) says supports() accepted SOMETHING; if neither
            // query above fired, the two disagree and that is a bug in this file
            // rather than a shape the caller can fix.
            getrf_throw_native_unimplemented<T>(route, "getrf_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        // A vendor-free build with no native route left is the NoRouteError this
        // work package exists to remove; with one, the native term is the whole
        // answer. GATED ON native_fired, not on native_need != 0, for the reason
        // above.
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::getrf, B, dispatch::kFactorizationLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::getrf_vendor_buffer_size<B, T>(ctx, A));
    }
}

template <Backend Back, typename T>
Event getrs(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            const MatrixView<T,MatrixFormat::Dense>& B,
            Transpose transA,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
    // VALIDATION FIRST -- it must precede the shape builder, which reads
    // A.rows()/B.cols(). Deliberately one negative-extent test; see getrs.hh on
    // why squareness, A/B agreement and the pivot span's length are NOT checked
    // here.
    getrs_validate_params<T>(A, B);

    const dispatch::Route route = backend::getrs_route<Back, T>(
        ctx, A, B, transA,
        /*vendor_available=*/dispatch::factorization_vendor_available<Back>);

    // LIVE, AND TWO-TIERED. Both native arms exist for every scalar type. In the
    // vendor-free build native_tier_preferred sends every shape the FUSED tier can
    // hold to Algorithm::CTA and the rest to Algorithm::Blocked; in the
    // vendor-present build neither is reached, because preferred() is all-false.
    // See the block comment above getrf.
    //
    // MEASURED, and it matters if a preferred() window is ever written here: the
    // COMPOSITION is a crossover on nrhs (geomean 0.32x of cuBLAS at nrhs=1 rising
    // to 1.36x at nrhs=128, docs/perf/lu.md#the-vendor-baseline-and-saturation), while the FUSED
    // tier is 2.10x of cuBLAS at nrhs=1 with no losses over 15 cells and crosses
    // the other way as nrhs grows (docs/perf/lu.md#the-fused-narrow-rhs-getrs).
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            // THE FUSED NARROW-RHS TIER. It injects NOTHING -- no trsm, no gemm,
            // no laswp -- because it calls no other BLAS operation at all: the
            // permutation and both substitutions are one kernel. That is why this
            // arm has no seam where the Blocked one below has one.
            return sycl_getrs::getrs_fused_dispatch<T>(
                ctx, A, B, transA, pivots, work_space);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // BOTH TRIANGULAR SOLVES GO THROUGH THE ROUTER. Calling a native trsm
            // entry point from the driver TU is the recorded WP3-step-16 defect
            // (trsm_native.hh:82-104, fix at level3.cc:186-231), and injection is
            // in any case the only way to reach trsm<Back,T> from a TU
            // instantiated per scalar type with no Backend parameter.
            //
            // Signature is the routed batchlas::trsm's positional form verbatim --
            // alpha THIRD (functions/trsm.hh:100-108); the old spelling is a
            // deleted overload at :121-138 so a stale call cannot silently compile.
            //
            // NO BUFFER-SIZE TWIN is injected alongside it, unlike orgqr's
            // OrgqrApplyQ/OrgqrApplyQBufferSize pair (:349-363 and :431-440). That
            // pair exists because the routed ormqr HAS a workspace whose size must
            // come from the same resolution as the call; the public trsm takes no
            // workspace at all, so there is nothing here for a query and a call to
            // disagree about.
            return sycl_getrs::getrs_blocked_dispatch<T>(
                ctx, A, B, transA, pivots, work_space,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ta,
                   const MatrixView<T, MatrixFormat::Dense>& tb,
                   T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
                    return trsm<Back, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
                });
        }
        getrs_throw_native_unimplemented<T>(route, "getrs");
    }

    if constexpr (!dispatch::factorization_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrs, Back, dispatch::kFactorizationLibrary<Back>);
    } else {
        return backend::getrs_vendor<Back, T>(ctx, A, B, transA, pivots, work_space);
    }
}

template <Backend Back, typename T>
size_t getrs_buffer_size(Queue& ctx,
                         const MatrixView<T,MatrixFormat::Dense>& A,
                         const MatrixView<T,MatrixFormat::Dense>& B,
                         Transpose transA) {
    // THE QUERY MOVES WITH THE CALL: same validator, same builder, same *_route
    // function, SAME ARGUMENTS -- including transA, which is a live routing input
    // for this op and would silently split the two resolutions if the query
    // dropped it. The double resolution is real: options.hh:651 sizes and :652
    // calls, two getenv reads inside one API call.
    getrs_validate_params<T>(A, B);

    const dispatch::Route route = backend::getrs_route<Back, T>(
        ctx, A, B, transA,
        /*vendor_available=*/dispatch::factorization_vendor_available<Back>);

    // max(native, vendor) -- with ONE native tier this is max over that tier and
    // the vendor. It is still not "whatever the chosen route needs": the two
    // resolutions can differ (see above), and max() turns a disagreement into a
    // harmless over-allocation where a chosen-only size turns it into the ormqr
    // UNDER-allocation. Safe only because both terms are alignment multiples.
    //
    // `native_fired` rather than `native_need != 0`, for the same reason as getrf:
    // an in-place interchange walk legitimately needs ZERO workspace, and only the
    // collapsed-gather strategy needs a buffer.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::getrs_op_shape<Back, T>(ctx, A, B, transA);
        using Tbl = dispatch::RouteTable<dispatch::Op::getrs, T>;
        if (shape) {
            // max OVER EVERY NATIVE TIER THAT COULD SERVE THIS SHAPE, not over the
            // one the route named. The query and the call resolve independently
            // (options.hh:651 sizes and :652 calls, two getenv reads inside one
            // API call), so a lease sized for only one tier is a lease the other
            // tier can overrun. Both are 0 today -- the fused tier is entirely
            // local-memory resident and the composition is in-place -- and that is
            // exactly why the max must be written now rather than when one of them
            // grows a workspace.
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA},
                              *shape)) {
                native_need = std::max(
                    native_need,
                    sycl_getrs::getrs_fused_buffer_size<T>(ctx, A, B, transA));
                native_fired = true;
            }
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                              *shape)) {
                native_need = std::max(
                    native_need,
                    sycl_getrs::getrs_blocked_buffer_size<T>(ctx, A, B, transA));
                native_fired = true;
            }
        }
        if (!native_fired) {
            getrs_throw_native_unimplemented<T>(route, "getrs_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<Back>) {
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::getrs, Back, dispatch::kFactorizationLibrary<Back>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::getrs_vendor_buffer_size<Back, T>(ctx, A, B, transA));
    }
}

template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
    // VALIDATION FIRST -- it must precede the shape builder, which reads
    // A.rows()/A.cols(). The two-argument arity is used here and the one-argument
    // arity in the query, because getri_buffer_size has no C; see getri.hh on why
    // that split is forced, and src/backends/getri_route.hh on why the ROUTE is a
    // function of A alone.
    getri_validate_params<T>(A, C);

    const dispatch::Route route = backend::getri_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // LIVE. getri_blocked_available is true for every scalar type, and this arm
    // serves every getri in the vendor-free build -- it is what closed
    // inverse_tests. Unreached in the vendor-present build only because
    // preferred() is all-false; see the block comment above getrf.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::Blocked) {
            // BOTH TRIANGULAR SOLVES GO THROUGH THE ROUTER -- the WP3-step-16
            // rule, and the only way to reach trsm<B,T> from a TU instantiated per
            // scalar type with no Backend parameter. Signature is the routed
            // batchlas::trsm's positional form verbatim, alpha THIRD.
            //
            // NOTE WHAT IS *NOT* INJECTED: the row permutation. getri's measured
            // design writes P straight into C rather than writing an identity and
            // permuting it -- same store count, one kernel, ZERO workspace -- so
            // there is no second routed op here, and no buffer-size twin either
            // (the routed trsm takes no workspace).
            return sycl_getri::getri_blocked_dispatch<T>(
                ctx, A, C, pivots, work_space, info,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ta,
                   const MatrixView<T, MatrixFormat::Dense>& tb,
                   T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
                    return trsm<B, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
                });
        }
        getri_throw_native_unimplemented<T>(route, "getri");
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getri, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getri_vendor<B, T>(ctx, A, C, pivots, work_space, info);
    }
}

template <Backend B, typename T>
size_t getri_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A) {
    // THE QUERY MOVES WITH THE CALL: same builder, same *_route function, SAME
    // ARGUMENTS -- which for getri means A alone, in both places, because
    // getri_buffer_size's signature has no C. That is exactly why
    // backend::getri_op_shape takes A alone: a builder that read C could be called
    // from only one of the two sites, which is the split factorization.cc:8-10
    // forbids.
    //
    // TWO double-resolution sites exist for this op: options.hh:695 sizes and :696
    // calls, and src/extensions/inv.cc:35 sizes and :49 calls.
    //
    // THIS QUERY RUNS UNDER BumpAllocator::measuring(). inv_buffer_size
    // (inv.cc:54-57) replays inv_layout, which calls this at :35, so per
    // mempool.hh:180-186 everything reachable from here must be PURE WITH RESPECT
    // TO THE WORKSPACE -- no workspace read or write, no kernel launch -- and must
    // not dereference A.data_ptr().
    getri_validate_params<T>(A);

    const dispatch::Route route = backend::getri_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor); `native_fired` rather than `native_need != 0`, because
    // the native arm's workspace is EXPECTED to be zero (P is written straight into
    // C and the routed trsm allocates nothing) and reading the consistency check
    // off the size would throw on every call the table had just promised.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::getri_op_shape<B, T>(ctx, A);
        using Tbl = dispatch::RouteTable<dispatch::Op::getri, T>;
        if (shape) {
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                              *shape)) {
                native_need = std::max(native_need,
                                       sycl_getri::getri_blocked_buffer_size<T>(ctx, A));
                native_fired = true;
            }
        }
        if (!native_fired) {
            getri_throw_native_unimplemented<T>(route, "getri_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::getri, B, dispatch::kFactorizationLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::getri_vendor_buffer_size<B, T>(ctx, A));
    }
}

// The diagnostic for a route the build cannot service. Written as a function so
// the call and the buffer-size query cannot drift into two different messages.
//
// It is a THROW rather than a fall-through to the vendor on purpose. Omitting
// the native branch would silently take the vendor at the exact moment
// potrf_cta_max_n<T>() first comes off zero -- a kernel LINKED but never
// REACHED, and a test suite passing green over it. route_compiled.hh:1-24 names
// that defect class.
template <typename T>
[[noreturn]] inline void potrf_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native potrf kernel is linked into this build. "
        "sycl_potrf::potrf_cta_max_n / potrf_blocked_available reported a "
        "capability the facade cannot service.");
}

template <Backend B, typename T>
Event potrf(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& descrA,
                Uplo uplo,
                Span<std::byte> workspace,
                Span<int32_t> info_out) {
    // VALIDATION FIRST, and hoisted to here on purpose -- the trsm precedent at
    // level3.cc:167-174. The positional potrf validated nothing anywhere, and
    // this must precede the shape builder, which reads A.rows()/A.cols().
    potrf_validate_params<T>(descrA, uplo);

    // THE ROUTE IS RESOLVED BEFORE THE VENDOR-AVAILABLE TEST. Anything below
    // the `if constexpr` at the bottom of this body is UNREACHABLE in the
    // vendor-free build, which is the build WP4 exists for. spec:463's
    // instruction to route through src/linalg-impl.hh has no referent -- grep
    // for potrf there returns exactly one line, :732, inside a comment.
    //
    // solver_vendor_available, NOT factorization_vendor_available: potrf comes
    // from cuSOLVER on NVIDIA and the two differ on CUDA
    // (vendor_available.hh:41-45 vs :47-52).
    const dispatch::Route route = backend::potrf_route<B, T>(
        ctx, descrA, uplo,
        /*vendor_available=*/dispatch::solver_vendor_available<B>);

    // BOTH NATIVE TIERS ARE REACHED, not merely linked. route_compiled.hh:1-24
    // names the defect class where a kernel is compiled into the library and no
    // call ever arrives at it; the throw at the end of this block is what makes
    // a third tier impossible to ship silently.
    //
    // In a VENDOR-PRESENT build neither arm is reached by default --
    // RouteTable<Op::potrf,T>::preferred() is still all-false, so Origin::Auto
    // returns {Vendor, Auto} for every shape (route_resolve.hh:57-65) and this
    // whole block is entered only when a caller pins BATCHLAS_POTRF_ROUTE. In a
    // VENDOR-FREE build :60-63 hands over any SUPPORTED native route, which is
    // the point of the work package.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_potrf::potrf_cta_dispatch<T>(ctx, descrA, uplo, workspace, info_out);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // THE TRAILING UPDATE AND THE PANEL SOLVE GO THROUGH THE ROUTER, not
            // straight to the native kernels. Calling sycl_gemm::gemm_custom from
            // the driver is a RECORDED DEFECT (WP3 step 16, trsm_native.hh:82-104):
            // it bypasses RouteTable<Op::gemm> entirely, so the driver would get
            // the native kernel even on the shapes WP2 had already measured it
            // losing. For potrf the stake is larger than it was for trsm -- the
            // trailing update is 65-95% of a vendor-free blocked factorisation,
            // and native/vendor on those exact shapes is 0.13-0.18x for float,
            // 0.21-0.23x for cfloat and 0.33-0.34x for cdouble (double is the one
            // type where native WINS, at 1.15-1.19x). Only the router knows that.
            //
            // The panel solve is injected for a STRONGER reason than preference:
            // the driver is instantiated per scalar type with no Backend
            // parameter (potrf_cta.cc:706-726), and trsm<Back,T> needs one, so
            // injection is the only way to reach the routed trsm from that TU at
            // all. It is the right choice on the numbers too -- the routed trsm
            // beats the vendor in 46 of 48 measured panel cells.
            //
            // Injection rather than an include keeps the kernel TU free of the
            // dispatch layer: tests call potrf_blocked_dispatch directly and get
            // the native gemm and the native trsm, and a vendor-free build is
            // unaffected because the resolver falls back to native there anyway
            // (route_resolve.hh:60-63). Both signatures are identical to the
            // native entry points', so nothing adapts.
            return sycl_potrf::potrf_blocked_dispatch<T>(
                ctx, descrA, uplo, workspace, info_out,
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ga,
                   const MatrixView<T, MatrixFormat::Dense>& gb,
                   const MatrixView<T, MatrixFormat::Dense>& gc,
                   T galpha, T gbeta, Transpose gta, Transpose gtb,
                   ComputePrecision gp) {
                    return gemm<B, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
                },
                [](Queue& c,
                   const MatrixView<T, MatrixFormat::Dense>& ta,
                   const MatrixView<T, MatrixFormat::Dense>& tb,
                   T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
                    return trsm<B, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
                });
        }
        // Kept as the trailing default: Algorithm::Auto, and any future native
        // tier, must not fall through to the vendor.
        potrf_throw_native_unimplemented<T>(route, "potrf");
    }

    if constexpr (!dispatch::solver_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
    } else {
        return backend::potrf_vendor<B, T>(ctx, descrA, uplo, workspace, info_out);
    }
}

template <Backend B, typename T>
size_t potrf_buffer_size(Queue& ctx,
                        const MatrixView<T,MatrixFormat::Dense>& A,
                        Uplo uplo) {
    // THE QUERY MOVES WITH THE CALL, and that is not a style rule: the note at
    // the top of this file records why -- "Splitting them would let the two
    // resolve differently, which is the defect class S4d found in ormqr (buffer
    // size 2560 bytes, call demanded 276480)". Same validation, same builder,
    // same arguments, therefore the same PotrfShape and the same Route by
    // construction rather than by a comment asking for it.
    potrf_validate_params<T>(A, uplo);

    const dispatch::Route route = backend::potrf_route<B, T>(
        ctx, A, uplo,
        /*vendor_available=*/dispatch::solver_vendor_available<B>);

    // max(native, vendor), NOT "whatever the chosen route needs". options.hh:546-552
    // resolves the route TWICE -- once for the size at :550 and once for the call
    // at :551 -- and both reads hit getenv afresh, so a chosen-only size turns a
    // disagreement between them into an UNDER-allocation, which is the ormqr
    // failure mode (2560 bytes reported, 276480 demanded). max() turns the same
    // disagreement into a harmless over-allocation. It is safe only because every
    // term is an alignment multiple: potrf_cta_buffer_size replays its layout
    // through BumpAllocator::measuring() (mempool.hh:185-190) rather than
    // hand-summing, and every vendor query sums allocation_size terms
    // (cusolver.cc:35,37; rocsolver.cc:24,26; netlib_lapack.cc returns 0).
    //
    // It is also why src/backends/cusolver.cc:56 had to stop calling the PUBLIC
    // query: that line hands its result to cusolverDnXpotrf as the workspace-size
    // argument, and this max() is the moment it would have started lying.
    //
    // AND WITH TWO NATIVE TIERS IT IS max OVER EVERY SUPPORTED ONE, not over the
    // one THIS resolution chose. The argument above is the whole reason: the
    // double resolution re-reads the environment, so query and call can disagree
    // -- and a CTA-sized answer against a Blocked call is the ormqr
    // under-allocation one level down (potrf_cta_buffer_size is batch int32,
    // while the blocked layout adds a W x W x batch product buffer, i.e.
    // kilobytes against tens of megabytes). With one native tier "whatever the
    // chosen route needs" was airtight; with two it is not. Over-allocation is
    // harmless, under-allocation is the entire defect class, and the extra cost
    // here is one shape build -- pure arithmetic plus the two device-property
    // reads the resolution already performs.
    //
    // WHAT THIS max() DOES NOT COVER, stated because the paragraph above reads
    // as though it covered everything. The native terms are computed only inside
    // `if (dispatch::is_native(route))`. In a vendor-present build with nothing
    // pinned the query resolves {Vendor, Auto} (preferred() is all-false), so
    // native_need stays 0 and the answer is cuSOLVER's -- 512 bytes at the shapes
    // tests/potrf_tests.cc B12 measures. If the environment changed between the
    // query at options.hh:550 and the call at :551 in the vendor -> native
    // direction, the driver would be handed 512 bytes and BumpAllocator::allocate
    // would THROW from potrf_blocked_layout. The native -> native direction is
    // closed by the max() below; the vendor -> native direction is not.
    //
    // It is left open on purpose. Closing it means computing the blocked layout
    // unconditionally, which adds W*W*batch*sizeof(T) -- megabytes at large batch
    // -- to every vendor-present potrf that will never touch it, to defend
    // against a getenv that changes inside a single API call. A throw is also the
    // benign end of this defect class; the ormqr failure it is modelled on was an
    // under-allocation that ran. If potrf's preferred() ever comes off all-false,
    // revisit: query and call could then disagree for reasons other than getenv.
    //
    // Do not "simplify" this back to the chosen route.
    std::size_t native_need = 0;
    if (dispatch::is_native(route)) {
        const auto shape = backend::potrf_op_shape<B, T>(ctx, A, uplo);
        using Tbl = dispatch::RouteTable<dispatch::Op::potrf, T>;
        if (shape) {
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::CTA}, *shape)) {
                native_need = std::max(native_need,
                                       sycl_potrf::potrf_cta_buffer_size<T>(ctx, A));
            }
            if (Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked},
                              *shape)) {
                native_need = std::max(
                    native_need, sycl_potrf::potrf_blocked_buffer_size<T>(ctx, A, uplo));
            }
        }
        if (native_need == 0) {
            // is_native(route) says supports() accepted SOMETHING; if neither
            // query above fired, the two disagree and that is a bug in this file
            // rather than a shape the caller can fix.
            potrf_throw_native_unimplemented<T>(route, "potrf_buffer_size");
        }
    }

    if constexpr (!dispatch::solver_vendor_available<B>) {
        // A vendor-free build with no native route left is the NoRouteError this
        // work package exists to remove; with one, the native term is the whole
        // answer.
        if (native_need == 0) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::potrf, B, dispatch::kSolverLibrary<B>);
        }
        return native_need;
    } else {
        return std::max(native_need,
                        backend::potrf_vendor_buffer_size<B, T>(ctx, A, uplo));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

#define FACTORIZATION_ONE(B_, fp)              \
    OP_INSTANTIATE(geqrf, B_, fp)              \
    OP_INSTANTIATE(geqrf_buffer_size, B_, fp)  \
    OP_INSTANTIATE(orgqr, B_, fp)              \
    OP_INSTANTIATE(orgqr_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getrf, B_, fp)              \
    OP_INSTANTIATE(getrf_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getrs, B_, fp)              \
    OP_INSTANTIATE(getrs_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getri, B_, fp)              \
    OP_INSTANTIATE(getri_buffer_size, B_, fp)

#define FACTORIZATION_ALL(B_)                       \
    FACTORIZATION_ONE(B_, float)                    \
    FACTORIZATION_ONE(B_, double)                   \
    FACTORIZATION_ONE(B_, std::complex<float>)      \
    FACTORIZATION_ONE(B_, std::complex<double>)

#define POTRF_ALL(B_)                               \
    OP_INSTANTIATE(potrf, B_, float)                \
    OP_INSTANTIATE(potrf, B_, double)               \
    OP_INSTANTIATE(potrf, B_, std::complex<float>)  \
    OP_INSTANTIATE(potrf, B_, std::complex<double>) \
    OP_INSTANTIATE(potrf_buffer_size, B_, float)                \
    OP_INSTANTIATE(potrf_buffer_size, B_, double)               \
    OP_INSTANTIATE(potrf_buffer_size, B_, std::complex<float>)  \
    OP_INSTANTIATE(potrf_buffer_size, B_, std::complex<double>)

// geqrf/orgqr/getrf/getrs/getri come from cuBLAS on NVIDIA; potrf from cuSOLVER.
// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
FACTORIZATION_ALL(Backend::CUDA)
POTRF_ALL(Backend::CUDA)
#endif

// On ROCm all of them come from rocSOLVER.
#if BATCHLAS_HAS_ROCM_BACKEND
FACTORIZATION_ALL(Backend::ROCM)
POTRF_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
FACTORIZATION_ALL(Backend::NETLIB)
POTRF_ALL(Backend::NETLIB)
#endif

#undef POTRF_ALL
#undef FACTORIZATION_ALL
#undef FACTORIZATION_ONE
#undef OP_INSTANTIATE

}  // namespace batchlas
