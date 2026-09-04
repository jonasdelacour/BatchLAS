// The public factorization entry points -- geqrf, orgqr, getrf, getrs, getri and
// potrf -- defined once here rather than inside a vendor TU, so dropping a vendor
// library does not drop the public symbol.
// See docs/design/vendor-independence.md#the-entry-point-facade.
//
// Each op MUST stay next to its buffer-size query: separated, the two can resolve
// differently and the workspace is then under-allocated.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/potrf.hh>

// The routed ops the blocked drivers inject: their kernel TUs are instantiated
// per scalar type with no Backend parameter, so only this layer can name a route.
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// Routing adapters and native drivers. Each is a src/ header over public
// includes only, so the facade can include it in a vendor-free build.
#include "../../backends/potrf_route.hh"
#include "../../extensions/potrf_native.hh"

#include "../../backends/geqrf_route.hh"
#include "../../backends/orgqr_route.hh"
#include "../../extensions/geqrf_native.hh"
#include "../../extensions/orgqr_native.hh"

#include "../../backends/getrf_route.hh"
#include "../../backends/getrs_route.hh"
#include "../../backends/getri_route.hh"
#include "../../extensions/getrf_native.hh"
#include "../../extensions/getrs_native.hh"
#include "../../extensions/getri_native.hh"

// orgqr's native arm is ormqr against an identity, applied through the router.
#include <batchlas/blas/functions/ormqr.hh>

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

// potrf uses solver_vendor_available<B> (cuSOLVER); geqrf/orgqr and the LU family
// use factorization_vendor_available<B> (cuBLAS). The two differ on CUDA and are
// NOT interchangeable -- swapping one also changes which builds get the entry
// point. A latent defect in that gate is recorded in docs/design/known-defects.md.

// One diagnostic per op, so the call and the buffer-size query cannot drift into
// two different messages. Each THROWS rather than falling through to the vendor,
// which would silently keep taking it the day a native capability comes off zero.
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
    // Validation first: it must precede the shape builder, which reads
    // A.rows()/A.cols(). Deliberately only a negative-extent test -- see geqrf.hh.
    geqrf_validate_params<T>(A);

    // Resolved before the vendor-available test, so a vendor-free build routes
    // natively instead of falling into the `if constexpr` at the end of this body.
    const dispatch::Route route = backend::geqrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_geqrf::geqrf_cta_dispatch<T>(ctx, A, tau, work_space);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // The trailing GEMM goes through the ROUTER: a direct
            // sycl_gemm::gemm_custom call bypasses RouteTable<Op::gemm> and takes
            // the native kernel even on the shapes it loses.
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
    // The query mirrors the call exactly -- same validator, builder, route
    // function and arguments -- so both resolve to the same Route by construction.
    geqrf_validate_params<T>(A);

    const dispatch::Route route = backend::geqrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max over EVERY supported native tier and the vendor, not the chosen route:
    // query and call resolve independently, so a chosen-only size under-allocates
    // where max() merely over-allocates. Safe only because every term is an
    // alignment multiple -- the native queries replay their layout through
    // BumpAllocator::measuring() rather than hand-summing. A vendor -> native
    // change between the two is left throwing from BumpAllocator::allocate.
    //
    // geqrf ONLY: band_reduction.cc sizes against an (m_max x nb_max) dummy view
    // and calls with a smaller sub-view, so any native geqrf_*_buffer_size must be
    // MONOTONE NON-DECREASING in (rows, cols, batch) and must never dereference
    // A.data_ptr() or tau.data() -- both are nullptr there.
    //
    // `native_fired`, not `native_need != 0`: the CTA tier's workspace is
    // legitimately zero, so the consistency check cannot be read off the size.
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
            // supports() accepted something but no query fired: a bug here.
            geqrf_throw_native_unimplemented<T>(route, "geqrf_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
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

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::Blocked) {
            // The apply goes through the ROUTER, and its SIZE is injected from
            // the same routed query the call uses so the two cannot resolve
            // differently. Argument order is the positional ormqr entry point's.
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
    // Same validator, builder, route function and arguments as the call above.
    orgqr_validate_params<T>(A);

    const dispatch::Route route = backend::orgqr_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor) and `native_fired` rather than `native_need != 0` -- see
    // geqrf_buffer_size; a zero workspace is a legitimate answer here too.
    // evidence: docs/perf/qr.md#the-orgqr_buffer_size-latent-defect
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::orgqr_op_shape<B, T>(ctx, A);
        using Tbl = dispatch::RouteTable<dispatch::Op::orgqr, T>;
        if (shape &&
            Tbl::supports({dispatch::Origin::Native, dispatch::Algorithm::Blocked}, *shape)) {
            // The apply's size comes from the SAME routed query the call uses.
            // `tau` travels with it only because ormqr_buffer_size validates
            // tau.size() >= k * batch; the C view built for it is a null one.
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
            orgqr_throw_native_unimplemented<T>(route, "orgqr_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
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

// The LU family. preferred() is all-false in all three route tables, so a
// vendor-present build always resolves {Vendor, Auto} while a vendor-free build
// takes the native arm for every square shape.
// evidence: docs/perf/lu.md#the-shipped-preferred-windows

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
    // Must precede the shape builder, which reads A.rows()/A.cols().
    getrf_validate_params<T>(A);

    const dispatch::Route route = backend::getrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_getrf::getrf_cta_dispatch<T>(ctx, A, pivots, work_space, info);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // The trailing GEMM and the panel TRSM go through the ROUTER: a direct
            // native call bypasses RouteTable and takes the native kernel even on
            // shapes it loses. trsm's alpha comes THIRD (functions/trsm.hh); the
            // old order is a deleted overload, so a stale call cannot compile.
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
    // The query mirrors the call exactly, so both resolve to the same Route.
    getrf_validate_params<T>(A);

    const dispatch::Route route = backend::getrf_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max over every supported native tier and the vendor, and `native_fired`
    // rather than `native_need != 0` -- see geqrf_buffer_size for both.
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
            getrf_throw_native_unimplemented<T>(route, "getrf_buffer_size");
        }
    }

    if constexpr (!dispatch::factorization_vendor_available<B>) {
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
    getrs_validate_params<T>(A, B);

    const dispatch::Route route = backend::getrs_route<Back, T>(
        ctx, A, B, transA,
        /*vendor_available=*/dispatch::factorization_vendor_available<Back>);

    // Two native tiers. In a vendor-free build native_tier_preferred sends every
    // shape the fused tier can hold to Algorithm::CTA and the rest to Blocked.
    // evidence: docs/perf/lu.md#getrs-fused-window-evidence
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            // The fused tier injects NOTHING: the permutation and both
            // substitutions are one kernel, so it has no seam.
            return sycl_getrs::getrs_fused_dispatch<T>(
                ctx, A, B, transA, pivots, work_space);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // Both triangular solves go through the ROUTER, and injection is the
            // only way to reach trsm<Back,T> from a TU with no Backend parameter.
            // No buffer-size twin is needed: the public trsm takes no workspace.
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
    // The query mirrors the call -- SAME ARGUMENTS, including transA, which is a
    // live routing input here and would split the two resolutions if dropped.
    getrs_validate_params<T>(A, B);

    const dispatch::Route route = backend::getrs_route<Back, T>(
        ctx, A, B, transA,
        /*vendor_available=*/dispatch::factorization_vendor_available<Back>);

    // max(native, vendor); `native_fired`, not a zero size -- see geqrf_buffer_size.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        const auto shape = backend::getrs_op_shape<Back, T>(ctx, A, B, transA);
        using Tbl = dispatch::RouteTable<dispatch::Op::getrs, T>;
        if (shape) {
            // max over every native tier that could serve this shape, not the one
            // the route named: a lease sized for one tier is a lease the other can
            // overrun. Both are 0 today, which is why the max is written now.
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
    // Validation takes C here; the query below validates A alone, because
    // getri_buffer_size has no C and the route is a function of A alone.
    getri_validate_params<T>(A, C);

    const dispatch::Route route = backend::getri_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::Blocked) {
            // Both triangular solves go through the ROUTER. The row permutation
            // is NOT injected: P is written straight into C rather than permuting
            // an identity, so there is no second routed op and no workspace.
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
    // The query mirrors the call with A alone in both places -- which is why
    // backend::getri_op_shape takes A alone.
    //
    // THIS QUERY RUNS UNDER BumpAllocator::measuring() (inv.cc replays inv_layout
    // through it): everything reachable from here must be pure with respect to the
    // workspace -- no read, no write, no kernel launch -- and must not dereference
    // A.data_ptr().
    getri_validate_params<T>(A);

    const dispatch::Route route = backend::getri_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // max(native, vendor); `native_fired`, not a zero size -- the native arm's
    // workspace is expected to be zero.
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
    potrf_validate_params<T>(descrA, uplo);

    // solver_vendor_available, NOT factorization_vendor_available -- see the file
    // header. Resolved before the vendor-available test, as above.
    const dispatch::Route route = backend::potrf_route<B, T>(
        ctx, descrA, uplo,
        /*vendor_available=*/dispatch::solver_vendor_available<B>);

    // preferred() is all-false, so a vendor-present build enters this block only
    // when a caller pins BATCHLAS_POTRF_ROUTE; a vendor-free build takes any
    // supported native route. evidence: docs/perf/potrf.md#preferred-is-false-everywhere
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_potrf::potrf_cta_dispatch<T>(ctx, descrA, uplo, workspace, info_out);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // The trailing GEMM and the panel TRSM go through the ROUTER: a direct
            // native call bypasses RouteTable and takes the native kernel even on
            // shapes it loses. Injection is also the only way to reach trsm<B,T>
            // from a TU with no Backend parameter.
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
    // The query mirrors the call exactly, so both resolve to the same Route.
    potrf_validate_params<T>(A, uplo);

    const dispatch::Route route = backend::potrf_route<B, T>(
        ctx, A, uplo,
        /*vendor_available=*/dispatch::solver_vendor_available<B>);

    // max over EVERY supported native tier and the vendor, not the chosen route --
    // see geqrf_buffer_size; a CTA-sized answer against a Blocked call is kilobytes
    // against tens of megabytes. Unlike the ops above, potrf reads the consistency
    // check off `native_need == 0` rather than a fired flag, which is safe only
    // while both tiers have a non-zero workspace.
    // evidence: docs/perf/qr.md#the-orgqr_buffer_size-latent-defect
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
            potrf_throw_native_unimplemented<T>(route, "potrf_buffer_size");
        }
    }

    if constexpr (!dispatch::solver_vendor_available<B>) {
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

// Explicit instantiations, one block per backend whose vendor TU is compiled.

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

// Keyed on the DEVICE FAMILY, not on the vendor library: the bodies above compile
// to a throw when the library is absent, so the public entry point exists as a
// symbol in every build that has the device.
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
