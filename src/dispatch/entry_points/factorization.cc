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

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

template <Backend B, typename T>
Event geqrf(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> work_space) {
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
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::geqrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::geqrf_vendor_buffer_size<B, T>(ctx, A, tau);
    }
}

template <Backend B, typename T>
Event orgqr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<T> tau,
            Span<std::byte> workspace) {
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
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::orgqr, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::orgqr_vendor_buffer_size<B, T>(ctx, A, tau);
    }
}

template <Backend B, typename T>
Event getrf(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
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
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrf, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getrf_vendor_buffer_size<B, T>(ctx, A);
    }
}

template <Backend Back, typename T>
Event getrs(Queue& ctx,
            const MatrixView<T,MatrixFormat::Dense>& A,
            const MatrixView<T,MatrixFormat::Dense>& B,
            Transpose transA,
            Span<int64_t> pivots,
            Span<std::byte> work_space) {
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
    if constexpr (!dispatch::factorization_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getrs, Back, dispatch::kFactorizationLibrary<Back>);
    } else {
        return backend::getrs_vendor_buffer_size<Back, T>(ctx, A, B, transA);
    }
}

template <Backend B, typename T>
Event getri(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& C,
            Span<int64_t> pivots,
            Span<std::byte> work_space,
            Span<int32_t> info) {
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
    if constexpr (!dispatch::factorization_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::getri, B, dispatch::kFactorizationLibrary<B>);
    } else {
        return backend::getri_vendor_buffer_size<B, T>(ctx, A);
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
