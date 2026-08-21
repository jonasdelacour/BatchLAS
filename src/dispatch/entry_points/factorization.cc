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

    // UNREACHABLE TODAY, and deliberately written anyway:
    // RouteTable<Op::potrf,T>::supports() returns false for both native routes
    // while sycl_potrf::potrf_cta_max_n<T>() == 0 and
    // potrf_blocked_available<T>() == false, and route_resolve.hh:101 and :62
    // both gate on supports().
    // THE NATIVE ARM IS REACHED, not merely linked. route_compiled.hh:1-24 names
    // the defect class where a kernel is compiled into the library and no call
    // ever arrives at it; the throw below is what makes the Blocked half of that
    // impossible to ship silently.
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_potrf::potrf_cta_dispatch<T>(ctx, descrA, uplo, workspace, info_out);
        }
        // Algorithm::Blocked is Phase 2 and is not written. It is unreachable
        // today for the same reason it was before: potrf_blocked_available<T>()
        // is false, so RouteTable<Op::potrf,T>::supports() rejects that arm and
        // route_resolve.hh:101 / :62 both gate on supports().
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
    std::size_t native_need = 0;
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            native_need = sycl_potrf::potrf_cta_buffer_size<T>(ctx, A);
        } else {
            // Phase 2, unreachable: potrf_blocked_available<T>() is false.
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
