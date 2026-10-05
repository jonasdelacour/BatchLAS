// The public factorization entry points -- getrs, getri and gesv (posv, geqrf, orgqr and
// getrf are in src/ops/<op>/) -- defined once here rather than inside a vendor TU, so
// dropping a vendor library does not drop the public symbol.
// See docs/design/vendor-independence.md#the-entry-point-facade.
//
// Each op MUST stay next to its buffer-size query: separated, the two can resolve
// differently and the workspace is then under-allocated.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/potrf.hh>

// P2: the two ops with no vendor arm anywhere. Their composed routes are built from
// the entry points above, in this file, because only this layer can name a route for
// getrf / getrs / potrf / trsm.
#include <batchlas/blas/functions/gesv.hh>

// The routed ops the blocked drivers inject: their kernel TUs carry no Backend
// parameter, so only this layer can name a route for them.
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// Routing adapters and native drivers: each is a src/ header over public includes
// only, so the facade can include it in a vendor-free build.
#include "../../backends/getrs_route.hh"
#include "../../backends/getri_route.hh"
#include "../../backends/gesv_route.hh"
#include "../../extensions/solve_native.hh"
#include "../../extensions/getrf_native.hh"
#include "../../extensions/getrs_native.hh"
#include "../../extensions/getri_native.hh"

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

// The LU family uses factorization_vendor_available<B> (cuBLAS); potrf
// (src/ops/potrf/potrf.cc) uses solver_vendor_available<B> (cuSOLVER). The two differ on
// CUDA and are NOT interchangeable -- swapping one also changes which builds get the
// entry point.
// A latent defect in that gate: docs/design/known-defects.md.

// These THROW rather than falling through to the vendor, which would silently keep
// taking the vendor the day a native capability comes off zero.
template <typename T>
[[noreturn]] inline void getrs_throw_native_unimplemented(dispatch::Route route,
                                                          const char* who) {
    throw batchlas::internal_error(
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
    throw batchlas::internal_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native getri driver is linked into this build. "
        "sycl_getri::getri_blocked_available reported a capability the facade "
        "cannot service.");
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

    // Two native tiers: in a vendor-free build native_tier_preferred sends every shape
    // the fused tier can hold to Algorithm::CTA and the rest to Blocked.
    // evidence: docs/perf/lu.md#getrs-fused-window-evidence
    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            // The fused tier injects nothing: the permutation and both
            // substitutions are one kernel, so it has no seam.
            return sycl_getrs::getrs_fused_dispatch<T>(
                ctx, A, B, transA, pivots, work_space);
        }
        if (route.algo == dispatch::Algorithm::Blocked) {
            // Both solves go through the ROUTER; injection is the only way to reach
            // trsm<Back,T> here. No size twin: the public trsm takes no workspace.
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
            // Both solves go through the ROUTER. The permutation is NOT injected:
            // P is written straight into C rather than permuting an identity, so
            // there is no second routed op and no workspace.
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
    // THIS QUERY RUNS UNDER BumpAllocator::measuring() (inv.cc replays inv_layout
    // through it): everything reachable from here must be pure with respect to the
    // workspace -- no read, no write, no kernel launch -- and must not dereference
    // A.data_ptr().
    getri_validate_params<T>(A);

    const dispatch::Route route = backend::getri_route<B, T>(
        ctx, A,
        /*vendor_available=*/dispatch::factorization_vendor_available<B>);

    // `native_fired`, not a zero size: the native arm's workspace is expected
    // to be zero.
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

// ---------------------------------------------------------------------------
// P2: gesv. With posv (now src/ops/posv/), THE OPS WITH NO VENDOR ARM ON ANY BACKEND,
// so gesv has no `*_vendor` declaration, does not read factorization_vendor_available,
// and cannot fall back the way every other op in this file does. Its composed arm,
// `getrf; getrs`, is assembled here out of the routed entry points above, which is
// also why the composition lives in the facade rather than in a driver TU.
// evidence: docs/perf/lu.md#p2-the-window-this-tier-expects

template <typename T>
[[noreturn]] inline void solve_throw_unroutable(dispatch::Route route, const char* who) {
    throw batchlas::internal_error(
        std::string(who) + ": resolved to " +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ", which names no implementation. This op has no vendor arm, so a "
        "{vendor, auto} answer means the shape builder returned nullopt for a pair "
        "the validator had already accepted -- a routing defect, not a missing "
        "library.");
}

// The workspace split, and why it is safe to cut the caller's span at the first
// leg's reported size: BumpAllocator's sizing results are alignment multiples and a
// real pool's base is device-aligned, so the tail sub-span is device-aligned too
// (mempool.hh). Sharing ONE span between the two legs would also work on an
// in-order queue, but only by an argument about kernel ordering; this one needs no
// such argument.
template <Backend Back, typename T>
Event gesv(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           Span<int64_t> pivots,
           Span<std::byte> work_space,
           Span<int32_t> info) {
    gesv_validate_params<T>(A, B);

    const dispatch::Route route = backend::gesv_route<Back, T>(ctx, A, B);

    if (route.algo == dispatch::Algorithm::Tiny) {
        return sycl_gesv::gesv_tiny_dispatch<T>(ctx, A, B, pivots, work_space, info);
    }
    if (route.algo == dispatch::Algorithm::Blocked) {
        const std::size_t split = getrf_buffer_size<Back, T>(ctx, A);
        if (work_space.size() < split) {
            throw batchlas::workspace_error(
                "gesv: workspace is shorter than the composed route's getrf leg needs");
        }
        (void)getrf<Back, T>(ctx, A, pivots, work_space.subspan(0, split), info);
        return getrs<Back, T>(ctx, A, B, Transpose::NoTrans, pivots,
                              work_space.subspan(split));
    }
    solve_throw_unroutable<T>(route, "gesv");
}

template <Backend Back, typename T>
size_t gesv_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B) {
    // The query mirrors the call: SAME ARGUMENTS, so the two reach the same route by
    // construction rather than by a comment asking for it.
    gesv_validate_params<T>(A, B);

    const dispatch::Route route = backend::gesv_route<Back, T>(ctx, A, B);

    if (route.algo == dispatch::Algorithm::Tiny) {
        return sycl_gesv::gesv_tiny_buffer_size<T>(ctx, A, B);
    }
    if (route.algo == dispatch::Algorithm::Blocked) {
        // A SUM, not a max: the two legs hold disjoint halves of the span.
        return getrf_buffer_size<Back, T>(ctx, A) +
               getrs_buffer_size<Back, T>(ctx, A, B, Transpose::NoTrans);
    }
    solve_throw_unroutable<T>(route, "gesv_buffer_size");
}

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

#define FACTORIZATION_ONE(B_, fp)              \
    OP_INSTANTIATE(getrs, B_, fp)              \
    OP_INSTANTIATE(getrs_buffer_size, B_, fp)  \
    OP_INSTANTIATE(getri, B_, fp)              \
    OP_INSTANTIATE(getri_buffer_size, B_, fp)

#define FACTORIZATION_ALL(B_)                       \
    FACTORIZATION_ONE(B_, float)                    \
    FACTORIZATION_ONE(B_, double)                   \
    FACTORIZATION_ONE(B_, std::complex<float>)      \
    FACTORIZATION_ONE(B_, std::complex<double>)

// P2's gesv. A separate driver from FACTORIZATION_ALL only because it landed
// separately; it has the same four-type domain.
#define SOLVE_ONE(B_, fp)                      \
    OP_INSTANTIATE(gesv, B_, fp)               \
    OP_INSTANTIATE(gesv_buffer_size, B_, fp)

#define SOLVE_ALL(B_)                               \
    SOLVE_ONE(B_, float)                            \
    SOLVE_ONE(B_, double)                           \
    SOLVE_ONE(B_, std::complex<float>)              \
    SOLVE_ONE(B_, std::complex<double>)

// Keyed on the DEVICE FAMILY, not on the vendor library: the bodies above compile to
// a throw when the library is absent, so the symbol exists in every build with the
// device.
#if BATCHLAS_HAS_CUDA_BACKEND
FACTORIZATION_ALL(Backend::CUDA)
SOLVE_ALL(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
FACTORIZATION_ALL(Backend::ROCM)
SOLVE_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
FACTORIZATION_ALL(Backend::NETLIB)
SOLVE_ALL(Backend::NETLIB)
#endif

#undef SOLVE_ALL
#undef SOLVE_ONE
#undef FACTORIZATION_ALL
#undef FACTORIZATION_ONE
#undef OP_INSTANTIATE

}  // namespace batchlas
