// The public gesv entry points (posv, geqrf, orgqr, getrf, getrs and getri are in
// src/ops/<op>/) -- defined once here rather than inside a vendor TU, so
// dropping a vendor library does not drop the public symbol.
// See docs/design/vendor-independence.md#the-entry-point-facade.
//
// Each op MUST stay next to its buffer-size query: separated, the two can resolve
// differently and the workspace is then under-allocated.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
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
#include "../../backends/gesv_route.hh"
#include "../../extensions/solve_native.hh"
#include "../../extensions/getrf_native.hh"

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

// ---------------------------------------------------------------------------
// P2: gesv. With posv (now src/ops/posv/), THE OPS WITH NO VENDOR ARM ON ANY BACKEND,
// so gesv has no `*_vendor` declaration, does not read factorization_vendor_available,
// and cannot fall back the way every other op in this file does. Its composed arm,
// `getrf; getrs`, is assembled here out of the public entry points, which is
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
SOLVE_ALL(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SOLVE_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SOLVE_ALL(Backend::NETLIB)
#endif

#undef SOLVE_ALL
#undef SOLVE_ONE
#undef OP_INSTANTIATE

}  // namespace batchlas
