// The public sparse entry points, defined once, outside every vendor TU.
//
// Same move and same reason as entry_points/level3.cc: spmm was DEFINED in
// cusparse.cc, netlib_lapack.cc and rocsparse.cc, so dropping a vendor library
// dropped the public entry point along with the vendor path.
//
// spmm carries a third template parameter (MatrixFormat), which is why its
// instantiations are spelled out here rather than going through the shared
// per-type macros the dense facades use.
//
// WP8 puts the route resolution in between. spmm moves TOGETHER WITH ITS
// BUFFER-SIZE QUERY -- same builder, same *_route call, SAME ARGUMENTS -- for
// the reason factorization.cc:8-9 records: splitting them would let the two
// resolve differently, which is the defect class S4d found in ormqr (buffer
// size 2560 bytes, call demanded 276480).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/spmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// WP8: the native SpMM arm. Same shape as level3.cc:38-39's gemv pair -- the
// adapter reaches only public headers plus src/sycl/spmm_native.hh, so the
// vendor-free facade can include it. route_spmm.hh arrives transitively through
// the adapter, but this file names RouteTable<Op::spmm, T> directly in the
// sizing path, so it is included in its own right.
#include <batchlas/blas/dispatch/route_spmm.hh>

#include "../../backends/spmm_route.hh"
#include "../../sycl/spmm_native.hh"

#include "../../util/template-instantiations.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace batchlas {

// The diagnostic for a native route the build cannot service. There is NO
// dispatch::throw_native_unimplemented: the convention is one file-local
// [[noreturn]] helper per op, defined in the facade TU that owns the op --
// geqrf_throw_native_unimplemented (entry_points/factorization.cc:114), orgqr_
// (:127), getrf_ (:524), getrs_ (:536), getri_ (:548), potrf_ (:1008). One
// function per op so the call and the buffer-size query cannot drift into two
// different messages.
//
// IT IS A THROW RATHER THAN A FALL-THROUGH TO THE VENDOR, which is the whole
// point of it. Omitting the native branch would silently take the vendor at the
// exact moment a capability first comes off zero -- a kernel LINKED but never
// REACHED, and a test suite passing green over it. route_compiled.hh:1-24 names
// that defect class, and WP5's break B9 measured it: deleting geqrf's native arm
// turned NOTHING red anywhere, in either build.
template <typename T>
[[noreturn]] inline void spmm_throw_native_unimplemented(dispatch::Route route,
                                                         const char* who) {
    throw std::logic_error(
        std::string(who) + ": resolved to a native route (" +
        std::string(dispatch::to_string(route.origin)) + ":" +
        std::string(dispatch::to_string(route.algo)) +
        ") but no native spmm kernel is linked into this build. "
        "sycl_spmm::spmm_gather_available / spmm_scatter_available reported a "
        "capability the facade cannot service.");
}

template <Backend B, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx,
           const MatrixView<T, MFormat>& A,
           const MatrixView<T, MatrixFormat::Dense>& B_mat,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           Span<std::byte> workspace) {
    // THE ROUTE IS RESOLVED BEFORE THE VENDOR-AVAILABLE TEST. Everything below
    // the `if constexpr` at the bottom of this body is UNREACHABLE in the
    // vendor-free build, which is the build this campaign exists for -- and for
    // spmm that build is not hypothetical: build-novendor has
    // BATCHLAS_HAS_HOST_BACKEND 1 with BATCHLAS_HAS_LAPACKE and
    // BATCHLAS_HAS_CBLAS both 0, so SPMM_ALL(Backend::NETLIB) below IS
    // instantiated and every one of those calls throws NoRouteError today.
    //
    // NO VALIDATION CALL IS HOISTED HERE, unlike trsm and the LU family. spmm
    // has never had a spmm_validate_params and WP8 deliberately does not add
    // one: the native kernel must accept exactly what the vendor accepts, and a
    // new throw would turn today's silent bugs into crashes in live paths. The
    // agreement checks live in the shape builder instead, where their answer is
    // "hand it to the vendor" (spmm_route.hh:35-44).
    const dispatch::Route route = backend::spmm_route<B, T, MFormat>(
        ctx, A, B_mat, C, transA, transB,
        /*vendor_available=*/dispatch::sparse_vendor_available<B>);

    // ROUTE NEUTRALITY, AND WHY IT IS GUARANTEED RATHER THAN HOPED FOR. In a
    // vendor-present build with BATCHLAS_SPMM_ROUTE unset, legacy_unset_default
    // returns {Auto, Auto} for every op unconditionally (route_env.hh:145-148),
    // so resolve_route enters `automatic()` (route_resolve.hh:109). Its first
    // pass needs supports() && preferred(), and RouteTable<Op::spmm, T>::
    // preferred() is false for every route, every type and every shape
    // (route_spmm.hh:271). Its two remaining native passes sit inside
    // `if (!vendor_available)` (route_resolve.hh:113-127) and are therefore
    // never entered. What is left is `return Route{Origin::Vendor,
    // Algorithm::Auto}` at :129, so this block is skipped and the call reaches
    // backend::spmm_vendor with exactly the arguments it received before WP8 --
    // byte-identical routing to today, and provable as a coverage diff rather
    // than as an argument.
    if (dispatch::is_native(route)) {
        // Only CSR has native bodies, and supports() refuses every other format
        // (route_spmm.hh, gate 1) -- including on a FORCED route, which bypasses
        // preferred() but never supports() (route_resolve.hh:165). So this
        // `if constexpr` cannot hide a reachable arm; it exists because the
        // template is written for a format parameter the kernel does not take.
        // Inside it MatrixView<T, MFormat> IS MatrixView<T, MatrixFormat::CSR>,
        // so no cast or adapter is needed.
        if constexpr (MFormat == MatrixFormat::CSR) {
            if (route.algo == dispatch::Algorithm::Direct) {
                // ONE ROUTE, THREE BODIES. The launcher picks the gather (body
                // 1) or the scale + atomic scatter pair (bodies 0 + 2) on
                // transA, exactly as {Native, Direct} names both of gemv's
                // direct kernels. Body selection is a decomposition, not an
                // algorithm, which is why no new Algorithm enumerator ships and
                // why to_string(Algorithm) and parse_algorithm_word need no
                // edit.
                //
                // NO WORKSPACE ARGUMENT. The native tier allocates nothing, and
                // that is what makes spmm_buffer_size agree with this call by
                // construction rather than by a comment (spmm_native.hh:27).
                return sycl_spmm::spmm_native_csr<T>(ctx, A, B_mat, C, alpha,
                                                     beta, transA, transB);
            }
        }
        // Kept as the trailing default: Algorithm::Auto, and any future native
        // tier, must not fall through to the vendor.
        spmm_throw_native_unimplemented<T>(route, "spmm");
    }

    if constexpr (!dispatch::sparse_vendor_available<B>) {
        // Still honest about the gap. What reaches here in a vendor-free build
        // is exactly what supports() refuses: a non-CSR format, a heterogeneous
        // dense operand, a negative extent, an empty batch, or a set of views
        // whose lengths do not describe one spmm (the shape builder's nullopt,
        // which resolves to {Vendor, Auto}). Those have no route at all without
        // a vendor, and they say so by name rather than dying downstream.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::spmm, B, dispatch::kSparseLibrary<B>);
    } else {
        return backend::spmm_vendor<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB, workspace);
    }
}

template <Backend B, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx,
                        const MatrixView<T, MFormat>& A,
                        const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C,
                        T alpha,
                        T beta,
                        Transpose transA,
                        Transpose transB) {
    // THE QUERY MOVES WITH THE CALL: same builder, same *_route function, SAME
    // ARGUMENTS in the same order, therefore the same SpmmShape and the same
    // Route BY CONSTRUCTION rather than by a comment asking for it. alpha and
    // beta are not among those arguments in either place -- neither the shape
    // nor the route depends on a scalar value -- so the two resolutions cannot
    // diverge on them either; they are forwarded only to the vendor query below,
    // exactly as before.
    const dispatch::Route route = backend::spmm_route<B, T, MFormat>(
        ctx, A, B_mat, C, transA, transB,
        /*vendor_available=*/dispatch::sparse_vendor_available<B>);

    // max(native, vendor), NOT "whatever the chosen route needs", and taken over
    // every SUPPORTED native tier rather than over the one THIS resolution
    // chose. Both halves of the argument are potrf's and getrf's and transfer
    // verbatim: a chosen-only size turns a query/call disagreement into an
    // UNDER-allocation, which is the ormqr failure mode, while max() turns it
    // into a harmless over-allocation.
    //
    // `native_fired` IS NOT `native_need != 0`, and for spmm that difference is
    // not hypothetical the way it is elsewhere -- it is the normal case: the
    // native tier's need is EXACTLY ZERO for every shape and every type. Reading
    // the internal-consistency check off the size would make this function throw
    // on every call the route table had just promised. orgqr_buffer_size shipped
    // with precisely that latent defect and it was fixed in the WP5 repair pass.
    //
    // NOTHING BELOW TOUCHES DEVICE MEMORY. spmm_op_shape reads only the views'
    // int metadata (spmm_route.hh:46-52) -- never data_ptr(), never
    // row_offsets()[k], never A.nnz(b), which matrix.hh:1081-1083 says is not
    // host-reachable for a malloc_device-backed view. A sizing path that
    // dereferenced would be an immediate segfault rather than a wrong route, and
    // returning 0 without reading device memory is the property the whole
    // zero-workspace story rests on.
    std::size_t native_need = 0;
    bool native_fired = false;
    if (dispatch::is_native(route)) {
        if constexpr (MFormat == MatrixFormat::CSR) {
            const auto shape = backend::spmm_op_shape<B, T, MFormat>(
                ctx, A, B_mat, C, transA, transB);
            using Tbl = dispatch::RouteTable<dispatch::Op::spmm, T>;
            if (shape && Tbl::supports({dispatch::Origin::Native,
                                        dispatch::Algorithm::Direct}, *shape)) {
                // ZERO, AS A NAMED CONSTANT FOLDED THROUGH THE SAME max() every
                // other facade uses, so a second native tier is a one-line
                // addition here rather than a restructuring. It is a literal
                // rather than a query because spmm_native_csr takes no
                // Span<std::byte> at all -- there is no sizing function that
                // could disagree with the launcher, which is invariant 2 of
                // src/sycl/spmm_native.hh:27.
                constexpr std::size_t kSpmmNativeDirectNeed = 0;
                native_need = std::max(native_need, kSpmmNativeDirectNeed);
                native_fired = true;
            }
        }
        if (!native_fired) {
            // is_native(route) says supports() accepted SOMETHING; if the query
            // above did not fire, the two disagree and that is a bug in this
            // file rather than a shape the caller can fix.
            spmm_throw_native_unimplemented<T>(route, "spmm_buffer_size");
        }
    }

    if constexpr (!dispatch::sparse_vendor_available<B>) {
        // A vendor-free build with no native route left is the NoRouteError this
        // work package exists to remove; with one, the native term is the whole
        // answer -- and it is 0, which is why the caller's BumpAllocator asks for
        // nothing. GATED ON native_fired, not on native_need != 0, for the
        // reason above.
        if (!native_fired) {
            dispatch::throw_no_vendor_route<T>(
                dispatch::Op::spmm, B, dispatch::kSparseLibrary<B>);
        }
        return native_need;
    } else {
        // With the environment unset this build resolves to {Vendor, Auto}, so
        // native_need is still 0 and the vendor term is returned unchanged --
        // the sizing half of the route-neutrality claim above.
        return std::max(native_need,
                        backend::spmm_vendor_buffer_size<B, T, MFormat>(ctx, A, B_mat, C, alpha, beta, transA, transB));
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define SPMM_ONE(B_, fp, F)                                            \
    BATCHLAS_INSTANTIATE(sig::spmm<fp BATCHLAS_COMMA F>, spmm, B_, fp, F) \
    BATCHLAS_INSTANTIATE(sig::spmm_buffer_size<fp BATCHLAS_COMMA F>, spmm_buffer_size, B_, fp, F)

// CSR is the only sparse format any backend instantiates today.
#define SPMM_ALL(B_)                                    \
    SPMM_ONE(B_, float, MatrixFormat::CSR)              \
    SPMM_ONE(B_, double, MatrixFormat::CSR)             \
    SPMM_ONE(B_, std::complex<float>, MatrixFormat::CSR)\
    SPMM_ONE(B_, std::complex<double>, MatrixFormat::CSR)

// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
SPMM_ALL(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SPMM_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SPMM_ALL(Backend::NETLIB)
#endif

#undef SPMM_ALL
#undef SPMM_ONE

}  // namespace batchlas
