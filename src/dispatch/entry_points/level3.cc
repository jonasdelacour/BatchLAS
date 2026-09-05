// The public level-3 entry points, defined once, outside every vendor TU.
//
// Vendor TUs define backend::<op>_vendor<B, T>; this file defines and explicitly
// instantiates the public <op><B, T> that routes to it. Keeping the public
// definitions out of the vendor TUs is what lets the API link in a build with no
// vendor library. See docs/design/vendor-independence.md#the-entry-point-facade.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "../../backends/trsm_route.hh"
#include "../../backends/gemv_route.hh"
#include "../../sycl/gemv_native.hh"
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/trmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#include "../../backends/gemm_variant.hh"
#include "../../backends/gemm_heterogeneous.hh"
#include "../../sycl/gemm_kernels.hh"

// The four level-3 custom-route gates. They have to run before the
// vendor-available test, so they live here rather than in cublas.cc.
#include "../../backends/symm_custom_dispatch.hh"
#include "../../backends/syrk_custom_dispatch.hh"
#include "../../backends/syr2k_custom_dispatch.hh"
#include "../../backends/trmm_custom_dispatch.hh"
#include "../../backends/level3_coverage.hh"


#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

template <Backend Back, typename T>
Event gemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           ComputePrecision precision) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        // A heterogeneous batch is handled BEFORE routing -- no strided-batched
        // call can serve members of differing shape. The recursive call below is
        // safe: each batch_item() is homogeneous by construction and cannot
        // re-enter this branch.
        if (backend::gemm_has_heterogeneous_batch(A, B, C)) {
            return backend::detail::gemm_heterogeneous_loop<T>(
                ctx, A, B, C, beta, transA, transB,
                [&](const MatrixView<T, MatrixFormat::Dense>& a,
                    const MatrixView<T, MatrixFormat::Dense>& b,
                    const MatrixView<T, MatrixFormat::Dense>& c) {
                    return ::batchlas::gemm<Back, T>(ctx, a, b, c, alpha, beta,
                                                     transA, transB, precision);
                });
        }

        // Not a second routing policy: backend::gemm_route is the adapter
        // cublas.cc consults, over the same RouteTable<Op::gemm, T>, with
        // vendor_available = false as the only differing input.
        const auto route = backend::gemm_route<T>(ctx, A, B, C, transA, transB,
                                                  precision, /*vendor_available=*/false);
        if (dispatch::is_native(route)) {
            return sycl_gemm::gemm_custom<T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }
        // Shapes the native kernel does not serve (non-Default precision,
        // degenerate dims) have no route at all without a vendor.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }
}

template <Backend Back, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA) {
    // The gate runs BEFORE the vendor-available test: anything below that test
    // is unreachable in the vendor-free build.
    //
    // No validation call is hoisted here, unlike trsm. The native kernel must
    // accept exactly what the vendor accepts, and a new throw would turn an
    // existing silent bug into a crash in a live path (docs/design/known-defects.md,
    // defect 1).
    const dispatch::Route route = backend::gemv_route<Back, T>(
        ctx, A, X, Y, transA,
        /*vendor_available=*/dispatch::level3_vendor_available<Back>);

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_gemv::gemv_native_cta<T>(ctx, A, X, Y, alpha, beta, transA);
        }
        if (route.algo == dispatch::Algorithm::Direct) {
            return sycl_gemv::gemv_native_direct<T>(ctx, A, X, Y, alpha, beta, transA);
        }
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        // What reaches here is what supports() refuses: a heterogeneous A, a
        // negative extent, an empty batch, or views that do not describe one gemv.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemv, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::gemv_vendor<Back, T>(ctx, A, X, Y, alpha, beta, transA);
    }
}

template <Backend Back, typename T>
Event trsm(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const MatrixView<T,MatrixFormat::Dense>& B,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag) {
    // Validation first, hoisted here so every backend gets it -- netlib_lapack.cc
    // never called trsm_validate_params. It must precede the shape builder, which
    // reads A.rows()/B.rows()/B.cols() and would index a non-conforming shape.
    trsm_validate_params(A, B, side, uplo, transA, diag);

    // The gate runs BEFORE the vendor-available test: anything below that test
    // is unreachable in the vendor-free build.
    const dispatch::Route route = backend::trsm_route<T>(
        ctx, A, B, side, uplo, transA, diag,
        /*vendor_available=*/dispatch::level3_vendor_available<Back>);

    {
        if (dispatch::is_native(route)) {
            if (route.algo == dispatch::Algorithm::CTA) {
                return sycl_trsm::trsm_native_v1_dispatch<T>(
                    ctx, A, B, alpha, side, uplo, transA, diag);
            }
            if (route.algo == dispatch::Algorithm::Blocked) {
                // The trailing update goes through the ROUTER, not straight to
                // sycl_gemm::gemm_custom: every operand trsm issues is a sub-view
                // carrying its parent's leading dimension, and the native GEMM
                // collapses on a strided ld where the vendor does not.
                // evidence: docs/perf/gemm.md#the-strided-ld-defect-and-the-routing-fix
                // Injected as a lambda so the kernel TU stays free of dispatch.
                return sycl_trsm::trsm_native_blocked<T>(
                    ctx, A, B, alpha, side, uplo, transA, diag,
                    [](Queue& c,
                       const MatrixView<T, MatrixFormat::Dense>& ga,
                       const MatrixView<T, MatrixFormat::Dense>& gb,
                       const MatrixView<T, MatrixFormat::Dense>& gc,
                       T galpha, T gbeta, Transpose gta, Transpose gtb,
                       ComputePrecision gp) {
                        return gemm<Back, T>(c, ga, gb, gc, galpha, gbeta,
                                             gta, gtb, gp);
                    });
            }
        }
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::trsm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::trsm_vendor<Back, T>(ctx, A, B, side, uplo, transA, diag, alpha);
    }
}

template <Backend Back, RealScalar T>
Event symm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::symm_use_cuda_custom(ctx, A, B, C, side, uplo)) {
            return backend::symm_cuda_custom(ctx, A, B, C, alpha, beta, side, uplo);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::symm,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown,
            {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::symm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::symm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }
}

template <Backend Back, ComplexScalar T>
Event hemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::hemm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::hemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }
}

template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha,
           float_t<T> beta,
           Uplo uplo,
           Transpose transA) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::herk, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::herk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, ComplexScalar T>
Event her2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            float_t<T> beta,
            Uplo uplo,
            Transpose transA) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::her2k, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::her2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, RealScalar T>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syrk_use_cuda_custom(ctx, A, C, uplo, transA)) {
            return backend::syrk_cuda_custom(ctx, A, C, alpha, beta, uplo, transA);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::syrk,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::syrk, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::syrk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syr2k_use_cuda_custom(ctx, A, B, C, uplo, transA)) {
            return backend::syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::syr2k,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::syr2k, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::syr2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, typename T>
Event trmm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag) {
    // Native tile gate, CUDA + float only. evidence: docs/perf/level3.md#the-shipped-predicates
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::trmm_use_cuda_custom(ctx, A, B, C, side, uplo, transA, diag)) {
            return backend::trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
        }
        // Record the decline: a shape moving OFF a native kernel shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::trmm,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown, {uplo, side, diag, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::trmm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per device family.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

// symm/syrk/syr2k are RealScalar-constrained and hemm/herk/her2k
// ComplexScalar-constrained, hence the split.
#define REAL_ONLY_OPS(B_)             \
    OP_INSTANTIATE(symm,  B_, float)  \
    OP_INSTANTIATE(symm,  B_, double) \
    OP_INSTANTIATE(syrk,  B_, float)  \
    OP_INSTANTIATE(syrk,  B_, double) \
    OP_INSTANTIATE(syr2k, B_, float)  \
    OP_INSTANTIATE(syr2k, B_, double)

#define COMPLEX_ONLY_OPS(B_)                            \
    OP_INSTANTIATE(hemm,  B_, std::complex<float>)      \
    OP_INSTANTIATE(hemm,  B_, std::complex<double>)     \
    OP_INSTANTIATE(herk,  B_, std::complex<float>)      \
    OP_INSTANTIATE(herk,  B_, std::complex<double>)     \
    OP_INSTANTIATE(her2k, B_, std::complex<float>)      \
    OP_INSTANTIATE(her2k, B_, std::complex<double>)

#define ALL_TYPE_OPS_ONE(B_, fp)  \
    OP_INSTANTIATE(gemm, B_, fp)  \
    OP_INSTANTIATE(gemv, B_, fp)  \
    OP_INSTANTIATE(trsm, B_, fp)  \
    OP_INSTANTIATE(trmm, B_, fp)

#define LEVEL3_INSTANTIATE(B_)                       \
    ALL_TYPE_OPS_ONE(B_, float)                      \
    ALL_TYPE_OPS_ONE(B_, double)                     \
    ALL_TYPE_OPS_ONE(B_, std::complex<float>)        \
    ALL_TYPE_OPS_ONE(B_, std::complex<double>)       \
    REAL_ONLY_OPS(B_)                                \
    COMPLEX_ONLY_OPS(B_)

// Keyed on the DEVICE FAMILY, not on the vendor library: the bodies above
// compile to a throw when the library is absent, so the public entry point is a
// symbol in every build that has the device.
#if BATCHLAS_HAS_CUDA_BACKEND
LEVEL3_INSTANTIATE(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
// rocblas.cc has no hemm/herk/her2k/symm wrapper, so the ROCm backend
// instantiates only the ops it implements.
ALL_TYPE_OPS_ONE(Backend::ROCM, float)
ALL_TYPE_OPS_ONE(Backend::ROCM, double)
ALL_TYPE_OPS_ONE(Backend::ROCM, std::complex<float>)
ALL_TYPE_OPS_ONE(Backend::ROCM, std::complex<double>)
OP_INSTANTIATE(syrk,  Backend::ROCM, float)
OP_INSTANTIATE(syrk,  Backend::ROCM, double)
OP_INSTANTIATE(syr2k, Backend::ROCM, float)
OP_INSTANTIATE(syr2k, Backend::ROCM, double)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
LEVEL3_INSTANTIATE(Backend::NETLIB)
#endif

#undef LEVEL3_INSTANTIATE
#undef ALL_TYPE_OPS_ONE
#undef COMPLEX_ONLY_OPS
#undef REAL_ONLY_OPS
#undef OP_INSTANTIATE

}  // namespace batchlas
