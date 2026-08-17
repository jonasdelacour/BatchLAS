// The public level-3 entry points, defined once, outside every vendor TU.
//
// WHY THIS FILE EXISTS -- the actual obstacle to vendor independence
//
// `gemm<Backend::CUDA, float>` was DEFINED at src/backends/cublas.cc:1568 and
// explicitly instantiated in the same file. So "build without cuBLAS" did not
// mean "lose the cuBLAS gemm path"; it meant "lose `batchlas::gemm` entirely",
// because the only definition of the public template lived in the file that was
// dropped. The same held in rocblas.cc and netlib_lapack.cc.
//
// That is a *definition ownership* problem, and no amount of enum, CMake or
// routing work addresses it -- which is why S1-S4 could reach a configuration
// that CONFIGURES with `BATCHLAS_HAS_CUDA_BACKEND 1` and `CUBLAS 0` but cannot
// LINK. Moving the definitions here is what makes the public API independent of
// which vendor libraries were compiled in.
//
// What each op looks like now:
//
//     vendor TU   defines and instantiates  backend::<op>_vendor<B, T>
//     this file   defines and instantiates  <op><B, T>, which calls it
//
// The instantiation guards below are per LIBRARY, not per device family, so a
// backend appears here exactly when the TU that defines its vendor entry point
// is compiled. Today every `<op>` still forwards straight to the vendor; S6
// puts the route resolution in between, at which point a backend can be
// instantiated with no vendor at all.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/trmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// WP1 S5: the native GEMM arm. Both are vendor-free -- gemm_kernels.hh reaches
// only enums/matrix/queue, and gemm_variant.hh stopped including linalg-impl.hh
// (its sole CUDA tie) in the same step.
#include "../../backends/gemm_variant.hh"
#include "../../sycl/gemm_kernels.hh"

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
        // WP1 S5. Without this arm the whole work package delivers nothing
        // vendor-free: S2 pointed the level-3 expansions at this entry point,
        // and this entry point threw. VENDOR_FREE_BASELINE.md claimed gemm
        // "fails only on the shapes outside gemm_custom_problem_supported";
        // it threw on EVERY call, measured at 48 passed / 136 failed where all
        // 48 passes are pure route-resolution tests that never run a kernel.
        //
        // The register-tiled family was linked the whole time -- it lives in
        // the vendor-free batchlas_sycl component -- just unreachable. LINKED
        // is not REACHABLE, and that distinction is what the coverage table's
        // `linked` rows do and do not tell you.
        //
        // The routing is NOT duplicated here. backend::gemm_route is the same
        // adapter cublas.cc consults, over the same RouteTable<Op::gemm, T>;
        // `vendor_available = false` is the one input that differs, and
        // resolve_route already has a branch for it (route_resolve.hh: a
        // requested vendor that does not exist falls back to the ordinary
        // automatic choice rather than being honoured).
        const auto route = backend::gemm_route<T>(ctx, A, B, C, transA, transB,
                                                  precision, /*vendor_available=*/false);
        if (dispatch::is_native(route)) {
            return sycl_gemm::gemm_custom<T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }
        // Still honest about the gap: shapes the native kernel does not serve
        // (heterogeneous batches, non-Default precision, degenerate dims) have
        // no route at all without a vendor, and say so by name.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }
}

template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA) {
    if constexpr (!dispatch::level3_vendor_available<B>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemv, B, dispatch::kLevel3Library<B>);
    } else {
        return backend::gemv_vendor<B, T>(ctx, A, X, Y, alpha, beta, transA);
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
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::trmm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

// Real- and complex-only ops are separated because symm/syrk/syr2k are
// RealScalar-constrained and hemm/herk/her2k ComplexScalar-constrained.
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

// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
LEVEL3_INSTANTIATE(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
// rocBLAS has no hemm/herk/her2k/symm wrapper in rocblas.cc, so the ROCm
// backend instantiates only the ops it actually implements. That asymmetry
// predates S5 -- rocblas.cc's own instantiation block listed exactly these.
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
