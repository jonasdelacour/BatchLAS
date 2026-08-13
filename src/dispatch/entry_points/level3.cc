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
    return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
}

template <Backend B, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA) {
    return backend::gemv_vendor<B, T>(ctx, A, X, Y, alpha, beta, transA);
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
    return backend::trsm_vendor<Back, T>(ctx, A, B, side, uplo, transA, diag, alpha);
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
    return backend::symm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
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
    return backend::hemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
}

template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha,
           float_t<T> beta,
           Uplo uplo,
           Transpose transA) {
    return backend::herk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
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
    return backend::her2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
}

template <Backend Back, RealScalar T>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA) {
    return backend::syrk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
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
    return backend::syr2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
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
    return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
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

#if BATCHLAS_HAS_CUBLAS
LEVEL3_INSTANTIATE(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCBLAS
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

#if BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS
LEVEL3_INSTANTIATE(Backend::NETLIB)
#endif

#undef LEVEL3_INSTANTIATE
#undef ALL_TYPE_OPS_ONE
#undef COMPLEX_ONLY_OPS
#undef REAL_ONLY_OPS
#undef OP_INSTANTIATE

}  // namespace batchlas
