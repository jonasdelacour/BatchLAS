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

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define GEMM_INSTANTIATE(B_, fp) BATCHLAS_INSTANTIATE(sig::gemm<fp>, gemm, B_, fp)

#define GEMM_INSTANTIATE_ALL_TYPES(B_)      \
    GEMM_INSTANTIATE(B_, float)             \
    GEMM_INSTANTIATE(B_, double)            \
    GEMM_INSTANTIATE(B_, std::complex<float>)  \
    GEMM_INSTANTIATE(B_, std::complex<double>)

#if BATCHLAS_HAS_CUBLAS
GEMM_INSTANTIATE_ALL_TYPES(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCBLAS
GEMM_INSTANTIATE_ALL_TYPES(Backend::ROCM)
#endif

#if BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS
GEMM_INSTANTIATE_ALL_TYPES(Backend::NETLIB)
#endif

#undef GEMM_INSTANTIATE_ALL_TYPES
#undef GEMM_INSTANTIATE

}  // namespace batchlas
