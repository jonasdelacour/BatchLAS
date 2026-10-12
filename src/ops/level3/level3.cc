// The public hemm/herk/her2k entry points, defined outside every vendor TU so the API
// links in a build with no vendor library. They go straight to backend::<op>_vendor<B, T>
// (whose expand/fold-into-gemm arms are still hand-written in cublas.cc), or throw
// NoRouteError when no vendor library is compiled in. symm, syrk, syr2k and trmm use
// flat kernel selection: src/ops/<op>/<op>.cc.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/her2k.hh>

#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

// rocBLAS is linked but src/backends/rocblas.cc wraps none of these three.
// No build switch restores these on ROCm: rocblas.cc has no wrapper at all.
template <class T>
[[noreturn]] void throw_no_rocm_wrapper(Op op) {
    coverage::record_miss(op, scalar_kind_of<T>, Backend::ROCM, "rocBLAS");
    throw NoRouteError(op, Backend::ROCM, scalar_kind_of<T>,
                       "rocblas.cc has no hemm/herk/her2k wrapper; no build option adds one");
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
    if constexpr (Back == Backend::ROCM) {
        throw_no_rocm_wrapper<T>(Op::hemm);
    } else if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::hemm, Back, select::kLevel3Library<Back>);
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
    if constexpr (Back == Backend::ROCM) {
        throw_no_rocm_wrapper<T>(Op::herk);
    } else if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::herk, Back, select::kLevel3Library<Back>);
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
    if constexpr (Back == Backend::ROCM) {
        throw_no_rocm_wrapper<T>(Op::her2k);
    } else if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::her2k, Back, select::kLevel3Library<Back>);
    } else {
        return backend::her2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per device family.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

#define LEVEL3_INSTANTIATE(B_)                          \
    OP_INSTANTIATE(hemm,  B_, std::complex<float>)      \
    OP_INSTANTIATE(hemm,  B_, std::complex<double>)     \
    OP_INSTANTIATE(herk,  B_, std::complex<float>)      \
    OP_INSTANTIATE(herk,  B_, std::complex<double>)     \
    OP_INSTANTIATE(her2k, B_, std::complex<float>)      \
    OP_INSTANTIATE(her2k, B_, std::complex<double>)

// Keyed on the DEVICE FAMILY, not on the vendor library: the bodies above
// compile to a throw when the library is absent (always on ROCm), so the public
// entry point is a symbol in every build that has the device; the dispatch macro
// emits a <ROCM> call in every ROCm build.
#if BATCHLAS_HAS_CUDA_BACKEND
LEVEL3_INSTANTIATE(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
LEVEL3_INSTANTIATE(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
LEVEL3_INSTANTIATE(Backend::NETLIB)
#endif

#undef LEVEL3_INSTANTIATE
#undef OP_INSTANTIATE

}  // namespace batchlas
