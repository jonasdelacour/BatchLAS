// The public symm/hemm/herk/her2k/syr2k/trmm entry points (syrk: src/ops/syrk), defined outside every
// vendor TU so the API links in a build with no vendor library. The float CUDA tile
// routes are chosen by rule in src/backends/*_custom_dispatch.cc (BATCHLAS_<OP>_ROUTE);
// everything else goes to backend::<op>_vendor<B, T>, or throws NoRouteError when no
// vendor library is compiled in.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/trmm.hh>

#include <batchlas/no_route.hh>
#include "../../select/vendor.hh"

// The four level-3 custom-route gates. They have to run before the
// vendor-available test, so they live here rather than in cublas.cc.
#include "../../backends/symm_custom_dispatch.hh"
#include "../../backends/syr2k_custom_dispatch.hh"
#include "../../backends/trmm_custom_dispatch.hh"
#include "../../backends/level3_coverage.hh"

#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

// gemm lives in src/ops/gemm/gemm.cc (flat kernel selection).

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
            Op::symm, "vendor",
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown,
            {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::symm, Back, select::kLevel3Library<Back>);
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
    if constexpr (!select::level3_vendor_available<Back>) {
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
    if constexpr (!select::level3_vendor_available<Back>) {
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
    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::her2k, Back, select::kLevel3Library<Back>);
    } else {
        return backend::her2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
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
            Op::syr2k, "vendor",
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::syr2k, Back, select::kLevel3Library<Back>);
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
            Op::trmm, "vendor",
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown, {uplo, side, diag, transA});
    }

    if constexpr (!select::level3_vendor_available<Back>) {
        select::throw_no_vendor_route<T>(
            Op::trmm, Back, select::kLevel3Library<Back>);
    } else {
        return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per device family.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

// symm/syr2k are RealScalar-constrained and hemm/herk/her2k
// ComplexScalar-constrained, hence the split.
#define REAL_ONLY_OPS(B_)             \
    OP_INSTANTIATE(symm,  B_, float)  \
    OP_INSTANTIATE(symm,  B_, double) \
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
