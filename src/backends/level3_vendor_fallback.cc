#include "level3_vendor_fallback.hh"

#include <batchlas/backend_config.h>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

#if BATCHLAS_HAS_CUBLAS
#include "symm_custom_dispatch.hh"
#include "syrk_custom_dispatch.hh"
#include "syr2k_custom_dispatch.hh"
#include "trmm_custom_dispatch.hh"
#endif

// Compiled in EVERY configuration; the only level-3 file naming a vendor symbol.
// The #if lives here, not in the dispatchers, because only #if drops an #include.

namespace batchlas::backend::detail {

namespace {
#if !BATCHLAS_HAS_CUBLAS
// The same NoRouteError the facade throws, so both failure sites read alike.
[[noreturn]] void no_vendor(dispatch::Op op) {
    dispatch::throw_no_vendor_route<float>(
        op, Backend::CUDA, dispatch::kLevel3Library<Backend::CUDA>);
}
#endif
} // namespace

Event symm_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Side side,
                           Uplo uplo) {
#if BATCHLAS_HAS_CUBLAS
    return symm_vendor_cuda_raw(ctx, A, B, C, alpha, beta, side, uplo);
#else
    (void)ctx; (void)A; (void)B; (void)C; (void)alpha; (void)beta; (void)side; (void)uplo;
    no_vendor(dispatch::Op::symm);
#endif
}

Event syrk_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Uplo uplo,
                           Transpose transA) {
#if BATCHLAS_HAS_CUBLAS
    return syrk_vendor_cuda_raw(ctx, A, C, alpha, beta, uplo, transA);
#else
    (void)ctx; (void)A; (void)C; (void)alpha; (void)beta; (void)uplo; (void)transA;
    no_vendor(dispatch::Op::syrk);
#endif
}

Event syr2k_vendor_fallback(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            float alpha,
                            float beta,
                            Uplo uplo,
                            Transpose transA) {
#if BATCHLAS_HAS_CUBLAS
    return syr2k_vendor_cuda_raw(ctx, A, B, C, alpha, beta, uplo, transA);
#else
    (void)ctx; (void)A; (void)B; (void)C; (void)alpha; (void)beta; (void)uplo; (void)transA;
    no_vendor(dispatch::Op::syr2k);
#endif
}

Event trmm_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag) {
#if BATCHLAS_HAS_CUBLAS
    return trmm_vendor_cuda_raw(ctx, A, B, C, alpha, side, uplo, transA, diag);
#else
    (void)ctx; (void)A; (void)B; (void)C; (void)alpha; (void)side; (void)uplo;
    (void)transA; (void)diag;
    no_vendor(dispatch::Op::trmm);
#endif
}

} // namespace batchlas::backend::detail
