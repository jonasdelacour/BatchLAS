#pragma once

// The cuBLASDx fused device kernels, behind a portable declaration.
//
// WHY A HOOK RATHER THAN #if IN THE DISPATCHERS. An `if constexpr` cannot
// discard a file-scope #include -- only a #if can. So fencing the fused tails
// where they sit would leave symm/syrk/syr2k/trmm_custom_dispatch.cc striped
// with preprocessor AND still reaching <cuda_runtime_api.h> through
// gemm_cublasdx_dispatch.hh, *_cublasdx_fused.hh and
// cublasdx_dispatch_common.hh. Moving the tails out instead leaves the four
// dispatchers with no preprocessor and no CUDA header at all, which is the
// actual WP1 goal.
//
// ZERO ROUTE RISK, and that is checkable rather than hoped for: MathDx is not
// present in this build (BATCHLAS_HAS_CUBLASDX 0, mathdx_DIR-NOTFOUND), so
// every *_cublasdx::available() is false, cublasdx_variant_needs_fallback is
// unconditionally true, and not one of these tails is reachable. They are moved
// exactly as they are.

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend::detail {

// THREE outcomes, not two, because the four ops genuinely disagree about what
// each one means and flattening them would change behaviour:
//
//   Ran                -- the fused kernel ran; `event` is its completion.
//   NoKernel           -- no compatible fused variant exists in this build.
//                         symm and syrk fall back to their GEMM shim, trmm to
//                         the vendor, and syr2k THROWS (syr2k_custom_dispatch's
//                         throw is not guarded by `forced` -- pre-existing, and
//                         recorded in WP1_LEVEL3_SPEC.md as out of scope).
//   DeviceUnsupported  -- the kernel exists but the device refused it at launch
//                         (cudaErrorNotSupported). Every op falls back.
//
// A hard launch failure is neither: it throws from inside the CUDA TU, exactly
// as it does today.
struct FusedResult {
    enum class Outcome { Ran, NoKernel, DeviceUnsupported };
    Event event{};
    Outcome outcome = Outcome::NoKernel;
};

FusedResult symm_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Side side,
                           Uplo uplo);

FusedResult syrk_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Uplo uplo,
                           Transpose transA);

FusedResult syr2k_fused_try(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            float alpha,
                            float beta,
                            Uplo uplo,
                            Transpose transA);

FusedResult trmm_fused_try(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag);

} // namespace batchlas::backend::detail
