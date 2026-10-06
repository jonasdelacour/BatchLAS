#pragma once

// The cuBLASDx fused kernels behind a portable declaration (no CUDA header in
// the dispatchers). evidence: docs/perf/level3.md#level-3-the-cublasdx-fused-tail-hook

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend::detail {

// THREE outcomes, not two: the four ops react differently, so do not flatten.
//   Ran                -- the fused kernel ran; `event` is its completion.
//   NoKernel           -- no compatible fused variant in this build.
//   DeviceUnsupported  -- the device refused the launch (cudaErrorNotSupported).
// A hard launch failure is neither and throws from the CUDA TU.
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
