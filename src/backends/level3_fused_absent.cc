// Vendor-free stand-in for level3_fused_cuda.cc: always NoKernel, the same
// answer the CUDA TU gives without MathDx. Each op's reaction stays in its
// dispatcher. evidence: docs/perf/level3.md#level-3-the-cublasdx-fused-tail-hook

#include "level3_fused.hh"

namespace batchlas::backend::detail {

namespace {
FusedResult none() { return FusedResult{Event{}, FusedResult::Outcome::NoKernel}; }
} // namespace

FusedResult symm_fused_try(Queue&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           float, float, Side, Uplo) {
    return none();
}

FusedResult syrk_fused_try(Queue&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           float, float, Uplo, Transpose) {
    return none();
}

FusedResult syr2k_fused_try(Queue&,
                            const MatrixView<float, MatrixFormat::Dense>&,
                            const MatrixView<float, MatrixFormat::Dense>&,
                            const MatrixView<float, MatrixFormat::Dense>&,
                            float, float, Uplo, Transpose) {
    return none();
}

FusedResult trmm_fused_try(Queue&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           const MatrixView<float, MatrixFormat::Dense>&,
                           float, Side, Uplo, Transpose, Diag) {
    return none();
}

} // namespace batchlas::backend::detail
