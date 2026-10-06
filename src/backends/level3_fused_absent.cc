// The vendor-free answer to "is there a fused device kernel for this shape".
//
// Compiled INSTEAD of level3_fused_cuda.cc when cuBLAS is absent
// (src/backends/CMakeLists.txt), so the four dispatchers link in a build that
// has no CUDA object library at all. `NoKernel` is not a degraded answer here
// -- it is the same answer the CUDA version already gives on this machine,
// where MathDx is missing and cublasdx_variant_needs_fallback is
// unconditionally true.
//
// Each op's reaction to NoKernel stays where it belongs, in the dispatcher:
// symm and syrk fall back to their GEMM shim, trmm to the vendor seam, syr2k
// throws. Putting the reaction here instead would have flattened four
// deliberately different behaviours into one.

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
