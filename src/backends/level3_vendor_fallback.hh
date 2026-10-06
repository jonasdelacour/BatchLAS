#pragma once

// The seam where a level-3 tile route gives up and asks for the vendor: the
// vendor where one is compiled, NoRouteError where none is.
// Trap: never call the PUBLIC symm/syrk/syr2k/trmm from a fallback site. The
// gate already said yes, re-entry says yes again: unbounded recursion.
// evidence: docs/perf/level3.md#level-3-the-sideways-vendor-seam

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend::detail {

// Signatures copied VERBATIM from the *_custom_dispatch.hh headers, never
// regenerated from the public ones: vendor and public argument orders differ.

Event symm_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Side side,
                           Uplo uplo);

Event syrk_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Uplo uplo,
                           Transpose transA);

Event syr2k_vendor_fallback(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            float alpha,
                            float beta,
                            Uplo uplo,
                            Transpose transA);

Event trmm_vendor_fallback(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag);

} // namespace batchlas::backend::detail
