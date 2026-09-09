#pragma once

// The seam where a level-3 tile route gives up and asks for the vendor.
//
// WHY IT IS NOT SIMPLY THE PUBLIC symm/syrk/syr2k/trmm -- the single most
// valuable finding of the WP1 design pass, and it is a real bug if ignored.
//
// The obvious move is to make the four dispatchers' `*_vendor_cuda_raw(...)`
// fallbacks call the PUBLIC entry point, exactly as their downward GEMM
// terminal becomes the public gemm. It does not work, and it does not fail
// loudly: every one of those fallback sites is reached AFTER a gate that
// already returned true. `symm_vendor` calls `symm_use_cuda_custom`, that
// returns true, `symm_cuda_custom` runs, decides the shape is unsupported --
// and a public `symm` call from there re-enters `symm_use_cuda_custom` with the
// same environment and the same views. It returns true again. Unbounded
// recursion, reachable today with BATCHLAS_SYMM_ROUTE=custom on a CPU queue,
// where route_common.hh's should_use_cublasdx returns true for a forced custom
// variant BEFORE the problem_supported test.
//
// So the sideways terminal needs its own seam: forward to the vendor where one
// is compiled, and throw the ordinary NoRouteError where none is. That is what
// lets the four TUs stop naming cuBLAS symbols without pretending a vendor-free
// build can serve every shape.
//
// Portable by construction -- enums, matrix views and the queue, nothing else.
// The #if that picks vendor-or-throw is in the .cc, so the four dispatchers
// contain no preprocessor of their own.

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend::detail {

// Signatures are copied VERBATIM from {symm,syrk,syr2k,trmm}_custom_dispatch.hh
// rather than regenerated from the public declarations. That is deliberate: the
// vendor forms and the public forms genuinely disagree -- trsm's vendor form
// takes `alpha` last while the public form takes it third -- and the last time
// a facade was generated from public declarations instead of lifted verbatim it
// would have passed `alpha` where `side` was expected on every backend.

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
