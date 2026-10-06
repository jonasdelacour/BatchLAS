#pragma once

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend {

// True unless BATCHLAS_SYRK_VARIANT pins the vendor. Non-float callers need this
// bit too, or `=vendor` would silently measure the new route.
bool syrk_route_prefers_vendor();

// True only when BATCHLAS_SYRK_VARIANT names the Gram kernel. herk never takes it
// automatically (it loses to GEMM-plus-fold); it stays reachable to stay tested.
// evidence: docs/perf/level3.md#herk-on-the-gram-tile-kernel
bool syrk_route_requests_gram();

bool syrk_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Uplo uplo,
                          Transpose transA);

Event syrk_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       float beta,
                       Uplo uplo,
                       Transpose transA);

Event syrk_vendor_cuda_raw(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           float beta,
                           Uplo uplo,
                           Transpose transA);

} // namespace batchlas::backend