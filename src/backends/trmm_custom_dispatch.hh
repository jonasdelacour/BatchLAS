#pragma once

#include "../queue.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>

namespace batchlas::backend {

bool trmm_cuda_custom_forced();

// True when BATCHLAS_TRMM_VARIANT pins the vendor. The float router reads the
// variable through its own enum; double and complex reach only the tile kernel,
// so they need just this one bit of it -- but they do need it, or `=vendor`
// would silently measure the new route and report it as the old one.
bool trmm_route_prefers_vendor();

bool trmm_use_cuda_custom(const Queue& ctx,
                          const MatrixView<float, MatrixFormat::Dense>& A,
                          const MatrixView<float, MatrixFormat::Dense>& B,
                          const MatrixView<float, MatrixFormat::Dense>& C,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag diag);

Event trmm_cuda_custom(Queue& ctx,
                       const MatrixView<float, MatrixFormat::Dense>& A,
                       const MatrixView<float, MatrixFormat::Dense>& B,
                       const MatrixView<float, MatrixFormat::Dense>& C,
                       float alpha,
                       Side side,
                       Uplo uplo,
                       Transpose transA,
                       Diag diag);

Event trmm_vendor_cuda_raw(Queue& ctx,
                           const MatrixView<float, MatrixFormat::Dense>& A,
                           const MatrixView<float, MatrixFormat::Dense>& B,
                           const MatrixView<float, MatrixFormat::Dense>& C,
                           float alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag);

} // namespace batchlas::backend