#pragma once

#include "../queue.hh"

#include <blas/enums.hh>
#include <blas/matrix.hh>

namespace batchlas::backend {

// True unless BATCHLAS_SYRK_VARIANT pins the vendor. The float router reads the
// variable through its own enum; double and complex reach only the single-tile
// Gram kernel, so they need just this one bit of it -- but they do need it, or
// `=vendor` would silently measure the new route and report it as the old one.
bool syrk_route_prefers_vendor();

// True only when BATCHLAS_SYRK_VARIANT names the Gram kernel outright. HERK
// does not take it automatically: measured on RTX 4090 / sm_89 in complex float
// against the GEMM-plus-Hermitian-fold route it would replace, the tile kernel
// loses at every Gram shape -- 0.217 vs 0.206 ms at n=32/batch 2048, 2.08 vs
// 1.57 at n=128/batch 512. A complex multiply is four real ones, so herk is
// compute bound where real syrk is bandwidth bound, and cuBLAS's cgemm is
// simply better at compute than this kernel is. The route stays reachable so it
// remains measurable and so the conjugation stays under test.
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