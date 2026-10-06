#pragma once

// syr2k selects in src/ops/syr2k/syr2k.cc. What is left here is the raw cuBLAS terminal that the
// dead level-3 vendor fallback (level3_vendor_fallback.cc) still names; both go with it.

#include "../queue.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

namespace batchlas::backend {

Event syr2k_vendor_cuda_raw(Queue& ctx,
                            const MatrixView<float, MatrixFormat::Dense>& A,
                            const MatrixView<float, MatrixFormat::Dense>& B,
                            const MatrixView<float, MatrixFormat::Dense>& C,
                            float alpha,
                            float beta,
                            Uplo uplo,
                            Transpose transA);

} // namespace batchlas::backend
