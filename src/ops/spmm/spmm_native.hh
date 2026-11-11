// SPDX-License-Identifier: MIT
#pragma once

// Native batched CSR SpMM. TRAPS: A.nnz() is the batch-max CAPACITY (bound loops by row_offsets);
// \f$\beta = 0\f$ never reads C; no __restrict__ (LOBPCG slices alias); the scatter is not bitwise-repeatable.
// evidence: docs/perf/spmm.md#spmm-the-kernel-contract

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::ops::spmm {

template <typename T>
BATCHLAS_INTERNAL_API Event spmm_native_csr(Queue& ctx,  // all nine (transA, transB) spellings, dispatched on transA
                      const MatrixView<T, MatrixFormat::CSR>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B_mat,  // own ld/stride: never derive ld*cols
                      const MatrixView<T, MatrixFormat::Dense>& C,  // evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
                      T alpha,
                      T beta,
                      Transpose transA,
                      Transpose transB);

}  // namespace batchlas::ops::spmm
