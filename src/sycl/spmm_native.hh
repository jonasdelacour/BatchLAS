#pragma once

// Native batched CSR SpMM, C = alpha*op(A)*op(B) + beta*C. TRAPS: A.nnz() is the batch-max
// CAPACITY (bound loops by row_offsets); beta == 0 never reads C, alpha == 0 still scales C;
// no __restrict__ (LOBPCG slices alias); the scatter is not bitwise-repeatable.
// evidence: docs/perf/spmm.md#spmm-the-kernel-contract

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_spmm {

template <typename T>
bool spmm_gather_available();  // compiled into this build; not a device query

template <typename T>
bool spmm_scatter_available();  // independent of gather; each gates supports()

template <typename T>
Event spmm_native_csr(Queue& ctx,  // all nine (transA, transB) spellings, dispatched on transA
                      const MatrixView<T, MatrixFormat::CSR>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B_mat,  // own ld/stride: never derive ld*cols
                      const MatrixView<T, MatrixFormat::Dense>& C,  // evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
                      T alpha,
                      T beta,
                      Transpose transA,
                      Transpose transB);

}  // namespace batchlas::sycl_spmm
