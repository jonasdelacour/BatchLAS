#pragma once

// Native batched CSR SpMM declarations: C = alpha*op(A)*op(B) + beta*C, with A
// batched CSR (one strided slab per item) and B, C dense column-major. Three
// kernel bodies sit behind one {Native, Direct} route, picked on transA.
// evidence: docs/perf/spmm.md
//
// CSR indexing (src/matrix.cc): row offsets are ITEM-LOCAL, indexed
// b*offset_stride(); values and col_indices indexed b*matrix_stride(). A.nnz()
// is the batch-maximum CAPACITY, not a count, so the only legal bound on the
// nonzero loop is row_offsets[ro+i+1] -- slots above an item's own nnz are
// uninitialised garbage. Getting this wrong is correct at batch 1, wrong at 2.
//
// beta == 0 must not read C: callers pass never-zeroed BumpAllocator memory, so
// an unconditional beta*C_old returns NaN. Dually alpha == 0 leaves A and B
// unread but still requires C = beta*C.
//
// No __restrict__ on any pointer, and no body materialises a pointer array:
// LOBPCG passes X, P, R as element-disjoint slices of one buffer that alias.
//
// The transposed arm scatters through global atomics: summation order varies run
// to run (no test may compare two runs bitwise) and its FP64 instantiations
// carry an atomic64 device requirement the FP32 ones do not.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_spmm {

// Was the kernel COMPILED into this build? Not a device query; gates supports().
// Gather (transA == NoTrans) and scatter are independent capabilities.
template <typename T>
bool spmm_gather_available();

template <typename T>
bool spmm_scatter_available();

// All nine (transA, transB) spellings are served; dispatches on transA.
// evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
// B_mat and C carry their own ld and stride; read them from the view, never
// derive them as ld*cols.
template <typename T>
Event spmm_native_csr(Queue& ctx,
                      const MatrixView<T, MatrixFormat::CSR>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B_mat,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Transpose transA,
                      Transpose transB);

}  // namespace batchlas::sycl_spmm
