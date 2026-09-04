#pragma once

// Native batched CSR SpMM -- declarations. C = alpha*op(A)*op(B) + beta*C, with
// A batched CSR (one strided slab per item) and B, C dense column-major. Three
// kernel bodies sit behind one {Native, Direct} route; the launcher picks on
// transA. evidence: docs/perf/spmm.md
//
// CSR INDEXING CONTRACT (src/matrix.cc). Getting it wrong is correct at batch 1
// and wrong at batch 2:
//   * Row offsets are ITEM-LOCAL and start at 0, indexed b*offset_stride()
//     (== rows+1); values and col_indices are indexed b*matrix_stride().
//   * A.nnz() is the batch-maximum CAPACITY, not a count: the only legal bound
//     on the nonzero loop is row_offsets[ro+i+1], and the value/index slots
//     above an item's own nnz are uninitialised garbage.
//
// beta == 0 MEANS C IS NOT READ. Callers pass BumpAllocator memory, which is
// never zeroed, so an unconditional beta*C_old returns NaN. Dually alpha == 0
// leaves A and B unread but C = beta*C still happens -- do not copy gemv's quick
// return, which also fires at alpha == 0 && beta == 1.
//
// NO __restrict__ ON ANY POINTER, and no body materialises a pointer array:
// LOBPCG passes X, P, R as element-disjoint slices of one buffer that alias at
// the object level. Every operand is reached as base + b*stride.
//
// The transposed arm scatters through global atomics, so summation order varies
// run to run -- no test may compare two runs bitwise -- and its FP64
// instantiations carry an atomic64 device requirement the FP32 ones do not.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_spmm {

// Was the kernel COMPILED into this build? Not a device query; this gates
// supports() for the native route. The gather (transA == NoTrans) and the
// scatter pair are independent capabilities, hence two flags.
template <typename T>
bool spmm_gather_available();

template <typename T>
bool spmm_scatter_available();

// All nine (transA, transB) spellings are served; dispatches on transA.
// evidence: docs/perf/spmm.md#supports-and-what-is-deliberately-not-in-it
//
// There is no workspace parameter: this route allocates nothing, so the sizing
// query and the call agree by construction. A is the sparse operand and must be
// CSR; B_mat and C carry their own ld and stride, which must be read from the
// view, never derived as ld*cols.
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
