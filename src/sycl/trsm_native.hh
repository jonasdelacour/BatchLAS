#pragma once

// Native batched TRSM declarations: V1 (CTA solver, one work-group per matrix)
// and V2 (blocked driver that calls V1 on each diagonal block). See docs/perf/trsm.md.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <functional>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_trsm {

// Zero means "no native TRSM kernel in this build": RouteTable<Op::trsm,T> reads
// this through TrsmShape::cta_max_n and reports both native routes unsupported.
template <typename T>
int trsm_cta_max_n();

template <typename T>
bool trsm_blocked_available();

template <typename T>
Event trsm_native_v1_dispatch(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              T alpha,
                              Side side,
                              Uplo uplo,
                              Transpose transA,
                              Diag diag);

// Trailing-update GEMM. An EMPTY function means sycl_gemm::gemm_custom, keeping
// this layer dispatch-free; inject the routed gemm where dispatch is available,
// since the native kernel collapses on the strided sub-views a panel passes.
// evidence: docs/perf/trsm.md#the-final-grid-after-the-routed-trailing-gemm
template <typename T>
using TrsmTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

template <typename T>
Event trsm_native_blocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag diag,
                          TrsmTrailingGemm<T> trailing_gemm = {});

} // namespace batchlas::sycl_trsm
