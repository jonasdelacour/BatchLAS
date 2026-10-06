#pragma once

// Native batched GEMV. No is_gpu gate: Direct must build for native_cpu, so keep
// this TU out of NO_CPU_TARGETS. evidence: docs/perf/gemv.md#the-five-kernel-bodies
// m == 0, n == 0 or (alpha == 0 && beta == 1) leaves Y untouched; A unread at alpha == 0.

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemv {

// Compiled into this build? Not a device query -- gates supports() for native.
template <typename T>
BATCHLAS_INTERNAL_API bool gemv_direct_available();

template <typename T>
BATCHLAS_INTERNAL_API bool gemv_cta_available();

template <typename T>
Event gemv_native_direct(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const VectorView<T>& X,
                         const VectorView<T>& Y,
                         T alpha,
                         T beta,
                         Transpose transA);

// transA MUST NOT be NoTrans; a direct caller that violates it gets a throw.
template <typename T>
Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha,
                      T beta,
                      Transpose transA);

// TEST-ONLY, launcher's own gate: 1 = body 3, W >= 2 = body 5 at W; pass A.cols()*batch.
template <typename T>
BATCHLAS_INTERNAL_API int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch);

}  // namespace batchlas::sycl_gemv
