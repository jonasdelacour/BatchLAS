#pragma once

// Native batched GEMV declarations. Bodies and windows: docs/perf/gemv.md
//
// gemv is the one native tier NOT gated on is_gpu: the Direct arm must build for
// native_cpu, so this TU stays out of any NO_CPU_TARGETS object library.
// Reference-BLAS quick-return is matched exactly -- m == 0, n == 0 or
// (alpha == 0 && beta == 1) leaves Y untouched, and A is unread when alpha == 0.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemv {

// Compiled into this build? Not a device query -- gates supports() for native.
template <typename T>
bool gemv_direct_available();

template <typename T>
bool gemv_cta_available();

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

// TEST-ONLY, via the same gate the launcher uses: 1 = one sub-group per output,
// W >= 2 = the segmented kernel at that W. Pass A.cols() * A.batch_size().
template <typename T>
int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch);

}  // namespace batchlas::sycl_gemv
