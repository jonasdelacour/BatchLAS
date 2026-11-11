// SPDX-License-Identifier: MIT
#pragma once

// Native batched GEMV. No is_gpu gate: Direct must build for native_cpu, so keep
// this TU out of NO_CPU_TARGETS. evidence: docs/perf/gemv.md#the-five-kernel-bodies
// m == 0, n == 0, (alpha == 0 && beta == 1) or a masked-off item leaves y untouched; A unread at alpha == 0.

#include <cstdint>

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::ops::gemv {

// The work-group ladder of gemv and native spmm over a FLATTENED 1-D grid (out_len * batch).
// units_per_wg_shift: 0 = one work-ITEM per unit, 5 = one 32-lane SUB-GROUP.
inline int gemv_wg_ladder(int64_t work_units, int max_wg, int cu,
                          int units_per_wg_shift) {
    int wg = 32;
    for (int cand : {256, 128, 64, 32}) {
        if (cand > max_wg) continue;
        wg = cand;
        const int64_t per_wg = cand >> units_per_wg_shift;
        if (per_wg < 1) continue;
        const int64_t groups = (work_units + per_wg - 1) / per_wg;
        if (groups >= static_cast<int64_t>(4) * cu) break;
    }
    // No admissible candidate leaves 32, which may exceed max_wg (an INVALID nd_range).
    if (max_wg > 0 && wg > max_wg) wg = max_wg;
    return wg;
}

template <typename T>
BATCHLAS_INTERNAL_API Event gemv_native_direct(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const VectorView<T>& X,
                         const VectorView<T>& Y,
                         T alpha,
                         T beta,
                         Transpose transA);

// transA MUST NOT be NoTrans; a direct caller that violates it gets a throw.
template <typename T>
BATCHLAS_INTERNAL_API Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha,
                      T beta,
                      Transpose transA);

// TEST-ONLY, launcher's own gate: 1 = body 3, W >= 2 = body 5 at W; pass A.cols()*batch.
template <typename T>
BATCHLAS_INTERNAL_API int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch);

}  // namespace batchlas::ops::gemv
