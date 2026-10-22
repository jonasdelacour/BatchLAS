// SPDX-License-Identifier: MIT
#pragma once

// The heterogeneous-batch GEMM loop, shared by every backend and the vendor-free
// facade; only the per-item terminal is a parameter. Its semantics are not the
// vendor's: m == 0 or n == 0 members are skipped, a k == 0 member is C := beta*C,
// and an all-skipped batch still returns a valid Event.
// evidence: docs/perf/gemm.md#gemm-the-heterogeneous-batch-loop
// Also the batch-shape predicates the gemm entry point and the vendor TUs share.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <stdexcept>
#include <utility>

namespace batchlas::backend {

template <typename T>
inline bool gemm_has_heterogeneous_batch(const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         const MatrixView<T, MatrixFormat::Dense>& C) {
    return A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous();
}

// Every batch member's (m, n, k) agrees across A, B and C.
template <typename T>
inline bool gemm_batch_dimensions_compatible(const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Transpose transA,
                                             Transpose transB) {
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        if (k != k_b) {
            return false;
        }
        if (C.rows(batch_index) != m || C.cols(batch_index) != n) {
            return false;
        }
        if (m < 0 || n < 0 || k < 0) {
            return false;
        }
    }

    return true;
}

} // namespace batchlas::backend

namespace batchlas::backend::detail {

// launch_item(A.batch_item(i), B.batch_item(i), C.batch_item(i)) -> Event.
template <typename T, typename LaunchItem>
Event gemm_heterogeneous_loop(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              const MatrixView<T, MatrixFormat::Dense>& C,
                              T beta,
                              Transpose transA,
                              Transpose transB,
                              LaunchItem&& launch_item) {
    if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
        throw batchlas::invalid_argument("gemm: incompatible per-batch matrix dimensions for heterogeneous dispatch");
    }

    bool launched = false;
    Event last_event;
    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        static_cast<void>(k_b);
        if (m == 0 || n == 0) {
            continue;
        }
        if (k == 0) {
            // k == 0 is not a GEMM: the defined result is C := beta*C, and beta == 0 never reads C.
            last_event = beta == T(0) ? C.batch_item(batch_index).fill(ctx, T(0))
                                      : scale(ctx, beta, C.batch_item(batch_index));
            launched = true;
            continue;
        }

        last_event = launch_item(A.batch_item(batch_index),
                                 B.batch_item(batch_index),
                                 C.batch_item(batch_index));
        launched = true;
    }

    if (launched) {
        return std::move(last_event);
    }
    return ctx.create_event_after_external_work();
}

} // namespace batchlas::backend::detail
