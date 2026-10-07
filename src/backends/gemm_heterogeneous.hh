#pragma once

// The heterogeneous-batch GEMM loop, shared by every backend and the vendor-free
// facade; only the per-item terminal is a parameter. Its semantics are not the
// vendor's: m == 0 or n == 0 members are skipped, a k == 0 member is C := beta*C,
// and an all-skipped batch still returns a valid Event.
// evidence: docs/perf/gemm.md#gemm-the-heterogeneous-batch-loop

#include "gemm_variant.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <stdexcept>
#include <utility>

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
        throw batchlas::invalid_argument("GEMM: incompatible per-batch matrix dimensions for heterogeneous dispatch");
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
            // k == 0 is not a GEMM: the defined result is C := beta*C.
            last_event = scale(ctx, beta, C.batch_item(batch_index));
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
