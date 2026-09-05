#pragma once

// The heterogeneous-batch GEMM loop, extracted from cublas.cc so it is not
// vendor-only.
//
// WHY. A heterogeneous batch is one whose members differ in shape, so no
// strided-batched call can serve it -- every backend has to walk the batch.
// That loop lived inside gemm_heterogeneous_vendor_impl (src/backends/cublas.cc),
// a cuBLAS-gated TU, and it carries semantics that are NOT about the vendor at
// all:
//
//   * batch members with m == 0 or n == 0 are SKIPPED, not launched;
//   * a member with k == 0 is not a GEMM but a scale: C := beta * C;
//   * if nothing launched, the caller still gets a valid Event.
//
// So in a vendor-free build those three behaviours simply did not exist, which
// is why all 17 remaining vendor-free gemm_tests failures are heterogeneous
// batch. Hoisting the loop is what lets the vendor-free facade reuse it
// verbatim instead of growing a second, subtly different copy -- and this
// codebase has already paid twice for restating one behaviour in two places.
//
// The per-item TERMINAL is a parameter. That is the whole design: the loop, the
// skips and the k == 0 substitution are backend-independent; only what runs on
// one homogeneous batch item differs. cublas.cc passes gemm_vendor_impl; the
// vendor-free facade passes the public gemm.

#include "gemm_variant.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <stdexcept>
#include <utility>

namespace batchlas::backend::detail {

// `launch_item` is invoked as
//     launch_item(A.batch_item(i), B.batch_item(i), C.batch_item(i))
// and must return an Event. Everything else -- validation, the skips, the
// scale, the empty-batch Event -- is fixed here so it cannot diverge per
// backend.
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
        throw std::invalid_argument("GEMM: incompatible per-batch matrix dimensions for heterogeneous dispatch");
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
            // Not a degenerate GEMM -- a different operation. With k == 0 the
            // product contributes nothing, so the defined result is C := beta*C.
            // scale() is pure SYCL (src/matrix.cc), which is why this branch
            // needs no vendor at all.
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
