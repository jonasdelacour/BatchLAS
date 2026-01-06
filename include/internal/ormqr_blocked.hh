#pragma once

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

namespace batchlas {

// Blocked application of Q from a QR factorization using the classic WY representation:
//   H = I - V T V^H   (LAPACK LARFT/LARFB style)
// and level-3 BLAS (GEMM).
//
// Intended for medium sizes where the CTA kernels are not applicable.
//
// Notes:
// - Assumes reflectors come from GEQRF (QR, Forward/Columnwise, unit-lower V).
// - Supports batched inputs via strided-batch views.
// - Requires an in-order Queue for correct sequencing across the packing/LARFT/GEMM steps.
// - Workspace is required for explicit V, T, and intermediate W buffers.

template <Backend B, typename T>
Event ormqr_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a,
                    const MatrixView<T, MatrixFormat::Dense>& c,
                    Side side,
                    Transpose trans,
                    Span<T> tau,
                    Span<std::byte> workspace,
                    int32_t block_size = 32);

template <Backend B, typename T>
size_t ormqr_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const MatrixView<T, MatrixFormat::Dense>& c,
                                 Side side,
                                 Transpose trans,
                                 Span<T> tau,
                                 int32_t block_size = 32);

} // namespace batchlas
