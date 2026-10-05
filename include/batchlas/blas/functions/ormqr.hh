#pragma once

#include <batchlas/export.hh>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <complex>

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/tuning_params.hh>

#include <batchlas/internal/ormqr_blocked.hh>

#include <batchlas/blas/queue-dispatch.hh>

namespace batchlas {

// Signature aliases for explicit instantiation; see BATCHLAS_INSTANTIATE in
// src/util/template-instantiations.hh. Keep in sync with the declarations below.
namespace sig {
template <typename T>
using ormqr = Event(Queue&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    const MatrixView<T, MatrixFormat::Dense>&,
                    Side, Transpose, Span<T>, Span<std::byte>,
                    int32_t);

template <typename T>
using ormqr_buffer_size = size_t(Queue&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 const MatrixView<T, MatrixFormat::Dense>&,
                                 Side, Transpose, Span<T>,
                                 int32_t);

// The vendor entry points deliberately do NOT take the block-size hint: it
// selects a WY panel width in the blocked implementation and means nothing to a
// vendor kernel. So these are spelled out rather than aliased to the two above.
template <typename T>
using ormqr_vendor = Event(Queue&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           const MatrixView<T, MatrixFormat::Dense>&,
                           Side, Transpose, Span<T>, Span<std::byte>);

template <typename T>
using ormqr_vendor_buffer_size = size_t(Queue&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        const MatrixView<T, MatrixFormat::Dense>&,
                                        Side, Transpose, Span<T>);
}  // namespace sig


// Public API. Defined in src/ops/ormqr/ormqr.cc, which picks the kernel (flat selection).
template <Backend B, typename T>
BATCHLAS_API Event ormqr(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const MatrixView<T, MatrixFormat::Dense>& C,
                         Side side,
                         Transpose trans,
                         Span<T> tau,
                         Span<std::byte> workspace,
                         int32_t block_size_hint = 0);

template <Backend B, typename T>
BATCHLAS_API size_t ormqr_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& C,
                                      Side side,
                                      Transpose trans,
                                      Span<T> tau,
                                      int32_t block_size_hint = 0);

} // namespace batchlas

namespace batchlas::backend {

// Implemented by backend wrapper TUs (e.g. cuSOLVER / rocSOLVER / LAPACKE).
template <Backend B, typename T>
BATCHLAS_API Event ormqr_vendor(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& A,
                                const MatrixView<T, MatrixFormat::Dense>& C,
                                Side side,
                                Transpose trans,
                                Span<T> tau,
                                Span<std::byte> workspace);

template <Backend B, typename T>
BATCHLAS_API size_t ormqr_vendor_buffer_size(Queue& ctx,
                                             const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Side side,
                                             Transpose trans,
                                             Span<T> tau);

} // namespace batchlas::backend


namespace batchlas {

// Owning-argument and backend-deducing overloads: `f(ctx, Matrix, ...)` accepts
// owning containers where the primary takes views, and `f(ctx, ...)` uses
// ctx.backend(). See BATCHLAS_ACCEPT_OWNING and BATCHLAS_DISPATCH_ON_QUEUE in
// blas/queue-dispatch.hh.

BATCHLAS_ACCEPT_OWNING(ormqr)
BATCHLAS_ACCEPT_OWNING(ormqr_buffer_size)

BATCHLAS_DISPATCH_ON_QUEUE(ormqr)
BATCHLAS_DISPATCH_ON_QUEUE(ormqr_buffer_size)

}  // namespace batchlas
