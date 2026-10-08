#pragma once

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include "../../util/internal-api.hh"

#include <cstddef>

namespace batchlas {

/// The largest workspace of every family this device can run at A, for callers that size once at a
/// bounding shape and factor sub-views. `evidence: docs/design/flat-kernel-selection.md#phase-5-geqrf`
template <Backend B, typename T>  /// @ingroup api_selection_ops
BATCHLAS_INTERNAL_API std::size_t geqrf_buffer_size_bound(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                          Span<T> tau);

}  // namespace batchlas
