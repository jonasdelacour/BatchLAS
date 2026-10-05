#pragma once

// For callers that size once at a bounding shape and factor sub-views of it: the largest need of
// every family this device can run there. evidence: docs/design/flat-select-p5/geqrf.md

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include "../../util/internal-api.hh"

#include <cstddef>

namespace batchlas {

template <Backend B, typename T>
BATCHLAS_INTERNAL_API std::size_t geqrf_buffer_size_bound(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                          Span<T> tau);

}  // namespace batchlas
