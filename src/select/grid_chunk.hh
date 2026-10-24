#pragma once

#include <src/select/select.hh>

#include <algorithm>
#include <cstdint>

namespace batchlas::select {

// chunk, shrunk so no part of a remainder-last part loop exceeds max_groups_yz / 2 (DPC++ range
// rounding lifts a plain range's batch past the grid). evidence: docs/perf/qr.md#qr-l2-batch-chunking
inline int grid_capped_chunk(const Queue& q, int chunk, int batch) {
    const std::int64_t limit = describe(q.device(), q.backend(), false).max_groups_yz / 2;
    if (batch <= limit) return chunk;
    return static_cast<int>(std::min<std::int64_t>(chunk, (limit + 1) / 2));
}

}  // namespace batchlas::select
