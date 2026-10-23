#pragma once

#include <src/select/select.hh>

#include <algorithm>
#include <cstdint>

namespace batchlas::select {

/// @p chunk for a part loop whose last part takes the remainder (up to 2 chunk - 1 items), shrunk
/// past half the grid z/y ceiling so no part exceeds max_groups_yz / 2. Half, not all: a plain
/// range launch is range-rounded by DPC++, which lifts a batch of 65505 in dimension 0 to 65536.
/// evidence: docs/perf/qr.md#qr-l2-batch-chunking
inline int grid_capped_chunk(const Queue& q, int chunk, int batch) {
    const std::int64_t limit = describe(q.device(), q.backend(), false).max_groups_yz / 2;
    if (batch <= limit) return chunk;
    return static_cast<int>(std::min<std::int64_t>(chunk, (limit + 1) / 2));
}

}  // namespace batchlas::select
