// SPDX-License-Identifier: MIT
#pragma once

#include <src/queue.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cstddef>

namespace batchlas::ops::syev::detail {

// Test seam: nonzero replaces Device::scratch_budget in the blocked and two_stage chunk caps. No quick
// test batch passes the real budget, and on a 96 GB device not even C120 does.
// evidence: docs/perf/syev.md#syev-vendor-free-double-band
std::size_t& chunk_budget_override();

class ScopedChunkBudget {
public:
    explicit ScopedChunkBudget(std::size_t bytes) : saved_(chunk_budget_override()) { chunk_budget_override() = bytes; }
    ~ScopedChunkBudget() { chunk_budget_override() = saved_; }
    ScopedChunkBudget(const ScopedChunkBudget&) = delete;
    ScopedChunkBudget& operator=(const ScopedChunkBudget&) = delete;

private:
    std::size_t saved_;
};

inline std::size_t chunk_budget(const Queue& q) {
    const std::size_t bytes = chunk_budget_override();
    if (bytes != 0) return bytes;
    return static_cast<std::size_t>(select::describe(q.device(), q.backend(), false).scratch_budget);
}

// The largest pass of a remainder-last part loop: under two chunks there is one.
inline int largest_part(int chunk, int batch) { return batch < 2 * chunk ? batch : chunk + batch % chunk; }

// `chunk` when its largest part fits `budget`, sized by the driver's own layout (`part_bytes(items)`);
// else the chunk of the fewest near-even parts that fits. A bisection on the items finds the most one
// part holds; only a chunk whose largest part was sized to fit is returned (1 when none is).
template <typename PartBytes>
int budget_capped_chunk(int chunk, int batch, std::size_t budget, PartBytes&& part_bytes) {
    const auto fits = [&](int c) { return part_bytes(largest_part(c, batch)) <= budget; };
    if (budget == 0 || fits(chunk)) return chunk;
    int most = 1;
    for (int lo = 2, hi = largest_part(chunk, batch) - 1; lo <= hi;) {
        const int mid = lo + (hi - lo) / 2;
        if (part_bytes(mid) <= budget) most = mid, lo = mid + 1;
        else hi = mid - 1;
    }
    for (int parts = (batch + most - 1) / most;; ++parts) {
        const int c = std::min(chunk, batch / parts);
        if (c <= 1) return 1;
        if (largest_part(c, batch) <= most && fits(c)) return c;
    }
}

}  // namespace batchlas::ops::syev::detail
