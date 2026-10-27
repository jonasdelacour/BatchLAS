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

// The workspace of a remainder-last part loop at `chunk`: the larger of a full part and the last one,
// because a family's size need not be monotone in its items (two_stage needs more for 1 than for 2).
template <typename PartBytes>
std::size_t chunk_bytes(int chunk, int batch, PartBytes&& part_bytes) {
    if (batch < 2 * chunk) return part_bytes(batch);
    return std::max(part_bytes(chunk), part_bytes(chunk + batch % chunk));
}

// `chunk` when its parts fit `budget`, sized by the driver's own layout (`part_bytes(items)`, plus
// `extra(largest part)` once for the loop); else the chunk of the fewest near-even parts that fits. A
// bisection on the items finds the most one part holds; 1 when not even two items fit.
template <typename PartBytes, typename Extra>
int budget_capped_chunk(int chunk, int batch, std::size_t budget, PartBytes&& part_bytes, Extra&& extra) {
    const auto fits = [&](int c) {
        return chunk_bytes(c, batch, part_bytes) + extra(largest_part(c, batch)) <= budget;
    };
    if (budget == 0 || fits(chunk)) return chunk;
    int most = 1;
    for (int lo = 2, hi = largest_part(chunk, batch) - 1; lo <= hi;) {
        const int mid = lo + (hi - lo) / 2;
        if (part_bytes(mid) + extra(mid) <= budget) most = mid, lo = mid + 1;
        else hi = mid - 1;
    }
    for (int parts = (batch + most - 1) / most;; ++parts) {
        const int c = std::min(chunk, batch / parts);
        if (c <= 1) return 1;
        if (largest_part(c, batch) <= most && fits(c)) return c;
    }
}

}  // namespace batchlas::ops::syev::detail
