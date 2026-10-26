// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

namespace batchlas::ops::syev::detail {

// Test seam: nonzero replaces Device::scratch_budget in syev's part split, which no ordinary test batch
// reaches on a real device. evidence: docs/perf/syev.md#syev-vendor-free-double-band
std::size_t& part_budget_override();

class ScopedPartBudget {
public:
    explicit ScopedPartBudget(std::size_t bytes) : saved_(part_budget_override()) { part_budget_override() = bytes; }
    ~ScopedPartBudget() { part_budget_override() = saved_; }
    ScopedPartBudget(const ScopedPartBudget&) = delete;
    ScopedPartBudget& operator=(const ScopedPartBudget&) = delete;

private:
    std::size_t saved_;
};

}  // namespace batchlas::ops::syev::detail
