// SPDX-License-Identifier: MIT
// Which batch items a check reads (docs/design/verification.md).
#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

namespace batchlas::verify {

/// {0, batch/2, batch-1}, sorted and de-duplicated.
inline std::vector<int> default_items(int batch) {
    if (batch < 1) return {};
    std::vector<int> items{0, batch / 2, batch - 1};
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());
    return items;
}

inline std::vector<int> all_items(int batch) {
    std::vector<int> items(batch > 0 ? batch : 0);
    std::iota(items.begin(), items.end(), 0);
    return items;
}

}  // namespace batchlas::verify
