#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace batchlas {

// Level-synchronous STEDC merge tree; own header so tests can assert on it (a bad leaf is only slow).
struct StedcLevelPlan {
    int64_t leaf = 0;      // leaf sub-problem size
    int32_t levels = 0;    // merge levels; the tree has 2^levels leaves
    int64_t padded_n = 0;  // leaf << levels
};

// Merge tree for an n x n tridiagonal and the tuned leaf threshold; `levels == 0` means
// no level plan applies (the caller falls back to the recursive driver).
inline StedcLevelPlan plan_stedc_levels(int64_t n, int64_t threshold) {
    StedcLevelPlan plan{n, 0, n};
    if (n <= threshold || threshold <= 0) {
        return plan;
    }
    // HARD CAP, not a preference: the leaf never exceeds the threshold (the sub-group
    // width; steqr_cta throws above it and one step over costs ~14x). Below the cap,
    // prefer the tree that pads least; weighting width against padding regressed syev.
    // evidence: docs/perf/stedc.md#stedc-the-leaf-cap-at-the-sub-group-width
    const int64_t lo = std::max<int64_t>(2, threshold / 2);
    const int64_t hi = threshold;
    double best_score = std::numeric_limits<double>::max();
    for (int32_t L = 1; L <= 24; ++L) {
        const int64_t k = int64_t(1) << L;
        if (k > n) break;
        const int64_t leaf = (n + k - 1) / k;
        if (leaf > hi) continue;
        if (leaf < lo) break;
        const int64_t N = leaf * k;
        const double score = static_cast<double>(N - n) / static_cast<double>(n)
                           + 1e-3 * std::abs(static_cast<double>(leaf - threshold)) / static_cast<double>(threshold);
        if (score < best_score) {
            best_score = score;
            plan = StedcLevelPlan{leaf, L, N};
        }
    }
    return plan;
}

} // namespace batchlas
