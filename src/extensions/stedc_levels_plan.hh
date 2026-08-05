#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace batchlas {

// ---------------------------------------------------------------------------
// Shape of the level-synchronous ("flattened") STEDC merge tree.
//
// Lives in its own header purely so tests can assert on the plan directly:
// picking a bad leaf is a *performance* defect that produces perfectly correct
// eigenvalues, so no numerical test can catch it. See stedc_levels_plan_tests.
// ---------------------------------------------------------------------------
struct StedcLevelPlan {
    int64_t leaf = 0;      // leaf sub-problem size
    int32_t levels = 0;    // merge levels; the tree has 2^levels leaves
    int64_t padded_n = 0;  // leaf << levels
};

// Choose the merge tree for an n x n tridiagonal, given the tuned leaf
// threshold. `levels == 0` means "no level plan applies"; the caller falls back
// to the recursive driver.
inline StedcLevelPlan plan_stedc_levels(int64_t n, int64_t threshold) {
    StedcLevelPlan plan{n, 0, n};
    if (n <= threshold || threshold <= 0) {
        return plan;
    }
    // The leaf may never exceed the tuned threshold. That is a hard constraint,
    // not a preference: the threshold is the device sub-group width, and `steqr`
    // only takes the fast `steqr_cta` path for n <= that (steqr_cta throws above
    // it, see steqr_cta.cc). One step over the edge costs ~14x on the leaf solve
    // -- measured, batch 10416, eigenvectors: n=32 0.26us, n=36 3.76us, n=40
    // 4.87us, n=80 54.4us. The recursive driver got this for free by bisecting;
    // this planner has to say it.
    //
    // Below the cap, prefer the tree that pads least, since padding costs
    // (N/n)^3 in the top-level GEMM -- but note the cap outranks that. Weighting
    // leaf width against padding rather than bounding it is what regressed
    // n = 320 and n = 640 (leaf 40 over leaf 20, 3.25x and 1.51x on syev): the
    // padding term dominates the score, so a zero-padding wide leaf always won.
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
