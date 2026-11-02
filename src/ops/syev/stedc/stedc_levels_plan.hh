// SPDX-License-Identifier: MIT
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

// Merge tree for an \f$n \times n\f$ tridiagonal and the tuned leaf threshold; `levels == 0` means
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

// The padded tree of a plan over `bs` items. Level l has 2^l nodes of size N >> l per item, merged
// as one launch over P = 2^l * bs sub-problems (node p of item p >> l); level L is the leaves.
struct MergeTree {
    int64_t N = 0;
    int32_t L = 0;
    int64_t bs = 0;

    struct Level {
        int64_t node_size;
        int64_t half;    // the cut: left child size
        int64_t nodes;   // per item
        int64_t P;       // nodes * bs
    };

    static MergeTree of(const StedcLevelPlan& plan, int64_t bs) { return {plan.padded_n, plan.levels, bs}; }

    constexpr Level level(int32_t l) const {
        const int64_t k = int64_t(1) << l;
        return {N >> l, (N >> l) >> 1, k, k * bs};
    }

    // Tear coefficients of level l: [rho_offset(l), rho_offset(l + 1)); levels 0..L-1.
    constexpr int64_t rho_offset(int32_t l) const { return ((int64_t(1) << l) - 1) * bs; }
    constexpr int64_t rho_count() const { return rho_offset(L); }
    // Coefficient g is node p % 2^l of item p / 2^l at level l.
    struct Slot { int32_t l; int64_t p; };
    constexpr Slot rho_slot(int64_t g) const {
        int32_t l = 0;
        int64_t start = 0;
        while (l < L) {
            const int64_t count = (int64_t(1) << l) * bs;
            if (g < start + count) break;
            start += count;
            ++l;
        }
        return {l, g - start};
    }

    // Expanded batch mask of level l >= 1: [mask_offset(l), mask_offset(l + 1)); level 0 is the call's.
    constexpr int64_t mask_offset(int32_t l) const { return ((int64_t(1) << l) - 2) * bs; }
    constexpr int64_t mask_count() const { return L > 0 ? mask_offset(L + 1) : 0; }
    constexpr int32_t mask_level(int64_t g) const {
        int32_t l = 1;
        while (g >= mask_offset(l + 1)) ++l;
        return l;
    }

    // Eigenvector storage of level l: the root in `top`, odd levels ping-pong through `odd`, and
    // even levels >= 2 (<= N*N*bs/4) in the upper half of `qprime`, whose first N*N*bs/2^l a
    // level-l merge uses.
    template <typename T> T* buffer(int32_t l, T* top, T* odd, T* qprime) const {
        if (l == 0) return top;
        if (l & 1) return odd;
        return qprime + static_cast<size_t>(N) * N * bs / 2;
    }
};

} // namespace batchlas
