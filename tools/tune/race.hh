#pragma once

// Paired-ratio race between candidates of one cell (docs/design/tiered-tuning.md). Host-only.

#include <cstddef>
#include <string>
#include <vector>

#include "tier.hh"

namespace batchlas::tune {

struct RaceState {
    std::vector<std::string> cands;
    std::vector<std::vector<double>> ms;  // ms[c][r]: candidate c in round r; NaN when it did not run
    std::vector<bool> alive;
};

enum class RaceVerdict { more, winner, tie, cap };

inline constexpr double kGrossLoserRatio = 4.0;  // median paired ratio; any round count, see race_step

// Largest k with P(Binomial(n, 0.5) < k) <= 1 - confidence; 0 when none.
std::size_t lower_order_stat(int n, double confidence);

RaceVerdict race_step(RaceState& s, const TierParams& p, double tie = 0.03);

bool race_over(RaceVerdict v, int rounds, const TierParams& p);  // a winner by default runs min_reps rounds

// Survivors ranked by rank() on their medians, then the eliminated by median.
std::vector<std::string> race_ranking(const RaceState& s, const std::vector<std::string>& order, double tie = 0.03);

}  // namespace batchlas::tune
