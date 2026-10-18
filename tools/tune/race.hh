#pragma once

// Paired-ratio race between candidates of one cell (docs/design/tiered-tuning.md). Host-only.

#include <cstddef>
#include <string>
#include <vector>

#include <tools/tune/spec.hh>
#include <tools/tune/tier.hh>

namespace batchlas::tune {

struct RaceState {
    std::vector<std::string> cands;
    std::vector<std::vector<double>> ms;  // ms[c][r]: candidate c in round r; NaN when it did not run
    std::vector<bool> alive;
};

enum class RaceVerdict { more, winner, tie, cap };

inline constexpr double kGrossLoserRatio = 4.0;  // lower median paired ratio, from round 2 on; see race_step

// The lean cell: docs/design/tiered-tuning.md#engine-the-lean-cell
inline constexpr double kWarmTol = 0.02;  // two consecutive warm round totals within this: clocks are up
inline constexpr int kWarmMinRounds = 2;
inline constexpr double kWarmCapPerArmS = 0.2;
inline constexpr double kRepBudgetS = 0.05;  // timed work per arm
inline constexpr int kRepCap = 16;

bool warm_converged(double prev, double total, int rounds);  // warm round `rounds` (1-based) after `prev`
int min_reps_for(double confidence);  // the first n with lower_order_stat(n, confidence) >= 1
int lean_max_reps(double slowest_ms, const TierParams& p);
// An arm's lean cost besides its reps: 1 untimed + 2 cut runs, min(cap, kWarmMinRounds runs); verify_s is per cell.
double lean_arm_fixed_s(double ms);  // reps: kRepBudgetS / slowest in [min_reps_for, kRepCap]

// Largest k with P(Binomial(n, 0.5) < k) <= 1 - confidence; 0 when none.
std::size_t lower_order_stat(int n, double confidence);

RaceVerdict race_step(RaceState& s, const TierParams& p, double tie = 0.03);

bool race_over(RaceVerdict v, int rounds, const TierParams& p);  // a winner by default runs min_reps rounds

// evidence: docs/design/tiered-tuning.md#engine-tie-set-verification
std::vector<std::string> verify_next(const std::vector<ArmOutcome>& arms, const std::vector<std::string>& extra,
                                     double tie = 0.03);  // unverified arms that can rank first, or are in extra

// Survivors ranked by rank() on their medians, then the eliminated by median.
std::vector<std::string> race_ranking(const RaceState& s, const std::vector<std::string>& order, double tie = 0.03);

}  // namespace batchlas::tune
