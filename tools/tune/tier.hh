// SPDX-License-Identifier: MIT
#pragma once

// Tuning tiers (docs/design/tiered-tuning.md, "Engine: tiers"). Host-only.

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace batchlas::tune {

// Declared in ascending precedence: a deeper tier replaces a shallower record. transcribed, prior and
// converted carry no kernel hash and are never stale (docs/design/tuning-data.md).
enum class Tier { transcribed, prior, converted, custom, preview, calibrate, coarse, deep };

inline int tier_rank(Tier t) { return static_cast<int>(t); }

inline bool unhashed(Tier t) { return t == Tier::transcribed || t == Tier::prior || t == Tier::converted; }

inline std::string to_string(Tier t) {
    switch (t) {
        case Tier::transcribed: return "transcribed";
        case Tier::prior: return "prior";
        case Tier::converted: return "converted";
        case Tier::custom: return "custom";
        case Tier::preview: return "preview";
        case Tier::calibrate: return "calibrate";
        case Tier::coarse: return "coarse";
        case Tier::deep: return "deep";
    }
    return "?";
}

inline std::optional<Tier> parse_tier(std::string_view s) {
    for (Tier t : {Tier::transcribed, Tier::prior, Tier::converted, Tier::custom, Tier::preview, Tier::calibrate,
                   Tier::coarse, Tier::deep})
        if (to_string(t) == s) return t;
    return std::nullopt;
}

// Bracket refinement (grid.hh): geometric midpoints, or bisection in the axis's full value list.
enum class RefineMode { geometric, index };

struct TierParams {
    int stride;            // lattice subsampling: every stride-th point
    double refine_ratio;   // bisect while hi/lo >= this; 0 = no bisection
    int min_reps;          // min_reps_for(confidence): fewer rounds cannot decide
    int max_reps;          // the replay's race cap; the live race caps at lean_max_reps (race.hh)
    double confidence;     // one-sided, for the paired-ratio elimination
    bool alternate_reverse;
    double audit_fraction;
    RefineMode refine_mode = RefineMode::geometric;
    double refine_margin = 0;  // also refine agreeing lattice brackets whose runner-up is within this of the winner; 0 = off
    double refine_cap_factor = 1.0;  // refinement cells per (op, dtype) <= this x round-0 cells
    bool round0_only = false;  // no frontier or bisection rounds: select's nearest row serves the rest
    int hash_stride = 0;       // tier_subsample's key-hash stride (a demand-driven grid); 0 = stride
};

// custom has no fixed protocol (expert overrides): it takes the coarse values as its base. calibrate: coarse
// races on a stride-4 probe lattice (docs/design/tiered-tuning.md#engine-calibrate-tier-from-a-borrowed-table).
// preview: round 0 only, a demand-driven grid at hash % 12 (inside coarse's full grid and the old preview's % 2).
inline const TierParams& params(Tier t) {
    static const TierParams preview{2, 1.1, 3, 6, 0.80, false, 0.02, RefineMode::index, 0.10, 3.0, true, 12};
    static const TierParams coarse{1, 1.1, 4, 12, 0.90, false, 0.02, RefineMode::geometric, 0, 1.0};
    static const TierParams deep{1, 1.1, 6, 16, 0.98, true, 0.01, RefineMode::geometric, 0, 2.0};
    static const TierParams calibrate{4, 1.1, 4, 12, 0.90, false, 0.02, RefineMode::index, 0, 1.0};
    switch (t) {
        case Tier::preview: return preview;
        case Tier::calibrate: return calibrate;
        case Tier::deep: return deep;
        default: return coarse;
    }
}

}  // namespace batchlas::tune
