#pragma once

// Tuning tiers (docs/design/tiered-tuning.md, "Engine: tiers"). Host-only.

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace batchlas::tune {

// Declared in ascending precedence: a deeper tier replaces a shallower record.
enum class Tier { transcribed, custom, preview, coarse, deep };

inline int tier_rank(Tier t) { return static_cast<int>(t); }

inline std::string to_string(Tier t) {
    switch (t) {
        case Tier::transcribed: return "transcribed";
        case Tier::custom: return "custom";
        case Tier::preview: return "preview";
        case Tier::coarse: return "coarse";
        case Tier::deep: return "deep";
    }
    return "?";
}

inline std::optional<Tier> parse_tier(std::string_view s) {
    for (Tier t : {Tier::transcribed, Tier::custom, Tier::preview, Tier::coarse, Tier::deep})
        if (to_string(t) == s) return t;
    return std::nullopt;
}

// Bracket refinement (grid.hh): geometric midpoints, or bisection in the axis's full value list.
enum class RefineMode { geometric, index };

struct TierParams {
    int stride;            // lattice subsampling: every stride-th point
    double refine_ratio;   // bisect while hi/lo >= this; 0 = no bisection
    int min_reps;
    int max_reps;
    double confidence;     // one-sided, for the paired-ratio elimination
    bool alternate_reverse;
    double audit_fraction;
    double warm_topup_s;
    RefineMode refine_mode = RefineMode::geometric;
    double refine_margin = 0;  // also refine agreeing brackets whose runner-up is within this of the winner; 0 = off
};

// custom has no fixed protocol (expert overrides): it takes the coarse values as its base.
inline const TierParams& params(Tier t) {
    static const TierParams preview{2, 1.1, 3, 6, 0.80, false, 0.02, 0.2, RefineMode::index, 0.10};
    static const TierParams coarse{1, 1.1, 4, 12, 0.90, false, 0.02, 0.2};
    static const TierParams deep{1, 1.1, 6, 16, 0.98, true, 0.10, 0.2};
    switch (t) {
        case Tier::preview: return preview;
        case Tier::deep: return deep;
        default: return coarse;
    }
}

}  // namespace batchlas::tune
