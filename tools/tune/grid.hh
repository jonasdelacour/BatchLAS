#pragma once

// Tier lattices and multi-axis bisection (docs/design/tiered-tuning.md, "Engine: tiers"). Host-only.

#include "tier.hh"
#include "tune_core.hh"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace batchlas::tune {

struct AxisSpec {
    std::string name;
    bool log;
    std::vector<std::string> values;
    double weight = 1;  // distance weight of a log axis ("order:log:2")
};

std::uint64_t fnv1a64(std::string_view s);

// key_names entries are "name:exact" or "name:log[:w]"; an axis with no entry (hidden grid axes) is exact.
std::vector<AxisSpec> axis_specs(const std::vector<std::string>& key_names,
                                 const std::vector<std::pair<std::string, std::vector<std::string>>>& axes);

std::vector<std::string> subsample_axis(const std::vector<std::string>& values, int stride);

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, Tier t);

// For grids that are not lattices: keep a cell when fnv1a64(key_arg(cell)) % stride == 0.
std::vector<CellKey> tier_subsample(const std::vector<CellKey>& grid, Tier t);

// What the refinement rules read of one ranked cell. evidence: docs/design/tiered-tuning.md#engine-refinement-convergence-rules
struct RefineCell {
    std::map<std::string, double> ms;  // timed candidates (ok or eliminated): median ms
    std::set<std::string> out;         // not runnable here, or eliminated without a median
    bool lattice = false;              // a starting-lattice (round 0) cell of this run
};

// Flips (both ends decisive by > tie) and margin hedges (two lattice ends) refine; index mode refills
// AxisSpec::values, then geometric; `batch` only refills. The rules: the evidence page above.
struct RefineOpts {
    RefineMode mode = RefineMode::geometric;
    double margin = 0;
    const std::map<CellKey, RefineCell>* cells = nullptr;  // a missing cell: differing winners flip, no margin
    double tie = 0.03;
};

// The refinement cells an (op, dtype) may still measure: floor(cap_factor x lattice_cells) - refined.
std::size_t refine_allowance(std::size_t lattice_cells, std::size_t refined, double cap_factor);

// RefineRound::next = flip midpoints, then margin midpoints; stalled = a midpoint without a winner.
RefineRound refine_all_axes(const std::map<CellKey, std::vector<std::string>>& ranked,
                            const std::vector<AxisSpec>& axes, double ratio, const RefineOpts& opts = {});

}  // namespace batchlas::tune
