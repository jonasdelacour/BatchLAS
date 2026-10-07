#pragma once

// Tier lattices and multi-axis bisection (docs/design/tiered-tuning.md, "Engine: tiers"). Host-only.

#include "tier.hh"
#include "tune_core.hh"

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace batchlas::tune {

struct AxisSpec {
    std::string name;
    bool log;
    std::vector<std::string> values;
};

std::uint64_t fnv1a64(std::string_view s);

// key_names entries are "name:exact" or "name:log[:w]"; an axis with no entry (hidden grid axes) is exact.
std::vector<AxisSpec> axis_specs(const std::vector<std::string>& key_names,
                                 const std::vector<std::pair<std::string, std::vector<std::string>>>& axes);

std::vector<std::string> subsample_axis(const std::vector<std::string>& values, int stride);

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, Tier t);

// For grids that are not lattices: keep a cell when fnv1a64(key_arg(cell)) % stride == 0.
std::vector<CellKey> tier_subsample(const std::vector<CellKey>& grid, Tier t);

// New cells to measure: per log axis and per line (cells equal in every other key), the midpoints
// between neighbours whose winners (ranked.front()) differ. Empty ranking = no winner; ratio 0 = off.
std::vector<CellKey> refine_all_axes(const std::map<CellKey, std::vector<std::string>>& ranked,
                                     const std::vector<AxisSpec>& axes, double ratio);

}  // namespace batchlas::tune
