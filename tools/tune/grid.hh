#pragma once

// Tier lattices and multi-axis bisection (docs/design/tiered-tuning.md, "Engine: tiers"). Host-only.

#include <tools/tune/tier.hh>
#include <tools/tune/tune_core.hh>

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
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

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, int stride);  // log axes: every stride-th value + last
std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, Tier t);       // at params(t).stride

// For grids that are not lattices: keep a cell when fnv1a64(key_arg(cell)) % stride == 0.
std::vector<CellKey> tier_subsample(const std::vector<CellKey>& grid, Tier t);

// What the refinement rules read of one ranked cell. evidence: docs/design/tiered-tuning.md#engine-refinement-convergence-rules
struct RefineCell {
    std::map<std::string, double> ms;  // timed candidates (ok or eliminated): median ms
    std::set<std::string> out;         // not runnable here, or eliminated without a median
    bool lattice = false;              // a starting-lattice (round 0) cell of this run
    std::string single;                // an untimed runnable arm (skip:single): the winner when ms is empty
};

// Flips (decisive by > tie at either end) and margin hedges (two lattice ends) refine; index mode refills
// AxisSpec::values, then geometric; `batch` only refills. The rules: the evidence page above.
struct RefineOpts {
    RefineMode mode = RefineMode::geometric;
    double margin = 0;
    const std::map<CellKey, RefineCell>* cells = nullptr;  // a missing cell: differing winners flip, no margin
    double tie = 0.03;
};

// The refinement cells an (op, dtype) may still measure: floor(cap_factor x lattice_cells) - refined.
std::size_t refine_allowance(std::size_t lattice_cells, std::size_t refined, double cap_factor);

// The one host port of select::choose's row distance (Table::nearest_scan, sweep_to_table.py nearest()).
class NearestIndex {
public:
    NearestIndex(const std::vector<CellKey>& rows, const std::vector<AxisSpec>& axes);  // keys in `axes` order
    static constexpr std::size_t npos = std::size_t(-1);
    // Exact-key prefix dropped from the right, never past a leading `dtype` (select keeps one table per dtype):
    // npos when no row shares it. Ties: smaller log keys.
    std::size_t find(const CellKey& key) const;
    std::vector<std::size_t> k_nearest(const CellKey& key, std::size_t k) const;  // own exact-key group only

private:
    struct Row {
        std::vector<std::string> exact;
        std::vector<std::int64_t> logs;
        std::vector<double> lg;
    };
    Row prepare(const CellKey& k, bool query) const;
    double distance(const Row& r, const Row& q) const;
    std::vector<AxisSpec> axes_;
    std::vector<double> weights_;
    std::size_t min_prefix_ = 0;
    std::vector<Row> rows_;
    std::vector<std::size_t> order_;                                      // (exact strings, log integers) order
    std::map<std::vector<std::string>, std::vector<std::size_t>> group_;  // exact keys -> rows, in order_
};

std::size_t nearest_row(const std::vector<CellKey>& rows, const CellKey& key, const std::vector<AxisSpec>& axes);

// The unmeasured `universe` cells (in order) whose k nearest measured cells hold a decisive flip at > `slower`.
// evidence: docs/design/tiered-tuning.md#engine-flip-frontier-and-the-slowdown-replay
std::vector<CellKey> flip_frontier(const std::map<CellKey, RefineCell>& measured, const std::vector<CellKey>& universe,
                                   const std::vector<AxisSpec>& axes, std::size_t k = 6, double slower = 0.05);

// The calibrate tier's rounds: measured cells against the prior's pick at any cell ("" = no prior row; no Pick:
// flip_frontier).
// evidence: docs/design/tiered-tuning.md#engine-calibrate-tier-from-a-borrowed-table
class CalibrateRound {
public:
    using Pick = std::function<std::string(const CellKey&)>;
    CalibrateRound(const std::map<CellKey, RefineCell>& measured, const std::vector<AxisSpec>& axes, Pick pick,
                   std::size_t k = 6, double slower = 0.05);
    bool disagrees(const CellKey& m) const;  // the device's winner beats m's pick by > slower, or the pick cannot run
    bool disagrees_near(const CellKey& u, std::size_t n) const;  // one of u's n nearest measured cells disagrees
    std::vector<CellKey> frontier(const std::vector<CellKey>& open) const;  // u's pick beaten at a disagreeing neighbour
    std::set<std::string> winners(const CellKey& u) const;                  // of u's k nearest measured cells
    std::vector<CellKey> flips(const std::vector<CellKey>& open) const;     // flip_frontier's rule; ignores the pick

private:
    const std::map<CellKey, RefineCell>& measured_;
    std::vector<AxisSpec> axes_;
    Pick pick_;
    std::size_t k_;
    double slower_;
    std::vector<CellKey> rows_;
    std::vector<const RefineCell*> info_;
    std::vector<std::string> winner_;
    std::vector<bool> dis_;
    std::optional<NearestIndex> index_;
};

// Round 0: cells on tier_lattice(axes, stride) if global; a group without one: its own values' stride, else whole.
std::vector<CellKey> frontier_start(const std::vector<CellKey>& cells, const std::vector<AxisSpec>& axes, int stride,
                                    bool global = true);

// The CU-aligned batch ladder: max(1, round(c x cus)) per class c, plus `extra`, ascending and unique.
// evidence: docs/design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder
inline const std::vector<double>& default_batch_classes() {
    static const std::vector<double> c{0.25, 2, 16, 128};
    return c;
}
std::vector<double> without_latency_class(const std::vector<double>& classes);  // the classes >= 1 (one CU or more)
std::vector<std::int64_t> batch_ladder(int cus, const std::vector<double>& classes, const std::vector<int>& extra = {});
std::vector<std::string> batch_strings(const std::vector<std::int64_t>& batches);
// `batches` plus 65536 and 131072: level-3 runs past the grid-y cap as slices, so its ladder extras and
// choice.hh axes take rows there. evidence: docs/design/tiered-tuning.md#engine-the-cu-aligned-batch-ladder
std::vector<int> with_slice_batches(std::vector<int> batches);

// `grid` with each line's (the cell less `batch`) batch values replaced by `batches`, in first-seen line order.
// A line the grid declares at one batch only (syrk's trans=C at batch 1) keeps it.
std::vector<CellKey> rebatch(const std::vector<CellKey>& grid, const std::vector<std::string>& batches);

// RefineRound::next = flip midpoints, then margin midpoints; stalled = a midpoint without a winner.
RefineRound refine_all_axes(const std::map<CellKey, std::vector<std::string>>& ranked,
                            const std::vector<AxisSpec>& axes, double ratio, const RefineOpts& opts = {});

}  // namespace batchlas::tune
