#pragma once

// Host-only. evidence: docs/design/tiered-tuning.md#engine-tiers-and-the-per-cell-algorithm

#include "grid.hh"
#include "ledger.hh"
#include "spec.hh"
#include "tier.hh"
#include "tune_core.hh"

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace batchlas::tune {

// Task 0's measured child start-up (threadripper02): docs/design/tiered-tuning.md#engine-measured-per-child-overhead
inline constexpr double kChildOverheadS = 0.49;
inline constexpr double kVerifyS = 0.05;        // per candidate, the untimed verification run
inline constexpr double kModelBytesPerS = 500e9;

struct PlanSpec {  // plain data, so tests need no OpSpec
    std::vector<std::string> candidates;  // the full current list, in tie order
    std::function<double(const CellKey&)> bytes;
    std::function<std::int64_t(const CellKey&)> max_dim;  // empty = no extents known (never skip:dim)
};

struct PlannedCell {
    CellKey key;
    std::string reason;  // "" = measure, "skip:current", "skip:single", "skip:cap", "skip:dim", "partial:<fams>"
    std::vector<std::string> arms;
    double est_s = 0;
    Tier tier = Tier::preview;            // a partial re-race keeps the stored record's tier
    const CellRecord* stored = nullptr;   // partial: the record the new arms merge into
};

double estimate_ms(const Ledger& l, const CellKey& key, const std::string& cand,
                   double bytes);  // nearest record (equal non-int fields, log distance), else bytes / 500 GB/s

double cell_estimate_s(const Ledger& l, const CellKey& key, const std::vector<std::string>& arms, double bytes,
                       const TierParams& p, double overhead_s);

// Ascending bytes (stable). `runnable`: probe results per cell, nullptr = unknown (no skip:single).
std::vector<PlannedCell> plan_round(const PlanSpec& spec, Tier tier, const std::vector<CellKey>& cells, const Ledger& l,
                                    const std::map<std::string, std::string>& family_hash, double cap_gib,
                                    double per_cell_overhead_s,
                                    const std::map<CellKey, std::vector<std::string>>* runnable = nullptr,
                                    std::int64_t max_dim = 0);  // > 0: a cell with a larger matrix dimension is skip:dim

// op -> the ops whose tables its timings read. evidence: docs/design/tiered-tuning.md#engine-op-order-over-all-19-ops
const std::map<std::string, std::vector<std::string>>& op_dependencies();
const std::vector<std::string>& canonical_op_order();  // the order `all` starts from
std::vector<std::string> all_ops(std::vector<std::string> registered);  // canonical order, unknown ones last
// Stable topological sort: an op follows every dependency in the list; input order breaks ties.
std::vector<std::string> op_order(std::vector<std::string> ops);

// Empty when the estimate (lattice and refinement) fits the budget (or there is none); the lattice runs either way.
std::string budget_warning(double est_s, double budget_h);

// Ranked by rank() at tie 0.03; merging into `stored` keeps its unchanged candidates, drops removed ones.
CellRecord record_from_arms(const CellKey& key, int round, const std::vector<ArmOutcome>& arms,
                            const std::vector<std::string>& order, const std::map<std::string, std::string>& family_hash,
                            const CellRecord* stored = nullptr);

// skip:single: `only` ranked untimed, every other candidate skipped.
CellRecord single_record(const CellKey& key, int round, const std::string& only, const std::vector<std::string>& order,
                         const std::map<std::string, std::string>& family_hash);

// `arms`, the nearest record's winner first (as estimate_ms picks the nearest): the race's seed order.
std::vector<std::string> seed_order(const std::map<CellKey, CellRecord>& done, const CellKey& key,
                                    const std::vector<std::string>& arms);

double item_footprint(double bytes, const CellKey& key);  // bytes / batch (bytes without a batch key)
// A worker's share: ascending per-item footprint, then bytes (stable): the carve-out order.
void sort_for_worker(std::vector<const PlannedCell*>& cells, const std::function<double(const CellKey&)>& bytes);

bool audit_pick(const std::string& run_id, const CellKey& key, double fraction);

// ok | mismatch:feasibility | mismatch:winner (> margin in the fresh run) | inconclusive (the fresh child ran nothing)
struct AuditResult {
    std::string verdict;
    double fresh_ms = NAN, warm_ms = NAN;
    bool mismatch() const { return verdict.rfind("mismatch", 0) == 0; }
};
AuditResult audit_compare(const std::vector<ArmOutcome>& warm, const std::vector<ArmOutcome>& fresh,
                          const std::vector<std::string>& order, double tie = 0.03, double margin = 0.10);

// One (op, dtype)'s ld audit (spec.hh kLdAuditPad). An arm the cell did not verify stays wanted.
struct LdAuditBook {
    std::set<std::string> done;  // arms with a verdict: pass, fail or skipped
    std::size_t passed = 0, skipped = 0;
    std::vector<std::string> failed;  // "<arm> @ <key>: <reason>"
    std::vector<std::string> want(const std::vector<std::string>& arms) const;  // those with no verdict yet
    std::vector<std::string> note(const CellKey& key, const std::vector<ArmOutcome>& out);  // returns this cell's fails
};

// The refinement view of a record (grid.hh RefineCell); `lattice` marks a round-0 cell of this run.
RefineCell refine_cell(const CellRecord& r, bool lattice);

// Refinement cells per lattice cell: this ledger's refined / round-0 records at `tier` (<= cap_factor), else cap_factor x 0.5.
double refine_ratio_estimate(const Ledger& l, Tier tier, double cap_factor, bool* from_history = nullptr);

}  // namespace batchlas::tune
