#pragma once

// Host-only. evidence: docs/design/tiered-tuning.md#engine-tiers-and-the-per-cell-algorithm

#include "ledger.hh"
#include "spec.hh"
#include "tier.hh"
#include "tune_core.hh"

#include <cmath>
#include <functional>
#include <map>
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
};

struct PlannedCell {
    CellKey key;
    std::string reason;  // "" = measure, "skip:current", "skip:single", "skip:cap", "partial:<fams>"
    std::vector<std::string> arms;
    double est_s = 0;
    Tier tier = Tier::preview;            // a partial re-race keeps the stored record's tier
    const CellRecord* stored = nullptr;   // partial: the record the new arms merge into
};

// Nearest record's median (equal non-integer fields, log distance on integer ones), else bytes / 500 GB/s.
double estimate_ms(const Ledger& l, const CellKey& key, const std::string& cand, double bytes);

double cell_estimate_s(const Ledger& l, const CellKey& key, const std::vector<std::string>& arms, double bytes,
                       const TierParams& p, double overhead_s);

// Ascending bytes (stable). `runnable`: probe results per cell, nullptr = unknown (no skip:single).
std::vector<PlannedCell> plan_round(const PlanSpec& spec, Tier tier, const std::vector<CellKey>& cells, const Ledger& l,
                                    const std::map<std::string, std::string>& family_hash, double cap_gib,
                                    double per_cell_overhead_s,
                                    const std::map<CellKey, std::vector<std::string>>* runnable = nullptr);

// potrf and trsm before posv; otherwise input order.
std::vector<std::string> op_order(std::vector<std::string> ops);

// Empty when the estimate fits the budget (or there is none); the lattice runs either way.
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

bool audit_pick(const std::string& run_id, const CellKey& key, double fraction);

// verdict ok | mismatch:feasibility | mismatch:winner (beyond the tie in the fresh run) | inconclusive.
struct AuditResult {
    std::string verdict;
    double fresh_ms = NAN, warm_ms = NAN;
    bool mismatch() const { return verdict.rfind("mismatch", 0) == 0; }
};
AuditResult audit_compare(const std::vector<ArmOutcome>& warm, const std::vector<ArmOutcome>& fresh,
                          const std::vector<std::string>& order, double tie = 0.03);

// runner-up time / winner time - 1 (the preview refinement margin); +inf without a timed runner-up.
double runner_up_gap(const CellRecord& r);

}  // namespace batchlas::tune
