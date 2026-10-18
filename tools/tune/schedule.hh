#pragma once

// Host-only. evidence: docs/design/tiered-tuning.md#engine-tiers-and-the-per-cell-algorithm

#include <tools/tune/grid.hh>
#include <tools/tune/ledger.hh>
#include <tools/tune/spec.hh>
#include <tools/tune/tier.hh>
#include <tools/tune/tune_core.hh>

#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace batchlas::tune {

// Task 0's measured child start-up (threadripper02): docs/design/tiered-tuning.md#engine-measured-per-child-overhead
inline constexpr double kChildOverheadS = 0.49;
inline constexpr double kVerifyS = 0.05;        // per cell: the tie set's untimed verification run
inline constexpr double kModelBytesPerS = 500e9;
inline constexpr double kFrontierShare = 0.86;  // --plan. evidence: docs/design/tiered-tuning.md#engine-round-planning-by-flip-frontier

struct PlanSpec {  // plain data, so tests need no OpSpec
    std::vector<std::string> candidates;  // the full current list, in tie order
    std::function<double(const CellKey&)> bytes;
    std::function<std::int64_t(const CellKey&)> max_dim;  // empty = no extents known (never skip:dim)
    std::string mode;  // worker | fresh: the phase history the estimates read; "" = either
};

struct PlannedCell {
    CellKey key;
    std::string reason;  // "" = measure, "remeasure", "skip:current", "skip:single", "skip:cap", "skip:dim", "partial:<fams>"
    std::vector<std::string> arms;
    double est_s = 0;
    Tier tier = Tier::preview;            // a partial re-race keeps the stored record's tier
    const CellRecord* stored = nullptr;   // partial: the record the new arms merge into
    int round = -1;                       // remeasure: the stored record's round (-1: the running round)
    bool history = false;                 // est_s from recorded phases, not the model
};

double estimate_ms(const Ledger& l, const CellKey& key, const std::string& cand,
                   double bytes);  // nearest record (equal non-int fields, log distance), else bytes / 500 GB/s

double cell_estimate_s(const Ledger& l, const CellKey& key, const std::vector<std::string>& arms, double bytes,
                       Tier tier, double overhead_s, bool* from_history = nullptr,
                       const std::string& mode = "");  // the plan's rule: kProtocol history at tier + mode, else the model

double history_cell_overhead_s(const Ledger& l, Tier tier, const std::string& mode, double audit_fraction,
                               bool* from_history = nullptr,  // per history cell: worker starts + audits
                               double* ld_once_s = nullptr);  // one run's ld audits (its cells' sum, median)

// Ascending bytes (stable). `runnable`: probe results per cell, nullptr = unknown (no skip:single).
std::vector<PlannedCell> plan_round(const PlanSpec& spec, Tier tier, const std::vector<CellKey>& cells, const Ledger& l,
                                    const std::map<std::string, std::string>& family_hash, double cap_gib,
                                    double per_cell_overhead_s,
                                    const std::map<CellKey, std::vector<std::string>>* runnable = nullptr,
                                    std::int64_t max_dim = 0,  // > 0: a cell with a larger matrix dimension is skip:dim
                                    const std::set<CellKey>* remeasure = nullptr);  // missing, past max_dim too: "remeasure"

// --remeasure-keys: "<op> <dtype> <key_arg>" lines -> "op.dtype" -> cells; throws std::runtime_error on a bad
// line. evidence: docs/design/tiered-tuning.md#engine-re-measuring-named-cells
std::map<std::string, std::set<CellKey>> read_remeasure_keys(const std::string& path);

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
// A fail is a verdict on the kernel, so the arm is `bad` in every cell of the (op, dtype).
struct LdAuditBook {
    std::set<std::string> done;     // arms with a verdict: pass or fail
    std::set<std::string> refused;  // the pin refused the padded shape (skipped): still wanted
    std::map<std::string, std::string> bad;  // failed arm -> the reason its other cells record
    std::size_t passed = 0;
    std::vector<std::string> failed;  // "<arm> @ <key>: <reason>"
    std::vector<std::string> want(const std::vector<std::string>& arms) const;  // those with no verdict yet
    std::vector<std::string> note(const CellKey& key, const std::vector<ArmOutcome>& out);  // returns this cell's fails
    std::size_t skipped() const;  // refused at every cell that asked, never audited
};

// evidence: docs/design/tiered-tuning.md#engine-tie-set-verification
std::string verify_bucket(const CellKey& key, const std::vector<AxisSpec>& axes, double item_bytes);  // exact keys, decade

struct VerifyBook {  // one (op, dtype)'s, per verify_bucket
    std::map<std::string, std::set<std::string>> done;              // bucket -> arms with a verdict
    std::map<std::string, std::map<std::string, std::string>> bad;  // bucket -> failed arm -> reason
    std::vector<std::string> want(const std::string& bucket, const std::vector<std::string>& arms) const;
    std::vector<std::string> note(const std::string& bucket, const CellKey& key,
                                  const std::vector<ArmOutcome>& out);  // returns the bucket's new fails
};

// `arm` ok or eliminated in r -> bad with `reason`, out of r.ranked; false when r did not rank it.
bool ld_retract(CellRecord& r, const std::string& arm, const std::string& reason,
                bool unverified_only = false);  // true: only where r did not verify it (a VerifyBook fail)

// r.ranked's front becomes an arm r verified: unverified arms above it move below it; none verified: emptied.
bool verified_front(CellRecord& r);  // true: the front changed (the driver races the cell again)

// The refinement view of a record (grid.hh RefineCell); `lattice` marks a round-0 cell of this run.
RefineCell refine_cell(const CellRecord& r, bool lattice);

// Refinement cells per lattice cell: this ledger's refined / round-0 records at `tier` (<= cap_factor), else cap_factor x 0.5.
double refine_ratio_estimate(const Ledger& l, Tier tier, double cap_factor, bool* from_history = nullptr);

}  // namespace batchlas::tune
