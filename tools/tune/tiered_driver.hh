#pragma once

// Breadth-first tiered rounds (docs/design/tiered-tuning.md, "Engine: driver interface"). SYCL-free:
// the GPU side stays in batchlas_tune.cc behind CellMeasurer.

#include "schedule.hh"
#include "spec.hh"
#include "tier.hh"

#include <map>
#include <string>
#include <vector>

namespace batchlas::tune {

struct TieredOpts {
    std::vector<std::string> ops;  // in op_order
    std::vector<std::string> dtypes;
    std::vector<int> devices;
    Tier tier = Tier::preview;
    std::int64_t max_dim = 2048;  // 0 = off: skip:dim above this matrix dimension
    double budget_h = 0, cap_gib = 4, overhead_s = kChildOverheadS;
    double audit_fraction = -1;     // < 0: the tier's
    double refine_cap_factor = -1;  // < 0: the tier's
    bool plan = false;
    int progress_fd = -1;
    std::string repo, ledger_root, out, argv;
    std::string run_id;  // empty: make_run_id()
    std::map<std::string, std::vector<std::string>> grid;
};

struct RunIdentity {
    std::string device, device_name;
};

// One cell to race. `arms` are in seed order: the nearest finished cell's winner first.
struct CellJob {
    int gpu = 0;
    const OpSpec* spec = nullptr;
    std::string dtype;
    CellKey key;
    std::vector<std::string> arms;
    Tier tier = Tier::preview;
    TierParams p{};
    double footprint = 0;  // item_footprint: a worker restarts before a smaller one
    std::vector<std::string> ld_audit;  // arms not yet ld-audited in this (op, dtype): CellRequest::ld_audit
};

// One outcome per requested arm; `error` names a child failure that hit every arm.
struct ArmBatch {
    std::vector<ArmOutcome> arms;
    std::string error;
    int worker_restarts = 0;  // the worker died on this cell and was restarted
    bool fallback = false;    // ... and the cell ran in a fresh child after all
    std::vector<std::string> alone;  // arms raced alone in fresh children: not the worker's numbers
};

// The GPU seam: measure() is the default path, the persistent worker when persistent();
// measure_fresh() is one fresh `--cell --mode race` child (the audit, and an op and dtype marked fresh).
class CellMeasurer {
public:
    virtual ~CellMeasurer() = default;
    virtual ArmBatch measure(const CellJob& j) = 0;
    virtual ArmBatch measure_fresh(const CellJob& j) { return measure(j); }
    virtual bool persistent() const { return false; }
};

// `m` may be null only with o.plan, which prints the starting lattice's plan and touches no GPU.
int run_tiered(const TieredOpts& o, const RunIdentity& id, CellMeasurer* m);

// The op x dtype x device matrix from the ledgers and tables; no GPU.
int status_main(const std::string& repo, const std::string& ledger_root, const std::string& tuned_dir);

// A schema-1 raw sweep into the ledger as a deep run, hashed against the current sources.
int import_raw_main(const std::string& repo, const std::string& ledger_root, const std::string& raw);

std::string git_head(const std::string& repo);  // 8-hex HEAD, "-dirty" when tracked files differ

}  // namespace batchlas::tune
