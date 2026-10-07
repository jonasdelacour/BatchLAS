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
    double budget_h = 0, cap_gib = 4, overhead_s = kChildOverheadS;
    bool plan = false;
    int progress_fd = -1;
    std::string repo, ledger_root, out, argv;
    std::map<std::string, std::vector<std::string>> grid;
};

struct RunIdentity {
    std::string device, device_name;
};

// One outcome per requested arm; `error` names a child failure that hit every arm.
struct ArmBatch {
    std::vector<ArmOutcome> arms;
    std::string error;
};

// THE SEAM Task 8 replaces: today one fresh `--cell --mode time` child per cell, one pass of
// max_reps reps after a warm_topup_s warm-up per arm; then the persistent worker's `--mode race`.
class CellMeasurer {
public:
    virtual ~CellMeasurer() = default;
    virtual ArmBatch measure(int gpu, const OpSpec& spec, const std::string& dtype, const CellKey& key,
                             const std::vector<std::string>& arms, const TierParams& p) = 0;
};

// `m` may be null only with o.plan, which prints the starting lattice's plan and touches no GPU.
int run_tiered(const TieredOpts& o, const RunIdentity& id, CellMeasurer* m);

// The op x dtype x device matrix from the ledgers and tables; no GPU.
int status_main(const std::string& repo, const std::string& ledger_root, const std::string& tuned_dir);

// A schema-1 raw sweep into the ledger as a deep run, hashed against the current sources.
int import_raw_main(const std::string& repo, const std::string& ledger_root, const std::string& raw);

std::string git_head(const std::string& repo);  // 8-hex HEAD, "-dirty" when tracked files differ

}  // namespace batchlas::tune
