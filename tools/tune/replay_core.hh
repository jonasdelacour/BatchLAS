#pragma once

// Offline replay of the tiered tuner against an exhaustive raw sweep (docs/design/tiered-tuning.md,
// "Engine: replay results on the trsm data"). Host-only: no SYCL, no GPU.

#include "grid.hh"
#include "tier.hh"
#include "tune_core.hh"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace batchlas::tune {

// One measured cell of the raw file: the final attempt's reps of the candidates that were "ok" in
// every pass of it. rounds[c][r] pairs the reps in time order p1r0, p2r0, p1r1, p2r1, ...
struct ReplayCell {
    CellKey key;  // the table keys only, in '# keys:' order (hidden grid fields dropped)
    std::vector<std::string> cands;
    std::vector<std::vector<double>> rounds;
    std::map<std::string, double> exhaustive;  // final_times: the mean of the pass medians
    int round = 0;                              // 0 = starting lattice of the raw run
};

struct ReplayMeta {
    std::vector<std::string> key_names;  // "order:log:2" entries of meta.keys
    std::vector<std::string> candidates;
    std::vector<AxisSpec> axes;  // round-0 values per key, skipped cells included
};

// Throws std::runtime_error without a meta record. Cells whose status is not "ok" are not returned.
std::vector<ReplayCell> load_replay(const std::string& raw_jsonl, ReplayMeta* meta = nullptr);

struct ReplayReport {
    std::size_t cells = 0, cells_measured = 0;
    double reps_fraction = 0;  // candidate-reps the replay timed / candidate-reps in the file
    double race_misrank = 0;   // measured cells whose first-ranked candidate is > 1.03x the exhaustive best
    double table_misrank = 0;  // all cells whose select::choose pick (first runnable entry of the nearest row) is > 1.03x, or none runs
    double table_misrank_lattice = 0;  // table_misrank over round-0 cells only
    double mean_loss = 0, p95_loss = 0, p99_loss = 0, max_loss = 0;  // exhaustive(chosen) / best - 1, runnable cells
    double time_weighted_loss = 0;                                   // sum(chosen) / sum(best) - 1, runnable cells
    std::size_t unrunnable = 0;                                      // cells where no entry of the nearest row can run
    double est_gpu_h = 0;                                            // set by the caller: reps_fraction x exhaustive hours
    std::size_t refine_unavailable = 0;  // distinct bisection midpoints that are not in the raw file
    std::vector<std::string> worst;      // the five worst misranks of either kind
};

ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t,
                    const TierParams& p, double tie = 0.03, const RefineOpts& ro = {});
ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t, double tie = 0.03);

// "--axis-keep name=v1:v2" / "--axis-stride name=k": shrink one axis's starting lattice in place
// (stride keeps every k-th value and the last). Throws std::invalid_argument on an unknown axis, a value
// the axis lacks or a bad stride.
void shrink_axis(std::vector<AxisSpec>& axes, const std::string& name, const std::string& spec, bool keep);

// Port of nearest() in scripts/sweep_to_table.py (exact-key prefix, weighted log2 distance, ties as there).
std::size_t nearest_row(const std::vector<CellKey>& rows, const CellKey& key, const std::vector<AxisSpec>& axes);

}  // namespace batchlas::tune
