#pragma once

// Offline replay of the tiered tuner against an exhaustive raw sweep, and the slowdown replay of planning
// schemes (docs/design/tiered-tuning.md, "Engine: replay results on trsm" and the next section). Host-only.

#include <tools/tune/grid.hh>
#include <tools/tune/tier.hh>
#include <tools/tune/tune_core.hh>

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
    int passes = 2;
};

struct ReplayMeta {
    std::vector<std::string> key_names;  // "order:log:2" entries of meta.keys
    std::vector<std::string> candidates;
    std::vector<AxisSpec> axes;  // round-0 values per key, skipped cells included
    double total_rep_ms = 0;     // every `rep` record of the file: the exhaustive run's summed rep time
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
    double floor_race = 0, floor_table = 0, excess_race = 0, excess_table = 0;  // see apply_floor
    double measure_s = 0;  // rep time the replay consumed, alone
    double est_gpu_h = 0;  // (measure_s + lean_arm_fixed_s per candidate + verify_s + cell_overhead_s per cell) / 3600
    std::size_t refine_unavailable = 0;  // distinct bisection midpoints that are not in the raw file
    std::size_t cells_lattice = 0;       // round-0 cells measured; cells_measured - this = refinement cells
    std::size_t refine_capped = 0;       // midpoints the refinement cap dropped (0 = the cap was not hit)
    std::vector<std::string> worst;      // the five worst misranks of either kind
};

// holdout: race on pass-1 reps only (no reversed round), score against the pass-2 median; off: shared samples.
struct ReplayOpts {
    bool holdout = false;
    double verify_s = 0.05;        // per cell: the tie set's residual check
    double cell_overhead_s = 0;    // per cell; 0 = persistent worker
    bool all_cells = false;        // race every cell of the file: no lattice, no bisection (the oracle)
    std::vector<std::string> order;  // tie order (meta.candidates); empty = first appearance in the cells
};
ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t,
                    const TierParams& p, double tie = 0.03, const ReplayOpts& o = {});
ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t, double tie = 0.03);

// "--axis-keep name=v1:v2" / "--axis-stride name=k": shrink one axis's starting lattice in place
// (stride keeps every k-th value and the last). Throws std::invalid_argument on an unknown axis, a value
// the axis lacks or a bad stride.
void shrink_axis(std::vector<AxisSpec>& axes, const std::string& name, const std::string& spec, bool keep);

// Holdout noise floor: the no-elimination oracle (run with ReplayOpts::all_cells) on the same pass-2 reference.
TierParams oracle_params();
void apply_floor(ReplayReport& r, const ReplayReport& oracle);

// Slowdown replay: a measured universe scored t(pick) / t(fastest) per cell under a planning scheme.
// evidence: docs/design/tiered-tuning.md#engine-flip-frontier-and-the-slowdown-replay
struct OracleSet {
    std::string label;
    std::vector<AxisSpec> axes;           // log axes: the round-0 lattice values when known, else every distinct value
    std::map<CellKey, RefineCell> cells;  // ms = each timed arm's median; out = the set's other arms; dtype is a key
    std::string lattice = "inferred";     // ledger (round-0 records), raw (sweep meta), inferred (distinct values)
};

OracleSet load_table_oracle(const std::string& tuned_txt);  // timed rows only; transcribed (`-`) rows are dropped
std::map<std::string, std::string> table_family_hashes(const std::string& tuned_txt);  // header family_kernels=
// Current best records; the lattice is the round-0 records' values (rare strays dropped).
OracleSet load_ledger_oracle(const std::string& ledger_jsonl, const std::map<std::string, std::string>& family_hash);
OracleSet raw_oracle(const std::string& label, const std::vector<ReplayCell>& cells, const ReplayMeta& meta);
void adopt_lattice(OracleSet& o, const std::vector<AxisSpec>& lattice);  // log axes take lattice's values, by name

struct Scheme {
    std::string name;
    int stride = 1;         // round 0: the universe cells on tier_lattice(axes, stride); 0 = every universe cell
    bool frontier = false;  // then flip_frontier rounds until none is left
    double prune = 0;       // > 0: a frontier cell races only arms within prune x the best at one of its neighbours
};
std::vector<Scheme> replay_schemes();  // full, lattice, stride2, stride4, stride2+frontier, stride4+frontier, stride2+frontier+prune

struct SlowdownReport {
    std::size_t cells = 0, measured = 0, rounds = 0;
    double arm_runs = 0, arm_total = 0;          // timed arms raced / timed arms in the universe
    std::vector<double> ratio, ratio_measured;   // t(pick) / t(fastest): every scored cell; measured cells alone
    double worst = 1;
    std::string worst_key;
    std::vector<std::string> failures;  // no entry of the nearest measured row runs, or no row of the dtype (not scored)
    std::string lattice;
};
// Round 0: the stride lattice; an exact-key group with no cell on it takes the stride of its own values.
// Picks: a measured cell's fastest raced arm; any other cell reads the nearest measured row (NearestIndex::find,
// as select::choose) and takes its first entry, fastest first, that runs there.
SlowdownReport slowdown_replay(const OracleSet& o, const Scheme& s, std::size_t k = 6, double slower = 0.05);

struct SlowdownStats {
    double geomean = 0;  // geomean(ratio) - 1
    double p99 = 1, p999 = 1, worst = 1;  // nearest rank: sorted[ceil(f n) - 1]
    double slow10 = 0, misrank = 0;  // share of cells > 1.10, > 1.03
};
SlowdownStats slowdown_stats(std::vector<double> ratio);

constexpr double kGateGeomean = 0.0005, kGateP999 = 1.10;  // evidence: docs/design/tiered-tuning.md#engine-flip-frontier-and-the-slowdown-replay
// Every cell scored (no failures, something measured), geomean <= kGateGeomean and p99.9 <= kGateP999.
bool slowdown_gate(const SlowdownReport& r);

}  // namespace batchlas::tune
