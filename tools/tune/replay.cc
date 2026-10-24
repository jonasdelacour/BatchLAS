// SPDX-License-Identifier: MIT
// tune_replay: runs the tiered tuner's race and bisection against an exhaustive raw sweep, or scores planning schemes, no GPU.
//   tune_replay --raw <jsonl> --tier preview|coarse|deep [--confidence x] [--min-reps n] [--max-reps n] [--stride n] [--refine r] [--verify-s s] [--cell-overhead-s s] [--no-holdout] [--axis-keep name=v:v] [--axis-stride name=k] [--refine-mode geometric|index] [--refine-margin m] [--refine-cap f] [--print-axes]
// Prints the ReplayReport as one JSON line (docs/design/tiered-tuning.md).
//   tune_replay --slowdown [--table <tuned/op.dev.txt>]... [--ledger <op.dev.jsonl>]... [--raw <jsonl>]... [--scheme a,b] [--k n] [--slower f]
//               [--prior <tuned/op.dev.txt>]... [--batch-ladder CUS [--batch-classes c,c] [--batch-extra b,b]]
//               [--drop-nonmonotone F] [--min-batch B]
// Slowdown replay: one JSON line per (scheme, source), then per (scheme, overall/<kind>). --table or --ledger
// implies --slowdown; a ledger takes its family hashes from the --table of the same <op>.<device>. A --prior adds
// the calibrate schemes for the sources of its op (calibrate_replay). --batch-ladder adds the schemes `ladder` and
// `ladder+frontier+bisect`: only each group's batches nearest the CU classes are measured, every cell is scored.
// --drop-nonmonotone drops co-tenant-contaminated cells first (drop_nonmonotone_batch).

#include <tools/tune/replay_core.hh>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace batchlas::tune;

namespace {

void print_slowdown(const std::string& scheme, const std::string& source, const SlowdownReport& r) {
    const SlowdownStats all = slowdown_stats(r.ratio), mine = slowdown_stats(r.ratio_measured);
    std::string fails;
    for (const std::string& f : r.failures) fails += (fails.empty() ? "" : "; ") + f;
    Json j;
    j.str("scheme", scheme).str("source", source).str("lattice", r.lattice).integer("cells", std::int64_t(r.cells)).integer("measured", std::int64_t(r.measured));
    j.num("kept", r.cells ? double(r.measured) / double(r.cells) : 0).num("arm_runs", r.arm_total > 0 ? r.arm_runs / r.arm_total : 0);
    j.integer("rounds", std::int64_t(r.rounds)).integer("bisect_missing", std::int64_t(r.bisect_missing));
    j.num("geomean", all.geomean).num("p99", all.p99).num("p999", all.p999);
    j.num("slow10", all.slow10).num("misrank", all.misrank).num("worst", r.worst).str("worst_key", r.worst_key);
    j.num("measured_geomean", mine.geomean).num("measured_p99", mine.p99).num("measured_p999", mine.p999);
    j.num("measured_slow10", mine.slow10).num("measured_worst", mine.worst);
    j.integer("probes", std::int64_t(r.probes)).integer("disagreed", std::int64_t(r.disagreed));
    j.integer("prior_rows", std::int64_t(r.prior_rows));
    j.integer("failures", std::int64_t(r.failures.size())).str("failure_list", fails);
    j.str("gate", slowdown_gate(r) ? "PASS" : "FAIL");
    std::fputs(j.line().c_str(), stdout);
}

void pool(SlowdownReport& into, const SlowdownReport& r) {
    into.lattice = into.lattice.empty() || into.lattice == r.lattice ? r.lattice : "mixed";
    into.cells += r.cells, into.measured += r.measured, into.rounds = std::max(into.rounds, r.rounds);
    into.bisect_missing += r.bisect_missing;
    into.probes += r.probes, into.disagreed += r.disagreed, into.prior_rows += r.prior_rows;
    into.arm_runs += r.arm_runs, into.arm_total += r.arm_total;
    into.ratio.insert(into.ratio.end(), r.ratio.begin(), r.ratio.end());
    into.ratio_measured.insert(into.ratio_measured.end(), r.ratio_measured.begin(), r.ratio_measured.end());
    into.failures.insert(into.failures.end(), r.failures.begin(), r.failures.end());
    if (r.worst > into.worst) into.worst = r.worst, into.worst_key = r.worst_key;
}

int slowdown_main(int argc, char** argv) {
    std::vector<std::string> tables, ledgers, raws, want, priors;
    std::size_t k = 6;
    double slower = 0.05, drop = 0;
    int cus = 0;
    std::int64_t min_batch = 0;
    std::vector<double> classes = default_batch_classes();
    std::vector<int> extra;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slowdown") continue;
        if (i + 1 >= argc) {
            std::fprintf(stderr, "tune_replay: %s needs a value\n", a.c_str());
            return 2;
        }
        const std::string v = argv[++i];
        if (a == "--table") tables.push_back(v);
        else if (a == "--ledger") ledgers.push_back(v);
        else if (a == "--raw") raws.push_back(v);
        else if (a == "--prior") priors.push_back(v);
        else if (a == "--scheme") want = split(v, ',');
        else if (a == "--k") k = std::size_t(std::stoul(v));
        else if (a == "--slower") slower = std::stod(v);
        else if (a == "--batch-ladder") cus = std::stoi(v);
        else if (a == "--batch-classes") {
            classes.clear();
            for (const std::string& c : split(v, ',')) classes.push_back(std::stod(c));
        } else if (a == "--batch-extra") {
            for (const std::string& b : split(v, ',')) extra.push_back(std::stoi(b));
        } else if (a == "--drop-nonmonotone") drop = std::stod(v);
        else if (a == "--min-batch") min_batch = std::stoll(v);
        else {
            std::fprintf(stderr, "tune_replay: unknown argument %s in --slowdown mode\n", a.c_str());
            return 2;
        }
    }
    std::vector<Scheme> known = replay_schemes(), schemes;
    for (const Scheme& s : calibrate_schemes())
        if (!priors.empty()) known.push_back(s);
    if (cus > 0) known.push_back({"ladder", 0, false, 0}), known.push_back({"ladder+frontier+bisect", 2, true, 0, true});
    for (const Scheme& s : known)
        if (want.empty() || std::find(want.begin(), want.end(), s.name) != want.end()) schemes.push_back(s);
    if (schemes.size() != (want.empty() ? known.size() : want.size())) {
        std::fprintf(stderr, "tune_replay: --scheme takes a comma list of full, lattice, stride2, stride4, stride2+frontier, "
                             "stride4+frontier, stride2+frontier+prune, lattice+bisect, stride2+frontier+bisect; with "
                             "--prior also calibrate, calibrate8, calibrate-probe, prior; with --batch-ladder also ladder, "
                             "ladder+frontier+bisect\n");
        return 2;
    }
    std::map<std::string, std::unique_ptr<Prior>> prior_of;  // op -> its --prior
    for (const std::string& p : priors) {
        const std::string stem = std::filesystem::path(p).stem().string();
        prior_of[stem.substr(0, stem.find('.'))] = std::make_unique<Prior>(p);
    }
    std::vector<std::pair<std::string, OracleSet>> sources;  // kind, set
    std::map<std::string, std::string> table_of;              // <op>.<device> -> table path
    for (const std::string& t : tables) {
        sources.emplace_back("table", load_table_oracle(t));
        table_of[std::filesystem::path(t).stem().string()] = t;
    }
    const std::size_t n_tables = sources.size();
    for (const std::string& l : ledgers) {
        const std::string stem = std::filesystem::path(l).stem().string();
        if (!table_of.count(stem)) {
            std::fprintf(stderr, "tune_replay: --ledger %s needs --table .../%s.txt for its family hashes\n", l.c_str(), stem.c_str());
            return 2;
        }
        sources.emplace_back("ledger", load_ledger_oracle(l, table_family_hashes(table_of[stem])));
        sources.back().second.label = stem.substr(0, stem.find('.')) + "/ledger";
        for (std::size_t i = 0; i < n_tables; ++i)  // a table starts from its ledger's round-0 lattice
            if (std::filesystem::path(tables[i]).stem().string() == stem) {
                adopt_lattice(sources[i].second, sources.back().second.axes);
                sources[i].second.lattice = "ledger";
            }
    }
    for (const std::string& r : raws) {
        ReplayMeta meta;
        const auto cells = load_replay(r, &meta);
        sources.emplace_back("raw", raw_oracle(std::filesystem::path(r).stem().string() + "/raw", cells, meta));
    }
    const auto ladder = cus > 0 ? batch_ladder(cus, classes, extra) : std::vector<std::int64_t>{};
    std::string ladder_text;
    for (const std::string& b : batch_strings(ladder)) ladder_text += (ladder_text.empty() ? "" : ",") + b;
    std::vector<std::set<std::string>> keep(sources.size());
    for (std::size_t i = 0; i < sources.size(); ++i) {
        OracleSet& o = sources[i].second;
        const std::size_t before = o.cells.size(), dropped = drop > 0 ? drop_nonmonotone_batch(o, drop) : 0;
        for (auto it = o.cells.begin(); it != o.cells.end();)  // --min-batch: score (and measure) only batch >= it
            it = key_get(it->first, "batch") && key_int(it->first, "batch") < min_batch ? o.cells.erase(it) : std::next(it);
        std::string kept;
        if (cus > 0) keep[i] = nearest_batches(o, ladder, extra, &kept);
        if (drop > 0 || cus > 0 || min_batch > 0)
            std::fputs(Json().str("source", o.label).integer("cells", std::int64_t(before)).integer("dropped", std::int64_t(dropped))
                           .str("ladder", ladder_text).str("kept_batches", kept).line().c_str(), stdout);
    }
    for (Scheme s : schemes) {
        std::map<std::string, SlowdownReport> overall;
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto& [kind, o] = sources[i];
            if (s.name.rfind("ladder", 0) == 0) s.batch_keep = keep[i];
            const auto prior = prior_of.find(o.label.substr(0, o.label.find('/')));
            if (s.calibrate && prior == prior_of.end()) continue;
            SlowdownReport r = s.calibrate ? calibrate_replay(o, *prior->second, s, k, slower) : slowdown_replay(o, s, k, slower);
            r.lattice = o.lattice;
            print_slowdown(s.name, o.label, r);
            pool(overall[kind], r);
        }
        for (const auto& [kind, r] : overall) print_slowdown(s.name, "overall/" + kind, r);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--slowdown" || a == "--table" || a == "--ledger") {
            try {
                return slowdown_main(argc, argv);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "tune_replay: %s\n", e.what());
                return 1;
            }
        }
    }
    std::string raw, tier_name;
    TierParams p = params(Tier::coarse);
    std::optional<double> confidence;
    std::optional<int> min_reps, max_reps, stride;
    std::optional<double> refine;
    ReplayOpts ro;
    ro.holdout = true;
    bool print_axes = false;
    std::optional<std::string> mode;
    std::optional<double> margin, cap;
    std::vector<std::pair<std::string, std::pair<std::string, bool>>> shrink;  // axis, spec, keep
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "tune_replay: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--raw") raw = next();
        else if (a == "--tier") tier_name = next();
        else if (a == "--confidence") confidence = std::stod(next());
        else if (a == "--min-reps") min_reps = std::stoi(next());
        else if (a == "--max-reps") max_reps = std::stoi(next());
        else if (a == "--refine-mode") mode = next();
        else if (a == "--refine-margin") margin = std::stod(next());
        else if (a == "--refine-cap") cap = std::stod(next());
        else if (a == "--print-axes") print_axes = true;
        else if (a == "--axis-keep" || a == "--axis-stride") {
            const std::string v = next();
            const auto eq = v.find('=');
            if (eq == std::string::npos) {
                std::fprintf(stderr, "tune_replay: %s wants name=value\n", a.c_str());
                return 2;
            }
            shrink.push_back({v.substr(0, eq), {v.substr(eq + 1), a == "--axis-keep"}});
        }
        else if (a == "--refine") refine = std::stod(next());
        else if (a == "--verify-s") ro.verify_s = std::stod(next());
        else if (a == "--cell-overhead-s") ro.cell_overhead_s = std::stod(next());
        else if (a == "--no-holdout") ro.holdout = false;
        else if (a == "--stride") stride = std::stoi(next());
        else {
            std::fprintf(stderr, "tune_replay: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (mode && *mode != "index" && *mode != "geometric") {
        std::fprintf(stderr, "tune_replay: --refine-mode must be geometric or index, not %s\n", mode->c_str());
        return 2;
    }
    const auto tier = parse_tier(tier_name);
    if (raw.empty() || !tier || unhashed(*tier) || *tier == Tier::custom) {
        std::fprintf(stderr, "usage: tune_replay --raw <jsonl> --tier preview|coarse|deep [--confidence x] [--min-reps n] "
                             "[--max-reps n] [--stride n] [--refine r] [--verify-s s] [--cell-overhead-s s] [--no-holdout] [--axis-keep name=v:v] [--axis-stride name=k] [--refine-mode geometric|index] [--refine-margin m] [--refine-cap f] [--print-axes]\n");
        return 2;
    }
    p = params(*tier);
    if (confidence) p.confidence = *confidence;
    if (min_reps) p.min_reps = *min_reps;
    if (max_reps) p.max_reps = *max_reps;
    if (stride) p.stride = *stride;
    if (refine) p.refine_ratio = *refine;
    if (mode) p.refine_mode = *mode == "index" ? RefineMode::index : RefineMode::geometric;
    if (margin) p.refine_margin = *margin;
    if (cap) p.refine_cap_factor = *cap;
    try {
        ReplayMeta meta;
        const auto cells = load_replay(raw, &meta);
        if (print_axes) {
            for (const AxisSpec& a : meta.axes) {
                std::string v;
                for (const std::string& x : a.values) v += (v.empty() ? "" : " ") + x;
                std::printf("%s%s: %s\n", a.name.c_str(), a.log ? " (log)" : "", v.c_str());
            }
            return 0;
        }
        ro.order = meta.candidates;
        const auto raw_axes = meta.axes;
        for (const auto& [name, sv] : shrink) shrink_axis(meta.axes, name, sv.first, sv.second);
        ReplayReport r = replay(cells, meta.axes, *tier, p, 0.03, ro);
        if (ro.holdout) {
            ReplayOpts all = ro;
            all.all_cells = true;
            apply_floor(r, replay(cells, raw_axes, *tier, oracle_params(), 0.03, all));
        }
        std::string worst;
        for (const std::string& w : r.worst) worst += (worst.empty() ? "" : "; ") + w;
        Json j;
        j.str("raw", raw).str("tier", to_string(*tier));
        j.integer("stride", p.stride).num("refine_ratio", p.refine_ratio).integer("min_reps", p.min_reps);
        j.integer("max_reps", p.max_reps).num("confidence", p.confidence);
        j.integer("cells", static_cast<std::int64_t>(r.cells)).integer("cells_measured", static_cast<std::int64_t>(r.cells_measured));
        j.num("reps_fraction", r.reps_fraction).num("race_misrank", r.race_misrank).num("table_misrank", r.table_misrank);
        j.num("table_misrank_lattice", r.table_misrank_lattice).num("mean_loss", r.mean_loss).num("p95_loss", r.p95_loss);
        j.num("p99_loss", r.p99_loss).num("max_loss", r.max_loss).num("time_weighted_loss", r.time_weighted_loss);
        j.integer("unrunnable", static_cast<std::int64_t>(r.unrunnable)).num("est_gpu_h", r.est_gpu_h);
        j.num("measure_s", r.measure_s).boolean("holdout", ro.holdout).num("exhaustive_rep_s", meta.total_rep_ms / 1000);
        j.num("exhaustive_rep_gpu_h", meta.total_rep_ms / 3.6e6);
        const double bound = *tier == Tier::deep ? 0.002 : *tier == Tier::coarse ? 0.01 : 0.05;  // spec "Engine: testing and acceptance"
        const double race = ro.holdout ? r.excess_race : r.race_misrank, table = ro.holdout ? r.excess_table : r.table_misrank;
        j.num("bound", bound).str("verdict", race <= bound && table <= bound ? "PASS" : "FAIL");
        j.num("floor_race", r.floor_race).num("floor_table", r.floor_table).num("excess_race", r.excess_race).num("excess_table", r.excess_table);
        j.integer("refine_unavailable", static_cast<std::int64_t>(r.refine_unavailable));
        j.integer("cells_lattice", static_cast<std::int64_t>(r.cells_lattice)).num("refine_cap", p.refine_cap_factor);
        j.integer("refine_capped", static_cast<std::int64_t>(r.refine_capped)).str("worst", worst);
        std::fputs(j.line().c_str(), stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tune_replay: %s\n", e.what());
        return 1;
    }
    return 0;
}
