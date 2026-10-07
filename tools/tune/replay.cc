// tune_replay: runs the tiered tuner's race and bisection against an exhaustive raw sweep, no GPU.
//   tune_replay --raw <jsonl> --tier preview|coarse|deep [--confidence x] [--min-reps n] [--max-reps n] [--stride n] [--refine r] [--verify-s s] [--cell-overhead-s s] [--no-holdout] [--axis-keep name=v:v] [--axis-stride name=k] [--refine-mode geometric|index] [--refine-margin m] [--print-axes]
// Prints the ReplayReport as one JSON line (docs/design/tiered-tuning.md).

#include "replay_core.hh"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

using namespace batchlas::tune;

int main(int argc, char** argv) {
    std::string raw, tier_name;
    TierParams p = params(Tier::coarse);
    std::optional<double> confidence;
    std::optional<int> min_reps, max_reps, stride;
    std::optional<double> refine;
    ReplayOpts ro;
    ro.holdout = true;
    bool print_axes = false;
    std::optional<std::string> mode;
    std::optional<double> margin;
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
    if (raw.empty() || !tier || *tier == Tier::transcribed || *tier == Tier::custom) {
        std::fprintf(stderr, "usage: tune_replay --raw <jsonl> --tier preview|coarse|deep [--confidence x] [--min-reps n] "
                             "[--max-reps n] [--stride n] [--refine r] [--verify-s s] [--cell-overhead-s s] [--no-holdout] [--axis-keep name=v:v] [--axis-stride name=k] [--refine-mode geometric|index] [--refine-margin m] [--print-axes]\n");
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
        for (const auto& [name, sv] : shrink) shrink_axis(meta.axes, name, sv.first, sv.second);
        ReplayReport r = replay(cells, meta.axes, *tier, p, 0.03, ro);
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
        j.num("bound", bound).str("verdict", r.race_misrank <= bound && r.table_misrank <= bound ? "PASS" : "FAIL");
        j.integer("refine_unavailable", static_cast<std::int64_t>(r.refine_unavailable)).str("worst", worst);
        std::fputs(j.line().c_str(), stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tune_replay: %s\n", e.what());
        return 1;
    }
    return 0;
}
