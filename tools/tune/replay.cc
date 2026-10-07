// tune_replay: runs the tiered tuner's race and bisection against an exhaustive raw sweep, no GPU.
//   tune_replay --raw <jsonl> --tier ultra|coarse|deep [--confidence x] [--min-reps n] [--max-reps n] [--stride n]
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
        else if (a == "--stride") stride = std::stoi(next());
        else {
            std::fprintf(stderr, "tune_replay: unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    const auto tier = parse_tier(tier_name);
    if (raw.empty() || !tier || *tier == Tier::transcribed || *tier == Tier::custom) {
        std::fprintf(stderr, "usage: tune_replay --raw <jsonl> --tier ultra|coarse|deep [--confidence x] [--min-reps n] "
                             "[--max-reps n] [--stride n]\n");
        return 2;
    }
    p = params(*tier);
    if (confidence) p.confidence = *confidence;
    if (min_reps) p.min_reps = *min_reps;
    if (max_reps) p.max_reps = *max_reps;
    if (stride) p.stride = *stride;
    try {
        ReplayMeta meta;
        const auto cells = load_replay(raw, &meta);
        const ReplayReport r = replay(cells, meta.axes, *tier, p);
        std::string worst;
        for (const std::string& w : r.worst) worst += (worst.empty() ? "" : "; ") + w;
        Json j;
        j.str("raw", raw).str("tier", to_string(*tier));
        j.integer("stride", p.stride).num("refine_ratio", p.refine_ratio).integer("min_reps", p.min_reps);
        j.integer("max_reps", p.max_reps).num("confidence", p.confidence);
        j.integer("cells", static_cast<std::int64_t>(r.cells)).integer("cells_measured", static_cast<std::int64_t>(r.cells_measured));
        j.num("reps_fraction", r.reps_fraction).num("race_misrank", r.race_misrank).num("table_misrank", r.table_misrank);
        j.integer("refine_unavailable", static_cast<std::int64_t>(r.refine_unavailable)).str("worst", worst);
        std::fputs(j.line().c_str(), stdout);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tune_replay: %s\n", e.what());
        return 1;
    }
    return 0;
}
