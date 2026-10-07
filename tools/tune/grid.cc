#include "grid.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace batchlas::tune {

std::uint64_t fnv1a64(std::string_view s) {
    std::uint64_t h = 14695981039346656037ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

std::vector<AxisSpec> axis_specs(const std::vector<std::string>& key_names,
                                 const std::vector<std::pair<std::string, std::vector<std::string>>>& axes) {
    std::map<std::string, bool> log;
    std::map<std::string, double> weight;
    for (const std::string& kn : key_names) {
        const auto part = split_fields(kn, ':');
        if (part.size() < 2) continue;
        log[part[0]] = part[1] == "log";
        if (part.size() > 2 && part[1] == "log") weight[part[0]] = std::stod(part[2]);
    }
    std::vector<AxisSpec> out;
    for (const auto& [name, values] : axes) {
        const auto it = log.find(name);
        const auto w = weight.find(name);
        out.push_back({name, it != log.end() && it->second, values, w != weight.end() ? w->second : 1.0});
    }
    return out;
}

std::vector<std::string> subsample_axis(const std::vector<std::string>& values, int stride) {
    std::vector<std::string> out;
    for (std::size_t i = 0; i < values.size(); ++i)
        if (i % std::size_t(std::max(stride, 1)) == 0 || i + 1 == values.size()) out.push_back(values[i]);
    return out;
}

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, Tier t) {
    const int stride = params(t).stride;
    std::vector<CellKey> cells{{}};
    for (const AxisSpec& a : axes) {
        const auto vals = a.log ? subsample_axis(a.values, stride) : a.values;
        std::vector<CellKey> next;
        for (const CellKey& c : cells)
            for (const std::string& v : vals) {
                CellKey k = c;
                k.push_back({a.name, v});
                next.push_back(std::move(k));
            }
        cells = std::move(next);
    }
    return cells;
}

std::vector<CellKey> tier_subsample(const std::vector<CellKey>& grid, Tier t) {
    const auto stride = std::uint64_t(std::max(params(t).stride, 1));
    std::vector<CellKey> out;
    for (const CellKey& c : grid)
        if (fnv1a64(key_arg(c)) % stride == 0) out.push_back(c);
    return out;
}


std::size_t refine_allowance(std::size_t lattice_cells, std::size_t refined, double cap_factor) {
    const auto cap = std::size_t(std::floor(std::max(cap_factor, 0.0) * double(lattice_cells) + 1e-9));
    return cap > refined ? cap - refined : 0;
}

namespace {

// Batch is measured at saturation-relevant points only: it refills its own values, never a geometric midpoint.
// evidence: docs/design/tiered-tuning.md#engine-refinement-convergence-rules
constexpr const char* kBatchAxis = "batch";

const RefineCell* info_of(const RefineOpts& o, const CellKey& k) {
    if (!o.cells) return nullptr;
    const auto it = o.cells->find(k);
    return it == o.cells->end() ? nullptr : &it->second;
}

// At a cell won by `w`: `other` is more than the tie slower, or cannot win there (not runnable, eliminated untimed).
bool decisive(const RefineCell& c, const std::string& w, const std::string& other, double tie) {
    const auto o = c.ms.find(other);
    if (o == c.ms.end()) return c.out.count(other) != 0;
    const auto t = c.ms.find(w);
    return t == c.ms.end() || o->second / t->second - 1 > tie;
}

double runner_up_gap(const RefineCell& c, const std::string& w) {
    double g = std::numeric_limits<double>::infinity();
    const auto t = c.ms.find(w);
    if (t == c.ms.end()) return g;
    for (const auto& [cand, ms] : c.ms)
        if (cand != w) g = std::min(g, std::max(0.0, ms / t->second - 1));
    return g;
}

}  // namespace

RefineRound refine_all_axes(const std::map<CellKey, std::vector<std::string>>& ranked,
                            const std::vector<AxisSpec>& axes, double ratio, const RefineOpts& opts) {
    RefineRound out;
    if (opts.mode == RefineMode::geometric && ratio <= 0) return out;
    std::vector<CellKey> by_kind[2];  // [0] flips, [1] margin hedges
    for (const AxisSpec& a : axes) {
        if (!a.log) continue;
        const bool batch = a.name == kBatchAxis;
        std::vector<std::int64_t> lattice;
        for (const std::string& v : a.values) lattice.push_back(std::stoll(v));
        std::sort(lattice.begin(), lattice.end());
        // line = cells equal in every key but a.name, keyed by that remainder
        std::map<CellKey, std::vector<LinePoint>> lines;
        std::map<CellKey, CellKey> sample;
        for (const auto& [cell, rk] : ranked) {
            const std::string* v = key_get(cell, a.name);
            if (!v || rk.empty()) continue;
            CellKey rest;
            for (const KV& kv : cell)
                if (kv.name != a.name) rest.push_back(kv);
            lines[rest].push_back({key_int(cell, a.name), rk.front()});
            sample.emplace(rest, cell);
        }
        for (auto& [rest, pts] : lines) {
            std::sort(pts.begin(), pts.end(), [](const LinePoint& x, const LinePoint& y) { return x.n < y.n; });
            auto at = [&](std::int64_t n) { return key_with(sample[rest], a.name, std::to_string(n)); };
            for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                const LinePoint &lo = pts[i], &hi = pts[i + 1];
                const RefineCell *cl = info_of(opts, at(lo.n)), *ch = info_of(opts, at(hi.n));
                bool flip = lo.winner != hi.winner;
                if (flip && cl && ch)
                    flip = decisive(*cl, lo.winner, hi.winner, opts.tie) || decisive(*ch, hi.winner, lo.winner, opts.tie);
                const bool near = !flip && opts.margin > 0 && cl && ch && cl->lattice && ch->lattice &&
                                  (runner_up_gap(*cl, lo.winner) <= opts.margin || runner_up_gap(*ch, hi.winner) <= opts.margin);
                if (!flip && !near) continue;
                std::vector<std::int64_t> mids, inside;
                for (std::int64_t v : lattice)
                    if (v > lo.n && v < hi.n) inside.push_back(v);
                if ((opts.mode == RefineMode::index || batch) && !inside.empty())
                    mids.push_back(inside[(inside.size() - 1) / 2]);
                else if (!batch && ratio > 0)
                    mids = refine_midpoints({{lo.n, "lo"}, {hi.n, "hi"}}, ratio);
                for (std::int64_t mid : mids) {
                    CellKey k = at(mid);
                    if (!ranked.count(k)) {
                        by_kind[flip ? 0 : 1].push_back(std::move(k));
                        continue;
                    }
                    out.stalled.push_back(key_text(rest) + ": edge between " + a.name + "=" + std::to_string(lo.n) + " (" +
                                          lo.winner + ") and " + std::to_string(hi.n) + " (" + hi.winner + ") stays wide: " +
                                          a.name + "=" + std::to_string(mid) + " has no winner");
                }
            }
        }
    }
    std::set<CellKey> seen;
    for (auto& kind : by_kind)
        for (CellKey& k : kind)
            if (seen.insert(k).second) out.next.push_back(std::move(k));
    return out;
}

}  // namespace batchlas::tune
