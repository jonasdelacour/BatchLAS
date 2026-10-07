#include "grid.hh"

#include <algorithm>
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

RefineRound refine_all_axes(const std::map<CellKey, std::vector<std::string>>& ranked,
                            const std::vector<AxisSpec>& axes, double ratio, const RefineOpts& opts) {
    RefineRound out;
    const bool plain = opts.mode == RefineMode::geometric && opts.margin <= 0;
    if (opts.mode == RefineMode::geometric && ratio <= 0) return out;
    std::set<CellKey> seen;
    auto gap_of = [&](const CellKey& k) -> double {
        if (!opts.gap) return std::numeric_limits<double>::infinity();
        const auto it = opts.gap->find(k);
        return it == opts.gap->end() ? std::numeric_limits<double>::infinity() : it->second;
    };
    for (const AxisSpec& a : axes) {
        if (!a.log) continue;
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
            std::vector<std::int64_t> mids;
            if (plain) {
                mids = refine_midpoints(pts, ratio);
            } else {
                auto index_of = [&](std::int64_t n) {
                    for (std::size_t i = 0; i < a.values.size(); ++i)
                        if (std::stoll(a.values[i]) == n) return std::int64_t(i);
                    return std::int64_t(-1);
                };
                auto at = [&](std::int64_t n) { return key_with(sample[rest], a.name, std::to_string(n)); };
                for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
                    const LinePoint &lo = pts[i], &hi = pts[i + 1];
                    const bool near = opts.margin > 0 && (gap_of(at(lo.n)) <= opts.margin || gap_of(at(hi.n)) <= opts.margin);
                    if (lo.winner == hi.winner && !near) continue;
                    const std::int64_t li = index_of(lo.n), hj = index_of(hi.n);
                    if (opts.mode == RefineMode::index && li >= 0 && hj - li > 1) {
                        mids.push_back(std::stoll(a.values[std::size_t((li + hj) / 2)]));
                    } else if (ratio > 0) {
                        for (std::int64_t m : refine_midpoints({{lo.n, "lo"}, {hi.n, "hi"}}, ratio)) mids.push_back(m);
                    }
                }
            }
            for (std::int64_t mid : mids) {
                CellKey k = key_with(sample[rest], a.name, std::to_string(mid));
                if (!ranked.count(k)) {
                    if (seen.insert(k).second) out.next.push_back(std::move(k));
                    continue;
                }
                const auto hi = std::find_if(pts.begin(), pts.end(), [&](const LinePoint& p) { return p.n > mid; });
                out.stalled.push_back(key_text(rest) + ": edge between " + a.name + "=" + std::to_string((hi - 1)->n) +
                                      " (" + (hi - 1)->winner + ") and " + std::to_string(hi->n) + " (" + hi->winner +
                                      ") stays wide: " + a.name + "=" + std::to_string(mid) + " has no winner");
            }
        }
    }
    return out;
}

}  // namespace batchlas::tune
