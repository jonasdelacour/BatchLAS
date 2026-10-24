// SPDX-License-Identifier: MIT
#include <tools/tune/grid.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <batchlas/error.hh>

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

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, Tier t) { return tier_lattice(axes, params(t).stride); }

std::vector<CellKey> tier_lattice(const std::vector<AxisSpec>& axes, int stride) {
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

NearestIndex::NearestIndex(const std::vector<CellKey>& rows, const std::vector<AxisSpec>& axes)
    : axes_(axes), min_prefix_(!axes.empty() && !axes[0].log && axes[0].name == "dtype") {
    for (const AxisSpec& a : axes)
        if (a.log) weights_.push_back(a.weight);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        order_.push_back(i);
        rows_.push_back(prepare(rows[i], false));
    }
    std::sort(order_.begin(), order_.end(), [&](std::size_t a, std::size_t b) {
        return std::tie(rows_[a].exact, rows_[a].logs) < std::tie(rows_[b].exact, rows_[b].logs);
    });
    for (std::size_t i : order_) group_[rows_[i].exact].push_back(i);
}

NearestIndex::Row NearestIndex::prepare(const CellKey& k, bool query) const {
    Row r;
    for (std::size_t i = 0; i < axes_.size(); ++i) {
        if (!axes_[i].log) {
            r.exact.push_back(k[i].value);
            continue;
        }
        r.logs.push_back(std::stoll(k[i].value));
        r.lg.push_back(std::log2(double(query ? std::max<std::int64_t>(1, r.logs.back()) : r.logs.back())));
    }
    return r;
}

double NearestIndex::distance(const Row& r, const Row& q) const {
    double d = 0;
    for (std::size_t j = 0; j < weights_.size(); ++j) d += weights_[j] * std::abs(r.lg[j] - q.lg[j]);
    return d;
}

std::size_t NearestIndex::find(const CellKey& key) const {
    const Row q = prepare(key, true);
    for (std::size_t p = q.exact.size() + 1; p-- > min_prefix_;) {
        const auto g = group_.find(q.exact);
        if (p == q.exact.size() && g == group_.end()) continue;
        bool have = false;
        std::size_t best = 0;
        double best_d = 0;
        for (std::size_t i : p == q.exact.size() ? g->second : order_) {
            if (!std::equal(q.exact.begin(), q.exact.begin() + std::ptrdiff_t(p), rows_[i].exact.begin())) continue;
            const double d = distance(rows_[i], q);
            const bool tie = have && std::abs(d - best_d) <= 1e-9;
            if (!have || (!tie && d < best_d) || (tie && rows_[i].logs < rows_[best].logs)) best = i, best_d = d, have = true;
        }
        if (have) return best;
    }
    return npos;
}

std::vector<std::size_t> NearestIndex::k_nearest(const CellKey& key, std::size_t k) const {
    const Row q = prepare(key, true);
    const auto g = group_.find(q.exact);
    if (g == group_.end()) return {};
    std::vector<std::pair<double, std::size_t>> by;
    for (std::size_t i : g->second) by.emplace_back(distance(rows_[i], q), i);
    const auto n = std::ptrdiff_t(std::min(k, by.size()));
    std::partial_sort(by.begin(), by.begin() + n, by.end(), [&](const auto& a, const auto& b) {
        if (std::abs(a.first - b.first) > 1e-9) return a.first < b.first;
        return rows_[a.second].logs < rows_[b.second].logs;
    });
    std::vector<std::size_t> out;
    for (std::ptrdiff_t i = 0; i < n; ++i) out.push_back(by[std::size_t(i)].second);
    return out;
}

std::size_t nearest_row(const std::vector<CellKey>& rows, const CellKey& key, const std::vector<AxisSpec>& axes) {
    return NearestIndex(rows, axes).find(key);
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

std::string winner_of(const RefineCell& c) {
    if (c.ms.empty()) return c.single;
    return std::min_element(c.ms.begin(), c.ms.end(), [](const auto& x, const auto& y) { return x.second < y.second; })->first;
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

std::vector<CellKey> flip_frontier(const std::map<CellKey, RefineCell>& measured, const std::vector<CellKey>& universe,
                                   const std::vector<AxisSpec>& axes, std::size_t k, double slower) {
    return CalibrateRound(measured, axes, {}, k, slower).flips(universe);
}

CalibrateRound::CalibrateRound(const std::map<CellKey, RefineCell>& measured, const std::vector<AxisSpec>& axes, Pick pick,
                               std::size_t k, double slower)
    : measured_(measured), axes_(axes), pick_(std::move(pick)), k_(k), slower_(slower) {
    for (const auto& [key, c] : measured) {
        if (c.ms.empty() && c.single.empty()) continue;
        rows_.push_back(key);
        info_.push_back(&c);
        winner_.push_back(winner_of(c));
        const std::string p = pick_ ? pick_(key) : std::string();
        dis_.push_back(pick_ && (p.empty() || (p != winner_.back() && decisive(c, winner_.back(), p, slower_))));
    }
    if (!rows_.empty()) index_.emplace(rows_, axes_);
}

bool CalibrateRound::disagrees(const CellKey& m) const {
    const auto it = std::find(rows_.begin(), rows_.end(), m);
    return it != rows_.end() && dis_[std::size_t(it - rows_.begin())];
}

bool CalibrateRound::disagrees_near(const CellKey& u, std::size_t n) const {
    if (!index_) return false;
    const auto nb = index_->k_nearest(u, n);
    return std::any_of(nb.begin(), nb.end(), [&](std::size_t r) { return bool(dis_[r]); });
}

// A cell with no prior pick has nothing to copy: flip_frontier's rule over the device's cells decides it.
std::vector<CellKey> CalibrateRound::frontier(const std::vector<CellKey>& open) const {
    std::vector<CellKey> out, unpicked;
    if (!index_) return out;
    for (const CellKey& u : open) {
        if (measured_.count(u)) continue;
        const std::string p = pick_(u);
        if (p.empty()) {
            unpicked.push_back(u);
            continue;
        }
        for (std::size_t r : index_->k_nearest(u, k_))
            if (dis_[r] && winner_[r] != p && decisive(*info_[r], winner_[r], p, slower_)) {
                out.push_back(u);
                break;
            }
    }
    for (CellKey& u : flips(unpicked)) out.push_back(std::move(u));
    return out;
}

// Two of u's k nearest measured cells have different winners and one decisively beats the other there.
std::vector<CellKey> CalibrateRound::flips(const std::vector<CellKey>& open) const {
    std::vector<CellKey> out;
    if (!index_) return out;
    for (const CellKey& u : open) {
        if (measured_.count(u)) continue;
        const auto nb = index_->k_nearest(u, k_);
        bool flip = false;
        for (std::size_t a = 0; a < nb.size() && !flip; ++a)
            for (std::size_t b = 0; b < nb.size() && !flip; ++b)
                flip = a != b && winner_[nb[a]] != winner_[nb[b]] &&
                       decisive(*info_[nb[b]], winner_[nb[b]], winner_[nb[a]], slower_);
        if (flip) out.push_back(u);
    }
    return out;
}

std::set<std::string> CalibrateRound::winners(const CellKey& u) const {
    std::set<std::string> out;
    if (index_)
        for (std::size_t r : index_->k_nearest(u, k_)) out.insert(winner_[r]);
    return out;
}

std::vector<std::int64_t> batch_ladder(int cus, const std::vector<double>& classes, const std::vector<int>& extra) {
    if (cus < 1) throw batchlas::invalid_argument("batch ladder: the device's compute-unit count is unknown (--cus N)");
    std::set<std::int64_t> b(extra.begin(), extra.end());
    for (double c : classes) {
        if (!(c > 0)) throw batchlas::invalid_argument("batch ladder: a class multiplier must be positive");
        b.insert(std::max<std::int64_t>(1, std::llround(c * double(cus))));
    }
    return {b.begin(), b.end()};
}

std::vector<int> with_slice_batches(std::vector<int> batches) {
    batches.insert(batches.end(), {65536, 131072});
    return batches;
}

std::vector<double> without_latency_class(const std::vector<double>& classes) {
    std::vector<double> out;
    for (double c : classes)
        if (c >= 1) out.push_back(c);
    return out.empty() ? classes : out;
}

std::vector<std::string> batch_strings(const std::vector<std::int64_t>& batches) {
    std::vector<std::string> out;
    for (std::int64_t b : batches) out.push_back(std::to_string(b));
    return out;
}

std::vector<CellKey> rebatch(const std::vector<CellKey>& grid, const std::vector<std::string>& batches) {
    std::vector<CellKey> order;
    std::map<CellKey, std::vector<CellKey>> lines;  // line -> its cells, in grid order
    for (const CellKey& c : grid) {
        if (!key_get(c, kBatchAxis)) return grid;
        CellKey rest;
        for (const KV& kv : c)
            if (kv.name != kBatchAxis) rest.push_back(kv);
        auto& l = lines[rest];
        if (l.empty()) order.push_back(rest);
        l.push_back(c);
    }
    std::vector<CellKey> out;
    for (const CellKey& rest : order) {
        const auto& cells = lines[rest];
        if (cells.size() == 1) {
            out.push_back(cells.front());
            continue;
        }
        for (const std::string& b : batches) out.push_back(key_with(cells.front(), kBatchAxis, b));
    }
    return out;
}

std::vector<CellKey> frontier_start(const std::vector<CellKey>& cells, const std::vector<AxisSpec>& axes, int stride,
                                    bool global) {
    std::vector<std::set<std::string>> start(axes.size());
    for (std::size_t i = 0; i < axes.size(); ++i)
        if (axes[i].log)
            for (const std::string& v : subsample_axis(axes[i].values, stride)) start[i].insert(v);
    std::set<const CellKey*> keep;
    std::map<std::vector<std::string>, std::vector<const CellKey*>> groups;
    for (const CellKey& c : cells) {
        bool on = global;
        for (std::size_t i = 0; i < c.size() && on; ++i) on = !axes[i].log || start[i].count(c[i].value);
        if (on) keep.insert(&c);
        std::vector<std::string> g;
        for (std::size_t i = 0; i < c.size(); ++i)
            if (!axes[i].log) g.push_back(c[i].value);
        groups[g].push_back(&c);
    }
    for (const auto& [g, group] : groups) {
        if (std::any_of(group.begin(), group.end(), [&](const CellKey* c) { return keep.count(c) != 0; })) continue;
        std::vector<std::set<std::string>> sub(axes.size());
        for (std::size_t i = 0; i < axes.size(); ++i) {
            if (!axes[i].log) continue;
            std::set<std::int64_t> own;
            for (const CellKey* c : group) own.insert(std::stoll((*c)[i].value));
            std::vector<std::string> v;
            for (std::int64_t x : own) v.push_back(std::to_string(x));
            for (const std::string& x : subsample_axis(v, stride)) sub[i].insert(x);
        }
        std::size_t hit = 0;
        for (const CellKey* c : group) {
            bool on = true;
            for (std::size_t i = 0; i < axes.size() && on; ++i) on = !axes[i].log || sub[i].count((*c)[i].value);
            if (on) keep.insert(c), ++hit;
        }
        if (hit == 0) keep.insert(group.begin(), group.end());
    }
    std::vector<CellKey> out;
    for (const CellKey& c : cells)
        if (keep.count(&c)) out.push_back(c);
    return out;
}

}  // namespace batchlas::tune
