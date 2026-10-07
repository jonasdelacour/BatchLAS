#include "replay_core.hh"

#include "race.hh"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace batchlas::tune {

namespace {

constexpr double kMisrank = 1.03;
const double kNaN = std::numeric_limits<double>::quiet_NaN();

struct RawCell {
    std::map<int, Attempt> attempts;
    std::map<int, std::map<std::string, std::map<std::pair<int, int>, double>>> reps;  // attempt, cand, (pass, rep)
    int final_attempt = -1, round = 0;
    std::string status;
};

}  // namespace

std::vector<ReplayCell> load_replay(const std::string& raw_jsonl, ReplayMeta* meta_out) {
    std::ifstream f(raw_jsonl);
    if (!f) throw std::runtime_error("cannot open " + raw_jsonl);
    ReplayMeta meta;
    std::vector<std::string> names;
    int passes = 0;
    bool have_meta = false;
    std::map<CellKey, RawCell> raw;
    std::string line;
    while (std::getline(f, line)) {
        const auto rec = parse_record(line);
        if (!rec) continue;
        const std::string kind = rec->get("kind");
        if (kind == "meta") {
            meta.key_names = split(rec->get("keys"), ' ');
            meta.candidates = split(rec->get("candidates"), '|');
            for (const std::string& kn : meta.key_names) names.push_back(split_fields(kn, ':')[0]);
            passes = static_cast<int>(rec->number("passes"));
            have_meta = true;
            continue;
        }
        if (kind != "rep" && kind != "pass" && kind != "cell") continue;
        if (!have_meta) throw std::runtime_error(raw_jsonl + ": record before meta");
        CellKey key;
        for (const std::string& n : names) key.push_back({n, rec->get(n)});
        RawCell& c = raw[key];
        if (kind == "cell") {
            c.status = rec->get("status");
            c.final_attempt = static_cast<int>(rec->number("final_attempt"));
            c.round = static_cast<int>(rec->number("round"));
            continue;
        }
        const int attempt = static_cast<int>(rec->number("attempt")), pass = static_cast<int>(rec->number("pass"));
        const std::string cand = rec->get("cand");
        if (kind == "rep") {
            c.reps[attempt][cand][{pass, static_cast<int>(rec->number("rep"))}] = rec->number("ms");
            continue;
        }
        Attempt& a = c.attempts[attempt];
        a.resize(std::max<std::size_t>({a.size(), std::size_t(passes), std::size_t(pass)}));
        a[pass - 1].arms[cand] = {rec->get("status"), rec->get("reason"), rec->number("median_ms")};
    }
    if (!have_meta) throw std::runtime_error(raw_jsonl + ": no meta record");

    std::vector<std::set<std::string>> seen(names.size());
    std::vector<ReplayCell> out;
    for (auto& [key, c] : raw) {
        if (c.round == 0 && !c.status.empty())
            for (std::size_t i = 0; i < names.size(); ++i) seen[i].insert(key[i].value);
        if (c.status != "ok" || !c.attempts.count(c.final_attempt)) continue;
        ReplayCell rc;
        rc.key = key;
        rc.round = c.round;
        rc.exhaustive = attempt_times(c.attempts[c.final_attempt], meta.candidates);
        const auto& reps = c.reps[c.final_attempt];
        int per_pass = 0;
        for (const auto& [cand, m] : reps)
            for (const auto& [pr, ms] : m) per_pass = std::max(per_pass, pr.second + 1);
        for (const std::string& cand : meta.candidates) {
            if (!rc.exhaustive.count(cand) || !reps.count(cand)) continue;
            std::vector<double> v(static_cast<std::size_t>(per_pass) * passes, kNaN);
            for (const auto& [pr, ms] : reps.at(cand)) v[static_cast<std::size_t>(pr.second) * passes + (pr.first - 1)] = ms;
            rc.cands.push_back(cand);
            rc.rounds.push_back(std::move(v));
        }
        if (!rc.cands.empty()) out.push_back(std::move(rc));
    }
    std::vector<std::pair<std::string, std::vector<std::string>>> values;
    for (std::size_t i = 0; i < names.size(); ++i) values.emplace_back(names[i], std::vector<std::string>(seen[i].begin(), seen[i].end()));
    meta.axes = axis_specs(meta.key_names, values);
    for (AxisSpec& a : meta.axes)
        if (a.log)
            std::sort(a.values.begin(), a.values.end(),
                      [](const std::string& x, const std::string& y) { return std::stoll(x) < std::stoll(y); });
    if (meta_out) *meta_out = std::move(meta);
    return out;
}

namespace {

struct Prepared {
    std::vector<std::string> exact;
    std::vector<std::int64_t> logs;
};

Prepared prepare(const CellKey& k, const std::vector<AxisSpec>& axes) {
    Prepared p;
    for (std::size_t i = 0; i < axes.size(); ++i) {
        if (axes[i].log) p.logs.push_back(std::stoll(k[i].value));
        else p.exact.push_back(k[i].value);
    }
    return p;
}

// Rows are visited in the converter's (exact strings, log integers) order, so a full tie goes to the earlier one.
class NearestIndex {
public:
    NearestIndex(const std::vector<CellKey>& rows, const std::vector<AxisSpec>& axes) : axes_(axes) {
        for (const AxisSpec& a : axes)
            if (a.log) weights_.push_back(a.weight);
        for (std::size_t i = 0; i < rows.size(); ++i) {
            order_.push_back(i);
            prep_.push_back(prepare(rows[i], axes));
        }
        std::sort(order_.begin(), order_.end(), [&](std::size_t a, std::size_t b) {
            return std::pair(prep_[a].exact, prep_[a].logs) < std::pair(prep_[b].exact, prep_[b].logs);
        });
    }

    std::size_t find(const CellKey& key) const {
        const Prepared q = prepare(key, axes_);
        std::vector<double> lq;
        for (std::int64_t v : q.logs) lq.push_back(std::log2(double(std::max<std::int64_t>(1, v))));
        for (std::size_t p = q.exact.size() + 1; p-- > 0;) {
            bool have = false;
            std::size_t best = 0;
            double best_d = 0;
            for (std::size_t i : order_) {
                if (!std::equal(q.exact.begin(), q.exact.begin() + p, prep_[i].exact.begin())) continue;
                double d = 0;
                for (std::size_t j = 0; j < weights_.size(); ++j)
                    d += weights_[j] * std::abs(std::log2(double(prep_[i].logs[j])) - lq[j]);
                const bool tie = have && std::abs(d - best_d) <= 1e-9;
                if (!have || (!tie && d < best_d) || (tie && prep_[i].logs < prep_[best].logs)) best = i, best_d = d, have = true;
            }
            if (have) return best;
        }
        return 0;
    }

private:
    std::vector<AxisSpec> axes_;
    std::vector<double> weights_;
    std::vector<Prepared> prep_;
    std::vector<std::size_t> order_;
};

std::vector<CellKey> lattice(const std::vector<AxisSpec>& axes, int stride) {
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

double best_time(const ReplayCell& c) {
    double b = std::numeric_limits<double>::infinity();
    for (const auto& [cand, t] : c.exhaustive) b = std::min(b, t);
    return b;
}

}  // namespace

void shrink_axis(std::vector<AxisSpec>& axes, const std::string& name, const std::string& spec, bool keep) {
    const auto it = std::find_if(axes.begin(), axes.end(), [&](const AxisSpec& a) { return a.name == name; });
    if (it == axes.end()) throw std::invalid_argument("unknown axis " + name);
    if (!keep) {
        const int k = std::stoi(spec);
        if (k < 1) throw std::invalid_argument("axis stride must be >= 1");
        it->values = subsample_axis(it->values, k);
        return;
    }
    std::vector<std::string> kept;
    for (const std::string& v : split(spec, ':')) {
        if (std::find(it->values.begin(), it->values.end(), v) == it->values.end())
            throw std::invalid_argument("axis " + name + " has no value " + v);
        kept.push_back(v);
    }
    it->values = std::move(kept);
}

std::size_t nearest_row(const std::vector<CellKey>& rows, const CellKey& key, const std::vector<AxisSpec>& axes) {
    return NearestIndex(rows, axes).find(key);
}

ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier, const TierParams& p,
                    double tie, const RefineOpts& ro) {
    ReplayReport rep;
    rep.cells = cells.size();
    std::map<CellKey, std::size_t> at;
    std::vector<std::string> order;
    double total_reps = 0, used_reps = 0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
        at[cells[i].key] = i;
        for (const auto& r : cells[i].rounds)
            total_reps += double(std::count_if(r.begin(), r.end(), [](double x) { return !std::isnan(x); }));
        for (const std::string& c : cells[i].cands)
            if (std::find(order.begin(), order.end(), c) == order.end()) order.push_back(c);
    }

    std::map<CellKey, std::vector<std::string>> ranked;
    std::map<CellKey, double> gap;
    RefineOpts opts = ro;
    opts.gap = &gap;
    auto measure = [&](const CellKey& key) {
        const ReplayCell& c = cells[at.at(key)];
        RaceState s;
        s.cands = c.cands;
        s.ms.assign(c.cands.size(), {});
        s.alive.assign(c.cands.size(), true);
        for (std::size_t r = 0; r < c.rounds.front().size(); ++r) {
            for (std::size_t k = 0; k < c.cands.size(); ++k) {
                const bool run = s.alive[k] && !std::isnan(c.rounds[k][r]);
                s.ms[k].push_back(run ? c.rounds[k][r] : kNaN);
                used_reps += run;
            }
            if (race_step(s, p, tie) != RaceVerdict::more) break;
        }
        ranked[key] = race_ranking(s, order, tie);
        std::vector<double> med;
        for (std::size_t k = 0; k < c.cands.size(); ++k) {
            std::vector<double> v;
            for (double x : s.ms[k])
                if (!std::isnan(x)) v.push_back(x);
            med.push_back(median(v));
        }
        const std::size_t w = std::size_t(std::find(c.cands.begin(), c.cands.end(), ranked[key].front()) - c.cands.begin());
        double g = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < c.cands.size(); ++k)
            if (k != w && !std::isnan(med[k]) && !std::isnan(med[w])) g = std::min(g, std::max(0.0, med[k] / med[w] - 1));
        gap[key] = g;
    };

    std::vector<CellKey> todo;
    for (const CellKey& k : lattice(axes, p.stride))
        if (at.count(k)) todo.push_back(k);
    std::set<CellKey> unavailable;
    while (!todo.empty()) {
        for (const CellKey& k : todo) measure(k);
        todo.clear();
        for (CellKey& k : refine_all_axes(ranked, axes, p.refine_ratio, opts).next) {
            if (unavailable.count(k)) continue;
            if (at.count(k)) todo.push_back(std::move(k));
            else unavailable.insert(std::move(k));
        }
    }
    rep.refine_unavailable = unavailable.size();
    rep.cells_measured = ranked.size();
    rep.reps_fraction = total_reps > 0 ? used_reps / total_reps : 0;

    std::vector<std::pair<double, std::string>> worst;
    auto note = [&](const char* what, const ReplayCell& c, const std::string& winner, double ratio) {
        worst.emplace_back(ratio, std::string(what) + " " + key_text(c.key) + " picked " + winner + " x" + std::to_string(ratio));
    };
    std::size_t race_bad = 0, table_bad = 0;
    std::vector<CellKey> rows;
    for (const auto& [key, rk] : ranked) {
        const ReplayCell& c = cells[at.at(key)];
        const double ratio = c.exhaustive.at(rk.front()) / best_time(c);
        if (ratio > kMisrank) ++race_bad, note("race", c, rk.front(), ratio);
        rows.push_back(key);
    }
    std::vector<double> losses;
    double sum_chosen = 0, sum_best = 0;
    std::size_t lattice_cells = 0, lattice_bad = 0;
    if (!rows.empty()) {
        const NearestIndex nearest(rows, axes);
        for (const ReplayCell& c : cells) {
            const auto& row = ranked.at(rows[nearest.find(c.key)]);
            const auto pick = std::find_if(row.begin(), row.end(), [&](const std::string& n) { return c.exhaustive.count(n) != 0; });
            const bool runs = pick != row.end();
            const std::string& winner = runs ? *pick : row.front();
            const double ratio = runs ? c.exhaustive.at(winner) / best_time(c) : std::numeric_limits<double>::infinity();
            if (runs) losses.push_back(ratio - 1), sum_chosen += c.exhaustive.at(winner), sum_best += best_time(c);
            else ++rep.unrunnable;
            if (ratio > kMisrank) ++table_bad, note("table", c, winner, ratio);
            if (c.round == 0) ++lattice_cells, lattice_bad += ratio > kMisrank;
        }
    }
    rep.race_misrank = rows.empty() ? 0 : double(race_bad) / double(rows.size());
    rep.table_misrank = cells.empty() ? 0 : double(table_bad) / double(cells.size());
    rep.table_misrank_lattice = lattice_cells ? double(lattice_bad) / double(lattice_cells) : 0;
    if (!losses.empty()) {
        std::sort(losses.begin(), losses.end());
        auto q = [&](double f) { return losses[std::min(losses.size() - 1, std::size_t(f * double(losses.size())))]; };
        for (double l : losses) rep.mean_loss += l / double(losses.size());
        rep.p95_loss = q(0.95), rep.p99_loss = q(0.99), rep.max_loss = losses.back();
        rep.time_weighted_loss = sum_chosen / sum_best - 1;
    }
    std::sort(worst.rbegin(), worst.rend());
    for (std::size_t i = 0; i < worst.size() && i < 5; ++i) rep.worst.push_back(worst[i].second);
    return rep;
}

ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t, double tie) {
    return replay(cells, axes, t, params(t), tie);
}

}  // namespace batchlas::tune
