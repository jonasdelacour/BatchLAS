#include <tools/tune/replay_core.hh>

#include <tools/tune/ledger.hh>
#include <tools/tune/race.hh>

#include <batchlas/error.hh>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

namespace batchlas::tune {

namespace {

constexpr double kMisrank = 1.03;
constexpr double kStrayShare = 0.01;  // a round-0 value this rare beside its axis's commonest is a re-measure, not lattice
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
    if (!f) throw batchlas::invalid_argument("cannot open " + raw_jsonl);
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
        if (!have_meta) throw batchlas::invalid_argument(raw_jsonl + ": record before meta");
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
            if (const double ms = rec->number("ms"); !std::isnan(ms)) meta.total_rep_ms += ms;
            c.reps[attempt][cand][{pass, static_cast<int>(rec->number("rep"))}] = rec->number("ms");
            continue;
        }
        Attempt& a = c.attempts[attempt];
        a.resize(std::max<std::size_t>({a.size(), std::size_t(passes), std::size_t(pass)}));
        a[pass - 1].arms[cand] = {rec->get("status"), rec->get("reason"), rec->number("median_ms")};
    }
    if (!have_meta) throw batchlas::invalid_argument(raw_jsonl + ": no meta record");

    std::vector<std::set<std::string>> seen(names.size());
    std::vector<ReplayCell> out;
    for (auto& [key, c] : raw) {
        if (c.round == 0 && !c.status.empty())
            for (std::size_t i = 0; i < names.size(); ++i) seen[i].insert(key[i].value);
        if (c.status != "ok" || !c.attempts.count(c.final_attempt)) continue;
        ReplayCell rc;
        rc.key = key;
        rc.round = c.round;
        rc.passes = passes;
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

double best_time(const std::map<std::string, double>& times) {
    double b = std::numeric_limits<double>::infinity();
    for (const auto& [cand, t] : times) b = std::min(b, t);
    return b;
}

}  // namespace

void shrink_axis(std::vector<AxisSpec>& axes, const std::string& name, const std::string& spec, bool keep) {
    const auto it = std::find_if(axes.begin(), axes.end(), [&](const AxisSpec& a) { return a.name == name; });
    if (it == axes.end()) throw batchlas::invalid_argument("unknown axis " + name);
    if (!keep) {
        const int k = std::stoi(spec);
        if (k < 1) throw batchlas::invalid_argument("axis stride must be >= 1");
        it->values = subsample_axis(it->values, k);
        return;
    }
    std::vector<std::string> kept;
    for (const std::string& v : split(spec, ':')) {
        if (std::find(it->values.begin(), it->values.end(), v) == it->values.end())
            throw batchlas::invalid_argument("axis " + name + " has no value " + v);
        kept.push_back(v);
    }
    it->values = std::move(kept);
}

ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier, const TierParams& p,
                    double tie, const ReplayOpts& o) {
    ReplayReport rep;
    rep.cells = cells.size();
    std::map<CellKey, std::size_t> at;
    std::vector<std::string> order = o.order;
    double total_reps = 0, used_reps = 0, measure_ms = 0, fixed_s = 0;
    std::vector<std::map<std::string, double>> ref(cells.size());
    for (std::size_t i = 0; i < cells.size(); ++i) {
        at[cells[i].key] = i;
        if (!o.holdout) ref[i] = cells[i].exhaustive;
        for (std::size_t k = 0; o.holdout && k < cells[i].cands.size(); ++k) {
            std::vector<double> v;
            for (std::size_t r = 1; r < cells[i].rounds[k].size(); r += std::size_t(cells[i].passes))
                if (!std::isnan(cells[i].rounds[k][r])) v.push_back(cells[i].rounds[k][r]);
            if (!v.empty()) ref[i][cells[i].cands[k]] = median(v);
        }
        for (const auto& r : cells[i].rounds)
            total_reps += double(std::count_if(r.begin(), r.end(), [](double x) { return !std::isnan(x); }));
        for (const std::string& c : cells[i].cands)
            if (std::find(order.begin(), order.end(), c) == order.end()) order.push_back(c);
    }

    std::map<CellKey, std::vector<std::string>> ranked;
    std::map<CellKey, RefineCell> info;
    RefineOpts opts{p.refine_mode, p.refine_margin, &info, tie};
    bool lattice_round = true;
    auto measure = [&](const CellKey& key) {
        const ReplayCell& c = cells[at.at(key)];
        RaceState s;
        s.cands = c.cands;
        s.ms.assign(c.cands.size(), {});
        s.alive.assign(c.cands.size(), true);
        const std::size_t step = o.holdout ? std::size_t(c.passes) : 1, rounds = c.rounds.front().size() / step;
        fixed_s += o.cell_overhead_s + o.verify_s;
        for (std::size_t r = 0; r < rounds; ++r) {
            for (std::size_t k = 0; k < c.cands.size(); ++k) {
                const double v = c.rounds[k][r * step];
                const bool run = s.alive[k] && !std::isnan(v);
                s.ms[k].push_back(run ? v : kNaN);
                used_reps += run;
                if (run) measure_ms += v;
            }
            if (race_step(s, p, tie) != RaceVerdict::more) break;
        }
        ranked[key] = race_ranking(s, order, tie);
        RefineCell& rc = info[key];
        rc.lattice = lattice_round;
        for (std::size_t k = 0; k < c.cands.size(); ++k) {
            std::vector<double> v;
            for (double x : s.ms[k])
                if (!std::isnan(x)) v.push_back(x);
            if (!v.empty()) rc.ms[c.cands[k]] = median(v);
            fixed_s += lean_arm_fixed_s(v.empty() ? 0.0 : median(v));
        }
        for (const std::string& cand : order)  // the raw file's failed candidates are not runnable here
            if (!rc.ms.count(cand)) rc.out.insert(cand);
    };

    std::vector<CellKey> todo;
    if (o.all_cells) {
        for (const ReplayCell& c : cells) todo.push_back(c.key);
    } else {
        for (const CellKey& k : tier_lattice(axes, p.stride))
            if (at.count(k)) todo.push_back(k);
    }
    std::set<CellKey> unavailable;
    std::size_t refined = 0;
    while (!todo.empty()) {
        for (const CellKey& k : todo) measure(k);
        if (lattice_round) rep.cells_lattice = todo.size();
        lattice_round = false;
        todo.clear();
        if (rep.refine_capped) break;
        for (CellKey& k : refine_all_axes(ranked, axes, p.refine_ratio, opts).next) {
            if (unavailable.count(k)) continue;
            if (at.count(k)) todo.push_back(std::move(k));
            else unavailable.insert(std::move(k));
        }
        // The driver's cap (tiered_driver.cc refine), in refine_all_axes's order: flips before margins.
        const std::size_t allow = refine_allowance(rep.cells_lattice, refined, p.refine_cap_factor);
        if (todo.size() > allow) rep.refine_capped = todo.size() - allow, todo.resize(allow);
        refined += todo.size();
    }
    rep.refine_unavailable = unavailable.size();
    rep.cells_measured = ranked.size();
    rep.reps_fraction = total_reps > 0 ? used_reps / total_reps : 0;
    rep.measure_s = measure_ms / 1000;
    rep.est_gpu_h = (rep.measure_s + fixed_s) / 3600;

    std::vector<std::pair<double, std::string>> worst;
    auto note = [&](const char* what, const ReplayCell& c, const std::string& winner, double ratio) {
        worst.emplace_back(ratio, std::string(what) + " " + key_text(c.key) + " picked " + winner + " x" + std::to_string(ratio));
    };
    std::size_t race_bad = 0, table_bad = 0;
    std::vector<CellKey> rows;
    for (const auto& [key, rk] : ranked) {
        const ReplayCell& c = cells[at.at(key)];
        const auto& ex = ref[at.at(key)];
        const auto won = ex.find(rk.front());
        const double ratio = won == ex.end() ? std::numeric_limits<double>::infinity() : won->second / best_time(ex);
        if (ratio > kMisrank) ++race_bad, note("race", c, rk.front(), ratio);
        rows.push_back(key);
    }
    std::vector<double> losses;
    double sum_chosen = 0, sum_best = 0;
    std::size_t lattice_cells = 0, lattice_bad = 0;
    if (!rows.empty()) {
        const NearestIndex nearest(rows, axes);
        for (std::size_t ci = 0; ci < cells.size(); ++ci) {
            const ReplayCell& c = cells[ci];
            const auto& ex = ref[ci];
            const std::size_t at_row = nearest.find(c.key);
            static const std::vector<std::string> none{"(no row)"};
            const auto& row = at_row == NearestIndex::npos ? none : ranked.at(rows[at_row]);
            const auto pick = std::find_if(row.begin(), row.end(), [&](const std::string& n) { return ex.count(n) != 0; });
            const bool runs = pick != row.end();
            const std::string& winner = runs ? *pick : row.front();
            const double ratio = runs ? ex.at(winner) / best_time(ex) : std::numeric_limits<double>::infinity();
            if (runs) losses.push_back(ratio - 1), sum_chosen += ex.at(winner), sum_best += best_time(ex);
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

TierParams oracle_params() {
    TierParams p = params(Tier::deep);
    p.stride = 1, p.confidence = 1.0, p.min_reps = p.max_reps = 16, p.refine_ratio = 0;
    p.refine_mode = RefineMode::geometric, p.refine_margin = 0;
    return p;
}

void apply_floor(ReplayReport& r, const ReplayReport& oracle) {
    r.floor_race = oracle.race_misrank, r.floor_table = oracle.table_misrank;
    r.excess_race = r.race_misrank - oracle.race_misrank, r.excess_table = r.table_misrank - oracle.table_misrank;
}

ReplayReport replay(const std::vector<ReplayCell>& cells, const std::vector<AxisSpec>& axes, Tier t, double tie) {
    return replay(cells, axes, t, params(t), tie);
}

namespace {

// Axes from the cells' distinct values (log keys numerically) and `out` = the set's arms a cell did not time.
void finish(OracleSet& o, const std::vector<std::string>& key_names) {
    std::set<std::string> arms;
    std::vector<std::pair<std::string, std::vector<std::string>>> values;
    std::vector<std::set<std::string>> seen;
    for (const auto& [key, c] : o.cells) {
        if (values.empty())
            for (const KV& kv : key) values.push_back({kv.name, {}});
        seen.resize(key.size());
        for (std::size_t i = 0; i < key.size(); ++i) seen[i].insert(key[i].value);
        for (const auto& [arm, ms] : c.ms) arms.insert(arm);
    }
    for (std::size_t i = 0; i < values.size(); ++i) values[i].second.assign(seen[i].begin(), seen[i].end());
    o.axes = axis_specs(key_names, values);
    for (AxisSpec& a : o.axes)
        if (a.log)
            std::sort(a.values.begin(), a.values.end(),
                      [](const std::string& x, const std::string& y) { return std::stoll(x) < std::stoll(y); });
    for (auto& [key, c] : o.cells)
        for (const std::string& arm : arms)
            if (!c.ms.count(arm)) c.out.insert(arm);
}

std::vector<std::string> header_field(const std::string& tuned_txt, const std::string& prefix) {
    std::ifstream f(tuned_txt);
    if (!f) throw batchlas::invalid_argument("cannot open " + tuned_txt);
    std::string line;
    while (std::getline(f, line) && line.rfind("#", 0) == 0)
        for (const std::string& w : split(line.substr(1), ' '))
            if (w.rfind(prefix, 0) == 0) return {w.substr(prefix.size())};
    return {};
}

}  // namespace

std::map<std::string, std::string> table_family_hashes(const std::string& tuned_txt) {
    std::map<std::string, std::string> fh;
    for (const std::string& fam : header_field(tuned_txt, "family_kernels="))
        for (const std::string& pair : split(fam, ',')) {
            const auto colon = pair.find(':');
            if (colon != std::string::npos) fh[pair.substr(0, colon)] = pair.substr(colon + 1);
        }
    return fh;
}

OracleSet load_table_oracle(const std::string& tuned_txt) {
    std::ifstream f(tuned_txt);
    if (!f) throw batchlas::invalid_argument("cannot open " + tuned_txt);
    OracleSet o;
    const auto op = header_field(tuned_txt, "op=");
    o.label = op.empty() ? tuned_txt : op.front();
    std::vector<std::string> key_names, names;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("# keys:", 0) == 0) {
            key_names = split(line.substr(7), ' ');
            for (const std::string& kn : key_names) names.push_back(split_fields(kn, ':')[0]);
        }
        if (line.empty() || line[0] == '#') continue;
        if (names.empty()) throw batchlas::invalid_argument(tuned_txt + ": row before '# keys:'");
        auto parts = split_fields(line.substr(0, line.find('#')), '|');
        std::map<std::string, std::string> kv;
        for (const std::string& w : split(parts.front(), ' '))
            if (const auto eq = w.find('='); eq != std::string::npos) kv[w.substr(0, eq)] = w.substr(eq + 1);
        CellKey key;
        for (const std::string& n : names) {
            if (!kv.count(n)) throw batchlas::invalid_argument(tuned_txt + ": row lacks key " + n + ": " + line);
            key.push_back({n, kv[n]});
        }
        RefineCell c;
        bool timed = true;
        std::vector<std::string>& entries = o.rows[key];
        for (std::size_t i = 1; i < parts.size(); ++i) {
            const auto w = split(parts[i], ' ');
            if (!w.empty()) entries.push_back(w[0]);
            if (w.size() != 2 || w[1] == "-") timed = false;
            else if (timed) c.ms[w[0]] = std::stod(w[1]);
        }
        if (timed && !c.ms.empty()) o.cells[key] = std::move(c);
    }
    finish(o, key_names);
    o.key_names = key_names;
    return o;
}

OracleSet load_ledger_oracle(const std::string& ledger_jsonl, const std::map<std::string, std::string>& family_hash) {
    OracleSet o;
    o.label = ledger_jsonl;
    std::vector<std::string> key_names, names;
    std::map<std::string, std::map<std::string, std::size_t>> round0;  // key -> value -> round-0 records
    std::optional<RunMeta> gridded;  // the newest run that recorded its lattice
    for (const char* dtype : {"float", "double", "cfloat", "cdouble"}) {
        const Ledger l = read_ledger(ledger_jsonl, dtype);
        if (l.runs.empty()) continue;
        for (const RunMeta& m : l.runs)
            if (!m.grid.empty() && (!gridded || std::tie(m.date, m.run_id) > std::tie(gridded->date, gridded->run_id))) gridded = m;
        if (key_names.empty()) key_names = split("dtype:exact " + l.runs.back().keys, ' ');
        for (const auto& [rkey, r] : best_records(l, family_hash)) {
            if (names.empty())
                for (const KV& kv : rkey) names.push_back(kv.name);
            CellKey key{{"dtype", dtype}};
            for (const std::string& n : names)
                if (const std::string* v = key_get(rkey, n)) key.push_back({n, *v});
            // A calibrate cell off the probes raced a few arms: not an oracle for the rest.
            if (std::any_of(r->cands.begin(), r->cands.end(), [](const CandResult& x) { return x.reason == kUnraced; })) continue;
            RefineCell c;
            for (const CandResult& cr : r->cands)
                if ((cr.status == "ok" || cr.status == "eliminated") && cr.median_ms > 0) c.ms[cr.cand] = cr.median_ms;
            if (key.size() == names.size() + 1 && !c.ms.empty()) o.cells[key] = std::move(c);
        }
        for (const CellRecord& r : l.cells)
            if (r.round == 0)
                for (const KV& kv : r.key) ++round0[kv.name][kv.value];
    }
    finish(o, key_names);
    o.lattice = "ledger";
    std::vector<AxisSpec> lattice;
    // A frontier or calibrate round 0 is a stride-2 or stride-4 subset: only a run without `grid` infers it.
    for (const std::string& axis : gridded ? split(gridded->grid, ' ') : std::vector<std::string>{}) {
        const auto eq = axis.find('=');
        if (eq != std::string::npos) lattice.push_back({axis.substr(0, eq), true, split(axis.substr(eq + 1), ',')});
    }
    for (const auto& [name, count] : gridded ? decltype(round0){} : round0) {
        std::size_t most = 0;
        for (const auto& [v, n] : count) most = std::max(most, n);
        AxisSpec a{name, true, {}};
        for (const auto& [v, n] : count)
            if (double(n) >= kStrayShare * double(most)) a.values.push_back(v);
        lattice.push_back(std::move(a));
    }
    adopt_lattice(o, lattice);
    return o;
}

void adopt_lattice(OracleSet& o, const std::vector<AxisSpec>& lattice) {
    for (AxisSpec& a : o.axes) {
        const auto it = std::find_if(lattice.begin(), lattice.end(), [&](const AxisSpec& l) { return l.name == a.name; });
        if (!a.log || it == lattice.end() || it->values.empty()) continue;
        a.values = it->values;
        std::sort(a.values.begin(), a.values.end(),
                  [](const std::string& x, const std::string& y) { return std::stoll(x) < std::stoll(y); });
    }
}

OracleSet raw_oracle(const std::string& label, const std::vector<ReplayCell>& cells, const ReplayMeta& meta) {
    OracleSet o;
    o.label = label;
    for (const ReplayCell& c : cells)
        if (!c.exhaustive.empty()) o.cells[c.key].ms = c.exhaustive;
    finish(o, meta.key_names);
    adopt_lattice(o, meta.axes);
    o.lattice = "raw";
    return o;
}

std::vector<Scheme> replay_schemes() {
    return {{"full", 0, false, 0},
            {"lattice", 1, false, 0},
            {"stride2", 2, false, 0},
            {"stride4", 4, false, 0},
            {"stride2+frontier", 2, true, 0},
            {"stride4+frontier", 4, true, 0},
            {"stride2+frontier+prune", 2, true, 1.5},
            {"lattice+bisect", 1, false, 0, true},
            {"stride2+frontier+bisect", 2, true, 0, true}};
}

std::vector<Scheme> calibrate_schemes() {
    Scheme c{"calibrate", 4, true, 0, false, true}, c8{"calibrate8", 8, true, 0, false, true};
    return {c, c8, {"calibrate-probe", 4, false, 0, false, true}, {"prior", 0, false, 0, false, true}};
}

namespace {

// The universe cell `key` raced over `arms` (nullptr, or none of them runs there: every arm) into m.
void race_into(const OracleSet& o, std::map<CellKey, RefineCell>& m, SlowdownReport& rep, const CellKey& key,
               const std::set<std::string>* arms) {
    const RefineCell& t = o.cells.at(key);
    if (arms && std::none_of(t.ms.begin(), t.ms.end(), [&](const auto& a) { return arms->count(a.first) != 0; }))
        arms = nullptr;
    RefineCell& c = m[key];
    for (const auto& [arm, ms] : t.ms)
        if (!arms || arms->count(arm)) c.ms[arm] = ms, rep.arm_runs += 1;
    for (const std::string& arm : t.out)
        if (!arms || arms->count(arm)) c.out.insert(arm);
}

std::vector<std::string> fastest_first(const std::map<std::string, double>& ms) {
    std::vector<std::string> v;
    for (const auto& [arm, t] : ms) v.push_back(arm);
    std::stable_sort(v.begin(), v.end(), [&](const std::string& a, const std::string& b) { return ms.at(a) < ms.at(b); });
    return v;
}

}  // namespace

SlowdownReport slowdown_replay(const OracleSet& o, const Scheme& s, std::size_t k, double slower) {
    SlowdownReport rep;
    std::vector<CellKey> universe;
    for (const auto& [key, c] : o.cells) universe.push_back(key), rep.arm_total += double(c.ms.size());
    rep.cells = universe.size();
    std::map<CellKey, RefineCell> m;
    std::set<CellKey> missed;  // bisection midpoints the universe lacks
    auto measure = [&](const CellKey& key, const std::set<std::string>* arms) { race_into(o, m, rep, key, arms); };
    std::vector<std::set<std::string>> start_all;  // tier_lattice(o.axes, 1) per axis, without the product
    for (const AxisSpec& a : o.axes) start_all.emplace_back(a.values.begin(), a.values.end());
    // The driver's round 0 (frontier_start): the start lattice, own strides for exact-key groups it misses.
    for (const CellKey& c : s.stride > 0 ? frontier_start(universe, o.axes, s.stride, o.lattice != "inferred") : universe)
        measure(c, nullptr);
    // bisect: the driver's model. The frontier picks lattice cells only; midpoints off it come from bisection.
    std::vector<CellKey> lattice_cells;
    std::set<CellKey> on_lattice;
    for (const CellKey& key : universe) {
        bool on = true;
        for (std::size_t i = 0; i < key.size() && on; ++i) on = !o.axes[i].log || start_all[i].count(key[i].value);
        if (on) lattice_cells.push_back(key), on_lattice.insert(key);
    }
    while (s.frontier || s.bisect) {
        auto next = s.frontier ? flip_frontier(m, s.bisect ? lattice_cells : universe, o.axes, k, slower)
                               : std::vector<CellKey>{};
        if (s.bisect) {
            std::map<CellKey, std::vector<std::string>> ranked;
            for (const auto& [key, c] : m)
                if (!c.ms.empty()) ranked[key] = fastest_first(c.ms);
            RefineOpts ro;
            ro.mode = s.stride > 1 ? RefineMode::index : RefineMode::geometric;
            ro.cells = &m;
            const std::set<CellKey> picked(next.begin(), next.end());
            for (const CellKey& mid : refine_all_axes(ranked, o.axes, params(Tier::deep).refine_ratio, ro).next) {
                if (on_lattice.count(mid) || picked.count(mid) || m.count(mid)) continue;
                if (o.cells.count(mid)) next.push_back(mid);
                else rep.bisect_missing += missed.insert(mid).second;
            }
        }
        if (next.empty()) break;
        ++rep.rounds;
        std::vector<std::set<std::string>> arms(next.size());
        if (s.prune > 0) {
            std::vector<CellKey> rows;
            for (const auto& [key, c] : m)
                if (!c.ms.empty()) rows.push_back(key);
            const NearestIndex index(rows, o.axes);
            for (std::size_t i = 0; i < next.size(); ++i)
                for (std::size_t r : index.k_nearest(next[i], k)) {
                    const auto& ms = m.at(rows[r]).ms;
                    const double best = ms.at(fastest_first(ms).front());
                    for (const auto& [arm, t] : ms)
                        if (t <= s.prune * best) arms[i].insert(arm);
                }
        }
        for (std::size_t i = 0; i < next.size(); ++i) measure(next[i], s.prune > 0 && !arms[i].empty() ? &arms[i] : nullptr);
    }
    std::vector<CellKey> rows;
    for (const auto& [key, c] : m)
        if (!c.ms.empty()) rows.push_back(key);
    rep.measured = m.size();
    if (rows.empty()) {
        for (const CellKey& u : universe) rep.failures.push_back(key_text(u) + " (nothing measured)");
        return rep;
    }
    const NearestIndex index(rows, o.axes);
    for (const CellKey& u : universe) {
        const auto& truth = o.cells.at(u).ms;
        const auto own = m.find(u);
        const bool measured = own != m.end() && !own->second.ms.empty();
        const std::size_t near = measured ? 0 : index.find(u);
        if (near == NearestIndex::npos) {
            rep.failures.push_back(key_text(u) + " (no row of its dtype)");
            continue;
        }
        const CellKey& from = measured ? u : rows[near];
        const auto ranked = fastest_first(m.at(from).ms);
        const auto pick = std::find_if(ranked.begin(), ranked.end(), [&](const std::string& a) { return truth.count(a) != 0; });
        if (pick == ranked.end()) {
            rep.failures.push_back(key_text(u) + " (nearest " + key_text(from) + ")");
            continue;
        }
        const double ratio = truth.at(*pick) / truth.at(fastest_first(truth).front());
        rep.ratio.push_back(ratio);
        if (measured) rep.ratio_measured.push_back(ratio);
        if (ratio > rep.worst) rep.worst = ratio, rep.worst_key = key_text(u) + " picked " + *pick;
    }
    return rep;
}

namespace {

std::string family(const std::string& spelling) { return spelling.substr(0, spelling.find(':')); }

std::string dtype_of(const CellKey& k) {
    const std::string* d = key_get(k, "dtype");
    return d ? *d : std::string();
}

}  // namespace

Prior::Prior(const std::string& tuned_txt) : path(tuned_txt), t_(load_table_oracle(tuned_txt)) {
    const std::string stem = std::filesystem::path(tuned_txt).stem().string();
    device = stem.substr(stem.find('.') + 1);
    const auto d = header_field(tuned_txt, "date=");
    date = d.empty() ? "" : d.front();
    rows = t_.rows.size(), timed = t_.cells.size();
    std::vector<std::pair<std::string, std::vector<std::string>>> values;
    std::vector<std::set<std::string>> seen;
    for (const auto& [key, entries] : t_.rows) {
        keys_.push_back(key);
        if (values.empty())
            for (const KV& kv : key) values.push_back({kv.name, {}});
        seen.resize(key.size());
        for (std::size_t i = 0; i < key.size(); ++i) seen[i].insert(key[i].value);
        for (const std::string& e : entries) families_[dtype_of(key)].insert(family(e));
    }
    for (std::size_t i = 0; i < values.size(); ++i) values[i].second.assign(seen[i].begin(), seen[i].end());
    axes_ = axis_specs(t_.key_names, values);
    if (!keys_.empty()) index_.emplace(keys_, axes_);
}

std::vector<std::string> Prior::row(const std::string& dtype, const CellKey& key, const std::vector<std::string>& cands) const {
    if (!index_) return {};
    CellKey q;
    for (const AxisSpec& a : axes_) {
        const std::string* v = a.name == "dtype" ? &dtype : key_get(key, a.name);
        if (!v) return {};
        q.push_back({a.name, *v});
    }
    const std::size_t i = index_->find(q);
    std::vector<std::string> out;
    if (i != NearestIndex::npos)
        for (const std::string& e : t_.rows.at(keys_[i]))
            if (std::find(cands.begin(), cands.end(), e) != cands.end()) out.push_back(e);
    return out;
}

std::vector<std::string> Prior::arms(const std::string& dtype, const CellKey& key, const std::vector<std::string>& cands,
                                     const std::set<std::string>& also) const {
    const auto r = row(dtype, key, cands);
    std::set<std::string> want = also;
    want.insert(r.begin(), r.begin() + std::ptrdiff_t(std::min<std::size_t>(2, r.size())));
    const auto named = families_.find(dtype);
    std::vector<std::string> out;
    for (const std::string& c : cands)
        if (want.count(c) || named == families_.end() || !named->second.count(family(c))) out.push_back(c);
    return out;
}

// The newest table with timed rows wins (header date, then timed share, then device name); untimed-only tables
// only when no other exists. Never the device's own table, never cpu's.
std::string prior_table(const std::string& tuned_dir, const std::string& op, const std::string& device,
                        const std::string& want) {
    namespace fs = std::filesystem;
    if (!want.empty()) {
        const fs::path p = fs::path(tuned_dir) / (op + "." + want + ".txt");
        return fs::exists(p) ? p.string() : "";
    }
    std::string best;
    std::tuple<bool, std::string, double, std::string, std::string> best_rank;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(tuned_dir, ec)) {
        const std::string name = e.path().filename().string();
        if (name.rfind(op + ".", 0) != 0 || e.path().extension() != ".txt") continue;
        const std::string dev = e.path().stem().string().substr(op.size() + 1);
        if (dev.empty() || dev == device || dev == "cpu" || dev.find('.') != std::string::npos) continue;
        const Prior p(e.path().string());
        const double share = p.rows ? double(p.timed) / double(p.rows) : 0.0;
        const bool full = share >= 0.9;  // fully timed: the newest wins; otherwise the larger share
        const auto rank = std::tuple(full, full ? p.date : std::string(), share, p.date, dev);
        if (best.empty() || rank > best_rank) best = e.path().string(), best_rank = rank;
    }
    return best;
}

SlowdownReport calibrate_replay(const OracleSet& o, const Prior& prior, const Scheme& s, std::size_t k, double slower) {
    SlowdownReport rep;
    std::vector<CellKey> universe;
    std::map<std::string, std::vector<std::string>> cands;  // dtype -> every arm its cells name
    for (const auto& [key, c] : o.cells) {
        universe.push_back(key), rep.arm_total += double(c.ms.size());
        auto& v = cands[dtype_of(key)];
        for (const auto& [arm, ms] : c.ms)
            if (std::find(v.begin(), v.end(), arm) == v.end()) v.push_back(arm);
        for (const std::string& arm : c.out)
            if (std::find(v.begin(), v.end(), arm) == v.end()) v.push_back(arm);
    }
    rep.cells = universe.size();
    const std::vector<std::string> none;
    auto cands_of = [&](const CellKey& key) -> const std::vector<std::string>& {
        const auto it = cands.find(dtype_of(key));
        return it == cands.end() ? none : it->second;
    };
    auto row = [&](const CellKey& key) { return prior.row(dtype_of(key), key, cands_of(key)); };
    auto pick = [&](const CellKey& key) {
        const auto r = row(key);
        return r.empty() ? std::string() : r.front();
    };
    std::map<CellKey, RefineCell> m;
    if (s.stride > 0)
        for (const CellKey& c : frontier_start(universe, o.axes, s.stride, o.lattice != "inferred")) race_into(o, m, rep, c, nullptr);
    rep.probes = m.size();
    {
        const CalibrateRound cr(m, o.axes, pick, k, slower);
        for (const auto& [key, c] : m) rep.disagreed += cr.disagrees(key);
    }
    while (s.frontier && !m.empty()) {
        const CalibrateRound cr(m, o.axes, pick, k, slower);
        std::vector<CellKey> open;
        for (const CellKey& u : universe)
            if (!m.count(u)) open.push_back(u);
        const auto next = cr.frontier(open);
        if (next.empty()) break;
        ++rep.rounds;
        std::vector<std::set<std::string>> arms;
        for (const CellKey& u : next) {
            const auto a = prior.arms(dtype_of(u), u, cands_of(u), cr.winners(u));
            arms.emplace_back(a.begin(), a.end());
        }
        for (std::size_t i = 0; i < next.size(); ++i) race_into(o, m, rep, next[i], &arms[i]);
    }
    rep.measured = m.size();
    std::map<CellKey, std::vector<std::string>> table;  // the generated table: measured rows, then copied prior rows
    for (const auto& [key, c] : m)
        if (!c.ms.empty()) table[key] = fastest_first(c.ms);
    std::vector<CellKey> rest;
    for (const CellKey& u : universe)
        if (!m.count(u)) rest.push_back(u);
    std::set<CellKey> contradicted;
    if (!m.empty())
        for (const CellKey& u : CalibrateRound(m, o.axes, pick, k, slower).frontier(rest)) contradicted.insert(u);
    for (const CellKey& u : rest)
        if (!contradicted.count(u))
            if (auto r = row(u); !r.empty()) table[u] = std::move(r), ++rep.prior_rows;
    std::vector<CellKey> keys;
    for (const auto& [key, r] : table) keys.push_back(key);
    if (keys.empty()) {
        for (const CellKey& u : universe) rep.failures.push_back(key_text(u) + " (no row)");
        return rep;
    }
    const NearestIndex index(keys, o.axes);
    for (const CellKey& u : universe) {
        const auto& truth = o.cells.at(u).ms;
        const bool own = table.count(u) != 0;
        const std::size_t near = own ? 0 : index.find(u);
        if (near == NearestIndex::npos) {
            rep.failures.push_back(key_text(u) + " (no row of its dtype)");
            continue;
        }
        const CellKey& from = own ? u : keys[near];
        const auto& ranked = table.at(from);
        const auto p = std::find_if(ranked.begin(), ranked.end(), [&](const std::string& a) { return truth.count(a) != 0; });
        if (p == ranked.end()) {
            rep.failures.push_back(key_text(u) + " (row " + key_text(from) + ")");
            continue;
        }
        const double ratio = truth.at(*p) / truth.at(fastest_first(truth).front());
        rep.ratio.push_back(ratio);
        if (m.count(u) && !m.at(u).ms.empty()) rep.ratio_measured.push_back(ratio);
        if (ratio > rep.worst) rep.worst = ratio, rep.worst_key = key_text(u) + " picked " + *p;
    }
    return rep;
}

SlowdownStats slowdown_stats(std::vector<double> ratio) {
    SlowdownStats st;
    if (ratio.empty()) return st;
    std::sort(ratio.begin(), ratio.end());
    const double n = double(ratio.size());
    auto q = [&](double f) { return ratio[std::size_t(std::max(1.0, std::ceil(f * n - 1e-9))) - 1]; };
    double lg = 0;
    for (double r : ratio) {
        lg += std::log(r);
        st.slow10 += double(r > 1.10) / n;
        st.misrank += double(r > kMisrank) / n;
    }
    st.geomean = std::exp(lg / n) - 1;
    st.p99 = q(0.99), st.p999 = q(0.999), st.worst = ratio.back();
    return st;
}

bool slowdown_gate(const SlowdownReport& r) {
    const SlowdownStats st = slowdown_stats(r.ratio);
    return r.measured > 0 && r.failures.empty() && r.ratio.size() == r.cells && st.geomean <= kGateGeomean &&
           st.p999 <= kGateP999;
}

}  // namespace batchlas::tune
