#include "schedule.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>

namespace batchlas::tune {

namespace {

constexpr double kTie = 0.03;  // tune_core rank(); scripts/sweep_to_table.py TIE

const std::set<std::string> kStatuses{"ok", "skipped", "bad", "error", "eliminated"};

std::string family_of(const std::string& spelling) { return spelling.substr(0, spelling.find(':')); }

bool is_int(const std::string& s) {
    return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
}

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

// Ledger records grouped by their non-integer fields; within a group the integer fields give the
// log distance.
class TimeIndex {
public:
    explicit TimeIndex(const Ledger& l) {
        for (const CellRecord& r : l.cells) {
            Entry e;
            for (const KV& kv : r.key) e.logs.push_back(is_int(kv.value) ? std::log(std::max(std::stod(kv.value), 1.0)) : 0);
            for (const CandResult& c : r.cands)
                if ((c.status == "ok" || c.status == "eliminated") && std::isfinite(c.median_ms) && c.median_ms > 0)
                    e.median[c.cand] = c.median_ms;
            if (!e.median.empty()) groups_[signature(r.key)].push_back(std::move(e));
        }
    }

    // cand -> the median of the nearest record that timed it; absent when none did.
    std::map<std::string, double> nearest(const CellKey& key, const std::vector<std::string>& cands) const {
        std::map<std::string, double> out;
        const auto g = groups_.find(signature(key));
        if (g == groups_.end()) return out;
        std::vector<double> at;
        for (const KV& kv : key) at.push_back(is_int(kv.value) ? std::log(std::max(std::stod(kv.value), 1.0)) : 0);
        std::map<std::string, double> best_d;
        for (const Entry& e : g->second) {
            double d = 0;
            for (std::size_t i = 0; i < at.size(); ++i) d += std::abs(at[i] - e.logs[i]);
            for (const std::string& c : cands) {
                const auto m = e.median.find(c);
                if (m == e.median.end()) continue;
                const auto [it, fresh] = best_d.try_emplace(c, d);
                if (fresh || d < it->second) it->second = d, out[c] = m->second;
            }
        }
        return out;
    }

private:
    struct Entry {
        std::vector<double> logs;
        std::map<std::string, double> median;
    };
    static std::string signature(const CellKey& k) {
        std::string s;
        for (const KV& kv : k) s += kv.name + "=" + (is_int(kv.value) ? std::string("#") : kv.value) + ",";
        return s;
    }
    std::map<std::string, std::vector<Entry>> groups_;
};

double estimate_with(const TimeIndex& idx, const CellKey& key, const std::vector<std::string>& arms, double bytes,
                     const TierParams& p, double overhead_s) {
    const auto near = idx.nearest(key, arms);
    double s = overhead_s;
    for (const std::string& a : arms) {
        const auto it = near.find(a);
        const double ms = it != near.end() ? it->second : bytes / kModelBytesPerS * 1e3;
        s += p.max_reps * ms * 1e-3 + p.warm_topup_s + kVerifyS;
    }
    return s;
}

bool all_error(const CellRecord& r) {
    return !r.cands.empty() &&
           std::all_of(r.cands.begin(), r.cands.end(), [](const CandResult& c) { return c.status == "error"; });
}

// best_records without all-`error` records (a failed child, not a result): such a cell counts as missing.
std::map<CellKey, const CellRecord*> usable_best(const Ledger& l, const std::map<std::string, std::string>& family_hash) {
    Ledger kept;
    std::vector<const CellRecord*> orig;
    for (const CellRecord& r : l.cells)
        if (!all_error(r)) kept.cells.push_back(r), orig.push_back(&r);
    std::map<CellKey, const CellRecord*> out;
    for (const auto& [k, p] : best_records(kept, family_hash)) out[k] = orig[std::size_t(p - kept.cells.data())];
    return out;
}

}  // namespace

double estimate_ms(const Ledger& l, const CellKey& key, const std::string& cand, double bytes) {
    const auto near = TimeIndex(l).nearest(key, {cand});
    return near.empty() ? bytes / kModelBytesPerS * 1e3 : near.begin()->second;
}

double cell_estimate_s(const Ledger& l, const CellKey& key, const std::vector<std::string>& arms, double bytes,
                       const TierParams& p, double overhead_s) {
    return estimate_with(TimeIndex(l), key, arms, bytes, p, overhead_s);
}

std::vector<PlannedCell> plan_round(const PlanSpec& spec, Tier tier, const std::vector<CellKey>& cells, const Ledger& l,
                                    const std::map<std::string, std::string>& family_hash, double cap_gib,
                                    double per_cell_overhead_s,
                                    const std::map<CellKey, std::vector<std::string>>* runnable) {
    const TimeIndex idx(l);
    const auto best = usable_best(l, family_hash);
    const double cap = cap_gib * 1024.0 * 1024.0 * 1024.0;
    std::vector<std::pair<double, PlannedCell>> out;
    for (const CellKey& key : cells) {
        PlannedCell c;
        c.key = key;
        c.tier = tier;
        const double bytes = spec.bytes(key);
        auto finish = [&] { out.emplace_back(bytes, std::move(c)); };
        if (bytes > cap) {
            c.reason = "skip:cap";
            finish();
            continue;
        }
        std::vector<std::string> live = spec.candidates;
        const auto probed = runnable ? runnable->find(key) : decltype(runnable->end()){};
        if (runnable && probed != runnable->end()) {
            std::erase_if(live, [&](const std::string& s) {
                return std::find(probed->second.begin(), probed->second.end(), s) == probed->second.end();
            });
            if (live.size() <= 1) {
                c.reason = "skip:single";
                c.arms = live;
                c.est_s = per_cell_overhead_s;
                finish();
                continue;
            }
        }
        const auto b = best.find(key);
        if (b != best.end() && tier_rank(b->second->tier) >= tier_rank(tier)) {
            const CellRecord& rec = *b->second;
            if (freshness(rec, family_hash) == Freshness::current) {
                c.reason = "skip:current";
                finish();
                continue;
            }
            std::vector<std::string> fams;
            std::set<std::string> want;
            for (const std::string& f : stale_candidates(rec, family_hash))
                for (const std::string& s : live)
                    if (family_of(s) == f) {
                        if (fams.empty() || fams.back() != f) fams.push_back(f);
                        want.insert(s);
                    }
            if (want.empty()) {  // only removed families changed: nothing left to race
                c.reason = "skip:current";
                finish();
                continue;
            }
            for (std::size_t i = 0; i < std::min<std::size_t>(2, rec.ranked.size()); ++i) want.insert(rec.ranked[i]);
            for (const std::string& s : live)
                if (want.count(s)) c.arms.push_back(s);
            c.reason = "partial:" + join(fams, ",");
            c.tier = rec.tier;
            c.stored = &rec;
        } else {
            c.arms = live;
        }
        c.est_s = estimate_with(idx, key, c.arms, bytes, params(c.tier), per_cell_overhead_s);
        finish();
    }
    std::stable_sort(out.begin(), out.end(), [](const auto& x, const auto& y) { return x.first < y.first; });
    std::vector<PlannedCell> plan;
    for (auto& [bytes, c] : out) plan.push_back(std::move(c));
    return plan;
}

std::vector<std::string> op_order(std::vector<std::string> ops) {
    const auto posv = std::find(ops.begin(), ops.end(), "posv");
    std::ptrdiff_t last = -1;
    for (std::size_t i = 0; i < ops.size(); ++i)
        if (ops[i] == "potrf" || ops[i] == "trsm") last = std::ptrdiff_t(i);
    if (posv == ops.end() || posv - ops.begin() > last) return ops;
    ops.erase(posv);
    ops.insert(ops.begin() + last, "posv");
    return ops;
}

std::string budget_warning(double est_s, double budget_h) {
    if (budget_h <= 0 || est_s <= budget_h * 3600) return "";
    char b[200];
    std::snprintf(b, sizeof(b),
                  "--budget %.2f h is below the starting lattice estimate %.2f h: the lattice still completes, "
                  "refinement is skipped",
                  budget_h, est_s / 3600);
    return b;
}

CellRecord record_from_arms(const CellKey& key, int round, const std::vector<ArmOutcome>& arms,
                            const std::vector<std::string>& order, const std::map<std::string, std::string>& family_hash,
                            const CellRecord* stored) {
    auto hash_of = [&](const std::string& cand) {
        const auto it = family_hash.find(family_of(cand));
        return it == family_hash.end() ? std::string() : it->second;
    };
    std::map<std::string, CandResult> by;
    std::set<std::string> raced;
    for (const ArmOutcome& a : arms) raced.insert(a.arm);
    if (stored)
        for (const CandResult& c : stored->cands)
            if (std::find(order.begin(), order.end(), c.cand) != order.end() && hash_of(c.cand) == c.hash) by[c.cand] = c;
    for (const ArmOutcome& a : arms) {
        CandResult c;
        c.cand = a.arm;
        c.hash = hash_of(a.arm);
        c.status = kStatuses.count(a.status) ? a.status : "error";
        c.reason = a.reason;
        if (c.status == "ok" && !a.ms.empty()) {
            c.median_ms = median(a.ms);
            c.lo = *std::min_element(a.ms.begin(), a.ms.end());
            c.hi = *std::max_element(a.ms.begin(), a.ms.end());
            c.reps = static_cast<int>(a.ms.size());
        }
        by[a.arm] = c;
    }
    CellRecord r;
    r.key = key;
    r.round = round;
    std::map<std::string, double> times;
    for (const std::string& s : order) {
        const auto it = by.find(s);
        if (it == by.end()) continue;
        if (raced.count(s) && it->second.status == "ok" && std::isfinite(it->second.median_ms)) times[s] = it->second.median_ms;
        r.cands.push_back(it->second);
        by.erase(it);
    }
    for (auto& [s, c] : by) r.cands.push_back(c);  // arms outside the current list (none from the driver)
    if (!times.empty()) r.ranked = rank(times, order, kTie);
    // Carried-over candidates keep their stored order below every re-raced one: their stored times
    // come from another session, so they never outrank a fresh measurement.
    if (stored)
        for (const std::string& s : stored->ranked)
            if (!raced.count(s) && std::any_of(r.cands.begin(), r.cands.end(), [&](const CandResult& c) { return c.cand == s; }))
                r.ranked.push_back(s);
    return r;
}

CellRecord single_record(const CellKey& key, int round, const std::string& only, const std::vector<std::string>& order,
                         const std::map<std::string, std::string>& family_hash) {
    CellRecord r;
    r.key = key;
    r.round = round;
    for (const std::string& s : order) {
        CandResult c;
        c.cand = s;
        const auto h = family_hash.find(family_of(s));
        c.hash = h == family_hash.end() ? "" : h->second;
        c.status = s == only ? "ok" : "skipped";
        if (s != only) c.reason = "probe: not runnable";
        r.cands.push_back(std::move(c));
    }
    if (!only.empty()) r.ranked = {only};
    return r;
}

double runner_up_gap(const CellRecord& r) {
    const double inf = std::numeric_limits<double>::infinity();
    if (r.ranked.size() < 2) return inf;
    auto ms = [&](const std::string& s) -> double {
        for (const CandResult& c : r.cands)
            if (c.cand == s) return c.median_ms;
        return NAN;
    };
    const double a = ms(r.ranked[0]), b = ms(r.ranked[1]);
    return std::isfinite(a) && std::isfinite(b) && a > 0 ? b / a - 1 : inf;
}

}  // namespace batchlas::tune
