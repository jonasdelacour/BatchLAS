#include "schedule.hh"

#include "grid.hh"

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
                                    const std::map<CellKey, std::vector<std::string>>* runnable,
                                    std::int64_t max_dim) {
    const TimeIndex idx(l);
    const auto best = best_records(l, family_hash);
    const double cap = cap_gib * 1024.0 * 1024.0 * 1024.0;
    std::vector<std::pair<double, PlannedCell>> out;
    for (const CellKey& key : cells) {
        PlannedCell c;
        c.key = key;
        c.tier = tier;
        const double bytes = spec.bytes(key);
        auto finish = [&] { out.emplace_back(bytes, std::move(c)); };
        if (max_dim > 0 && spec.max_dim && spec.max_dim(key) > max_dim) {
            c.reason = "skip:dim";
            finish();
            continue;
        }
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

const std::map<std::string, std::vector<std::string>>& op_dependencies() {
    static const std::map<std::string, std::vector<std::string>> deps{
        {"trmm", {"gemm"}},
        {"symm", {"gemm"}},
        {"getrf", {"gemm", "trsm"}},
        {"getrs", {"trsm"}},
        {"geqrf", {"gemm"}},
        {"ormqr", {"gemm", "trmm"}},
        {"getri", {"trsm", "getrf"}},
        {"posv", {"potrf", "trsm"}},
        {"gesv", {"getrf", "getrs"}},
        {"orgqr", {"ormqr"}},
        {"syev", {"gemm", "trmm", "syr2k", "geqrf", "ormqr"}},
        {"gesvd", {"gemm", "trmm", "syr2k"}},
    };
    return deps;
}

const std::vector<std::string>& canonical_op_order() {
    static const std::vector<std::string> order{"gemm",  "trsm",  "syr2k", "gemv",  "spmm", "trmm",  "symm",
                                                "syrk",  "potrf", "getrf", "getrs", "geqrf", "ormqr", "getri",
                                                "posv",  "gesv",  "orgqr", "syev",  "gesvd"};
    return order;
}

std::vector<std::string> all_ops(std::vector<std::string> registered) {
    const auto& c = canonical_op_order();
    auto rank = [&](const std::string& op) { return std::find(c.begin(), c.end(), op) - c.begin(); };
    std::stable_sort(registered.begin(), registered.end(),
                     [&](const std::string& a, const std::string& b) { return rank(a) < rank(b); });
    return registered;
}

std::vector<std::string> op_order(std::vector<std::string> ops) {
    const auto& deps = op_dependencies();
    std::vector<std::string> out;
    std::vector<bool> done(ops.size(), false);
    auto emitted = [&](const std::string& op) { return std::find(out.begin(), out.end(), op) != out.end(); };
    auto present = [&](const std::string& op) { return std::find(ops.begin(), ops.end(), op) != ops.end(); };
    while (out.size() < ops.size()) {
        std::size_t pick = ops.size();
        for (std::size_t i = 0; i < ops.size() && pick == ops.size(); ++i) {
            if (done[i]) continue;
            const auto it = deps.find(ops[i]);
            if (it == deps.end() || std::all_of(it->second.begin(), it->second.end(),
                                                [&](const std::string& d) { return !present(d) || emitted(d); }))
                pick = i;
        }
        if (pick == ops.size()) pick = std::size_t(std::find(done.begin(), done.end(), false) - done.begin());
        done[pick] = true;
        out.push_back(ops[pick]);
    }
    return out;
}

std::string budget_warning(double est_s, double budget_h) {
    if (budget_h <= 0 || est_s <= budget_h * 3600) return "";
    char b[200];
    std::snprintf(b, sizeof(b),
                  "--budget %.2f h is below the estimate %.2f h (lattice and refinement): the lattice still "
                  "completes, refinement stops when the budget is spent",
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
        if ((c.status == "ok" || c.status == "eliminated") && !a.ms.empty()) {
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
    std::vector<std::pair<double, std::size_t>> dropped;  // eliminated: (median, position in order)
    for (const std::string& s : order) {
        const auto it = by.find(s);
        if (it == by.end()) continue;
        const bool timed = raced.count(s) && std::isfinite(it->second.median_ms);
        if (timed && it->second.status == "ok") times[s] = it->second.median_ms;
        if (timed && it->second.status == "eliminated") dropped.emplace_back(it->second.median_ms, &s - order.data());
        r.cands.push_back(it->second);
        by.erase(it);
    }
    for (auto& [s, c] : by) r.cands.push_back(c);  // arms outside the current list (none from the driver)
    if (!times.empty()) r.ranked = rank(times, order, kTie);
    // race_ranking: the survivors, then the eliminated by median.
    std::sort(dropped.begin(), dropped.end());
    for (const auto& d : dropped) r.ranked.push_back(order[d.second]);
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

RefineCell refine_cell(const CellRecord& r, bool lattice) {
    RefineCell c;
    c.lattice = lattice;
    for (const CandResult& x : r.cands) {
        const bool timed = (x.status == "ok" || x.status == "eliminated") && std::isfinite(x.median_ms) && x.median_ms > 0;
        if (timed) c.ms[x.cand] = x.median_ms;
        else if (x.status != "ok") c.out.insert(x.cand);  // an untimed "ok" (skip:single) is runnable
    }
    return c;
}

double refine_ratio_estimate(const Ledger& l, Tier tier, double cap_factor, bool* from_history) {
    std::size_t lattice = 0, refined = 0;
    for (const CellRecord& r : l.cells)
        if (r.tier == tier) (r.round == 0 ? lattice : refined)++;
    if (from_history) *from_history = lattice > 0;
    return lattice > 0 ? std::min(cap_factor, double(refined) / double(lattice)) : cap_factor * 0.5;
}

std::vector<std::string> seed_order(const std::map<CellKey, CellRecord>& done, const CellKey& key,
                                    const std::vector<std::string>& arms) {
    auto logs = [](const CellKey& k, std::string* sig) {
        std::vector<double> v;
        for (const KV& kv : k) {
            const bool num = is_int(kv.value);
            v.push_back(num ? std::log(std::max(std::stod(kv.value), 1.0)) : 0);
            *sig += kv.name + "=" + (num ? std::string("#") : kv.value) + ",";
        }
        return v;
    };
    std::string want;
    const std::vector<double> at = logs(key, &want);
    std::string winner;
    double best = std::numeric_limits<double>::infinity();
    for (const auto& [k, r] : done) {
        std::string sig;
        const std::vector<double> l = logs(k, &sig);
        if (sig != want || r.ranked.empty() || k == key) continue;
        double d = 0;
        for (std::size_t i = 0; i < at.size(); ++i) d += std::abs(at[i] - l[i]);
        if (d < best) best = d, winner = r.ranked.front();
    }
    std::vector<std::string> out;
    if (std::find(arms.begin(), arms.end(), winner) != arms.end()) out.push_back(winner);
    for (const std::string& a : arms)
        if (a != winner) out.push_back(a);
    return out;
}

double item_footprint(double bytes, const CellKey& key) {
    const std::string* b = key_get(key, "batch");
    return b && is_int(*b) && std::stod(*b) > 0 ? bytes / std::stod(*b) : bytes;
}

void sort_for_worker(std::vector<const PlannedCell*>& cells, const std::function<double(const CellKey&)>& bytes) {
    std::stable_sort(cells.begin(), cells.end(), [&](const PlannedCell* a, const PlannedCell* b) {
        const double ba = bytes(a->key), bb = bytes(b->key);
        return std::pair(item_footprint(ba, a->key), ba) < std::pair(item_footprint(bb, b->key), bb);
    });
}

bool audit_pick(const std::string& run_id, const CellKey& key, double fraction) {
    return double(fnv1a64(run_id + key_arg(key)) % 1000) < fraction * 1000;
}

namespace {

bool feasible(const std::string& status) { return status == "ok" || status == "eliminated"; }

// A per-candidate disagreement: usable in one process (ok or eliminated; run_race verifies both),
// refused or failing in the other.
bool differ(const std::string& w, const std::string& f) { return feasible(w) != feasible(f); }

// rank()'s winner among the survivors, else the fastest eliminated arm; "" when nothing was timed.
std::string winner_of(const std::vector<ArmOutcome>& arms, const std::vector<std::string>& order, double tie) {
    std::map<std::string, double> ok;
    std::string fallback;
    double best = std::numeric_limits<double>::infinity();
    for (const ArmOutcome& a : arms) {
        const double m = median(a.ms);
        if (!std::isfinite(m)) continue;
        if (a.status == "ok") ok[a.arm] = m;
        else if (a.status == "eliminated" && m < best) best = m, fallback = a.arm;
    }
    return ok.empty() ? fallback : rank(ok, order, tie).front();
}

double median_of(const std::vector<ArmOutcome>& arms, const std::string& arm) {
    for (const ArmOutcome& a : arms)
        if (a.arm == arm) return median(a.ms);
    return NAN;
}

}  // namespace

AuditResult audit_compare(const std::vector<ArmOutcome>& warm, const std::vector<ArmOutcome>& fresh,
                          const std::vector<std::string>& order, double tie, double margin) {
    AuditResult r;
    const std::string ww = winner_of(warm, order, tie), fw = winner_of(fresh, order, tie);
    r.warm_ms = median_of(warm, ww);
    r.fresh_ms = median_of(fresh, fw);
    if (std::none_of(fresh.begin(), fresh.end(), [](const ArmOutcome& a) {
            return feasible(a.status) || a.status == "bad" || a.status == "skipped";
        })) {
        r.verdict = "inconclusive";
        return r;
    }
    std::vector<std::string> bad;
    for (const ArmOutcome& w : warm)
        for (const ArmOutcome& f : fresh)
            if (w.arm == f.arm && differ(w.status, f.status)) bad.push_back(w.arm + " " + w.status + "/" + f.status);
    if (!bad.empty()) {
        r.verdict = "mismatch:feasibility " + join(bad, ",");
        return r;
    }
    const double ww_fresh = median_of(fresh, ww);
    if (ww != fw && !(std::isfinite(ww_fresh) && ww_fresh <= r.fresh_ms * (1 + margin))) {
        r.verdict = "mismatch:winner " + ww + "/" + fw;
        return r;
    }
    r.verdict = "ok";
    return r;
}

std::vector<DominanceLoss> dominance_losses(const CellRecord& r, double bytes, double ratio) {
    std::vector<DominanceLoss> out;
    if (r.ranked.empty()) return out;
    const std::string& winner = r.ranked.front();
    double best = NAN;
    for (const CandResult& c : r.cands)
        if (c.cand == winner) best = c.median_ms;
    if (!(best > 0)) return out;
    for (const CandResult& c : r.cands)
        if (c.status == "eliminated" && std::isfinite(c.median_ms) && c.median_ms > ratio * best)
            out.push_back({c.cand, winner, r.key, bytes});
    return out;
}

// `big` lies beyond `small`: equal non-integer keys and batch, and either larger along `refine_key`
// with every other key equal, or every other integer key >= with more bytes. Batch never carries a
// loss: a one-group-per-matrix kernel that starves at small batch catches up.
bool beyond(const CellKey& small, double small_bytes, const CellKey& big, double big_bytes, const std::string& refine_key) {
    if (small.size() != big.size()) return false;
    bool all_ge = true, refine_only = true, refine_larger = false;
    for (std::size_t i = 0; i < small.size(); ++i) {
        const KV &s = small[i], &b = big[i];
        if (s.name != b.name) return false;
        if (s.value == b.value) continue;
        if (!is_int(s.value) || !is_int(b.value) || s.name == "batch") return false;
        const bool ge = std::stoll(b.value) >= std::stoll(s.value);
        all_ge = all_ge && ge;
        if (s.name == refine_key) refine_larger = ge;
        else refine_only = false;
    }
    return (refine_only && refine_larger) || (all_ge && big_bytes > small_bytes);
}

}  // namespace batchlas::tune
