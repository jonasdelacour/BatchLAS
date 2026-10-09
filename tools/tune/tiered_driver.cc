#include "tiered_driver.hh"

#include "grid.hh"
#include "ledger.hh"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace batchlas::tune {

namespace fs = std::filesystem;

namespace {

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

std::string family_of(const std::string& spelling) { return spelling.substr(0, spelling.find(':')); }

std::string fmt(double v, const char* f) {
    char b[64];
    std::snprintf(b, sizeof(b), f, v);
    return b;
}

std::string hours(double s) { return fmt(s / 3600, "%.2f h"); }

std::string capture(const std::string& cmd) {
    std::string out;
    if (FILE* p = ::popen(cmd.c_str(), "r")) {
        char buf[512];
        while (std::fgets(buf, sizeof(buf), p)) out += buf;
        ::pclose(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    return out;
}

std::string today_utc() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char b[16];
    std::strftime(b, sizeof(b), "%Y-%m-%d", &tm);
    return b;
}

std::string host_name() {
    char host[256] = "host";
    ::gethostname(host, sizeof(host) - 1);
    return host;
}

std::vector<std::string> families(const std::vector<std::string>& cands) {
    std::vector<std::string> out;
    for (const std::string& c : cands)
        if (std::find(out.begin(), out.end(), family_of(c)) == out.end()) out.push_back(family_of(c));
    return out;
}

std::map<std::string, std::string> current_hashes(const std::string& repo, const OpSpec& spec, const std::string& dtype) {
    return family_hashes(repo, spec.kernel_block(repo), families(spec.candidates(dtype)));
}

bool is_int(const std::string& s) { return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos; }

// The requested domain of a run, from its full grid: refinement never leaves it.
struct Region {
    std::map<std::string, std::pair<std::int64_t, std::int64_t>> range;
    std::map<std::string, std::set<std::string>> exact;

    explicit Region(const std::vector<CellKey>& grid) {
        std::map<std::string, std::set<std::string>> seen;
        for (const CellKey& k : grid)
            for (const KV& kv : k) seen[kv.name].insert(kv.value);
        for (const auto& [name, vals] : seen) {
            if (!std::all_of(vals.begin(), vals.end(), is_int)) {
                exact[name] = vals;
                continue;
            }
            std::int64_t lo = INT64_MAX, hi = INT64_MIN;
            for (const std::string& v : vals) {
                const std::int64_t x = std::stoll(v);
                lo = std::min(lo, x), hi = std::max(hi, x);
            }
            range[name] = {lo, hi};
        }
    }
    bool contains(const CellKey& k) const {
        for (const KV& kv : k) {
            if (const auto r = range.find(kv.name); r != range.end()) {
                if (!is_int(kv.value) || std::stoll(kv.value) < r->second.first || std::stoll(kv.value) > r->second.second)
                    return false;
            } else if (const auto e = exact.find(kv.name); e != exact.end() && !e->second.count(kv.value)) {
                return false;
            }
        }
        return true;
    }
};

struct Job {
    const OpSpec* spec = nullptr;
    std::string dtype, dir;
    std::vector<std::string> cands;
    std::map<std::string, std::string> fh;
    Ledger ledger;  // read once; plan_round's `stored` pointers point into it
    std::vector<AxisSpec> axes;
    std::unique_ptr<Region> region;
    std::vector<CellKey> next;
    std::map<CellKey, CellRecord> mine;
    std::map<CellKey, std::vector<std::string>> runnable;
    std::set<CellKey> remeasure;  // --remeasure-keys for this op and dtype
    std::set<CellKey> planned;  // never planned twice in a run: a skip:cap or skip:dim midpoint would come back forever
    std::unique_ptr<LedgerWriter> writer;
    RunMeta meta;
    bool fresh = false;  // a failed audit: the rest of this run uses fresh children
    bool audited = false;  // until set, the next eligible worker cell is audited whatever the hash says
    LdAuditBook ld;        // under TieredRun::mu_
    bool capped = false;
    double est_refine_cells = 0;
    std::vector<std::size_t> per_round;  // cells measured per round
    // Scheduler state, under TieredRun::sq_. `queue` points into `plan`, which is replaced only
    // once the round is drained and nothing of it runs (the next round depends on its results).
    std::vector<PlannedCell> plan;
    std::deque<const PlannedCell*> queue;
    std::size_t running = 0;
    int round = 0;
    bool planning = false, done = false;
    double wall = 0;
    PlanSpec plan_spec() const {
        return {cands, [s = spec, d = dtype](const CellKey& k) { return s->bytes(d, k); },
                [s = spec, d = dtype](const CellKey& k) {
                    const auto v = s->dims(d, k);
                    return v.empty() ? std::int64_t(0) : *std::max_element(v.begin(), v.end());
                }};
    }
};

bool measurable(const PlannedCell& c) {
    return c.reason.empty() || c.reason == "remeasure" || c.reason.rfind("partial:", 0) == 0;
}

class TieredRun {
public:
    TieredRun(const TieredOpts& o, const RunIdentity& id, CellMeasurer* m)
        : o_(o), id_(id), m_(m), run_id_(o.run_id.empty() ? make_run_id() : o.run_id), date_(today_utc()) {}
    int go();

private:
    const TieredOpts& o_;
    const RunIdentity& id_;
    CellMeasurer* m_;
    std::string run_id_, date_, batchlas_;
    std::vector<std::unique_ptr<Job>> jobs_;
    std::mutex mu_;
    std::chrono::steady_clock::time_point t0_;
    std::mutex sq_;  // the cell queues; taken before mu_, never inside it
    std::condition_variable cv_;
    std::atomic<std::size_t> budget_left_{0};
    std::vector<double> gpu_max_;  // per GPU: the largest per-item footprint since its last carve-out restart
    std::vector<const Job*> gpu_job_;
    std::size_t restarts_ = 0, switch_restarts_ = 0, switches_ = 0;

    void emit(const Json& j);
    void build(const std::string& op, const std::string& dtype);
    std::vector<PlannedCell> plan(Job& j);
    void print_plan(const Job& j, const std::vector<PlannedCell>& plan, bool cells);
    double cap_factor() const { return o_.refine_cap_factor >= 0 ? o_.refine_cap_factor : params(o_.tier).refine_cap_factor; }
    std::pair<std::size_t, std::size_t> cap_base(const Job& j) const;
    void cap(Job& j, std::vector<PlannedCell>& plan);
    std::size_t stage(Job& j);
    void advance(Job& j);
    void gpu_loop(std::size_t g);
    void note_pop(std::size_t g, const Job& j, const PlannedCell& c);
    void measure_one(Job& j, const PlannedCell& c, int gpu, int round);
    void record(Job& j, const CellRecord& r, Tier tier);
    void audit(Job& j, const CellJob& job, const ArmBatch& warm);
    bool over_budget() const;
    void refine(Job& j);
};

void TieredRun::emit(const Json& j) {
    if (o_.progress_fd < 0) return;
    const std::string line = j.line();
    std::lock_guard<std::mutex> lock(mu_);
    for (std::size_t done = 0; done < line.size();) {
        const ssize_t n = ::write(o_.progress_fd, line.data() + done, line.size() - done);
        if (n <= 0) return;
        done += std::size_t(n);
    }
}

void TieredRun::build(const std::string& op, const std::string& dtype) {
    auto j = std::make_unique<Job>();
    j->spec = find_spec(op);
    if (!j->spec) throw std::runtime_error("unknown op '" + op + "' (--list)");
    const OpSpec& s = *j->spec;
    j->dtype = dtype;
    j->cands = s.candidates(dtype);
    j->fh = current_hashes(o_.repo, s, dtype);
    j->dir = ledger_dir(o_.ledger_root, op, dtype, id_.device);
    j->ledger = read_ledger(j->dir);
    for (const std::string& w : j->ledger.warnings) std::fprintf(stderr, "batchlas_tune: warning: %s\n", w.c_str());
    auto ax = s.axes();
    for (const auto& [name, values] : o_.grid)
        for (auto& a : ax)
            if (a.first == name) a.second = values;
    j->axes = axis_specs(s.key_names(), ax);
    const std::vector<CellKey> grid = s.grid(dtype, o_.grid);
    const std::vector<CellKey> full = tier_lattice(j->axes, Tier::deep);
    // A lattice op subsamples its axes; a demand-driven grid (gemm's grid() override) by key hash.
    const bool lattice = std::set<CellKey>(grid.begin(), grid.end()) == std::set<CellKey>(full.begin(), full.end());
    j->next = lattice ? tier_lattice(j->axes, o_.tier) : tier_subsample(grid, o_.tier);
    // Named cells join round 0 even off the lattice (refinement midpoints), so no budget stops them.
    if (const auto r = o_.remeasure.find(op + "." + dtype); r != o_.remeasure.end()) {
        j->remeasure = r->second;
        std::vector<std::string> names;
        for (const KV& f : j->next.empty() ? grid.front() : j->next.front()) names.push_back(f.name);
        for (const CellKey& k : r->second) {
            std::vector<std::string> got;
            for (const KV& f : k) got.push_back(f.name);
            if (got != names)
                throw std::runtime_error("--remeasure-keys: " + op + " " + dtype + " " + key_arg(k) + ": want the fields " +
                                         join(names, ",") + " in that order");
            if (std::find(j->next.begin(), j->next.end(), k) == j->next.end()) j->next.push_back(k);
        }
    }
    j->region = std::make_unique<Region>(grid);
    jobs_.push_back(std::move(j));
}

std::vector<PlannedCell> TieredRun::plan(Job& j) {
    j.planned.insert(j.next.begin(), j.next.end());
    return plan_round(j.plan_spec(), o_.tier, j.next, j.ledger, j.fh, o_.cap_gib, o_.overhead_s, &j.runnable,
                      o_.max_dim, j.round == 0 ? &j.remeasure : nullptr);
}

void TieredRun::print_plan(const Job& j, const std::vector<PlannedCell>& plan, bool cells) {
    std::map<std::string, int> n;
    double est = 0;
    for (const PlannedCell& c : plan) {
        n[c.reason.empty() ? "measure" : c.reason.rfind("partial:", 0) == 0 ? "partial" : c.reason]++;
        est += c.est_s;
    }
    std::string mix;
    for (const auto& [why, count] : n) mix += (mix.empty() ? "" : ", ") + std::to_string(count) + " " + why;
    std::printf("== plan %s %s %s tier=%s: %zu cells (%s); est %s\n", j.spec->op().c_str(), j.dtype.c_str(),
                id_.device.c_str(), to_string(o_.tier).c_str(), plan.size(), mix.c_str(), hours(est).c_str());
    if (!cells) return;
    for (const PlannedCell& c : plan) {
        std::string line = "  " + key_text(c.key) + "  " + (c.reason.empty() ? std::string("measure") : c.reason);
        if (!c.arms.empty()) line += "  arms=" + join(c.arms, "|");
        if (c.tier != o_.tier) line += "  tier=" + to_string(c.tier);
        if (c.est_s > 0) line += "  est=" + fmt(c.est_s, "%.2fs");
        std::puts(line.c_str());
    }
}

bool TieredRun::over_budget() const {
    return o_.budget_h > 0 &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count() >= o_.budget_h * 3600;
}

void TieredRun::record(Job& j, const CellRecord& r, Tier tier) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!j.writer) {
        RunMeta m;
        m.run_id = run_id_;
        m.host = host_name();
        m.device = id_.device;
        m.device_name = id_.device_name;
        m.batchlas = batchlas_;
        m.argv = o_.argv;
        m.date = date_;
        m.tier = o_.tier;
        // Always the full lists: table generation derives the current families from the newest run.
        m.keys = join(j.spec->key_names(), " ");
        m.candidates = join(j.cands, "|");
        m.worker_mode[j.spec->op() + "." + j.dtype] = m_ && m_->persistent() ? "worker" : "fresh";
        j.meta = m;
        j.writer = std::make_unique<LedgerWriter>(j.dir, m);
    }
    CellRecord c = r;
    c.date = date_;
    c.tier = tier;
    j.writer->cell(c, tier);
    j.mine[c.key] = c;
}

// A fresh child re-measures the cell; a mismatch sends the rest of this op and dtype to fresh children.
void TieredRun::audit(Job& j, const CellJob& job_in, const ArmBatch& warm) {
    // The cell already had its ld audit; an arm it failed is a verdict on ld, not on the worker.
    CellJob job = job_in;
    job.ld_audit.clear();
    auto ld_failed = [&](const std::string& arm) {
        return std::any_of(warm.arms.begin(), warm.arms.end(), [&](const ArmOutcome& x) {
            return x.arm == arm && x.ld_audit == "fail";
        });
    };
    job.arms.erase(std::remove_if(job.arms.begin(), job.arms.end(), ld_failed), job.arms.end());
    const ArmBatch f = m_->measure_fresh(job);
    std::vector<ArmOutcome> on_worker;
    for (const ArmOutcome& x : warm.arms)
        if (std::find(warm.alone.begin(), warm.alone.end(), x.arm) == warm.alone.end() && !ld_failed(x.arm))
            on_worker.push_back(x);
    const AuditResult a = audit_compare(on_worker, f.arms, j.cands);
    const std::string od = j.spec->op() + "." + j.dtype;
    bool flipped = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (j.writer) j.writer->audit(job.key, a.verdict, a.fresh_ms, a.warm_ms);
        if (a.mismatch() && !j.fresh) {
            j.fresh = flipped = true;
            j.meta.worker_mode[od] = "fresh";
            if (j.writer) j.writer->update_run(j.meta);
        }
    }
    std::printf("[gpu%d] %s %s audit: %s%s\n", job.gpu, od.c_str(), key_text(job.key).c_str(), a.verdict.c_str(),
                flipped ? " -> fresh children for the rest of this run" : "");
    std::fflush(stdout);
    emit(Json().str("ev", "audit").str("op", j.spec->op()).str("dtype", j.dtype).key(job.key).str("verdict", a.verdict)
             .num("fresh_ms", a.fresh_ms).num("warm_ms", a.warm_ms).boolean("fresh", flipped));
}

void TieredRun::measure_one(Job& j, const PlannedCell& c, int gpu, int round) {
    const std::string& op = j.spec->op();
    emit(Json().str("ev", "cell_start").str("op", op).str("dtype", j.dtype).key(c.key).integer("gpu", gpu));
    const double bytes = j.spec->bytes(j.dtype, c.key);
    CellJob job{gpu, j.spec, j.dtype, c.key, {}, c.tier, params(c.tier), item_footprint(bytes, c.key)};
    bool worker = false;
    {
        std::lock_guard<std::mutex> lock(mu_);
        job.arms = seed_order(j.mine, c.key, c.arms);
        job.ld_audit = j.ld.want(c.arms);
        worker = m_->persistent() && !j.fresh;
    }
    ArmBatch b = worker ? m_->measure(job) : m_->measure_fresh(job);
    std::vector<std::string> ld_failed;
    {
        std::lock_guard<std::mutex> lock(mu_);
        ld_failed = j.ld.note(c.key, b.arms);
    }
    for (const ArmOutcome& a : b.arms) {
        if (std::find(ld_failed.begin(), ld_failed.end(), a.arm) == ld_failed.end()) continue;
        std::printf("[gpu%d] %s %s %s ld audit FAILED: %s %s\n", gpu, op.c_str(), j.dtype.c_str(),
                    key_text(c.key).c_str(), a.arm.c_str(), a.reason.c_str());
        std::fflush(stdout);
        emit(Json().str("ev", "ld_audit").str("op", op).str("dtype", j.dtype).key(c.key).str("cand", a.arm)
                 .str("verdict", "fail").str("reason", a.reason));
    }
    if (b.worker_restarts > 0)
        emit(Json().str("ev", "worker_restart").str("op", op).str("dtype", j.dtype).key(c.key).integer("gpu", gpu)
                 .integer("restarts", b.worker_restarts).boolean("fallback", b.fallback));
    for (const ArmOutcome& a : b.arms)
        if (a.status == "eliminated")
            emit(Json().str("ev", "eliminated").str("op", op).str("dtype", j.dtype).key(c.key).str("cand", a.arm)
                     .integer("round", static_cast<std::int64_t>(a.ms.size())));
    for (const std::string& a : c.arms)
        if (std::none_of(b.arms.begin(), b.arms.end(), [&](const ArmOutcome& x) { return x.arm == a; }))
            b.arms.push_back({a, "error", "child: " + b.error, {}, {}, 0, 0});
    // A child failure with no definitive arm is not a result: no record, so the next run measures the cell.
    if (!b.error.empty() && std::none_of(b.arms.begin(), b.arms.end(), [](const ArmOutcome& a) {
            return a.status == "ok" || a.status == "eliminated" || a.status == "bad" || a.status == "skipped";
        })) {
        std::printf("[gpu%d] %s %s %s r%d ERROR %s (not recorded)\n", gpu, op.c_str(), j.dtype.c_str(),
                    key_text(c.key).c_str(), round, b.error.c_str());
        std::fflush(stdout);
        emit(Json().str("ev", "cell_done").str("op", op).str("dtype", j.dtype).key(c.key).str("ranked", "")
                 .str("tier", to_string(c.tier)).str("error", b.error));
        return;
    }
    if (c.round >= 0) round = c.round;  // a re-measured refinement midpoint stays a refinement record
    CellRecord r = record_from_arms(c.key, round, b.arms, j.cands, j.fh, c.stored);
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto& live = j.runnable[c.key];
        live.clear();
        for (const ArmOutcome& a : b.arms)
            if (a.status != "skipped") live.push_back(a.arm);
    }
    record(j, r, c.tier);
    std::string summary;
    for (const std::string& s : r.ranked)
        for (const CandResult& cr : r.cands)
            if (cr.cand == s) summary += " | " + s + " " + fmt(cr.median_ms, "%.4g");
    std::printf("[gpu%d] %s %s %s r%d %s%s%s\n", gpu, op.c_str(), j.dtype.c_str(), key_text(c.key).c_str(), round,
                c.reason.empty() ? "" : (c.reason + " ").c_str(), b.error.empty() ? "" : ("ERROR " + b.error + " ").c_str(),
                summary.empty() ? "(nothing ranked)" : summary.c_str());
    std::fflush(stdout);
    emit(Json().str("ev", "cell_done").str("op", op).str("dtype", j.dtype).key(c.key)
             .str("ranked", join(r.ranked, "|")).str("tier", to_string(c.tier)));
    const double fraction = o_.audit_fraction >= 0 ? o_.audit_fraction : params(c.tier).audit_fraction;
    bool pick = audit_pick(run_id_, c.key, fraction);
    if (worker && !b.fallback && b.error.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        pick = pick || !j.audited;  // a hash-picked cell that fell back used up nothing
        j.audited = j.audited || pick;
    }
    if (worker && !b.fallback && b.error.empty() && pick) audit(j, job, b);
}

// Records j.plan's skip:single cells and queues the rest sorted by (per-item footprint, bytes), so
// every GPU popping the queue runs ascending footprints: the carve-out order. Returns the cells queued.
std::size_t TieredRun::stage(Job& j) {
    std::vector<const PlannedCell*> todo;
    for (const PlannedCell& c : j.plan) {
        if (c.reason == "skip:single") {
            record(j, single_record(c.key, j.round, c.arms.empty() ? "" : c.arms[0], j.cands, j.fh), c.tier);
            emit(Json().str("ev", "cell_done").str("op", j.spec->op()).str("dtype", j.dtype).key(c.key)
                     .str("ranked", join(c.arms, "|")).str("tier", to_string(c.tier)));
        } else if (measurable(c)) {
            todo.push_back(&c);
        }
    }
    std::printf("== %s %s %s round %d (%s): %zu of %zu cells to measure\n", j.spec->op().c_str(), j.dtype.c_str(),
                id_.device.c_str(), j.round, to_string(o_.tier).c_str(), todo.size(), j.plan.size());
    std::fflush(stdout);
    sort_for_worker(todo, [&](const CellKey& k) { return j.spec->bytes(j.dtype, k); });
    j.per_round.push_back(todo.size());
    std::lock_guard<std::mutex> lock(sq_);
    j.queue.assign(todo.begin(), todo.end());
    return todo.size();
}

// The job's round is fully recorded: refine, plan and queue the next one, or finish the job.
// Runs without sq_; no other thread touches j meanwhile (its queue is empty and nothing of it runs).
void TieredRun::advance(Job& j) {
    for (;;) {
        if (j.capped) j.next.clear();
        else refine(j);
        if (!j.next.empty() && over_budget()) budget_left_ += j.next.size(), j.next.clear();
        if (j.next.empty()) break;
        j.plan = plan(j);
        cap(j, j.plan);
        ++j.round;
        if (stage(j) > 0) return;
    }
    j.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
    std::lock_guard<std::mutex> lock(sq_);
    j.done = true;
}

// Under sq_. Mirrors WorkerGate's rule to count the carve-out restarts the pop order costs.
void TieredRun::note_pop(std::size_t g, const Job& j, const PlannedCell& c) {
    const double fp = item_footprint(j.spec->bytes(j.dtype, c.key), c.key);
    const bool moved = gpu_job_[g] && gpu_job_[g] != &j;
    switches_ += moved;
    gpu_job_[g] = &j;
    if (fp < gpu_max_[g]) ++restarts_, switch_restarts_ += moved, gpu_max_[g] = fp;
    else gpu_max_[g] = std::max(gpu_max_[g], fp);
}

// One thread per GPU: the next cell of the earliest job (op order) with queued cells. A job whose
// round is drained but still running waits for its stragglers while this GPU works on later jobs.
void TieredRun::gpu_loop(std::size_t g) {
    std::unique_lock<std::mutex> lk(sq_);
    for (;;) {
        Job* j = nullptr;
        for (auto& x : jobs_)
            if (!x->queue.empty()) {
                j = x.get();
                break;
            }
        if (!j) {
            if (std::all_of(jobs_.begin(), jobs_.end(), [](const auto& x) { return x->done; })) return;
            cv_.wait(lk);
            continue;
        }
        const PlannedCell* c = j->queue.front();
        j->queue.pop_front();
        if (j->round == 0 || !over_budget()) {  // round 0 is never budgeted
            note_pop(g, *j, *c);
            ++j->running;
            const int round = j->round;
            lk.unlock();
            measure_one(*j, *c, o_.devices[g], round);
            lk.lock();
            --j->running;
        }
        if (j->queue.empty() && j->running == 0 && !j->planning && !j->done) {
            j->planning = true;
            lk.unlock();
            advance(*j);
            lk.lock();
            j->planning = false;
            cv_.notify_all();
        }
    }
}

// (round-0, refined) records at the running tier: the ledger's current ones overlaid by this run's,
// so a resumed or budget-stopped run refines against the lattice an earlier run measured.
std::pair<std::size_t, std::size_t> TieredRun::cap_base(const Job& j) const {
    std::map<CellKey, int> round;
    for (const auto& [k, r] : best_records(j.ledger, j.fh))
        if (r->tier == o_.tier) round[k] = r->round;
    for (const auto& [k, r] : j.mine)
        if (r.tier == o_.tier) round[k] = r.round;
        else round.erase(k);
    std::pair<std::size_t, std::size_t> n{0, 0};
    for (const auto& [k, r] : round) ++(r == 0 ? n.first : n.second);
    return n;
}

// Refinement records per (op, dtype) <= cap_factor x round-0 records: past it, the first cells in
// refine_all_axes's order (flips, then margin hedges) run and refinement stops, reported.
void TieredRun::cap(Job& j, std::vector<PlannedCell>& plan) {
    const auto [lattice, done] = cap_base(j);
    const std::size_t want = std::size_t(std::count_if(plan.begin(), plan.end(), measurable));
    const std::size_t allow = refine_allowance(lattice, done, cap_factor());
    if (want <= allow) return;
    std::set<CellKey> keep;
    for (const CellKey& k : j.next) {
        if (keep.size() == allow) break;
        const auto c = std::find_if(plan.begin(), plan.end(), [&](const PlannedCell& p) { return p.key == k; });
        if (c != plan.end() && measurable(*c)) keep.insert(k);
    }
    std::erase_if(plan, [&](const PlannedCell& c) { return measurable(c) && !keep.count(c.key); });
    const std::size_t refined = done + allow;
    j.capped = true;
    const std::size_t limit = refine_allowance(lattice, 0, cap_factor());
    std::printf("== %s %s refinement cap hit: %zu refinement cells (cap %.2f x %zu round-0 cells); %zu midpoints left "
                "unmeasured, refinement stops\n", j.spec->op().c_str(), j.dtype.c_str(), refined, cap_factor(),
                lattice, want - allow);
    std::fflush(stdout);
    emit(Json().str("ev", "refine_cap").str("op", j.spec->op()).str("dtype", j.dtype)
             .integer("lattice", std::int64_t(lattice)).integer("refined", std::int64_t(refined))
             .integer("cap", std::int64_t(limit)).integer("dropped", std::int64_t(want - allow)));
}

void TieredRun::refine(Job& j) {
    std::map<CellKey, std::vector<std::string>> ranked;
    std::map<CellKey, RefineCell> cells;
    for (const auto& [k, r] : best_records(j.ledger, j.fh))
        // A lower-tier record is no bracket end: a midpoint on it is measured again. The margin
        // hedges lattice cells of this tier, stored or measured now (a resumed run hedges too).
        if (j.region->contains(k) && tier_rank(r->tier) >= tier_rank(o_.tier))
            ranked[k] = r->ranked, cells[k] = refine_cell(*r, r->round == 0 && r->tier == o_.tier);
    for (const auto& [k, r] : j.mine) ranked[k] = r.ranked, cells[k] = refine_cell(r, r.round == 0);
    const TierParams& p = params(o_.tier);
    RefineOpts ro;
    ro.mode = p.refine_mode;
    ro.margin = p.refine_margin;
    ro.cells = &cells;
    const RefineRound rr = refine_all_axes(ranked, j.axes, p.refine_ratio, ro);
    for (const std::string& s : rr.stalled) std::printf("== %s %s refinement stalled: %s\n", j.spec->op().c_str(), j.dtype.c_str(), s.c_str());
    j.next.clear();
    for (const CellKey& k : rr.next)
        if (!j.planned.count(k)) j.next.push_back(k);
}

int TieredRun::go() {
    for (const std::string& op : o_.ops)
        for (const std::string& dtype : o_.dtypes) {
            const OpSpec* s = find_spec(op);
            if (s && o_.ops.size() > 1) {
                try {
                    (void)s->candidates(dtype);
                } catch (const std::invalid_argument& e) {
                    std::printf("== note: skipping %s %s: %s\n", op.c_str(), dtype.c_str(), e.what());
                    continue;
                }
            }
            build(op, dtype);
        }
    for (const auto& [od, keys] : o_.remeasure)
        if (std::none_of(jobs_.begin(), jobs_.end(), [&](const auto& j) { return j->spec->op() + "." + j->dtype == od; }))
            throw std::runtime_error("--remeasure-keys names " + od + ", which this run does not tune");
    double est = 0, est_refine = 0, refine_cells = 0;
    std::size_t to_measure = 0;
    for (auto& j : jobs_) {
        j->plan = plan(*j);
        print_plan(*j, j->plan, o_.plan);
        double est_measure = 0;
        std::size_t n_measure = 0;
        for (const PlannedCell& c : j->plan) {
            est += c.est_s;
            to_measure += measurable(c) || c.reason == "skip:single";
            if (measurable(c)) est_measure += c.est_s, ++n_measure;
        }
        // Refinement cells follow the lattice cells measured now, at their mean estimate.
        bool history = false;
        const double ratio = refine_ratio_estimate(j->ledger, o_.tier, cap_factor(), &history);
        j->est_refine_cells = ratio * double(n_measure);
        const double s = n_measure ? j->est_refine_cells * est_measure / double(n_measure) : 0;
        refine_cells += j->est_refine_cells, est_refine += s;
        const auto [stored, refined] = cap_base(*j);
        std::printf("   refinement: ~%.0f cells (%.2f per lattice cell, %s), est %s; cap %.2f x round-0 records "
                    "(%zu stored, %zu refined stored, %zu to measure)\n", j->est_refine_cells, ratio,
                    history ? "ledger history" : "no history: cap x 0.5", hours(s).c_str(), cap_factor(), stored,
                    refined, n_measure);
    }
    std::printf("== plan total: %zu cells to measure or probe; est %s lattice only, %s with refinement (~%.0f refinement "
                "cells; %.2f s per child)\n", to_measure, hours(est).c_str(), hours(est + est_refine).c_str(), refine_cells,
                o_.overhead_s);
    if (const std::string w = budget_warning(est + est_refine, o_.budget_h); !w.empty()) std::printf("warning: %s\n", w.c_str());
    std::fflush(stdout);
    emit(Json().str("ev", "plan").integer("cells", std::int64_t(to_measure)).num("est_s", est)
             .integer("refine_cells", std::int64_t(std::llround(refine_cells))).num("est_refine_s", est_refine)
             .num("est_total_s", est + est_refine));
    if (o_.plan) return 0;
    if (!m_) throw std::logic_error("run_tiered: no measurer");
    batchlas_ = git_head(o_.repo);
    t0_ = std::chrono::steady_clock::now();
    for (auto& j : jobs_)
        if (stage(*j) == 0) advance(*j);
    gpu_max_.assign(o_.devices.size(), 0);
    gpu_job_.assign(o_.devices.size(), nullptr);
    std::vector<std::thread> gpus;
    for (std::size_t g = 0; g < o_.devices.size(); ++g) gpus.emplace_back([this, g] { gpu_loop(g); });
    for (auto& t : gpus) t.join();
    if (budget_left_ > 0)
        std::printf("== budget %.2f h spent: %zu refinement cells left for a later run\n", o_.budget_h,
                    budget_left_.load());
    for (auto& j : jobs_) {
        std::string rounds;
        for (std::size_t n : j->per_round) rounds += (rounds.empty() ? "" : ",") + std::to_string(n);
        const std::size_t refined = std::accumulate(j->per_round.begin() + 1, j->per_round.end(), std::size_t(0));
        std::printf("== summary %s %s: measured %zu lattice + %zu refinement cells, per round %s; planned ~%.0f "
                    "refinement%s; wall %.1f s\n", j->spec->op().c_str(), j->dtype.c_str(), j->per_round[0], refined,
                    rounds.c_str(), j->est_refine_cells, j->capped ? "; refinement cap hit" : "", j->wall);
    }
    for (auto& j : jobs_) {
        std::printf("== summary %s %s ld audit (ld +%d): %zu passed, %zu skipped, %zu failed\n", j->spec->op().c_str(),
                    j->dtype.c_str(), kLdAuditPad, j->ld.passed, j->ld.skipped, j->ld.failed.size());
        for (const std::string& f : j->ld.failed) std::printf("   ld audit FAILED %s\n", f.c_str());
        emit(Json().str("ev", "ld_audit_summary").str("op", j->spec->op()).str("dtype", j->dtype)
                 .integer("passed", std::int64_t(j->ld.passed)).integer("skipped", std::int64_t(j->ld.skipped))
                 .integer("failed", std::int64_t(j->ld.failed.size())).str("failures", join(j->ld.failed, "; ")));
    }
    std::printf("== summary workers: %zu (op, dtype) switches; %zu carve-out restarts by footprint order, %zu of "
                "them at a switch\n", switches_, restarts_, switch_restarts_);
    std::fflush(stdout);
    emit(Json().str("ev", "schedule").integer("switches", std::int64_t(switches_))
             .integer("restarts", std::int64_t(restarts_)).integer("switch_restarts", std::int64_t(switch_restarts_)));
    for (auto& j : jobs_) {
        if (!j->writer) continue;
        std::printf("== wrote %s/%s.jsonl: %zu cells\n", j->dir.c_str(), run_id_.c_str(), j->mine.size());
        if (o_.out.empty()) continue;
        const std::string cmd = "python3 '" + o_.repo + "/scripts/sweep_to_table.py' --ledger '" + j->dir + "' --out '" + o_.out + "'";
        if (std::system(cmd.c_str()) != 0) throw std::runtime_error("table generation failed: " + cmd);
    }
    emit(Json().str("ev", "done"));
    return 0;
}

std::time_t parse_date(const std::string& d) {
    std::tm tm{};
    std::istringstream in(d.substr(0, 10));
    in >> std::get_time(&tm, "%Y-%m-%d");
    return in.fail() ? 0 : timegm(&tm);
}

struct StatusRow {
    bool ledger = false;
    std::size_t runs = 0, cells = 0, stale = 0, partial = 0;
    std::map<Tier, int> tiers;
    std::string newest, table, error;
};

std::string tier_mix(const std::map<Tier, int>& t) {
    std::string s;
    for (Tier x : {Tier::deep, Tier::coarse, Tier::preview, Tier::custom, Tier::transcribed})
        if (t.count(x)) s += (s.empty() ? "" : ",") + to_string(x) + ":" + std::to_string(t.at(x));
    return s.empty() ? "-" : s;
}

// "source tiers date" from a table's header; without a tiers= word, its row count.
std::string table_summary(const fs::path& p) {
    std::ifstream f(p);
    std::string line, date, source, tiers;
    std::size_t rows = 0;
    while (std::getline(f, line)) {
        if (line.rfind("#", 0) != 0) {
            rows += !line.empty();
            continue;
        }
        for (const std::string& w : split(line, ' ')) {
            if (w.rfind("date=", 0) == 0) date = w.substr(5);
            else if (w.rfind("source=", 0) == 0) source = w.substr(7, w.find(':') == std::string::npos ? std::string::npos : w.find(':') - 7);
            else if (w.rfind("tiers=", 0) == 0) tiers = w.substr(6);
        }
    }
    return (source.empty() ? "?" : source) + " " + (tiers.empty() ? "rows:" + std::to_string(rows) : tiers) + " " + date;
}

}  // namespace

std::string git_head(const std::string& repo) {
    std::string sha = capture("git -C '" + repo + "' rev-parse --short=8 HEAD 2>/dev/null");
    if (!capture("git -C '" + repo + "' status --porcelain --untracked-files=no 2>/dev/null").empty()) sha += "-dirty";
    return sha.empty() ? "unknown" : sha;
}

int run_tiered(const TieredOpts& o, const RunIdentity& id, CellMeasurer* m) { return TieredRun(o, id, m).go(); }

int status_main(const std::string& repo, const std::string& ledger_root, const std::string& tuned_dir) {
    std::map<std::tuple<std::string, std::string, std::string>, StatusRow> rows;
    std::map<std::pair<std::string, std::string>, std::map<std::string, std::string>> hashes;
    auto hashes_of = [&](const std::string& op, const std::string& dtype) -> const std::map<std::string, std::string>& {
        auto [it, fresh] = hashes.try_emplace({op, dtype});
        if (fresh)
            if (const OpSpec* s = find_spec(op)) {
                try {
                    it->second = current_hashes(repo, *s, dtype);
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "batchlas_tune: %s %s: %s\n", op.c_str(), dtype.c_str(), e.what());
                }
            }
        return it->second;
    };
    auto ident = [](const std::string& name) -> std::optional<std::tuple<std::string, std::string, std::string>> {
        const auto p = split(name, '.');
        if (p.size() < 3) return std::nullopt;
        return std::tuple(p[0], p[1], join(std::vector<std::string>(p.begin() + 2, p.end()), "."));
    };
    std::error_code ec;
    if (fs::is_directory(ledger_root, ec))
        for (const auto& e : fs::directory_iterator(ledger_root)) {
            const auto id = e.is_directory() ? ident(e.path().filename().string()) : std::nullopt;
            if (!id) continue;
            StatusRow& r = rows[*id];
            r.ledger = true;
            try {
                const Ledger l = read_ledger(e.path().string());
                const auto& fh = hashes_of(std::get<0>(*id), std::get<1>(*id));
                const auto best = best_records(l, fh);
                std::set<CellKey> keys;
                for (const CellRecord& c : l.cells) keys.insert(c.key);
                r.runs = l.runs.size();
                r.cells = keys.size();
                r.stale = keys.size() - best.size();
                for (const auto& [k, c] : best) {
                    r.tiers[c->tier]++;
                    r.partial += freshness(*c, fh) == Freshness::partly_stale;
                }
                for (const RunMeta& m : l.runs) r.newest = std::max(r.newest, m.date);
            } catch (const std::exception& ex) {
                r.error = ex.what();
            }
        }
    if (fs::is_directory(tuned_dir, ec))
        for (const auto& e : fs::directory_iterator(tuned_dir)) {
            if (e.path().extension() != ".txt") continue;
            const auto id = ident(e.path().stem().string());
            if (!id || (!find_spec(std::get<0>(*id)) && !rows.count(*id))) continue;
            rows[*id].table = table_summary(e.path());
        }
    const std::time_t now = std::time(nullptr);
    std::printf("%-6s %-7s %-7s | %4s %6s %-40s %5s %7s %-16s | %s\n", "op", "dtype", "device", "runs", "cells",
                "ledger tiers (best record per cell)", "stale", "partial", "newest run", "table: source tiers date");
    for (const auto& [id, r] : rows) {
        std::string newest = "-";
        if (!r.newest.empty()) {
            const std::time_t t = parse_date(r.newest);
            newest = r.newest.substr(0, 10) + (t ? " (" + std::to_string((now - t) / 86400) + " d)" : "");
        }
        std::printf("%-6s %-7s %-7s | %4zu %6zu %-40s %5zu %7zu %-16s | %s%s\n", std::get<0>(id).c_str(),
                    std::get<1>(id).c_str(), std::get<2>(id).c_str(), r.runs, r.cells,
                    r.ledger ? tier_mix(r.tiers).c_str() : "(no ledger)", r.stale, r.partial, newest.c_str(),
                    r.table.empty() ? "(no table)" : r.table.c_str(), r.error.empty() ? "" : ("  ERROR " + r.error).c_str());
    }
    return 0;
}

int import_raw_main(const std::string& repo, const std::string& ledger_root, const std::string& raw) {
    std::ifstream f(raw);
    std::string first;
    if (!f || !std::getline(f, first)) throw std::runtime_error("cannot read " + raw);
    const auto meta = parse_record(first);
    if (!meta || meta->get("kind") != "meta") throw std::runtime_error(raw + ": the first line is not a meta record");
    const std::string op = meta->get("op"), dtype = meta->get("dtype"), device = meta->get("device");
    const OpSpec* spec = find_spec(op);
    if (!spec) throw std::runtime_error(raw + ": no spec for op '" + op + "'");
    const auto op_hash = kernel_hash(repo, spec->kernel_sources());
    import_schema1(raw, ledger_root, current_hashes(repo, *spec, dtype), op_hash.value_or(""), Tier::deep, spec->axes());
    std::printf("imported %s into %s as a deep run%s\n", raw.c_str(), ledger_dir(ledger_root, op, dtype, device).c_str(),
                op_hash && *op_hash == meta->get("kernels") ? "" : " (kernels changed since: hashes are legacy:<kernels>)");
    return 0;
}

}  // namespace batchlas::tune
