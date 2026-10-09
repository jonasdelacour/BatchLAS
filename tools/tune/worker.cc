#include "worker.hh"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <thread>

namespace batchlas::tune {

namespace {

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

}  // namespace

std::string request_line(const std::string& op, const CellRequest& r) {
    return Json().str("op", op).str("dtype", r.dtype).str("key", key_arg(r.key)).str("arms", join(r.arms, ","))
        .str("mode", r.mode).integer("reps", r.reps).num("warm", r.warm_s).boolean("reverse", r.reverse)
        .integer("ld_pad", r.ld_pad).str("tier", r.tier).integer("min_reps", r.min_reps)
        .integer("max_reps", r.max_reps).num("confidence", r.confidence)
        .boolean("alternate_reverse", r.alternate_reverse).str("seed_order", join(r.seed_order, ","))
        .str("ld_audit", join(r.ld_audit, ",")).line();
}

std::optional<std::pair<std::string, CellRequest>> parse_request(const std::string& line, std::string* err) {
    const auto rec = parse_record(line, err);
    if (!rec) return std::nullopt;
    CellRequest r;
    try {
        r.dtype = rec->get("dtype");
        r.key = parse_key_arg(rec->get("key")).value_or(CellKey{});
        r.arms = split(rec->get("arms"), ',');
        r.mode = rec->get("mode");
        r.reps = static_cast<int>(rec->number("reps"));
        r.warm_s = rec->number("warm");
        r.reverse = rec->get("reverse") == "true";
        r.ld_pad = static_cast<int>(rec->number("ld_pad"));
        r.tier = rec->get("tier");
        r.min_reps = static_cast<int>(rec->number("min_reps"));
        r.max_reps = static_cast<int>(rec->number("max_reps"));
        r.confidence = rec->number("confidence");
        r.alternate_reverse = rec->get("alternate_reverse") == "true";
        r.seed_order = split(rec->get("seed_order"), ',');
        r.ld_audit = split(rec->get("ld_audit"), ',');
    } catch (const std::exception& e) {
        if (err) *err = e.what();
        return std::nullopt;
    }
    const std::string op = rec->get("op");
    if (op.empty() || r.key.empty() || r.arms.empty() || std::isnan(r.warm_s)) {
        if (err) *err = "a request needs op, key, arms and warm";
        return std::nullopt;
    }
    return std::pair(op, std::move(r));
}

std::string outcome_text(const std::vector<ArmOutcome>& arms) {
    std::string out;
    for (const ArmOutcome& o : arms) {
        out += Json().str("kind", "arm").str("arm", o.arm).str("status", o.status).str("reason", o.reason)
                   .num("median_ms", median(o.ms)).num("residual", o.residual).integer("info_nonzero", o.info_nonzero)
                   .integer("reps", static_cast<std::int64_t>(o.ms.size())).str("ld_audit", o.ld_audit).line();
        for (std::size_t r = 0; r < o.ms.size(); ++r)
            out += Json().str("kind", "rep").str("arm", o.arm).integer("rep", static_cast<std::int64_t>(r))
                       .integer("slot", r < o.slot.size() ? o.slot[r] : 0).num("ms", o.ms[r]).line();
    }
    return out;
}

std::vector<ArmOutcome> outcomes_from_records(const std::vector<Record>& recs) {
    std::vector<ArmOutcome> out;
    std::map<std::string, std::size_t> at;
    auto get = [&](const std::string& arm) -> ArmOutcome& {
        const auto [it, fresh] = at.try_emplace(arm, out.size());
        if (fresh) out.push_back({arm, "", "", {}, {}, 0.0, 0});
        return out[it->second];
    };
    for (const Record& r : recs) {
        const std::string kind = r.get("kind");
        if (kind == "rep") {
            ArmOutcome& o = get(r.get("arm"));
            o.ms.push_back(r.number("ms"));
            o.slot.push_back(static_cast<int>(r.number("slot")));
        } else if (kind == "arm") {
            ArmOutcome& o = get(r.get("arm"));
            o.status = r.get("status");
            o.reason = r.get("reason");
            o.residual = r.number("residual");
            o.info_nonzero = static_cast<int>(std::atoll(r.get("info_nonzero", "0").c_str()));
            o.ld_audit = r.get("ld_audit");
        }
    }
    return out;
}

bool WorkerProcess::start(const std::vector<std::string>& argv,
                          const std::vector<std::pair<std::string, std::string>>& env, const std::string& log) {
    stop();
    log_ = log;
    ::signal(SIGPIPE, SIG_IGN);  // a dead worker is an EOF, not a dead driver
    int in[2], out[2];
    if (::pipe2(in, O_CLOEXEC) != 0) return false;
    if (::pipe2(out, O_CLOEXEC) != 0) {
        ::close(in[0]);
        ::close(in[1]);
        return false;
    }
    pid_ = ::fork();
    if (pid_ < 0) {
        for (int fd : {in[0], in[1], out[0], out[1]}) ::close(fd);
        return false;
    }
    if (pid_ == 0) {
        for (const auto& [k, v] : env) ::setenv(k.c_str(), v.c_str(), 1);
        ::dup2(in[0], 0);
        ::dup2(out[1], 1);
        const int fd = log.empty() ? -1 : ::open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) ::dup2(fd, 2);
        std::vector<char*> args;
        for (const auto& a : argv) args.push_back(const_cast<char*>(a.c_str()));
        args.push_back(nullptr);
        ::execv(args[0], args.data());
        std::fprintf(stderr, "execv %s: %s\n", args[0], std::strerror(errno));
        ::_exit(127);
    }
    ::close(in[0]);
    ::close(out[1]);
    to_ = in[1];
    from_ = out[0];
    buf_.clear();
    return true;
}

bool WorkerProcess::send(const std::string& line) {
    if (to_ < 0) return false;
    for (std::size_t done = 0; done < line.size();) {
        const ssize_t n = ::write(to_, line.data() + done, line.size() - done);
        if (n <= 0) return false;
        done += static_cast<std::size_t>(n);
    }
    return true;
}

WorkerProcess::Got WorkerProcess::receive(double timeout_s, std::vector<Record>* out) {
    using clock = std::chrono::steady_clock;
    const auto end = clock::now() + std::chrono::duration<double>(timeout_s);
    while (true) {
        for (std::size_t nl; (nl = buf_.find('\n')) != std::string::npos;) {
            const std::string line = buf_.substr(0, nl);
            buf_.erase(0, nl + 1);
            const auto rec = parse_record(line);
            if (!rec) continue;  // stray output is not a record
            if (rec->get("kind") == "done") return Got::done;
            out->push_back(*rec);
        }
        if (from_ < 0) return Got::eof;
        const double left = std::chrono::duration<double, std::milli>(end - clock::now()).count();
        if (left <= 0) return Got::timeout;
        pollfd p{from_, POLLIN, 0};
        const int ready = ::poll(&p, 1, static_cast<int>(std::min(left, 1000.0)) + 1);
        if (ready < 0 && errno != EINTR) return Got::eof;
        if (ready <= 0) continue;
        char b[4096];
        const ssize_t n = ::read(from_, b, sizeof(b));
        if (n <= 0) return Got::eof;
        buf_.append(b, static_cast<std::size_t>(n));
    }
}

void WorkerProcess::stop() {
    if (to_ >= 0) ::close(to_);
    if (from_ >= 0) ::close(from_);
    to_ = from_ = -1;
    if (pid_ <= 0) return;
    int st = 0;
    for (int i = 0; i < 50 && ::waitpid(pid_, &st, WNOHANG) == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    if (::waitpid(pid_, &st, WNOHANG) == 0) {
        ::kill(pid_, SIGKILL);
        ::waitpid(pid_, &st, 0);
    }
    pid_ = -1;
}

bool ArmErrors::benched(const std::string& arm) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = streak_.find(arm);
    return it != streak_.end() && it->second >= 2;
}

void ArmErrors::note(const std::string& arm, bool confirmed) {
    std::lock_guard<std::mutex> lock(mu_);
    streak_[arm] = confirmed ? streak_[arm] + 1 : 0;
}

bool ArmErrors::dropped(const std::string& arm) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto it = alone_streak_.find(arm);
    return it != alone_streak_.end() && it->second >= kDropAfterErrors;
}

bool ArmErrors::note_alone(const std::string& arm, bool error) {
    std::lock_guard<std::mutex> lock(mu_);
    int& n = alone_streak_[arm];
    n = error ? n + 1 : 0;
    return n == kDropAfterErrors;
}

namespace {

bool is_error(const ArmOutcome& a) { return a.status == "error"; }

ArmBatch race_pool(const std::vector<std::string>& arms, ArmErrors& errs,
                   const std::function<WorkerTry(const std::vector<std::string>&)>& attempt,
                   const std::function<void()>& restart, const std::function<ArmBatch(const std::vector<std::string>&)>& fresh) {
    int restarts = 0;
    std::vector<std::string> erred;
    for (int i = 0; i < 2; ++i) {
        WorkerTry r = attempt(arms);
        if (r.ok) {
            for (const ArmOutcome& a : r.arms)
                if (is_error(a)) erred.push_back(a.arm);
            if (erred.empty()) {
                for (const ArmOutcome& a : r.arms) errs.note(a.arm, false);
                return {std::move(r.arms), "", restarts, false};
            }
            restart();
            ++restarts;
            break;
        }
        if (!r.guard) {
            restart();
            ++restarts;
        }
    }
    ArmBatch b = fresh(arms);
    b.worker_restarts = restarts;
    b.fallback = true;
    for (const ArmOutcome& a : b.arms) {
        const bool worker_erred = std::find(erred.begin(), erred.end(), a.arm) != erred.end();
        if (!is_error(a)) errs.note(a.arm, false);
        else if (worker_erred) errs.note(a.arm, true);  // reproduced in a fresh child: the candidate's own error
    }
    return b;
}

}  // namespace

ArmBatch race_on_worker(const std::vector<std::string>& arms, ArmErrors& errs,
                        const std::function<WorkerTry(const std::vector<std::string>&)>& attempt,
                        const std::function<void()>& restart,
                        const std::function<ArmBatch(const std::vector<std::string>&)>& fresh) {
    std::vector<std::string> pool, alone;
    for (const std::string& a : arms) (errs.benched(a) ? alone : pool).push_back(a);
    ArmBatch b;
    if (!pool.empty()) b = race_pool(pool, errs, attempt, restart, fresh);
    // Errors may depend on the shape, so a benched arm is still raced in every cell, alone.
    for (const std::string& a : alone) {
        b.alone.push_back(a);
        if (errs.dropped(a)) {
            b.arms.push_back({a, "error", "dropped after " + std::to_string(kDropAfterErrors) + " consecutive errors", {}, {}, 0, 0});
            continue;
        }
        ArmBatch f = fresh({a});
        const auto it = std::find_if(f.arms.begin(), f.arms.end(), [&](const ArmOutcome& x) { return x.arm == a; });
        if (it != f.arms.end()) b.arms.push_back(std::move(*it));
        else b.arms.push_back({a, "error", "child: " + f.error, {}, {}, 0, 0});
        if (errs.note_alone(a, is_error(b.arms.back()))) {
            std::printf("arm %s dropped after %d consecutive errors in fresh children: not run again for this op and dtype\n",
                        a.c_str(), kDropAfterErrors);
            std::fflush(stdout);
        }
    }
    b.fallback = b.fallback || pool.empty();  // nothing ran on the worker
    std::stable_sort(b.arms.begin(), b.arms.end(), [&](const ArmOutcome& x, const ArmOutcome& y) {
        return std::find(arms.begin(), arms.end(), x.arm) < std::find(arms.begin(), arms.end(), y.arm);
    });
    return b;
}

}  // namespace batchlas::tune
