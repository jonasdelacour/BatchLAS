#pragma once

// The per-GPU worker (host-only): one request_line() per cell on stdin; back come the cell's
// outcome_text() records (as a --cell child writes them) and a {"kind":"done"} line.
// evidence: docs/design/tiered-tuning.md#engine-persistent-workers-and-the-carve-out-audit

#include "spec.hh"
#include "tiered_driver.hh"
#include "tune_core.hh"

#include <sys/types.h>

#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace batchlas::tune {

std::string request_line(const std::string& op, const CellRequest& r);
// (op, request); nullopt with *err on a malformed line.
std::optional<std::pair<std::string, CellRequest>> parse_request(const std::string& line, std::string* err = nullptr);

std::string outcome_text(const std::vector<ArmOutcome>& arms);
std::vector<ArmOutcome> outcomes_from_records(const std::vector<Record>& recs);

class WorkerProcess {
public:
    enum class Got { done, eof, timeout };
    WorkerProcess() = default;
    ~WorkerProcess() { stop(); }
    WorkerProcess(const WorkerProcess&) = delete;
    WorkerProcess& operator=(const WorkerProcess&) = delete;

    bool start(const std::vector<std::string>& argv, const std::vector<std::pair<std::string, std::string>>& env,
               const std::string& log);
    bool alive() const { return pid_ > 0; }
    pid_t pid() const { return pid_; }
    const std::string& log() const { return log_; }
    bool send(const std::string& line);
    // Records up to the next "done" line; eof when the worker exits or closes stdout first.
    Got receive(double timeout_s, std::vector<Record>* out);
    void stop();  // close stdin, then SIGKILL after a short grace, then reap

private:
    pid_t pid_ = -1;
    int to_ = -1, from_ = -1;
    std::string buf_, log_;
};

// Per-worker carve-out and guard bookkeeping. A cell with a smaller per-item footprint than one the
// worker already ran needs a fresh worker; a full guard (utilization too) runs before the first cell
// after a start and then every `every_s` seconds of worker time.
class WorkerGate {
public:
    explicit WorkerGate(double every_s = 60) : every_s_(every_s) {}
    bool restart_before(double footprint) const { return footprint < max_; }
    bool full_guard_due(double now_s) const { return due_ || now_s - last_s_ >= every_s_; }
    void started() { max_ = 0, due_ = true; }
    void ran(double footprint) { max_ = std::max(max_, footprint); }
    void guarded(double now_s) { last_s_ = now_s, due_ = false; }

private:
    double every_s_, max_ = 0, last_s_ = 0;
    bool due_ = true;
};

struct WorkerTry {
    bool ok = false, guard = false;  // guard: numbers discarded for a foreign process, the worker is fine
    std::string error;
    std::vector<ArmOutcome> arms;
};

// Consecutive confirmed errors per arm in one (op, dtype): the worker and the fresh child that re-raced
// the cell both reported `error`. Two in a row bench the arm: it is the candidate, not a poisoned worker.
inline constexpr int kDropAfterErrors = 5;  // evidence: docs/design/tiered-tuning.md#engine-repeat-crashers-are-dropped

class ArmErrors {
public:
    bool benched(const std::string& arm) const;
    bool dropped(const std::string& arm) const;
    void note(const std::string& arm, bool confirmed);  // a confirmed error, or a run without one
    // A benched arm's own fresh child; true when this error dropped it.
    bool note_alone(const std::string& arm, bool error);

private:
    mutable std::mutex mu_;
    std::map<std::string, int> streak_, alone_streak_;
};

// The worker path's failure handling: a failed try restarts the worker (not after a guard discard)
// and retries once, then `fresh` races the cell. An `error` arm in a finished cell may be a sticky
// CUDA error that poisons the process: restart, and the fresh child's result is the one recorded.
// A benched arm skips the worker and races alone in its own fresh child.
ArmBatch race_on_worker(const std::vector<std::string>& arms, ArmErrors& errs,
                        const std::function<WorkerTry(const std::vector<std::string>&)>& attempt,
                        const std::function<void()>& restart,
                        const std::function<ArmBatch(const std::vector<std::string>&)>& fresh);

}  // namespace batchlas::tune
