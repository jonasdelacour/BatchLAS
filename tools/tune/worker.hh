#pragma once

// The per-GPU worker (host-only): one request_line() per cell on stdin; back come the cell's
// outcome_text() records (as a --cell child writes them) and a {"kind":"done"} line.
// evidence: docs/design/tiered-tuning.md#engine-persistent-workers-and-the-carve-out-audit

#include "spec.hh"
#include "tune_core.hh"

#include <sys/types.h>

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

}  // namespace batchlas::tune
