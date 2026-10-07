#pragma once

// The per-run result ledger (docs/design/tiered-tuning.md, "Ledger"). Host-only. One JSONL file per
// run under <root>/<op>.<dtype>.<device>/; reading takes the union of the files, and each cell's
// best record is picked by tier precedence unless a kernel hash says it is stale.

#include "tier.hh"
#include "tune_core.hh"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace batchlas::tune {

// status: ok | skipped | bad | error | eliminated (raced and dropped by the race); an eliminated
// candidate must carry the median of the reps it did time (rows print it; no median = omitted).
struct CandResult {
    std::string cand, hash, status, reason;
    double median_ms = NAN, lo = NAN, hi = NAN;
    int reps = 0;
};

struct CellRecord {
    std::string run_id;
    Tier tier = Tier::deep;
    CellKey key;
    int round = 0;
    std::vector<CandResult> cands;
    std::vector<std::string> ranked;  // winner first; empty = nothing timed (still a record)
    std::string date;                 // ISO, compares as text
};

struct RunMeta {
    std::string run_id, host, device, device_name, batchlas, argv, date;
    std::string keys;        // the op's '# keys:' spec text, e.g. "uplo:exact n:log:3 batch:log"
    std::string candidates;  // '|'-joined spellings in candidate order (the tie order)
    Tier tier = Tier::deep;
    std::map<std::string, std::string> worker_mode;
};

std::string make_run_id();  // <UTC yyyymmddThhmmss>-<host>-<pid>
std::string ledger_dir(const std::string& root, const std::string& op, const std::string& dtype,
                       const std::string& device);

// Each line is one write() of a complete line, so a kill leaves at most one truncated last line.
// cell() stamps the writer's run id and tier on the record.
class LedgerWriter {
public:
    LedgerWriter(std::string dir, RunMeta meta);
    ~LedgerWriter();
    LedgerWriter(const LedgerWriter&) = delete;
    LedgerWriter& operator=(const LedgerWriter&) = delete;
    void cell(const CellRecord& r);
    void audit(const CellKey& key, const std::string& verdict, double fresh_ms, double warm_ms);

private:
    void put(const std::string& line);
    void repair_tail(const std::string& path);
    RunMeta meta_;
    int fd_ = -1;
};

struct Ledger {
    std::vector<RunMeta> runs;
    std::vector<CellRecord> cells;
    std::vector<std::string> warnings;
};

// Union of <dir>/*.jsonl except *.reps.jsonl. A malformed last line of a file is a warning and is
// skipped; a malformed earlier line throws std::runtime_error.
Ledger read_ledger(const std::string& dir);

enum class Freshness { current, partly_stale, stale };

// Family of a spelling = text before the first ':'. `family_hash` maps family -> current hash.
// stale: the winner's hash differs or its family is gone. partly_stale: another candidate's hash
// differs, or a family exists that the record never saw. An empty ranking has no winner to go stale.
Freshness freshness(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

// Family names (sorted), not spellings: changed, removed (still listed) or never timed. The consumer
// expands them to current spellings and drops removed families.
std::vector<std::string> stale_candidates(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

// Per key: highest tier_rank among non-stale records, then newest date, then larger run_id.
std::map<CellKey, const CellRecord*> best_records(const Ledger& l, const std::map<std::string, std::string>& family_hash);

// Converts a schema-1 raw sweep (streamed) into one deep run file. A candidate's hash is its current
// family hash when the raw `kernels` equals `op_hash_now`, else "legacy:<kernels>".
void import_schema1(const std::string& raw_jsonl, const std::string& ledger_root,
                    const std::map<std::string, std::string>& family_hash, const std::string& op_hash_now);

}  // namespace batchlas::tune
