#pragma once

// Per-run result ledger, host-only. evidence: docs/design/tiered-tuning.md#engine-the-ledger-and-table-generation

#include "tier.hh"
#include "tune_core.hh"

#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace batchlas::tune {

// status vocabulary and the eliminated-median rule: see the evidence page above.
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

// One write() per line; cell() stamps the writer's run id and tier.
class LedgerWriter {
public:
    LedgerWriter(std::string dir, RunMeta meta);
    ~LedgerWriter();
    LedgerWriter(const LedgerWriter&) = delete;
    LedgerWriter& operator=(const LedgerWriter&) = delete;
    void cell(const CellRecord& r);
    void cell(const CellRecord& r, Tier tier);  // a partial re-race keeps its stored record's tier
    void audit(const CellKey& key, const std::string& verdict, double fresh_ms, double warm_ms);
    void update_run(const RunMeta& meta);  // a later run line with the same run id; readers keep the last

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

// Union of <dir>/*.jsonl except *.reps.jsonl; a malformed last line warns, an earlier one throws.
// A repeated run line (update_run) replaces the earlier one with its run id.
Ledger read_ledger(const std::string& dir);

enum class Freshness { current, partly_stale, stale };

// Family = spelling text before the first ':'; `family_hash` maps family -> current hash.
Freshness freshness(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

// Sorted family names that changed, were removed or were never timed.
std::vector<std::string> stale_candidates(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

// Per key: highest tier, then newest date, then larger run_id (later line on a full tie).
std::map<CellKey, const CellRecord*> best_records(const Ledger& l, const std::map<std::string, std::string>& family_hash);

// Schema-1 raw sweep -> one run file at `tier` (a custom run records itself this way); hashes are
// "legacy:<kernels>" unless `kernels` == `op_hash_now`.
void import_schema1(const std::string& raw_jsonl, const std::string& ledger_root,
                    const std::map<std::string, std::string>& family_hash, const std::string& op_hash_now, Tier tier = Tier::deep);

}  // namespace batchlas::tune
