#pragma once

// The tuning ledger, one JSONL file per (op, device), host-only.
// evidence: docs/design/tuning-data.md#tuning-data-ledger-file

#include <tools/tune/tier.hh>
#include <tools/tune/tune_core.hh>

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
    bool verified = true;  // false: timed but never verified in this cell (field v<i> = 0)
};

struct CellRecord {
    std::string run_id;
    Tier tier = Tier::deep;
    CellKey key;
    int round = 0;
    std::vector<CandResult> cands;
    std::vector<std::string> ranked;  // winner first; empty = nothing timed (still a record)
    std::string date;                 // ISO, compares as text
    Phases phases;                    // this measurement's; none on a copy (retract) or an untimed record
};

// RunMeta::protocol of this engine's runs; --plan estimates from history of the same protocol only.
inline constexpr const char* kProtocol = "lean";  // "full": the fixed warm-up engine before it

inline constexpr const char* kUnraced = "unraced:calibrate";  // off-probe arm: skipped, not stale, not unrunnable

struct RunMeta {
    std::string run_id, dtype, host, device, device_name, batchlas, argv, date;
    std::string keys;        // the op's '# keys:' spec text, e.g. "uplo:exact n:log:3 batch:log"
    std::string candidates;  // '|'-joined spellings in candidate order (the tie order)
    Tier tier = Tier::deep;
    std::map<std::string, std::string> worker_mode;
    std::string protocol;  // the per-cell protocol (kProtocol); "" = a ledger from before the field
    std::string guard;     // guard backend: nvml, nvidia-smi or off; "" = a ledger from before the field
    std::string prior;     // calibrate: the borrowed table, <device>@<header date>; "" otherwise
};

std::string make_run_id();  // <UTC yyyymmddThhmmss>-<host>-<pid>
std::string ledger_file(const std::string& root, const std::string& op, const std::string& device);  // <root>/<op>.<device>.jsonl

// Appends to one ledger file shared by every dtype and process: each line under flock(LOCK_EX), one
// write() per line, a torn last line repaired under the same lock. Every line carries meta.dtype;
// cell() stamps the writer's run id and tier.
class LedgerWriter {
public:
    LedgerWriter(std::string path, RunMeta meta);
    ~LedgerWriter();
    LedgerWriter(const LedgerWriter&) = delete;
    LedgerWriter& operator=(const LedgerWriter&) = delete;
    void cell(const CellRecord& r);
    void cell(const CellRecord& r, Tier tier);  // a partial re-race keeps its stored record's tier
    void audit(const CellKey& key, const std::string& verdict, double fresh_ms, double warm_ms, double audit_s = NAN);
    void update_run(const RunMeta& meta);  // a later run line with the same run id; readers keep the last

private:
    void put(const std::string& line);
    void repair_tail();
    std::string path_;
    RunMeta meta_;
    int fd_ = -1;
};

struct Ledger {
    std::vector<RunMeta> runs;
    std::vector<CellRecord> cells;
    std::vector<std::string> warnings;
    std::map<std::string, std::vector<double>> audit_s;  // run id -> its fresh-process audits' wall seconds
};

// `dtype`'s lines of one ledger file in file order (a missing file is an empty ledger); a malformed
// last line warns, an earlier one throws. A repeated (run id, dtype) run line replaces the earlier
// one. `rehash` lines rewrite stored family hashes; a run whose batchlas ends in -dirty is dropped
// with its cells (a warning). sweep_to_table.py's read_ledger is the same rule.
Ledger read_ledger(const std::string& path, const std::string& dtype);

enum class Freshness { current, partly_stale, stale };

// Family = spelling text before the first ':'; `family_hash` maps family -> current hash.
Freshness freshness(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

// Sorted family names that changed, were removed or were never timed.
std::vector<std::string> stale_candidates(const CellRecord& r, const std::map<std::string, std::string>& family_hash);

bool all_error(const CellRecord& r);  // every candidate "error": a failed child, no record

// Per key, skipping all_error and stale records (sweep_to_table.py's rule too): highest tier, then
// newest date, then larger run_id (later line on a full tie).
std::map<CellKey, const CellRecord*> best_records(const Ledger& l, const std::map<std::string, std::string>& family_hash);

// Schema-1 raw sweep -> one run in <ledger_root>/<op>.<device>.jsonl at `tier` (a custom run records itself this way); hashes are
// "legacy:<kernels>" unless `kernels` == `op_hash_now`. A cell's key is the meta's table keys, or,
// when `axes` (the spec's grid axes) is given, every axis in its order, as the tiered driver keys
// cells: a hidden axis (trsm's uplo, diag) missing from a record takes its axis's only value.
void import_schema1(const std::string& raw_jsonl, const std::string& ledger_root,
                    const std::map<std::string, std::string>& family_hash, const std::string& op_hash_now, Tier tier = Tier::deep,
                    const std::vector<std::pair<std::string, std::vector<std::string>>>& axes = {});

}  // namespace batchlas::tune
