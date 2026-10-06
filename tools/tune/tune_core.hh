#pragma once

// The tuner's host-only logic (flat-kernel-selection.md §6): §6.3 statistics and tie rule, §6.2
// refinement, the JSONL records, the §6.4 hash. No SYCL: tests/tune_tests.cc links it alone.

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace batchlas::tune {

struct KV {
    std::string name;
    std::string value;
    bool operator==(const KV&) const = default;
    bool operator<(const KV& o) const { return std::pair(name, value) < std::pair(o.name, o.value); }
};
using CellKey = std::vector<KV>;  // in the op's '# keys:' order

std::string key_text(const CellKey& k);                    // "uplo=L n=64 batch=8192"
std::string key_arg(const CellKey& k);                     // "uplo=L,n=64,batch=8192"
std::optional<CellKey> parse_key_arg(std::string_view s);  // inverse of key_arg
const std::string* key_get(const CellKey& k, std::string_view name);
std::int64_t key_int(const CellKey& k, std::string_view name);  // throws if absent or not an int
CellKey key_with(CellKey k, std::string_view name, std::string value);

std::vector<std::string> split(std::string_view s, char sep);         // empty fields dropped
std::vector<std::string> split_fields(std::string_view s, char sep);  // empty fields kept
std::vector<std::string> parse_csv_line(std::string_view line);       // "..." quoting, empties kept
std::string csv_field(std::string_view s);                           // quoted when it needs it

double median(std::vector<double> v);  // even count: mean of the middle two; NaN when empty

// Arm indices for timed rep `rep`: the base order (reversed for pass 2) rotated left by rep.
std::vector<std::size_t> rep_order(std::size_t k, int rep, bool reverse);

// Every candidate within `tie` of the best ranks tied, ordered by `order` (candidates<T>()
// order); the rest follow by time, then by order. Spellings outside `order` go last.
std::vector<std::string> rank(const std::map<std::string, double>& times, const std::vector<std::string>& order,
                              double tie = 0.03);

// True when any candidate's pass medians differ by more than `limit`.
bool needs_remeasure(const std::map<std::string, std::vector<double>>& pass_medians, double limit = 0.10);

struct LinePoint {
    std::int64_t n;
    std::string winner;
};

// Next midpoints along one line: between neighbours with different winners while hi/lo >= ratio,
// at round(sqrt(lo*hi)) strictly inside (lo, hi); a bracket of adjacent integers is done.
std::vector<std::int64_t> refine_midpoints(std::vector<LinePoint> line, double ratio = 1.1);

struct ArmSummary {
    std::string status, reason;
    double median = 0;
};
struct PassData {
    std::map<std::string, ArmSummary> arms;
    std::string error;
};
using Attempt = std::vector<PassData>;  // one entry per pass

// §6.3: mean pass median of each candidate "ok" in every pass. final_times uses the newest attempt
// that timed anything (*used = its index, or -1), so a re-measure that failed outright keeps the first.
std::map<std::string, double> attempt_times(const Attempt& a, const std::vector<std::string>& cands);
std::map<std::string, double> final_times(const std::vector<Attempt>& attempts, const std::vector<std::string>& cands,
                                          int* used = nullptr);
bool attempt_needs_remeasure(const Attempt& a, const std::vector<std::string>& cands, double limit);

// One §6.2 round over every cell so far (empty times = no winner), lines varying only `refine_key`.
// A midpoint that exists without a winner is not re-measured; its bracket goes to `stalled`.
struct RefineRound {
    std::vector<CellKey> next;
    std::vector<std::string> stalled;
};
RefineRound refine_round(const std::map<CellKey, std::map<std::string, double>>& measured, const std::string& refine_key,
                         const std::vector<std::string>& order, double tie, double ratio);

// (chosen_origin, chosen_algo) of the last `reached` row for `op`; columns from the file's header.
std::optional<std::pair<std::string, std::string>> reached_route(std::string_view coverage, std::string_view op);

class Json {
public:
    Json& str(std::string_view k, std::string_view v);
    Json& num(std::string_view k, double v);  // shortest round-trip text; non-finite -> null
    Json& integer(std::string_view k, std::int64_t v);
    Json& boolean(std::string_view k, bool v);
    Json& key(const CellKey& key);  // every field; log-key values as integers
    std::string line() const;       // "{...}\n"

private:
    std::string body_;
    void sep(std::string_view k);
};

struct Record {
    std::map<std::string, std::string> s;  // strings, and numbers/bools in their JSON text
    bool has(const std::string& k) const { return s.count(k) != 0; }
    std::string get(const std::string& k, const std::string& dflt = "") const;
    double number(const std::string& k) const;  // NaN when absent or null
};

std::optional<Record> parse_record(std::string_view line, std::string* err = nullptr);
std::vector<Record> read_records(const std::string& path);  // throws std::runtime_error

std::string sha256_hex(std::string_view data);

// Quoted paths between the kernel-sources-begin/-end markers (CI and CMake parse the same block).
std::vector<std::string> parse_kernel_list(std::string_view spec_source);

// sha256 of "<sha256(file)>  <path>\n" lines (sha256sum's format), first 8 hex; nullopt if missing.
std::optional<std::string> kernel_hash(const std::string& repo, const std::vector<std::string>& paths,
                                       std::string* missing = nullptr);

std::string gate_verdict(double ratio_p1, double ratio_p2, bool bad_row, double limit = 1.05);  // §10.3

// --old-csv (dtype, every key, old): a row of another width or of a dtype outside non-empty `dtypes`
// is an error, never a skipped cell.
struct OldChoice {
    std::string dtype;
    CellKey key;
    std::string old;
};
std::optional<std::vector<OldChoice>> parse_old_csv(std::string_view text, const std::vector<std::string>& key_names,
                                                    const std::vector<std::string>& dtypes, std::string* err);
std::pair<std::string, std::string> split_origin(const std::string& spelling);  // "vendor:auto" -> both parts

// Verdicts same|pass|FAIL|BAD_ROW|ERROR|cap. Exit 1 on a FAIL, else 3 on ERROR/BAD_ROW or none gated.
struct GateCounts {
    std::size_t same = 0, pass = 0, fail = 0, bad = 0, error = 0, cap = 0;
};
void gate_count(GateCounts& c, const std::string& verdict);
int gate_exit_code(const GateCounts& c);
std::string gate_summary(const GateCounts& c);

// --devices are nvidia-smi (PCI order) indices. When the launcher saw CUDA_VISIBLE_DEVICES set,
// --devices must be given and every entry must be in that list; otherwise the fence is ignored.
std::optional<std::string> devices_problem(const std::vector<int>& devices, bool given,
                                           const std::optional<std::string>& parent_visible);

// nvidia-smi --query-compute-apps=pid output: every nonblank line that is not self_pid is foreign,
// unparseable ones ("[N/A]", "[Insufficient Permissions]") included.
struct AppScan {
    std::vector<std::string> foreign;
    bool self = false;
};
AppScan scan_compute_apps(std::string_view nvidia_smi_out, long self_pid);

// Before/after a child (README "Guard"): refuse "" = measure; after: untolerated entries = discard.
struct GuardCheck {
    std::string refuse;
    std::vector<std::string> tolerated;
};
GuardCheck guard_before(const AppScan& scan, double util, double ceiling, bool allow_idle_foreign);
std::vector<std::string> guard_new_foreign(const AppScan& after, const std::vector<std::string>& tolerated);

}  // namespace batchlas::tune
