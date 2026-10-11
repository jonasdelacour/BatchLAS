#include "ledger.hh"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <tuple>

namespace batchlas::tune {

namespace fs = std::filesystem;

namespace {

const std::set<std::string> kStatuses{"ok", "skipped", "bad", "error", "eliminated"};

std::string family_of(const std::string& spelling) { return spelling.substr(0, spelling.find(':')); }

std::string join_bar(const std::vector<std::string>& v) {
    std::string s;
    for (const auto& x : v) s += (s.empty() ? "" : "|") + x;
    return s;
}

std::string host_name() {
    char host[256] = "host";
    gethostname(host, sizeof(host) - 1);
    return host;
}

std::string run_line(const RunMeta& m) {
    Json j;
    j.str("kind", "run").str("run_id", m.run_id).str("host", m.host).str("device", m.device)
        .str("device_name", m.device_name).str("batchlas", m.batchlas).str("argv", m.argv).str("date", m.date)
        .str("tier", to_string(m.tier)).str("keys", m.keys).str("candidates", m.candidates);
    for (const auto& [k, v] : m.worker_mode) j.str("wm." + k, v);
    return j.line();
}

// Candidate i of a cell is spread over fields h<i> s<i> r<i> m<i> lo<i> hi<i> n<i>, so a cell is one line.
std::string cell_line(const CellRecord& r) {
    Json j;
    j.str("kind", "cell").str("run_id", r.run_id).str("tier", to_string(r.tier)).str("key", key_arg(r.key))
        .integer("round", r.round).str("date", r.date).str("ranked", join_bar(r.ranked));
    std::vector<std::string> names;
    for (const auto& c : r.cands) names.push_back(c.cand);
    j.str("cands", join_bar(names));
    for (std::size_t i = 0; i < r.cands.size(); ++i) {
        const CandResult& c = r.cands[i];
        const std::string n = std::to_string(i);
        j.str("h" + n, c.hash).str("s" + n, c.status).str("r" + n, c.reason).num("m" + n, c.median_ms)
            .num("lo" + n, c.lo).num("hi" + n, c.hi).integer("n" + n, c.reps);
    }
    return j.line();
}

Tier need_tier(const Record& r) {
    const auto t = parse_tier(r.get("tier"));
    if (!t) throw std::runtime_error("unknown tier '" + r.get("tier") + "'");
    return *t;
}

RunMeta parse_run(const Record& r) {
    RunMeta m;
    m.run_id = r.get("run_id");
    m.host = r.get("host");
    m.device = r.get("device");
    m.device_name = r.get("device_name");
    m.batchlas = r.get("batchlas");
    m.argv = r.get("argv");
    m.date = r.get("date");
    m.keys = r.get("keys");
    m.candidates = r.get("candidates");
    m.tier = need_tier(r);
    for (const auto& [k, v] : r.s)
        if (k.rfind("wm.", 0) == 0) m.worker_mode[k.substr(3)] = v;
    return m;
}

CellRecord parse_cell(const Record& r) {
    CellRecord c;
    c.run_id = r.get("run_id");
    c.tier = need_tier(r);
    const auto key = parse_key_arg(r.get("key"));
    if (!key) throw std::runtime_error("bad key '" + r.get("key") + "'");
    c.key = *key;
    c.round = static_cast<int>(r.number("round"));
    c.date = r.get("date");
    c.ranked = split(r.get("ranked"), '|');
    const auto names = split(r.get("cands"), '|');
    for (std::size_t i = 0; i < names.size(); ++i) {
        const std::string n = std::to_string(i);
        CandResult cr;
        cr.cand = names[i];
        cr.hash = r.get("h" + n);
        cr.status = r.get("s" + n);
        if (!kStatuses.count(cr.status)) throw std::runtime_error("unknown status '" + cr.status + "'");
        cr.reason = r.get("r" + n);
        cr.median_ms = r.number("m" + n);
        cr.lo = r.number("lo" + n);
        cr.hi = r.number("hi" + n);
        cr.reps = static_cast<int>(r.number("n" + n));
        c.cands.push_back(std::move(cr));
    }
    return c;
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

std::string make_run_id() {
    const std::time_t now = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&now, &tm);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%dT%H%M%S", &tm);
    return std::string(ts) + "-" + host_name() + "-" + std::to_string(getpid());
}

std::string ledger_dir(const std::string& root, const std::string& op, const std::string& dtype,
                       const std::string& device) {
    return root + "/" + op + "." + dtype + "." + device;
}

LedgerWriter::LedgerWriter(std::string dir, RunMeta meta) : meta_(std::move(meta)) {
    fs::create_directories(dir);
    const std::string path = dir + "/" + meta_.run_id + ".jsonl";
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    repair_tail(path);
    put(run_line(meta_));
}

// A reopened file may end in a torn line. A complete record just missing its newline gets one; any
// other fragment is cut, since appending after it would make it a malformed middle line.
void LedgerWriter::repair_tail(const std::string& path) {
    std::string text;
    {
        std::ifstream in(path, std::ios::binary);
        text.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    if (text.empty() || text.back() == '\n') return;
    const std::size_t nl = text.rfind('\n');
    const std::size_t start = nl == std::string::npos ? 0 : nl + 1;
    if (parse_record(std::string_view(text).substr(start))) put("\n");
    else if (::ftruncate(fd_, static_cast<off_t>(start)) != 0) throw std::runtime_error("cannot repair " + path);
}

LedgerWriter::~LedgerWriter() {
    if (fd_ >= 0) ::close(fd_);
}

// An unbuffered write() per line: a kill lands between lines or tears only the one in flight.
void LedgerWriter::put(const std::string& line) {
    std::size_t done = 0;
    while (done < line.size()) {
        const ssize_t n = ::write(fd_, line.data() + done, line.size() - done);
        if (n < 0) throw std::runtime_error("ledger write failed");
        done += static_cast<std::size_t>(n);
    }
}

void LedgerWriter::cell(const CellRecord& r) { cell(r, meta_.tier); }

void LedgerWriter::cell(const CellRecord& r, Tier tier) {
    CellRecord c = r;
    c.run_id = meta_.run_id;
    c.tier = tier;
    put(cell_line(c));
}

void LedgerWriter::audit(const CellKey& key, const std::string& verdict, double fresh_ms, double warm_ms) {
    put(Json().str("kind", "audit").str("run_id", meta_.run_id).str("key", key_arg(key)).str("verdict", verdict)
            .num("fresh_ms", fresh_ms).num("warm_ms", warm_ms).line());
}

void LedgerWriter::update_run(const RunMeta& meta) {
    meta_ = meta;
    put(run_line(meta_));
}

Ledger read_ledger(const std::string& dir) {
    Ledger l;
    std::vector<std::string> files;
    if (fs::is_directory(dir))
        for (const auto& e : fs::directory_iterator(dir)) {
            const std::string n = e.path().filename().string();
            if (ends_with(n, ".jsonl") && !ends_with(n, ".reps.jsonl")) files.push_back(e.path().string());
        }
    std::sort(files.begin(), files.end());
    for (const std::string& path : files) {
        std::ifstream f(path);
        std::vector<std::pair<int, std::string>> lines;
        std::string line;
        for (int no = 1; std::getline(f, line); ++no)
            if (line.find_first_not_of(" \t\r") != std::string::npos) lines.emplace_back(no, line);
        if (!lines.empty() && lines[0].second.rfind("version https://git-lfs.github.com/spec/v1", 0) == 0)
            throw std::runtime_error(path + ": Git LFS pointer: run git lfs pull");
        for (std::size_t i = 0; i < lines.size(); ++i) {
            try {
                std::string err;
                const auto rec = parse_record(lines[i].second, &err);
                if (!rec) throw std::runtime_error(err);
                const std::string kind = rec->get("kind");
                if (kind == "run") {
                    RunMeta m = parse_run(*rec);
                    const auto same = std::find_if(l.runs.begin(), l.runs.end(), [&](const RunMeta& x) { return x.run_id == m.run_id; });
                    if (same == l.runs.end()) l.runs.push_back(std::move(m));
                    else *same = std::move(m);
                } else if (kind == "cell") l.cells.push_back(parse_cell(*rec));
            } catch (const std::exception& e) {
                const std::string where = path + ":" + std::to_string(lines[i].first) + ": " + e.what();
                if (i + 1 != lines.size()) throw std::runtime_error(where);
                l.warnings.push_back("skipping truncated last line, " + where);
            }
        }
    }
    return l;
}

std::vector<std::string> stale_candidates(const CellRecord& r, const std::map<std::string, std::string>& family_hash) {
    std::set<std::string> seen, out;
    for (const CandResult& c : r.cands) {
        const std::string fam = family_of(c.cand);
        seen.insert(fam);
        // A dominated skip was never tried there (removed carry-forward): stale, re-raced.
        if (c.status == "skipped" && c.reason.rfind("dominated:", 0) == 0) {
            out.insert(fam);
            continue;
        }
        if (c.status == "skipped") continue;  // could not run there: its hash cannot change the ranking
        const auto it = family_hash.find(fam);
        if (it == family_hash.end() || it->second != c.hash) out.insert(fam);
    }
    for (const auto& [fam, h] : family_hash)
        if (!seen.count(fam)) out.insert(fam);
    return {out.begin(), out.end()};
}

Freshness freshness(const CellRecord& r, const std::map<std::string, std::string>& family_hash) {
    if (!r.ranked.empty()) {
        const auto it = family_hash.find(family_of(r.ranked.front()));
        const auto win = std::find_if(r.cands.begin(), r.cands.end(),
                                      [&](const CandResult& c) { return c.cand == r.ranked.front(); });
        if (it == family_hash.end() || win == r.cands.end() || win->hash != it->second) return Freshness::stale;
    }
    return stale_candidates(r, family_hash).empty() ? Freshness::current : Freshness::partly_stale;
}

bool all_error(const CellRecord& r) {
    return !r.cands.empty() && std::all_of(r.cands.begin(), r.cands.end(), [](const CandResult& c) { return c.status == "error"; });
}

std::map<CellKey, const CellRecord*> best_records(const Ledger& l, const std::map<std::string, std::string>& family_hash) {
    const auto order = [](const CellRecord& x) { return std::tuple(tier_rank(x.tier), x.date, x.run_id); };
    std::map<CellKey, const CellRecord*> best;
    for (const CellRecord& c : l.cells) {
        if (all_error(c) || freshness(c, family_hash) == Freshness::stale) continue;
        const auto [it, fresh] = best.try_emplace(c.key, &c);
        // >=: on a full tie the later line wins (a re-race appended to the same run).
        if (!fresh && order(c) >= order(*it->second)) it->second = &c;
    }
    return best;
}

namespace {

struct PassInfo {
    std::string status, reason;
    double median = NAN;
    int reps = 0;
};

// One pass record per entry: attempt -> cand -> passes.
using CellPasses = std::map<int, std::map<std::string, std::vector<PassInfo>>>;

CandResult summarize(const std::string& cand, const std::vector<PassInfo>& ps) {
    CandResult c;
    c.cand = cand;
    c.status = "ok";
    double sum = 0, lo = INFINITY, hi = -INFINITY;
    for (const PassInfo& p : ps) {
        if (p.status != "ok" && c.status == "ok") {
            c.status = kStatuses.count(p.status) ? p.status : "error";
            c.reason = p.reason;
        }
        sum += p.median;
        lo = std::min(lo, p.median);
        hi = std::max(hi, p.median);
        c.reps += p.reps;
    }
    if (c.status == "ok") {
        c.median_ms = sum / static_cast<double>(ps.size());
        c.lo = lo;
        c.hi = hi;
    }
    return c;
}

}  // namespace

void import_schema1(const std::string& raw_jsonl, const std::string& ledger_root,
                    const std::map<std::string, std::string>& family_hash, const std::string& op_hash_now, Tier tier,
                    const std::vector<std::pair<std::string, std::vector<std::string>>>& axes) {
    std::ifstream f(raw_jsonl);
    if (!f) throw std::runtime_error("cannot open " + raw_jsonl);
    std::vector<std::string> names, cands;
    std::string kernels, date;
    std::unique_ptr<LedgerWriter> w;
    // Pass records arrive before their cell record; rep records (nearly all the bytes) are skipped unparsed.
    std::map<CellKey, CellPasses> pending;
    std::string line;
    while (std::getline(f, line)) {
        if (line.find("\"kind\": \"rep\"") != std::string::npos) continue;
        const auto rec = parse_record(line);
        if (!rec) continue;
        const std::string kind = rec->get("kind");
        if (kind == "meta") {
            if (rec->get("schema") != "1") throw std::runtime_error(raw_jsonl + ": not a schema 1 sweep");
            for (const std::string& kn : split(rec->get("keys"), ' ')) names.push_back(split_fields(kn, ':')[0]);
            cands = split(rec->get("candidates"), '|');
            kernels = rec->get("kernels");
            RunMeta m;
            m.run_id = make_run_id();
            m.host = host_name();
            m.device = rec->get("device");
            m.device_name = rec->get("device_name");
            m.batchlas = rec->get("batchlas");
            m.argv = rec->get("argv");
            m.date = date = rec->get("date");
            m.tier = tier;
            m.keys = rec->get("keys");
            m.candidates = rec->get("candidates");
            w = std::make_unique<LedgerWriter>(ledger_dir(ledger_root, rec->get("op"), rec->get("dtype"), m.device), m);
            continue;
        }
        if (kind != "pass" && kind != "cell") continue;
        if (!w) throw std::runtime_error(raw_jsonl + ": record before meta");
        CellKey key;
        for (const std::string& n : names)
            if (axes.empty()) key.push_back({n, rec->get(n)});
        for (const auto& [n, values] : axes) {
            const bool fixed = values.size() == 1 && std::find(names.begin(), names.end(), n) == names.end();
            if (!rec->has(n) && !fixed) throw std::runtime_error(raw_jsonl + ": a record lacks grid axis '" + n + "'");
            key.push_back({n, rec->has(n) ? rec->get(n) : values[0]});
        }
        if (kind == "pass") {
            pending[key][static_cast<int>(rec->number("attempt"))][rec->get("cand")].push_back(
                {rec->get("status"), rec->get("reason"), rec->number("median_ms"), static_cast<int>(rec->number("reps"))});
            continue;
        }
        const CellPasses passes = std::move(pending[key]);
        pending.erase(key);
        CellRecord c;
        c.tier = tier;
        c.key = key;
        c.round = static_cast<int>(rec->number("round"));
        c.date = date;
        c.ranked = split(rec->get("ranked"), '|');
        int attempt = static_cast<int>(rec->number("final_attempt"));
        if (!passes.count(attempt) && !passes.empty()) attempt = passes.rbegin()->first;  // nothing timed: show the last try
        const auto at = passes.find(attempt);
        for (const std::string& cand : cands) {
            CandResult cr;
            if (at != passes.end() && at->second.count(cand)) cr = summarize(cand, at->second.at(cand));
            else {
                cr.cand = cand;
                cr.status = "skipped";
                cr.reason = rec->get("reason");
            }
            const auto fh = family_hash.find(family_of(cand));
            cr.hash = (kernels == op_hash_now && fh != family_hash.end()) ? fh->second : "legacy:" + kernels;
            c.cands.push_back(std::move(cr));
        }
        w->cell(c);
    }
    if (!w) throw std::runtime_error(raw_jsonl + ": no meta record");
}

}  // namespace batchlas::tune
