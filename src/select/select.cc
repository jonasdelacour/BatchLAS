#include "select.hh"

#include "../util/resident_capacity.hh"

#include <batchlas/settings.hh>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <tuple>
#include <unordered_map>

namespace batchlas::select {

struct Table::Memo {
    std::mutex mu;
    std::unordered_map<std::string, std::size_t> row;  // kNoRow: no row matched
};

namespace {

std::string_view trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    return s;
}

std::vector<std::string_view> words(std::string_view s) {
    std::vector<std::string_view> out;
    std::size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        const std::size_t start = i;
        while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i > start) out.push_back(s.substr(start, i - start));
    }
    return out;
}

// A plain positive decimal: digits with at most one '.', no sign, no exponent.
bool parse_weight(std::string_view s, double& out) {
    if (s.empty() || std::count(s.begin(), s.end(), '.') > 1) return false;
    if (!std::all_of(s.begin(), s.end(), [](char c) { return c == '.' || (c >= '0' && c <= '9'); })) return false;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
    return r.ec == std::errc{} && r.ptr == s.data() + s.size() && std::isfinite(out) && out > 0;
}

int leading_int(std::string_view s) {
    int v = 0;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

std::optional<Op> op_from_name(std::string_view op) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(Op::COUNT); ++i) {
        const auto o = static_cast<Op>(i);
        if (op_name(o) == op) return o;
    }
    return std::nullopt;
}

std::string_view dtype_from_scalar(ScalarKind s) {
    switch (s) {
        case ScalarKind::F32: return "float";
        case ScalarKind::F64: return "double";
        case ScalarKind::C32: return "cfloat";
        case ScalarKind::C64: return "cdouble";
    }
    return "?";
}

// "<op>.<dtype>.<device>.txt" -> {op, dtype, device}; empty strings when it does not fit.
std::array<std::string, 3> split_file_name(std::string_view file) {
    std::array<std::string, 3> out;
    const auto slash = file.find_last_of('/');
    if (slash != std::string_view::npos) file = file.substr(slash + 1);
    const auto parts = detail::split(file, '.');
    if (parts.size() == 4 && parts[3] == "txt")
        for (std::size_t i = 0; i < 3; ++i) out[i] = std::string(parts[i]);
    return out;
}

std::mutex& state_mutex() {
    static auto* m = new std::mutex();  // leaked: choose() may run from static destructors
    return *m;
}

std::set<std::string>& warned() {
    static auto* s = new std::set<std::string>();
    return *s;
}

bool warn_once(const std::string& tag) {
    std::lock_guard<std::mutex> lock(state_mutex());
    return warned().insert(tag).second;
}

struct TableState {
    std::optional<std::vector<std::pair<std::string, std::string>>> test_builtin;
    int generation = 0;
    std::map<std::string, std::vector<const Table*>> cache;
};

TableState& tables_state() {
    static auto* s = new TableState();  // leaked, and Table objects are never freed
    return *s;
}

std::vector<const Table*> load_tables(const TableState& st, std::string_view op, std::string_view dtype,
                                      const std::string& dir) {
    std::map<std::string, const Table*> by_name;
    // Only <op>.<dtype>.<device>.txt names take part: a stray README.txt or a .bak.txt copy in
    // the override dir would otherwise throw from every choose() or compete with the real table.
    // The embed step rejects such names in tuned/ at configure time.
    auto consider = [&](const std::string& name, std::string_view text, bool is_override) {
        const auto f = split_file_name(name);
        if (f[0].empty()) {
            if (is_override && warned().insert("tuned_name:" + dir + "/" + name).second)
                std::fprintf(stderr, "batchlas: BATCHLAS_TUNED_DIR: skipping %s (not <op>.<dtype>.<device>.txt)\n",
                             name.c_str());
            return;
        }
        if (f[0] != op || f[1] != dtype) return;
        auto* t = new Table(parse_table(text, name));
        t->is_override = is_override;
        if (t->op == op && t->dtype == dtype) by_name[t->file] = t;
    };
    if (st.test_builtin) {
        for (const auto& [name, text] : *st.test_builtin) consider(name, text, false);
    } else {
        for (const auto& e : embedded_tables()) consider(std::string(e.name), e.text, false);
    }
    if (!dir.empty()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            if (warned().insert("tuned_dir:" + dir).second)
                std::fprintf(stderr, "batchlas: BATCHLAS_TUNED_DIR=%s is not a directory; ignoring it\n",
                             dir.c_str());
        } else {
            for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
                if (!ent.is_regular_file() || ent.path().extension() != ".txt") continue;
                std::ifstream in(ent.path());
                std::stringstream buf;
                buf << in.rdbuf();
                consider(ent.path().filename().string(), buf.str(), true);
            }
        }
    }
    std::vector<const Table*> out;
    for (const auto& [name, t] : by_name) out.push_back(t);
    return out;
}

int family_rank(std::string_view f) {
    if (f == "sm") return 0;
    if (f == "gfx") return 1;
    if (f == "intel") return 2;
    if (f == "cpu") return 4;
    return 3;
}

struct Decision {
    std::string op, spelling, detail, tag;
};

thread_local std::vector<std::pair<std::string, std::string>> t_pins;
thread_local std::vector<Decision> t_decisions;
thread_local int t_depth = 0;

std::string indent() { return std::string(static_cast<std::size_t>(2 * t_depth), ' '); }

std::string format_ms(double ms) {
    char buf[32];
    std::snprintf(buf, sizeof buf, ms < 1.0 ? "%.3g" : "%.2f", ms);
    return buf;
}

}  // namespace

Device device_from_key(std::string_view key) {
    Device d;
    d.key = std::string(key);
    d.family = d.key;
    if (key.rfind("sm_", 0) == 0) {
        d.family = "sm";
        d.arch_number = leading_int(key.substr(3));
    } else if (key.rfind("gfx", 0) == 0) {
        d.family = "gfx";
        d.arch_number = leading_int(key.substr(3));
    } else if (key == "rocm") {
        d.family = "gfx";  // Device has no gcnArchName query yet; borrows like an unknown gfx
    } else if (key.rfind("intel", 0) == 0) {
        d.family = "intel";
        if (key.size() > 6) d.arch_number = leading_int(key.substr(6));
    }
    return d;
}

const Device& describe(const batchlas::Device& dev, Backend b, bool has_vendor) {
    static auto* memo = new std::map<std::tuple<int, std::size_t, int, bool>, Device>();
    const auto k = std::make_tuple(static_cast<int>(dev.type), dev.idx, static_cast<int>(b), has_vendor);
    {
        std::lock_guard<std::mutex> lock(state_mutex());
        if (auto it = memo->find(k); it != memo->end()) return it->second;
    }
    const bool is_gpu = dev.type == DeviceType::GPU;
    std::string key = "gpu";
    if (!is_gpu) key = "cpu";
    else if (b == Backend::CUDA && dev.cuda_compute_capability() > 0)
        key = "sm_" + std::to_string(dev.cuda_compute_capability());
    else if (b == Backend::ROCM) key = "rocm";
    else if (dev.get_vendor() == Vendor::INTEL) key = "intel";
    Device d = device_from_key(key);
    d.is_gpu = is_gpu;
    d.has_sg32 = dev.supports_sub_group_size(32);
    d.has_vendor = has_vendor;
    d.slm_budget = static_cast<std::int64_t>(resident::device_slm_budget(
        static_cast<std::size_t>(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE))));
    d.max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    std::lock_guard<std::mutex> lock(state_mutex());
    return memo->emplace(k, std::move(d)).first->second;
}

Table parse_table(std::string_view text, std::string_view file) {
    Table t;
    t.file = std::string(file);
    if (const auto slash = t.file.find_last_of('/'); slash != std::string::npos) t.file = t.file.substr(slash + 1);
    std::map<std::string, std::string> header;
    std::map<std::string, int> header_line;
    std::map<std::vector<std::string>, int> seen_rows;
    bool have_keys = false;
    int line_no = 0;
    auto fail = [&](const std::string& why) {
        throw std::runtime_error(t.file + ":" + std::to_string(line_no) + ": " + why);
    };
    for (std::string_view raw : detail::split(text, '\n')) {
        ++line_no;
        std::string_view line = trim(raw);
        if (line.empty()) continue;
        if (line.front() == '#') {
            if (!t.rows.empty()) continue;
            const std::string_view body = trim(line.substr(1));
            if (body.rfind("keys:", 0) == 0) {
                if (have_keys) fail("second '# keys:' line");
                for (std::string_view w : words(body.substr(5))) {
                    const auto part = detail::split(w, ':');
                    const std::string_view kind = part.size() > 1 ? part[1] : "";
                    if (kind != "exact" && kind != "log") fail("key '" + std::string(w) + "' needs :exact or :log");
                    if (part.size() > (kind == "log" ? 3u : 2u))
                        fail("key '" + std::string(w) + "': only a :log key takes a weight");
                    double weight = 1.0;
                    if (part.size() == 3 && !parse_weight(part[2], weight))
                        fail("key '" + std::string(w) + "': weight must be a positive number like 3 or 0.5");
                    for (const auto& k : t.keys)
                        if (k.name == part[0]) fail("key '" + k.name + "' declared twice");
                    t.keys.push_back({std::string(part[0]), kind == "log", weight});
                }
                if (t.keys.empty()) fail("'# keys:' names no keys");
                have_keys = true;
                continue;
            }
            for (std::string_view w : words(body))
                if (const auto eq = w.find('='); eq != std::string_view::npos) {
                    header[std::string(w.substr(0, eq))] = std::string(w.substr(eq + 1));
                    header_line[std::string(w.substr(0, eq))] = line_no;
                }
            continue;
        }
        if (!have_keys) fail("row before the '# keys:' line");
        if (const auto hash = line.find('#'); hash != std::string_view::npos) line = trim(line.substr(0, hash));
        const auto segs = detail::split(line, '|');
        TableRow row;
        row.line = line_no;
        row.keys.resize(t.keys.size());
        row.log2_keys.assign(t.keys.size(), 0.0);
        std::vector<bool> got(t.keys.size(), false);
        for (std::string_view w : words(segs[0])) {
            const auto eq = w.find('=');
            const std::string name(w.substr(0, eq));
            std::size_t i = 0;
            while (i < t.keys.size() && t.keys[i].name != name) ++i;
            if (eq == std::string_view::npos || i == t.keys.size()) fail("unknown key '" + std::string(w) + "'");
            if (got[i]) fail("key '" + name + "' given twice");
            got[i] = true;
            row.keys[i] = std::string(w.substr(eq + 1));
            if (t.keys[i].log) {
                std::int64_t v = 0;
                const auto& s = row.keys[i];
                const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
                if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || v <= 0)
                    fail("log key '" + name + "' must be a positive integer, got '" + s + "'");
                row.log2_keys[i] = std::log2(static_cast<double>(v));
            }
        }
        for (std::size_t i = 0; i < got.size(); ++i)
            if (!got[i]) fail("row lacks key '" + t.keys[i].name + "'");
        std::size_t untimed = 0;
        for (std::size_t s = 1; s < segs.size(); ++s) {
            const auto w = words(segs[s]);
            if (w.size() != 2)
                fail("ranked entry '" + std::string(trim(segs[s])) + "' is not '<choice> <ms>' or '<choice> -'");
            double ms = 0.0;  // from_chars: strtod obeys a host app's comma-decimal LC_NUMERIC
            const auto r = std::from_chars(w[1].data(), w[1].data() + w[1].size(), ms);
            if (w[1] == "-") ++untimed;
            else if (r.ec != std::errc{} || r.ptr != w[1].data() + w[1].size() || !std::isfinite(ms) || ms < 0)
                fail("time '" + std::string(w[1]) + "' is not a non-negative number or '-'");
            for (const auto& e : row.ranked)
                if (e.spelling == w[0]) fail("'" + e.spelling + "' ranked twice");
            row.ranked.push_back({std::string(w[0]), ms});
        }
        if (row.ranked.empty()) fail("row has no ranked entries");
        if (untimed != 0 && untimed != row.ranked.size()) fail("row mixes timed and untimed ('-') entries");
        row.timed = untimed == 0;
        if (const auto [it, fresh] = seen_rows.emplace(row.keys, line_no); !fresh)
            fail("duplicate row (first at line " + std::to_string(it->second) + ")");
        t.rows.push_back(std::move(row));
    }
    const auto from_name = split_file_name(t.file);
    const char* names[3] = {"op", "dtype", "device"};
    std::string* fields[3] = {&t.op, &t.dtype, &t.device};
    line_no = 1;
    for (int i = 0; i < 3; ++i) {
        const auto h = header.find(names[i]);
        if (h != header.end() && !from_name[i].empty() && h->second != from_name[i])
            fail(std::string(names[i]) + "=" + h->second + " disagrees with the file name");
        *fields[i] = h != header.end() ? h->second : from_name[i];
        if (fields[i]->empty()) fail(std::string("no ") + names[i] + " in the header or the file name");
    }
    if (const auto h = header.find("source"); h != header.end()) t.source = h->second;
    // Each row is all-timed or all-untimed (checked above); a table may hold both kinds, but
    // an untimed row needs a transcribed:<hex sha> source. sweep_to_table.py --check agrees.
    const std::string_view tr = "transcribed:";
    if (t.source.rfind(tr, 0) == 0) {
        const std::string sha = t.source.substr(tr.size());
        line_no = header_line["source"];
        if (sha.empty() || !std::all_of(sha.begin(), sha.end(), [](char c) { return std::isxdigit(
                static_cast<unsigned char>(c)); }))
            fail("source=" + t.source + ": the commit after 'transcribed:' must be a hex sha");
    } else if (const auto u = std::find_if(t.rows.begin(), t.rows.end(), [](const TableRow& r) { return !r.timed; });
               u != t.rows.end()) {
        line_no = u->line;
        fail("untimed ('-') row needs a 'source=transcribed:<sha>' header");
    }
    const Device d = device_from_key(t.device);
    t.family = d.family;
    t.arch_number = d.arch_number;
    t.memo = std::make_shared<Table::Memo>();
    return t;
}

namespace {
constexpr std::size_t kNoRow = static_cast<std::size_t>(-1);
constexpr std::size_t kMemoCap = 1 << 16;  // distinct shapes per table before the memo resets
}  // namespace

const TableRow* Table::nearest(const Key& key) const {
    std::vector<std::string> kv(keys.size());
    std::vector<double> kl(keys.size(), 0.0);
    std::string memo_key;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const auto it = std::find_if(key.begin(), key.end(), [&](const KeyField& f) { return f.name == keys[i].name; });
        if (it == key.end()) throw std::invalid_argument(file + ": key '" + keys[i].name + "' not supplied by the op");
        kv[i] = it->value;
        memo_key += it->value;
        memo_key += '\x1f';
        if (keys[i].log) kl[i] = std::log2(std::max(1.0, std::strtod(it->value.c_str(), nullptr)));
    }
    if (memo) {
        std::lock_guard<std::mutex> lock(memo->mu);
        if (const auto hit = memo->row.find(memo_key); hit != memo->row.end())
            return hit->second < rows.size() ? &rows[hit->second] : nullptr;
    }
    const TableRow* found = nearest_scan(kv, kl);
    if (memo) {
        std::lock_guard<std::mutex> lock(memo->mu);
        if (memo->row.size() >= kMemoCap) memo->row.clear();
        memo->row.emplace(std::move(memo_key), found ? static_cast<std::size_t>(found - rows.data()) : kNoRow);
    }
    return found;
}

const TableRow* Table::nearest_scan(const std::vector<std::string>& kv, const std::vector<double>& kl) const {
    std::vector<std::size_t> exact;
    for (std::size_t i = 0; i < keys.size(); ++i)
        if (!keys[i].log) exact.push_back(i);
    std::size_t prefix = exact.size();
    auto matches = [&](const TableRow& r) {
        for (std::size_t j = 0; j < prefix; ++j)
            if (r.keys[exact[j]] != kv[exact[j]]) return false;
        return true;
    };
    while (prefix > 0 && std::none_of(rows.begin(), rows.end(), matches)) --prefix;
    const TableRow* best = nullptr;
    double best_d = 0.0;
    for (const auto& r : rows) {
        if (!matches(r)) continue;
        double dist = 0.0;
        for (std::size_t i = 0; i < keys.size(); ++i)
            if (keys[i].log) dist += keys[i].weight * std::fabs(r.log2_keys[i] - kl[i]);
        const bool tie = best && std::fabs(dist - best_d) <= 1e-9;
        if (!best || (!tie && dist < best_d) || (tie && r.log2_keys < best->log2_keys)) {
            best = &r;
            best_d = dist;
        }
    }
    return best;
}

std::vector<const Table*> tables_in_borrow_order(std::string_view op, std::string_view dtype, const Device& d) {
    const EnvValue& dir_env = settings().selection.tuned_dir;
    const std::string dir = dir_env.is_set() ? dir_env.value() : std::string();
    std::vector<const Table*> all;
    {
        std::lock_guard<std::mutex> lock(state_mutex());
        auto& st = tables_state();
        const std::string ck = std::string(op) + "|" + std::string(dtype) + "|" + dir + "|" +
                               std::to_string(st.generation);
        auto it = st.cache.find(ck);
        if (it == st.cache.end()) it = st.cache.emplace(ck, load_tables(st, op, dtype, dir)).first;
        all = it->second;
    }
    auto rank = [&](const Table* t) {
        if (t->device == d.key) return std::make_tuple(0, 0, 0);
        if (t->family == d.family) {
            if (t->arch_number <= d.arch_number) return std::make_tuple(1, 0, d.arch_number - t->arch_number);
            return std::make_tuple(1, 1, t->arch_number - d.arch_number);
        }
        return std::make_tuple(2 + family_rank(t->family), 0, -t->arch_number);
    };
    // §3/§7.6: the CPU never borrows a GPU table; without its own it reaches the last resort.
    if (d.family == "cpu")
        std::erase_if(all, [&](const Table* t) { return t->device != d.key; });
    std::stable_sort(all.begin(), all.end(), [&](const Table* a, const Table* b) { return rank(a) < rank(b); });
    return all;
}

namespace detail {

bool trace_enabled() noexcept { return settings().diagnostics.select_trace; }

void note_decision(std::string_view op, std::string spelling, std::string detail, std::string tag) {
    Decision dec{std::string(op), std::move(spelling), std::move(detail), std::move(tag)};
    for (auto& d : t_decisions)
        if (d.op == op) {
            d = std::move(dec);
            return;
        }
    t_decisions.push_back(std::move(dec));
}

void note_borrow(std::string_view op, std::string_view dtype, const Device& d, const Table& t) {
    if (!warn_once("borrow:" + std::string(op))) return;
    std::fprintf(stderr, "batchlas: %.*s has no %.*s table for %s; borrowing %s (run tools/tune to tune this device)\n",
                 static_cast<int>(op.size()), op.data(), static_cast<int>(dtype.size()), dtype.data(),
                 d.key.c_str(), t.device.c_str());
}

void warn_pin_fallback(std::string_view op, std::string_view word, const Device& d) {
    if (!warn_once(std::string(word) + ":" + std::string(op))) return;
    const int ol = static_cast<int>(op.size()), wl = static_cast<int>(word.size());
    std::fprintf(stderr, "batchlas: %.*s pinned \"%.*s\", but no %.*s candidate can run this shape on %s; "
                 "using the automatic choice\n", ol, op.data(), wl, word.data(), wl, word.data(), d.key.c_str());
}

std::string table_tag(const Table& t, const Device& d, bool own_table_exists) {
    const std::string pre = t.is_override ? "override " : "";
    if (t.device == d.key) return pre + t.device;
    return pre + t.device + (t.is_override ? "" : " table") +
           (own_table_exists ? ", fallthrough from " : ", borrowed for ") + d.key;
}

std::string format_detail(double ms, const std::string* next, double next_ms) {
    std::string s = format_ms(ms) + " ms";
    if (!next) return s;
    const double rel = ms > 0 ? std::fabs(next_ms - ms) / ms : (next_ms == ms ? 0.0 : 1.0);
    if (rel <= 0.03)
        return s + ", tied with " + *next + " " + format_ms(next_ms) + " ms (" +
               std::to_string(static_cast<int>(std::lround(rel * 100))) + "%)";
    return s + ", next " + *next + " " + format_ms(next_ms) + " ms";
}

std::string entry_detail(const TableRow& row, std::size_t i, const TableEntry* next) {
    if (!row.timed) return "transcribed";
    return format_detail(row.ranked[i].ms, next ? &next->spelling : nullptr, next ? next->ms : 0.0);
}

std::optional<std::string> pin_text(std::string_view op, std::string* source) {
    std::string text;
    if (auto it = std::find_if(t_pins.rbegin(), t_pins.rend(), [&](const auto& p) { return p.first == op; });
        it != t_pins.rend()) {
        text = it->second;
        *source = "ScopedPin";
    } else if (RoutingSettings::index_of(op) < RoutingSettings::ops.size()) {
        const char* raw = settings().routing.route(op).get();
        *source = "BATCHLAS_";
        for (const char c : op) *source += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        *source += "_ROUTE";
        if (!raw) return std::nullopt;
        text = raw;
    } else {
        return std::nullopt;
    }
    text = std::string(trim(text));
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (text.empty()) return std::nullopt;
    return text;
}

void push_pin(std::string_view op, std::string text) { t_pins.emplace_back(std::string(op), std::move(text)); }

void pop_pin(std::string_view op) {
    const auto it = std::find_if(t_pins.rbegin(), t_pins.rend(), [&](const auto& p) { return p.first == op; });
    if (it != t_pins.rend()) t_pins.erase(std::next(it).base());
}

bool trace_open(std::string_view op, const std::string& spelling, bool vendor, const coverage::Shape& shape,
                const NativeFacts& facts, const Key& fields) {
    if (coverage::dynamic_enabled()) {
        if (const auto o = op_from_name(op)) {
            coverage::Shape s = shape;
            s.op = *o;
            coverage::record_choice(*o, s.scalar, s.backend, s, vendor ? "vendor" : "native",
                                    spelling.c_str(), facts.existed, facts.supported);
        }
    }
    if (!trace_enabled()) return false;
    const auto dec = std::find_if(t_decisions.begin(), t_decisions.end(),
                                  [&](const Decision& d) { return d.op == op && d.spelling == spelling; });
    std::string line = indent() + std::string(op) + " " + std::string(dtype_from_scalar(shape.scalar));
    if (fields.empty()) line += " n=" + std::to_string(shape.n) + " batch=" + std::to_string(shape.batch);
    for (const KeyField& f : fields) line += " " + f.name + "=" + f.value;
    line += " -> " + spelling;
    if (dec != t_decisions.end()) {
        if (!dec->detail.empty()) line += "  " + dec->detail;
        line += "  [" + dec->tag + "]";
        t_decisions.erase(dec);
    } else {
        line += "  [untraced]";
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    ++t_depth;
    return true;
}

void trace_close() {
    if (t_depth > 0) --t_depth;
}

}  // namespace detail

namespace testing {

void set_builtin_tables(std::vector<std::pair<std::string, std::string>> files) {
    std::lock_guard<std::mutex> lock(state_mutex());
    tables_state().test_builtin = std::move(files);
    ++tables_state().generation;
}

void use_embedded_tables() {
    std::lock_guard<std::mutex> lock(state_mutex());
    tables_state().test_builtin.reset();
    ++tables_state().generation;
}

void reset_warnings() {
    std::lock_guard<std::mutex> lock(state_mutex());
    warned().clear();
}

}  // namespace testing

}  // namespace batchlas::select
