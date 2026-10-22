// SPDX-License-Identifier: MIT
#include <src/select/select.hh>

#include <src/util/internal-api.hh>
#include <src/util/resident_capacity.hh>

#include <batchlas/settings.hh>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <locale>
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

// Whole-string, locale-independent (strtod obeys a host app's comma-decimal LC_NUMERIC).
// Not std::from_chars: Apple libc++ headers mark the floating-point overload unavailable.
bool parse_double(std::string_view s, double& out) {
    if (s.empty() || s.front() == '+' || std::isspace(static_cast<unsigned char>(s.front()))) return false;
    std::istringstream in{std::string(s)};
    in.imbue(std::locale::classic());
    in >> std::noskipws >> out;
    return !in.fail() && in.peek() == std::char_traits<char>::eof();
}

// A plain positive decimal: digits with at most one '.', no sign, no exponent.
bool parse_weight(std::string_view s, double& out) {
    if (s.empty() || std::count(s.begin(), s.end(), '.') > 1) return false;
    if (!std::all_of(s.begin(), s.end(), [](char c) { return c == '.' || (c >= '0' && c <= '9'); })) return false;
    return parse_double(s, out) && std::isfinite(out) && out > 0;
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

// The runtime spelling of dtype_name<T>, for a trace line that has only the shape's ScalarKind.
std::string_view dtype_from_scalar(ScalarKind s) {
    switch (s) {
        case ScalarKind::F32: return dtype_name<float>();
        case ScalarKind::F64: return dtype_name<double>();
        case ScalarKind::C32: return dtype_name<std::complex<float>>();
        case ScalarKind::C64: return dtype_name<std::complex<double>>();
    }
    return "?";
}

// "<op>.<device>.txt" -> {op, "", device} (the dtype is a key column); empty strings when it does not fit.
std::array<std::string, 3> split_file_name(std::string_view file) {
    std::array<std::string, 3> out;
    const auto slash = file.find_last_of('/');
    if (slash != std::string_view::npos) file = file.substr(slash + 1);
    const auto parts = detail::split(file, '.');
    if (parts.size() == 3 && parts[2] == "txt" && !parts[0].empty() && !parts[1].empty()) {
        out[0] = std::string(parts[0]);
        out[2] = std::string(parts[1]);
    }
    return out;
}

std::mutex& state_mutex() {
    static auto* m = new std::mutex();  // leaked: choose() may run from static destructors
    return *m;
}

struct TableState {
    std::optional<std::vector<std::pair<std::string, std::string>>> test_builtin;
    int generation = 0;
    std::size_t parses = 0;  // parse_tables calls, for testing::parse_count
    std::map<std::string, std::vector<const Table*>> cache;  // op|dtype|dir|generation -> tables
    // file key|generation -> one Table per dtype: a multi-dtype file (gemm ~14 MB) is parsed once,
    // not once per dtype that asks for it, under the state mutex.
    std::map<std::string, std::vector<const Table*>> parsed;
};

TableState& tables_state() {
    static auto* s = new TableState();  // leaked, and Table objects are never freed
    return *s;
}

std::vector<const Table*> load_tables(TableState& st, std::string_view op, std::string_view dtype,
                                      const std::string& dir) {
    std::map<std::string, const Table*> by_name;
    // Only <op>.<device>.txt names take part: a stray README.txt or a .bak.txt copy in the
    // override dir would otherwise throw from every choose() or compete with the real table.
    // The embed step rejects such names in tuned/ at configure time.
    auto consider = [&](const std::string& name, auto&& read_text, bool is_override) {
        const auto f = split_file_name(name);
        if (f[0].empty()) {
            if (is_override)
                ::batchlas::detail::warn_once("tuned_name:" + dir + "/" + name,
                                              "BATCHLAS_TUNED_DIR: skipping " + name + " (not <op>.<device>.txt)");
            return;
        }
        if (f[0] != op) return;
        const std::string key = (is_override ? dir + "/" : std::string("builtin:")) + name + "|" +
                                std::to_string(st.generation);
        auto it = st.parsed.find(key);
        if (it == st.parsed.end()) {
            ++st.parses;
            std::vector<const Table*> parts;
            std::vector<Table> parsed;
            try {
                parsed = parse_tables(read_text(), name);
            } catch (const internal_error& e) {
                if (!is_override) throw;
                throw invalid_argument(std::string("BATCHLAS_TUNED_DIR: ") + e.what());
            }
            for (Table& part : parsed) {
                auto* t = new Table(std::move(part));
                t->is_override = is_override;
                parts.push_back(t);
            }
            it = st.parsed.emplace(key, std::move(parts)).first;
        }
        for (const Table* t : it->second)
            if (t->op == op && t->dtype == dtype) by_name[t->file] = t;
    };
    if (st.test_builtin) {
        for (const auto& [name, text] : *st.test_builtin)
            consider(name, [&] { return std::string_view(text); }, false);
    } else {
        for (const auto& e : embedded_tables()) consider(std::string(e.name), [&] { return e.text; }, false);
    }
    if (!dir.empty()) {
        std::error_code ec;
        if (!std::filesystem::is_directory(dir, ec)) {
            ::batchlas::detail::warn_once("tuned_dir:" + dir,
                                          "BATCHLAS_TUNED_DIR=" + dir + " is not a directory; ignoring it");
        } else {
            for (const auto& ent : std::filesystem::directory_iterator(dir, ec)) {
                if (!ent.is_regular_file() || ent.path().extension() != ".txt") continue;
                consider(ent.path().filename().string(), [&] {
                    std::ifstream in(ent.path());
                    std::stringstream buf;
                    buf << in.rdbuf();
                    return buf.str();
                }, true);
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
    if (f == "apple") return 3;
    if (f == "cpu") return 5;
    return 4;
}

struct Decision {
    std::string op, spelling, detail, tag;
};

struct PinEntry {
    std::string first, second;  // op, pin text
    bool strict = false;
};
thread_local std::vector<PinEntry> t_pins;
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
    } else if (key.rfind("apple", 0) == 0) {
        d.family = "apple";
        const auto digit = key.find_first_of("0123456789");
        if (digit != std::string_view::npos) d.arch_number = leading_int(key.substr(digit));
    }
    return d;
}

std::string apple_device_key(std::string_view device_name) {
    std::string key;
    for (const char c : device_name) {
        if (std::isalnum(static_cast<unsigned char>(c)))
            key += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!key.empty() && key.back() != '_')
            key += '_';
    }
    while (!key.empty() && key.back() == '_') key.pop_back();
    if (key.rfind("apple", 0) != 0) key = key.empty() ? "apple" : "apple_" + key;
    return key;
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
    // Only acpp's Metal backend exposes an Apple GPU; tables are per chip, named after it.
    const bool metal = is_gpu && dev.get_vendor() == Vendor::APPLE;
    if (metal) key = apple_device_key(dev.get_name());
    Device d = device_from_key(key);
    d.is_gpu = is_gpu;
    d.has_sg32 = dev.supports_sub_group_size(32);
    d.has_vendor = has_vendor;
    // acpp's host device reports SIZE_MAX ("unbounded"); a plain cast would make it negative.
    d.slm_budget = static_cast<std::int64_t>(std::min<std::size_t>(
        resident::device_slm_budget(static_cast<std::size_t>(dev.get_property(DeviceProperty::LOCAL_MEM_SIZE))),
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())));
    d.max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    d.scratch_budget = static_cast<std::int64_t>(std::min<std::size_t>(
        {static_cast<std::size_t>(dev.get_property(DeviceProperty::MAX_MEM_ALLOC_SIZE)),
         static_cast<std::size_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_SIZE)) / 4,
         static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())}));
    if (metal) {
        // impl::local_mem_scratch_per_item and impl::kMetalMaxGlobalRange (src/sycl/), which
        // this sycl-free TU cannot include; resident_capacity_tests holds the two to agree.
        d.slm_per_item = 8;
        d.max_groups_yz = 0xffffffffll;
    }
    std::lock_guard<std::mutex> lock(state_mutex());
    return memo->emplace(k, std::move(d)).first->second;
}

namespace {

// One table text. by_dtype: a tuned/<op>.<device>.txt file, whose first key is dtype:exact and
// whose header names no dtype (parse_tables splits it).
Table parse_one(std::string_view text, std::string_view file, bool by_dtype) {
    Table t;
    t.file = std::string(file);
    if (const auto slash = t.file.find_last_of('/'); slash != std::string::npos) t.file = t.file.substr(slash + 1);
    std::map<std::string, std::string> header;
    std::map<std::string, int> header_line;
    std::map<std::vector<std::string>, int> seen_rows;
    bool have_keys = false;
    int line_no = 0;
    auto fail = [&](const std::string& why) {
        throw internal_error(t.file + ":" + std::to_string(line_no) + ": " + why);
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
            double ms = 0.0;
            const bool parsed = parse_double(w[1], ms);
            if (w[1] == "-") ++untimed;
            else if (!parsed || !std::isfinite(ms) || ms < 0)
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
        if (by_dtype && i == 1) {
            if (h != header.end()) fail("dtype=" + h->second + " in the header: the dtype is a key column");
            continue;
        }
        if (h != header.end() && !from_name[i].empty() && h->second != from_name[i])
            fail(std::string(names[i]) + "=" + h->second + " disagrees with the file name");
        *fields[i] = h != header.end() ? h->second : from_name[i];
        if (fields[i]->empty()) fail(std::string("no ") + names[i] + " in the header or the file name");
    }
    if (by_dtype && (t.keys.empty() || t.keys[0].name != "dtype" || t.keys[0].log)) {
        line_no = 1;
        fail("the first key must be dtype:exact");
    }
    if (const auto h = header.find("source"); h != header.end()) t.source = h->second;
    // Each row is all-timed or all-untimed (checked above); a table may hold both kinds, but
    // an untimed row needs a transcribed:<hex sha> or ledger:<dir> source. sweep_to_table.py --check agrees.
    const std::string_view tr = "transcribed:";
    if (t.source.rfind(tr, 0) == 0) {
        const std::string sha = t.source.substr(tr.size());
        line_no = header_line["source"];
        if (sha.empty() || !std::all_of(sha.begin(), sha.end(), [](char c) { return std::isxdigit(
                static_cast<unsigned char>(c)); }))
            fail("source=" + t.source + ": the commit after 'transcribed:' must be a hex sha");
    } else if (t.source.rfind("ledger:", 0) == 0) {
        // A ledger table keeps transcribed rows beside measured ones: untimed rows are allowed.
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

}  // namespace

Table parse_table(std::string_view text, std::string_view file) { return parse_one(text, file, false); }

std::vector<Table> parse_tables(std::string_view text, std::string_view file) {
    Table all = parse_one(text, file, true);
    // The rows leave `all` first, so each dtype's Table copies only the header fields.
    std::vector<TableRow> rows = std::move(all.rows);
    all.rows.clear();
    std::vector<Table> out;
    for (TableRow& r : rows) {
        const std::string& dtype = r.keys[0];
        if (dtype != "float" && dtype != "double" && dtype != "cfloat" && dtype != "cdouble")
            throw internal_error(all.file + ":" + std::to_string(r.line) + ": unknown dtype '" + dtype + "'");
        auto t = std::find_if(out.begin(), out.end(), [&](const Table& x) { return x.dtype == dtype; });
        if (t == out.end()) {
            Table part = all;
            part.dtype = dtype;
            part.keys.erase(part.keys.begin());
            part.rows.clear();
            part.memo = std::make_shared<Table::Memo>();
            out.push_back(std::move(part));
            t = std::prev(out.end());
        }
        r.keys.erase(r.keys.begin());
        r.log2_keys.erase(r.log2_keys.begin());
        t->rows.push_back(std::move(r));
    }
    return out;
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
        if (it == key.end()) {
            const std::string what = file + ": key '" + keys[i].name + "' not supplied by the op";
            if (is_override) throw invalid_argument("BATCHLAS_TUNED_DIR: " + what);
            throw internal_error(what);
        }
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
    // §5.5: the CPU never borrows a GPU table; without its own it reaches the last resort.
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
    ::batchlas::detail::warn_once("borrow:" + std::string(op),
                                  std::string(op) + " has no " + std::string(dtype) + " table for " + d.key +
                                      "; borrowing " + t.device + " (run tools/tune to tune this device)");
}

void warn_pin_fallback(std::string_view op, std::string_view word, const Device& d) {
    const std::string w(word);
    ::batchlas::detail::warn_once(w + ":" + std::string(op), std::string(op) + " pinned \"" + w + "\", but no " + w +
                                                              " candidate can run this shape on " + d.key +
                                                              "; using the automatic choice");
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

void push_pin(std::string_view op, std::string text, bool strict) {
    t_pins.push_back({std::string(op), std::move(text), strict});
}

bool pin_strict(std::string_view op) {
    const auto it = std::find_if(t_pins.rbegin(), t_pins.rend(), [&](const auto& p) { return p.first == op; });
    return it != t_pins.rend() && it->strict;
}

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

void reset_warnings() { ::batchlas::detail::reset_warnings(); }

std::size_t parse_count() {
    std::lock_guard<std::mutex> lock(state_mutex());
    return tables_state().parses;
}

}  // namespace testing

}  // namespace batchlas::select
