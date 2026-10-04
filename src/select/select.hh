#pragma once

// Flat kernel selection: one choose() per op over measured per-device tables.
// Spec: docs/design/flat-kernel-selection.md §4-§5. Everything here is generic over a
// choice std::variant whose alternatives are family structs (§4.2). State that a test and
// the library must share (pins, table cache, trace depth) lives in select.cc behind
// BATCHLAS_API: a header-local static would be one copy per DSO under -fvisibility=hidden.

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/export.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <algorithm>
#include <array>
#include <charconv>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace batchlas::select {

// ---- device facts (§5.1) ---------------------------------------------------------------

struct Device {
    std::string key;     // table key: "sm_120", "gfx90a", "intel", "cpu"
    std::string family;  // "sm", "gfx", "intel", "cpu": borrowing stays in a family first
    int arch_number = 0; // 120, 89, 90 (gfx90a), 0 for cpu
    bool is_gpu = false;
    bool has_sg32 = false;
    bool has_vendor_solver = false;  // potrf/syev library (solver_vendor_available)
    bool has_vendor_blas = false;    // level-3 library: gemm/trsm/... (level3_vendor_available)
    std::int64_t slm_budget = 0;
    int max_wg = 0;
};

// Key, family and arch only; the capability fields stay default. Tables and tests use it.
BATCHLAS_API Device device_from_key(std::string_view key);

// Memoized per (device, backend, has_vendor_solver, has_vendor_blas).
BATCHLAS_API const Device& describe(const batchlas::Device& dev, Backend b, bool has_vendor_solver,
                                    bool has_vendor_blas);

template <Backend B>
const Device& device_of(const Queue& q, bool has_vendor_solver = dispatch::solver_vendor_available<B>,
                        bool has_vendor_blas = dispatch::level3_vendor_available<B>) {
    return describe(q.device(), B, has_vendor_solver, has_vendor_blas);
}

template <class T>
constexpr std::string_view dtype_name() {
    if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "cfloat";
    else {
        static_assert(std::is_same_v<T, std::complex<double>>, "unsupported scalar");
        return "cdouble";
    }
}

// ---- families and spelling (§4.2, §5.2) -----------------------------------------------

template <std::size_t N>
struct FamilyName {
    char s[N]{};
    constexpr FamilyName(const char (&a)[N]) { std::copy_n(a, N, s); }
};
// `struct Cta : NoFields<"cta"> {};` -- generic code builds field-less families as A{}.
template <FamilyName L>
struct NoFields {
    static constexpr std::string_view name{L.s, sizeof(L.s) - 1};
    static constexpr std::array<std::string_view, 0> fields{};
    std::array<int, 0> values() const { return {}; }
    bool operator==(const NoFields&) const = default;
};

namespace detail {

template <class Choice, class F>
void for_each_family(F&& f) {
    [&]<std::size_t... I>(std::index_sequence<I...>) {
        (f(std::type_identity<std::variant_alternative_t<I, Choice>>{}), ...);
    }(std::make_index_sequence<std::variant_size_v<Choice>>{});
}

inline std::vector<std::string_view> split(std::string_view s, char sep) {
    std::vector<std::string_view> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

inline bool parse_int(std::string_view t, int& out) {
    if (t.empty()) return false;
    const auto r = std::from_chars(t.data(), t.data() + t.size(), out);
    return r.ec == std::errc{} && r.ptr == t.data() + t.size();
}

}  // namespace detail

template <class Choice>
std::string_view family_of(const Choice& c) {
    return std::visit([](const auto& a) { return std::decay_t<decltype(a)>::name; }, c);
}

// Output spelling, always the long form: family[:field=value]...
template <class Choice>
std::string to_string(const Choice& c) {
    return std::visit([](const auto& a) {
        using A = std::decay_t<decltype(a)>;
        std::string s(A::name);
        const auto v = a.values();
        for (std::size_t i = 0; i < v.size(); ++i) {
            s += ':';
            s += A::fields[i];
            s += '=';
            s += std::to_string(v[i]);
        }
        return s;
    }, c);
}

// Accepts the long form and positional shorthand ("lpanel:8"); exact case, no whitespace.
template <class Choice>
std::optional<Choice> parse(std::string_view text, std::string* err = nullptr) {
    const auto tok = detail::split(text, ':');
    std::optional<Choice> out;
    std::string why = "unknown family '" + std::string(tok[0]) + "'";
    detail::for_each_family<Choice>([&](auto id) {
        using A = typename decltype(id)::type;
        if (out || tok[0] != A::name) return;
        constexpr std::size_t K = A::fields.size();
        if (tok.size() - 1 != K) {
            why = std::string(A::name) + " takes " + std::to_string(K) + " field(s), got " +
                  std::to_string(tok.size() - 1);
            return;
        }
        std::array<int, K> vals{};
        std::array<bool, K> seen{};
        for (std::size_t i = 1; i < tok.size(); ++i) {
            std::string_view t = tok[i];
            std::size_t idx = i - 1;
            if (const auto eq = t.find('='); eq != std::string_view::npos) {
                const auto f = std::find(A::fields.begin(), A::fields.end(), t.substr(0, eq));
                if (f == A::fields.end()) {
                    why = "unknown field '" + std::string(t.substr(0, eq)) + "' of " + std::string(A::name);
                    return;
                }
                idx = static_cast<std::size_t>(f - A::fields.begin());
                t = t.substr(eq + 1);
            }
            if (seen[idx]) {
                why = "field '" + std::string(A::fields[idx]) + "' given twice";
                return;
            }
            if (!detail::parse_int(t, vals[idx])) {
                why = "'" + std::string(t) + "' is not an integer";
                return;
            }
            seen[idx] = true;
        }
        if constexpr (K == 0) out = Choice{A{}};
        else out = Choice{A::from(vals)};
    });
    if (!out && err) *err = "'" + std::string(text) + "': " + why;
    return out;
}

// ---- op-supplied rules ------------------------------------------------------------------

struct Alias {
    std::string_view name;      // e.g. "native:tiny"
    std::string_view spelling;  // e.g. "tiny"
};

struct Rules {
    std::span<const Alias> aliases{};
    // Families in generality order (§5.5); the rest follow in candidate-list order.
    std::span<const std::string_view> last_resort{};
};

// ---- keys and tables (§5.4) -------------------------------------------------------------

struct KeyField {
    std::string name;
    std::string value;
    KeyField(std::string_view n, std::string_view v) : name(n), value(v) {}
    KeyField(std::string_view n, std::int64_t v) : name(n), value(std::to_string(v)) {}
};
using Key = std::vector<KeyField>;

struct TableKey {
    std::string name;
    bool log = false;     // false: ":exact", compared as text
    double weight = 1.0;  // ":log:<w>": this key's share of the distance
};

struct TableEntry {
    std::string spelling;
    double ms = 0.0;
};

struct TableRow {
    std::vector<std::string> keys;  // in Table::keys order
    std::vector<double> log2_keys;  // log2 of each :log key, 0 for :exact
    std::vector<TableEntry> ranked;
    int line = 0;
    // False for a transcribed row ("<spelling> -" entries, every ms 0): ranked, never timed.
    // Untimed rows need a source=transcribed:<sha> header; such a table may also hold timed rows.
    bool timed = true;
};

struct Table {
    std::string file;  // basename, e.g. "potrf.float.sm_120.txt"
    std::string op, dtype, device, family;
    int arch_number = 0;
    std::string source;  // the header's source= value, e.g. "transcribed:2b46acab"
    bool is_override = false;
    std::vector<TableKey> keys;
    std::vector<TableRow> rows;

    // Rows matching the longest prefix of the :exact keys (in '# keys:' order; exact keys are
    // dropped from the right until some row matches, possibly all of them), then min
    // sum w*|log2(row/key)|, ties lexicographic by the :log keys in declared order.
    // Throws if `key` lacks a table key.
    BATCHLAS_API const TableRow* nearest(const Key& key) const;
};

// Throws std::runtime_error("<file>:<line>: <reason>").
BATCHLAS_API Table parse_table(std::string_view text, std::string_view file);

struct EmbeddedTable {
    std::string_view name;
    std::string_view text;
};
BATCHLAS_API std::span<const EmbeddedTable> embedded_tables();  // generated from tuned/*.txt

// §5.5: own key; same family at-or-below by nearest arch, then above; then sm, gfx, intel,
// other, cpu. Built-in tables plus BATCHLAS_TUNED_DIR, parsed once per (op, dtype, dir).
BATCHLAS_API std::vector<const Table*> tables_in_borrow_order(std::string_view op, std::string_view dtype,
                                                              const Device& d);

namespace detail {

BATCHLAS_API bool trace_enabled() noexcept;
BATCHLAS_API void note_decision(std::string_view op, std::string spelling, std::string detail, std::string tag);
BATCHLAS_API void note_borrow(std::string_view op, std::string_view dtype, const Device& d, const Table& t);
// A class-word pin ("native", "vendor") that nothing in its class can serve: once per op and word.
BATCHLAS_API void warn_pin_fallback(std::string_view op, std::string_view word, const Device& d);
// `own_table_exists`: the device has a table but its row had nothing runnable ("fallthrough"),
// as opposed to an untuned device ("borrowed").
BATCHLAS_API std::string table_tag(const Table& t, const Device& d, bool own_table_exists);
BATCHLAS_API std::string format_detail(double ms, const std::string* next, double next_ms);
// The trace detail for a ranked entry: "transcribed" on an untimed row, else format_detail.
BATCHLAS_API std::string entry_detail(const TableRow& row, std::size_t i, const TableEntry* next);
BATCHLAS_API std::optional<std::string> pin_text(std::string_view op, std::string* source);
BATCHLAS_API void push_pin(std::string_view op, std::string text);
BATCHLAS_API void pop_pin(std::string_view op);
struct NativeFacts;
BATCHLAS_API bool trace_open(std::string_view op, const std::string& spelling, bool vendor,
                             const dispatch::OpShape& shape, const NativeFacts& facts);
BATCHLAS_API void trace_close();

// A bad spelling in a table is a build defect, so it fails loudly on first use.
template <class Choice>
void validate(const Table& t) {
    static std::mutex mu;
    static std::set<const Table*> done;  // tables are never freed, so pointers stay unique
    std::lock_guard<std::mutex> lock(mu);
    if (done.count(&t)) return;
    for (const auto& row : t.rows)
        for (const auto& e : row.ranked) {
            std::string err;
            if (!parse<Choice>(e.spelling, &err))
                throw std::runtime_error(t.file + ":" + std::to_string(row.line) + ": " + err);
        }
    done.insert(&t);
}

template <class Choice, std::size_t N>
bool is_candidate(const Choice& c, const std::array<Choice, N>& candidates) {
    return std::find(candidates.begin(), candidates.end(), c) != candidates.end();
}

// The ranked walk of §5.4, then the last resort. nullopt only when native_only.
template <class Choice, std::size_t N, class CanRun>
std::optional<Choice> walk(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
                           const std::array<Choice, N>& candidates, CanRun& can_run, const Rules& rules,
                           bool native_only) {
    auto eligible = [&](const Choice& c) {
        return is_candidate(c, candidates) && !(native_only && family_of(c) == "vendor") && can_run(c);
    };
    const char* pin_note = native_only ? ", pinned native" : "";
    const auto tables = tables_in_borrow_order(op, dtype, d);
    const bool own = std::any_of(tables.begin(), tables.end(), [&](const Table* t) { return t->device == d.key; });
    for (const Table* t : tables) {
        validate<Choice>(*t);
        const TableRow* row = t->nearest(key);
        if (!row) continue;
        for (std::size_t i = 0; i < row->ranked.size(); ++i) {
            const auto c = parse<Choice>(row->ranked[i].spelling);
            if (!eligible(*c)) continue;
            // A tuned device whose row has nothing runnable is expected (Upper on Lower-only
            // rows, vendor-free builds); only an untuned device gets the R8 warning.
            if (t->device != d.key && !own) note_borrow(op, dtype, d, *t);
            if (trace_enabled()) {
                const TableEntry* next = nullptr;
                for (std::size_t j = i + 1; j < row->ranked.size() && !next; ++j)
                    if (eligible(*parse<Choice>(row->ranked[j].spelling))) next = &row->ranked[j];
                note_decision(op, to_string(*c), entry_detail(*row, i, next), table_tag(*t, d, own) + pin_note);
            }
            return c;
        }
    }
    std::vector<Choice> order;
    for (std::string_view fam : rules.last_resort)
        for (const Choice& c : candidates)
            if (family_of(c) == fam) order.push_back(c);
    for (const Choice& c : candidates)
        if (std::find(order.begin(), order.end(), c) == order.end()) order.push_back(c);
    for (const Choice& c : order) {
        if (!eligible(c)) continue;
        if (trace_enabled()) note_decision(op, to_string(c), "", std::string("last resort") + pin_note);
        return c;
    }
    if (native_only) return std::nullopt;
    throw std::runtime_error(std::string(op) + ": no runnable kernel on " + d.key);
}

// §5.3 / R6. nullopt means "auto": the caller runs the normal walk. The class words
// `native` and `vendor` fall back to auto with a warning; every other spelling throws.
template <class Choice, std::size_t N, class CanRun>
std::optional<Choice> resolve_pin(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
                                  const std::array<Choice, N>& candidates, CanRun& can_run, const Rules& rules,
                                  std::string text, const std::string& source) {
    const std::string where = std::string(op) + ": " + source + "=\"" + text + "\"";
    if (text == "auto") return std::nullopt;
    if (text == "native") {
        if (auto c = walk(op, dtype, d, key, candidates, can_run, rules, true)) return c;
        warn_pin_fallback(op, "native", d);
        return std::nullopt;
    }
    if (text == "vendor") {
        for (const Choice& k : candidates)
            if (family_of(k) == "vendor" && can_run(k)) {
                if (trace_enabled()) note_decision(op, to_string(k), "", "pinned");
                return k;
            }
        warn_pin_fallback(op, "vendor", d);  // a vendor-free build: the old router fell through too
        return std::nullopt;
    }
    for (const Alias& a : rules.aliases)
        if (text == a.name) text = std::string(a.spelling);
    std::string err;
    const std::optional<Choice> c = parse<Choice>(text, &err);
    if (!c) throw std::invalid_argument(where + " is not a valid choice: " + err);
    if (!is_candidate(*c, candidates))
        throw std::invalid_argument(where + ": " + to_string(*c) + " is not a compiled " + std::string(op) + " " +
                                    std::string(dtype) + " candidate");
    if (!can_run(*c))
        throw std::invalid_argument(where + ": " + to_string(*c) + " cannot run this shape on " + d.key);
    if (trace_enabled()) note_decision(op, to_string(*c), "", "pinned");
    return c;
}

}  // namespace detail

// ---- pins (§5.3) ------------------------------------------------------------------------

// Wins over BATCHLAS_<OP>_ROUTE on this thread; nests, restoring the outer pin on exit.
// Takes a choice, or any pin word ("auto", "native", "vendor", an alias, a spelling).
template <class Choice>
class ScopedPin {
public:
    ScopedPin(std::string_view op, const Choice& c) : op_(op) { detail::push_pin(op_, to_string(c)); }
    ScopedPin(std::string_view op, std::string_view word) : op_(op) { detail::push_pin(op_, std::string(word)); }
    ~ScopedPin() { detail::pop_pin(op_); }
    ScopedPin(const ScopedPin&) = delete;
    ScopedPin& operator=(const ScopedPin&) = delete;

private:
    std::string op_;
};

// ---- the selection algorithm (§5.4) -----------------------------------------------------

template <class Choice, std::size_t N, class CanRun>
Choice choose(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
              const std::array<Choice, N>& candidates, CanRun&& can_run, const Rules& rules = {}) {
    std::string source;
    if (auto text = detail::pin_text(op, &source))
        if (auto c = detail::resolve_pin(op, dtype, d, key, candidates, can_run, rules, *text, source))
            return *c;
    return *detail::walk(op, dtype, d, key, candidates, can_run, rules, false);
}

// ---- trace and coverage (§5.6) ----------------------------------------------------------

// The coverage key for a square op; the op sets uplo/side/... on the result.
template <Backend B, class T>
dispatch::OpShape square_shape(std::int64_t n, std::int64_t batch) {
    dispatch::OpShape s;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;
    s.m = s.n = s.k = n;
    s.batch = batch;
    return s;
}

// The coverage row's native_route_existed / native_route_supported (tri-state, -1 unknown).
namespace detail {
struct NativeFacts {
    bool existed = true;
    int supported = -1;
};
}  // namespace detail
using detail::NativeFacts;

// Over the op's candidates: any non-vendor compiled, and any of those passing can_run.
template <class Choice, std::size_t N, class CanRun>
NativeFacts native_facts(const std::array<Choice, N>& candidates, CanRun&& can_run) {
    NativeFacts f{false, 0};
    for (const Choice& c : candidates)
        if (family_of(c) != "vendor") {
            f.existed = true;
            if (can_run(c)) f.supported = 1;
        }
    return f;
}

// Prints the last choose() decision for `op` under BATCHLAS_SELECT_TRACE=1, indents nested
// scopes, and records the coverage row. The shape is required: with coverage on and trace
// off no decision is noted, so scalar, backend and uplo can come from nowhere else.
class TraceScope {
public:
    template <class Choice>
    TraceScope(std::string_view op, const Choice& c, const dispatch::OpShape& shape, NativeFacts facts = {}) {
        if (detail::trace_enabled() || dispatch::coverage::dynamic_enabled())
            active_ = detail::trace_open(op, to_string(c), family_of(c) == "vendor", shape, facts);
    }
    ~TraceScope() {
        if (active_) detail::trace_close();
    }
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;

private:
    bool active_ = false;
};

namespace testing {
// Replace the embedded tables with {file name, text} pairs; caches are dropped.
BATCHLAS_API void set_builtin_tables(std::vector<std::pair<std::string, std::string>> files);
BATCHLAS_API void use_embedded_tables();
BATCHLAS_API void reset_warnings();
}  // namespace testing

}  // namespace batchlas::select
