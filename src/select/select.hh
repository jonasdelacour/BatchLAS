#pragma once

/// @file
/// @brief Flat kernel selection: one choose() per op over measured per-device tables.
/// Spec: docs/design/flat-kernel-selection.md §4-§5. Everything here is generic over a choice
/// std::variant whose alternatives are family structs (§4.2). @ingroup selection
// State that a test and the library must share (pins, table cache, trace depth) lives in select.cc
// behind BATCHLAS_API: a header-local static would be one copy per DSO under -fvisibility=hidden.

#include "coverage.hh"
#include "vendor.hh"
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
#include <memory>
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
/// @addtogroup selection
/// @{

// ---- device facts (§5.1) ---------------------------------------------------------------

/// The device facts a `can_run` predicate and the table lookup read (§5.1).
struct Device {
    std::string key;     ///< table key: "sm_120", "gfx90a", "intel", "cpu"
    std::string family;  ///< "sm", "gfx", "intel", "cpu": borrowing stays in a family first
    int arch_number = 0; ///< 120, 89, 90 (gfx90a), 0 for cpu
    bool is_gpu = false;
    bool has_sg32 = false;
    bool has_vendor = false;  ///< the asking op's library group (OpSpec::vendor) is compiled in
    std::int64_t slm_budget = 0;  ///< LOCAL_MEM_SIZE less a 4 KiB reserve, bytes
    int max_wg = 0;  ///< MAX_WORK_GROUP_SIZE
};

/// Key, family and arch only; the capability fields stay default. Tables and tests use it.
BATCHLAS_API Device device_from_key(std::string_view key);

/// The facts of @p dev for backend @p b; memoized per (device, backend, has_vendor).
BATCHLAS_API const Device& describe(const batchlas::Device& dev, Backend b, bool has_vendor);

/// The facts of @p q's device, with Device::has_vendor answering for the library group @p vendor.
template <Backend B>
const Device& device_of(const Queue& q, Lib vendor = Lib::none) {
    return describe(q.device(), B, has_library<B>(vendor));
}

/// The dtype token of a table file name: float, double, cfloat or cdouble.
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

/// A string literal as a template argument: the name of a NoFields family.
template <std::size_t N>
struct FamilyName {
    char s[N]{};
    constexpr FamilyName(const char (&a)[N]) { std::copy_n(a, N, s); }
};
/// Base of a family with no knobs: `struct Cta : NoFields<"cta"> {};`.
/// Generic code builds such families as `A{}`.
template <FamilyName L>
struct NoFields {
    static constexpr std::string_view name{L.s, sizeof(L.s) - 1};
    static constexpr std::array<std::string_view, 0> fields{};
    std::array<int, 0> values() const { return {}; }
    bool operator==(const NoFields&) const = default;
};

/// Every alternative of a field-less choice variant, in declaration (= tie-break) order.
template <class Choice>
constexpr auto all_of() {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::array<Choice, sizeof...(I)>{Choice{std::variant_alternative_t<I, Choice>{}}...};
    }(std::make_index_sequence<std::variant_size_v<Choice>>{});
}

/// std::visit over a choice with one lambda per family.
template <class... F>
struct overloaded : F... { using F::operator()...; };
template <class... F>
overloaded(F...) -> overloaded<F...>;

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

/// The family name of @p c, e.g. "lpanel" for `lpanel:panel=8`.
template <class Choice>
std::string_view family_of(const Choice& c) {
    return std::visit([](const auto& a) { return std::decay_t<decltype(a)>::name; }, c);
}

/// Output spelling, always the long form: family[:field=value]...
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

/// Accepts the long form and positional shorthand ("lpanel:8"); exact case, no whitespace.
/// @return the choice, or nullopt with the reason in `*err` (when non-null).
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

/// What an op adds to the generic walk: its last-resort order.
struct Rules {
    /// Families in generality order (§5.5); the rest follow in candidate-list order.
    std::span<const std::string_view> last_resort{};
};

// ---- keys and tables (§5.4) -------------------------------------------------------------

/// One field of a lookup key, value as text; a Key lists the fields an op's table keys name.
struct KeyField {
    std::string name;
    std::string value;
    KeyField(std::string_view n, std::string_view v) : name(n), value(v) {}
    KeyField(std::string_view n, std::int64_t v) : name(n), value(std::to_string(v)) {}
};
using Key = std::vector<KeyField>;

/// One entry of a table's `# keys:` line.
struct TableKey {
    std::string name;
    bool log = false;     ///< false: ":exact", compared as text
    double weight = 1.0;  ///< ":log:<w>": this key's share of the distance
};

/// One ranked candidate of a row: its spelling and milliseconds per call for the whole batch.
struct TableEntry {
    std::string spelling;
    double ms = 0.0;
};

/// One measured (or transcribed) shape and every candidate ranked there, fastest first.
struct TableRow {
    std::vector<std::string> keys;  ///< in Table::keys order
    std::vector<double> log2_keys;  ///< log2 of each :log key, 0 for :exact
    std::vector<TableEntry> ranked;
    int line = 0;
    /// False for a transcribed row ("<spelling> -" entries, every ms 0): ranked, never timed.
    /// Untimed rows need a `source=transcribed:<sha>` header; such a table may also hold timed rows.
    bool timed = true;
};

/// One parsed `tuned/<op>.<dtype>.<device>.txt` file.
struct Table {
    std::string file;  ///< basename, e.g. "potrf.float.sm_120.txt"
    std::string op, dtype, device, family;
    int arch_number = 0;
    std::string source;  ///< the header's source= value, e.g. "transcribed:2b46acab"
    bool is_override = false;  ///< read from BATCHLAS_TUNED_DIR
    std::vector<TableKey> keys;
    std::vector<TableRow> rows;
    // nearest() results by key text, as row indices (copies may share it); set by parse_table.
    // gemm tables hold thousands of rows and gemm is called per panel, so the scan is memoized.
    struct Memo;
    std::shared_ptr<Memo> memo;

    /// Rows matching the longest prefix of the :exact keys (in '# keys:' order; exact keys are
    /// dropped from the right until some row matches, possibly all of them), then min
    /// sum w*|log2(row/key)|, ties lexicographic by the :log keys in declared order.
    /// @throws std::invalid_argument if @p key lacks a table key.
    BATCHLAS_API const TableRow* nearest(const Key& key) const;
    const TableRow* nearest_scan(const std::vector<std::string>& kv, const std::vector<double>& kl) const;
};

/// Parses one table file. @throws std::runtime_error `"<file>:<line>: <reason>"`.
BATCHLAS_API Table parse_table(std::string_view text, std::string_view file);

/// A `tuned/*.txt` file compiled into the library: its basename and text.
struct EmbeddedTable {
    std::string_view name;
    std::string_view text;
};
BATCHLAS_API std::span<const EmbeddedTable> embedded_tables();  ///< generated from tuned/*.txt

/// §5.5: own key; same family at-or-below by nearest arch, then above; then sm, gfx, intel,
/// other, cpu. Built-in tables plus BATCHLAS_TUNED_DIR, parsed once per (op, dtype, dir).
/// A CPU device gets only its own table.
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
BATCHLAS_API void push_pin(std::string_view op, std::string text, bool strict = false);
BATCHLAS_API bool pin_strict(std::string_view op);  // the innermost ScopedPin of op is strict
BATCHLAS_API void pop_pin(std::string_view op);
struct NativeFacts;
// `fields` is what the trace line prints after the dtype; empty means the shape's n and batch.
BATCHLAS_API bool trace_open(std::string_view op, const std::string& spelling, bool vendor,
                             const coverage::Shape& shape, const NativeFacts& facts, const Key& fields);
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
// `strict` (the tuner's arms): a class word that cannot serve the shape throws too, and
// `vendor` must resolve to the candidate spelled `vendor`.
template <class Choice, std::size_t N, class CanRun>
std::optional<Choice> resolve_pin(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
                                  const std::array<Choice, N>& candidates, CanRun& can_run, const Rules& rules,
                                  const std::string& text, const std::string& source, bool strict = false) {
    const std::string where = std::string(op) + ": " + source + "=\"" + text + "\"";
    if (text == "auto") return std::nullopt;
    const std::string refused = where + " (strict): no " + text + " candidate can run this shape on " + d.key;
    if (text == "native") {
        if (auto c = walk(op, dtype, d, key, candidates, can_run, rules, true)) return c;
        if (strict) throw std::invalid_argument(refused);
        warn_pin_fallback(op, "native", d);
        return std::nullopt;
    }
    if (text == "vendor") {
        for (const Choice& k : candidates)
            if (family_of(k) == "vendor" && (!strict || to_string(k) == text) && can_run(k)) {
                if (trace_enabled()) note_decision(op, to_string(k), "", "pinned");
                return k;
            }
        if (strict) throw std::invalid_argument(refused);
        warn_pin_fallback(op, "vendor", d);  // a vendor-free build: the old router fell through too
        return std::nullopt;
    }
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

/// Tag for a strict ScopedPin: a class word ("native", "vendor") that cannot serve the shape
/// throws std::invalid_argument instead of falling back to auto.
struct StrictPin {};

/// RAII pin for @p op: wins over BATCHLAS_<OP>_ROUTE on this thread; nests, restoring the outer
/// pin on exit. Takes a choice, or any pin word ("auto", "native", "vendor", a spelling).
template <class Choice>
class ScopedPin {
public:
    ScopedPin(std::string_view op, std::string_view word, StrictPin) : op_(op) {
        detail::push_pin(op_, std::string(word), true);
    }
    ScopedPin(std::string_view op, const Choice& c) : op_(op) { detail::push_pin(op_, to_string(c)); }
    ScopedPin(std::string_view op, std::string_view word) : op_(op) { detail::push_pin(op_, std::string(word)); }
    ~ScopedPin() { detail::pop_pin(op_); }
    ScopedPin(const ScopedPin&) = delete;
    ScopedPin& operator=(const ScopedPin&) = delete;

private:
    std::string op_;
};

// ---- the selection algorithm (§5.4) -----------------------------------------------------

/// The only selection algorithm: the pin, else the first candidate passing @p can_run in the nearest
/// row of each table in borrow order, else the first runnable candidate in @p rules' last-resort order.
/// @throws std::invalid_argument for a bad pin (R6); std::runtime_error when nothing can run.
template <class Choice, std::size_t N, class CanRun>
Choice choose(std::string_view op, std::string_view dtype, const Device& d, const Key& key,
              const std::array<Choice, N>& candidates, CanRun&& can_run, const Rules& rules = {}) {
    std::string source;
    if (auto text = detail::pin_text(op, &source))
        if (auto c = detail::resolve_pin(op, dtype, d, key, candidates, can_run, rules, *text, source,
                                         detail::pin_strict(op)))
            return *c;
    return *detail::walk(op, dtype, d, key, candidates, can_run, rules, false);
}

// ---- trace and coverage (§5.6) ----------------------------------------------------------

namespace detail {
/// The coverage row's native_route_existed / native_route_supported (tri-state, -1 unknown).
struct NativeFacts {
    bool existed = true;
    int supported = -1;
};
}  // namespace detail
using detail::NativeFacts;

/// Over the op's candidates: any non-vendor compiled, and any of those passing can_run.
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

/// Prints the last choose() decision for `op` under BATCHLAS_SELECT_TRACE=1, indents nested
/// scopes, and records the coverage row. The shape is required: with coverage on and trace
/// off no decision is noted, so scalar, backend and uplo can come from nowhere else.
/// `fields` are the key fields the line shows (posv: n, nrhs, batch); empty prints n and batch.
class TraceScope {
public:
    template <class Choice>
    TraceScope(std::string_view op, const Choice& c, const coverage::Shape& shape, NativeFacts facts = {},
               const Key& fields = {}) {
        if (detail::trace_enabled() || coverage::dynamic_enabled())
            active_ = detail::trace_open(op, to_string(c), family_of(c) == "vendor", shape, facts, fields);
    }
    ~TraceScope() {
        if (active_) detail::trace_close();
    }
    TraceScope(const TraceScope&) = delete;
    TraceScope& operator=(const TraceScope&) = delete;

private:
    bool active_ = false;
};

// ---- an op's whole selection path (§4.3) -------------------------------------------------

inline constexpr std::array<std::string_view, 2> kBlockedThenVendor{"blocked", "vendor"};  ///< default last resort

/// What an op's choice.hh declares once: its name, the library its Vendor family calls (one
/// gate for can_run, the launch arm and the NoRouteError), and its last-resort order.
struct OpSpec {
    Op op;
    Lib vendor = Lib::none;  ///< Lib::none: the op has no vendor family
    Rules rules{kBlockedThenVendor};
    constexpr std::string_view name() const { return op_name(op); }  ///< table, pin and trace name
};

/// The Vendor arm of an op without its library: record the miss and throw NoRouteError.
template <Backend B, class T>
[[noreturn]] void no_vendor(const OpSpec& op) {
    throw_no_vendor_route<T>(op.op, B, library_name<B>(op.vendor));
}

/// choose(), with "nothing runnable" in a build without the op's library reported as
/// NoRouteError plus a coverage `miss` row (the vendor-free burn-down reads them).
/// can_run(choice, device) is the op's R3 predicate. `<op>_buffer_size` calls this (R5).
template <Backend B, class T, class Choice, std::size_t N, class CanRun>
Choice pick(const OpSpec& op, const Device& d, const Key& key, const std::array<Choice, N>& candidates,
            CanRun&& can_run) {
    auto ok = [&](const Choice& c) { return can_run(c, d); };
    try {
        return choose(op.name(), dtype_name<T>(), d, key, candidates, ok, op.rules);
    } catch (const std::runtime_error&) {
        if (op.vendor != Lib::none && !has_library<B>(op.vendor) &&
            std::none_of(candidates.begin(), candidates.end(), ok))
            no_vendor<B, T>(op);
        throw;
    }
}

template <Backend B, class T, class Choice, std::size_t N, class CanRun>
Choice pick(const OpSpec& op, const Queue& q, const Key& key, const std::array<Choice, N>& candidates,
            CanRun&& can_run) {
    return pick<B, T>(op, device_of<B>(q, op.vendor), key, candidates, can_run);
}

/// A public entry point after validation: pick, open the trace/coverage scope (the shape's
/// scalar and backend are filled here; `trace_fields` as for TraceScope), then launch(choice)
/// inside it, so children nest under this line. @return whatever @p launch returns.
template <Backend B, class T, class Choice, std::size_t N, class CanRun, class Launch>
decltype(auto) run(const OpSpec& op, Queue& q, const Key& key, const std::array<Choice, N>& candidates,
                   CanRun&& can_run, coverage::Shape shape, const Key& trace_fields, Launch&& launch) {
    const Device& d = device_of<B>(q, op.vendor);
    const Choice c = pick<B, T>(op, d, key, candidates, can_run);
    shape.scalar = scalar_kind_of<T>;
    shape.backend = B;
    NativeFacts facts;
    if (coverage::dynamic_enabled()) facts = native_facts(candidates, [&](const Choice& k) { return can_run(k, d); });
    TraceScope trace(op.name(), c, shape, facts, trace_fields);
    return launch(c);
}

/// Runs f(queue) on q if it is in order, else on an in-order queue that first waits on q's
/// pending work (the multi-launch drivers need one).
template <class F>
decltype(auto) on_in_order_queue(Queue& q, F&& f) {
    if (q.in_order()) return f(q);
    Queue in_order(q, true);
    Event dep = q.get_event();
    in_order.enqueue(dep);
    return f(in_order);
}

/// @}

namespace testing {
// Replace the embedded tables with {file name, text} pairs; caches are dropped.
BATCHLAS_API void set_builtin_tables(std::vector<std::pair<std::string, std::string>> files);
BATCHLAS_API void use_embedded_tables();
BATCHLAS_API void reset_warnings();
}  // namespace testing

}  // namespace batchlas::select
