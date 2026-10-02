#pragma once

// The op-agnostic half of select / size / run. An op supplies its Table (descriptors), its Shape
// and its choosers; pin parsing, the pin policy, decide-with-plan, sizing, the checked launch,
// explain-with-kernel and the coverage record are all here, once.

#include "route_table.hh"

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/settings.hh>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

namespace batchlas::dispatch::sel {

// BATCHLAS_PIN_POLICY=strict makes an unhonoured pin throw; the override is a test hook.
inline std::optional<PinPolicy>& pin_policy_override() {
    static std::optional<PinPolicy> p;
    return p;
}
inline PinPolicy pin_policy() {
    if (pin_policy_override()) return *pin_policy_override();
    const char* v = std::getenv("BATCHLAS_PIN_POLICY");
    return (v && std::strcmp(v, "strict") == 0) ? PinPolicy::Strict : PinPolicy::Warn;
}
inline std::atomic<std::size_t>& pin_warnings() {
    static std::atomic<std::size_t> n{0};
    return n;
}

// Leaked on purpose: no static-destruction ordering hazard.
inline std::string_view intern(const std::string& s) {
    static std::mutex mu;
    static auto* pool = new std::set<std::string>;
    const std::lock_guard<std::mutex> lock(mu);
    return *pool->insert(s).first;
}

// Exact table key first ("native:cta_wg"), then the shared route grammar ("native", "cta").
template <class Table>
std::optional<Pin> parse_pin() {
    constexpr Op op = Table::OpT::op;
    const char* raw = settings().routing.canonical_route(op).get();
    const bool canonical = raw && *raw;
    const std::string text = canonical ? raw : std::string();
    if (canonical) {
        for (int i = 0; i < Table::size; ++i) {
            if (Table::keys[i] == text) return Pin{intern(text), false, {}, Table::keys[i]};
        }
    }
    const auto parsed = parse_route_env(op);
    if (!parsed.found) return std::nullopt;
    const std::string_view shown = intern(canonical ? text : std::string(parsed.source.value));
    if (parsed.route.algo == Algorithm::Auto) return Pin{shown, true, parsed.route.origin, {}};
    for (int i = 0; i < Table::size; ++i) {
        if (Table::routes[i] == parsed.route) return Pin{shown, false, {}, Table::keys[i]};
    }
    return Pin{shown, false, {}, shown};   // names no row: recorded as not honoured
}

template <class Table>
std::string explain(const Selection<Table>& s) {
    if (s.decision.index < 0) return explain(s.candidates, s.decision);
    const std::string kernel = Table::with_row(
        s.plan, [&]<class R>(const typename R::Geometry& g) { return R::kernel(g); });
    return explain(s.candidates, s.decision, kernel);
}

// Legality of every row, the pin filter, the chooser chain, then the chosen row's plan.
template <class Table, Chooser... C>
Selection<Table> decide_selection(const typename Table::Shape& s, const std::optional<Pin>& pin,
                                  const C&... ch) {
    typename Table::Geoms g;
    Selection<Table> out;
    out.shape = s;
    out.candidates = Table::candidates(s, g);
    out.decision = decide(out.candidates, pin, ch...);
    if (out.decision.index >= 0) out.plan = Table::plan_at(out.decision.index, g);
    return out;
}

// Warn once per (op, pin) or, under Strict, throw. A no-op when the pin was honoured.
template <class Table>
void enforce_pin(const Selection<Table>& s) {
    const Decision& d = s.decision;
    if (d.pin.empty() || d.pin_honoured) return;
    const std::string op(op_name(Table::OpT::op));
    const std::string head = op + " pin '" + std::string(d.pin) + "' not honoured:\n";
    if (pin_policy() == PinPolicy::Strict) throw std::invalid_argument(head + explain(s));
    static std::mutex mu;
    static auto* seen = new std::set<std::string>;
    {
        const std::lock_guard<std::mutex> lock(mu);
        if (!seen->insert(op + "|" + std::string(d.pin)).second) return;
    }
    pin_warnings().fetch_add(1);
    std::fprintf(stderr, "batchlas: %s%s", head.c_str(), explain(s).c_str());
}

template <class Table>
void size_selection(Queue& q, Selection<Table>& s, const typename Table::Args& a) {
    if (s.decision.index < 0) return;
    s.workspace = Table::with_row(s.plan, [&]<class R>(const typename R::Geometry& g) {
        return R::workspace(q, a, g);
    });
    s.sized = true;
}

// The launch of a Selection, checked against the call it is applied to.
template <class Table>
Event run_selection(Queue& q, const Selection<Table>& s, const typename Table::Args& a,
                    std::size_t ws_bytes) {
    const std::string op(op_name(Table::OpT::op));
    if (s.decision.index < 0) throw std::invalid_argument(op + ": the selection holds no route");
    if (!s.sized) {
        throw std::invalid_argument(op + ": the selection was made without sizing (DecideOnly)");
    }
    if (!Table::OpT::matches(s.shape, a)) {
        throw std::invalid_argument(op + ": the selection was made for a different call");
    }
    if (ws_bytes < s.workspace) {
        throw std::length_error(op + ": workspace " + std::to_string(ws_bytes) + " B < " +
                                std::to_string(s.workspace) + " B needed by " +
                                std::string(s.key()));
    }
    return Table::with_row(s.plan, [&]<class R>(const typename R::Geometry& g) {
        return R::launch(q, a, g);
    });
}

// The one dynamic-coverage record; `row` is the op's OpShape with device facts filled.
template <class Table>
void record_coverage(const Selection<Table>& s, OpShape row) {
    if (!coverage::dynamic_enabled() || s.decision.index < 0) return;
    row.cost_extrapolated = s.decision.extrapolated;
    int native_supported = 0;
    for (int i = 0; i < s.candidates.size; ++i) {
        native_supported += is_native(s.candidates.row[i].route) && s.candidates.row[i].legal.ok;
    }
    coverage::record_if_enabled(Table::OpT::op, row.scalar, row.backend, row,
                                s.candidates.row[s.decision.index].route, true,
                                native_supported > 0);
}

}  // namespace batchlas::dispatch::sel
