#pragma once

// Selection policy as data: a generated first-match RuleSet per (arch, op) ranks tier ids,
// and one engine returns the first ranked tier whose legal() admits the shape. SYCL-free,
// so tools and offline tests include it. See evaluation/routing/compile_rules.py.

#include <batchlas/blas/dispatch/route.hh>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace batchlas::routing {

inline constexpr int kMaxF = 4, kMaxRank = 6, kMaxKnobs = 3;
using Features = std::array<std::int64_t, kMaxF>;
inline constexpr std::int64_t kInf = std::numeric_limits<std::int64_t>::max();

struct Knobs { std::array<std::int16_t, kMaxKnobs> v{}; };   // 0 = the plan's own default
struct Candidate { std::uint8_t name = 0; Knobs knobs{}; };  // name: index into RuleSet::names
enum class Source : std::uint8_t { Hand, Model, Measured, Override };

struct Rule {
    std::uint32_t id;
    std::uint16_t key;
    Features lo, hi;   // closed box
    std::array<Candidate, kMaxRank> rank;
    std::uint8_t nrank;
    Source source;
};

struct RuleSet {
    std::string_view op, arch, provenance;
    const Rule* begin;
    const Rule* end;
    // Rules are sorted by key; key_off[k]..key_off[k+1] is key k's slice.
    const std::uint32_t* key_off;
    std::uint16_t nkeys;
    Features compiled_for;   // cus, local_mem, max_wg, sub-group

    const Rule* match(std::uint16_t key, const Features& f) const {
        if (key >= nkeys) return nullptr;
        for (const Rule* r = begin + key_off[key]; r != begin + key_off[key + 1]; ++r) {
            bool in = true;
            for (int i = 0; i < kMaxF; ++i) in = in && f[i] >= r->lo[i] && f[i] <= r->hi[i];
            if (in) return r;
        }
        return nullptr;
    }
};

enum class Reason : std::uint8_t { Rule, RankFallback, NoRule, Pinned, PinRefused };

inline constexpr std::string_view to_string(Reason r) {
    switch (r) {
        case Reason::Rule: return "rule";
        case Reason::RankFallback: return "rank_fallback";
        case Reason::NoRule: return "no_rule";
        case Reason::Pinned: return "pinned";
        case Reason::PinRefused: return "pin_refused";
    }
    return "?";
}
inline constexpr std::string_view to_string(Source s) {
    switch (s) {
        case Source::Hand: return "hand";
        case Source::Model: return "model";
        case Source::Measured: return "measured";
        case Source::Override: return "override";
    }
    return "?";
}

// A pin names a tier id ("native:lpanel"), or only an origin ("native"), which restricts the
// rule's ranking to that origin -- the vendor-free walk, forced.
struct Pin {
    std::string text;
    int tier = -1;
    bool origin_only = false;
    dispatch::Origin origin = dispatch::Origin::Auto;
    Knobs knobs{};
    bool active = false, strict = false;
};

template <class... Tiers>
struct TierList {
    static constexpr std::size_t size = sizeof...(Tiers);
    static constexpr std::array<std::string_view, size> ids{Tiers::id...};
    static constexpr std::array<dispatch::Route, size> routes{Tiers::route...};
    using PlanVariant = std::variant<typename Tiers::Plan...>;   // index == tier index

    static constexpr int index_of(std::string_view s) {
        for (std::size_t i = 0; i < size; ++i) {
            if (ids[i] == s) return static_cast<int>(i);
        }
        return -1;
    }
    static consteval bool ids_unique() {
        for (std::size_t i = 0; i < size; ++i) {
            for (std::size_t j = i + 1; j < size; ++j) {
                if (ids[i] == ids[j]) return false;
            }
        }
        return true;
    }
    static_assert(ids_unique(), "duplicate tier id in a TierList");

    template <class Ctx>
    static bool legal(std::size_t i, const Ctx& c) {
        bool ok = false;
        std::size_t j = 0;
        ((ok = (j++ == i) ? Tiers::legal(c) : ok), ...);
        return ok;
    }
    template <class Ctx>
    static PlanVariant plan(std::size_t i, const Ctx& c, const Knobs& k) {
        return plan_impl(i, c, k, std::index_sequence_for<Tiers...>{});
    }
    template <class Ctx, std::size_t... I>
    static PlanVariant plan_impl(std::size_t i, const Ctx& c, const Knobs& k,
                                 std::index_sequence<I...>) {
        PlanVariant out;
        ((i == I ? (out.template emplace<I>(Tiers::plan(c, k)), 0) : 0), ...);
        return out;
    }
    template <class Ctx>
    static std::size_t workspace(const PlanVariant& p, const Ctx& c) {
        return ws_impl(p, c, std::index_sequence_for<Tiers...>{});
    }
    template <class Ctx, std::size_t... I>
    static std::size_t ws_impl(const PlanVariant& p, const Ctx& c, std::index_sequence<I...>) {
        std::size_t out = 0;
        ((p.index() == I ? (out = Tiers::workspace(std::get<I>(p), c), 0) : 0), ...);
        return out;
    }
    template <class Ev, class Call>
    static Ev launch(const PlanVariant& p, Call& k) {
        return launch_impl<Ev>(p, k, std::index_sequence_for<Tiers...>{});
    }
    template <class Ev, class Call, std::size_t... I>
    static Ev launch_impl(const PlanVariant& p, Call& k, std::index_sequence<I...>) {
        Ev out{};
        ((p.index() == I ? (out = Tiers::launch(std::get<I>(p), k), 0) : 0), ...);
        return out;
    }
};

// A rules file naming a tier the TierList does not register is a compile error here.
template <class Tiers, std::size_t N>
consteval std::array<std::uint8_t, N> bind(const std::array<std::string_view, N>& names) {
    std::array<std::uint8_t, N> m{};
    for (std::size_t i = 0; i < N; ++i) {
        const int t = Tiers::index_of(names[i]);
        if (t < 0) throw "rules name a tier the TierList does not register";
        m[i] = static_cast<std::uint8_t>(t);
    }
    return m;
}

template <class Tiers>
struct Selection {
    std::uint8_t tier = 0;
    typename Tiers::PlanVariant plan;
    std::size_t ws_bytes = 0;
    Reason reason = Reason::NoRule;
    std::uint32_t rule_id = 0;
    Source source = Source::Hand;
    std::string forced;   // the pin text, empty when none
    const RuleSet* rules = nullptr;
    std::string_view id() const { return Tiers::ids[tier]; }
    dispatch::Route route() const { return Tiers::routes[tier]; }
};

class no_route_error : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Warn once per (op, pin) under a lock; strict mode throws instead.
inline void report_refused_pin(std::string_view op, std::string_view pin, bool strict) {
    const std::string msg = "BatchLAS routing: pin \"" + std::string(pin) + "\" for " +
                            std::string(op) +
                            " is not legal for this shape; using the rules instead";
    if (strict) throw std::invalid_argument(msg + " (BATCHLAS_ROUTING_STRICT=1)");
    static auto* mu = new std::mutex();
    static auto* seen = new std::set<std::string>();
    std::lock_guard<std::mutex> lock(*mu);
    if (seen->insert(std::string(op) + "|" + std::string(pin)).second) {
        std::fprintf(stderr, "%s\n", msg.c_str());
    }
}

template <class Tiers>
Pin parse_pin(const char* raw, bool strict) {
    Pin p;
    p.strict = strict;
    if (!raw || !*raw) return p;
    p.text = raw;
    p.active = true;
    p.tier = Tiers::index_of(p.text);
    if (p.tier >= 0) return p;
    if (p.text == "native" || p.text == "batchlas") {
        p.origin_only = true;
        p.origin = dispatch::Origin::Native;
        return p;
    }
    // Unknown words keep active with tier -1: legal() never admits them, so they are refused
    // loudly rather than silently meaning Auto.
    return p;
}

template <class Tiers, class Ctx, std::size_t N>
Selection<Tiers> select(const RuleSet& rs, const std::array<std::uint8_t, N>& bound,
                        const Ctx& c, const Pin& pin) {
    auto make = [&](std::size_t t, const Knobs& k, Reason why, const Rule* r) {
        Selection<Tiers> s;
        s.tier = static_cast<std::uint8_t>(t);
        s.plan = Tiers::plan(t, c, k);
        s.ws_bytes = Tiers::workspace(s.plan, c);
        s.reason = why;
        s.rule_id = r ? r->id : 0u;
        s.source = r ? r->source : (why == Reason::Pinned ? Source::Override : Source::Hand);
        s.forced = pin.active ? pin.text : std::string();
        s.rules = &rs;
        return s;
    };
    bool restrict_origin = pin.active && pin.origin_only;
    auto admitted = [&](std::size_t t) {
        if (restrict_origin && Tiers::routes[t].origin != pin.origin) return false;
        return Tiers::legal(t, c);
    };
    Reason miss = Reason::RankFallback;
    if (pin.active && !pin.origin_only) {
        if (pin.tier >= 0 && Tiers::legal(static_cast<std::size_t>(pin.tier), c)) {
            return make(static_cast<std::size_t>(pin.tier), pin.knobs, Reason::Pinned, nullptr);
        }
        report_refused_pin(rs.op, pin.text, pin.strict);
        miss = Reason::PinRefused;
    }
    if (restrict_origin) {
        bool any = false;
        for (std::size_t t = 0; t < Tiers::size; ++t) any = any || admitted(t);
        if (any) {
            miss = Reason::Pinned;
        } else {
            report_refused_pin(rs.op, pin.text, pin.strict);
            restrict_origin = false;
            miss = Reason::PinRefused;
        }
    }
    if (const Rule* r = rs.match(c.key(), c.features())) {
        for (std::uint8_t i = 0; i < r->nrank; ++i) {
            const std::size_t t = bound[r->rank[i].name];
            if (admitted(t)) {
                const Reason why = (i == 0 && miss == Reason::RankFallback) ? Reason::Rule : miss;
                return make(t, r->rank[i].knobs, why, r);
            }
        }
    }
    for (std::size_t t = 0; t < Tiers::size; ++t) {
        if (admitted(t)) return make(t, Knobs{}, Reason::NoRule, nullptr);
    }
    throw no_route_error("BatchLAS routing: no legal tier for " + std::string(rs.op));
}

}  // namespace batchlas::routing
