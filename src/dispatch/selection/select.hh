#pragma once

// Route selection, op-agnostic and SYCL-free: candidate rows, the strategies that choose among
// them, the pin filter and the pure decide(). Prototype of the descriptor-registry design
// (experiments/kernel_selection/descriptor-registry/README.md). No op name appears here.

#include <batchlas/blas/dispatch/route.hh>

#include "../../util/launch_plan.hh"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace batchlas::dispatch::sel {

inline constexpr int kMaxRoutes = 8;
using Mask = std::uint32_t;

// Legality WITH a reason: the only correctness gate.
struct Verdict {
    bool ok = false;
    std::string_view why{};
    static constexpr Verdict yes() { return {true, {}}; }
    static constexpr Verdict no(std::string_view w) { return {false, w}; }
};

// One row per descriptor; all a strategy may read.
struct Candidate {
    Route route{};
    std::string_view key{};   // "native:lpanel": env word, cost-row key, coverage key
    Verdict legal{};
    launch_plan::LaunchPlan lp{};
    bool eligible() const { return legal.ok && lp.fits; }
};

struct Candidates {
    std::array<Candidate, kMaxRoutes> row{};
    int size = 0;
    std::uint8_t variant = 0;               // cost-row discriminator (potrf: uplo)
    std::array<std::int64_t, 4> coords{};   // support-box point, `dims` used
    int dims = 0;
};

enum class Strategy : std::uint8_t { None, Model, Windows };   // None: legality alone decided

struct Decision {
    int index = -1;   // row == the Plan variant index
    Strategy by = Strategy::None;
    std::string_view pin{};
    bool pin_honoured = false;
    std::string_view why{};
    std::array<double, kMaxRoutes> cost{-1, -1, -1, -1, -1, -1, -1, -1};   // < 0: not priced
    std::int8_t extrapolated = -1;   // set only when by == Model
};

template <class C>
concept Chooser = requires(const C& c, const Candidates& k, Mask m) {
    { c.choose(k, m) } -> std::same_as<std::optional<Decision>>;
};

// Generated (here: re-keyed from the positional profile), keyed by NAME.
struct CostRow {
    std::string_view key;
    std::uint8_t variant = 0;
    launch_plan::CostConstants c{};
    std::array<std::int64_t, 8> box{};   // lo/hi per coordinate
    int fitted_rows = 0;
};
struct CostBook {   // one per (arch, op, dtype): its own gate and margin
    bool gate = false;
    double margin = 0;
    std::span<const CostRow> rows{};
};

struct ModelChooser {
    const CostBook* book = nullptr;
    launch_plan::DeviceFacts facts{};

    const CostRow* find(std::string_view k, std::uint8_t v) const {
        for (const CostRow& r : book->rows) {
            if (r.key == k && r.variant == v) return &r;
        }
        return nullptr;
    }
    static bool outside(const CostRow& r, const Candidates& c) {
        if (r.fitted_rows < 1) return true;
        for (int i = 0; i < c.dims; ++i) {
            if (c.coords[i] < r.box[2 * i] || c.coords[i] > r.box[2 * i + 1]) return true;
        }
        return false;
    }
    // argmin natives vs argmin vendors; the vendor keeps the call unless beaten by the margin.
    // An eligible vendor with no row DECLINES the model: nothing can be shown to beat it.
    std::optional<Decision> choose(const Candidates& c, Mask m) const {
        if (!book || !book->gate) return std::nullopt;
        Decision d;
        d.by = Strategy::Model;
        int best_nat = -1, best_ven = -1;
        for (int i = 0; i < c.size; ++i) {
            const Candidate& r = c.row[i];
            if (!(m >> i & 1u) || !r.eligible()) continue;
            const CostRow* row = find(r.key, c.variant);
            if (!row && is_vendor(r.route)) return std::nullopt;
            if (!row) continue;   // a native the book never fitted is not a model candidate
            d.cost[i] = launch_plan::cost(r.lp, facts, row->c);
            int& best = is_vendor(r.route) ? best_ven : best_nat;
            if (best < 0 || d.cost[i] < d.cost[best]) best = i;
        }
        if (best_nat < 0 && best_ven < 0) return std::nullopt;
        const bool native_wins = best_nat >= 0 &&
            (best_ven < 0 || d.cost[best_nat] < (1 - book->margin) * d.cost[best_ven]);
        d.index = native_wins ? best_nat : best_ven;
        d.why = native_wins ? (best_ven < 0 ? "model: cheapest native (no vendor eligible)"
                                            : "model: cheapest native beats vendor by margin")
                            : "model: vendor within margin";
        d.extrapolated = static_cast<std::int8_t>(outside(*find(c.row[d.index].key, c.variant), c));
        return d;
    }
};

enum class Question : std::uint8_t { VsVendor, AmongNative };

template <class Shape>
struct Window {
    std::string_view key;
    Question q;
    bool (*holds)(const Shape&);
    std::string_view evidence;
};

// First match wins: VsVendor windows, then a vendor, then AmongNative windows, then the first
// eligible native. Windows need not be mutually exclusive and never call each other.
template <class Shape>
struct WindowChooser {
    std::span<const Window<Shape>> windows;
    const Shape* s = nullptr;
    std::optional<Decision> choose(const Candidates& c, Mask m) const {
        auto pick = [&](Question q) -> int {
            for (const auto& w : windows) {
                if (w.q != q || !w.holds(*s)) continue;
                for (int i = 0; i < c.size; ++i) {
                    if ((m >> i & 1u) && c.row[i].eligible() && c.row[i].key == w.key) return i;
                }
            }
            return -1;
        };
        auto first = [&](bool vendor) {
            for (int i = 0; i < c.size; ++i) {
                if ((m >> i & 1u) && c.row[i].eligible() && is_vendor(c.row[i].route) == vendor) {
                    return i;
                }
            }
            return -1;
        };
        Decision d;
        d.by = Strategy::Windows;
        if ((d.index = pick(Question::VsVendor)) >= 0) { d.why = "window: native beats vendor"; return d; }
        if ((d.index = first(true)) >= 0) { d.why = "window: no native window, vendor"; return d; }
        if ((d.index = pick(Question::AmongNative)) >= 0) { d.why = "window: native tier"; return d; }
        if ((d.index = first(false)) >= 0) { d.why = "window: first eligible native"; return d; }
        return std::nullopt;
    }
};

template <Chooser... C>
std::optional<Decision> first_of(const Candidates& c, Mask m, const C&... ch) {
    std::optional<Decision> d;
    ((d = d ? d : ch.choose(c, m)), ...);
    return d;
}

// A pin FILTERS eligible rows; strategies never see it. Either a key or a bare origin.
struct Pin {
    std::string_view text{};
    bool origin_only = false;
    Origin origin{};
    std::string_view key{};
};

inline Mask pin_mask(const Candidates& c, const Pin& p) {
    Mask m = 0;
    for (int i = 0; i < c.size; ++i) {
        const bool match = p.origin_only ? c.row[i].route.origin == p.origin : c.row[i].key == p.key;
        if (match && c.row[i].eligible()) m |= 1u << i;
    }
    return m;
}

inline int only_eligible(const Candidates& c, Mask m) {   // -1: none or several
    int found = -1;
    for (int i = 0; i < c.size; ++i) {
        if (!(m >> i & 1u) || !c.row[i].eligible()) continue;
        if (found >= 0) return -1;
        found = i;
    }
    return found;
}

// Pure: pin filter -> "only one eligible" -> the bound chooser chain.
template <Chooser... C>
Decision decide(const Candidates& c, const std::optional<Pin>& pin, const C&... ch) {
    const Mask all = (c.size >= 32) ? ~Mask{0} : ((Mask{1} << c.size) - 1);
    auto run = [&](Mask m) -> std::optional<Decision> {
        if (const int i = only_eligible(c, m); i >= 0) {
            Decision d;
            d.index = i;
            d.why = "only eligible route";
            return d;
        }
        return first_of(c, m, ch...);
    };
    if (pin) {
        if (const Mask m = pin_mask(c, *pin)) {
            if (auto d = run(m)) {
                d->pin = pin->text;
                d->pin_honoured = true;
                return *d;
            }
        }
    }
    Decision d = run(all).value_or(Decision{});
    if (pin) {
        d.pin = pin->text;
        d.pin_honoured = false;
    }
    return d;
}

inline const char* to_cstr(Strategy s) {
    switch (s) {
        case Strategy::Model: return "model";
        case Strategy::Windows: return "windows";
        default: return "none";
    }
}

}  // namespace batchlas::dispatch::sel
