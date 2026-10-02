#pragma once

// The ONE generic planner, the three strategies and EXPLAIN. SYCL-free.

#include "plan.hh"

#include <cstdio>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

namespace batchlas::plan {

inline std::string_view to_string(Why w) {
    switch (w) {
        case Why::Forced: return "forced";
        case Why::PinRejected: return "pin-rejected";
        case Why::Cheapest: return "cheapest";
        case Why::VendorMargin: return "vendor-within-margin";
        case Why::Window: return "measured-window";
        case Why::NoWindow: return "no-window";
        case Why::FirstLegal: return "first-legal";
        case Why::OnlyLegal: return "only-legal";
    }
    return "?";
}

inline namespace BATCHLAS_PLAN_ABI {   // see plan.hh: ODR isolation

inline std::string route_name(Route r) {
    return std::string(dispatch::to_string(r.origin)) + ":" + std::string(dispatch::to_string(r.algo));
}

// The incumbent is the first vendor candidate: it keeps a shape unless a native beats its
// price by the margin.
inline int incumbent(std::span<const Candidate> cs) {
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        if (dispatch::is_vendor(cs[i].route)) return i;
    }
    return -1;
}

inline Choice CostModelChooser::choose(std::span<const Candidate> cs) const {
    int best = -1;
    const int v = incumbent(cs);
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        const Candidate& c = cs[i];
        if (dispatch::is_vendor(c.route)) continue;
        if (!c.legal.ok || !c.selectable || !c.candidate || c.seconds < 0) continue;
        if (best < 0 || c.seconds < cs[best].seconds) best = i;
    }
    if (v >= 0 && cs[v].legal.ok && cs[v].seconds >= 0) {
        if (best < 0 || !(cs[best].seconds < (1 - margin) * cs[v].seconds)) {
            return {v, Why::VendorMargin};
        }
        return {best, Why::Cheapest};
    }
    if (best >= 0) return {best, Why::Cheapest};
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        if (cs[i].legal.ok && cs[i].selectable) {
            return {i, dispatch::is_vendor(cs[i].route) ? Why::OnlyLegal : Why::FirstLegal};
        }
    }
    return {};
}

inline int find_tier(std::span<const Candidate> cs, std::string_view name) {
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        if (cs[i].name == name) return i;
    }
    return -1;
}

inline Choice WindowChooser::choose(std::span<const Candidate> cs, ScalarKind dt,
                                    std::uint8_t mode, std::int64_t key) const {
    for (const WindowRow& r : rows) {
        if (r.dtype != dt || r.mode != mode || key < r.lo || key > r.hi) continue;
        const int i = find_tier(cs, r.tier);
        if (i >= 0 && cs[i].legal.ok && cs[i].selectable && (cs[i].candidate || !require_candidate)) {
            return {i, Why::Window};
        }
        if (first_match_decides) return {-2, Why::NoWindow};
    }
    return {-1, Why::NoWindow};
}

inline Choice FirstLegalChooser::choose(std::span<const Candidate> cs) const {
    for (int i = 0; i < static_cast<int>(cs.size()); ++i) {
        if (cs[i].legal.ok && cs[i].selectable && dispatch::is_native(cs[i].route)) return {i, Why::FirstLegal};
    }
    return {};
}

// vendor_ok == false is the vendor-free walk: the incumbent is struck from the table.
inline Choice run_strategy(const Strategy& st, std::vector<Candidate> pool, ScalarKind dt,
                           std::uint8_t mode, std::int64_t key, bool vendor_ok) {
    const int v = incumbent(pool);
    if (!vendor_ok && v >= 0) pool[static_cast<std::size_t>(v)].legal = Legality::no("excluded");
    if (const auto* cm = std::get_if<CostModelChooser>(&st)) return cm->choose(pool);
    if (const auto* w = std::get_if<WindowPolicy>(&st)) {
        const Choice c = w->vs_vendor.choose(pool, dt, mode, key);
        if (c.index >= 0) return c;
        if (vendor_ok && v >= 0 && pool[static_cast<std::size_t>(v)].legal.ok) {
            return {v, Why::NoWindow};
        }
        const Choice t = w->native_tier.choose(pool, dt, mode, key);
        if (t.index >= 0) return t;
        return FirstLegalChooser{}.choose(pool);
    }
    const Choice c = FirstLegalChooser{}.choose(pool);
    if (c.index >= 0 || !vendor_ok || v < 0) return c;
    return {v, Why::OnlyLegal};
}

inline std::string_view strategy_name(const Strategy& st) {
    if (std::holds_alternative<CostModelChooser>(st)) return "cost-model";
    if (std::holds_alternative<WindowPolicy>(st)) return "windows";
    return "first-legal";
}

// Warn once per (op, pin): the decision is recorded on the Node either way.
inline void warn_pin_once(Op op, const std::string& pin, const std::string& served) {
    static std::mutex* mu = new std::mutex();
    static auto* seen = new std::set<std::string>();
    const std::string key = std::string(dispatch::op_name(op)) + "|" + pin;
    {
        std::lock_guard<std::mutex> lock(*mu);
        if (!seen->insert(key).second) return;
    }
    std::fprintf(stderr,
                 "BatchLAS: %s pin '%s' cannot serve this shape; running %s instead "
                 "(BATCHLAS_ROUTE_STRICT=1 makes this an error)\n",
                 std::string(dispatch::op_name(op)).c_str(), pin.c_str(), served.c_str());
}

// Builds the candidate table over Tiers, picks ONE strategy from policy data, applies the pin,
// and expands only the winner. Exec binds the workspace afterwards (it needs the queue).
template <class Tiers, class Shape>
Node plan_op(const OpContext& ctx, const Shape& s, const Facts& f, const Request& req) {
    constexpr std::size_t N = Tiers::size;
    std::vector<Candidate> cs(N);
    std::vector<Node> sk(N);
    Tiers::each([&]<class D>(std::size_t i) {
        static_assert(TierDescriptor<D, Shape>, "not a tier descriptor");
        Candidate& c = cs[i];
        c.route = D::route;
        c.name = D::name;
        c.legal = D::legal(s, f);
        c.candidate = ctx.measured && ctx.measured(f.arch, D::name, ctx.dt, ctx.mode);
        c.selectable = !ctx.selectable || ctx.selectable(D::name);
        if (!c.legal.ok && !dispatch::is_vendor(D::route)) return;   // the incumbent is priced always
        sk[i] = D::sketch(s, f, req);
        sk[i].tier = D::name;
        if (!ctx.model_row || !sk[i].launch.fits) return;
        if (const auto e = ctx.model_row(f.arch, D::name, ctx.dt, ctx.mode)) {
            const auto pr = launch_plan::predict(sk[i].launch, f.launch, e->c, e->box, ctx.box_n,
                                                 ctx.box_batch);
            c.seconds = pr.seconds;
            c.extrapolated = pr.extrapolated;
        }
    });

    const int vi = incumbent(cs);
    const bool model = ctx.model_gate && f.is_gpu && vi >= 0 && cs[static_cast<std::size_t>(vi)].seconds >= 0;
    Strategy st = FirstLegalChooser{};
    if (req.strategy == "model") st = CostModelChooser{ctx.margin};
    else if (req.strategy == "windows" && ctx.windows) st = *ctx.windows;
    else if (req.strategy == "first-legal") st = FirstLegalChooser{};
    else if (model) st = CostModelChooser{ctx.margin};
    else if (ctx.windows) st = *ctx.windows;

    Decision d;
    d.forced = req.forced;
    d.forced_name = req.forced_name;
    d.chooser = strategy_name(st);
    auto automatic = [&](bool vendor_ok) {
        return run_strategy(st, cs, ctx.dt, ctx.mode, ctx.key, vendor_ok);
    };
    Choice ch;
    if (req.forced_by_name()) {
        const int i = find_tier(cs, req.forced_name);
        if (i >= 0 && cs[static_cast<std::size_t>(i)].legal.ok) ch = {i, Why::Forced};
        else { d.honoured = false; ch = automatic(f.vendor_available); ch.why = Why::PinRejected; }
    } else if (req.forced.origin == Origin::Auto) {
        ch = automatic(f.vendor_available);
    } else if (req.forced.algo == Algorithm::Auto && dispatch::is_native(req.forced)) {
        ch = automatic(false);   // a bare `native` pin IS the vendor-free walk
        if (ch.index < 0) { d.honoured = false; ch = automatic(f.vendor_available); ch.why = Why::PinRejected; }
    } else {
        // A route pin means the route's OWNER, never "the first tier with that route".
        const int i = Tiers::owner_index(req.forced);
        if (i >= 0 && cs[static_cast<std::size_t>(i)].legal.ok) ch = {i, Why::Forced};
        else { d.honoured = false; ch = automatic(f.vendor_available); ch.why = Why::PinRejected; }
    }
    if (ch.index < 0) ch.index = vi;   // nothing serves it: the vendor arm throws NoRoute
    if (ch.index < 0) throw internal_error("plan_op: no tier and no incumbent");
    d.why = ch.why;
    d.candidates = cs;

    const std::size_t w = static_cast<std::size_t>(ch.index);
    Node out = (sk[w].op == Op::COUNT) ? Node{} : std::move(sk[w]);
    if (out.op == Op::COUNT) {   // an illegal incumbent was never sketched
        Tiers::visit(cs[w].name, [&]<class D>() { out = D::sketch(s, f, req); return 0; });
        out.tier = cs[w].name;
    }
    if (!d.honoured) {
        const std::string pin = req.forced_by_name() ? req.forced_name : route_name(req.forced);
        if (req.strict) {
            throw invalid_argument("BatchLAS: " + std::string(dispatch::op_name(ctx.op)) + " pin '" +
                                   pin + "' cannot serve this shape (BATCHLAS_ROUTE_STRICT=1)");
        }
        warn_pin_once(ctx.op, (req.forced_by_name() ? "name:" : "route:") + pin, std::string(out.tier));
    }
    out.decision = std::move(d);
    Tiers::visit(out.tier, [&]<class D>() { D::expand(out, s, f, req); return 0; });
    return out;
}

// One tier's expanded plan outside any choice: the covering size binds one per legal native tier.
template <class Tiers, class Shape>
Node plan_tier(std::string_view name, const Shape& s, const Facts& f, const Request& req) {
    Node out;
    Tiers::visit(name, [&]<class D>() {
        out = D::sketch(s, f, req);
        out.tier = D::name;
        D::expand(out, s, f, req);
        return 0;
    });
    return out;
}

}  // inline namespace BATCHLAS_PLAN_ABI

inline std::string explain(const Node& nd, int indent) {
    std::ostringstream o;
    const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
    o << pad << dispatch::op_name(nd.op) << " " << nd.m << "x" << nd.n;
    if (nd.k) o << "x" << nd.k;
    o << " b=" << nd.batch << " -> " << (nd.tier.empty() ? route_name(nd.route) : std::string(nd.tier))
      << "  [" << nd.kernel << "]";
    for (int i = 0; i < nd.nparams; ++i) {
        o << " " << nd.params[static_cast<std::size_t>(i)].key << "=" << nd.params[static_cast<std::size_t>(i)].value;
    }
    if (nd.launch.fits && nd.launch.wg_size > 0) {
        o << " wg=" << nd.launch.wg_size << " groups=" << nd.launch.groups;
    }
    if (nd.bound) o << " ws=" << nd.workspace() << "B";
    o << "\n";
    const Decision& d = nd.decision;
    if (!d.candidates.empty()) {
        o << pad << "  why: " << to_string(d.why) << " (chooser " << d.chooser << ")";
        if (!d.forced_name.empty() || d.forced.origin != Origin::Auto) {
            o << "; forced " << (d.forced_name.empty() ? route_name(d.forced) : d.forced_name)
              << (d.honoured ? " honoured" : " NOT honoured");
        }
        o << "\n";
        for (const Candidate& c : d.candidates) {
            o << pad << "    " << c.name << ": " << (c.legal.ok ? "legal" : c.legal.reason);
            if (c.legal.ok) o << (c.candidate ? ", candidate" : ", not a candidate here");
            if (c.seconds >= 0) {
                o << ", " << c.seconds * 1e6 << " us" << (c.extrapolated ? " (extrapolated)" : "");
            }
            o << "\n";
        }
    }
    for (const Node& ch : nd.children) o << explain(ch, indent + 1);
    return o.str();
}

}  // namespace batchlas::plan
