#pragma once

// The per-op route table IS a type list of descriptors. The Plan variant, the key array and
// the candidate rows are derived from it, so no parallel list exists to forget. SYCL-free:
// descriptors declare workspace()/launch(); only the op's .cc odr-uses them.

#include "select.hh"

#include <batchlas/util/sycl-device-queue.hh>

#include <concepts>
#include <cstddef>
#include <cstdio>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace batchlas::dispatch::sel {

template <class R, class Op>
concept RouteDescriptor = requires(const typename Op::Shape& s, const typename R::Geometry& g,
                                   Queue& q, const typename Op::Args& a) {
    { R::route } -> std::convertible_to<Route>;
    { R::key } -> std::convertible_to<std::string_view>;
    { R::legal(s) } -> std::same_as<Verdict>;
    { R::plan(s) } -> std::same_as<typename R::Geometry>;
    { R::cost_plan(s, g) } -> std::same_as<launch_plan::LaunchPlan>;
    { R::kernel(g) } -> std::same_as<std::string>;
    { R::workspace(q, a, g) } -> std::same_as<std::size_t>;
    { R::launch(q, a, g) } -> std::same_as<Event>;
    { g.fits } -> std::convertible_to<bool>;
};

template <class Op, class... Rs>
    requires(RouteDescriptor<Rs, Op> && ...)
struct Table {
    using Shape = typename Op::Shape;
    using Plan = std::variant<typename Rs::Geometry...>;   // index == row == Decision::index
    using Geoms = std::tuple<typename Rs::Geometry...>;
    static constexpr int size = static_cast<int>(sizeof...(Rs));
    static constexpr std::array<std::string_view, sizeof...(Rs)> keys{Rs::key...};
    static constexpr std::array<Route, sizeof...(Rs)> routes{Rs::route...};
    static_assert(size <= kMaxRoutes, "raise kMaxRoutes");

    static consteval bool unique_keys() {
        for (int i = 0; i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                if (keys[i] == keys[j]) return false;
            }
        }
        return true;
    }
    static_assert(unique_keys(), "two descriptors share a key");

    static constexpr int index_of(std::string_view k) {
        for (int i = 0; i < size; ++i) {
            if (keys[i] == k) return i;
        }
        return -1;
    }

    // Legality, plan and cost plan of EVERY row, once per call.
    static Candidates candidates(const Shape& s, Geoms& g) {
        Candidates c;
        c.size = size;
        c.variant = Op::cost_variant(s);
        c.dims = Op::coords(s, c.coords);
        fill(s, g, c, std::index_sequence_for<Rs...>{});
        return c;
    }

    static Plan plan_at(int i, const Geoms& g) { return pick(i, g, std::index_sequence_for<Rs...>{}); }

    // The ONE row-to-descriptor map: f.template operator()<R>(geometry).
    template <class F>
    static decltype(auto) with_row(const Plan& p, F&& f) {
        return visit_row(p, f, std::index_sequence_for<Rs...>{});
    }

private:
    template <std::size_t... I>
    static void fill(const Shape& s, Geoms& g, Candidates& c, std::index_sequence<I...>) {
        ((c.row[I] = row<Rs>(s, std::get<I>(g))), ...);
    }
    template <class R>
    static Candidate row(const Shape& s, typename R::Geometry& g) {
        Candidate r{R::route, R::key, R::legal(s), {}};
        if (r.legal.ok) {
            g = R::plan(s);
            r.lp = R::cost_plan(s, g);
            r.lp.fits = g.fits;   // the launcher's geometry is the capacity gate
        }
        return r;
    }
    template <std::size_t... I>
    static Plan pick(int i, const Geoms& g, std::index_sequence<I...>) {
        Plan out;
        ((static_cast<int>(I) == i ? (out.template emplace<I>(std::get<I>(g)), 0) : 0), ...);
        return out;
    }
    template <class F, std::size_t... I>
    static decltype(auto) visit_row(const Plan& p, F& f, std::index_sequence<I...>) {
        using R0 = std::tuple_element_t<0, std::tuple<Rs...>>;
        using Ret = decltype(f.template operator()<R0>(std::get<0>(p)));
        if constexpr (std::is_void_v<Ret>) {
            ((p.index() == I ? (f.template operator()<Rs>(std::get<I>(p)), 0) : 0), ...);
        } else {
            Ret out{};
            ((p.index() == I ? (out = f.template operator()<Rs>(std::get<I>(p)), 0) : 0), ...);
            return out;
        }
    }
};

// The decision as a value, consumed by sizing and by running.
template <class Tbl>
struct Selection {
    typename Tbl::Plan plan;
    Candidates candidates;
    Decision decision;
    std::size_t workspace = 0;   // of THIS route only
    std::uint64_t settings_epoch = 0;
    std::string_view key() const { return candidates.row[decision.index].key; }
};

enum class PinPolicy : std::uint8_t { Warn, Strict };

// Generated "why": every row's legality, fit, geometry and price, then the decision.
inline std::string explain(const Candidates& c, const Decision& d, const std::string& kernel = {}) {
    std::string out;
    char buf[320];
    for (int i = 0; i < c.size; ++i) {
        const Candidate& r = c.row[i];
        const char* state = !r.legal.ok ? "illegal" : !r.lp.fits ? "unfit" : "eligible";
        char cost[32] = "-";
        if (d.cost[i] >= 0) std::snprintf(cost, sizeof cost, "%.1fus", d.cost[i] * 1e6);
        std::snprintf(buf, sizeof buf, "  %c %-16.*s %-8s %-28.*s groups=%-7lld wg=%-4d cost=%s\n",
                      i == d.index ? '*' : ' ', static_cast<int>(r.key.size()), r.key.data(), state,
                      static_cast<int>(r.legal.why.size()), r.legal.why.data(),
                      static_cast<long long>(r.lp.groups), r.lp.wg_size, cost);
        out += buf;
    }
    std::snprintf(buf, sizeof buf, "  by=%s why=\"%.*s\" pin=\"%.*s\" honoured=%d extrapolated=%d\n",
                  to_cstr(d.by), static_cast<int>(d.why.size()), d.why.data(),
                  static_cast<int>(d.pin.size()), d.pin.data(), d.pin_honoured ? 1 : 0,
                  d.extrapolated);
    out += buf;
    if (!kernel.empty()) out += "  kernel: " + kernel + "\n";
    return out;
}

}  // namespace batchlas::dispatch::sel
