#pragma once

// PlanTree core: kernel selection as a VALUE. plan_op() builds a Node tree from one candidate
// table and one strategy; Exec<B, Tier>::run executes the Node and re-derives nothing.
// SYCL-free. Prototype; see experiments/kernel_selection/planner-tree/README.md.

#include <batchlas/arch/arch_key.hh>
#include <batchlas/blas/dispatch/route.hh>
#include <batchlas/error.hh>

#include "../util/launch_plan.hh"

#include <algorithm>
#include <any>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

// ODR isolation. Every inline/template body whose meaning a macro can change lives in
// `inline namespace BATCHLAS_PLAN_ABI`, whose NAME encodes those macros. A test TU built
// with -DBATCHLAS_PLAN_MUTANT=k therefore mangles every planner symbol differently from the
// library's copy, so the dynamic linker cannot interpose one onto the other. The types the
// library API passes across the .so boundary (Node, Request, Facts, ...) stay outside it:
// they are macro-independent, and tagging them would change potrf_plan_node's mangling.
#ifndef BATCHLAS_PLAN_MUTANT   // test-only deliberate breaks (mutants.sh); 0 = none
#define BATCHLAS_PLAN_MUTANT 0
#endif
#ifdef BATCHLAS_PLAN_MOCK_DROP_CTA_WG_EXEC
#define BATCHLAS_PLAN_MOCK_BIT 1
#else
#define BATCHLAS_PLAN_MOCK_BIT 0
#endif
#define BATCHLAS_PLAN_CAT4_(a, b, c, d) a##b##c##d
#define BATCHLAS_PLAN_CAT4(a, b, c, d) BATCHLAS_PLAN_CAT4_(a, b, c, d)
#define BATCHLAS_PLAN_ABI BATCHLAS_PLAN_CAT4(abi_m, BATCHLAS_PLAN_MUTANT, _x, BATCHLAS_PLAN_MOCK_BIT)
#define BATCHLAS_PLAN_STR_(x) #x
#define BATCHLAS_PLAN_STR(x) BATCHLAS_PLAN_STR_(x)
#define BATCHLAS_PLAN_ABI_STR BATCHLAS_PLAN_STR(BATCHLAS_PLAN_ABI)

namespace batchlas::plan {

using dispatch::Algorithm;
using dispatch::Op;
using dispatch::Origin;
using dispatch::Route;
using dispatch::ScalarKind;

// The ONE device-facts struct a plan reads: replaces dispatch::DeviceFacts + launch_plan's.
struct Facts {
    launch_plan::DeviceFacts launch{};   // SLM, max wg, CUs, and the profile's thread/group caps
    arch::RoutingProfile arch = arch::RoutingProfile::Unset;
    bool arch_nearest = false;
    bool is_gpu = false;
    bool has_sg32 = false;
    int max_sub_group = 0;               // what sub-op OpShapes carry (sub_group_sizes()[0])
    int cuda_cc = 0;
    Backend backend = Backend::AUTO;
    bool vendor_available = false;       // solver_vendor_available<B>, or false for a vendor-free walk
    bool level3_vendor_available = false;   // the sub-ops' (gemm/trsm) library
};

enum class Why : std::uint8_t {
    Forced, PinRejected, Cheapest, VendorMargin, Window, NoWindow, FirstLegal, OnlyLegal
};

struct Legality {
    bool ok = false;
    const char* reason = "";
    static constexpr Legality yes() { return {true, ""}; }
    static constexpr Legality no(const char* r) { return {false, r}; }
};

struct Candidate {   // one EXPLAIN row
    Route route{};
    std::string_view name;
    Legality legal{};
    bool candidate = false;   // measured vs the vendor for (arch, op, dtype, mode)
    bool selectable = true;   // Auto may pick it; false = measured nowhere, reachable by pin only
    double seconds = -1;      // < 0: unpriced
    bool extrapolated = true;
};

struct Param {
    std::string_view key;
    std::int64_t value = 0;
};

struct Decision {
    Route forced{};
    std::string forced_name;   // a name-scoped pin (BATCHLAS_ROUTE=potrf=<tier name>)
    bool honoured = true;
    Why why = Why::OnlyLegal;
    std::string_view chooser;
    std::vector<Candidate> candidates;
};

struct Node {
    Op op = Op::COUNT;
    Route route{};
    std::string_view tier;     // the descriptor's name: Exec is found by it, never by index
    std::string_view kernel;
    std::int64_t m = 0, n = 0, k = 0, batch = 0;
    launch_plan::LaunchPlan launch{};   // the geometry that WILL launch
    std::any geometry;                  // the tier's typed geometry; Exec reads it back
    std::array<Param, 8> params{};
    int nparams = 0;
    std::size_t own_ws = 0;             // bound by Exec<B, Tier>::bind: a measuring replay
    std::size_t cover_ws = 0;           // what buffer_size reports: >= workspace(), see potrf_exec.hh
    bool bound = false;
    Decision decision;
    std::vector<Node> children;         // sub-op plans, distinct shapes, issue order

    Node& set(std::string_view key, std::int64_t v) {
        if (nparams == static_cast<int>(params.size())) {
            throw internal_error("plan::Node: too many params");
        }
        params[static_cast<std::size_t>(nparams++)] = {key, v};
        return *this;
    }
    std::int64_t param(std::string_view key) const {
        for (int i = 0; i < nparams; ++i) {
            if (params[static_cast<std::size_t>(i)].key == key) {
                return params[static_cast<std::size_t>(i)].value;
            }
        }
        throw internal_error("plan::Node: no param '" + std::string(key) + "' on " +
                             std::string(tier) + " (plan/exec drift)");
    }
    const Node* find_child(Op o, std::int64_t cm, std::int64_t cn, std::int64_t ck) const {
        for (const Node& c : children) {
            if (c.op == o && c.m == cm && c.n == cn && c.k == ck) return &c;
        }
        return nullptr;
    }
    const Node& child(Op o, std::int64_t cm, std::int64_t cn, std::int64_t ck) const {
        if (const Node* c = find_child(o, cm, cn, ck)) return *c;
        throw internal_error("plan::Node: " + std::string(tier) + " issued " +
                             std::string(dispatch::op_name(o)) + " " + std::to_string(cm) + "x" +
                             std::to_string(cn) + "x" + std::to_string(ck) +
                             ", which the plan did not predict (plan/exec drift)");
    }
    std::size_t workspace() const {   // children run sequentially on the tail
        std::size_t c = 0;
        for (const Node& ch : children) c = std::max(c, ch.workspace());
        return own_ws + c;
    }
};

// Everything a plan reads from settings/env, snapshotted once per call. It IS the cache key's
// settings half, so a changed pin can never serve a stale plan.
struct Request {
    Route forced{};
    std::string forced_name;     // tier name from BATCHLAS_ROUTE="potrf=<name>"; wins over forced
    bool strict = false;         // BATCHLAS_ROUTE_STRICT=1: an unhonourable pin throws
    int potrf_nb = 0, potrf_w = 0;
    int potrf_lpanel_nb = 0;     // BATCHLAS_POTRF_LPANEL_NB: the LPanel block, 0 = the type's default
    std::string strategy;        // BATCHLAS_STRATEGY = model | windows | first-legal; "" = policy
    int profile_override = -1;   // BATCHLAS_ROUTING_PROFILE as an int, -1 = unset
    Route gemm_forced{}, trsm_forced{};   // sub-op pins: children are planned under them
    bool forced_by_name() const { return !forced_name.empty(); }
    bool pinned() const { return forced.origin != Origin::Auto || forced_by_name(); }
};

// A descriptor is one route: correctness gate, geometry, cost key, sub-op expansion.
template <class D, class Shape>
concept TierDescriptor = requires(const Shape& s, const Facts& f, const Request& r, Node& nd) {
    { D::route } -> std::convertible_to<Route>;
    { D::name } -> std::convertible_to<std::string_view>;
    { D::legal(s, f) } -> std::same_as<Legality>;
    { D::sketch(s, f, r) } -> std::same_as<Node>;
    { D::expand(nd, s, f, r) };
};

// Tier identity is the NAME. A Route may be shared (CtaWg runs the CTA kernel), and then
// exactly one tier OWNS it: a route pin and today's resolver both mean the owner. A tier that
// shares a route says `static constexpr bool owns_route = false`.
template <class D>
constexpr bool owns_route_v() {
    if constexpr (requires { D::owns_route; }) return D::owns_route;
    else return true;
}

template <class... D>
struct TierList {
    static constexpr std::size_t size = sizeof...(D);
    static constexpr std::array<std::string_view, size> names{D::name...};
    static constexpr std::array<Route, size> routes{D::route...};
    static constexpr std::array<bool, size> owners{owns_route_v<D>()...};

    static constexpr bool names_unique() {
        for (std::size_t i = 0; i < size; ++i) {
            for (std::size_t j = i + 1; j < size; ++j) {
                if (names[i] == names[j]) return false;
            }
        }
        return true;
    }
    static_assert(names_unique(), "two tiers share a name: plans and profiles key on it");

    static constexpr bool one_owner_per_route() {
        for (std::size_t i = 0; i < size; ++i) {
            int owners_of_i = 0;
            for (std::size_t j = 0; j < size; ++j) owners_of_i += (routes[j] == routes[i] && owners[j]);
            if (owners_of_i != 1) return false;
        }
        return true;
    }
    static_assert(one_owner_per_route(),
                  "a Route shared by several tiers needs exactly one owner (owns_route = false on the rest)");

    // The tier a route pin, a coverage row and today's resolver mean by `r`; order-independent.
    static constexpr int owner_index(Route r) {
        for (std::size_t i = 0; i < size; ++i) {
            if (routes[i] == r && owners[i]) return static_cast<int>(i);
        }
        return -1;
    }
    static constexpr std::string_view owner_name(Route r) {
        const int i = owner_index(r);
        return i < 0 ? std::string_view{} : names[static_cast<std::size_t>(i)];
    }

    template <class F>
    static void each(F&& f) {
        std::size_t i = 0;
        (f.template operator()<D>(i++), ...);
    }

    // name -> descriptor type; an unknown name is plan/exec drift.
    template <class F>
    static auto visit(std::string_view name, F&& f) {
        using First = std::tuple_element_t<0, std::tuple<D...>>;
        using R = decltype(f.template operator()<First>());
        std::optional<R> out;
        static_cast<void>(((D::name == name && (out.emplace(f.template operator()<D>()), true)) || ...));
        if (!out) throw internal_error("plan: no tier named '" + std::string(name) + "'");
        return R(std::move(*out));
    }
};

// ---- strategies: pure functions of the candidate table; one runs per plan --------------------
// Every chooser reads Candidate::selectable: an Auto walk never returns a tier measured
// nowhere, whatever its position in the TierList. Pins bypass the choosers.

inline namespace BATCHLAS_PLAN_ABI {

struct Choice {
    int index = -1;   // -1: nothing; -2: a window matched a route that is not usable
    Why why = Why::OnlyLegal;
};

struct CostModelChooser {
    double margin = 0;
    Choice choose(std::span<const Candidate>) const;
};

struct WindowRow {   // keyed by tier NAME, like the model rows
    ScalarKind dtype;
    std::uint8_t mode;
    std::int64_t lo, hi;
    std::string_view tier;
};

struct WindowChooser {
    std::span<const WindowRow> rows;
    bool require_candidate = true;     // vs-vendor windows: yes; the native tie-break: no
    bool first_match_decides = true;   // a window: yes; a priority list: no
    Choice choose(std::span<const Candidate>, ScalarKind, std::uint8_t mode, std::int64_t key) const;
};

struct FirstLegalChooser {
    Choice choose(std::span<const Candidate>) const;
};

// AGENTS.md s9's two questions, as two tables.
struct WindowPolicy {
    WindowChooser vs_vendor;
    WindowChooser native_tier;
};

using Strategy = std::variant<CostModelChooser, WindowPolicy, FirstLegalChooser>;

// What plan_op needs to know about one op instance besides the descriptors.
struct ModelRow {
    launch_plan::CostConstants c{};
    launch_plan::SupportRegion box{};
};

struct OpContext {
    Op op = Op::COUNT;
    ScalarKind dt = ScalarKind::F32;
    std::uint8_t mode = 0;                 // uplo, trans, ...
    std::int64_t key = 0;                  // the window axis (n)
    std::int64_t box_n = 0, box_batch = 0; // the support-box coordinates
    std::optional<ModelRow> (*model_row)(arch::RoutingProfile, std::string_view, ScalarKind,
                                         std::uint8_t) = nullptr;
    bool (*measured)(arch::RoutingProfile, std::string_view, ScalarKind, std::uint8_t) = nullptr;
    bool (*selectable)(std::string_view) = nullptr;   // nullptr: every tier is
    const WindowPolicy* windows = nullptr;
    bool model_gate = false;               // PolicyRow: the (arch, op) ship gate passed
    double margin = 0;
};

}  // inline namespace BATCHLAS_PLAN_ABI

std::string_view to_string(Why);
std::string explain(const Node&, int indent = 0);

}  // namespace batchlas::plan
