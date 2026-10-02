#pragma once

// The execution half of potrf's PlanTree: Exec<B, Tier> runs a Node (re-deriving nothing),
// binds its workspace, and the facade-side cache shares ONE Node between buffer_size and run.
// Included by src/dispatch/entry_points/factorization.cc only (plus tests, for the API below).

#include "potrf_tiers.hh"

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/settings.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include "../util/internal-api.hh"

#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace batchlas::plan {

// ---- the library-side API (defined in factorization.cc); macro-independent, untagged ---------

struct LogRow {   // one executed plan node: what coverage cannot hold yet (forced/honoured/path)
    std::string path, tier, route, forced;
    bool honoured = true;
    Why why = Why::OnlyLegal;
    Backend backend = Backend::AUTO;
    ScalarKind dt = ScalarKind::F32;
    std::int64_t m = 0, n = 0, k = 0, batch = 0;
};
struct CacheStats { std::uint64_t hits = 0, misses = 0, evictions = 0, size = 0; };

BATCHLAS_INTERNAL_API void set_potrf_planner(int mode);   // -1: BATCHLAS_POTRF_PLANNER, 0 off, 1 on
BATCHLAS_INTERNAL_API bool potrf_planner_enabled();
BATCHLAS_INTERNAL_API void log_enable(bool on);
BATCHLAS_INTERNAL_API bool log_enabled();
BATCHLAS_INTERNAL_API void log_row(LogRow row);
BATCHLAS_INTERNAL_API std::vector<LogRow> log_take();
// Sharded (shared_mutex per shard), bounded (FIFO eviction past BATCHLAS_PLAN_CACHE_CAPACITY,
// default 1024 entries); the per-thread memo below is the lock-free hit path in front of it.
BATCHLAS_INTERNAL_API std::shared_ptr<const Node> cache_find(const std::string& key);
BATCHLAS_INTERNAL_API std::shared_ptr<const Node> cache_insert(const std::string& key, Node nd);
BATCHLAS_INTERNAL_API void cache_note_hit();
BATCHLAS_INTERNAL_API std::uint64_t cache_generation();   // bumped by cache_clear()
BATCHLAS_INTERNAL_API CacheStats cache_stats();
BATCHLAS_INTERNAL_API void cache_clear();

// The library's plan for this call (vendor_free: the native-only walk, unbound, uncached).
template <Backend B, typename T>
BATCHLAS_INTERNAL_API Node potrf_plan_node(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A,
                                           Uplo uplo, bool vendor_free);

inline namespace BATCHLAS_PLAN_ABI {   // plan.hh: ODR isolation for everything below

// ---- facts and request ---------------------------------------------------------------------

struct HwFacts {
    launch_plan::DeviceFacts launch{};
    dispatch::DeviceFacts routing{};
    bool has_sg32 = false;
};

inline HwFacts hw_facts(const Device& dev) {
    static auto* mu = new std::mutex();   // leaked: routed calls may run during static teardown
    static auto* memo = new std::map<std::pair<int, std::size_t>, HwFacts>();
    const auto key = std::make_pair(static_cast<int>(dev.type), dev.idx);
    {
        std::lock_guard<std::mutex> lock(*mu);
        if (auto it = memo->find(key); it != memo->end()) return it->second;
    }
    HwFacts h;
    h.launch = sycl_potrf::potrf_device_facts(dev);
    h.routing = dispatch::device_facts(dev);
    h.launch.compute_units = h.routing.compute_units;
    h.has_sg32 = dev.supports_sub_group_size(32);
    std::lock_guard<std::mutex> lock(*mu);
    return memo->emplace(key, h).first->second;
}

template <Backend B>
Facts facts_for(const Queue& q, bool vendor_available) {
    const HwFacts h = hw_facts(q.device());
    Facts f;
    f.launch = h.launch;
    const arch::ProfileChoice pc = arch::select_profile(h.routing.key, dispatch::routing_profile_override());
    f.arch = pc.profile;
    f.arch_nearest = pc.nearest;
    if (const auto* p = potrf_profile::profile_for(f.arch)) {
        f.launch.max_threads_per_cu = p->max_threads_per_cu;
        f.launch.max_groups_per_cu = p->max_groups_per_cu;
    }
    f.is_gpu = h.routing.is_gpu;
    f.has_sg32 = h.has_sg32;
    f.max_sub_group = h.routing.max_sub_group;
    f.cuda_cc = h.routing.key.cuda_cc;
    f.backend = B;
    f.vendor_available = vendor_available;
    f.level3_vendor_available = dispatch::level3_vendor_available<B>;
    return f;
}

inline Route env_route(Op op) {
    const auto parsed = dispatch::parse_route_env(op);
    return parsed.found ? parsed.route : dispatch::legacy_unset_default(op);
}

// BATCHLAS_ROUTE="potrf=native:cta_wg;gemm=..." -- the name-scoped pin grammar.
inline std::string scoped_pin(std::string_view op) {
    const char* v = std::getenv("BATCHLAS_ROUTE");
    if (!v) return {};
    std::string_view s(v);
    while (!s.empty()) {
        const auto semi = s.find(';');
        const std::string_view item = s.substr(0, semi);
        const auto eq = item.find('=');
        if (eq != std::string_view::npos && item.substr(0, eq) == op) return std::string(item.substr(eq + 1));
        if (semi == std::string_view::npos) break;
        s.remove_prefix(semi + 1);
    }
    return {};
}

inline Request request_for_potrf() {
    Request r;
    r.forced = env_route(Op::potrf);
    r.forced_name = scoped_pin("potrf");
    const char* strict = std::getenv("BATCHLAS_ROUTE_STRICT");
    r.strict = strict && std::string_view(strict) == "1";
    r.potrf_nb = settings().geometry.potrf_nb;
    r.potrf_w = settings().geometry.potrf_w;
    if (const char* nb = std::getenv("BATCHLAS_POTRF_LPANEL_NB"); nb && *nb) r.potrf_lpanel_nb = std::atoi(nb);
    if (const char* st = std::getenv("BATCHLAS_STRATEGY")) r.strategy = st;
    if (const auto o = dispatch::routing_profile_override()) r.profile_override = static_cast<int>(*o);
    r.gemm_forced = env_route(Op::gemm);
    r.trsm_forced = env_route(Op::trsm);
    return r;
}

inline bool same_request(const Request& a, const Request& b) {
    return a.forced == b.forced && a.forced_name == b.forced_name && a.strict == b.strict &&
           a.potrf_nb == b.potrf_nb && a.potrf_w == b.potrf_w &&
           a.potrf_lpanel_nb == b.potrf_lpanel_nb && a.strategy == b.strategy &&
           a.profile_override == b.profile_override && a.gemm_forced == b.gemm_forced &&
           a.trsm_forced == b.trsm_forced;
}

inline std::string fingerprint(const Request& r) {
    auto rt = [](Route x) { return std::to_string(int(x.origin)) + "." + std::to_string(int(x.algo)); };
    return rt(r.forced) + "|" + r.forced_name + "|" + std::to_string(r.strict) + "|" +
           std::to_string(r.potrf_nb) + "|" + std::to_string(r.potrf_w) + "|" +
           std::to_string(r.potrf_lpanel_nb) + "|" + r.strategy + "|" +
           std::to_string(r.profile_override) + "|" + rt(r.gemm_forced) + "|" + rt(r.trsm_forced);
}

// ---- Exec: one specialisation per tier, NO primary definition ------------------------------

template <Backend B, class D>
struct Exec;

template <Backend B, class D>
concept Executable = requires { sizeof(Exec<B, D>); };

template <Backend B, class T>
using PView = MatrixView<T, MatrixFormat::Dense>;

template <Backend B, class T>
struct Exec<B, potrf::Tiny<T>> {
    static Event run(Queue& q, const Node&, const PView<B, T>& A, Uplo u, Span<std::byte> ws, Span<int32_t> info) {
        return sycl_potrf::potrf_tiny_dispatch<T>(q, A, u, ws, info);
    }
    static std::size_t bind(Queue& q, const Node&, const PView<B, T>& A, Uplo) {
        return sycl_potrf::potrf_tiny_buffer_size<T>(q, A);
    }
};

template <Backend B, class T>
struct Exec<B, potrf::Cta<T>> {
    static Event run(Queue& q, const Node& nd, const PView<B, T>& A, Uplo u, Span<std::byte> ws, Span<int32_t> info) {
        if constexpr (potrf::kMutant == 8) {   // re-derives a different scope/geometry
            return sycl_potrf::potrf_cta_dispatch_planned<T>(
                q, A, u, ws, info,
                potrf_plan::cta_wg_geometry<T>(static_cast<int>(A.rows()), A.batch_size(), hw_facts(q.device()).launch));
        }
        return sycl_potrf::potrf_cta_dispatch_planned<T>(q, A, u, ws, info,
                                                         std::any_cast<const potrf_plan::CtaGeometry&>(nd.geometry));
    }
    static std::size_t bind(Queue& q, const Node&, const PView<B, T>& A, Uplo) {
        return sycl_potrf::potrf_cta_buffer_size<T>(q, A);
    }
};

#ifndef BATCHLAS_PLAN_MOCK_DROP_CTA_WG_EXEC   // the extension's compile-time safety net
template <Backend B, class T>
struct Exec<B, potrf::CtaWg<T>> : Exec<B, potrf::Cta<T>> {};   // same launcher, the node's geometry
#endif

template <Backend B, class T>
struct Exec<B, potrf::LPanel<T>> {
    static Event run(Queue& q, const Node& nd, const PView<B, T>& A, Uplo u, Span<std::byte> ws, Span<int32_t> info) {
        return sycl_potrf::potrf_lpanel_dispatch<T>(q, A, u, ws, info, resident::kMinBlocksPerSm,
                                                    potrf::kMutant == 7 ? 0 : static_cast<int>(nd.param("NB")));
    }
    static std::size_t bind(Queue& q, const Node&, const PView<B, T>& A, Uplo) {
        return sycl_potrf::potrf_lpanel_buffer_size<T>(q, A);
    }
};

inline void log_child(const Node& parent, const Node& c, Backend b, ScalarKind dt) {
    if (!log_enabled()) return;
    LogRow r;
    r.path = std::string("potrf/") + std::string(parent.tier) + "/" + std::string(dispatch::op_name(c.op));
    r.route = route_name(c.route);
    r.tier = r.route;
    r.backend = b;
    r.dt = dt;
    r.m = c.m; r.n = c.n; r.k = c.k; r.batch = c.batch;
    log_row(std::move(r));
}

template <Backend B, class T>
struct Exec<B, potrf::Blocked<T>> {
    static Event run(Queue& q, const Node& nd, const PView<B, T>& A, Uplo u, Span<std::byte> ws, Span<int32_t> info) {
        constexpr ScalarKind dt = dispatch::scalar_kind_of<T>;
        return sycl_potrf::potrf_blocked_dispatch_planned<T>(
            q, A, u, ws, info, std::any_cast<const potrf_plan::BlockedParams&>(nd.geometry),
            [&nd](Queue& c, const PView<B, T>& ga, const PView<B, T>& gb, const PView<B, T>& gc,
                  T al, T be, Transpose ta, Transpose tb, ComputePrecision p) {
                const auto k = (ta == Transpose::NoTrans) ? ga.cols() : ga.rows();
                log_child(nd, nd.child(Op::gemm, gc.rows(), gc.cols(), k), B, dt);   // drift -> throw
                return gemm<B, T>(c, ga, gb, gc, al, be, ta, tb, p);
            },
            [&nd](Queue& c, const PView<B, T>& ta, const PView<B, T>& tb, T al, Side s, Uplo ul,
                  Transpose t, Diag d) {
                log_child(nd, nd.child(Op::trsm, tb.rows(), tb.cols(), ta.rows()), B, dt);
                return trsm<B, T>(c, ta, tb, al, s, ul, t, d);
            });
    }
    static std::size_t bind(Queue& q, const Node& nd, const PView<B, T>& A, Uplo) {
        return sycl_potrf::potrf_blocked_buffer_size_planned<T>(
            q, A, std::any_cast<const potrf_plan::BlockedParams&>(nd.geometry));
    }
};

template <Backend B, class T>
struct Exec<B, potrf::Vendor<T>> {
    static Event run(Queue& q, const Node&, const PView<B, T>& A, Uplo u, Span<std::byte> ws, Span<int32_t> info) {
        if constexpr (!dispatch::solver_vendor_available<B>) {
            dispatch::throw_no_vendor_route<T>(Op::potrf, B, dispatch::kSolverLibrary<B>);
        } else {
            return backend::potrf_vendor<B, T>(q, A, u, ws, info);
        }
    }
    // A vendor-free build binds 0 and lets run() throw NoRoute: PLANNING an unservable shape
    // (EXPLAIN, the equivalence grid, a buffer_size query) is not an error; running it is.
    static std::size_t bind(Queue& q, const Node&, const PView<B, T>& A, Uplo u) {
        if constexpr (!dispatch::solver_vendor_available<B>) {
            return 0;
        } else {
            return backend::potrf_vendor_buffer_size<B, T>(q, A, u);
        }
    }
};

template <Backend B, class... D>
constexpr bool all_executable(TierList<D...>*) { return (Executable<B, D> && ...); }

// ---- the facade-side driver ----------------------------------------------------------------

// Only what the plan reads: no stride, and ld only at batch == 1 (cuSOLVER's single-matrix
// bufferSize takes lda; nothing else in the tree reads ld or stride).
struct ShapeKey {
    std::int64_t rows = 0, cols = 0, batch = 0, ld = 0;
    int uplo = 0, dev_type = 0;
    bool hetero = false;
    std::size_t dev_idx = 0;
    bool operator==(const ShapeKey&) const = default;
};

template <Backend B, typename T>
ShapeKey shape_key(const Queue& q, const PView<B, T>& A, Uplo uplo) {
    const Device dev = q.device();
    ShapeKey k;
    k.rows = A.rows();
    k.cols = A.cols();
    k.batch = A.batch_size();
    k.ld = (A.batch_size() == 1) ? A.ld() : 0;
    k.uplo = static_cast<int>(uplo);
    k.dev_type = static_cast<int>(dev.type);
    k.hetero = A.is_heterogeneous();
    k.dev_idx = dev.idx;
    return k;
}

template <Backend B, typename T>
std::string potrf_cache_key(const ShapeKey& k, const Request& r) {
    // The ABI tag keeps a mutant TU's plans out of the library's entries (same shared cache).
    return "potrf|" BATCHLAS_PLAN_ABI_STR "|" + std::to_string(int(B)) + "|" + std::to_string(int(dispatch::scalar_kind_of<T>)) + "|" +
           std::to_string(k.uplo) + "|" + std::to_string(k.hetero) + "|" + std::to_string(k.rows) + "x" +
           std::to_string(k.cols) + "|" + std::to_string(k.batch) + "|" + std::to_string(k.ld) + "|" +
           std::to_string(k.dev_type) + "." + std::to_string(k.dev_idx) + "|" + fingerprint(r);
}

template <Backend B, typename T>
potrf::Shape potrf_shape(const PView<B, T>& A, Uplo uplo, const Facts& f, const Request& r) {
    return potrf::shape_of<T>(A.rows(), A.cols(), A.batch_size(), uplo, A.is_heterogeneous(), f,
                              r.potrf_lpanel_nb);
}

template <Backend B, typename T>
Node potrf_build(Queue& q, const PView<B, T>& A, Uplo uplo, const Request& r, bool vendor_free) {
    const Facts f = facts_for<B>(q, vendor_free ? false : dispatch::solver_vendor_available<B>);
    return potrf::plan_potrf<T>(potrf_shape<B, T>(A, uplo, f, r), f, r);
}

// The SIZING CONTRACT (main's, kept): buffer_size reports the vendor's size and, when the
// chosen tree is native, the max over the bound plans of EVERY legal native tier, because
// options.hh re-reads the route pin between the size query and the call. Each of those plans is exact (Blocked's W is
// clamped to n - nb), so this is a max over real trees, not the 512 MiB over-estimate.
// potrf() itself requires only the CHOSEN tree's workspace().
template <Backend B, typename T>
Node potrf_build_bound(Queue& q, const PView<B, T>& A, Uplo uplo, const Request& r) {
    const Facts f = facts_for<B>(q, dispatch::solver_vendor_available<B>);
    const potrf::Shape s = potrf_shape<B, T>(A, uplo, f, r);
    Node nd = potrf::plan_potrf<T>(s, f, r);
    auto bind = [&](const Node& x) {
        return potrf::Tiers<T>::visit(x.tier, [&]<class D>() { return Exec<B, D>::bind(q, x, A, uplo); });
    };
    nd.own_ws = bind(nd);
    nd.bound = true;
    nd.cover_ws = nd.workspace();
    if (dispatch::is_native(nd.route)) {
        for (const Candidate& c : nd.decision.candidates) {
            if (!c.legal.ok || !dispatch::is_native(c.route) || c.name == nd.tier) continue;
            Node other = plan_tier<potrf::Tiers<T>>(c.name, s, f, r);
            other.own_ws = bind(other);
            nd.cover_ws = std::max(nd.cover_ws, other.workspace());
        }
    }
    if constexpr (dispatch::solver_vendor_available<B>) {   // main sizes the vendor arm always
        nd.cover_ws = std::max(nd.cover_ws, Exec<B, potrf::Vendor<T>>::bind(q, nd, A, uplo));
    }
    return nd;
}

template <Backend B, typename T>
std::shared_ptr<const Node> potrf_plan_cached(Queue& q, const PView<B, T>& A, Uplo uplo) {
    static_assert(all_executable<B>(static_cast<potrf::Tiers<T>*>(nullptr)),
                  "a potrf tier has no Exec<B, Tier> specialisation");
    struct Memo {
        std::uint64_t gen = 0;
        ShapeKey shape;
        Request req;
        std::shared_ptr<const Node> node;
    };
    thread_local Memo memo;   // the lock-free hit path: buffer_size and its potrf() share it
    const Request r = request_for_potrf();
    const ShapeKey sk = shape_key<B, T>(q, A, uplo);
    const std::uint64_t gen = cache_generation();
    if (memo.node && memo.gen == gen && memo.shape == sk && same_request(memo.req, r)) {
        cache_note_hit();
        return memo.node;
    }
    const std::string key = potrf_cache_key<B, T>(sk, r);
    std::shared_ptr<const Node> hit = cache_find(key);
    if (!hit) {
        Node nd = potrf_build_bound<B, T>(q, A, uplo, r);
        if (const char* e = std::getenv("BATCHLAS_EXPLAIN"); e && *e == '1') {
            std::fprintf(stderr, "%s", explain(nd).c_str());
        }
        hit = cache_insert(key, std::move(nd));
    }
    memo = Memo{gen, sk, r, hit};
    return hit;
}

template <Backend B, typename T>
void record_tree(const Node& nd) {
    if (dispatch::coverage::dynamic_enabled()) {   // the existing choke point, one row per root
        dispatch::OpShape row;
        row.op = Op::potrf;
        row.scalar = dispatch::scalar_kind_of<T>;
        row.backend = B;
        row.m = nd.m; row.n = nd.n; row.k = nd.m; row.batch = nd.batch;
        row.cost_extrapolated = -1;
        bool native_supported = false;
        for (const Candidate& c : nd.decision.candidates) {
            if (dispatch::is_native(c.route) && c.legal.ok) native_supported = true;
            if (nd.decision.chooser == "cost-model" && c.name == nd.tier && c.seconds >= 0) {
                row.cost_extrapolated = c.extrapolated ? 1 : 0;
            }
        }
        dispatch::coverage::record_if_enabled(Op::potrf, row.scalar, B, row, nd.route, true, native_supported);
    }
    if (!log_enabled()) return;
    LogRow r;
    r.path = "potrf";
    r.tier = std::string(nd.tier);
    r.route = route_name(nd.route);
    r.forced = nd.decision.forced_name.empty()
                   ? (nd.decision.forced.origin == Origin::Auto ? "" : route_name(nd.decision.forced))
                   : nd.decision.forced_name;
    r.honoured = nd.decision.honoured;
    r.why = nd.decision.why;
    r.backend = B;
    r.dt = dispatch::scalar_kind_of<T>;
    r.m = nd.m; r.n = nd.n; r.batch = nd.batch;
    log_row(r);
    for (const Node& c : nd.children) {
        if (c.op != Op::potrf) continue;   // gemm/trsm children are logged when they are ISSUED
        LogRow lr = r;
        lr.path = "potrf/" + std::string(nd.tier) + "/leaf";
        lr.tier = std::string(c.tier);
        lr.route = route_name(c.route);
        lr.forced.clear();
        lr.m = c.m; lr.n = c.n;
        log_row(lr);
    }
}

template <Backend B, typename T>
std::size_t potrf_planned_buffer_size(Queue& q, const PView<B, T>& A, Uplo uplo) {
    const std::shared_ptr<const Node> nd = potrf_plan_cached<B, T>(q, A, uplo);
    if constexpr (!dispatch::solver_vendor_available<B>) {
        if (dispatch::is_vendor(nd->route)) {   // main's contract: an unservable size query throws
            dispatch::throw_no_vendor_route<T>(Op::potrf, B, dispatch::kSolverLibrary<B>);
        }
    }
    return nd->cover_ws;
}

template <Backend B, typename T>
Event potrf_planned(Queue& q, const PView<B, T>& A, Uplo uplo, Span<std::byte> ws, Span<int32_t> info) {
    const std::shared_ptr<const Node> nd = potrf_plan_cached<B, T>(q, A, uplo);
    if (ws.size() < nd->workspace()) {
        throw workspace_error("potrf: the plan (" + std::string(nd->tier) + ") needs " +
                              std::to_string(nd->workspace()) + " B of workspace, got " +
                              std::to_string(ws.size()) + " B");
    }
    record_tree<B, T>(*nd);
    return potrf::Tiers<T>::visit(nd->tier, [&]<class D>() {
        return Exec<B, D>::run(q, *nd, A, uplo, ws, info);
    });
}

}  // inline namespace BATCHLAS_PLAN_ABI
}  // namespace batchlas::plan
