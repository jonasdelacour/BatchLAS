#pragma once

// potrf as tier descriptors run by routing::select: legal() is RouteTable::supports verbatim,
// plan() is the launch-plan geometry, launch() is today's dispatcher. Nothing here prices a
// route; the policy is the generated RuleSet. See evaluation/routing/compile_rules.py.

#include <batchlas/routing/rules.hh>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/device_facts.hh>
#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/error.hh>

#include "../backends/potrf_route.hh"
#include "../extensions/potrf_launch_plan.hh"
#include "../extensions/potrf_native.hh"
#include "../sycl/trsm_native.hh"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include "generated/potrf_rules.inc"

namespace batchlas::routing::potrf {
using dispatch::Algorithm;
using dispatch::Origin;
using launch_plan::DeviceFacts;

template <class T>
using Tbl = dispatch::RouteTable<dispatch::Op::potrf, T>;
template <class T>
using View = MatrixView<T, MatrixFormat::Dense>;

template <class T>
struct Ctx {
    dispatch::PotrfShape s;   // ceilings only: model_cost / model_enabled stay unset
    DeviceFacts d;
    int leaf_trsm = 0;
    int nb_env = 0, w_env = 0;
    bool vendor_legal = true;   // false = the vendor-free walk
    Queue* q = nullptr;         // null: offline (no workspace query)
    const View<T>* A = nullptr;
    bool size_ws = true;

    std::uint16_t key() const {
        return static_cast<std::uint16_t>(backend::potrf_dtype_index<T>() * 2 +
                                          (s.uplo == Uplo::Upper ? 1 : 0));
    }
    Features features() const { return {s.order(), s.batch, 0, 0}; }
};

template <class T>
struct Call {
    Queue& q;
    const View<T>& A;
    Uplo uplo;
    Span<std::byte> ws;
    Span<int32_t> info;
};

[[noreturn]] inline void plan_mismatch(const char* tier, unsigned want, unsigned got) {
    throw batchlas::internal_error(std::string("routing: ") + tier +
                                   " dispatcher geometry differs from the Selection's plan (" +
                                   std::to_string(want) + " vs " + std::to_string(got) + ")");
}

template <class T>
struct TinyTier {
    static constexpr std::string_view id = "native:tiny";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::Tiny};
    using Plan = potrf_plan::TinyGeometry;
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>& c, Knobs) {
        return potrf_plan::tiny_geometry<T>(static_cast<int>(c.s.order()), c.s.batch,
                                            c.d.max_wg_size);
    }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return (c.q && c.size_ws) ? sycl_potrf::potrf_tiny_buffer_size<T>(*c.q, *c.A) : 0;
    }
    static Event launch(const Plan& p, Call<T>& k) {
        const int S = p.wg_size / 32;
        const unsigned want = (static_cast<unsigned>(S) << 16) | static_cast<unsigned>(p.per_wg / S);
        const unsigned got = sycl_potrf::potrf_tiny_debug_launch<T>(k.q, static_cast<int>(k.A.rows()));
        if (!p.fits || want != got) plan_mismatch("native:tiny", want, got);
        return sycl_potrf::potrf_tiny_dispatch<T>(k.q, k.A, k.uplo, k.ws, k.info);
    }
};

template <class T>
struct CtaTier {
    static constexpr std::string_view id = "native:cta";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::CTA};
    using Plan = potrf_plan::CtaGeometry;
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>& c, Knobs) {
        return potrf_plan::cta_geometry<T>(static_cast<int>(c.s.order()), c.s.batch, c.d);
    }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return (c.q && c.size_ws) ? sycl_potrf::potrf_cta_buffer_size<T>(*c.q, *c.A) : 0;
    }
    static Event launch(const Plan& p, Call<T>& k) {
        const unsigned want = static_cast<unsigned>(p.G) | (static_cast<unsigned>(p.L) << 16);
        const unsigned got = sycl_potrf::potrf_cta_debug_launch<T>(
            k.q, static_cast<int>(k.A.rows()), static_cast<int>(k.A.batch_size()));
        if (!p.fits || want != got) plan_mismatch("native:cta", want, got);
        return sycl_potrf::potrf_cta_dispatch<T>(k.q, k.A, k.uplo, k.ws, k.info);
    }
};

// CTA at work-group scope. No RouteTable arm exists for it: legal() is CTA's correctness gate
// plus this tier's own geometry, and coverage records it under CTA's Route.
template <class T>
struct CtaWgTier {
    static constexpr std::string_view id = "native:cta_wg";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::CTA};
    using Plan = potrf_plan::CtaGeometry;
    static Plan plan(const Ctx<T>& c, Knobs) {
        return potrf_plan::cta_wg_geometry<T>(static_cast<int>(c.s.order()), c.s.batch, c.d);
    }
    static bool legal(const Ctx<T>& c) {
        return Tbl<T>::supports(CtaTier<T>::route, c.s) && plan(c, {}).fits;
    }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return (c.q && c.size_ws) ? sycl_potrf::potrf_cta_buffer_size<T>(*c.q, *c.A) : 0;
    }
    static Event launch(const Plan& p, Call<T>& k) {
        const unsigned want = static_cast<unsigned>(p.G) | (static_cast<unsigned>(p.L) << 16);
        const unsigned got = sycl_potrf::potrf_cta_wg_debug_launch<T>(
            k.q, static_cast<int>(k.A.rows()), static_cast<int>(k.A.batch_size()));
        if (!p.fits || want != got) plan_mismatch("native:cta_wg", want, got);
        return sycl_potrf::potrf_cta_wg_dispatch<T>(k.q, k.A, k.uplo, k.ws, k.info);
    }
};

// Knob 0 = nb (0: the type's default NB).
template <class T>
struct LpanelTier {
    static constexpr std::string_view id = "native:lpanel";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::LPanel};
    using Plan = potrf_plan::LpanelGeometry;
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>& c, Knobs k) {
        return potrf_plan::lpanel_geometry<T>(static_cast<int>(c.s.order()), c.s.batch, c.d,
                                              resident::kMinBlocksPerSm, k.v[0]);
    }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return (c.q && c.size_ws) ? sycl_potrf::potrf_lpanel_buffer_size<T>(*c.q, *c.A) : 0;
    }
    static Event launch(const Plan& p, Call<T>& k) {
        const unsigned want = (static_cast<unsigned>(p.nb) << 16) |
                              (static_cast<unsigned>(p.L) << 4) | static_cast<unsigned>(p.G);
        const unsigned got = sycl_potrf::potrf_lpanel_debug_launch<T>(
            k.q, static_cast<int>(k.A.rows()), static_cast<int>(k.A.batch_size()),
            resident::kMinBlocksPerSm, p.nb);
        if (!p.fits || want != got) plan_mismatch("native:lpanel", want, got);
        return sycl_potrf::potrf_lpanel_dispatch<T>(k.q, k.A, k.uplo, k.ws, k.info,
                                                    resident::kMinBlocksPerSm, p.nb);
    }
};

// Child selections of the Blocked tier: one row per sub-op call, with the parent named.
inline void record_child(std::string_view parent, std::string_view op, Backend b, std::int64_t m,
                  std::int64_t n, std::int64_t k, std::int64_t batch);

// Knobs: 0 = nb, 1 = W (0: the env override, else the type's default).
template <class T>
struct BlockedPlan {
    potrf_plan::BlockedParams p{};
    potrf_plan::CtaGeometry leaf{};
    bool fits = false;
};

template <Backend B, class T>
struct BlockedTier {
    static constexpr std::string_view id = "native:blocked";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::Blocked};
    using Plan = BlockedPlan<T>;
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>& c, Knobs k) {
        const int n = static_cast<int>(c.s.order());
        Plan out;
        out.p = potrf_plan::blocked_params<T>(n, c.d.local_mem_bytes, c.leaf_trsm,
                                              k.v[0] ? k.v[0] : c.nb_env,
                                              k.v[1] ? k.v[1] : c.w_env);
        out.leaf = potrf_plan::cta_geometry<T>(std::min(out.p.nb, n), c.s.batch, c.d,
                                               out.p.leaf_min_blocks);
        out.fits = out.leaf.fits;
        return out;
    }
    // The driver's own layout (W unclamped: the driver reads W from its statics).
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return (c.q && c.size_ws) ? sycl_potrf::potrf_blocked_buffer_size<T>(*c.q, *c.A, c.s.uplo)
                                  : 0;
    }
    static Event launch(const Plan& p, Call<T>& k) {
        const unsigned want = static_cast<unsigned>(p.p.nb) | (static_cast<unsigned>(p.p.W) << 16);
        const unsigned got =
            sycl_potrf::potrf_blocked_debug_params<T>(k.q, static_cast<int>(k.A.rows()));
        if (!p.fits || want != got) plan_mismatch("native:blocked", want, got);
        return sycl_potrf::potrf_blocked_dispatch<T>(
            k.q, k.A, k.uplo, k.ws, k.info,
            [](Queue& c, const View<T>& ga, const View<T>& gb, const View<T>& gc, T al, T be,
               Transpose ta, Transpose tb, ComputePrecision pr) {
                record_child("potrf:native:blocked", "gemm", B, gc.rows(), gc.cols(),
                             ta == Transpose::NoTrans ? ga.cols() : ga.rows(), gc.batch_size());
                return gemm<B, T>(c, ga, gb, gc, al, be, ta, tb, pr);
            },
            [](Queue& c, const View<T>& ta, const View<T>& tb, T al, Side sd, Uplo ul,
               Transpose tr, Diag dg) {
                record_child("potrf:native:blocked", "trsm", B, tb.rows(), tb.cols(), ta.rows(),
                             tb.batch_size());
                return trsm<B, T>(c, ta, tb, al, sd, ul, tr, dg);
            });
    }
};

template <Backend B, class T>
struct VendorTier {
    static constexpr std::string_view id = "vendor";
    static constexpr dispatch::Route route{Origin::Vendor, Algorithm::Auto};
    struct Plan { bool fits = true; };
    static bool legal(const Ctx<T>& c) {
        return dispatch::solver_vendor_available<B> && c.vendor_legal;
    }
    static Plan plan(const Ctx<T>&, Knobs) { return {}; }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        if constexpr (dispatch::solver_vendor_available<B>) {
            if (c.q && c.size_ws) return backend::potrf_vendor_buffer_size<B, T>(*c.q, *c.A, c.s.uplo);
        }
        return 0;
    }
    static Event launch(const Plan&, Call<T>& k) {
        if constexpr (dispatch::solver_vendor_available<B>) {
            return backend::potrf_vendor<B, T>(k.q, k.A, k.uplo, k.ws, k.info);
        } else {
            throw batchlas::internal_error("potrf: vendor launched although legal() is false");
        }
    }
};

template <Backend B, class T>
using Tiers = TierList<TinyTier<T>, CtaTier<T>, LpanelTier<T>, BlockedTier<B, T>, VendorTier<B, T>,
                       CtaWgTier<T>>;

template <Backend B, class T>
inline constexpr auto kBound = bind<Tiers<B, T>>(potrf_rules::kNames);

// Offline context from described facts: what tests and tools use instead of a Queue.
template <Backend B, class T>
Ctx<T> ctx_from_facts(const DeviceFacts& d, std::int64_t n, std::int64_t batch, Uplo uplo,
                      arch::RoutingProfile profile, bool vendor_legal) {
    Ctx<T> c;
    c.s = backend::potrf_op_shape_from_facts<B, T>(d, n, n, batch, uplo, true, true, false,
                                                   sycl_potrf::potrf_blocked_available<T>());
    c.s.profile = profile;
    c.d = d;
    c.leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    c.vendor_legal = vendor_legal;
    return c;
}

struct MemoFacts {
    DeviceFacts d;
    bool has_sg32 = false;
};

// Device facts are memoized per device; a routed call queries nothing twice.
inline MemoFacts memo_facts(const Queue& q) {
    static auto* mu = new std::mutex();
    static auto* memo = new std::map<std::pair<int, std::size_t>, MemoFacts>();
    const Device dev = q.device();
    const auto key = std::make_pair(static_cast<int>(dev.type), dev.idx);
    {
        std::lock_guard<std::mutex> lock(*mu);
        if (auto it = memo->find(key); it != memo->end()) return it->second;
    }
    MemoFacts f;
    f.d = sycl_potrf::potrf_device_facts(dev);
    f.d.compute_units = dispatch::device_facts(dev).compute_units;
    f.has_sg32 = dev.supports_sub_group_size(32);
    std::lock_guard<std::mutex> lock(*mu);
    return memo->emplace(key, f).first->second;
}

template <Backend B, class T>
Ctx<T> ctx_from_queue(Queue& q, const View<T>& A, Uplo uplo, bool vendor_legal, bool size_ws) {
    const MemoFacts f = memo_facts(q);
    Ctx<T> c;
    c.s = backend::potrf_op_shape_from_facts<B, T>(
        f.d, A.rows(), A.cols(), A.batch_size(), uplo, q.device().type == DeviceType::GPU,
        f.has_sg32, A.is_heterogeneous(), sycl_potrf::potrf_blocked_available<T>());
    dispatch::fill_device_facts(c.s, q);
    c.d = f.d;
    c.leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    sycl_potrf::potrf_blocked_overrides(c.nb_env, c.w_env);
    c.vendor_legal = vendor_legal;
    c.q = &q;
    c.A = &A;
    c.size_ws = size_ws;
    return c;
}

// The pin grammar: a tier id, or today's BATCHLAS_POTRF_ROUTE words mapped onto tier ids.
template <class Tiers>
Pin pin_from_env() {
    const char* raw = std::getenv("BATCHLAS_POTRF_ROUTE");
    const char* st = std::getenv("BATCHLAS_ROUTING_STRICT");
    const bool strict = st && *st && *st != '0';
    Pin p = parse_pin<Tiers>(raw, strict);
    if (!p.active || p.tier >= 0 || p.origin_only) return p;
    if (const auto r = dispatch::parse_route_value(p.text)) {
        if (r->origin == Origin::Auto && r->algo == Algorithm::Auto) return Pin{};
        if (r->algo == Algorithm::Auto && r->origin == Origin::Native) {
            p.origin_only = true;
            p.origin = Origin::Native;
            return p;
        }
        for (std::size_t i = 0; i < Tiers::size; ++i) {
            if (Tiers::routes[i] == *r) p.tier = static_cast<int>(i);
        }
    }
    return p;
}

template <Backend B, class T>
Selection<Tiers<B, T>> select_ctx(const Ctx<T>& c, const Pin& pin) {
    return routing::select<Tiers<B, T>>(potrf_rules::rules_for_profile(c.s.profile), kBound<B, T>,
                                        c, pin);
}

inline void record_explain(std::string_view line);

template <Backend B, class T>
void record(const Ctx<T>& c, const Selection<Tiers<B, T>>& sel) {
    using Ts = Tiers<B, T>;
    if (dispatch::coverage::dynamic_enabled()) {
        bool native_ok = false;
        for (std::size_t t = 0; t < Ts::size; ++t) {
            if (dispatch::is_native(Ts::routes[t]) && Ts::legal(t, c)) native_ok = true;
        }
        dispatch::coverage::record_if_enabled(dispatch::Op::potrf, dispatch::scalar_kind_of<T>, B,
                                              static_cast<const dispatch::OpShape&>(c.s),
                                              sel.route(), true, native_ok ? 1 : 0);
    }
    if (std::getenv("BATCHLAS_ROUTING_EXPLAIN_OUT")) {
        char buf[512];
        std::snprintf(buf, sizeof buf, "potrf,%d,%d,%lld,%lld,%c,%s,%s,R%04u,%s,%s,%s,%zu,",
                      backend::potrf_dtype_index<T>(), static_cast<int>(B),
                      static_cast<long long>(c.s.order()), static_cast<long long>(c.s.batch),
                      c.s.uplo == Uplo::Upper ? 'U' : 'L', std::string(sel.id()).c_str(),
                      std::string(to_string(sel.reason)).c_str(), sel.rule_id,
                      std::string(to_string(sel.source)).c_str(),
                      sel.rules ? std::string(sel.rules->arch).c_str() : "-",
                      sel.forced.empty() ? "-" : sel.forced.c_str(), sel.ws_bytes);
        record_explain(buf);
    }
}

template <Backend B, class T>
Selection<Tiers<B, T>> potrf_select(Queue& q, const View<T>& A, Uplo uplo,
                                    bool vendor_legal = dispatch::solver_vendor_available<B>,
                                    bool size_ws = true) {
    const Ctx<T> c = ctx_from_queue<B, T>(q, A, uplo, vendor_legal, size_ws);
    return select_ctx<B, T>(c, pin_from_env<Tiers<B, T>>());
}

template <Backend B, class T>
std::size_t potrf_rules_buffer_size(Queue& q, const View<T>& A, Uplo uplo) {
    potrf_validate_params<T>(A, uplo);
    return potrf_select<B, T>(q, A, uplo).ws_bytes;
}

template <Backend B, class T>
Event potrf_rules(Queue& q, const View<T>& A, Uplo uplo, Span<std::byte> ws,
                  Span<int32_t> info = {}) {
    potrf_validate_params<T>(A, uplo);
    const Ctx<T> c = ctx_from_queue<B, T>(q, A, uplo, dispatch::solver_vendor_available<B>, false);
    const auto sel = select_ctx<B, T>(c, pin_from_env<Tiers<B, T>>());
    record<B, T>(c, sel);
    Call<T> call{q, A, uplo, ws, info};
    return Tiers<B, T>::template launch<Event>(sel.plan, call);
}

// Out-of-line state for the two loggers; one definition per program.
inline std::mutex& log_mutex() {
    static auto* mu = new std::mutex();
    return *mu;
}
inline void record_explain(std::string_view line) {
    const char* path = std::getenv("BATCHLAS_ROUTING_EXPLAIN_OUT");
    if (!path || !*path) return;
    std::lock_guard<std::mutex> lock(log_mutex());
    if (FILE* f = std::fopen(path, "a")) {
        std::fprintf(f, "%.*s\n", static_cast<int>(line.size()), line.data());
        std::fclose(f);
    }
}
inline void record_child(std::string_view parent, std::string_view op, Backend b, std::int64_t m,
                         std::int64_t n, std::int64_t k, std::int64_t batch) {
    if (!std::getenv("BATCHLAS_ROUTING_EXPLAIN_OUT")) return;
    char buf[256];
    std::snprintf(buf, sizeof buf, "%.*s,child,%d,%lld,%lld,%lld,%lld,parent=%.*s",
                  static_cast<int>(op.size()), op.data(), static_cast<int>(b),
                  static_cast<long long>(m), static_cast<long long>(n), static_cast<long long>(k),
                  static_cast<long long>(batch), static_cast<int>(parent.size()), parent.data());
    record_explain(buf);
}

}  // namespace batchlas::routing::potrf
