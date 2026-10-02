#pragma once

// potrf's tiers as descriptors over the REAL plan functions (potrf_launch_plan.hh) and the
// generated profile. SYCL-free: Exec<B, Tier> (potrf_exec.hh) is the SYCL side.

#include "planner.hh"

#include <batchlas/blas/dispatch/route_gemm.hh>
#include <batchlas/blas/dispatch/route_trsm.hh>

#include "../backends/potrf_profile_constants.hh"
#include "../extensions/potrf_launch_plan.hh"
#include "../extensions/potrf_native.hh"
#include "../sycl/trsm_native.hh"

#include <complex>
#include <limits>
#include <type_traits>

namespace batchlas::plan::potrf {
inline namespace BATCHLAS_PLAN_ABI {   // plan.hh: a mutant TU must not interpose the library's copy

inline constexpr int kMutant = BATCHLAS_PLAN_MUTANT;

struct Shape {
    std::int64_t m = 0, n = 0, batch = 0;   // m != n is representable so the gate can fire
    Uplo uplo = Uplo::Lower;
    bool hetero = false;
    int tiny_max = 0, cta_max = 0, lpanel_max = 0;
    int lpanel_nb = 0;                      // the LPanel block hint (0 = default); lpanel_max is AT it
    bool blocked_built = false;
    int leaf_trsm = 0;
    std::int64_t order() const { return m; }
};

constexpr std::uint8_t mode_of(Uplo u) { return u == Uplo::Upper ? 1 : 0; }

template <class T>
Shape shape_of(std::int64_t rows, std::int64_t cols, std::int64_t batch, Uplo uplo, bool hetero,
               const Facts& f, int lpanel_nb = 0) {
    Shape s;
    s.m = rows;
    s.n = cols;
    s.batch = batch;
    s.uplo = uplo;
    s.hetero = hetero;
    s.lpanel_nb = lpanel_nb;
    const std::size_t budget = resident::device_slm_budget(f.launch.local_mem_bytes);
    s.cta_max = sycl_potrf::potrf_cta_max_n_for_slm<T>(budget);
    s.lpanel_max = potrf_plan::lpanel_nb_is_built<T>(potrf_plan::lpanel_nb_for<T>(lpanel_nb))
                       ? sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, f.launch.max_wg_size,
                                                                   resident::kMinBlocksPerSm, lpanel_nb)
                       : 0;
    s.tiny_max = sycl_potrf::potrf_tiny_max_n<T>();
    s.blocked_built = sycl_potrf::potrf_blocked_available<T>();
    s.leaf_trsm = sycl_trsm::trsm_cta_max_n<T>();
    return s;
}

inline Legality common(const Shape& s, const Facts& f) {
    if (s.m != s.n) return Legality::no("non-square");
    if (!f.is_gpu) return Legality::no("not a GPU");
    if (!f.has_sg32) return Legality::no("no sub-group size 32");
    if (s.hetero) return Legality::no("heterogeneous batch");
    if (s.order() < 1 || s.batch < 1) return Legality::no("empty");
    return Legality::yes();
}

template <class T>
potrf_plan::KernelRegs regs(const Facts& f) {
    constexpr int ti = static_cast<int>(dispatch::scalar_kind_of<T>);
    const auto* p = potrf_profile::profile_for(f.arch);
    return p ? p->regs[ti] : potrf_plan::KernelRegs{};
}

inline Node make_node(Route r, std::string_view kernel, const Shape& s) {
    Node nd;
    nd.op = Op::potrf;
    nd.route = r;
    nd.kernel = kernel;
    nd.m = s.m;
    nd.n = s.n;
    nd.batch = s.batch;
    return nd;
}

template <class T> struct Tiny {
    static constexpr Route route{Origin::Native, Algorithm::Tiny};
    static constexpr std::string_view name = "native:tiny";
    static Legality legal(const Shape& s, const Facts& f) {
        if (auto c = common(s, f); !c.ok) return c;
        if (s.tiny_max < 1) return Legality::no("tier not built");
        return s.order() <= s.tiny_max ? Legality::yes() : Legality::no("n > tiny_max_n");
    }
    static Node sketch(const Shape& s, const Facts& f, const Request&) {
        const int n = static_cast<int>(s.order());
        const auto g = potrf_plan::tiny_geometry<T>(n, s.batch, f.launch.max_wg_size);
        Node nd = make_node(route, "PotrfTinyKernel<T,N>", s);
        nd.launch = potrf_plan::tiny_plan<T>(n, s.batch, f.launch, regs<T>(f));
        nd.geometry = g;
        nd.set("N", g.N).set("per_wg", g.per_wg);
        return nd;
    }
    static void expand(Node&, const Shape&, const Facts&, const Request&) {}
};

template <class T> struct Cta {
    static constexpr Route route{Origin::Native, Algorithm::CTA};
    static constexpr std::string_view name = "native:cta";
    static Legality legal(const Shape& s, const Facts& f) {
        if (auto c = common(s, f); !c.ok) return c;
        if (s.cta_max < 1) return Legality::no("tier not built / no SLM");
        return s.order() <= s.cta_max ? Legality::yes() : Legality::no("n > cta_max_n");
    }
    static Node sketch_at(const Shape& s, const Facts& f, int min_blocks) {
        const int n = static_cast<int>(s.order());
        const auto g = potrf_plan::cta_geometry<T>(n, s.batch, f.launch, min_blocks);
        Node nd = make_node(route, g.subgroup_scope ? "PotrfCtaKernel<T,NB,TS,SubGroup>"
                                                    : "PotrfCtaKernel<T,NB,TS,WorkGroup>", s);
        nd.tier = name;
        nd.launch = potrf_plan::cta_plan<T>(n, s.batch, f.launch, regs<T>(f), min_blocks);
        nd.geometry = g;
        nd.set("L", g.L).set("G", g.G).set("min_blocks", min_blocks);
        return nd;
    }
    static Node sketch(const Shape& s, const Facts& f, const Request&) {
        return sketch_at(s, f, resident::kMinBlocksPerSm);
    }
    static void expand(Node&, const Shape&, const Facts&, const Request&) {}
};

// EXTENSION TIER: the CTA kernel at work-group scope, one matrix per group, L >= 64. Not
// measured anywhere, so Auto never picks it; a name pin or a strategy override can.
template <class T> struct CtaWg {
    static constexpr Route route{Origin::Native, Algorithm::CTA};
    static constexpr bool owns_route = false;   // `native:cta` (pin, coverage) means Cta<T>
    static constexpr std::string_view name = "native:cta_wg";
    static Legality legal(const Shape& s, const Facts& f) {
        if (auto c = Cta<T>::legal(s, f); !c.ok) return c;
        const auto g = potrf_plan::cta_wg_geometry<T>(static_cast<int>(s.order()), s.batch, f.launch);
        return g.fits ? Legality::yes() : Legality::no("work-group geometry does not fit");
    }
    static Node sketch(const Shape& s, const Facts& f, const Request&) {
        const int n = static_cast<int>(s.order());
        const auto g = potrf_plan::cta_wg_geometry<T>(n, s.batch, f.launch);
        Node nd = make_node(route, "PotrfCtaKernel<T,NB,TS,WorkGroup>", s);
        nd.launch = potrf_plan::cta_plan_from<T>(g, n, s.batch, f.launch, regs<T>(f));
        nd.geometry = g;
        nd.set("L", g.L).set("G", g.G);
        return nd;
    }
    static void expand(Node&, const Shape&, const Facts&, const Request&) {}
};

template <class T> struct LPanel {
    static constexpr Route route{Origin::Native, Algorithm::LPanel};
    static constexpr std::string_view name = "native:lpanel";
    static Legality legal(const Shape& s, const Facts& f) {
        if (auto c = common(s, f); !c.ok) return c;
        if (kMutant != 4 && s.uplo != Uplo::Lower) return Legality::no("Upper (the update reads the lower triangle)");
        if (s.lpanel_max < 1) return Legality::no("tier not built");
        return s.order() <= s.lpanel_max ? Legality::yes() : Legality::no("n > lpanel_max_n");
    }
    static Node sketch(const Shape& s, const Facts& f, const Request&) {
        const int n = static_cast<int>(s.order());
        const auto g = potrf_plan::lpanel_geometry<T>(n, s.batch, f.launch, resident::kMinBlocksPerSm, s.lpanel_nb);
        Node nd = make_node(route, "PotrfLpanelKernel<T,NB>", s);
        nd.launch = potrf_plan::lpanel_plan<T>(n, s.batch, f.launch, regs<T>(f), resident::kMinBlocksPerSm,
                                               s.lpanel_nb);
        nd.geometry = g;
        nd.set("NB", g.nb).set("L", g.L).set("G", g.G);
        return nd;
    }
    static void expand(Node&, const Shape&, const Facts&, const Request&) {}
};

template <class T>
constexpr Transpose kTrailingTransB =
    potrf_plan::kIsComplex<T> ? Transpose::ConjTrans : Transpose::Trans;

inline void add_unique(Node& parent, Node child) {
    if (!parent.find_child(child.op, child.m, child.n, child.k)) parent.children.push_back(std::move(child));
}

inline void sub_op_facts(dispatch::OpShape& o, const Facts& f) {
    o.is_gpu = f.is_gpu;
    o.max_sub_group = f.max_sub_group;
    o.compute_units = f.launch.compute_units;
    o.cuda_cc = f.cuda_cc;
    o.profile = f.arch;
    o.profile_nearest = f.arch_nearest;
}

// The panel solve and the trailing gemms, resolved through their OWN tables, under their own pins.
template <class T>
Node plan_trsm(std::int64_t m2, std::int64_t ib, std::int64_t batch, const Facts& f, const Request& r) {
    dispatch::TrsmShape t;
    t.op = Op::trsm;
    t.scalar = dispatch::scalar_kind_of<T>;
    t.backend = f.backend;
    t.m = m2; t.n = ib; t.k = ib; t.batch = batch;
    t.side = Side::Right; t.uplo = Uplo::Lower; t.transA = Transpose::ConjTrans; t.diag = Diag::NonUnit;
    sub_op_facts(t, f);
    t.cta_max_n = sycl_trsm::trsm_cta_max_n<T>();
    t.blocked_available = sycl_trsm::trsm_blocked_available<T>();
    Node c;
    c.op = Op::trsm;
    c.route = dispatch::resolve_route_uninstrumented<Op::trsm, T>(r.trsm_forced, t, f.level3_vendor_available);
    c.kernel = "routed trsm<B,T>";
    c.m = m2; c.n = ib; c.k = ib; c.batch = batch;
    return c;
}

template <class T>
Node plan_gemm(std::int64_t m, std::int64_t n, std::int64_t k, std::int64_t batch, const Facts& f,
               const Request& r) {
    dispatch::OpShape g;
    g.op = Op::gemm;
    g.scalar = dispatch::scalar_kind_of<T>;
    g.backend = f.backend;
    g.m = m; g.n = n; g.k = k; g.batch = batch;
    g.transA = Transpose::NoTrans; g.transB = kTrailingTransB<T>;
    sub_op_facts(g, f);
    Node c;
    c.op = Op::gemm;
    c.route = dispatch::resolve_route_uninstrumented<Op::gemm, T>(r.gemm_forced, g, f.level3_vendor_available);
    c.kernel = "routed gemm<B,T>";
    c.m = m; c.n = n; c.k = k; c.batch = batch;
    return c;
}

template <class T> struct Blocked {
    static constexpr Route route{Origin::Native, Algorithm::Blocked};
    static constexpr std::string_view name = "native:blocked";
    static Legality legal(const Shape& s, const Facts& f) {
        if (auto c = common(s, f); !c.ok) return c;
        if (s.uplo != Uplo::Lower) return Legality::no("Upper (the driver is Lower-only)");
        if (!s.blocked_built || s.cta_max < 1) return Legality::no("tier not built / no CTA leaf");
        return Legality::yes();
    }
    static Node sketch(const Shape& s, const Facts& f, const Request& r) {
        const int n = static_cast<int>(s.order());
        auto bp = potrf_plan::blocked_params<T>(n, f.launch.local_mem_bytes, s.leaf_trsm, r.potrf_nb, r.potrf_w);
        const int w_cfg = bp.W;
        bp.W = std::max(1, std::min(bp.W, n - bp.nb));   // W <= n - nb: the 512 MiB sizing fix
        Node nd = make_node(route, "potrf_blocked driver", s);
        // Priced at the configured W: the fitted constants saw that fill term.
        nd.launch = potrf_plan::blocked_plan<T>(n, s.batch, f.launch, s.leaf_trsm, regs<T>(f),
                                                r.potrf_nb, r.potrf_w);
        nd.geometry = bp;
        nd.set("nb", bp.nb).set("W", bp.W).set("W_cfg", w_cfg).set("leaf_min_blocks", bp.leaf_min_blocks);
        return nd;
    }
    static void expand(Node& nd, const Shape& s, const Facts& f, const Request& r) {
        const auto bp = std::any_cast<potrf_plan::BlockedParams>(nd.geometry);
        const int n = static_cast<int>(s.order());
        Shape leaf = s;
        leaf.m = leaf.n = std::min(bp.nb, n);
        nd.children.push_back(Cta<T>::sketch_at(leaf, f, bp.leaf_min_blocks));   // leaf: explicit
        potrf_plan::blocked_schedule(n, bp.nb, bp.W, [](int, int, int) {},
            [&](int, int ib, int m2) { add_unique(nd, plan_trsm<T>(m2, ib, s.batch, f, r)); },
            [&](int, int ib, int, int w, int mr) {
                add_unique(nd, plan_gemm<T>(w, w, ib, s.batch, f, r));
                if (mr > 0) add_unique(nd, plan_gemm<T>(mr, w, ib, s.batch, f, r));
            },
            [] {});
    }
};

template <class T> struct Vendor {
    static constexpr Route route{Origin::Vendor, Algorithm::Auto};
    static constexpr std::string_view name = "vendor";
    static Legality legal(const Shape& s, const Facts& f) {
        if (!f.vendor_available) return Legality::no("vendor library not built");
        return s.m == s.n ? Legality::yes() : Legality::no("non-square");
    }
    static Node sketch(const Shape& s, const Facts&, const Request&) {
        Node nd = make_node(route, "vendor potrfBatched", s);
        nd.launch = potrf_plan::vendor_pseudo_plan<T>(static_cast<int>(s.order()), s.batch);
        return nd;
    }
    static void expand(Node&, const Shape&, const Facts&, const Request&) {}
};

// THE list of potrf tiers. Order = the FirstLegal tie order among SELECTABLE tiers only; a
// shared Route resolves to its owner and an unmeasured tier is never chosen, so moving CtaWg
// (mutant 5) changes nothing.
template <class T>
using Tiers = std::conditional_t<kMutant == 3,
    TierList<Tiny<T>, Cta<T>, CtaWg<T>, Blocked<T>, LPanel<T>, Vendor<T>>,
    std::conditional_t<kMutant == 5,
    TierList<Tiny<T>, CtaWg<T>, Cta<T>, LPanel<T>, Blocked<T>, Vendor<T>>,
    TierList<Tiny<T>, Cta<T>, CtaWg<T>, LPanel<T>, Blocked<T>, Vendor<T>>>>;

// ---- data: profile rows by NAME, candidacy, hand windows ----------------------------------

// The generated table is still positional (e[route][dtype][uplo]); this is its ONE column map.
inline constexpr std::string_view kProfileColumns[5] = {
    "native:tiny", "native:cta", "native:lpanel", "native:blocked", "vendor"};

inline std::optional<ModelRow> model_row(arch::RoutingProfile a, std::string_view tier,
                                         ScalarKind dt, std::uint8_t mode) {
    const auto* p = potrf_profile::profile_for(a);
    if (!p) return std::nullopt;
    for (int i = 0; i < 5; ++i) {
        if (kProfileColumns[i] != tier) continue;
        const auto& e = p->e[i][static_cast<int>(dt)][mode];
        if (!e.present) return std::nullopt;
        return ModelRow{e.c, e.box};
    }
    return std::nullopt;
}

inline constexpr std::int64_t kInf = std::numeric_limits<std::int64_t>::max();

// route_potrf.hh's hand windows, as rows. vs-vendor: first match decides.
inline constexpr WindowRow kVsVendor[] = {
    {ScalarKind::F32, 1, 1, 32, "native:tiny"}, {ScalarKind::F64, 1, 1, 32, "native:tiny"},
    {ScalarKind::C32, 1, 1, 32, "native:tiny"}, {ScalarKind::C64, 1, 1, 32, "native:tiny"},
    {ScalarKind::F32, 0, 1, 32, "native:tiny"}, {ScalarKind::C32, 0, 1, 32, "native:tiny"},
    {ScalarKind::F64, 0, 2, 8, "native:tiny"},  {ScalarKind::F64, 0, 12, 16, "native:tiny"},
    {ScalarKind::C64, 0, 2, 8, "native:tiny"},  {ScalarKind::C64, 0, 12, 16, "native:tiny"},
    {ScalarKind::F32, 0, 33, 35, "native:cta"},
    {ScalarKind::F32, 0, 36, kMutant == 1 ? 255 : 256, "native:lpanel"},
    {ScalarKind::C32, 0, 33, 256, "native:lpanel"},
};

// The native tie-break: a priority list (first LEGAL row wins).
inline constexpr WindowRow kNativeTier[] = {
    {ScalarKind::F32, 1, 1, 32, "native:tiny"}, {ScalarKind::F64, 1, 1, 32, "native:tiny"},
    {ScalarKind::C32, 1, 1, 32, "native:tiny"}, {ScalarKind::C64, 1, 1, 32, "native:tiny"},
    {ScalarKind::F32, 0, 1, 32, "native:tiny"}, {ScalarKind::C32, 0, 1, 32, "native:tiny"},
    {ScalarKind::F64, 0, 2, 8, "native:tiny"},  {ScalarKind::F64, 0, 12, 16, "native:tiny"},
    {ScalarKind::C64, 0, 2, 8, "native:tiny"},  {ScalarKind::C64, 0, 12, 16, "native:tiny"},
    {ScalarKind::F32, 0, 36, kInf, "native:lpanel"},
    {ScalarKind::C32, 0, 33, kInf, "native:lpanel"},
    {ScalarKind::F32, 0, 1, kInf, "native:cta"}, {ScalarKind::F32, 1, 1, kInf, "native:cta"},
    {ScalarKind::F64, 0, 1, kInf, "native:cta"}, {ScalarKind::F64, 1, 1, kInf, "native:cta"},
    {ScalarKind::C32, 0, 1, kInf, "native:cta"}, {ScalarKind::C32, 1, 1, kInf, "native:cta"},
    {ScalarKind::C64, 0, 1, kInf, "native:cta"}, {ScalarKind::C64, 1, 1, kInf, "native:cta"},
    {ScalarKind::F32, 0, 1, kInf, "native:blocked"}, {ScalarKind::F64, 0, 1, kInf, "native:blocked"},
    {ScalarKind::C32, 0, 1, kInf, "native:blocked"}, {ScalarKind::C64, 0, 1, kInf, "native:blocked"},
};

inline const WindowPolicy kWindows{{kVsVendor, true, true}, {kNativeTier, false, false}};

// Tiers measured nowhere: an EXPLICIT decision, so a tier missing from every table is a test
// failure rather than a silent non-candidate.
inline constexpr std::string_view kUnmeasured[] = {"native:cta_wg"};

// Auto may pick every tier but these; a pin still can (mutant 6 drops the net).
inline bool selectable(std::string_view tier) {
    if (kMutant == 6) return true;
    for (std::string_view u : kUnmeasured) {
        if (u == tier) return false;
    }
    return true;
}

// Candidacy: ONE answer, read by the cost model and the windows alike.
inline bool measured(arch::RoutingProfile a, std::string_view tier, ScalarKind dt, std::uint8_t mode) {
    if (tier == "vendor") return true;
    if (model_row(a, tier, dt, mode)) return true;
    for (const WindowRow& w : kVsVendor) {
        if (w.dtype == dt && w.mode == mode && w.tier == tier) return true;
    }
    return false;
}

template <class T>
OpContext context(const Shape& s, const Facts& f) {
    OpContext c;
    c.op = Op::potrf;
    c.dt = dispatch::scalar_kind_of<T>;
    c.mode = mode_of(s.uplo);
    c.key = s.order();
    c.box_n = s.order();
    c.box_batch = s.batch;
    c.model_row = &model_row;
    c.measured = &measured;
    c.selectable = &selectable;
    c.windows = &kWindows;
    const auto* p = potrf_profile::profile_for(f.arch);
    c.model_gate = p && p->model_enabled && s.m == s.n && s.order() >= 1 && s.batch >= 1;
    c.margin = (p && kMutant != 2) ? p->margin : 0;
    return c;
}

template <class T>
Node plan_potrf(const Shape& s, const Facts& f, const Request& r) {
    return plan_op<Tiers<T>>(context<T>(s, f), s, f, r);
}

}  // inline namespace BATCHLAS_PLAN_ABI
}  // namespace batchlas::plan::potrf
