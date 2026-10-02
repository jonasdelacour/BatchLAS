#pragma once

// POTRF's routes as descriptors (prototype of the descriptor-registry design). Each struct is
// ONE route: legality with a reason, the geometry its launcher uses, the cost plan, workspace
// and launch. PotrfTable at the bottom is the only list of potrf routes. SYCL-free.
// evidence: experiments/kernel_selection/descriptor-registry/README.md

#include "../dispatch/selection/route_table.hh"
#include "../extensions/potrf_launch_plan.hh"
#include "../extensions/potrf_native.hh"

#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <algorithm>
#include <complex>
#include <string>
#include <type_traits>

namespace batchlas::potrf_routes {

namespace pp = potrf_plan;
namespace ds = dispatch::sel;
using dispatch::Algorithm;
using dispatch::Origin;
using dispatch::Route;
using ds::Verdict;

// Inputs only; nothing priced lives here.
struct PotrfShape {
    int n = 0;
    std::int64_t batch = 0;
    Uplo uplo = Uplo::Lower;
    bool square = true, homogeneous = true, is_gpu = false, has_sg32 = false;
    launch_plan::DeviceFacts dev{};   // local mem, max wg, CUs, profile caps: one struct
    pp::KernelRegs regs{};
    int leaf_trsm = 0, nb_knob = 0, w_knob = 0;   // one settings snapshot per select
};

template <class T>
struct PotrfArgs {
    const MatrixView<T, MatrixFormat::Dense>* A = nullptr;
    Uplo uplo = Uplo::Lower;
    Span<std::byte> ws{};
    Span<int32_t> info{};
};

template <class T>
struct PotrfOp {
    using Shape = PotrfShape;
    using Args = PotrfArgs<T>;
    static constexpr dispatch::Op op = dispatch::Op::potrf;
    static std::uint8_t cost_variant(const Shape& s) { return s.uplo == Uplo::Upper ? 1 : 0; }
    static int coords(const Shape& s, std::array<std::int64_t, 4>& c) {
        c = {s.n, s.batch, 0, 0};
        return 2;
    }
    static bool matches(const Shape& s, const Args& a) {
        return a.A && a.uplo == s.uplo && a.A->rows() == s.n && a.A->cols() == s.n &&
               a.A->batch_size() == s.batch && a.A->is_heterogeneous() == !s.homogeneous;
    }
};

inline Verdict native_common(const PotrfShape& s) {
    if (!s.square) return Verdict::no("non-square");
    if (!s.is_gpu || !s.has_sg32) return Verdict::no("needs a GPU with sub-group 32");
    if (!s.homogeneous) return Verdict::no("heterogeneous batch");
    if (s.n < 1 || s.batch < 1) return Verdict::no("empty");
    return Verdict::yes();
}
inline Verdict lower_only(const PotrfShape& s, Verdict v) {
    return (v.ok && s.uplo != Uplo::Lower) ? Verdict::no("Lower only") : v;
}

template <class T>
std::string type_name() {
    if constexpr (std::is_same_v<T, float>) return "float";
    else if constexpr (std::is_same_v<T, double>) return "double";
    else if constexpr (std::is_same_v<T, std::complex<float>>) return "cfloat";
    else return "cdouble";
}
inline std::string launch_text(std::int64_t groups, int wg) {
    return " groups=" + std::to_string(groups) + " wg=" + std::to_string(wg);
}

template <class T>
struct Tiny {
    static constexpr Route route{Origin::Native, Algorithm::Tiny};
    static constexpr std::string_view key = "native:tiny";
    using Geometry = pp::TinyGeometry;
    static Verdict legal(const PotrfShape& s) {
        const Verdict v = native_common(s);
        return (v.ok && s.n > pp::TinyCap<T>::kMaxN) ? Verdict::no("n > tiny cap") : v;
    }
    static Geometry plan(const PotrfShape& s) { return pp::tiny_geometry<T>(s.n, s.batch, s.dev.max_wg_size); }
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry&) {
        return pp::tiny_plan<T>(s.n, s.batch, s.dev, s.regs);
    }
    static std::string kernel(const Geometry& g) {
        return "PotrfTinyKernel<" + type_name<T>() + "," + std::to_string(g.N) + ">" +
               launch_text(g.num_wg, g.wg_size);
    }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};

template <class T>
struct Cta {
    static constexpr Route route{Origin::Native, Algorithm::CTA};
    static constexpr std::string_view key = "native:cta";
    using Geometry = pp::CtaGeometry;
    static Verdict legal(const PotrfShape& s) { return native_common(s); }   // capacity: plan().fits
    static Geometry plan(const PotrfShape& s) { return pp::cta_geometry<T>(s.n, s.batch, s.dev); }
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry&) {
        return pp::cta_plan<T>(s.n, s.batch, s.dev, s.regs);
    }
    static std::string kernel(const Geometry& g) {
        return "PotrfCtaKernel<" + type_name<T>() + "," + std::to_string(pp::CtaConst<T>::NB) + "," +
               std::to_string(pp::CtaConst<T>::TS) + (g.subgroup_scope ? ",SubGroup>" : ",WorkGroup>") +
               " L=" + std::to_string(g.L) + " G=" + std::to_string(g.G) + launch_text(g.num_wg, g.wg_size);
    }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};

template <class T>
struct LPanel {
    static constexpr Route route{Origin::Native, Algorithm::LPanel};
    static constexpr std::string_view key = "native:lpanel";
    using Geometry = pp::LpanelGeometry;
    static Verdict legal(const PotrfShape& s) { return lower_only(s, native_common(s)); }
    static Geometry plan(const PotrfShape& s) { return pp::lpanel_geometry<T>(s.n, s.batch, s.dev); }
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry&) {
        return pp::lpanel_plan<T>(s.n, s.batch, s.dev, s.regs);
    }
    static std::string kernel(const Geometry& g) {
        return "PotrfLpanelKernel<" + type_name<T>() + "," + std::to_string(g.nb) + "> L=" +
               std::to_string(g.L) + " G=" + std::to_string(g.G) + launch_text(g.num_wg, g.wg_size);
    }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};

template <Backend B, class T>
struct Blocked {
    static constexpr Route route{Origin::Native, Algorithm::Blocked};
    static constexpr std::string_view key = "native:blocked";
    struct Geometry {
        pp::BlockedParams p{};   // W clamped to n - nb: what the layout allocates
        pp::CtaGeometry leaf{};
        bool fits = false;
    };
    static Verdict legal(const PotrfShape& s) { return lower_only(s, native_common(s)); }
    static Geometry plan(const PotrfShape& s) {
        Geometry g;
        g.p = pp::blocked_params<T>(s.n, s.dev.local_mem_bytes, s.leaf_trsm, s.nb_knob, s.w_knob);
        g.p.W = std::max(1, std::min(g.p.W, s.n - g.p.nb));
        g.leaf = pp::cta_geometry<T>(std::min(g.p.nb, s.n), s.batch, s.dev, g.p.leaf_min_blocks);
        g.fits = g.leaf.fits;
        return g;
    }
    // Priced with the UNclamped W, as today: clamped pricing moves 12 cells (README, risk 1).
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry&) {
        return pp::blocked_plan<T>(s.n, s.batch, s.dev, s.leaf_trsm, s.regs, s.nb_knob, s.w_knob);
    }
    static std::string kernel(const Geometry& g) {
        return "blocked nb=" + std::to_string(g.p.nb) + " W=" + std::to_string(g.p.W) +
               ": leaf PotrfCtaKernel<" + type_name<T>() + "> L=" + std::to_string(g.leaf.L) +
               launch_text(g.leaf.num_wg, g.leaf.wg_size) + " + routed trsm + routed gemm";
    }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};

// ---- EXTENSION: a fifth native tier. CTA at WORK-GROUP scope, one matrix per group. ----------
template <class T>
struct CtaWg {
    static constexpr Route route{Origin::Native, Algorithm::CTA};   // key, not Route, is identity
    static constexpr std::string_view key = "native:cta_wg";
    using Geometry = pp::CtaGeometry;
    static Verdict legal(const PotrfShape& s) { return native_common(s); }
    static Geometry plan(const PotrfShape& s) {
        Geometry g = pp::cta_geometry<T>(s.n, s.batch, s.dev);
        g.G = 1;   // WorkGroup scope is a race with G > 1
        g.subgroup_scope = false;
        g.wg_size = g.L;
        g.num_wg = s.batch;
        g.slm_total = potrf_native::potrf_hole_padded(g.slm_per_matrix);
        g.fits = g.slm_total <= resident::device_slm_budget(s.dev.local_mem_bytes) &&
                 g.wg_size <= s.dev.max_wg_size;
        return g;
    }
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry& g) {
        launch_plan::LaunchPlan p = pp::cta_plan<T>(s.n, s.batch, s.dev, s.regs);
        p.groups = g.num_wg;
        p.wg_size = g.wg_size;
        p.slm_per_group = g.slm_total;
        return p;
    }
    static std::string kernel(const Geometry& g) { return Cta<T>::kernel(g); }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};
// ---- end EXTENSION --------------------------------------------------------------------------

template <Backend B, class T>
struct Cusolver {   // listed only where solver_vendor_available<B>
    static constexpr Route route{Origin::Vendor, Algorithm::Auto};
    static constexpr std::string_view key = "vendor:cusolver";
    struct Geometry { bool fits = true; };
    static Verdict legal(const PotrfShape& s) { return s.square ? Verdict::yes() : Verdict::no("non-square"); }
    static Geometry plan(const PotrfShape&) { return {}; }
    static launch_plan::LaunchPlan cost_plan(const PotrfShape& s, const Geometry&) {
        return pp::vendor_pseudo_plan<T>(s.n, s.batch);
    }
    static std::string kernel(const Geometry&) { return "cusolverDnXpotrf (batched vendor)"; }
    static std::size_t workspace(Queue&, const PotrfArgs<T>&, const Geometry&);
    static Event launch(Queue&, const PotrfArgs<T>&, const Geometry&);
};

// THE list of potrf routes. Vendor-free: the vendor row is not in the type, so never named.
// Order is a tie-break and the last-resort "first eligible native" walk, nothing else.
template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
using PotrfTable = std::conditional_t<Vendor,
    ds::Table<PotrfOp<T>, Tiny<T>, Cta<T>, LPanel<T>, Blocked<B, T>, CtaWg<T>, Cusolver<B, T>>,
    ds::Table<PotrfOp<T>, Tiny<T>, Cta<T>, LPanel<T>, Blocked<B, T>, CtaWg<T>>>;

}  // namespace batchlas::potrf_routes
