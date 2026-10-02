// Extension (b), offline: getrs -- a second op with an nrhs axis and a trans variant -- on the
// SAME op-agnostic select.hh/route_table.hh. No chooser code is copied: WindowChooser carries
// today's getrs windows as data, and ModelChooser prices a 3-D (n, nrhs, batch) support box.
// Decision-level only (descriptors declare workspace/launch, nothing odr-uses them).
// Build and run: experiments/kernel_selection/descriptor-registry/build_offline.sh (g++, no GPU).

#include "../../../src/dispatch/selection/route_table.hh"

#include <batchlas/blas/dispatch/route_getrs.hh>

#include <complex>
#include <cstdio>
#include <string>

using namespace batchlas;
namespace ds = dispatch::sel;
using dispatch::Algorithm;
using dispatch::Origin;
using dispatch::Route;

namespace getrs_routes {

struct GetrsShape {
    int n = 0, nrhs = 0;
    std::int64_t batch = 0;
    Transpose trans = Transpose::NoTrans;
    bool is_gpu = true, has_sg32 = true, homogeneous = true, blocked_available = true;
    std::int64_t fused_max_elems = 0, fused_max_nrhs = 0;   // device-derived capacity
};
template <class T> struct GetrsArgs {};

template <class T>
struct GetrsOp {
    using Shape = GetrsShape;
    using Args = GetrsArgs<T>;
    static std::uint8_t cost_variant(const Shape& s) { return static_cast<std::uint8_t>(s.trans); }
    static int coords(const Shape& s, std::array<std::int64_t, 4>& c) {
        c = {s.n, s.nrhs, s.batch, 0};
        return 3;
    }
};

inline ds::Verdict native_common(const GetrsShape& s) {
    if (!s.is_gpu || !s.has_sg32) return ds::Verdict::no("needs a GPU with sub-group 32");
    if (!s.homogeneous) return ds::Verdict::no("heterogeneous batch");
    if (s.n < 1 || s.nrhs < 1 || s.batch < 1) return ds::Verdict::no("empty");
    return ds::Verdict::yes();
}

struct Geo { std::int64_t groups = 0; int wg = 0; bool fits = false; };

// The CTA capacity gate as the PLAN's fit, stated exactly as supports() states it.
inline bool cta_fits(const GetrsShape& s) {
    return static_cast<std::int64_t>(s.n) * s.nrhs <= s.fused_max_elems && s.nrhs <= s.fused_max_nrhs;
}

template <class T>
struct Cta {
    static constexpr Route route{Origin::Native, Algorithm::CTA};
    static constexpr std::string_view key = "native:cta";
    using Geometry = Geo;
    static ds::Verdict legal(const GetrsShape& s) {
        if (s.fused_max_elems <= 0 || s.fused_max_nrhs <= 0) return ds::Verdict::no("fused tier absent");
        return native_common(s);
    }
    static Geometry plan(const GetrsShape& s) { return {s.batch, 32, cta_fits(s)}; }
    static launch_plan::LaunchPlan cost_plan(const GetrsShape& s, const Geometry& g) {
        launch_plan::LaunchPlan p;
        p.launches = p.wave_launches = 1;
        p.groups = g.groups;
        p.wg_size = g.wg;
        p.serial_steps = 2 * s.n;
        p.flops = 2.0 * s.n * s.n * s.nrhs * static_cast<double>(s.batch);
        return p;
    }
    static std::string kernel(const Geometry&) { return "getrs fused CTA"; }
    static std::size_t workspace(Queue&, const GetrsArgs<T>&, const Geometry&);
    static Event launch(Queue&, const GetrsArgs<T>&, const Geometry&);
};

template <class T>
struct Blocked {
    static constexpr Route route{Origin::Native, Algorithm::Blocked};
    static constexpr std::string_view key = "native:blocked";
    using Geometry = Geo;
    static ds::Verdict legal(const GetrsShape& s) {
        return s.blocked_available ? native_common(s) : ds::Verdict::no("blocked tier absent");
    }
    static Geometry plan(const GetrsShape& s) { return {s.batch, 128, true}; }
    static launch_plan::LaunchPlan cost_plan(const GetrsShape& s, const Geometry& g) {
        launch_plan::LaunchPlan p;
        p.launches = 3;
        p.wave_launches = 1;
        p.groups = g.groups;
        p.wg_size = g.wg;
        p.flops = 2.0 * s.n * s.n * s.nrhs * static_cast<double>(s.batch);
        p.batch_wide_work = true;
        return p;
    }
    static std::string kernel(const Geometry&) { return "laswp gather + 2 routed trsm"; }
    static std::size_t workspace(Queue&, const GetrsArgs<T>&, const Geometry&);
    static Event launch(Queue&, const GetrsArgs<T>&, const Geometry&);
};

template <class T>
struct Vendor {
    static constexpr Route route{Origin::Vendor, Algorithm::Auto};
    static constexpr std::string_view key = "vendor:cusolver";
    struct Geometry { bool fits = true; };
    static ds::Verdict legal(const GetrsShape&) { return ds::Verdict::yes(); }
    static Geometry plan(const GetrsShape&) { return {}; }
    static launch_plan::LaunchPlan cost_plan(const GetrsShape& s, const Geometry&) {
        launch_plan::LaunchPlan p;
        p.launches = 1;
        p.additive_work = true;
        p.item_steps = static_cast<double>(s.n) * s.nrhs * static_cast<double>(s.batch);
        return p;
    }
    static std::string kernel(const Geometry&) { return "cusolverDnXgetrs (int64 pivots)"; }
    static std::size_t workspace(Queue&, const GetrsArgs<T>&, const Geometry&);
    static Event launch(Queue&, const GetrsArgs<T>&, const Geometry&);
};

template <class T, bool V>
using GetrsTable = std::conditional_t<V, ds::Table<GetrsOp<T>, Cta<T>, Blocked<T>, Vendor<T>>,
                                      ds::Table<GetrsOp<T>, Cta<T>, Blocked<T>>>;

// Today's preferred() / native_tier_preferred() as first-match data rows.
template <class T>
bool cta_vs_vendor(const GetrsShape& s) {
    if (s.n < 32) return false;
    if (s.nrhs <= 2) return true;
    return std::is_same_v<T, float> && s.nrhs <= 4;
}
template <class T>
bool blocked_vs_vendor(const GetrsShape& s) {
    if (s.batch < 128) return false;
    if constexpr (std::is_same_v<T, float>) return s.nrhs >= 64;
    if constexpr (std::is_same_v<T, double>) return s.nrhs >= 128;
    return false;
}
template <class T>
inline constexpr ds::Window<GetrsShape> kGetrsWindows[] = {
    {"native:cta", ds::Question::VsVendor, &cta_vs_vendor<T>, "docs/perf/lu.md"},
    {"native:blocked", ds::Question::VsVendor, &blocked_vs_vendor<T>, "docs/perf/lu.md"},
    {"native:cta", ds::Question::AmongNative, [](const GetrsShape&) { return true; }, "docs/perf/lu.md"},
};

}  // namespace getrs_routes

namespace {

using namespace getrs_routes;

std::string key_of(Route r) {
    if (dispatch::is_vendor(r)) return "vendor:cusolver";
    return std::string(dispatch::to_string(r.origin)) + ":" + std::string(dispatch::to_string(r.algo));
}

template <class T>
long sweep(bool vendor, long& cells) {
    long diffs = 0;
    for (int n : {1, 8, 31, 32, 33, 64, 128, 256, 512})
    for (int nrhs : {1, 2, 3, 4, 5, 16, 63, 64, 65, 127, 128, 256})
    for (long b : {1L, 127L, 128L, 4096L})
    for (std::int64_t cap : {std::int64_t{0}, std::int64_t{4096}, std::int64_t{65536}})
    for (bool blocked : {true, false}) {
        ++cells;
        dispatch::GetrsShape o;
        o.op = dispatch::Op::getrs; o.backend = Backend::CUDA; o.m = n; o.n = nrhs; o.batch = b;
        o.is_gpu = true; o.has_sg32 = true; o.blocked_available = blocked;
        o.fused_max_elems = cap; o.fused_max_nrhs = cap ? 64 : 0;
        const Route old = dispatch::resolve_route_uninstrumented<dispatch::Op::getrs, T>(Route{}, o, vendor);
        GetrsShape s;
        s.n = n; s.nrhs = nrhs; s.batch = b; s.blocked_available = blocked;
        s.fused_max_elems = cap; s.fused_max_nrhs = cap ? 64 : 0;
        std::string got;
        auto run = [&]<class Tbl>() {
            typename Tbl::Geoms g;
            const ds::Candidates c = Tbl::candidates(s, g);
            const ds::WindowChooser<GetrsShape> w{kGetrsWindows<T>, &s};
            const ds::ModelChooser m{nullptr, {}};   // no getrs CostBook fitted: windows decide
            const ds::Decision d = ds::decide(c, std::nullopt, m, w);
            got = d.index < 0 ? "vendor:cusolver" : std::string(c.row[d.index].key);
        };
        if (vendor) run.template operator()<GetrsTable<T, true>>();
        else run.template operator()<GetrsTable<T, false>>();
        if (got != key_of(old)) {
            if (diffs < 10) {
                std::printf("DIFF vendor=%d n=%d nrhs=%d b=%ld cap=%ld blocked=%d old=%s new=%s\n",
                            vendor, n, nrhs, b, static_cast<long>(cap), blocked, key_of(old).c_str(),
                            got.c_str());
            }
            ++diffs;
        }
    }
    return diffs;
}

// The SAME ModelChooser on a 3-D support box and a trans variant, with a synthetic CostBook
// (no getrs profile exists): checks only that the op-agnostic chooser reads dims = 3.
int model_3d_check() {
    using T = float;
    const launch_plan::CostConstants cheap{1e-6, 1e-12, 0, 1e-8, 0, 0};
    const launch_plan::CostConstants vend{2e-6, 0, 0, 0, 0, 5e-9};
    const ds::CostRow rows[] = {
        {"native:cta", 0, cheap, {32, 512, 1, 64, 128, 1 << 20}, 10},   // NoTrans only
        {"native:blocked", 0, cheap, {32, 512, 1, 512, 128, 1 << 20}, 10},
        {"vendor:cusolver", 0, vend, {1, 4096, 1, 4096, 1, 1 << 20}, 10},
        {"vendor:cusolver", 1, vend, {1, 4096, 1, 4096, 1, 1 << 20}, 10},
    };
    const ds::CostBook book{true, 0.05, rows};
    launch_plan::DeviceFacts f;
    f.compute_units = 188;
    auto pick = [&](int n, int nrhs, long b, Transpose t) {
        GetrsShape s;
        s.n = n; s.nrhs = nrhs; s.batch = b; s.trans = t;
        s.fused_max_elems = 65536; s.fused_max_nrhs = 64;
        typename GetrsTable<T, true>::Geoms g;
        const ds::Candidates c = GetrsTable<T, true>::candidates(s, g);
        const ds::Decision d = ds::decide(c, std::nullopt, ds::ModelChooser{&book, f},
                                          ds::WindowChooser<GetrsShape>{kGetrsWindows<T>, &s});
        std::printf("%s", ds::explain(c, d, "getrs n=" + std::to_string(n) + " nrhs=" +
                                                std::to_string(nrhs) + " trans=" +
                                                std::to_string(static_cast<int>(t))).c_str());
        return std::pair{d, std::string(c.row[d.index].key)};
    };
    int bad = 0;
    const auto [a, ka] = pick(128, 8, 4096, Transpose::NoTrans);    // inside every box
    bad += !(a.by == ds::Strategy::Model && a.extrapolated == 0);
    const auto [b, kb] = pick(128, 100, 4096, Transpose::NoTrans);  // nrhs > 64: CTA unfit
    bad += !(b.by == ds::Strategy::Model && kb != "native:cta");
    const auto [c, kc] = pick(128, 8, 4096, Transpose::Trans);      // variant 1: only vendor priced
    bad += !(c.by == ds::Strategy::Model && kc == "vendor:cusolver");
    const auto [d, kd] = pick(600, 8, 4096, Transpose::NoTrans);    // n outside the native boxes
    bad += !(d.by == ds::Strategy::Model && d.extrapolated == (kd == "vendor:cusolver" ? 0 : 1));
    return bad;
}

}  // namespace

int main() {
    long cells = 0, diffs = 0;
    for (bool v : {true, false}) {
        diffs += sweep<float>(v, cells);
        diffs += sweep<double>(v, cells);
        diffs += sweep<std::complex<float>>(v, cells);
        diffs += sweep<std::complex<double>>(v, cells);
    }
    std::printf("getrs windows-as-data vs shipped RouteTable<getrs>: cells=%ld diffs=%ld\n", cells, diffs);
    const int bad = model_3d_check();
    std::printf("getrs 3-D ModelChooser checks failed: %d\n", bad);
    return (diffs || bad) ? 1 : 0;
}
