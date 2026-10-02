#pragma once

// POTRF's launch geometry, one definition per native tier: the launchers, the capability
// ceilings supports() reads, the *_debug_* introspection and tools/potrf_plan_dump all call
// these functions, so a cost model cannot price a geometry the kernel does not launch.
// SYCL-free. evidence: docs/perf/potrf.md#launch-plans

#include "../util/launch_plan.hh"
#include "../util/resident_capacity.hh"
#include "potrf_slm_hole.hh"
#include "tiny_geometry.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::potrf_plan {

using launch_plan::DeviceFacts;
using launch_plan::LaunchPlan;
using potrf_native::potrf_hole_padded;

template <typename T> struct RealOf { using type = T; };
template <typename R> struct RealOf<std::complex<R>> { using type = R; };
template <typename T>
inline constexpr bool kIsComplex = !std::is_same_v<T, typename RealOf<T>::type>;

// Sizes of the device scalar and its real part; the kernels static_assert sizeof(D)==sizeof(T).
template <typename T> inline constexpr std::size_t kSzD = sizeof(T);
template <typename T> inline constexpr std::size_t kSzR = sizeof(typename RealOf<T>::type);

constexpr double tri(double n) { return n * (n + 1) / 2; }

// LAPACK's potrf count; a complex multiply-add is four real ones.
template <typename T>
constexpr double useful_flops(int n, std::int64_t batch) {
    const double d = n;
    const double f = d * d * d / 3 + d * d / 2 + d / 6;
    return f * (kIsComplex<T> ? 4.0 : 1.0) * static_cast<double>(batch);
}

// ---- TINY: one matrix per SubGroupPartition<N>, registers only -----------------------------

// A flat compile-time ceiling: the tier owns no local memory. evidence: docs/perf/potrf.md#the-tiny-tier
template <typename T> struct TinyCap { static constexpr int kMaxN = 32; };
template <> struct TinyCap<std::complex<double>> { static constexpr int kMaxN = 16; };

// The worst probed demand across the tier's instantiations, the one the launch gate reads.
inline constexpr int kTinyWorstProbedRegs = 176;

struct TinyGeometry {
    int N = 0;        // register bucket
    int per_wg = 0;   // matrices per work-group
    int wg_size = 0;
    std::int64_t num_wg = 0;
    bool fits = false;
};

template <typename T>
constexpr TinyGeometry tiny_geometry(int n, std::int64_t batch, int max_wg) {
    using tiny_native::kTinyWgSize;
    TinyGeometry g;
    g.N = tiny_native::tiny_bucket_ge(n);
    if (g.N == 0 || n > TinyCap<T>::kMaxN || max_wg < kTinyWgSize) return g;
    // Byte argument a nominal 1 against an unbounded budget: the tier owns no local memory.
    g.per_wg = resident::pack_matrices_per_wg(1, g.N, ~std::size_t(0), max_wg, kTinyWgSize,
                                              kTinyWgSize / g.N);
    g.wg_size = g.per_wg * g.N;
    g.num_wg = (batch + g.per_wg - 1) / g.per_wg;
    g.fits = (g.wg_size % tiny_native::kTinySubGroupSize == 0);
    return g;
}

// ---- CTA: one matrix per work-group, whole matrix in local memory --------------------------

// NB is the panel width and the d[]/x[] register length, TS the thread tile.
// evidence: docs/perf/potrf.md#register-gate
template <typename T> struct CtaConst { static constexpr int NB = 8; static constexpr int TS = 4; };
template <> struct CtaConst<std::complex<double>> { static constexpr int NB = 8; static constexpr int TS = 2; };

inline constexpr int kCtaMaxL = 256;
inline constexpr int kCtaElemsPerItem = 24;

// lda = n | 1 is odd so a stride-lda row read is conflict-free; the 256 over-covers *fail
// plus alignment slack. evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
constexpr std::size_t cta_slm_per_matrix(int n, int NB, int TS, std::size_t sz_d, std::size_t sz_r) {
    const std::size_t lda = static_cast<std::size_t>(n | 1);
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    const int Rt0 = (m2_0 + TS - 1) / TS;
    return lda * static_cast<std::size_t>(n) * sz_d + static_cast<std::size_t>(NB) * sz_r + 256 +
           4 * static_cast<std::size_t>(Rt0 + 1);
}

struct CtaGeometry {
    int L = 32;               // work-items per matrix
    int G = 1;                // matrices per work-group; > 1 only when L == 32
    int wg_size = 32;
    std::int64_t num_wg = 0;
    int lda = 1;
    int Rt0 = 0;
    std::size_t slm_per_matrix = 0;
    std::size_t slm_total = 0;   // G * slm_per_matrix, after the hole pad
    bool subgroup_scope = true;  // L == 32; scope is derived here and nowhere else
    bool fits = false;
};

// slm_budget is ONE work-group's slice (already occupancy-scaled).
template <int NB, int TS>
constexpr CtaGeometry cta_geometry_raw(int n, std::int64_t batch, std::size_t sz_d,
                                       std::size_t sz_r, std::size_t slm_budget, int max_wg) {
    CtaGeometry p;
    p.lda = n | 1;
    const int m2_0 = (n > NB) ? (n - NB) : 0;
    p.Rt0 = (m2_0 + TS - 1) / TS;
    const long long ntiles_0 = static_cast<long long>(p.Rt0) * (p.Rt0 + 1) / 2;

    // L follows m2_0 = n - NB, the first trailing update, not n, and counts elements rather
    // than tiles because TS varies. evidence: docs/perf/potrf.md#the-l-ladder
    const long long work_elems = ntiles_0 * static_cast<long long>(TS) * TS;
    int want = 32;
    while (want < kCtaMaxL && static_cast<long long>(want) * kCtaElemsPerItem < work_elems) {
        want <<= 1;
    }
    p.L = want;
    while (p.L > 32 && p.L > max_wg) p.L >>= 1;

    p.slm_per_matrix = cta_slm_per_matrix(n, NB, TS, sz_d, sz_r);
    p.G = (p.L == 32 && p.slm_per_matrix > 0)
              ? resident::pack_matrices_per_wg(p.slm_per_matrix, p.L, slm_budget, max_wg)
              : 1;
    p.wg_size = p.G * p.L;
    p.num_wg = (batch + p.G - 1) / p.G;
    p.subgroup_scope = (p.L == 32);
    p.slm_total = potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix);
    p.fits = (p.slm_total <= slm_budget) && (p.wg_size <= max_wg);
    return p;
}

template <typename T>
constexpr CtaGeometry cta_geometry(int n, std::int64_t batch, const DeviceFacts& d,
                                   int min_blocks_per_sm = resident::kMinBlocksPerSm) {
    using C = CtaConst<T>;
    const std::size_t budget = resident::occupancy_budget(
        resident::device_slm_budget(d.local_mem_bytes), min_blocks_per_sm);
    return cta_geometry_raw<C::NB, C::TS>(n, batch, kSzD<T>, kSzR<T>, budget, d.max_wg_size);
}

// The ceiling supports() advertises; the pad is inside the walked function, so this and the
// launcher's `fits` stay one predicate. slm_budget_bytes is the DEVICE budget.
template <typename T>
constexpr int cta_max_n(std::size_t slm_budget_bytes,
                        int min_blocks_per_sm = resident::kMinBlocksPerSm) {
    return resident::resident_max_n(
        [](int n) {
            return potrf_hole_padded(cta_slm_per_matrix(n, CtaConst<T>::NB, CtaConst<T>::TS,
                                                        kSzD<T>, kSzR<T>));
        },
        slm_budget_bytes, min_blocks_per_sm);
}

// ---- LPANEL: one n x NB panel in local memory, one work-item per row -----------------------

template <typename T> struct LpanelConst { static constexpr int NB = 8; };

// A hint the type cannot honour is an ERROR, not a silent downgrade.
template <typename T>
constexpr bool lpanel_nb_is_built(int nb) {
    return nb == LpanelConst<T>::NB || (std::is_same_v<T, float> && nb == 16);
}
template <typename T>
constexpr int lpanel_nb_for(int hint) { return (hint == 0) ? LpanelConst<T>::NB : hint; }

// sA is the n x NB panel at ld = n, sB the NB x NB broadcast block, 256 covers *fail + slack.
constexpr std::size_t lpanel_slm_per_matrix(int n, int nb, std::size_t sz_d) {
    return (static_cast<std::size_t>(n) * static_cast<std::size_t>(nb) +
            static_cast<std::size_t>(nb) * static_cast<std::size_t>(nb)) * sz_d + 256;
}

// `L >= n` is CORRECTNESS: lane `row` carries rp/rS across the k loop. Whole sub-groups.
constexpr int lpanel_lanes(int n) { return ((n + 31) / 32) * 32; }

struct LpanelGeometry {
    int nb = 0;
    int L = 32;
    int G = 1;
    int wg_size = 32;
    std::int64_t num_wg = 0;
    int slda = 1;
    std::size_t slm_per_matrix = 0;
    std::size_t slm_total = 0;   // after the hole pad
    bool fits = false;
};

// G > 1 is legal: every barrier is a work-group barrier over a schedule that depends only on
// (n, NB). evidence: docs/perf/potrf.md#why-lpanel-may-pack-on-work-group-barriers
constexpr LpanelGeometry lpanel_geometry_raw(int n, int nb, std::int64_t batch, std::size_t sz_d,
                                             std::size_t slm_budget, int max_wg) {
    LpanelGeometry p;
    p.nb = nb;
    p.slda = n;
    p.L = lpanel_lanes(n);
    p.slm_per_matrix = lpanel_slm_per_matrix(n, nb, sz_d);
    p.G = resident::pack_matrices_per_wg(p.slm_per_matrix, p.L, slm_budget, max_wg);
    // The pad is applied to the TOTAL, so a G the byte test passed can still overflow it.
    while (p.G > 1 &&
           potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix) > slm_budget) {
        p.G >>= 1;
    }
    p.wg_size = p.G * p.L;
    p.num_wg = (batch + p.G - 1) / p.G;
    p.slm_total = potrf_hole_padded(static_cast<std::size_t>(p.G) * p.slm_per_matrix);
    p.fits = (p.L >= n) && (p.slm_total <= slm_budget) && (p.wg_size <= max_wg);
    return p;
}

// fits == false, nb == 0 when the hint names an NB this build does not instantiate.
template <typename T>
constexpr LpanelGeometry lpanel_geometry(int n, std::int64_t batch, const DeviceFacts& d,
                                         int min_blocks_per_sm = resident::kMinBlocksPerSm,
                                         int nb_hint = 0) {
    const int nb = lpanel_nb_for<T>(nb_hint);
    if (!lpanel_nb_is_built<T>(nb)) return LpanelGeometry{0};
    const std::size_t budget = resident::occupancy_budget(
        resident::device_slm_budget(d.local_mem_bytes), min_blocks_per_sm);
    return lpanel_geometry_raw(n, nb, batch, kSzD<T>, budget, d.max_wg_size);
}

// Two caps, and the work-group one is NOT slack: the body requires L >= n.
template <typename T>
constexpr int lpanel_max_n(std::size_t slm_budget_bytes, int max_wg_size,
                           int min_blocks_per_sm = resident::kMinBlocksPerSm, int nb_hint = 0) {
    const int nb = lpanel_nb_for<T>(nb_hint);
    if (!lpanel_nb_is_built<T>(nb)) return 0;
    const int by_wg = (max_wg_size / 32) * 32;
    const int by_slm = resident::resident_max_n(
        [nb](int n) { return potrf_hole_padded(lpanel_slm_per_matrix(n, nb, kSzD<T>)); },
        slm_budget_bytes, min_blocks_per_sm);
    return std::min(by_wg, by_slm);
}

// ---- BLOCKED: right-looking driver, CTA leaf + panel solve + folded trailing update --------

// nb is the diagonal block order and the update's k, W the trailing panel width.
// evidence: docs/perf/potrf.md#nb-and-w
template <typename T> struct BlockedConst;
template <> struct BlockedConst<float>                { static constexpr int NB = 128; static constexpr int W = 128; };
template <> struct BlockedConst<double>               { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct BlockedConst<std::complex<float>>  { static constexpr int NB = 96;  static constexpr int W = 32; };
template <> struct BlockedConst<std::complex<double>> { static constexpr int NB = 64;  static constexpr int W = 16; };

// Above this order the trailing update's k (= nb) dominates, so nb is clamped against the
// residency ceiling instead. evidence: docs/perf/potrf.md#the-occupancy-clamp-on-nb
inline constexpr int kOccupancyNbMaxOrder = 256;

struct BlockedParams {
    int nb;
    int W;
    int leaf_min_blocks;   // the occupancy target nb was clamped against, and the leaf's
};

// nb_env / w_env: the BATCHLAS_POTRF_NB / _W overrides, 0 = unset. leaf_trsm is
// sycl_trsm::trsm_cta_max_n<T>(), the block the panel solve's leaf rounds to.
template <typename T>
constexpr BlockedParams blocked_params(int n, std::size_t local_mem_bytes, int leaf_trsm,
                                       int nb_env = 0, int w_env = 0) {
    const std::size_t budget = resident::device_slm_budget(local_mem_bytes);
    const int leaf_min_blocks =
        (n > 0 && n <= kOccupancyNbMaxOrder) ? resident::kMinBlocksPerSm : 1;
    const int ceiling = cta_max_n<T>(budget, leaf_min_blocks);
    const int want = nb_env ? nb_env : BlockedConst<T>::NB;
    int nb = std::min(want, std::max(ceiling, 1));
    if (n > 0) nb = std::min(nb, n);
    if (leaf_trsm > 0 && nb >= leaf_trsm) nb = (nb / leaf_trsm) * leaf_trsm;
    if (nb < 1) nb = 1;
    int W = w_env ? w_env : BlockedConst<T>::W;
    if (W < 1) W = 1;
    return {nb, W, leaf_min_blocks};
}

// THE panel schedule: the driver issues its kernels from these callbacks and the plan counts
// them from the same walk. leaf(j, ib, m2); solve(j, ib, m2); update(j, ib, c, w, mr);
// panel_done(). A short final block (m2 == 0) issues neither a solve nor an update.
template <typename Leaf, typename Solve, typename Update, typename Done>
constexpr void blocked_schedule(int n, int nb, int W, Leaf&& leaf, Solve&& solve,
                                Update&& update, Done&& panel_done) {
    for (int j = 0; j < n; j += nb) {
        const int ib = std::min(nb, n - j);
        const int m2 = n - j - ib;
        leaf(j, ib, m2);
        if (m2 == 0) break;
        solve(j, ib, m2);
        for (int c = 0; c < m2; c += W) {
            const int w = std::min(W, m2 - c);
            update(j, ib, c, w, m2 - c - w);
        }
        panel_done();
    }
}

// ---- plans ---------------------------------------------------------------------------------

// Probed registers per thread of each kernel instantiation, for ONE scalar type on ONE arch.
// A per-arch capacity fact: it comes from evaluation/routing/profiles/registers.json (through
// potrf_plan_dump), never from here; 0 = unknown, and the occupancy estimate ignores it.
struct KernelRegs {
    int tiny[3] = {0, 0, 0};   // register buckets N = 8, 16, 32
    int cta_sg = 0;            // CTA at L == 32 (sub-group scope)
    int cta_wg = 0;            // CTA at L > 32 (work-group scope)
    int lpanel = 0;            // LPanel at the type's default NB
    int lpanel_nb16 = 0;       // LPanel float NB = 16
};

template <typename T>
constexpr LaunchPlan tiny_plan(int n, std::int64_t batch, const DeviceFacts& d,
                               const KernelRegs& r = {}) {
    const TinyGeometry g = tiny_geometry<T>(n, batch, d.max_wg_size);
    LaunchPlan p;
    p.fits = g.fits;
    if (!g.fits) return p;
    const double b = static_cast<double>(batch);
    p.launches = 1;
    p.wave_launches = 1;
    p.groups = g.num_wg;
    p.wg_size = g.wg_size;
    p.regs_per_item = r.tiny[g.N == 8 ? 0 : g.N == 16 ? 1 : 2];
    p.resident_groups_per_cu = launch_plan::resident_groups_per_cu(d, 0, g.wg_size, p.regs_per_item);
    // The unrolled recurrence runs the whole bucket; padding is masked, not skipped.
    p.flops = useful_flops<T>(g.N, batch);
    p.useful_flops = useful_flops<T>(n, batch);
    p.bytes = (2 * tri(n) * kSzD<T> + 4) * b;
    p.serial_steps = g.N;
    return p;
}

template <typename T>
constexpr LaunchPlan cta_plan_from(const CtaGeometry& g, int n, std::int64_t batch,
                                   const DeviceFacts& d, const KernelRegs& r = {}) {
    LaunchPlan p;
    p.fits = g.fits;
    if (!g.fits) return p;
    p.launches = 1;
    p.wave_launches = 1;
    p.groups = g.num_wg;
    p.wg_size = g.wg_size;
    p.slm_per_group = g.slm_total;
    p.regs_per_item = g.subgroup_scope ? r.cta_sg : r.cta_wg;
    p.resident_groups_per_cu =
        launch_plan::resident_groups_per_cu(d, g.slm_total, g.wg_size, p.regs_per_item);
    p.flops = useful_flops<T>(n, batch);
    p.useful_flops = p.flops;
    p.bytes = (2 * tri(n) * kSzD<T> + 4) * static_cast<double>(batch);
    // The body walks whole NB-wide panels, so the chain is NB*ceil(n/NB), not n.
    p.serial_steps = static_cast<std::int64_t>(CtaConst<T>::NB) *
                     ((n + CtaConst<T>::NB - 1) / CtaConst<T>::NB);
    return p;
}

template <typename T>
constexpr LaunchPlan cta_plan(int n, std::int64_t batch, const DeviceFacts& d,
                              const KernelRegs& r = {},
                              int min_blocks_per_sm = resident::kMinBlocksPerSm) {
    return cta_plan_from<T>(cta_geometry<T>(n, batch, d, min_blocks_per_sm), n, batch, d, r);
}

// The planner prototype's extension tier: the CTA body at work-group scope, ONE matrix per
// group (G == 1 is what makes work-group barriers correct) and at least two sub-groups.
template <typename T>
constexpr CtaGeometry cta_wg_geometry(int n, std::int64_t batch, const DeviceFacts& d,
                                      int min_blocks_per_sm = resident::kMinBlocksPerSm) {
    CtaGeometry p = cta_geometry<T>(n, batch, d, min_blocks_per_sm);
    p.L = (p.L < 64) ? 64 : p.L;
    p.G = 1;
    p.wg_size = p.L;
    p.num_wg = batch;
    p.subgroup_scope = false;
    p.slm_total = potrf_hole_padded(p.slm_per_matrix);
    const std::size_t budget = resident::occupancy_budget(
        resident::device_slm_budget(d.local_mem_bytes), min_blocks_per_sm);
    p.fits = n >= 1 && p.slm_total <= budget && p.wg_size <= d.max_wg_size;
    return p;
}

// Left-looking traffic: panel j re-reads every earlier column of its live rows plus one
// NB x NB broadcast block per earlier panel.
template <typename T>
constexpr LaunchPlan lpanel_plan(int n, std::int64_t batch, const DeviceFacts& d,
                                 const KernelRegs& r = {},
                                 int min_blocks_per_sm = resident::kMinBlocksPerSm,
                                 int nb_hint = 0) {
    const LpanelGeometry g = lpanel_geometry<T>(n, batch, d, min_blocks_per_sm, nb_hint);
    LaunchPlan p;
    p.fits = g.fits;
    if (!g.fits) return p;
    double elems = 0;
    for (int j = 0; j < n; j += g.nb) {
        const int ib = std::min(g.nb, n - j);
        elems += 2.0 * (n - j) * ib;   // panel read + factored panel write
        for (int k = 0; k < j; k += g.nb) {
            const int kb = std::min(g.nb, j - k);
            elems += static_cast<double>(ib) * kb + static_cast<double>(n - j) * kb;
        }
    }
    p.launches = 1;
    p.wave_launches = 1;
    p.groups = g.num_wg;
    p.wg_size = g.wg_size;
    p.slm_per_group = g.slm_total;
    p.regs_per_item = (g.nb == 16) ? r.lpanel_nb16 : r.lpanel;
    p.resident_groups_per_cu =
        launch_plan::resident_groups_per_cu(d, g.slm_total, g.wg_size, p.regs_per_item);
    p.flops = useful_flops<T>(n, batch);
    p.useful_flops = p.flops;
    p.bytes = (elems * kSzD<T> + 4) * static_cast<double>(batch);
    // Per-lane barrier chain: three barriers per column plus the k-loop's broadcast-block
    // steps, weighted by scalar width. evidence: docs/perf/potrf.md#launch-plans
    const double P = (n + g.nb - 1) / g.nb;
    const double chain = 3.0 * n + (static_cast<double>(kSzD<T>) / 2) * P * (P - 1) / 2;
    p.serial_steps = static_cast<std::int64_t>(chain);
    p.slot_chain = chain;
    return p;
}

// The groups of a wave launch are the LEAF's at the first (widest) panel. The solve and the
// gemms count as ONE submission each: they are routed calls whose own launch count is theirs.
template <typename T>
constexpr LaunchPlan blocked_plan(int n, std::int64_t batch, const DeviceFacts& d, int leaf_trsm,
                                  const KernelRegs& r = {}, int nb_env = 0, int w_env = 0) {
    const BlockedParams bp = blocked_params<T>(n, d.local_mem_bytes, leaf_trsm, nb_env, w_env);
    const CtaGeometry leaf = cta_geometry<T>(std::min(bp.nb, n), batch, d, bp.leaf_min_blocks);
    LaunchPlan p;
    p.fits = leaf.fits && n >= 1;
    if (!p.fits) return p;
    const double cx = kIsComplex<T> ? 4.0 : 1.0;
    double flops = 0;
    double elems = 0;
    int launches = (n > bp.nb) ? 2 : 1;   // info fill, and the product fill when one exists
    int panels = 0;
    blocked_schedule(
        n, bp.nb, bp.W,
        [&](int, int ib, int) {
            ++panels;
            launches += 2;   // leaf + info-merge fixup
            flops += useful_flops<T>(ib, 1);
            elems += 2 * tri(ib);
        },
        [&](int, int ib, int m2) {
            launches += 1;
            flops += cx * static_cast<double>(m2) * ib * ib;
            elems += tri(ib) + 2.0 * m2 * ib;
        },
        [&](int, int ib, int, int w, int mr) {
            launches += (mr > 0) ? 3 : 2;   // product gemm + fold (+ rectangular gemm)
            flops += cx * 2.0 * w * w * ib;
            elems += 2.0 * w * ib + 2.0 * w * w;   // epilogue reads prior even at beta == 0
            elems += static_cast<double>(w) * w + 2 * tri(w);
            if (mr > 0) {
                flops += cx * 2.0 * mr * w * ib;
                elems += static_cast<double>(mr) * ib + static_cast<double>(w) * ib + 2.0 * mr * w;
            }
        },
        [] {});
    const double b = static_cast<double>(batch);
    p.launches = launches;
    p.wave_launches = panels;
    p.groups = leaf.num_wg;
    p.wg_size = leaf.wg_size;
    p.slm_per_group = leaf.slm_total;
    p.regs_per_item = leaf.subgroup_scope ? r.cta_sg : r.cta_wg;
    p.resident_groups_per_cu =
        launch_plan::resident_groups_per_cu(d, leaf.slm_total, leaf.wg_size, p.regs_per_item);
    p.flops = flops * b;
    p.useful_flops = useful_flops<T>(n, batch);
    // The driver zero-fills the W x W product scratch per matrix whenever n > nb.
    const double fill = (n > bp.nb) ? static_cast<double>(bp.W) * bp.W * kSzD<T> : 0.0;
    p.bytes = (elems * kSzD<T> + 8.0 * panels + fill) * b;
    p.serial_steps = n;
    p.batch_wide_work = true;   // gemm/trsm/fold work is spread over the GPU, not per group
    return p;
}

// The vendor has no launch plan; this is its black-box form (V1c):
//   t = t0 + t_step*n_e + batch*(s_flop*f(n_e) + s_byte*n_e(n_e+1)*sz + t_item*n_e),
// n_e = n up to 16, then rounded up to whole 16-blocks (cuSOLVER's step), f = LAPACK's count.
// evidence: docs/perf/potrf.md#launch-plans
template <typename T>
constexpr LaunchPlan vendor_pseudo_plan(int n, std::int64_t batch) {
    LaunchPlan p;
    p.fits = n >= 1 && batch >= 1;
    const int ne = (n <= 16) ? n : 16 * ((n + 15) / 16);
    p.launches = 1;
    p.wave_launches = 1;
    p.groups = 1;
    p.resident_groups_per_cu = 1;
    p.flops = useful_flops<T>(ne, batch);
    p.useful_flops = useful_flops<T>(n, batch);
    p.bytes = 2 * tri(ne) * kSzD<T> * static_cast<double>(batch);
    p.serial_steps = ne;
    p.additive_work = true;
    p.item_steps = static_cast<double>(ne) * static_cast<double>(batch);
    return p;
}

}  // namespace batchlas::potrf_plan
