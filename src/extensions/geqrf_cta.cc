// Native batched GEQRF: the CTA tier, and the panel leaf both native tiers share. The device
// body lives in geqrf_cta_device.hh because geqrf_blocked.cc's panel step runs the SAME code
// against a global accessor -- correctness fixes belong there. preferred() ships a per-type
// order-floor plus tall-panel window and best_native_tier() can resolve it to THIS arm, so it
// is reachable in a vendor build, not only vendor-free or under a pin.
// evidence: docs/perf/qr.md#route-arms

#include "geqrf_native.hh"
#include "geqrf_cta_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_geqrf {

namespace {

namespace gn = ::batchlas::geqrf_native;

// Convenience capacity overloads only; every real decision reads LOCAL_MEM_SIZE from the
// device, never device_limits.hh's hardcoded constant. evidence: docs/perf/qr.md#cta-capacity
constexpr std::size_t kGeqrfReferenceSlmBudget = 97280;

// Exactly m*n scalars with NO leading-dimension padding: both hot access patterns are
// bank-conflict-free at any ld, and the missing pad keeps the element ceiling monotone.
template <typename T>
constexpr std::size_t geqrf_slm_bytes(int64_t m, int64_t n) {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(T);
}

// Allocations in (kGeqrfHoleLo, kGeqrfHoleHi] fail to launch once geqr2_panel_device's
// reduce_over_group calls add static shared, so such a request is padded past the band.
// The attribute is sticky per CUfunction, so a suite can be green by launch order alone.
// evidence: docs/perf/qr.md#the-48-kib-launch-hole
constexpr std::size_t kGeqrfHoleLo = 47104;
constexpr std::size_t kGeqrfHoleHi = 49664;
constexpr std::size_t kGeqrfHolePadTo = 49920;

constexpr std::size_t geqrf_hole_padded(std::size_t bytes) {
    return (bytes > kGeqrfHoleLo && bytes <= kGeqrfHoleHi) ? kGeqrfHolePadTo : bytes;
}

// A budget inside the band cannot host a tile inside it, so clamp to just below: the
// table's `m*n <= cta_max_elems` gate and geqrf_cta_fits must not disagree.
constexpr std::size_t geqrf_hole_safe_budget(std::size_t budget) {
    return (budget > kGeqrfHoleLo && budget < kGeqrfHolePadTo) ? kGeqrfHoleLo : budget;
}

// One work-group per matrix, 32 * teams work-items; teams track COLUMNS and lanes track
// ROWS, matching the apply in geqrf_cta_device.hh.
inline int geqrf_panel_wg(int n, int max_wg) {
    int teams = 1;
    while (teams < 8 && teams < n) teams *= 2;
    int wg = teams * 32;
    while (wg > max_wg && wg > 32) wg /= 2;
    return wg;
}

// The band in which one sub-group is a sensible unit of work for a whole panel, and so
// the band in which G-packing is offered. Above it the panel wants the several teams
// geqrf_panel_wg gives it, and with them work-group barriers.
inline bool geqrf_packable(int m, int n) { return m <= 32 && n <= 32; }

// Panels per work-group, and the scope that makes that number correct: G > 1 demands
// GeqrfScope::SubGroup, so the two are derived together.
struct GeqrfLeafLaunch {
    int wg = 32;
    int G = 1;
    bool packed = false;
};

template <typename T>
GeqrfLeafLaunch geqrf_leaf_launch(int m, int n, std::size_t wg_slm_budget, int max_wg) {
    GeqrfLeafLaunch p;
    if (geqrf_packable(m, n)) {
        p.G = resident::pack_matrices_per_wg(
            geqrf_slm_bytes<T>(m, n), 32, wg_slm_budget, max_wg);
    }
    if (p.G > 1) {
        p.packed = true;
        p.wg = p.G * 32;
    } else {
        p.G = 1;
        p.wg = geqrf_panel_wg(n, max_wg);
    }
    return p;
}

template <typename T, gn::GeqrfScope SC> class GeqrfPanelResidentKernel;
template <typename T> class GeqrfPanelGlobalKernel;

template <typename T, gn::GeqrfScope SC>
Event geqrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  T* tau_ptr, int tau_batch_stride, int tau_offset,
                                  int wg, int G) {
    // std::complex is re-typed to the POD device scalar HERE and never enters the
    // kernel body: its operator* is Annex-G conformant (isnan branch plus libcall).
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    static_assert(SC == gn::GeqrfScope::SubGroup || SC == gn::GeqrfScope::WorkGroup);

    const int kmax = std::min(m, n);
    const std::size_t tile_elems =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    // The allocation steps over the launch hole; the body indexes only tile_elems per
    // panel, G panels side by side.
    const std::size_t tile_alloc_elems =
        geqrf_hole_padded(static_cast<std::size_t>(G) * tile_elems * sizeof(D)) / sizeof(D);
    const int num_wg = (batch + G - 1) / G;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_alloc_elems), h);
        h.parallel_for<GeqrfPanelResidentKernel<T, SC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int wg_id = static_cast<int>(it.get_group_linear_id());

                int b, slot, tid, lwg;
                if constexpr (SC == gn::GeqrfScope::SubGroup) {
                    const int sg_id = static_cast<int>(sg.get_group_linear_id());
                    b = wg_id * G + sg_id;
                    slot = sg_id;
                    tid = static_cast<int>(sg.get_local_linear_id());
                    lwg = static_cast<int>(sg.get_local_linear_range());
                    // Sub-group-uniform, and this scope executes only sub-group barriers.
                    if (b >= batch) return;
                } else {
                    b = wg_id;             // G == 1 => num_wg == batch
                    slot = 0;
                    tid = static_cast<int>(it.get_local_linear_id());
                    lwg = static_cast<int>(it.get_local_range(0));
                }

                D* const mine = &tile[0] + static_cast<std::ptrdiff_t>(slot) *
                                               static_cast<std::ptrdiff_t>(tile_elems);
                D* const src = ap + static_cast<std::ptrdiff_t>(b) * stride;

                for (std::size_t e = static_cast<std::size_t>(tid); e < tile_elems;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    mine[e] = src[static_cast<std::ptrdiff_t>(r) +
                                  static_cast<std::ptrdiff_t>(c) * ld];
                }
                if constexpr (SC == gn::GeqrfScope::SubGroup) {
                    sycl::group_barrier(sg);                  // B0
                } else {
                    sycl::group_barrier(it.get_group());      // B0
                }

                gn::GeqrfRawTile<D> A{mine, m};
                gn::geqr2_panel_device<D, SC>(
                    it, A, m, n, kmax,
                    tp + static_cast<std::ptrdiff_t>(b) * tau_batch_stride + tau_offset);

                if constexpr (SC == gn::GeqrfScope::SubGroup) {
                    sycl::group_barrier(sg);                  // B4
                } else {
                    sycl::group_barrier(it.get_group());      // B4
                }
                for (std::size_t e = static_cast<std::size_t>(tid); e < tile_elems;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    src[static_cast<std::ptrdiff_t>(r) +
                        static_cast<std::ptrdiff_t>(c) * ld] = mine[e];
                }
            });
    });
    return ctx.get_event();
}

// The GLOBAL leaf: the same device body streamed from global memory, for panels too tall
// to hold resident. Deliberately does NOT stage v in SLM -- every team re-reading column
// j is an L1 broadcast, and the column has no bounded size to stage.
template <typename T>
Event geqrf_panel_global_launch(Queue& ctx,
                                T* a_ptr, int ld, int stride,
                                int m, int n, int batch,
                                T* tau_ptr, int tau_batch_stride, int tau_offset,
                                int wg) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    const int kmax = std::min(m, n);

    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<GeqrfPanelGlobalKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int b = static_cast<int>(it.get_group_linear_id());
                gn::GeqrfGlobalTile<D> A{ap + static_cast<std::ptrdiff_t>(b) * stride, ld};
                // Always WorkGroup: this leaf exists for panels one sub-group cannot hold.
                gn::geqr2_panel_device<D, gn::GeqrfScope::WorkGroup>(
                    it, A, m, n, kmax,
                    tp + static_cast<std::ptrdiff_t>(b) * tau_batch_stride + tau_offset);
            });
    });
    return ctx.get_event();
}

}  // namespace

// CAPABILITY. The capacity is an AREA -- the tile is m*n scalars, so per-extent ceilings
// would admit panels needing many times the budget. A speed threshold here rather than in
// preferred() would remove the vendor-free route. evidence: docs/perf/qr.md#cta-capacity
// The occupancy rule enters as a division of the budget, and the hole clamp is applied
// AFTER it: a scaled budget can land inside the band even when the whole one did not,
// and a budget inside the band cannot host a tile inside it.
template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    const std::size_t wg_budget = geqrf_hole_safe_budget(
        resident::occupancy_budget(slm_budget_bytes, min_blocks_per_sm));
    return static_cast<int64_t>(wg_budget / sizeof(T));
}

template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    const int64_t e = geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes, min_blocks_per_sm);
    return static_cast<int>(std::min<int64_t>(e, 0x7fffffff));
}

template <typename T>
int geqrf_cta_max_m() {
    return geqrf_cta_max_m_for_slm<T>(kGeqrfReferenceSlmBudget);
}

template <typename T>
int64_t geqrf_cta_max_elems() {
    return geqrf_cta_max_elems_for_slm<T>(kGeqrfReferenceSlmBudget);
}

// The TIER's fit predicate, occupancy-scaled by default: the table's capacity, the CTA
// entry point's gate and this must be one predicate, or a routed shape fails at enqueue.
// The blocked driver asks geqrf_leaf_fits instead -- see below.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    if (m < 1 || n < 1) return false;
    const int64_t elems = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    const std::size_t wg_budget =
        resident::occupancy_budget(slm_budget_bytes, min_blocks_per_sm);
    return static_cast<int64_t>(m) <=
               static_cast<int64_t>(geqrf_cta_max_m_for_slm<T>(slm_budget_bytes,
                                                               min_blocks_per_sm)) &&
           elems <= geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes, min_blocks_per_sm) &&
           geqrf_hole_padded(geqrf_slm_bytes<T>(m, n)) <= wg_budget;
}

// The RESIDENCY predicate, at the whole budget: "can this panel be held in local memory
// at all". The blocked driver's leading panel is chosen with it, because a panel that
// stops being resident streams from global memory -- a large-n regression, not an
// occupancy win. evidence: docs/perf/qr.md#the-panel-leaf-is-not-the-tier-ceiling
template <typename T>
bool geqrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes) {
    return geqrf_cta_fits<T>(m, n, slm_budget_bytes, 1);
}

// WORKSPACE. This tier needs none, but the size must stay monotone in (rows, cols, batch)
// and must dereference neither A nor tau: both are null when band_reduction.cc sizes.
template <typename T>
std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A) {
    static_cast<void>(A);
    return workspace_bytes([&](BumpAllocator& p) {
        static_cast<void>(ctx);
        return &p;
    });
}

template <typename T>
Event geqrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            T* tau_ptr, int tau_batch_stride, int tau_offset,
                            bool* used_resident_out) {
    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    const bool resident = geqrf_leaf_fits<T>(m, n, budget);
    if (used_resident_out) *used_resident_out = resident;

    if (!resident) {
        return geqrf_panel_global_launch<T>(ctx, a_ptr, ld, stride, m, n, batch, tau_ptr,
                                            tau_batch_stride, tau_offset,
                                            geqrf_panel_wg(n, max_wg));
    }

    const auto p = geqrf_leaf_launch<T>(m, n, resident::occupancy_budget(budget), max_wg);
    if (p.packed) {
        return geqrf_panel_resident_launch<T, gn::GeqrfScope::SubGroup>(
            ctx, a_ptr, ld, stride, m, n, batch, tau_ptr, tau_batch_stride, tau_offset,
            p.wg, p.G);
    }
    return geqrf_panel_resident_launch<T, gn::GeqrfScope::WorkGroup>(
        ctx, a_ptr, ld, stride, m, n, batch, tau_ptr, tau_batch_stride, tau_offset,
        p.wg, 1);
}

// Test hook: low 16 bits G (panels per work-group), high 16 the work-group width;
// 0 when the panel is not resident. See geqrf_native.hh.
template <typename T>
unsigned geqrf_cta_debug_launch(Queue& ctx, int m, int n) {
    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    if (!geqrf_leaf_fits<T>(m, n, budget)) return 0u;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const auto p = geqrf_leaf_launch<T>(m, n, resident::occupancy_budget(budget), max_wg);
    return (static_cast<unsigned>(p.wg) << 16) | static_cast<unsigned>(p.G);
}

// The CTA tier's direct entry point. Every gate supports() applies to the CTA arm is
// re-applied here, because a forced route reaches this without the table.
template <typename T>
Event geqrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace) {
    static_cast<void>(workspace);

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("geqrf_cta: degenerate extents");
    }
    if (m < n) {
        throw batchlas::invalid_argument(
            "geqrf_cta: m < n is not supported (route_geqrf.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with a single (m, n, ld, stride) tuple.
        throw batchlas::invalid_argument("geqrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("geqrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, not MAX_SUB_GROUP_SIZE >= 32: that returns the FIRST supported
        // size, so the weak test accepts a {64} device and the launch aborts.
        throw batchlas::unsupported(
            "geqrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    const std::size_t k = static_cast<std::size_t>(std::min(m, n));
    if (tau.size() < k * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("geqrf_cta: tau span is shorter than k * batch");
    }

    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    // The occupancy-scaled gate, matching supports().
    if (!geqrf_cta_fits<T>(m, n, budget)) {
        throw batchlas::invalid_argument(
            "geqrf_cta: " + std::to_string(m) + " x " + std::to_string(n) +
            " does not fit this device's per-work-group local-memory budget (needs " +
            std::to_string(geqrf_slm_bytes<T>(m, n)) + " B of " +
            std::to_string(resident::occupancy_budget(budget)) +
            " B); the element ceiling for this type is " +
            std::to_string(geqrf_cta_max_elems_for_slm<T>(budget)));
    }

    bool resident = false;
    Event e = geqrf_panel_factorize<T>(ctx, A.data_ptr(), A.ld(), A.stride(), m, n, batch,
                                       tau.data(), static_cast<int>(k), 0, &resident);
    if (!resident) {
        // Unreachable; asserted because a silent tier swap passes a pinned-route test.
        throw batchlas::internal_error(
            "geqrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- geqrf_cta_fits and geqrf_panel_factorize disagree");
    }
    return e;
}

// Per scalar type only, no Backend cross-product: the kernel has no vendor dependency.
#define BATCHLAS_GEQRF_CTA_INSTANTIATE(T)                                                     \
    template int geqrf_cta_max_m_for_slm<T>(std::size_t, int);                                \
    template int64_t geqrf_cta_max_elems_for_slm<T>(std::size_t, int);                        \
    template int geqrf_cta_max_m<T>();                                                        \
    template int64_t geqrf_cta_max_elems<T>();                                                \
    template bool geqrf_cta_fits<T>(int, int, std::size_t, int);                              \
    template bool geqrf_leaf_fits<T>(int, int, std::size_t);                                  \
    template unsigned geqrf_cta_debug_launch<T>(Queue&, int, int);                            \
    template std::size_t geqrf_cta_buffer_size<T>(Queue&,                                     \
                                                  const MatrixView<T, MatrixFormat::Dense>&); \
    template Event geqrf_panel_factorize<T>(Queue&, T*, int, int, int, int, int, T*, int,     \
                                            int, bool*);                                      \
    template Event geqrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,   \
                                         Span<T>, Span<std::byte>);

BATCHLAS_GEQRF_CTA_INSTANTIATE(float)
BATCHLAS_GEQRF_CTA_INSTANTIATE(double)
BATCHLAS_GEQRF_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_GEQRF_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GEQRF_CTA_INSTANTIATE

}  // namespace sycl_geqrf
}  // namespace batchlas
