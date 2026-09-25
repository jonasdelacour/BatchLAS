// Native batched GETRF: the CTA tier and the panel leaf both tiers share -- stage the tile into
// local memory, factor by ?GETF2's right-looking rank-1 recurrence with partial pivoting, store
// back. The device body lives in getrf_cta_device.hh because getrf_blocked.cc's panel step runs
// the SAME code from global memory, so a fix must not miss one residency. preferred()'s shipped
// window admits Blocked ONLY (`r.algo != Blocked` returns false), so this arm is reached by the
// vendor-free walk or a pin, never by a vendor build's preference.
// evidence: docs/perf/lu.md#getrf-window-evidence

#include "getrf_native.hh"
#include "getrf_cta_device.hh"

#include "../queue.hh"
#include "../util/resident_capacity.hh"
#include "../util/template-instantiations.hh"

#include <batchlas/util/mempool.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace batchlas {
namespace sycl_getrf {

namespace {

namespace gn = ::batchlas::getrf_native;

// Convenience overloads only; every real decision re-reads the device, never
// device_limits.hh. evidence: docs/perf/lu.md#one-spelling-per-ceiling
constexpr std::size_t kGetrfReferenceSlmBudget = 97280;

// ODD ld, unlike geqrf's packed ld = m: the row exchange walks a row at stride
// ld, and an even ld puts every work-item of it in the same local-memory bank.
constexpr int getrf_tile_ld(int m) { return m | 1; }

// Tile plus the pivot search's slots, and CONSTANT IN THE WORK-GROUP WIDTH: the
// capacity query, the fit predicate and the launcher each pick their own wg.
template <typename T>
constexpr std::size_t getrf_scratch_bytes() {
    using DM = sycl_device::DevMap<T>;
    return static_cast<std::size_t>(gn::kLuRedSlots) *
           (sizeof(typename DM::real) + sizeof(int));
}

template <typename T>
constexpr std::size_t getrf_slm_bytes(int m, int n) {
    using DM = sycl_device::DevMap<T>;
    return static_cast<std::size_t>(getrf_tile_ld(m)) * static_cast<std::size_t>(n) *
               sizeof(typename DM::type) +
           getrf_scratch_bytes<T>();
}

// The 48 KB launch hole: a request inside this band is refused at enqueue, so it is padded past.
// Band and pad must stay byte-identical to potrf_cta.cc's and geqrf_cta.cc's. No group collective
// here, so the pad is defensive -- adding one arms it. evidence: docs/perf/lu.md#the-48-kb-launch-hole
constexpr std::size_t kGetrfHoleLo = 47104;
constexpr std::size_t kGetrfHoleHi = 49664;
constexpr std::size_t kGetrfHolePadTo = 49920;

constexpr std::size_t getrf_hole_padded(std::size_t bytes) {
    return (bytes > kGetrfHoleLo && bytes <= kGetrfHoleHi) ? kGetrfHolePadTo : bytes;
}

// Work-group width: about four columns' worth of rows, clamped to [64, 512]. A pure performance
// knob -- getrf_slm_bytes does not depend on wg, so changing it cannot move a capacity -- but the
// 512 cap is deliberate, not a default. evidence: docs/perf/lu.md#negative-results
inline int getrf_leaf_wg(int m, int n, int max_wg) {
    const int cols = (n >= 4) ? 4 : ((n < 1) ? 1 : n);
    const std::int64_t target = static_cast<std::int64_t>(m) * cols;
    int wg = 32;
    while (wg < target && wg < 512) wg <<= 1;
    if (wg < 64) wg = 64;
    while (wg > max_wg && wg > 32) wg >>= 1;
    return wg;
}

// The band in which ONE sub-group is a sensible unit of work for a whole panel, hence the band
// in which G-packing is offered. Above it the panel wants getrf_leaf_wg's wider work-group and
// with it work-group barriers.
inline bool getrf_packable(int m, int n) { return m <= 32 && n <= 32; }

// Matrices per work-group, and the scope that makes that number correct. G > 1 demands
// LuScope::SubGroup; the two are derived here together so no caller can pair them wrong.
struct GetrfLeafLaunch {
    int wg = 64;
    int G = 1;
    bool packed = false;
};

template <typename T>
GetrfLeafLaunch getrf_leaf_launch(int m, int n, std::size_t wg_slm_budget, int max_wg) {
    GetrfLeafLaunch p;
    if (getrf_packable(m, n)) {
        p.G = resident::pack_matrices_per_wg(getrf_slm_bytes<T>(m, n), 32,
                                             wg_slm_budget, max_wg);
    }
    if (p.G > 1) {
        p.packed = true;
        p.wg = p.G * 32;
    } else {
        p.G = 1;
        p.wg = getrf_leaf_wg(m, n, max_wg);
    }
    return p;
}

template <typename T, gn::LuScope SC> class GetrfPanelResidentKernel;
template <typename T> class GetrfPanelGlobalKernel;

// The RESIDENT leaf: stage, factor in local memory, store. Under SubGroup scope G matrices
// share the work-group, one per sub-group; G == 1 keeps work-group scope and the wider group.
template <typename T, gn::LuScope SC>
Event getrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  int* piv_ptr, int piv_stride, int piv_base,
                                  int32_t* info_ptr,
                                  int wg, int G) {
    // std::complex is re-typed HERE, at the pointer boundary, and never enters the
    // kernel body: its Annex-G operator* costs an isnan branch and a library call.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    static_assert(SC == gn::LuScope::SubGroup || SC == gn::LuScope::WorkGroup);

    D* const ap = reinterpret_cast<D*>(a_ptr);
    const int kmax = std::min(m, n);
    const int ldt = getrf_tile_ld(m);
    const std::size_t tile_elems =
        static_cast<std::size_t>(ldt) * static_cast<std::size_t>(n);

    // The allocation steps over the 48 KB hole; the body indexes only tile_elems, so the pad is
    // capacity, never data. Same arithmetic as getrf_leaf_fits with G tiles side by side; the
    // argmax slots are shared because SubGroup scope, the only G > 1 scope, never reads them.
    const std::size_t scratch = getrf_scratch_bytes<T>();
    const std::size_t raw =
        static_cast<std::size_t>(G) * tile_elems * sizeof(D) + scratch;
    const std::size_t padded = getrf_hole_padded(raw);
    const std::size_t tile_alloc_elems = (padded - scratch + sizeof(D) - 1) / sizeof(D);
    const int num_wg = (batch + G - 1) / G;

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_alloc_elems), h);
        sycl::local_accessor<R, 1> rval(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        sycl::local_accessor<int, 1> ridx(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        h.parallel_for<GetrfPanelResidentKernel<T, SC>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(num_wg) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto sg = it.get_sub_group();
                const int wg_id = static_cast<int>(it.get_group_linear_id());

                int b, slot, tid, lwg;
                if constexpr (SC == gn::LuScope::SubGroup) {
                    const int sg_id = static_cast<int>(sg.get_group_linear_id());
                    b = wg_id * G + sg_id;
                    slot = sg_id;
                    tid = static_cast<int>(sg.get_local_linear_id());
                    lwg = static_cast<int>(sg.get_local_linear_range());
                    // Sub-group-uniform, and this scope executes only sub-group
                    // barriers -- an early return under a work-group barrier hangs.
                    if (b >= batch) return;
                } else {
                    b = wg_id;             // G == 1 => num_wg == batch
                    slot = 0;
                    tid = static_cast<int>(it.get_local_linear_id());
                    lwg = static_cast<int>(it.get_local_range(0));
                }

                D* const mine = &tile[0] + static_cast<std::ptrdiff_t>(slot) *
                                               static_cast<std::ptrdiff_t>(ldt) * n;
                D* const src = ap + static_cast<std::ptrdiff_t>(b) * stride;
                const std::size_t used =
                    static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

                // Logical m x n extent scattered into the padded tile ld, hence `used`.
                for (std::size_t e = static_cast<std::size_t>(tid); e < used;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    mine[static_cast<std::ptrdiff_t>(r) +
                         static_cast<std::ptrdiff_t>(c) * ldt] =
                        src[static_cast<std::ptrdiff_t>(r) +
                            static_cast<std::ptrdiff_t>(c) * ld];
                }
                if constexpr (SC == gn::LuScope::SubGroup) {
                    sycl::group_barrier(sg);                  // B0
                } else {
                    sycl::group_barrier(it.get_group());      // B0
                }

                gn::LuRawTile<D> A{mine, ldt};
                gn::getf2_panel_device<D, SC>(
                    it, A, m, n, kmax,
                    piv_ptr + static_cast<std::ptrdiff_t>(b) * piv_stride + piv_base,
                    piv_base,
                    info_ptr + b,
                    rval, ridx);

                if constexpr (SC == gn::LuScope::SubGroup) {
                    sycl::group_barrier(sg);                  // B5
                } else {
                    sycl::group_barrier(it.get_group());      // B5
                }
                for (std::size_t e = static_cast<std::size_t>(tid); e < used;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    src[static_cast<std::ptrdiff_t>(r) +
                        static_cast<std::ptrdiff_t>(c) * ld] =
                        mine[static_cast<std::ptrdiff_t>(r) +
                             static_cast<std::ptrdiff_t>(c) * ldt];
                }
            });
    });
    return ctx.get_event();
}

// The GLOBAL leaf: the same device body streamed from global memory, for the
// blocked driver's (m - j0) x nb panel, whose m the resident tile cannot hold.
template <typename T>
Event getrf_panel_global_launch(Queue& ctx,
                                T* a_ptr, int ld, int stride,
                                int m, int n, int batch,
                                int* piv_ptr, int piv_stride, int piv_base,
                                int32_t* info_ptr,
                                int wg) {
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    const int kmax = std::min(m, n);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<R, 1> rval(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        sycl::local_accessor<int, 1> ridx(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        h.parallel_for<GetrfPanelGlobalKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int b = static_cast<int>(it.get_group_linear_id());
                gn::LuGlobalTile<D> A{ap + static_cast<std::ptrdiff_t>(b) * stride, ld};
                // Always WorkGroup: the global leaf's panel is a full column block,
                // far taller than one sub-group should serve alone.
                gn::getf2_panel_device<D, gn::LuScope::WorkGroup>(
                    it, A, m, n, kmax,
                    piv_ptr + static_cast<std::ptrdiff_t>(b) * piv_stride + piv_base,
                    piv_base,
                    info_ptr + b,
                    rval, ridx);
            });
    });
    return ctx.get_event();
}

}  // namespace

// The ONE fit predicate the capacity query, the launcher and the blocked driver share.
template <typename T>
bool getrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes) {
    if (m < 1 || n < 1) return false;
    // int64 first: (m|1)*n overflows int at m ~ 46341, reachable as a panel height.
    const std::int64_t elems = static_cast<std::int64_t>(getrf_tile_ld(m)) *
                               static_cast<std::int64_t>(n);
    using DM = sycl_device::DevMap<T>;
    if (elems > static_cast<std::int64_t>(slm_budget_bytes / sizeof(typename DM::type))) {
        return false;
    }
    return getrf_hole_padded(getrf_slm_bytes<T>(m, n)) <= slm_budget_bytes;
}

// The TIER's admission test, and so occupancy-scaled by default: what supports()
// advertises and what getrf_cta_dispatch refuses must be one predicate. The blocked
// driver asks getrf_leaf_fits instead, at the whole budget -- a different question.
template <typename T>
bool getrf_cta_fits(int n, std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    return getrf_leaf_fits<T>(
        n, n, resident::occupancy_budget(slm_budget_bytes, min_blocks_per_sm));
}

// 0 spells "this tier is not in this build". The walk's `break` clause and why it is
// load-bearing are documented at resident_max_n.
template <typename T>
int getrf_cta_max_n_for_slm(std::size_t slm_budget_bytes, int min_blocks_per_sm) {
    using DM = sycl_device::DevMap<T>;
    const std::size_t wg_budget =
        resident::occupancy_budget(slm_budget_bytes, min_blocks_per_sm);
    const std::size_t scratch = getrf_scratch_bytes<T>();
    if (wg_budget <= scratch) return 0;
    // sqrt bound only: the walk still decides, and it must stop at the first miss.
    const double cap = static_cast<double>((wg_budget - scratch) /
                                           sizeof(typename DM::type));
    const int hi = static_cast<int>(std::sqrt(cap)) + 2;
    return resident::resident_max_n(
        [](int n) {
            return getrf_hole_padded(getrf_slm_bytes<T>(n, n));
        },
        slm_budget_bytes, min_blocks_per_sm, hi);
}

template <typename T>
int getrf_cta_max_n() {
    return getrf_cta_max_n_for_slm<T>(kGetrfReferenceSlmBudget);
}

// ONE term: the fallback `info` span for a caller that supplied none -- an empty OR SHORT span
// means "not requested". A may carry a null data_ptr() here.
namespace {

Span<int32_t> getrf_cta_layout(Queue& ctx, BumpAllocator& pool, int batch) {
    return pool.allocate<int32_t>(ctx, static_cast<std::size_t>(batch));
}

}  // namespace

template <typename T>
std::size_t getrf_cta_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    const int batch = static_cast<int>(A.batch_size());
    if (batch < 1) return 0;
    return workspace_bytes([&](BumpAllocator& p) {
        return getrf_cta_layout(ctx, p, batch);
    });
}

// The panel leaf, and the ONE place the residency is chosen.
template <typename T>
Event getrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            int* piv_ptr, int piv_stride, int piv_base,
                            int32_t* info_ptr,
                            bool* used_resident_out) {
    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    // The RESIDENCY question is asked at the whole budget: a blocked driver's panel that
    // stops being resident streams from global memory, which is a large-n regression, not
    // an occupancy win. The occupancy rule governs the advertised TIER capacity instead.
    const bool resident = getrf_leaf_fits<T>(m, n, budget);
    if (used_resident_out) *used_resident_out = resident;

    if (!resident) {
        return getrf_panel_global_launch<T>(ctx, a_ptr, ld, stride, m, n, batch, piv_ptr,
                                            piv_stride, piv_base, info_ptr,
                                            getrf_leaf_wg(m, n, max_wg));
    }

    // Packing is offered against the OCCUPANCY slice: G matrices share one work-group,
    // and the point of holding several is that the group stays small enough to be
    // resident several times over.
    const auto p = getrf_leaf_launch<T>(
        m, n, resident::occupancy_budget(budget), max_wg);
    if (p.packed) {
        return getrf_panel_resident_launch<T, gn::LuScope::SubGroup>(
            ctx, a_ptr, ld, stride, m, n, batch, piv_ptr, piv_stride, piv_base,
            info_ptr, p.wg, p.G);
    }
    return getrf_panel_resident_launch<T, gn::LuScope::WorkGroup>(
        ctx, a_ptr, ld, stride, m, n, batch, piv_ptr, piv_stride, piv_base,
        info_ptr, p.wg, 1);
}

// Test hook: low 16 bits G (matrices per work-group), high 16 the work-group width.
// 0 when the panel is not resident at all. See getrf_native.hh.
template <typename T>
unsigned getrf_cta_debug_launch(Queue& ctx, int m, int n) {
    const auto dev = ctx.device();
    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    if (!getrf_leaf_fits<T>(m, n, budget)) return 0u;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const auto p = getrf_leaf_launch<T>(m, n, resident::occupancy_budget(budget), max_wg);
    return (static_cast<unsigned>(p.wg) << 16) | static_cast<unsigned>(p.G);
}

// Direct entry point. Every supports() gate is re-applied here, because a forced
// route that fails one falls through to the vendor and passes green regardless.
template <typename T>
Event getrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<int64_t> pivots,
                         Span<std::byte> workspace,
                         Span<int32_t> info_out) {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || batch < 1) {
        throw batchlas::invalid_argument("getrf_cta: degenerate extents");
    }
    if (m != n) {
        // Contract, not fit: supports() refuses m != n and the two must agree.
        throw batchlas::invalid_argument(
            "getrf_cta: A must be square (route_getrf.hh's supports() refuses m != n)");
    }
    if (A.is_heterogeneous()) {
        // One (n, ld, stride) tuple at CAPACITY extents covers the whole batch.
        throw batchlas::invalid_argument("getrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw batchlas::invalid_argument("getrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports the
        // FIRST supported size, so the weak test accepts a {64}-only device here.
        throw batchlas::unsupported(
            "getrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    // int64 ON THE WIRE, PACKED 1-BASED int32 IN THE BUFFER: cuBLAS and rocSOLVER
    // reinterpret_cast this span, so any other format is silent garbage downstream.
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("getrf_cta: pivot span is shorter than n * batch");
    }

    const std::size_t budget = resident::device_slm_budget(
        dev.get_property(DeviceProperty::LOCAL_MEM_SIZE));
    // The occupancy-scaled gate, matching supports(): an order this refuses is one the
    // table never routes here, and one that would leave a single block resident per SM.
    if (!getrf_cta_fits<T>(n, budget)) {
        throw batchlas::invalid_argument(
            "getrf_cta: order " + std::to_string(n) +
            " does not fit this device's per-work-group local-memory budget (needs " +
            std::to_string(getrf_slm_bytes<T>(n, n)) + " B of " +
            std::to_string(resident::occupancy_budget(budget)) +
            " B); the ceiling for this type is " +
            std::to_string(getrf_cta_max_n_for_slm<T>(budget)));
    }

    BumpAllocator pool(workspace);
    // Empty or SHORT means "not requested"; the zero-fill below hits THIS span.
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : getrf_cta_layout(ctx, pool, batch);

    // The zero pre-pass is a read-after-write dependence, not a pure output:
    // getf2_panel_device READS info for first-failure-wins across panels, so on an
    // out-of-order queue the unguarded version returned the caller's own poison.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));
    if (!ctx.in_order()) ctx.wait();

    auto piv_i32 = pivots.as_span<int>();

    bool resident = false;
    Event e = getrf_panel_factorize<T>(ctx, A.data_ptr(), A.ld(), A.stride(), n, n, batch,
                                       piv_i32.data(), n, 0, info.data(), &resident);
    if (!resident) {
        // Unreachable: same budget and same arithmetic as the check above. Asserted
        // because a tier silently becoming the other is what a pinned test misses.
        throw batchlas::internal_error(
            "getrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- getrf_cta_fits and getrf_panel_factorize disagree");
    }
    return e;
}

// Per scalar type only, no Backend cross-product: this build is device-link-bound.
#define BATCHLAS_GETRF_CTA_INSTANTIATE(T)                                                     \
    template int getrf_cta_max_n_for_slm<T>(std::size_t, int);                                \
    template int getrf_cta_max_n<T>();                                                        \
    template bool getrf_cta_fits<T>(int, std::size_t, int);                                   \
    template bool getrf_leaf_fits<T>(int, int, std::size_t);                                  \
    template unsigned getrf_cta_debug_launch<T>(Queue&, int, int);                            \
    template std::size_t getrf_cta_buffer_size<T>(Queue&,                                     \
                                                  const MatrixView<T, MatrixFormat::Dense>&); \
    template Event getrf_panel_factorize<T>(Queue&, T*, int, int, int, int, int, int*, int,   \
                                            int, int32_t*, bool*);                            \
    template Event getrf_cta_dispatch<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,   \
                                         Span<int64_t>, Span<std::byte>, Span<int32_t>);

BATCHLAS_GETRF_CTA_INSTANTIATE(float)
BATCHLAS_GETRF_CTA_INSTANTIATE(double)
BATCHLAS_GETRF_CTA_INSTANTIATE(std::complex<float>)
BATCHLAS_GETRF_CTA_INSTANTIATE(std::complex<double>)

#undef BATCHLAS_GETRF_CTA_INSTANTIATE

}  // namespace sycl_getrf
}  // namespace batchlas
