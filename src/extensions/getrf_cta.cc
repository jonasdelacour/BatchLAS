// Native batched GETRF: the CTA tier and the panel leaf both tiers share. The
// tile is staged into local memory, factorised by ?GETF2's right-looking rank-1
// recurrence with partial pivoting, and stored back. The device body lives in
// getrf_cta_device.hh because getrf_blocked.cc's panel step runs the SAME code
// from global memory, so a fix must not miss one residency. preferred() is false
// for both native arms. evidence: docs/perf/lu.md#getrf-window-evidence

#include "getrf_native.hh"
#include "getrf_cta_device.hh"

#include "../queue.hh"
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

// local_mem_size minus the standard 4096 B reserve. It serves the convenience
// overload only -- every real decision re-reads the device, not device_limits.hh.
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

// The 48 KB launch hole: a local-memory request in (47104, 49664] is refused by
// the runtime, so a request inside the band is padded past it. Band and pad match
// potrf_cta.cc's and geqrf_cta.cc's byte for byte. This kernel uses no group
// collective and does not itself trip the hole; the pad is defensive.
// evidence: docs/perf/lu.md#the-48-kb-launch-hole
constexpr std::size_t kGetrfHoleLo = 47104;
constexpr std::size_t kGetrfHoleHi = 49664;
constexpr std::size_t kGetrfHolePadTo = 49920;

constexpr std::size_t getrf_hole_padded(std::size_t bytes) {
    return (bytes > kGetrfHoleLo && bytes <= kGetrfHoleHi) ? kGetrfHolePadTo : bytes;
}

// Work-group width: about four columns' worth of rows, clamped to [64, 512]. A
// heuristic, not a tuned table, and a pure performance knob -- getrf_slm_bytes
// does not depend on wg, so changing it cannot move a capacity. The 512 cap is
// deliberate; 1024 was measured. evidence: docs/perf/lu.md#negative-results
inline int getrf_leaf_wg(int m, int n, int max_wg) {
    const int cols = (n >= 4) ? 4 : ((n < 1) ? 1 : n);
    const std::int64_t target = static_cast<std::int64_t>(m) * cols;
    int wg = 32;
    while (wg < target && wg < 512) wg <<= 1;
    if (wg < 64) wg = 64;
    while (wg > max_wg && wg > 32) wg >>= 1;
    return wg;
}

template <typename T> class GetrfPanelResidentKernel;
template <typename T> class GetrfPanelGlobalKernel;

// The RESIDENT leaf: stage, factor in local memory, store.
template <typename T>
Event getrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  int* piv_ptr, int piv_stride, int piv_base,
                                  int32_t* info_ptr,
                                  int wg) {
    // std::complex is re-typed HERE, at the pointer boundary, and never enters the
    // kernel body: its Annex-G operator* costs an isnan branch and a library call.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    const int kmax = std::min(m, n);
    const int ldt = getrf_tile_ld(m);
    const std::size_t tile_elems =
        static_cast<std::size_t>(ldt) * static_cast<std::size_t>(n);

    // The allocation steps over the 48 KB hole; the body indexes only tile_elems,
    // so the pad is capacity, never data. Same arithmetic as getrf_leaf_fits.
    const std::size_t scratch = getrf_scratch_bytes<T>();
    const std::size_t raw = tile_elems * sizeof(D) + scratch;
    const std::size_t padded = getrf_hole_padded(raw);
    const std::size_t tile_alloc_elems = (padded - scratch + sizeof(D) - 1) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_alloc_elems), h);
        sycl::local_accessor<R, 1> rval(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        sycl::local_accessor<int, 1> ridx(
            sycl::range<1>(static_cast<std::size_t>(gn::kLuRedSlots)), h);
        h.parallel_for<GetrfPanelResidentKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());

                const int lwg = static_cast<int>(it.get_local_range(0));

                D* const src = ap + static_cast<std::ptrdiff_t>(b) * stride;
                const std::size_t used =
                    static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

                // Logical m x n extent scattered into the padded tile ld, hence
                // `used`. The 64-bit divisions stay; a power-of-two split lost.
                for (std::size_t e = static_cast<std::size_t>(tid); e < used;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    tile[static_cast<std::size_t>(r) +
                         static_cast<std::size_t>(c) * static_cast<std::size_t>(ldt)] =
                        src[static_cast<std::ptrdiff_t>(r) +
                            static_cast<std::ptrdiff_t>(c) * ld];
                }
                sycl::group_barrier(it.get_group());          // B0

                gn::LuLocalTile<D, sycl::local_accessor<D, 1>> A{tile, ldt};
                gn::getf2_panel_device<D>(
                    it, A, m, n, kmax,
                    piv_ptr + static_cast<std::ptrdiff_t>(b) * piv_stride + piv_base,
                    piv_base,
                    info_ptr + b,
                    rval, ridx);

                sycl::group_barrier(it.get_group());          // B5
                for (std::size_t e = static_cast<std::size_t>(tid); e < used;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    src[static_cast<std::ptrdiff_t>(r) +
                        static_cast<std::ptrdiff_t>(c) * ld] =
                        tile[static_cast<std::size_t>(r) +
                             static_cast<std::size_t>(c) * static_cast<std::size_t>(ldt)];
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
                gn::getf2_panel_device<D>(
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

template <typename T>
bool getrf_cta_fits(int n, std::size_t slm_budget_bytes) {
    return getrf_leaf_fits<T>(n, n, slm_budget_bytes);
}

// The walk's `break` is LOAD-BEARING: getrf_hole_padded is non-monotone once a
// pad exists, so the largest fitting n is not the largest n below the first miss
// unless the walk stops there. 0 spells "this tier is not in this build".
template <typename T>
int getrf_cta_max_n_for_slm(std::size_t slm_budget_bytes) {
    using DM = sycl_device::DevMap<T>;
    const std::size_t scratch = getrf_scratch_bytes<T>();
    if (slm_budget_bytes <= scratch) return 0;
    const double cap = static_cast<double>((slm_budget_bytes - scratch) /
                                           sizeof(typename DM::type));
    const int hi = static_cast<int>(std::sqrt(cap)) + 2;
    int best = 0;
    for (int n = 1; n <= hi; ++n) {
        if (!getrf_cta_fits<T>(n, slm_budget_bytes)) break;
        best = n;
    }
    return best;
}

template <typename T>
int getrf_cta_max_n() {
    return getrf_cta_max_n_for_slm<T>(kGetrfReferenceSlmBudget);
}

// ONE term: the fallback `info` span, for a caller that supplied none (an empty
// OR SHORT span means "not requested"). A may carry a null data_ptr() here.
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
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int wg = getrf_leaf_wg(m, n, max_wg);

    const bool resident = getrf_leaf_fits<T>(m, n, budget);
    if (used_resident_out) *used_resident_out = resident;

    if (resident) {
        return getrf_panel_resident_launch<T>(ctx, a_ptr, ld, stride, m, n, batch,
                                              piv_ptr, piv_stride, piv_base, info_ptr, wg);
    }
    return getrf_panel_global_launch<T>(ctx, a_ptr, ld, stride, m, n, batch,
                                        piv_ptr, piv_stride, piv_base, info_ptr, wg);
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
        throw std::invalid_argument("getrf_cta: degenerate extents");
    }
    if (m != n) {
        // Contract, not fit: supports() refuses m != n and the two must agree.
        throw std::invalid_argument(
            "getrf_cta: A must be square (route_getrf.hh's supports() refuses m != n)");
    }
    if (A.is_heterogeneous()) {
        // One (n, ld, stride) tuple at CAPACITY extents covers the whole batch.
        throw std::invalid_argument("getrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never MAX_SUB_GROUP_SIZE >= 32: that property reports the
        // FIRST supported size, so the weak test accepts a {64}-only device here.
        throw std::runtime_error(
            "getrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    // int64 ON THE WIRE, PACKED 1-BASED int32 IN THE BUFFER: cuBLAS and rocSOLVER
    // reinterpret_cast this span, so any other format is silent garbage downstream.
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrf_cta: pivot span is shorter than n * batch");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    if (!getrf_cta_fits<T>(n, budget)) {
        throw std::invalid_argument(
            "getrf_cta: order " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(getrf_slm_bytes<T>(n, n)) + " B of " +
            std::to_string(budget) + " B); the ceiling for this type is " +
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
        throw std::logic_error(
            "getrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- getrf_cta_fits and getrf_panel_factorize disagree");
    }
    return e;
}

// Per scalar type only, no Backend cross-product: this build is device-link-bound.
#define BATCHLAS_GETRF_CTA_INSTANTIATE(T)                                                     \
    template int getrf_cta_max_n_for_slm<T>(std::size_t);                                     \
    template int getrf_cta_max_n<T>();                                                        \
    template bool getrf_cta_fits<T>(int, std::size_t);                                        \
    template bool getrf_leaf_fits<T>(int, int, std::size_t);                                  \
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
