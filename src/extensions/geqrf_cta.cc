// Native batched GEQRF: the CTA tier, and the panel leaf both native tiers share.
// The device body is in geqrf_cta_device.hh because geqrf_blocked.cc's panel step
// runs the SAME code against a global accessor -- correctness fixes belong there.
// Route-neutral: preferred() is false for both native arms, so this is reachable
// only via BATCHLAS_GEQRF_ROUTE or a vendor-free build (docs/perf/qr.md#route-arms).

#include "geqrf_native.hh"
#include "geqrf_cta_device.hh"

#include "../queue.hh"
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

// Reference budget for the convenience capacity overloads only; every real
// decision reads local_mem_size from the device. NOT device_limits.hh's 49152,
// which cmake hardcodes for any nvidia_gpu_sm_* target and is 2.06x wrong here.
constexpr std::size_t kGeqrfReferenceSlmBudget = 97280;

// Exactly m*n scalars, with NO leading-dimension padding: both hot access patterns
// are bank-conflict-free at any ld, and the absence of a pad is what keeps
// geqrf_cta_max_elems_for_slm monotone.
template <typename T>
constexpr std::size_t geqrf_slm_bytes(int64_t m, int64_t n) {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(T);
}

// The 48 KiB launch hole: an allocation in (kGeqrfHoleLo, kGeqrfHoleHi] fails to
// launch once geqr2_panel_device's two reduce_over_group calls add static shared,
// so such a request is padded past the band. Order-dependent -- the attribute is
// sticky per CUfunction -- so a suite can be green by launch order alone.
// evidence: docs/perf/qr.md#the-48-kib-launch-hole
constexpr std::size_t kGeqrfHoleLo = 47104;
constexpr std::size_t kGeqrfHoleHi = 49664;
constexpr std::size_t kGeqrfHolePadTo = 49920;

constexpr std::size_t geqrf_hole_padded(std::size_t bytes) {
    return (bytes > kGeqrfHoleLo && bytes <= kGeqrfHoleHi) ? kGeqrfHolePadTo : bytes;
}

// A budget inside the band cannot host a tile inside it, so clamp to just below --
// the table's `m*n <= cta_max_elems` gate and geqrf_cta_fits must not disagree.
constexpr std::size_t geqrf_hole_safe_budget(std::size_t budget) {
    return (budget > kGeqrfHoleLo && budget < kGeqrfHolePadTo) ? kGeqrfHoleLo : budget;
}

// 32 * teams work-items, one work-group per matrix; teams track COLUMNS and lanes
// track ROWS, matching the apply in geqrf_cta_device.hh. Not a tuned number.
inline int geqrf_panel_wg(int n, int max_wg) {
    int teams = 1;
    while (teams < 8 && teams < n) teams *= 2;
    int wg = teams * 32;
    while (wg > max_wg && wg > 32) wg /= 2;
    return wg;
}

template <typename T> class GeqrfPanelResidentKernel;
template <typename T> class GeqrfPanelGlobalKernel;

template <typename T>
Event geqrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  T* tau_ptr, int tau_batch_stride, int tau_offset,
                                  int wg) {
    // std::complex is re-typed to the POD device scalar HERE and never enters the
    // kernel body: its operator* is Annex-G conformant (isnan branch plus libcall).
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    const int kmax = std::min(m, n);
    const std::size_t tile_elems =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    // The allocation steps over the launch hole; the body still indexes only
    // tile_elems, so the pad is capacity and never data.
    const std::size_t tile_alloc_elems =
        geqrf_hole_padded(tile_elems * sizeof(D)) / sizeof(D);

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<D, 1> tile(sycl::range<1>(tile_alloc_elems), h);
        h.parallel_for<GeqrfPanelResidentKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) *
                                             static_cast<std::size_t>(wg)),
                              sycl::range<1>(static_cast<std::size_t>(wg))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int b = static_cast<int>(it.get_group_linear_id());
                const int tid = static_cast<int>(it.get_local_linear_id());
                const int lwg = static_cast<int>(it.get_local_range(0));

                D* const src = ap + static_cast<std::ptrdiff_t>(b) * stride;

                for (std::size_t e = static_cast<std::size_t>(tid); e < tile_elems;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    tile[e] = src[static_cast<std::ptrdiff_t>(r) +
                                  static_cast<std::ptrdiff_t>(c) * ld];
                }
                sycl::group_barrier(it.get_group());          // B0

                gn::GeqrfLocalTile<D, sycl::local_accessor<D, 1>> A{tile, m};
                gn::geqr2_panel_device<D>(
                    it, A, m, n, kmax,
                    tp + static_cast<std::ptrdiff_t>(b) * tau_batch_stride + tau_offset);

                sycl::group_barrier(it.get_group());          // B4
                for (std::size_t e = static_cast<std::size_t>(tid); e < tile_elems;
                     e += static_cast<std::size_t>(lwg)) {
                    const int r = static_cast<int>(e % static_cast<std::size_t>(m));
                    const int c = static_cast<int>(e / static_cast<std::size_t>(m));
                    src[static_cast<std::ptrdiff_t>(r) +
                        static_cast<std::ptrdiff_t>(c) * ld] = tile[e];
                }
            });
    });
    return ctx.get_event();
}

// The GLOBAL leaf: the same device body streamed from global memory, for panels too
// tall to hold resident. NO LOCAL STAGING OF v: every team re-reads column j, which
// looks redundant but is an L1 broadcast of one contiguous column read by all teams
// at once; staging it would need a chunked SLM pipeline of an unbounded column.
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
                gn::geqr2_panel_device<D>(
                    it, A, m, n, kmax,
                    tp + static_cast<std::ptrdiff_t>(b) * tau_batch_stride + tau_offset);
            });
    });
    return ctx.get_event();
}

}  // namespace

// CAPABILITY. The capacity is an AREA: the tile is m*n scalars, so per-extent
// ceilings would admit a 155 x 155 float panel needing many times the budget.
// cta_max_m is that same bound at n = 1 today, kept separate because it is what
// moves if a per-row resident array is added; a speed threshold in supports() would
// remove the vendor-free route above it. evidence: docs/perf/qr.md#cta-capacity
template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes) {
    return static_cast<int64_t>(geqrf_hole_safe_budget(slm_budget_bytes) / sizeof(T));
}

template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes) {
    const int64_t e = geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes);
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

// The ONE fit predicate: the table's capacity and the blocked driver must agree.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes) {
    if (m < 1 || n < 1) return false;
    const int64_t elems = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    return static_cast<int64_t>(m) <= static_cast<int64_t>(geqrf_cta_max_m_for_slm<T>(slm_budget_bytes)) &&
           elems <= geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes) &&
           geqrf_hole_padded(geqrf_slm_bytes<T>(m, n)) <= slm_budget_bytes;
}

// WORKSPACE. This tier needs none. Zero still comes from a measuring() replay, not
// a literal, so the facade's max(native, vendor) compares required_bytes figures.
// Must stay monotone in (rows, cols, batch) and must dereference neither A nor tau:
// both are null when band_reduction.cc sizes.
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
    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    const int max_wg = static_cast<int>(dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int wg = geqrf_panel_wg(n, max_wg);

    const bool resident = geqrf_cta_fits<T>(m, n, budget);
    if (used_resident_out) *used_resident_out = resident;

    if (resident) {
        return geqrf_panel_resident_launch<T>(ctx, a_ptr, ld, stride, m, n, batch,
                                              tau_ptr, tau_batch_stride, tau_offset, wg);
    }
    return geqrf_panel_global_launch<T>(ctx, a_ptr, ld, stride, m, n, batch,
                                        tau_ptr, tau_batch_stride, tau_offset, wg);
}

// The CTA tier's direct entry point. Every gate supports() applies to the CTA arm is
// re-applied here, because this is reachable WITHOUT the table: a forced route whose
// gate disagrees falls through to the vendor and passes green over a dead kernel.
template <typename T>
Event geqrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace) {
    static_cast<void>(workspace);   // this tier needs none

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("geqrf_cta: degenerate extents");
    }
    if (m < n) {
        // Correctness, not fit: supports() refuses m < n for both native arms and
        // the two must agree, or a forced route reaches a shape promised the vendor.
        throw std::invalid_argument(
            "geqrf_cta: m < n is not supported (route_geqrf.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with a single (m, n, ld, stride) tuple, so
        // per-item active dims would factorise the wrong extents after item 0.
        throw std::invalid_argument("geqrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("geqrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that returns the
        // FIRST supported size, so the weak test accepts a {64} device, which is a
        // launch abort for a kernel carrying reqd_sub_group_size(32).
        throw std::runtime_error(
            "geqrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    const std::size_t k = static_cast<std::size_t>(std::min(m, n));
    if (tau.size() < k * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("geqrf_cta: tau span is shorter than k * batch");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    if (!geqrf_cta_fits<T>(m, n, budget)) {
        throw std::invalid_argument(
            "geqrf_cta: " + std::to_string(m) + " x " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(geqrf_slm_bytes<T>(m, n)) + " B of " + std::to_string(budget) +
            " B); the element ceiling for this type is " +
            std::to_string(geqrf_cta_max_elems_for_slm<T>(budget)));
    }

    bool resident = false;
    Event e = geqrf_panel_factorize<T>(ctx, A.data_ptr(), A.ld(), A.stride(), m, n, batch,
                                       tau.data(), static_cast<int>(k), 0, &resident);
    if (!resident) {
        // Unreachable: geqrf_cta_fits was just checked with the same budget and
        // arithmetic. Asserted because a silent tier swap passes a pinned-route test.
        throw std::logic_error(
            "geqrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- geqrf_cta_fits and geqrf_panel_factorize disagree");
    }
    return e;
}

// Per scalar type only, no Backend cross-product: the kernel has no vendor dependency.
#define BATCHLAS_GEQRF_CTA_INSTANTIATE(T)                                                     \
    template int geqrf_cta_max_m_for_slm<T>(std::size_t);                                     \
    template int64_t geqrf_cta_max_elems_for_slm<T>(std::size_t);                             \
    template int geqrf_cta_max_m<T>();                                                        \
    template int64_t geqrf_cta_max_elems<T>();                                                \
    template bool geqrf_cta_fits<T>(int, int, std::size_t);                                   \
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
