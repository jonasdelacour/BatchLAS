// Native batched GEQRF -- the CTA tier, and the panel leaf both tiers share.
//
// The whole m x n matrix is staged into local memory once, factorised there by
// LAPACK ?GEQR2's right-looking Householder recurrence, and stored once. All of
// the device code is in geqrf_cta_device.hh, because geqrf_blocked.cc's panel
// step calls the SAME body against a global-memory accessor: `at(r, c)` is the
// only difference between the two residencies, so a correctness fix cannot land
// in one and miss the other. That sharing is why both TUs sit in ONE device-code
// cluster (src/extensions/CMakeLists.txt:15-27, W12).
//
// WHAT IS IN THIS FILE AND NOT IN THE HEADER:
//   * the two CAPABILITY answers, which are the SAME expressions the launcher
//     uses to size its local_accessor -- so the ceiling supports() advertises and
//     the allocation the kernel makes cannot disagree (route_trsm.hh:62-72, and
//     potrf_cta.cc:442-454 for what disagreement costs);
//   * geqrf_cta_fits, the single predicate both the table's capacity and the
//     blocked driver's per-panel choice go through;
//   * the two launchers and the ONE decision site between them
//     (geqrf_panel_factorize).
//
// PERFORMANCE STATUS: ROUTE-NEUTRAL, DELIBERATELY.
// RouteTable<Op::geqrf,T>::preferred() is false for both arms, so a
// vendor-present build keeps taking cuSOLVER for every shape and this kernel is
// reachable only through BATCHLAS_GEQRF_ROUTE, through geqrf_cta_dispatch, or in
// a vendor-free build (route_resolve.hh:60-63). Flipping preferred() is a later
// step gated on a measured grid.

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

// The standard per-work-group local-memory budget: the RUNTIME local_mem_size
// minus the 4096 B reserve cmake/BatchLASDetectSYCL.cmake:57-67 applies to every
// other device-BLAS sizing decision in this library. 97,280 on this box.
//
// NOT build/include/batchlas/device_limits.hh's 49152: that number is HARDCODED
// by cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern, the
// detection routine never queries local_mem_size at all, and it is 2.06x wrong
// here (WP4 finding W1). This constant is only ever used to answer the
// "at this repository's reference budget" convenience overloads; every real
// decision reads the device.
constexpr std::size_t kGeqrfReferenceSlmBudget = 97280;

// The local-memory footprint of a resident m x n panel, in bytes.
//
// EXACTLY m*n scalars, with NO padding on the leading dimension, and that is a
// shape decision recorded in GeqrfLocalTile: the two hot access patterns are
// (lane -> consecutive rows of one column) and (team -> a different column,
// lane -> consecutive rows), both bank-conflict-free for any ld. A pad would buy
// nothing and would cost capacity -- and, worse, would make the walk in
// geqrf_cta_max_elems_for_slm NON-MONOTONE, which is the hazard
// potrf_cta.cc:459-468 had to close with a load-bearing `break`.
template <typename T>
constexpr std::size_t geqrf_slm_bytes(int64_t m, int64_t n) {
    return static_cast<std::size_t>(m) * static_cast<std::size_t>(n) * sizeof(T);
}

// ---------------------------------------------------------------------------
// THE 48 KB LAUNCH HOLE, AND THE PAD THAT STEPS OVER IT.
//
// This is potrf_cta.cc:258-296's hole, and unlike potrf's THIS KERNEL IS IN IT.
// WP4 measured the band cold and wrote down the condition that reopens it: "the
// hole's width EQUALS the kernel's static shared memory ... it stays because
// static shared is not something this source controls: one group algorithm added
// anywhere in the body -- a reduce_over_group, which is exactly what the probe
// kernel used -- reintroduces the hole". geqr2_panel_device runs TWO
// sycl::reduce_over_group calls per reflector, so the resident leaf carries
// static shared and the interval is not empty here.
//
// MEASURED COLD, one process per point, all four scalar types, through the
// public facade (BATCHLAS_GEQRF_ROUTE=blocked, whose leading panel takes this
// leaf):
//
//     48,896 B  PASS      49,152 B  FAIL      49,664 B  PASS
//
//   cdouble  96x32 and 192x16 -> 49152 FAIL;  191x16 -> 48896 PASS
//   cfloat  192x32           -> 49152 FAIL;  191x32 -> 48896 PASS
//   double  384x16           -> 49152 FAIL
//   float   384x32           -> 49152 FAIL
//
// so it is a BYTE threshold, not a shape or a type accident: exactly 48 KiB is
// too big for CUDA's non-opt-in limit once static shared is added, and not big
// enough for the UR adapter to raise MaxDynamicSharedMemorySize.
//
// IT IS ORDER-DEPENDENT, WHICH IS WHY IT SHIPPED. The attribute is sticky per
// CUfunction and one instantiation serves every panel shape, so a process that
// launches ANY larger panel first never sees it again. tests/geqrf_tests.cc's
// BlockedResidualAndOrthogonality reaches 100x32 (51,200 B) before it reaches
// 96x32 (49,152 B) and is green either way; the failure only appeared when
// tests/orgqr_tests.cc asked for cdouble 96x96 as the first blocked shape in its
// process. Every guard in this file's suite was therefore blind to it by
// EXECUTION ORDER rather than by construction.
//
// The band and the pad target are potrf's, deliberately: same box, same cause,
// and a second set of numbers would be two things to keep true instead of one.
// Padding costs at most 256 B of local memory and only inside the band.
// ---------------------------------------------------------------------------
constexpr std::size_t kGeqrfHoleLo = 47104;
constexpr std::size_t kGeqrfHoleHi = 49664;
constexpr std::size_t kGeqrfHolePadTo = 49920;

constexpr std::size_t geqrf_hole_padded(std::size_t bytes) {
    return (bytes > kGeqrfHoleLo && bytes <= kGeqrfHoleHi) ? kGeqrfHolePadTo : bytes;
}

// A BUDGET that lands inside the band cannot host a tile that lands inside it --
// the pad would take that tile over the budget -- so the largest usable tile
// there is the one just below the band. Applied to the capacity queries so that
// the ROUTE TABLE's `m*n <= cta_max_elems` gate and geqrf_cta_fits' padded byte
// test cannot disagree, which is the potrf_cta.cc:442-454 defect (a raw figure in
// the query against a padded one in the launcher, producing an unhandled throw on
// a call the table had promised).
//
// INERT ON THIS BOX: the budget is 97,280 B, far above the band, so no capacity
// number moves and the four pinned ceilings are unchanged.
constexpr std::size_t geqrf_hole_safe_budget(std::size_t budget) {
    return (budget > kGeqrfHoleLo && budget < kGeqrfHolePadTo) ? kGeqrfHoleLo : budget;
}

// The work-group shape. 32 * teams work-items, one work-group per matrix.
//
// TEAMS track the COLUMN count and lanes track the ROW count, because that is
// the mapping the apply uses (see geqrf_cta_device.hh's nd_range note). Capped
// at 8 teams / 256 work-items: past that the teams outnumber the trailing
// columns of the late reflectors and the extra sub-groups only add barrier cost,
// while the local-memory footprint (up to the full budget) already limits this
// launch to one resident block per SM.
//
// It is NOT a tuned number and is not claimed to be: preferred() is false, so
// nothing routes here yet. It is the shape a measured grid would tune, and it is
// derived from n rather than hardcoded so that tuning it later is a change to
// this function alone.
inline int geqrf_panel_wg(int n, int max_wg) {
    int teams = 1;
    while (teams < 8 && teams < n) teams *= 2;
    int wg = teams * 32;
    while (wg > max_wg && wg > 32) wg /= 2;
    return wg;
}

template <typename T> class GeqrfPanelResidentKernel;
template <typename T> class GeqrfPanelGlobalKernel;

// ---------------------------------------------------------------------------
// The RESIDENT leaf: stage, factor in local memory, store.
// ---------------------------------------------------------------------------
template <typename T>
Event geqrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  T* tau_ptr, int tau_batch_stride, int tau_offset,
                                  int wg) {
    // The whole kernel runs on the POD device scalar. std::complex is re-typed
    // HERE, at the pointer boundary, and never crosses into the kernel body: its
    // operator* is Annex-G conformant, which means an isnan branch and a
    // __mulsc3 / __muldc3 CALL in the inner loop (latrd_lower_panel.cc:148-190
    // measures what that costs).
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    D* const tp = reinterpret_cast<D*>(tau_ptr);
    const int kmax = std::min(m, n);
    const std::size_t tile_elems =
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n);

    // The ALLOCATION steps over the 48 KB launch hole; the kernel body still
    // indexes only tile_elems, so the pad is capacity and never data. Same
    // predicate geqrf_cta_fits applies, so the table cannot promise a tile this
    // launch would then be refused.
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

                // Consecutive work-items take consecutive elements of a column,
                // so both the global read and the local write are coalesced.
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

// ---------------------------------------------------------------------------
// The GLOBAL leaf: the same device body, streamed from global memory.
//
// It exists because a BLOCKED panel is (m - j0) x nb and m is unbounded, so the
// resident leaf cannot serve it -- a 1024 x 32 float panel is 128 KB against a
// 97 KB budget. WP5's scaffolding assumed the blocked driver's leaf would always
// be the resident one; it cannot be, and that is recorded in the report rather
// than papered over. The two share the algorithm, not the residency.
//
// NO LOCAL STAGING OF v. Every team re-reads column j from global memory, which
// looks redundant and is not: the column is contiguous, at most a few tens of KB,
// and read by all teams in the same instant -- i.e. it is an L1 broadcast, not
// DRAM traffic. Staging it would need a chunked SLM pipeline (the column is
// itself unbounded), which is a second algorithm.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// CAPABILITY.
//
// THE AREA IS THE REAL CAPACITY. The resident tile is m*n scalars, so what fits
// is governed by the product; a pair of per-extent ceilings would accept a
// 155 x 155 float panel because each extent is "within range" while the tile it
// needs is many times the budget. That is why GeqrfShape carries cta_max_elems
// at all and why supports() spells the test as a product in int64 arithmetic.
//
// THE HEIGHT LIMIT IS HONEST BUT NOT INDEPENDENTLY BINDING TODAY, and saying so
// is better than pretending otherwise. With this layout -- one tile, no per-row
// resident array -- the largest admissible m at n = 1 IS the area bound, so
// cta_max_m is the area bound specialised to a single column. It is kept, and
// kept as a separate number, for three reasons, none of which is decoration:
//
//   1. supports() already tests both, and RouteGeqrf.
//      CtaCapacityIsAnAreaAndAHeightNotTwoExtentBounds pins that it does.
//      Removing one test is a table change with a test to redo, not a
//      simplification.
//   2. It is the number that MOVES if a per-row resident array is ever added
//      (a staged v, a per-row norm cache). At that point the area bound stops
//      describing the height and the pair stops coinciding -- and the call site
//      that has to change is this one function, not supports().
//   3. It bounds the row index independently of the column count, which is what
//      keeps `r + c*ld` in range without an argument about the product.
//
// The alternative -- inventing a tighter height cap so the field looks
// load-bearing -- would be a SPEED threshold in supports(), which
// route_geqrf.hh's header forbids for the reason route_resolve.hh:60-63 gives:
// it would remove the vendor-free route for every panel above it.
// ---------------------------------------------------------------------------
template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes) {
    // No walk and no `break` are needed here, unlike potrf_cta_max_n_for_slm:
    // geqrf_slm_bytes is exactly linear in the element count and strictly
    // monotone, because there is no padding step to make it otherwise. That is
    // the second reason the tile carries no pad.
    return static_cast<int64_t>(geqrf_hole_safe_budget(slm_budget_bytes) / sizeof(T));
}

template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes) {
    const int64_t e = geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes);
    // Clamped into int because GeqrfShape::cta_max_m is an int and supports()
    // widens it to int64 before comparing. On any real device e is O(10^4).
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

// The ONE fit predicate. supports() reaches it through the two capacities above;
// the blocked driver reaches it directly, per panel. Both therefore answer the
// same question with the same arithmetic, which is what potrf_cta.cc:442-454
// records the cost of not doing.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes) {
    if (m < 1 || n < 1) return false;
    const int64_t elems = static_cast<int64_t>(m) * static_cast<int64_t>(n);
    return static_cast<int64_t>(m) <= static_cast<int64_t>(geqrf_cta_max_m_for_slm<T>(slm_budget_bytes)) &&
           elems <= geqrf_cta_max_elems_for_slm<T>(slm_budget_bytes) &&
           geqrf_hole_padded(geqrf_slm_bytes<T>(m, n)) <= slm_budget_bytes;
}

// ---------------------------------------------------------------------------
// WORKSPACE. The CTA tier needs NONE: the tile is local memory and tau is the
// caller's span.
//
// Zero is still produced by a BumpAllocator::measuring() replay of the (empty)
// layout rather than written as a literal, because the facade's
// max(native, vendor) is safe only when every term is an allocation_size /
// required_bytes figure (mempool.hh:52-58, and
// docs/perf/potrf.md's note on it). An empty sequence's required
// figure happens to be 0; a later term added here must arrive through the same
// replay.
//
// Trivially MONOTONE NON-DECREASING in (rows, cols, batch), which is the
// geqrf-only contract band_reduction.cc:1041-1044 imposes -- it sizes at
// (m_max x nb_max) and calls at :595 with a smaller sub-view -- and it
// dereferences neither A.data_ptr() nor tau.data(), both nullptr there.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A) {
    static_cast<void>(A);
    return workspace_bytes([&](BumpAllocator& p) {
        static_cast<void>(ctx);
        return &p;
    });
}

// ---------------------------------------------------------------------------
// THE PANEL LEAF, and the ONE place the residency is chosen.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE CTA TIER'S DIRECT ENTRY POINT.
//
// Every gate RouteTable<Op::geqrf,T>::supports() applies to the CTA arm is
// re-applied here, because this function is reachable WITHOUT the table -- and
// it must be. route_resolve.hh:101 tests `if (Table::supports(forced, s)) return
// forced;` and falls through to automatic() at :111, so a test that sets
// BATCHLAS_GEQRF_ROUTE=cta and gets one gate wrong runs cuSOLVER and passes
// GREEN over a kernel nothing executed (tests/potrf_tests.cc:6-25). A direct call
// cannot be served by a vendor.
// ---------------------------------------------------------------------------
template <typename T>
Event geqrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace) {
    static_cast<void>(workspace);   // this tier needs none; see the query above

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    const int batch = static_cast<int>(A.batch_size());

    if (m < 1 || n < 1 || batch < 1) {
        throw std::invalid_argument("geqrf_cta: degenerate extents");
    }
    if (m < n) {
        // CORRECTNESS, not fit. The panel body walks min(m, n) reflectors down
        // columns; handed a WIDE view the schedule is still well defined, but
        // route_geqrf.hh's supports() refuses m < n for BOTH native arms and the
        // two must agree, or a forced route reaches a shape the table promised
        // the vendor. Widening this is a table change with a test, not a local
        // one.
        throw std::invalid_argument(
            "geqrf_cta: m < n is not supported (route_geqrf.hh's supports() refuses it)");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with a single (m, n, ld, stride) tuple and
        // reads at data_ptr() + b*stride with the CAPACITY extents, so a view
        // with per-item active dims would factorise the wrong extents in place
        // for every item after the first. Unlike potrf there is no path in this
        // tree that gets heterogeneous-batch QR right -- netlib_lapack.cc:1406
        // hoists m and n outside its loop too -- so this is "the kernel cannot
        // serve it", full stop.
        throw std::invalid_argument("geqrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("geqrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that property
        // returns sub_group_sizes()[0] (src/util/queue-impl.cc:325), the FIRST
        // supported size, so the weak test refuses a {8,16,32} device and ACCEPTS
        // a {64} one -- and the kernel below carries
        // [[sycl::reqd_sub_group_size(32)]], for which the second is a launch
        // abort.
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
        // The tier is a CAPACITY, so this is the honest failure. It names the
        // ceiling rather than silently factoring a leading submatrix -- the
        // discipline WP3 had to learn the hard way (trsm_native.cc:96-115: a
        // 33-order solve silently solved the leading 32x32 system).
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
        // the same arithmetic. Asserted rather than assumed, because "the tier
        // silently became the other tier" is precisely the failure a
        // pinned-route test cannot see.
        throw std::logic_error(
            "geqrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- geqrf_cta_fits and geqrf_panel_factorize disagree");
    }
    return e;
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. Same shape and
// same reason as potrf_cta.cc:706-726 -- the kernel has no vendor dependency and
// no Backend parameter, so a 3x multiplication of a device-compiled family in a
// build that is device-link-bound is pure cost. Everything that needs a Backend
// arrives injected (GeqrfTrailingGemm).
// ---------------------------------------------------------------------------
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
