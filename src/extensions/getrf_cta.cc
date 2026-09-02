// Native batched GETRF -- the CTA tier, and the panel leaf both tiers share.
//
// The whole n x n matrix is staged into local memory once, factorised there by
// LAPACK ?GETF2's right-looking rank-1 recurrence with partial pivoting, and
// stored once. All of the device code is in getrf_cta_device.hh, because
// getrf_blocked.cc's panel step calls the SAME body against a global-memory
// accessor: `at(r, c)` is the only difference between the two residencies, so a
// correctness fix cannot land in one and miss the other. That sharing is why
// both TUs sit in ONE device-code cluster (src/extensions/CMakeLists.txt:29-42).
//
// WHAT IS IN THIS FILE AND NOT IN THE HEADER:
//   * the CAPABILITY answer, which is derived from the SAME expression the
//     launcher uses to size its local_accessor -- so the ceiling supports()
//     advertises and the allocation the kernel makes cannot disagree
//     (potrf_cta.cc:442-454 records what disagreement costs: an unhandled
//     std::invalid_argument on a call the route table had promised);
//   * getrf_cta_fits / getrf_leaf_fits, the single predicate the table's
//     capacity, the launcher and the blocked driver's per-panel residency choice
//     all go through;
//   * the two launchers and the ONE decision site between them
//     (getrf_panel_factorize).
//
// PERFORMANCE STATUS: ROUTE-NEUTRAL, DELIBERATELY.
// RouteTable<Op::getrf,T>::preferred() is false for both arms, so a
// vendor-present build keeps taking cuBLAS for every shape and this kernel is
// reachable only through BATCHLAS_GETRF_ROUTE, through getrf_cta_dispatch, or in
// a vendor-free build (route_resolve.hh:60-63). Flipping preferred() is a later
// step gated on a measured grid, and the vendor here is genuinely batched
// (cublas{S,D,C,Z}getrfBatched, cublas.cc:1509) -- there is no WP5-style
// per-item vendor loop to beat, so parity is the realistic good outcome.
//
// THE REGISTER PROBE, pointed at the right target (its default, batchlas_sycl,
// does not contain src/extensions/ at all):
//     scripts/register_probe.sh out.log '' batchlas_extensions_cta
// Gate on the ENTRY-FUNCTION spill line only; the all-functions count on this
// target reads 16 from a pre-existing 255-register gesvdj_cta_impl<cdouble>.

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

// The standard per-work-group local-memory budget: the RUNTIME local_mem_size
// minus the 4096 B reserve cmake/BatchLASDetectSYCL.cmake:57-67 applies to every
// other device-BLAS sizing decision in this library. 97,280 on this box.
//
// NOT build/include/batchlas/device_limits.hh's 49152: that number is HARDCODED
// by cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern, the
// detection routine never queries local_mem_size at all, and it is 2.06x wrong
// here (WP4 finding W1). This constant only ever answers the "at this
// repository's reference budget" convenience overload; every real decision reads
// the device.
constexpr std::size_t kGetrfReferenceSlmBudget = 97280;

// The resident tile's LEADING DIMENSION. ODD, unlike geqrf's packed ld = m.
//
// LU has an access pattern geqrf does not: the ROW EXCHANGE walks a row, i.e.
// `wg` work-items at stride ld, and an even ld puts every one of them in the
// same local-memory bank. potrf_cta.cc:555 makes the same choice for the same
// reason. It costs at most one extra column of tile.
constexpr int getrf_tile_ld(int m) { return m | 1; }

// The local-memory footprint of a resident m x n panel, in bytes: the tile plus
// the pivot search's per-sub-group slots.
//
// THE SCRATCH TERM IS CONSTANT IN THE WORK-GROUP WIDTH, and that is the whole
// reason the search is a sub-group butterfly rather than a per-work-item SLM
// tree (getrf_cta_device.hh's opening note). A per-work-item tree needs
// wg*(sizeof(real)+sizeof(int)) -- 2040 B at wg 256 for float, 3060 B for
// cdouble -- which (a) took a measured cdouble n=78 launch from 98,608 B to
// 101,668 B, past this device's 101,376 B hard cap, and (b) would make this
// function depend on a work-group width that the capacity query, the fit
// predicate and the launcher each choose separately. 32 slots is 384 B for
// cdouble and 256 B for float, and it is the same number everywhere.
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

// ---------------------------------------------------------------------------
// THE 48 KB LAUNCH HOLE, AND THE PAD THAT STEPS OVER IT.
//
// The band and the pad target are potrf_cta.cc:290-296's and geqrf_cta.cc:
// 115-122's, byte for byte and deliberately: same box, same cause, and a third
// set of numbers would be three things to keep true instead of one.
//
// WP6 RE-MEASURED THE HOLE FROM SCRATCH with a PAD= knob that holds kernel,
// shape and work-group fixed and moves ONLY the declared byte count, one process
// per point (docs/perf/lu.md#the-48-kb-launch-hole):
//     49,024 B PASS   49,152 B FAIL   49,280 B PASS
// 5/5 deterministic across five separate processes and independent of work-group
// width (32/64/128/256/512 all fail). For double and cdouble the collective also
// fails at 48,896 B. BOTH failure points lie inside (47104, 49664], so the
// inherited band already covers the wide-scalar case -- checked rather than
// assumed, because getrf_native.hh's note warns that it might not.
//
// THIS KERNEL SHOULD NOT BE IN THE HOLE AT ALL: WP6's measurement attributes it
// to sycl::reduce_over_group ALONE (an explicit tree at the IDENTICAL byte count
// launches fine, as does the unpivoted arm), and this file uses no group
// collective -- only permute_group_by_xor, which is a register shuffle. The pad
// is applied anyway because the hole is a property of the STATIC shared memory
// the compiler emits, which this source does not control: WP4 recorded exactly
// that ("one group algorithm added anywhere in the body ... reintroduces the
// hole") and WP5 walked into it regardless. Padding costs at most 256 B and only
// inside the band; on this box the budget is 97,280 B, far above it, so nothing
// here moves.
// ---------------------------------------------------------------------------
constexpr std::size_t kGetrfHoleLo = 47104;
constexpr std::size_t kGetrfHoleHi = 49664;
constexpr std::size_t kGetrfHolePadTo = 49920;

constexpr std::size_t getrf_hole_padded(std::size_t bytes) {
    return (bytes > kGetrfHoleLo && bytes <= kGetrfHoleHi) ? kGetrfHolePadTo : bytes;
}

// THE WORK-GROUP WIDTH. Not inherited from potrf or geqrf, and measured to
// matter more here than in either: float n=128 batch 4096, the unpivoted
// reference arm is 39.72 ms at wg=32 and 4.77 ms at wg=512 -- an 8.3x spread
// (docs/perf/lu.md#open-debts). Best measured wg is 256 at n=64 and 512
// at n=128.
//
// The rule below reproduces both of those and extends them to the blocked
// tier's panel shape, where the rows greatly outnumber the columns: the target
// is "about four columns' worth of rows", clamped to [64, 512].
//
// IT IS NOT A TUNED TABLE AND IS NOT CLAIMED TO BE. preferred() is false, so
// nothing routes here yet; this is the shape a measured grid would tune, kept as
// one pure function so that tuning it later is a change to one place. Note that
// it is a PURE PERFORMANCE KNOB: getrf_slm_bytes does not depend on it, so
// changing it cannot move a capacity or reopen a fit disagreement.
//
// THE 512 CAP IS NOT A SWEEP ARTEFACT, AND THAT WAS CHECKED. A review observed
// that the original wg sweep stopped at 512 with the curve still improving, that
// this device reports MAX_WORK_GROUP_SIZE = 1024, and that max_wg is only ever
// used as a DOWNWARD clamp -- all true -- and concluded the cap should be 1024.
// It was raised to 1024 and re-swept over all 156 saturation cells: geomean
// 0.974x, getrf cdouble 0.939x and float 0.960x, worst cells 0.814-0.837x (float
// n=256 batch 256: 2.046 -> 2.514 ms; cdouble n=1024 batch 16: 64.4 -> 77.9 ms),
// 0 cells discarded, 0 route changes.
//
// The prediction was RIGHT about the one shape it was aimed at and wrong about
// the knob: the only cells that improve are the blocked driver's global panel
// leaf at n=2048 (1.006-1.012x, i.e. under 1.2%), and this SAME function serves
// the resident tier, where 1024 work-items on an order-256 tile lose 16-19%.
// Splitting the cap by residency would buy at most that 1.2%; it was not taken.
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

// ---------------------------------------------------------------------------
// The RESIDENT leaf: stage, factor in local memory, store.
// ---------------------------------------------------------------------------
template <typename T>
Event getrf_panel_resident_launch(Queue& ctx,
                                  T* a_ptr, int ld, int stride,
                                  int m, int n, int batch,
                                  int* piv_ptr, int piv_stride, int piv_base,
                                  int32_t* info_ptr,
                                  int wg) {
    // The whole kernel runs on the POD device scalar. std::complex is re-typed
    // HERE, at the pointer boundary, and never crosses into the kernel body: its
    // operator* is Annex-G conformant, which means an isnan branch and a
    // __mulsc3 / __muldc3 CALL in the rank-1 update.
    using DM = sycl_device::DevMap<T>;
    using D = typename DM::type;
    using R = typename DM::real;
    static_assert(sizeof(D) == sizeof(T), "device scalar must be layout-compatible");

    D* const ap = reinterpret_cast<D*>(a_ptr);
    const int kmax = std::min(m, n);
    const int ldt = getrf_tile_ld(m);
    const std::size_t tile_elems =
        static_cast<std::size_t>(ldt) * static_cast<std::size_t>(n);

    // The ALLOCATION steps over the 48 KB launch hole; the kernel body still
    // indexes only tile_elems, so the pad is capacity and never data. It is the
    // SAME arithmetic getrf_leaf_fits applies, so the table cannot promise a
    // tile this launch would then be refused.
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

                // Consecutive work-items take consecutive elements of a column,
                // so both the global read and the local write are coalesced.
                // Indexed off the LOGICAL m x n extent and scattered into the
                // padded tile ld, which is why `used` is not `tile_elems`.
                //
                // THE 64-BIT DIVISIONS STAY. The same power-of-two (row, column)
                // split that was measured and rejected for the rank-1 update
                // (getrf_cta_device.hh) was measured here in the same build and is
                // part of the same 0.936x float regression: at the resident tier's
                // shape m is close to wg, so the split degenerates to a trip-count-
                // one inner loop and the loop overhead exceeds the division it
                // removes. Do not re-apply it without measuring the RESIDENT rung.
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

// ---------------------------------------------------------------------------
// The GLOBAL leaf: the same device body, streamed from global memory.
//
// It exists because a BLOCKED panel is (m - j0) x nb and m is unbounded, so the
// resident leaf cannot serve it -- a 2048 x 32 float panel is 256 KB against a
// 97 KB budget. The two share the algorithm, not the residency; geqrf_cta.cc
// carries the same pair for the same reason.
//
// The reduction slots are still LOCAL memory here: they are 256-384 B and the
// argmax has to land somewhere the whole work-group can read.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE FIT PREDICATE -- the ONE function the capacity, the launcher and the
// blocked driver's per-panel residency choice all go through, which is what
// potrf_cta.cc:442-454 records the cost of not doing.
// ---------------------------------------------------------------------------
template <typename T>
bool getrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes) {
    if (m < 1 || n < 1) return false;
    // int64 arithmetic before the byte count is formed: (m|1)*n overflows int at
    // m ~ 46341, which is far below anything that fits but is reachable through
    // the blocked driver's panel height on a large problem.
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

// ---------------------------------------------------------------------------
// CAPABILITY. ONE number, not geqrf's (height, area) pair, because getrf's
// operand is square: the order is the only extent.
//
// A WALK WITH A LOAD-BEARING `break`, not a closed form, and potrf_cta.cc:
// 458-472 is the precedent: getrf_hole_padded is NON-MONOTONE once a pad exists
// -- a raw figure just inside the band is padded ABOVE a larger raw figure just
// outside it (47,200 -> 49,920 while 49,700 stays 49,700) -- so the largest n
// that fits is not the largest n below the first miss unless the walk stops
// there. Stopping at the first miss is also what makes the table's
// `order <= cta_max_n` gate EQUIVALENT to getrf_leaf_fits for every order it
// admits, which is the property the dispatch's own re-check depends on.
//
// 0 IS the agreed spelling of "this tier is not in this build"
// (TrsmShape::cta_max_n's convention, pinned by
// RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable). It is now a real
// number on any GPU with local memory; it stays 0 for a budget too small to hold
// a 1 x 1 tile plus the 32 argmax slots.
// ---------------------------------------------------------------------------
template <typename T>
int getrf_cta_max_n_for_slm(std::size_t slm_budget_bytes) {
    using DM = sycl_device::DevMap<T>;
    const std::size_t scratch = getrf_scratch_bytes<T>();
    if (slm_budget_bytes <= scratch) return 0;
    // A generous upper probe: (n|1)*n <= (budget - scratch)/sizeof(D) bounds n
    // by the square root of that, and +2 covers the odd-ld rounding.
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

// ---------------------------------------------------------------------------
// WORKSPACE. ONE term: the fallback `info` span.
//
// The tile is local memory and the pivots are the caller's span, so the only
// thing this tier can need from the pool is the scratch that stands in for an
// `info` argument the caller did not supply -- and src/extensions/inv.cc:48 is
// exactly such a caller (it passes no info at all, which is why a singular item
// silently yields infinities there today). detail::info_target's rule
// (linalg-impl.hh:767-771, and potrf_cta.cc:688-695 inlined): an empty OR SHORT
// caller span means "not requested" and falls back to pool scratch. The
// direction matters -- supplying a span only ever REMOVES a pool draw -- which
// is what keeps this figure correct in both modes.
//
// Produced by a BumpAllocator::measuring() replay of the layout rather than
// hand-summed, because the facade's max(native, vendor) is safe only when every
// term is an allocation_size / required_bytes figure (mempool.hh:45-58): :96-104
// checks capacity as the alignment-rounded alloc_size measured from the
// UNALIGNED cursor while :111-113 advances the cursor by only size*sizeof(T).
//
// It dereferences nothing: A arrives with a null data_ptr() from inv_layout's
// measuring pass (inv.cc:36), and only batch_size() is read.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE PANEL LEAF, and the ONE place the residency is chosen. Called by this
// tier's dispatch with (n, n) and by the blocked driver with (m - j0, ib).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// THE CTA TIER'S DIRECT ENTRY POINT.
//
// Every gate RouteTable<Op::getrf,T>::supports() applies to the CTA arm is
// re-applied here, because this function is reachable WITHOUT the table -- and
// it must be. route_resolve.hh:165 tests `if (Table::supports(forced, s)) return
// forced;` and falls through to automatic() at :175, so a test that sets
// BATCHLAS_GETRF_ROUTE=cta and gets one gate wrong runs cuBLAS and passes GREEN
// over a kernel nothing executed. A direct call cannot be served by a vendor.
// ---------------------------------------------------------------------------
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
        // CORRECTNESS AND CONTRACT, not fit. route_getrf.hh's supports() refuses
        // m != n for both native arms and the two must agree, or a forced route
        // reaches a shape the table promised the vendor. The gate exists because
        // BatchLAS's public getrf IS square -- the pivot span is sized rows*batch
        // and cublas?getrfBatched takes one `n` -- not because LU needs it.
        throw std::invalid_argument(
            "getrf_cta: A must be square (route_getrf.hh's supports() refuses m != n)");
    }
    if (A.is_heterogeneous()) {
        // One launch covers the batch with a single (n, ld, stride) tuple and
        // reads at data_ptr() + b*stride with the CAPACITY extents, so a view
        // with per-item active dims would factorise the wrong extents in place
        // for every item after the first. netlib_lapack.cc:1291 hoists n outside
        // its loop too.
        throw std::invalid_argument("getrf_cta: heterogeneous batch is not supported");
    }
    const auto dev = ctx.device();
    if (dev.type != DeviceType::GPU) {
        throw std::invalid_argument("getrf_cta: GPU queues only");
    }
    if (!dev.supports_sub_group_size(32)) {
        // ENUMERATED, never get_property(MAX_SUB_GROUP_SIZE) >= 32: that property
        // returns sub_group_sizes()[0] (src/util/queue-impl.cc:325), the FIRST
        // supported size, so the weak test refuses a {8,16,32} device and ACCEPTS
        // a {64} one -- and the kernels below carry
        // [[sycl::reqd_sub_group_size(32)]], for which the second is a launch
        // abort.
        throw std::runtime_error(
            "getrf_cta: device does not offer sub-group size 32, which the kernel requires");
    }

    // THE PIVOT SPAN IS int64 ON THE WIRE AND PACKED int32 IN THE BUFFER. There
    // is no conversion anywhere on the GPU backends -- cublas.cc:1509 and
    // rocsolver.cc:227 both do pivots.as_span<int>(), which sycl-span.hh:45-47
    // implements as a reinterpret_cast with the size rescaled -- so the native
    // kernel must write the SAME packed 1-based int32 or a native getrf feeding a
    // vendor getri/getrs returns silent garbage. See getrf_native.hh's PIVOT
    // CONTRACT for the measurement. The one configuration the format cannot cover
    // -- Backend::NETLIB on a GPU queue, whose span is genuine int64 -- is refused
    // by supports() in all three route tables; it is NOT re-checked here, because
    // this entry point takes no Backend and could not see it.
    if (pivots.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("getrf_cta: pivot span is shorter than n * batch");
    }

    const std::size_t local_mem = dev.get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    if (!getrf_cta_fits<T>(n, budget)) {
        // The tier is a CAPACITY, so this is the honest failure. It names the
        // ceiling rather than silently factoring a leading submatrix -- the
        // discipline WP3 had to learn the hard way (trsm_native.cc:96-115: a
        // 33-order solve silently solved the leading 32x32 system).
        throw std::invalid_argument(
            "getrf_cta: order " + std::to_string(n) +
            " does not fit this device's local memory (needs " +
            std::to_string(getrf_slm_bytes<T>(n, n)) + " B of " +
            std::to_string(budget) + " B); the ceiling for this type is " +
            std::to_string(getrf_cta_max_n_for_slm<T>(budget)));
    }

    BumpAllocator pool(workspace);
    // detail::info_target's rule, inlined so this TU does not include
    // src/linalg-impl.hh: an empty or SHORT caller span means "not requested"
    // and falls back to pool scratch. THE SPAN THAT IS ZEROED IS THIS ONE, never
    // info_out -- zeroing the caller's would leave the span the kernel actually
    // reads full of whatever the pool last held
    // (docs/perf/potrf.md#what-the-spec-got-wrong:938-943).
    Span<int32_t> info = (info_out.size() >= static_cast<std::size_t>(batch))
                             ? info_out
                             : getrf_cta_layout(ctx, pool, batch);

    // THE ZERO PRE-PASS IS NOT OPTIONAL. getf2_panel_device READS info to keep
    // first-failure-wins across the blocked driver's panels
    // (getrf_cta_device.hh:262 loads *info_item into info_local), and a caller's
    // span arrives with garbage (options_api_tests.cc:498,509 seeds -12345), so
    // without this a non-singular item can report a stale failure.
    //
    // IT IS A READ-AFTER-WRITE DEPENDENCE, NOT A PURE OUTPUT, so it needs the
    // same guard every other dependent boundary in this family carries. On the
    // DEFAULT in-order queue (sycl-device-queue.hh:254) the ordering is free and
    // this costs nothing; an out-of-order queue is public API
    // (sycl-device-queue.hh:258, `Queue(const Queue& base, bool in_order)`) and
    // without the guard the panel reads the caller's pre-call garbage, decides an
    // earlier panel already failed, and writes that garbage straight back --
    // measured at 6,979 wrong items of 1,638,400 on an out-of-order queue, none
    // of them reporting the real singular column. A full `.wait()` unconditionally
    // is what potrf_blocked.cc:617-628 records the cost of; this drains only when
    // the queue cannot order it for us.
    ctx->fill(info.data(), int32_t(0), static_cast<std::size_t>(batch));
    if (!ctx.in_order()) ctx.wait();

    auto piv_i32 = pivots.as_span<int>();

    bool resident = false;
    Event e = getrf_panel_factorize<T>(ctx, A.data_ptr(), A.ld(), A.stride(), n, n, batch,
                                       piv_i32.data(), n, 0, info.data(), &resident);
    if (!resident) {
        // Unreachable: getrf_cta_fits was just checked with the same budget and
        // the same arithmetic. Asserted rather than assumed, because "the tier
        // silently became the other tier" is precisely the failure a
        // pinned-route test cannot see.
        throw std::logic_error(
            "getrf_cta: the panel leaf did not take the resident path after the fit "
            "check passed -- getrf_cta_fits and getrf_panel_factorize disagree");
    }
    return e;
}

// ---------------------------------------------------------------------------
// Instantiation: PER SCALAR TYPE ONLY, no Backend cross-product. Same shape and
// same reason as potrf_cta.cc:706-726 and geqrf_cta.cc:508-522 -- the kernel has
// no vendor dependency and no Backend parameter, so a 3x multiplication of a
// device-compiled family in a build that is device-link-bound is pure cost.
// Everything that needs a Backend arrives injected.
// ---------------------------------------------------------------------------
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
