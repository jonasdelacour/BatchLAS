#pragma once

// Native batched GEQRF -- declarations.
//
// WP5. THE KERNELS ARE IMPLEMENTED. Every capability below now answers from the
// device's real local-memory budget and both dispatch entry points run. The
// scaffolding this file started as did its job: landing the kernels was filling
// in bodies, not inventing an interface.
//
// IT IS NONETHELESS ROUTE-NEUTRAL. RouteTable<Op::geqrf,T>::preferred() is still
// false for both arms, so a vendor-present build keeps taking cuSOLVER for every
// shape and these kernels are reached only by a vendor-free build
// (route_resolve.hh:60-63), by BATCHLAS_GEQRF_ROUTE, or by the direct entry
// points below. Flipping preferred() is a later step gated on a measured grid.
//
// TWO NATIVE TIERS, ONE LEAF (the potrf shape, potrf_native.hh:9-21):
//
//   Algorithm::CTA      -- one m x n panel resident in local memory for the whole
//                          factorisation, a work-group per matrix. Serves panels
//                          that fit geqrf_cta_max_m_for_slm<T>() rows AND
//                          geqrf_cta_max_elems_for_slm<T>() scalars.
//   Algorithm::Blocked  -- the host-blocked right-looking driver for larger
//                          panels. Its panel leaf IS the CTA DEVICE FUNCTION,
//                          handed a raw sub-panel. That sharing is why both TUs
//                          must sit in ONE device-code cluster -- see
//                          src/extensions/CMakeLists.txt:15-27 and its :70-85
//                          grouping rule (W12).
//
// ONE CORRECTION TO THE SCAFFOLDING, MADE WHEN THE KERNELS LANDED. The sentence
// that used to sit here -- "so the crossover between the two tiers is a capacity
// and not a tuned guess" -- is true of the TIERS and was false of the LEAF. A
// blocked panel is (m - j0) x nb with m unbounded: a 1024 x 32 float panel is
// 128 KB against this box's 97 KB budget, so the leaf cannot always be RESIDENT.
// What it always is, is the same device function. It carries two accessors
// (local_accessor and raw global pointer), chooses between them per panel through
// the SAME geqrf_cta_fits predicate the route table's capacity uses, and reports
// which it took in geqrf_blocked_debug_params' high half. The crossover between
// the TIERS is still purely a capacity; the residency inside the blocked tier is
// too, just at a different shape.
//
// WHY THIS FILE IS SEPARATE FROM A FUTURE geqrf_cta_device.hh: the route table
// (route_geqrf.hh, through src/backends/geqrf_route.hh) must be able to ASK the
// capability questions without acquiring a dependency on device code, and the
// vendor-free facade must be able to CALL the launcher without including
// <sycl/sycl.hpp>. Both include only this. Same split, and same reason, as
// src/extensions/potrf_native.hh:28-35 and src/sycl/trsm_native.hh:38-125.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_geqrf {

// ---------------------------------------------------------------------------
// CAPABILITY. Two numbers, because the CTA tile's limit is an AREA and the
// Householder reduction's limit is a HEIGHT, and a single per-extent ceiling
// cannot express either honestly (see GeqrfShape::cta_max_elems).
//
// Both are functions of the per-work-group local-memory budget in BYTES rather
// than constants, for route_trsm.hh:62-72's reason and for WP4's finding W1: the
// 49152 in build/include/batchlas/device_limits.hh is HARDCODED by
// cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern -- the
// detection routine never queries local_mem_size at all -- and is 2.06x wrong on
// this box, which reports sycl::info::device::local_mem_size == 101,376 B. A
// build-time constant would make supports() claim an unlaunchable route on a
// smaller device AND leave a whole band of extents with NO ROUTE in a
// vendor-free build (route_resolve.hh:60-63).
//
// When the kernel lands, THE SAME two functions must size the launch's
// local_accessor, so that the ceiling the table advertises and the allocation
// the kernel makes cannot disagree.
//
// 0 REMAINS the agreed spelling of "this tier is not in this build"
// (TrsmShape::cta_max_n's convention, pinned by
// RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable). Neither returns 0 on
// a real GPU any more; both do on a device with no usable local memory, which is
// the honest answer there.
//
// ONE THING THE SCAFFOLDING PROMISED THAT THE IMPLEMENTATION HAD TO QUALIFY: the
// height limit is NOT independently binding with the shipped layout. The tile is
// exactly m*n scalars with no per-row resident array, so the largest admissible m
// at n = 1 IS the area bound. It is kept as a separate number because supports()
// tests both (and a test pins that it does), because it is the number that moves
// the moment a per-row array is added, and because inventing a tighter one would
// be a SPEED threshold in supports() -- which route_geqrf.hh forbids, since it
// would delete the vendor-free route above it. src/extensions/geqrf_cta.cc says
// so at the definitions.
// ---------------------------------------------------------------------------
template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes);

// The same questions at this repository's standard budget (the runtime
// local_mem_size minus the 4096 B reserve every other device-BLAS sizing
// decision in this library applies, BatchLASDetectSYCL.cmake:57-67). Convenience
// over the honest spellings above, kept so tests can pin the four per-type
// numbers once they exist.
template <typename T>
int geqrf_cta_max_m();

template <typename T>
int64_t geqrf_cta_max_elems();

// Whether the BLOCKED driver exists in this build. TRUE for all four scalar
// types as of WP5.
//
// IT IS DEFINED IN src/extensions/geqrf_blocked.cc, NOT IN THIS HEADER AND NOT
// IN geqrf_cta.cc, and that placement is load-bearing (potrf_native.hh:81-92):
// these are full explicit specialisations, so they link from wherever they sit
// -- and sitting anywhere but beside the driver would let a build advertise the
// tier while geqrf_blocked.cc is absent from EXTENSIONS_CTA_SOURCES or #if 0'd
// out. That is the state route_trsm.hh:99-110 names ("the table must describe the
// BUILD, not the design"). Co-located, "the flag is true" and "the file is
// compiled" are the same fact.
template <typename T>
bool geqrf_blocked_available();

// ---------------------------------------------------------------------------
// WORKSPACE
//
// The CTA tier needs NO workspace (its tile is local memory and tau is the
// caller's span), so its query returns 0 -- produced by an empty measuring replay
// rather than written as a literal, so that a later term arrives through the same
// path. Both figures MUST be obtained by replaying the layout through
// BumpAllocator::measuring() (mempool.hh:185-190's workspace_bytes), NEVER by
// hand-summing an expression. The reason, from potrf_native.hh:105-113:
// mempool.hh:82-86 checks capacity from the UNALIGNED cursor while :118-120
// advances only by the data extent, so an "exactly computed" figure fails the
// allocator's own capacity check; and required_bytes() (mempool.hh:52-58)
// additionally rounds to the coarsest quantum the sequence asked for, because
// callers re-serve the number through allocate<std::byte>()
// (src/extensions/ortho.cc:78 is such a caller).
//
// It is also what makes the facade's max(native, vendor) safe: every term must
// be such a rounded figure. WP4_POTRF_SPEC_CORRECTIONS.md states it -- "max(a,b)
// is safe only because both terms come from required_bytes()/allocation_size; do
// not 'optimise' the layout functions into a hand-summed arithmetic expression".
//
// TWO CONTRACTS THAT ARE geqrf-SPECIFIC AND HAVE NO potrf ANALOGUE. Both come
// from src/extensions/band_reduction.cc, and both are load-bearing:
//
//   (a) NEITHER QUERY MAY DEREFERENCE A.data_ptr() OR tau.data().
//       band_reduction.cc:1041-1044 (and the duplicate at :1185-1187) sizes
//       sytrd's band reduction with
//           MatrixView<T,Dense> dummyB(nullptr, m_max, nb_max, ...);
//           Span<T> dummyTau(nullptr, ...);
//           bytes += geqrf_buffer_size<B, T>(ctx, dummyB, dummyTau);
//       Any read of either pointer here is an immediate segfault in sytrd's
//       sizing path. Extents, ld, stride, batch and is_heterogeneous() are
//       metadata and are safe.
//
//   (b) THE QUERY AND THE CALL ARE MADE AGAINST DIFFERENT SHAPES, so the sizes
//       must be MONOTONE NON-DECREASING IN (rows, cols, batch). The same
//       band_reduction sizes at (m_max x nb_max) but calls geqrf at
//       band_reduction.cc:595 with `Bsub`, an m x r sub-view. max() over routes
//       at ONE shape does nothing about that. A native query that is not
//       monotone silently under-allocates sytrd, and it needs its own test over
//       a grid -- not an argument in a comment.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
std::size_t geqrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// The BLOCK WIDTHS the blocked driver WOULD use for this panel, for tests.
//
// Returns the panel width nb in the low 16 bits and, in the high 16 bits, WHICH
// PANEL LEAF the LEADING panel takes: 1 = the local-memory-resident leaf (the
// same device function the CTA tier runs), 2 = the global-memory leaf. 0 for the
// whole word means the driver is absent.
//
// THE HIGH HALF IS NOT THE "trailing-update width" THIS COMMENT ORIGINALLY
// PROMISED, and the change is deliberate. The driver issues ONE gemm per panel
// over the whole trailing block, so there is no second width to report -- while
// there IS a second code path, chosen per panel by capacity, and a test that
// cannot see which leaf it exercised is the blind guard this query exists to
// prevent. Reporting a constant would have been decorative.
//
// The leaf choice is per panel, and it can CHANGE WITHIN ONE CALL: panels get
// shorter as j0 advances, so a tall matrix can start on the global leaf and end
// on the resident one. This query reports the LEADING panel (j0 = 0), which is
// the tallest and therefore the least likely to be resident; a test that wants
// to exercise the resident leaf inside the blocked driver should either pick a
// small m or drive the last panels.
//
// THIS IS NOT OPTIONAL SCAFFOLDING, AND FOR WP5 IT IS THE MOST IMPORTANT ITEM IN
// THIS FILE. potrf_native.hh:246-266 records the shape of the failure: a test
// that must straddle a block boundary cannot see where the boundary is, and a
// test that hardcodes the width keeps passing after any of its inputs moves
// while silently no longer testing a short final panel. geqrf inherits that
// hazard from the sy2sb stage-1 SHORT FINAL PANEL bug, which produced wrong
// numbers with a green suite.
//
// WP5's baseline measured something worse, and it belongs next to this
// declaration because it invalidates the obvious regression test: THE
// SHORT-FINAL-PANEL BREAK IS VACUOUS ON A SQUARE REAL MATRIX. Deleting the LAST
// reflector left the residual BIT-IDENTICAL for float and double (4.072e-07 /
// 1.615e-15, unchanged) while turning complex red (2.137e-02), because LAPACK's
// larfg returns tau = 0 for a 1x1 real trailing reflector but must still rotate
// R's diagonal onto the real axis for a complex one (|tau[k-1]| measured as
// 0.000000e+00 real, 1.553e+00 complex). A short-final-panel test written on a
// square real matrix guards NOTHING. Use m > n, a MIDDLE panel, or complex.
//
// It is a QUERY over the same pure function the driver and the buffer-size query
// both call, so it cannot report a blocking the call does not use -- the
// potrf_cta_launch_params discipline (potrf_cta.cc:442-454), where a raw figure
// in the query against a padded one in the launcher produced an unhandled throw
// on a call the table had promised.
//
// MEASURED STARTING POINT FOR nb, recorded so it is not re-derived and NOT
// hardcoded anywhere (experiments/wp5_qr/baseline/README.md, summary_nb.txt):
// nb = 32 for float/cfloat/cdouble and nb = 16 for double. Do NOT inherit
// tuning::ormqr_block_size_for_n -- its 16/16/24/48/56 ladder was tuned on
// CUDA/float only (evaluation/tuning/tune.py:494 takes a single --type per run
// and the ormqr_blocked space has no type axis), and even in a VENDOR build the
// shipped width costs double 1.32-1.41x and cdouble 1.46-1.47x. Three measured
// constraints: keep it a MULTIPLE OF 16 (the transposed panel gemm's m IS the
// block width and 24/56 lose everywhere); never below 32 for complex
// (gemm_kernels.cc:700 gates the complex wide-scalar kernel on min_dim >= 32, and
// min_dim of the NN trailing update IS the block width -- at nb=24 complex falls
// to Tiled16 and costs 1.72-2.30x); and NOT WIDER than 32 despite what the
// trailing GEMMs alone say (nb=128 unlocks float's TN register kernel and is
// still the WORST width end to end, 83.0 ms against 36.8 at nb=32, because the
// panel cost a per-gemm probe cannot see dominates).
template <typename T>
unsigned geqrf_blocked_debug_params(Queue& ctx, int m, int n);

// ---------------------------------------------------------------------------
// THE TRAILING-UPDATE GEMM, INJECTED rather than hardcoded.
//
// WHY, and this is a RECORDED DEFECT rather than a style preference: WP3's V2
// called sycl_gemm::gemm_custom directly, which is the NATIVE kernel entry point
// and bypasses RouteTable<Op::gemm> entirely, so the trailing updates always got
// the native kernel whether or not it was the better one (WP3 step 16;
// trsm_native.hh:82-104 carries the measurement, and level3.cc:186-231 the fix).
//
// The signature is deliberately identical to both sycl_gemm::gemm_custom and the
// routed batchlas::gemm, so neither side adapts. An EMPTY function means "use
// gemm_custom", which is what keeps this kernel layer free of the dispatch layer:
// a direct caller (the tests, and any benchmark that must not be silently served
// by a vendor) gets the native kernel with no dispatch dependency, and a
// vendor-free build is unaffected because the resolver falls back to native there
// anyway (route_resolve.hh:60-63).
//
// TWO MEASURED FACTS ABOUT WHAT THIS SEAM WILL CARRY, from WP5's baseline:
//
//   * THE COMPLEX DEFICIT IS REAL AND IS NOT WP5's TO FIX. A vendor-free blocked
//     geqrf will pay 2.55x (float), 1.00x (double), 2.61x (cfloat), 2.01x
//     (cdouble) on its BLAS-3 core against the same driver in a vendor-present
//     build, and essentially ALL of it is the transposed panel gemm W = V^H A22:
//     that one alone is 4.81x / 1.00x / 4.99x / 3.12x while the NN update is
//     1.06x / 1.00x / 1.02x / 0.95x. gemm_kernels.cc:470-482 short-circuits every
//     transposed form to Direct/Tiled16 before the register ladder, the three
//     float TN escapes need m >= 128 (a block width measured worst end to end),
//     and complex cannot reach them at any width because :472 tests
//     transA == Transpose::Trans while a complex panel update is ConjTrans.
//     route_gemm.hh:113-114 refuses complex outright. Record and move on, as WP4
//     did; closing it is WP2 territory.
//   * DOUBLE IS ALREADY VENDOR-FREE FOR THE TRAILING UPDATE AT ZERO COST -- the
//     resolver hands double's trailing GEMMs to Native:RegisterTiled even with
//     cuBLAS present, and the two builds measure identically (11.8008 vs 11.8012
//     ms, separate processes and separate .so files).
// ---------------------------------------------------------------------------
template <typename T>
using GeqrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// ---------------------------------------------------------------------------
// DIRECT-CALL ENTRY POINTS.
//
// Exposed so tests can exercise a kernel without going through dispatch -- which
// is NOT convenience but a correctness requirement (potrf_native.hh:126-141, and
// tests/potrf_tests.cc:6-18 states it in full): route_resolve.hh:101 tests
// `if (Table::supports(forced, s)) return forced;` and falls through to
// automatic() at :111, so a test that sets BATCHLAS_GEQRF_ROUTE=cta and gets one
// gate wrong runs cuSOLVER and passes GREEN over a kernel nothing executed. A
// direct call cannot be served by a vendor.
//
// Each re-checks every RouteTable<Op::geqrf,T>::supports() gate and throws,
// because these are reachable WITHOUT the table.
//
// AND WHEN THE KERNEL LANDS, THE FACADE TEST MUST BE BIT-EXACT AGAINST THIS
// CALL, not a residual. tests/potrf_tests.cc:895-908 records this repository's
// fifth blind guard: a route-assertion-plus-residual test "stayed GREEN across
// all four scalar types while every number in it came from cuSOLVER", because a
// residual bound is satisfied by either implementation. The fix (:958-963) is
// that the vendor does not reproduce the kernel's reduction order, so an
// element-by-element ASSERT_EQ against this entry point discriminates and a
// residual does not.
// ---------------------------------------------------------------------------
template <typename T>
Event geqrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace);

template <typename T>
Event geqrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             GeqrfTrailingGemm<T> trailing_gemm = {});

// ---------------------------------------------------------------------------
// THE PANEL LEAF -- INTERNAL, and the one place the two tiers actually meet.
//
// It factorises an m x n panel in place (LAPACK ?GEQR2 semantics) and writes
// min(m, n) reflector scalars. It is what geqrf_cta_dispatch runs on a whole
// matrix and what geqrf_blocked_dispatch runs on each column panel, so the
// scaffolding's claim that "the blocked driver's panel leaf IS the CTA device
// function" is literally true -- ONE device body, `geqr2_panel_device` in
// src/extensions/geqrf_cta_device.hh.
//
// IT TAKES RAW POINTERS RATHER THAN A MatrixView ON PURPOSE, and this is the one
// place in WP5 where that is the safer spelling rather than the lazier one. A
// panel is a sub-view of the caller's matrix, and matrix.hh:1140 propagates the
// PARENT pointer array into any slice built with operator()(Slice, Slice) while
// the constructor DEFAULTS stride to ld*cols when 0 is passed
// (matrix.cc:1839-1842). Passing (ptr, ld, stride) explicitly removes both traps
// by removing the view.
//
// TWO RESIDENCIES, ONE DECISION SITE. The leaf stages the panel into local
// memory when it fits geqrf_cta_fits<T>() and streams it from global memory
// otherwise. Both run the SAME device function against different accessors, and
// `used_resident_out` (optional) reports which, so a test can assert it
// exercised the path it meant to -- the potrf_cta_debug_launch discipline.
//
// tau is addressed as tau_ptr[b * tau_batch_stride + tau_offset + j]. The stride
// is EXPLICIT because a panel's slice of tau is not contiguous with the panel:
// geqrf's contract packs tau per matrix with stride k = min(rows, cols) OF THE
// WHOLE MATRIX (cublas.cc:1259 does exactly this), while the panel's own
// min(m, n) is its width. Deriving the stride inside the leaf would give the
// blocked driver the panel's k and scatter tau across the wrong slots -- silently,
// and only for batch > 1.
// ---------------------------------------------------------------------------
template <typename T>
Event geqrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            T* tau_ptr, int tau_batch_stride, int tau_offset,
                            bool* used_resident_out = nullptr);

// Does an m x n panel fit the resident leaf at this local-memory budget? The
// SAME predicate the launcher applies, exposed so the route table's capacity and
// the blocked driver's per-panel choice cannot answer it differently -- the
// potrf_cta_launch_params defect (potrf_cta.cc:442-454), where a raw figure in
// the query against a padded one in the launcher produced an unhandled throw on
// a call the table had promised.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes);

}  // namespace batchlas::sycl_geqrf
