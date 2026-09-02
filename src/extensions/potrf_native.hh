#pragma once

// Native batched POTRF -- declarations.
//
// Read docs/perf/potrf.md#what-the-spec-got-wrong first, then docs/perf/potrf.md#what-the-spec-got-wrong. The spec
// predates WP0-WP3 and is stale in several places the corrections list; where
// they disagree the corrections win.
//
// TWO NATIVE TIERS, ONE LEAF.
//
//   Algorithm::CTA      -- one matrix resident in local memory for the whole
//                          factorisation, a work-group (or one 32-wide
//                          sub-group, with G matrices packed per work-group)
//                          per matrix. Serves order <= potrf_cta_max_n<T>().
//                          Both Uplo, all four scalar types.
//   Algorithm::Blocked  -- WP4 Phase 2. The host-blocked right-looking driver
//                          for larger orders. Its diagonal-block leaf IS the
//                          CTA kernel above, handed a SUB-VIEW, so the crossover
//                          between the two tiers is a capacity and not a tuned
//                          guess. Uplo::LOWER ONLY (see potrf_blocked_dispatch).
//
// ROUTE-NEUTRAL AS SHIPPED. RouteTable<Op::potrf,T>::preferred() is still false
// for both native arms, so a vendor-present build keeps taking cuSOLVER for
// every shape and the blocked driver is reachable only when a caller pins it
// (BATCHLAS_POTRF_ROUTE=blocked) or when the build has no vendor at all
// (route_resolve.hh:60-63). Flipping preferred() is a separate, measured step.
//
// WHY THIS FILE EXISTS SEPARATELY FROM potrf_cta_device.hh: the route table
// (route_potrf.hh, through src/backends/potrf_route.hh) must be able to ASK the
// capability questions without acquiring a dependency on device code, and the
// vendor-free facade must be able to CALL the launcher without including
// <sycl/sycl.hpp>. Both include only this. Same split, and same reason, as
// src/sycl/trsm_native.hh:38-125.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_potrf {

// The largest order the CTA kernel can hold resident, per scalar type, for a
// given per-work-group local-memory budget in BYTES.
//
// This is the honest spelling of the capability, and the no-argument overload
// below is a convenience over it. The ceiling is a pure function of the SLM
// budget and the (NB, TS) constants, and the budget is a DEVICE property -- so
// baking one device's answer into a constant is what would make supports()
// claim an unlaunchable route on a smaller device.
//
// It is a function and not a constexpr literal for route_trsm.hh:62-72's reason
// and one more of its own: docs/perf/potrf.md#what-the-spec-got-wrong:273's {105, 74, 74, 52} follow from
// `slm_budget = 45056`, and that budget is refuted (W1). The 49152 in
// build/include/batchlas/device_limits.hh is HARDCODED by
// cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern and is
// not a detected property at all -- the detection routine never queries
// local_mem_size. Measured, this box reports 101,376 B and launches a kernel
// with 0 B static shared at exactly that (docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings).
// At the 97,280 B budget (runtime - 4096 reserve) the ceilings are
// {float 155, double 109, complex<float> 109, complex<double> 77}, and all four
// were launched cold and computed the right answer before this kernel existed
// (docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings). Shipping 105 would leave float
// n in 106..155 with NO ROUTE AT ALL in a vendor-free build
// (route_resolve.hh:60-63).
template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes);

// The same question at this repository's standard budget (the 97,280 B above).
// Kept because RouteTable's PotrfShape::cta_max_n has always been documented in
// those terms and because the tests pin the four numbers.
template <typename T>
int potrf_cta_max_n();

// Whether the blocked driver exists in this build. TRUE for all four types
// since WP4 Phase 2. Kept symmetric with the capacity above so that "this tier
// is not in this build" is expressed the same way for both and cannot be
// forgotten when one of them lands.
//
// IT IS DEFINED IN src/extensions/potrf_blocked.cc, NOT HERE AND NOT IN
// potrf_cta.cc, and that placement is load-bearing. These are full explicit
// specialisations, so they link from wherever they sit -- and sitting anywhere
// but beside the driver would let a build advertise the tier while
// potrf_blocked.cc is absent from EXTENSIONS_CTA_SOURCES or #if 0'd out. That
// is the state route_trsm.hh:99-110 names ("the table must describe the BUILD,
// not the design"). Co-located, "the flag is true" and "the file is compiled"
// are the same fact. sycl_trsm::trsm_blocked_available has the identical shape
// (trsm_native.cc:927-930).
template <typename T>
bool potrf_blocked_available();

// Workspace the CTA route needs, in bytes.
//
// It is exactly the `info` fallback: one int32 per batch item, drawn only when
// the caller passes no (or a short) info span. No part of the kernel's own state
// is in global memory -- the tile, the pivot column, the failure flag and the
// tile-index table are all local memory sized at launch -- so this number does
// not depend on NB, TS, L or G, and no tuning knob can desynchronise the query
// from the call.
//
// Obtained by replaying the layout through BumpAllocator::measuring(), NEVER by
// hand-summing an expression: mempool.hh:82-86 checks capacity from the
// UNALIGNED cursor while :118-120 advances only by the data extent, so an
// "exactly computed" figure fails the allocator's own capacity check, and
// required_bytes() additionally rounds to the coarsest quantum the sequence
// asked for because callers re-serve the number through allocate<std::byte>()
// (src/extensions/ortho.cc:78 is such a caller).
template <typename T>
std::size_t potrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

// The launch geometry the kernel WOULD use for this shape, for tests.
//
// Returns G (matrices per work-group) in the low 16 bits and L (work-items per
// matrix) in the high 16 bits; 0 if the shape does not fit.
//
// This exists because a test cannot otherwise SEE the property it depends on.
// PackedBatchMatchesSolo is the only test that can detect the half of the (P1)
// `lane < ib` publish defect that writes into a NEIGHBOURING matrix, and it can
// only do so when the launch actually packs G > 1 matrices into a work-group.
// Which n do that is a consequence of kPotrfSlmSoftTarget, the clamp on G, and
// sizeof(T) -- none of which the test can see. It asserted the property in a
// comment and nowhere in code, so any change to those three would have silently
// collapsed it to G == 1 and left the defect invisible again while the test
// stayed green. That is this repository's recorded blind-guard shape, and the
// fix is to let the test ask.
template <typename T>
unsigned potrf_cta_debug_launch(Queue& ctx, int n, int batch);

// The CTA kernel. DIRECT-CALL ENTRY POINT.
//
// Exposed so tests can exercise the kernel without going through dispatch --
// which is not convenience but a correctness requirement for the tests: a forced
// route that supports() rejects falls back to automatic() (route_resolve.hh:101,
// :111), so a test that pins BATCHLAS_POTRF_ROUTE=cta and gets it wrong runs
// cuSOLVER and passes GREEN over an untested kernel. A direct call cannot be
// served by a vendor. Same reason trsm_native.hh:68-71 exposes V1.
//
// `A` must be square, GPU-resident, homogeneous-batch, and of order
// <= potrf_cta_max_n<T>() -- exactly RouteTable<Op::potrf,T>::supports()' gates.
// Each is re-checked here and throws, because this entry point is reachable
// without the table.
template <typename T>
Event potrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Uplo uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

// ---------------------------------------------------------------------------
// WP4 Phase 2 -- the BLOCKED driver.
// ---------------------------------------------------------------------------

// The trailing-update GEMM, INJECTED rather than hardcoded.
//
// WHY, and this is a RECORDED DEFECT rather than a style preference: WP3's V2
// called sycl_gemm::gemm_custom directly, which is the NATIVE kernel entry point
// and bypasses RouteTable<Op::gemm> entirely, so the trailing updates always got
// the native kernel whether or not it was the better one (WP3 step 16;
// trsm_native.hh:82-104 carries the measurement). potrf's trailing update is
// 65-95% of a vendor-free blocked factorisation and 33-48% of the end-to-end
// time is at stake in it, so the same mistake here would be the expensive one.
//
// Measured cost of vendor freedom in the trailing update. The figures first
// recorded here (float 0.13-0.18x, cfloat 0.21-0.23x, cdouble 0.33-0.34x,
// double 1.15-1.19x) were kernel-level and were taken at the OLD float W = 32,
// where the W x W diagonal-block gemm could not reach float's transposed
// register kernel. RE-MEASURED end to end in WP4 Phase 2 triage, at the shipped
// constants, on a correct factorisation, with the trsm seam held at NATIVE so
// the gemm seam is the only variable (whole-potrf ms, native gemm / vendor gemm,
// >1 = native faster):
//
//   float   n=512 b=256  3.393/2.256 = 0.665x   n=1024 b=256 15.802/10.663 = 0.675x
//                                               n=2048 b=128 46.473/29.569 = 0.636x
//   double  n=512 b=256 11.794/13.003 = 1.102x  n=1024 b=256 78.471/87.893 = 1.120x
//   cfloat  n=512 b=256  7.790/3.781 = 0.485x
//   cdouble n=512 b=128 60.575/25.770 = 0.425x
//
// The ORDERING is unchanged -- double wants the native gemm, the other three
// want the vendor when one exists -- but the float gap is 0.64-0.68x, not
// 0.13-0.18x, so do not quote the old numbers. Only the router knows which is
// which, and only the facade can ask it.
//
// The signature is deliberately identical to both sycl_gemm::gemm_custom and the
// routed batchlas::gemm, so neither side adapts. An EMPTY function means "use
// gemm_custom", which is what keeps this kernel layer free of the dispatch layer:
// a direct caller (the tests, and any benchmark that must not be silently served
// by a vendor) gets the native kernel with no dispatch dependency, and a
// vendor-free build is unaffected because the resolver falls back to native
// there anyway (route_resolve.hh:60-63).
template <typename T>
using PotrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// The panel solve, injected for the same reason and by a STRONGER argument than
// the GEMM's.
//
// potrf's kernel family is instantiated PER SCALAR TYPE with no Backend
// parameter (potrf_cta.cc:706-726: "no Backend cross-product ... pure cost" in a
// build that is device-link-bound). batchlas::trsm<Back,T> requires a Backend.
// So there is no way at all to reach the ROUTED trsm from this TU except by
// injection -- the alternative is to add a Backend parameter to the whole potrf
// family and triple its instantiation count.
//
// WHY ROUTED AND NOT A BESPOKE PANEL KERNEL (WP4 open question 5, settled by
// measurement): on the REAL panel shapes -- m2 x ib sub-views carrying the
// parent ld AND the parent batch stride -- at batch 128 and n in {512,1024,2048},
// the routed trsm beat the vendor in 46 of 48 panel-stage cells.
//
// THAT MEASUREMENT WAS TAKEN ON A KERNEL THAT WAS COMPUTING THE WRONG ANSWER,
// which WP4 Phase 2 triage found and fixed: V1 staged its triangle into local
// memory and read the diagonal back with no barrier between, and the launcher's
// work-group ladder leaves wg = 32 -- the only width at which the race cannot
// express itself -- exactly below q*batch ~ 65k, which no panel shape here is.
// So the old cells are not evidence of anything. RE-MEASURED post-fix, end to
// end, with the GEMM seam held at the VENDOR so the panel solve is the only
// variable (whole-potrf ms, vendor trsm / native trsm, >1 = native faster,
// bad == 0 in every arm):
//
//   float   n=512  b=256   3.174/2.256  = 1.407x
//   float   n=1024 b=256  15.144/10.663 = 1.420x
//   float   n=2048 b=128  38.674/29.569 = 1.308x
//   double  n=512  b=256  15.503/13.003 = 1.192x
//   double  n=1024 b=256  99.482/87.893 = 1.132x
//   cfloat  n=512  b=256   7.212/3.781  = 1.907x
//   cdouble n=512  b=128  69.349/25.770 = 2.691x
//
// The verdict stands, now on correct answers, and the routed trsm wins in every
// cell tried. Writing a bespoke kernel would also be aimed at the wrong stage:
// the panel solve is 5-22% of a vendor-free blocked potrf against 65-95% for the
// trailing update, so a hypothetical 2x here is worth 3-11% end to end.
//
// Argument order is the positional entry point's, functions/trsm.hh:100-108 --
// ALPHA IS IN POSITION 4, not last and not first. TrsmOptions puts it first
// (options.hh:257-264) and confusing the two is a recorded compile error (W13).
//
// An EMPTY function means "use sycl_trsm::trsm_native_blocked", which handles
// any triangular order (it falls through to a single V1 solve when the order
// fits the CTA capacity) and needs no workspace.
template <typename T>
using PotrfPanelSolve = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,   // A: the ib x ib factored L11
    const MatrixView<T, MatrixFormat::Dense>&,   // B: the m2 x ib panel, in place
    T,                                           // alpha
    Side, Uplo, Transpose, Diag)>;

// Workspace the BLOCKED route needs, in bytes.
//
// Obtained by replaying potrf_blocked_layout through BumpAllocator::measuring(),
// NEVER hand-summed -- same rule, and same reason, as potrf_cta_buffer_size
// above. The block width nb and the trailing panel width W come from ONE pure
// function that the driver also calls, so the query cannot size a layout the
// call does not build (the potrf_cta_launch_params discipline,
// potrf_cta.cc:442-454, where a raw figure in the query against a padded one in
// the launcher produced an unhandled throw on a call the table had promised).
template <typename T>
std::size_t potrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Uplo uplo);

// The BLOCK WIDTHS the driver WOULD use for this order, for tests.
//
// Returns nb (the diagonal-block order, hence the leaf's order and the trailing
// update's k) in the low 16 bits and W (the trailing-update column-panel width)
// in the high 16 bits.
//
// SAME REASON AS potrf_cta_debug_launch ABOVE, and it is the same recorded
// failure shape: a test that must straddle a block boundary cannot see where
// the boundary is. nb is a measured per-type constant, clamped by the DEVICE's
// SLM ceiling and then rounded down to a whole number of trsm_cta_max_n<T>()
// blocks (potrf_blocked.cc) -- three inputs a test cannot compute. A test that
// hardcoded {128, 96, 96, 64} would keep passing after any of the three moved,
// while silently no longer testing a short final block, a multi-block sweep, or
// a global info offset. That is this repository's blind-guard shape; letting the
// test ask is the fix.
//
// It is a QUERY over the same pure function the driver and the buffer-size query
// both call, so it cannot report a blocking the call does not use.
template <typename T>
unsigned potrf_blocked_debug_params(Queue& ctx, int n);

// The blocked driver. DIRECT-CALL ENTRY POINT, same reason as
// potrf_cta_dispatch above: a forced route that supports() rejects falls back to
// automatic() (route_resolve.hh:101, :111) and silently runs the vendor, so a
// test that pins BATCHLAS_POTRF_ROUTE=blocked and gets it wrong passes GREEN
// over cuSOLVER's numbers. A direct call cannot be served by a vendor.
//
// Uplo::LOWER ONLY, and this is a CORRECTNESS restriction, not a fit one: the
// right-looking schedule below (panel BELOW the diagonal block, trailing update
// A22 -= L21 L21^H) is Lower-shaped, and handed an Upper view it would read and
// overwrite the wrong triangle. It throws. RouteTable<Op::potrf,T>::supports()
// carries the matching gate at route_potrf.hh:278, and that line comes off only
// when the driver mirrors (syev.hh:840-847) or grows a transposed schedule.
//
// `info` follows LAPACK: 0 on success, and on failure the 1-based GLOBAL order
// of the leading minor that is not positive definite, FIRST FAILURE WINS. The
// leaf reports an index LOCAL to the sub-view it was handed
// (potrf_cta_device.hh:196) and writes it UNCONDITIONALLY
// (potrf_cta.cc:614-615), so the driver must both translate and merge; see
// potrf_blocked.cc for how, and for why that is done in the driver rather than
// by changing the leaf.
template <typename T>
Event potrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Uplo uplo,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             PotrfTrailingGemm<T> trailing_gemm = {},
                             PotrfPanelSolve<T> panel_solve = {});

}  // namespace batchlas::sycl_potrf
