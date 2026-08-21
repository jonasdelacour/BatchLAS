#pragma once

// Native batched POTRF -- declarations.
//
// Read WP4_POTRF_SPEC_CORRECTIONS.md first, then WP4_POTRF_SPEC.md. The spec
// predates WP0-WP3 and is stale in several places the corrections list; where
// they disagree the corrections win.
//
// PHASE 1 IS THE CTA KERNEL AND NOTHING ELSE.
//
//   Algorithm::CTA      -- one matrix resident in local memory for the whole
//                          factorisation, a work-group (or one 32-wide
//                          sub-group, with G matrices packed per work-group)
//                          per matrix. Serves order <= potrf_cta_max_n<T>().
//                          Both Uplo, all four scalar types.
//   Algorithm::Blocked  -- the host-blocked driver for larger orders, which
//                          would call the CTA body as its diagonal-block leaf.
//                          NOT WRITTEN. potrf_blocked_available<T>() returns
//                          false for every type and RouteTable<Op::potrf,T>
//                          therefore reports that arm unsupported, so an order
//                          above the CTA ceiling still has no native route.
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
// and one more of its own: WP4_POTRF_SPEC.md:273's {105, 74, 74, 52} follow from
// `slm_budget = 45056`, and that budget is refuted (W1). The 49152 in
// build/include/batchlas/device_limits.hh is HARDCODED by
// cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern and is
// not a detected property at all -- the detection routine never queries
// local_mem_size. Measured, this box reports 101,376 B and launches a kernel
// with 0 B static shared at exactly that (experiments/wp4_potrf/slm/README.md).
// At the 97,280 B budget (runtime - 4096 reserve) the ceilings are
// {float 155, double 109, complex<float> 109, complex<double> 77}, and all four
// were launched cold and computed the right answer before this kernel existed
// (experiments/wp4_potrf/slm/maxn_fitcheck.csv). Shipping 105 would leave float
// n in 106..155 with NO ROUTE AT ALL in a vendor-free build
// (route_resolve.hh:60-63).
template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes);

// The same question at this repository's standard budget (the 97,280 B above).
// Kept because RouteTable's PotrfShape::cta_max_n has always been documented in
// those terms and because the tests pin the four numbers.
template <typename T>
int potrf_cta_max_n();

// Whether the blocked driver exists in this build. FALSE for every type: it is
// Phase 2 and has not been written. Kept symmetric with the capacity above so
// that "this tier is not in this build" is expressed the same way for both and
// cannot be forgotten when one of them lands.
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

}  // namespace batchlas::sycl_potrf
