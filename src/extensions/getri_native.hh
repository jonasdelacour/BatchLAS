#pragma once

// Native batched GETRI -- declarations.
//
// STATUS: LIVE. getri_blocked_available<T>() is true for every scalar type and
// getri_blocked_dispatch is the driver in src/extensions/getri_blocked.cc, so
// RouteTable<Op::getri,T>::supports() admits the native arm for every square GPU
// shape and a vendor-free build takes it -- this is what closed inverse_tests. A
// vendor-PRESENT build is unchanged because preferred() is all-false, which is a
// withheld measured window, not a missing kernel.
//
// THE SHAPE OF THE OP, and it is the one LU op with a clear measured win to go and
// get. getri takes the FACTORED A and writes A^-1 into C
// (functions/getri.hh:55-61). Composed natively that is:
//     write the permutation P straight into C, then solve L (unit lower) and U
//     against it -- two routed trsm calls, ZERO permutation workspace.
//
// This is the `orgqr = ormqr on an identity` precedent (route_orgqr.hh:41-49) and
// it carries that precedent's obligation: RouteTable<Op::getri,T>::supports() must
// TRANSCRIBE the gates of the table that actually serves the call, because
// silently omitting an inherited gate is the wrong-answer class. route_getri.hh
// does that.
//
// getri_blocked.cc sits in EXTENSIONS_FACTORIZATION_SOURCES, not
// EXTENSIONS_CTA_SOURCES: it shares no device symbol with the getrf pair
// (src/extensions/CMakeLists.txt:77-86's cluster rule; orgqr_blocked.cc's
// precedent at :62-66). If a future getri ever calls the getrf CTA device body it
// MOVES, and getting that wrong is a hard `ptxas fatal: Unresolved extern
// function`, never a silent miscompile.
//
// ===========================================================================
// THE MEASUREMENT, and its crossover. Composed against cublas<t>getriBatched at
// saturating batch, in process, host oracle (experiments/wp6_lu/baseline/):
//
//     n(batch)    float   double  cfloat  cdouble
//      32(8192)    0.54    0.23    0.23    0.23
//      64(8192)    0.83    0.53    0.35    0.54
//     128(4096)    1.32    0.90    1.06    0.89
//     256(2048)    3.89    1.16    2.05    1.04
//     512(512)     5.75    1.28    3.01    1.02
//    1024(128)    15.66    1.16    6.05    1.11
//    2048(32)     74.87    3.93   25.88    4.30
//
// Geomean 1.60x over 28 cells, 18 wins, worst 0.23x, best 74.9x. Crossover
// n ~ 128 for float/cfloat and n ~ 256 for double/cdouble; BELOW it cuBLAS's
// small-n getriBatched path wins by up to 4.3x.
//
// TWO HONESTY CONSTRAINTS ON THOSE NUMBERS, both stated rather than silently
// corrected:
//   * EVERY n >= 512 CELL IS AGAINST AN UNSATURATED VENDOR. cuBLAS getrf at
//     float n=1024 is still falling 10% from batch 128 to 256, and cdouble
//     n=2048 does 64x the work for 1.03x the time from batch 1 to 64. The 74.9x
//     is a comparison against a routine barely using the GPU at that batch, NOT
//     "74.9x faster than cuBLAS".
//   * THE GRID'S BATCH SCHEDULE PENALISES THE VENDOR AT ONE CELL: cuBLAS getri
//     float n=256 is best at batch 256 (13.85 us/item) and DEGRADES to 20.38 at
//     batch 2048, so the 3.89x at that cell carries ~1.47x of pessimism.
//
// AND THE WIN IS MOSTLY THE PERMUTATION, WHICH getri GETS FOR FREE. A
// LAPACK-faithful laswp is 51% of the composed call at n=128 (measured by a
// deliberate break: getri_trsm 0.4580 -> 0.2251 ms without the row exchange). It
// is structurally slow -- ipiv is a SEQUENCE so it must be walked in order, one
// work-item per column, and in column-major consecutive work-items land ld apart,
// 32 transactions per warp access. Collapsing it (apply the interchanges to an
// identity index array once, then GATHER, so consecutive work-items land on
// consecutive rows) turns the geomean from 0.97x into 1.60x. getri needs NO
// permutation kernel and NO workspace to get it: write P straight into C instead
// of writing I and then permuting it -- same store count, one kernel.
// ===========================================================================
//
// TWO CONTRACT FACTS A NATIVE getri MUST HONOUR, both measured:
//
//   (a) A IS NOT WRITTEN. cuBLAS's prototype takes `const T* const A[]`
//       (cublas_api.h:5568-5576) and measured max|A_after - A_factored| == 0 for
//       all four types; max|A*Ainv - I| = 2.4e-07 / 3.3e-16 / 2.4e-07 / 4.4e-16.
//       cuBLAS also does NOT support in-place (A == C). The other two backends
//       synthesise the out-of-place contract with a copy and each has a latent
//       assumption worth knowing: rocsolver.cc:330-332 memcpy's
//       sizeof(T)*A.stride()*A.batch_size() and then inverts C in place, silently
//       assuming C.stride() == A.stride() and a contiguous batch;
//       netlib_lapack.cc:1362-1365 does a host std::copy of n*n per item and
//       IGNORES ld. A native arm may pick either mechanism but must not overwrite
//       A.
//
//   (b) info IS EXACT-ZERO SEMANTICS, NOT A TOLERANCE, and the full measurement is
//       in getrf_native.hh's PIVOT CONTRACT note 4. In one line: a native kernel
//       that flags |pivot| < eps reports non-zero where the vendor reports zero,
//       and that divergence is invisible to any native-vs-native test. Do not copy
//       potrf's first-failure masking or its quench: LAPACK's LU does neither and
//       neither does cuBLAS.
//
// AND THE PIVOT FORMAT IS getrf's, backend-dependent: PACKED 1-BASED INT32 on
// CUDA (cublas.cc:1537) and ROCm (rocsolver.cc:227), GENUINE 1-BASED INT64 on
// NETLIB (netlib_lapack.cc:1356 reads it back through piv.as_span<int64_t>(), an
// identity). An INTERCHANGE LIST, not a permutation vector. getri must not pick a
// different convention from getrf -- the two ops have independent env variables
// and independent preferred() windows, so every mixture of native and vendor arms
// is reachable, and a native getrf feeding a vendor getri reads garbage with no
// gate able to see it.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getri {

// ---------------------------------------------------------------------------
// CAPABILITY. ONE flag: getri has ONE native arm and no capacity to advertise --
// the two solves are the routed trsm, whose own tiers carry no upper bound on the
// order, and writing P into C needs no resident tile.
//
// DEFINED IN src/extensions/getri_blocked.cc, beside the driver
// (potrf_native.hh:81-92: full explicit specialisations link from wherever they
// sit, so co-location is what makes "the flag is true" and "the file is compiled"
// the same fact). TRUE for all four types.
// ---------------------------------------------------------------------------
template <typename T>
bool getri_blocked_available();

// ---------------------------------------------------------------------------
// WORKSPACE, in bytes, for the native arm.
//
// From BumpAllocator::measuring() (mempool.hh:186-190's workspace_bytes), NEVER
// hand-summed: mempool.hh:96-104 checks capacity as the alignment-rounded
// alloc_size from the UNALIGNED cursor while :111-113 advances by only the raw
// data extent, so an exactly-computed figure fails the allocator's own capacity
// check; and required_bytes() (mempool.hh:45-51) rounds to the coarsest quantum
// the sequence asked for, because callers re-serve the number through
// allocate<std::byte>(). It is also what makes the facade's max(native, vendor)
// safe: every term must be such a rounded figure.
//
// A ZERO ANSWER IS EXPECTED HERE, not merely legitimate. The measured design
// (write P straight into C, then two routed trsm calls) allocates NOTHING: the
// permutation is a store pattern rather than a buffer, and the routed trsm takes
// no workspace. The vendor's own term is small and in the same direction --
// getri_vendor_buffer_size is BumpAllocator::allocation_size<int>(batch), a
// per-item info array (cublas.cc:1552), 512 B at n=2048 batch=32. So the facade's
// max(native, vendor) will normally be the VENDOR term, and the LU family is never
// the workspace hazard WP5's ormqr was.
//
// (The rejected alternative is worth recording because its cost is not obvious:
// writing an identity and then PERMUTING it needs int32[n] per item -- 262,144 B
// at n=2048 batch=32, 1,048,576 B at n=32 batch=8192 -- for no benefit, since the
// permuted store is the same store count.)
//
// IT IS CALLED FROM INSIDE A LAYOUT FUNCTION UNDER measuring(): src/extensions/
// inv.cc:35, reached from inv_buffer_size at :54-57. Per mempool.hh:180-186 it
// must therefore be PURE WITH RESPECT TO THE WORKSPACE -- no workspace read or
// write, no kernel launch, and any nested query asked about the CALLER's views. It
// must also not dereference A.data_ptr(); extents, ld, stride, batch and
// is_heterogeneous() are metadata and are safe.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getri_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// ---------------------------------------------------------------------------
// THE TRIANGULAR SOLVE, INJECTED rather than hardcoded.
//
// Both solves go through the ROUTED trsm. Calling a native kernel entry point
// straight from a driver TU is the RECORDED DEFECT of WP3 step 16
// (trsm_native.hh:82-104, fix at level3.cc:186-231): it bypasses
// RouteTable<Op::trsm> entirely, so the driver would get the native kernel even on
// shapes measured to lose.
//
// It is ALSO the only way to reach trsm from here at all: this driver is
// instantiated per scalar type with NO Backend parameter, and trsm<B,T> needs one.
//
// Signature identical to the routed batchlas::trsm's positional form
// (functions/trsm.hh:100-108). NOTE alpha COMES THIRD, not last -- the old
// spelling is a DELETED overload at :121-138 so a stale call cannot silently
// compile.
//
// NO BUFFER-SIZE TWIN, unlike WP5's OrgqrApplyQ / OrgqrApplyQBufferSize pair
// (orgqr_native.hh:118, :137, injected together at factorization.cc:349-363 and
// :431-440). That pair exists because the routed ormqr HAS a workspace whose size
// must come from the same resolution as the call; the routed trsm has none, so
// there is nothing here for a query and a call to disagree about.
//
// AN ABSENT INJECTION THROWS rather than silently reaching for a native kernel: a
// direct caller (a test) injects trsm<Backend::CUDA, T> itself, which is still a
// call no vendor getri can serve.
// ---------------------------------------------------------------------------
template <typename T>
using GetriSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

// ---------------------------------------------------------------------------
// DIRECT-CALL ENTRY POINT.
//
// Exposed so tests can exercise the driver without going through dispatch -- a
// correctness requirement, not a convenience (potrf_native.hh:126-141,
// tests/potrf_tests.cc:6-18): route_resolve.hh:165 gates the forced route on
// supports() and falls through to automatic() at :175, so a test that sets
// BATCHLAS_GETRI_ROUTE and gets one gate wrong runs cuBLAS and passes GREEN over a
// driver nothing executed.
//
// It must re-check every RouteTable<Op::getri,T>::supports() gate and throw,
// because it is reachable WITHOUT the table. Today it throws unconditionally.
//
// ITS ORACLE IS THE HOST, NEVER THE VENDOR -- a vendor reference is INERT in a
// vendor-free build, where the resolver falls back to the code under test and the
// test compares the kernel with itself (tests/geqrf_tests.cc:8-12). For getri the
// oracle is ||A A^-1 - I|| in double regardless of T, on a matrix that has been
// ROW-PERMUTED so the pivot list is non-trivial by construction. THIS IS NOT
// PEDANTRY: on the obvious diagonally dominant test matrix, ignoring the pivot
// list entirely leaves the residual BIT-IDENTICAL (measured, getrf_native.hh note
// 5), and tests/inverse_tests.cc -- the one getri test in the tree -- is a single
// float case at n=40, batch=2 on Matrix<float>::Random().
// ---------------------------------------------------------------------------
template <typename T>
Event getri_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             GetriSolveTrsm<T> solve_trsm = {});

}  // namespace batchlas::sycl_getri
