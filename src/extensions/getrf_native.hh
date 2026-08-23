#pragma once

// Native batched GETRF -- declarations.
//
// STATUS: BOTH TIERS ARE LIVE FOR ALL FOUR SCALAR TYPES.
// getrf_cta_max_n_for_slm returns a real per-type ceiling derived from the
// RUNTIME local-memory budget, and getrf_blocked_available is true, so
// RouteTable<Op::getrf,T>::supports() accepts both native arms. What has NOT
// changed is preferred(), which is still false everywhere: a vendor-present build
// keeps taking cuBLAS for every shape and these kernels are reachable only
// through BATCHLAS_GETRF_ROUTE, through the direct entry points below, or in a
// vendor-free build (route_resolve.hh:60-63). Flipping preferred() is a later
// step gated on a measured grid.
//
// TWO NATIVE TIERS, ONE LEAF (the potrf/geqrf shape, potrf_native.hh:9-21):
//
//   Algorithm::CTA      -- one n x n matrix resident in local memory for the
//                          whole factorisation, a work-group per matrix. Serves
//                          orders up to getrf_cta_max_n_for_slm<T>().
//   Algorithm::Blocked  -- the host-blocked right-looking driver for larger
//                          orders. Its diagonal-panel leaf IS the CTA DEVICE
//                          FUNCTION, handed a raw sub-panel. That sharing is why
//                          both TUs must sit in ONE device-code cluster -- see
//                          src/extensions/CMakeLists.txt and its :77-86 grouping
//                          rule (W12).
//
// WHY THIS FILE IS SEPARATE FROM A FUTURE getrf_cta_device.hh: the route table
// (route_getrf.hh, through src/backends/getrf_route.hh) must be able to ASK the
// capability questions without acquiring a dependency on device code, and the
// vendor-free facade must be able to CALL the launcher without including
// <sycl/sycl.hpp>. Both include only this. Same split, and same reason, as
// src/extensions/potrf_native.hh:28-35, geqrf_native.hh:41-46 and
// src/sycl/trsm_native.hh:38-125.
//
// ===========================================================================
// THE PIVOT CONTRACT -- READ THIS BEFORE WRITING ONE LINE OF KERNEL.
//
// It has NO WP4/WP5 analogue and it is the single most likely silent-wrong-answer
// channel in WP6. Everything below is MEASURED against a host LAPACKE oracle
// through the public API, not read off the headers.
//
//   1. THE PUBLIC SPAN IS int64_t AND THE PHYSICAL FORMAT IS NOT. There is no
//      conversion anywhere on the GPU backends: cublas.cc:1508 does
//      `pivots.as_span<int>()`, which sycl-span.hh:45-47 implements as a
//      reinterpret_cast with the size rescaled -- zero cost, zero conversion --
//      and hands cuBLAS PACKED INT32 occupying the FIRST HALF of the caller's
//      int64 buffer. rocsolver.cc:227 is identical (used at :232, :239, :271,
//      :326). NETLIB is the opposite and genuinely int64:
//      netlib_lapack.cc:1291 allocates an int scratch, LAPACKE fills it, and
//      :1312-1320 launches a SYCL parallel_for that WIDENS it into the caller's
//      span.
//
//      Measured, n=6 batch=3, buffer pre-poisoned with 0x0BADBEEF0BADBEEF:
//          piv raw int64[0..3]      : 0x0000000400000003 0x0000000600000005 ...
//          piv as int32[0..5] (b=0) : 3 4 5 6 5 6
//          LAPACKE ipiv[0..5] (b=0) : 3 4 5 6 5 6
//          int32-packed mismatches vs LAPACKE: 0 / 18
//          int64        mismatches vs LAPACKE: 18 / 18
//      On CUDA/ROCm the TOP HALF of the pivot buffer is never written, although
//      options.hh:616-617 makes every caller allocate rows*batch int64.
//
//   2. THE VALUES ARE 1-BASED AND ARE AN INTERCHANGE LIST, NOT A PERMUTATION.
//      ipiv[k] is the row swapped WITH row k at step k, applied in order. The
//      exact-singular probe returned `2 2 3 4` for a 4x4 -- a permutation vector
//      could not repeat 2. Established by agreement with the host oracle, not by
//      the name.
//
//   3. THEREFORE A NATIVE getrf MUST PICK A FORMAT AND AGREE WITH WHATEVER SERVES
//      getrs/getri ON THE SAME CALL. That mixed combination is entirely
//      reachable: three independent env variables (BATCHLAS_GETRF_ROUTE /
//      _GETRS_ / _GETRI_) and three independent preferred() windows. A native
//      getrf writing int64 into a buffer a vendor getri reads as packed int32
//      returns silent garbage -- no gate anywhere can see it, because no shape
//      field can express "the op downstream of me resolved differently".
//
//      THE CONTRACT WP6 ADOPTED, and it is the FORMAT that is matched bit for
//      bit, not the VALUES: packed 1-based int32 through as_span<int>() on CUDA
//      and ROCm, an interchange LIST rather than a permutation, so that ANY
//      mixture of native and vendor arms composes.
//
//      WHAT SHIPPED, AND THE ONE PLACE IT DELIBERATELY DIVERGES FROM cuBLAS.
//      The kernel writes packed 1-based int32 with stride n per item, exactly the
//      layout cublas?getrfBatched uses. The VALUES agree with cuBLAS for float
//      and double and DO NOT for complex, because cuBLAS pivots on the MODULUS
//      where LAPACK's I?AMAX -- and therefore this kernel, and NETLIB -- pivot on
//      cabs1 = |Re| + |Im|. That is measured, not inferred: on a matrix built to
//      separate the two functionals, native gives ipiv[0] = 2 (== host LAPACKE)
//      and cuBLAS gives 1, for both cfloat and cdouble; substituting the modulus
//      into the kernel reproduces cuBLAS exactly. See getrf_cta_device.hh's
//      PIVOT METRIC note for the full record.
//
//      THE ONE CONFIGURATION THE FORMAT CANNOT COVER IS NOW GATED IN THE TABLE.
//      Backend::NETLIB on a GPU QUEUE is reachable (Queue's backend and its device
//      are independent, and s.is_gpu reads the DEVICE), and there the native
//      packed int32 and netlib's genuine int64 disagree. Measured before the gate:
//      a forced native getrf feeding netlib's getri gave ||A*C - I||_F / n =
//      5.32e-01 with info == 0, against 5.15e-07 when both arms agreed -- silent,
//      and invisible to tests/getrf_tests.cc, which skips every NETLIB row because
//      its queue is a CPU queue. All three tables now carry
//      `if (s.backend == Backend::NETLIB) return false;` in supports()
//      (route_getrf.hh, route_getrs.hh, route_getri.hh); the shape builders
//      already set s.backend = B, which is what made the gate one predicate.
//
//      MIXING ARMS IS STILL SAFE, and the reason is worth stating because it is
//      the question this whole note exists to answer: getrs and getri consume
//      ipiv together with the FACTOR the same getrf produced, and any valid pivot
//      sequence is self-consistent with its own factor. What no API in this tree
//      permits -- factoring with one implementation and permuting with another's
//      ipiv -- is the only thing the divergence could break.
//
//      IT DOES CONSTRAIN THE TESTS: a pivot oracle must be the HOST, never the
//      vendor, or complex goes red for the wrong reason. A CROSS-OP TEST (native
//      getrf -> vendor getri, and vendor getrf -> native getri) is still the
//      thing that pins the FORMAT, and it must assert on the RESIDUAL rather than
//      on pivot equality.
//
//   4. info IS EXACT-ZERO SEMANTICS, NOT A TOLERANCE. Measured on a matrix whose
//      step 2 cancels to a true binary zero: device info == 2, host LAPACK info
//      == 2, 1-based, per item, identical. On a float matrix with a duplicated
//      column the device produced U(3,3) = -1.375e-08 and reported info = 0 while
//      the host got a true 0.0 and reported 3. A native kernel that flags
//      |pivot| < eps reports non-zero where the vendor reports zero, and that
//      divergence is invisible to any native-vs-native test. Both implementations
//      also CONTINUE past a zero pivot and leave the rest finite (LAPACK skips
//      the reciprocal scale when the pivot is exactly zero); a kernel that divides
//      unconditionally produces Inf/NaN where the vendor gives finite garbage.
//
//      DO NOT COPY potrf's TWO EXTRA RULES. potrf_blocked.cc:425-433 masks
//      first-failure-wins across panels and :436-440 QUENCHES a failed item to
//      keep it finite. LAPACK's LU does neither and neither does cuBLAS, so getrf
//      keeps info-only semantics. What DOES transfer verbatim is potrf's
//      info_target fallback rule (potrf_cta.cc:688-695, potrf_blocked.cc:594-616):
//      a short-or-empty span means "not requested" and falls back to pool
//      scratch, and it is THAT span which is zeroed, not the caller's info_out.
//      src/linalg-impl.hh:767-771 is the shared helper.
//
//   5. THE OBVIOUS TEST MATRIX MAKES ALL OF THIS VACUOUS. Measured, and it cost
//      the baseline harness a full rewrite: on a diagonally dominant matrix
//      (A = rand + n*I -- the natural "well conditioned so the residual measures
//      the kernel" choice) partial pivoting selects the diagonal at EVERY step,
//      so ipiv is the identity and no row is ever exchanged. Two deliberate
//      breaks -- ignore the pivot list, and drop the row interchange from the
//      composition -- both left the residual BIT-IDENTICAL (2.446e-07 float /
//      1.055e-15 cdouble). The fix is in the SETUP, not the assertion: keep the
//      dominance, then ROW-PERMUTE each item by a per-item random permutation.
//      Conditioning is unchanged, partial pivoting must undo it, and ipiv is
//      non-trivial by construction; both breaks then go red (2.4e-07 -> 1.903e+00,
//      3.4e-07 -> 1.989e+00). WARNING FOR WP6's TESTS: tests/inverse_tests.cc is
//      ONE test, float, n=40, batch=2, on Matrix<float>::Random() -- if that
//      generator is pivot-free, the only getri test in the tree cannot see a
//      pivot bug either. An "number of non-diagonal pivots" column is an
//      anti-vacuity assertion about the CONFIGURATION: necessary, NOT sufficient.
// ===========================================================================

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getrf {

// ---------------------------------------------------------------------------
// CAPABILITY. ONE number, not geqrf's (height, area) pair, because getrf's
// operand is square: the order is the only extent.
//
// It is a function of the per-work-group local-memory budget in BYTES rather than
// a constant, for route_trsm.hh:62-72's reason and for WP4's finding W1: the
// 49152 in build/include/batchlas/device_limits.hh is HARDCODED by
// cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern -- the
// detection routine never queries local_mem_size at all -- and is 2.06x wrong on
// this box, which reports sycl::info::device::local_mem_size == 101,376 B. A
// build-time constant would make supports() claim an unlaunchable route on a
// smaller device AND leave a whole band of orders with NO ROUTE in a vendor-free
// build (route_resolve.hh:113-127).
//
// WHEN THE KERNEL LANDS, THE SAME FUNCTION MUST SIZE THE LAUNCH'S local_accessor,
// so the ceiling the table advertises and the allocation the kernel makes cannot
// disagree -- the potrf_cta_launch_params defect (potrf_cta.cc:442-454), where a
// raw figure in the query against a padded one in the launcher produced an
// unhandled throw on a call the table had promised.
//
// AND IT MUST COUNT THE PIVOT-SEARCH SCRATCH. Measured
// (experiments/wp6_lu/baseline/pivotcost.cpp): an explicit SLM tree argmax needs
// wg*(sizeof(real) + sizeof(int)) ON TOP OF the tile -- 2040 B at wg 256 for
// float, 3060 B for cdouble -- and at cdouble n=78 that took the request from
// 98,608 B to 101,668 B, past this device's 101,376 B hard cap: LAUNCH FAILURE,
// "Excessive allocation of local memory on the device". The identical shape
// without the scratch fits. A capacity function that sizes only the tile
// advertises orders whose launch the device rejects.
//
// 0 IS the agreed spelling of "this tier is not in this build"
// (TrsmShape::cta_max_n's convention, pinned by
// RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable). No specialisation
// returns 0 on a device with local memory any more: on an RTX 4090 the four
// answers are 155/109/109/77 for float/double/cfloat/cdouble.
// ---------------------------------------------------------------------------
template <typename T>
int getrf_cta_max_n_for_slm(std::size_t slm_budget_bytes);

// The same question at this repository's standard budget (the runtime
// local_mem_size minus the 4096 B reserve every other device-BLAS sizing decision
// in this library applies, BatchLASDetectSYCL.cmake:57-67). Convenience over the
// honest spelling above, kept so tests can pin the four per-type numbers once they
// exist.
template <typename T>
int getrf_cta_max_n();

// Whether the BLOCKED driver exists in this build. TRUE for all four scalar
// types.
//
// IT IS DEFINED IN src/extensions/getrf_blocked.cc, NOT IN THIS HEADER AND NOT IN
// getrf_cta.cc, and that placement is load-bearing (potrf_native.hh:81-92,
// geqrf_native.hh:114-121): these are full explicit specialisations, so they link
// from wherever they sit -- and sitting anywhere but beside the driver would let a
// build advertise the tier while getrf_blocked.cc is absent from
// EXTENSIONS_CTA_SOURCES or #if 0'd out. That is the state route_trsm.hh:99-110
// names ("the table must describe the BUILD, not the design"). Co-located, "the
// flag is true" and "the file is compiled" are the same fact.
template <typename T>
bool getrf_blocked_available();

// ---------------------------------------------------------------------------
// WORKSPACE
//
// BOTH FIGURES MUST BE OBTAINED BY REPLAYING THE LAYOUT THROUGH
// BumpAllocator::measuring() (mempool.hh:186-190's workspace_bytes), NEVER by
// hand-summing an expression. The reason, from potrf_native.hh:105-113:
// mempool.hh:96-104 checks capacity as the alignment-rounded alloc_size measured
// from the UNALIGNED cursor while :111-113 advances the cursor by only
// size*sizeof(T), so an "exactly computed" figure fails the allocator's own
// capacity check; and required_bytes() (mempool.hh:45-51) additionally rounds to
// the coarsest quantum the sequence asked for, because callers re-serve the number
// through allocate<std::byte>() and would otherwise be under-provisioned by up to
// one quantum.
//
// It is also what makes the facade's max(native, vendor) safe: every term must be
// such a rounded figure. WP4_POTRF_SPEC_CORRECTIONS.md states it -- "max(a,b) is
// safe only because both terms come from required_bytes()/allocation_size; do not
// 'optimise' the layout functions into a hand-summed arithmetic expression". The
// vendor terms already satisfy it (cublas.cc:1518, :1490; netlib_lapack.cc:1331-1336
// are all BumpAllocator::allocation_size<int> terms).
//
// A ZERO ANSWER IS LEGITIMATE, not a signal. The CTA tier plausibly needs NO
// workspace at all -- its tile is local memory, the pivots are the caller's span
// and info is the caller's span or the backend's scratch -- exactly as geqrf's CTA
// tier does. That is why the facade's internal-consistency check is gated on
// `native_fired` and NOT on `native_need != 0` (factorization.cc's geqrf_buffer_size
// note): reading the check off the size would make a CTA-only build throw on every
// call the route table had just promised. orgqr_buffer_size shipped with precisely
// that latent defect and it was fixed in the WP5 repair pass.
//
// NEITHER QUERY MAY DEREFERENCE A.data_ptr() OR pivots.data(). getrf_buffer_size is
// called from INSIDE a layout function under measuring() -- src/extensions/inv.cc:36,
// reached from inv_buffer_size at :54-57 -- so per mempool.hh:180-186 it must be
// PURE WITH RESPECT TO THE WORKSPACE: no workspace read or write, no kernel launch,
// and any nested query asked about the CALLER's views. Extents, ld, stride, batch
// and is_heterogeneous() are metadata and are safe.
//
// ONE CONTRACT geqrf HAS AND getrf DOES NOT INHERIT: geqrf's monotonicity
// requirement (geqrf_native.hh:159-165) exists because band_reduction.cc sizes at
// one shape and calls at another. Nothing in the LU consumers does that --
// inv_layout sizes against the CALLER's A (inv.cc:34-36) and calls against the
// shape-identical Acopy (inv.cc:17-23). Do not import a constraint that has no
// caller behind it.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
std::size_t getrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// The BLOCKING the blocked driver WOULD use for this order, for tests.
//
// Returns the panel width nb in the low 16 bits and, in the high 16 bits, WHICH
// PANEL LEAF the LEADING panel takes: 1 = the local-memory-resident leaf (the same
// device function the CTA tier runs), 2 = the global-memory leaf. 0 for the whole
// word means the driver is absent -- which it is NOT: the blocked driver is
// linked for all four scalar types, so this returns a real (nb, leaf) pair.
//
// THIS IS NOT OPTIONAL SCAFFOLDING. potrf_native.hh:246-266 records the shape of
// the failure it prevents: a test that must straddle a block boundary cannot see
// where the boundary is, and a test that hardcodes the width keeps passing after
// any of its inputs moves while silently no longer testing a short final panel.
// This family has produced exactly that failure before (the sy2sb stage-1
// short-final-panel bug: wrong numbers, green suite).
//
// It must be a QUERY over the SAME pure function the driver and the buffer-size
// query both call, so it cannot report a blocking the call does not use -- the
// potrf_cta_launch_params discipline again.
//
// AND THE WORK-GROUP WIDTH IS NOT INHERITABLE FROM potrf OR geqrf. Measured
// (experiments/wp6_lu/baseline/wg.csv): float n=128 batch 4096, the unpivoted
// reference arm is 39.72 ms at wg=32 and 4.77 ms at wg=512 -- an 8.3x spread. Best
// wg is 256 at n=64 and 512 at n=128, and the pivoted and unpivoted arms do not
// always prefer the same width. WP6 must tune wg per (type, n), and whatever it
// picks must be visible to a test through this query or a sibling of it.
template <typename T>
unsigned getrf_blocked_debug_params(Queue& ctx, int n);

// ---------------------------------------------------------------------------
// THE TWO ROUTED OPERATIONS THE BLOCKED DRIVER INJECTS, rather than hardcodes.
//
// WHY, and this is a RECORDED DEFECT rather than a style preference: WP3's V2
// called sycl_gemm::gemm_custom directly, which is the NATIVE kernel entry point
// and bypasses RouteTable<Op::gemm> entirely, so the trailing updates always got
// the native kernel whether or not it was the better one (WP3 step 16;
// trsm_native.hh:82-104 carries the measurement, level3.cc:186-231 the fix).
//
// It is ALSO the only way to reach gemm/trsm from these TUs at all: the drivers
// are instantiated per scalar type with NO Backend parameter, and gemm<B,T> /
// trsm<B,T> need one. Only the facade layer can name a routed entry point.
//
// The signatures are deliberately identical to sycl_gemm::gemm_custom and to the
// routed batchlas::gemm / batchlas::trsm, so neither side adapts. An EMPTY
// function means "use gemm_custom" for the gemm seam, which keeps this kernel
// layer free of the dispatch layer: a direct caller (the tests, and any benchmark
// that must not be silently served by a vendor) gets the native kernel with no
// dispatch dependency, and a vendor-free build is unaffected because the resolver
// falls back to native there anyway.
//
// NEITHER SEAM CARRIES A BUFFER-SIZE TWIN, and that is a real difference from
// WP5's orgqr (OrgqrApplyQ / OrgqrApplyQBufferSize, orgqr_native.hh:118 and :137,
// injected together at factorization.cc:349-363 and :431-440 so call and query
// cannot resolve differently). The public gemm and trsm entry points take NO
// workspace at all, so there is no size to keep in step -- verified against
// functions/gemm.hh and functions/trsm.hh, and consistent with the measured
// baseline's "the routed trsm allocates nothing".
//
// A MEASURED FACT ABOUT WHAT THE GEMM SEAM WILL CARRY, and it INVERTS the
// prediction WP6 inherited. The brief predicted complex would be stuck on the
// Tiled16 fallback; measured at the REAL batch and stride, for the LU trailing
// update (NN, k = nb = 32 or 64), vendor-free:
//     float             -> Tiled128x128RegisterK8 at m,n >= 128
//     cfloat, cdouble   -> Tiled64x64RegisterK16Wide at every cell but the
//                          N=2048 tail panel
//     double            -> Tiled16 at ALL 13 shapes
// DOUBLE is the type with no register kernel on this path, and it is STRUCTURAL:
// the wide-scalar CTA-count relaxation is `if constexpr (is_std_complex_v<T>)`,
// complex only, and the only other wide-scalar door (gemm_kernels.cc:642) needs
// min_dim >= 256, which k = nb can never satisfy. The deficit is bounded
// (gemm_kernels.cc:606-616 measures double at 1.01-1.08x of Tiled16, itself ~92%
// of the 4090's FP64 ceiling) but there is no WP6-local fix: it needs a
// transposed/predicated wide-scalar kernel and belongs to GEMM. Record it; do not
// fix GEMM inside WP6.
//
// (The harness artefact that produced the WRONG version of this answer is worth
// knowing: a first probe used batch=1 parent matrices on the reasoning that
// select_kernel_variant reads only m/n/k and the transposes. It reported Tiled16
// for every complex trailing update -- exactly the predicted answer -- and it was
// wrong, because gemm_kernels.cc:695-707's CTA-count gate multiplies by
// A.batch_size() and can_use_64x64_k16_wide_fast_path reads data_ptr(), ld() AND
// stride(). A harness that saves memory by shrinking the batch cannot ask this
// question.)
// ---------------------------------------------------------------------------
template <typename T>
using GetrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// The panel solve L11 \ A12 that turns the factored diagonal block into the row
// panel of U. Signature identical to the routed batchlas::trsm's positional form
// (functions/trsm.hh:100-108) -- note alpha comes THIRD, not last; the old
// spelling is a deleted overload at :121-138 precisely so a stale call cannot
// silently compile.
template <typename T>
using GetrfPanelSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

// ---------------------------------------------------------------------------
// DIRECT-CALL ENTRY POINTS.
//
// Exposed so tests can exercise a kernel without going through dispatch -- which
// is NOT convenience but a correctness requirement (potrf_native.hh:126-141, and
// tests/potrf_tests.cc:6-18 states it in full): route_resolve.hh:165 tests
// `if (Table::supports(forced, s)) return forced;` and falls through to
// automatic() at :175, so a test that sets BATCHLAS_GETRF_ROUTE=cta and gets one
// gate wrong runs cuBLAS and passes GREEN over a kernel nothing executed. A direct
// call cannot be served by a vendor.
//
// Each must re-check every RouteTable<Op::getrf,T>::supports() gate and throw,
// because these are reachable WITHOUT the table. Today they throw unconditionally.
//
// AND WHEN THE KERNEL LANDS, THE FACADE TEST MUST BE BIT-EXACT AGAINST THIS CALL,
// not a residual. tests/potrf_tests.cc:895-908 records this repository's fifth
// blind guard: a route-assertion-plus-residual test "stayed GREEN across all four
// scalar types while every number in it came from cuSOLVER", because a residual
// bound is satisfied by either implementation. For getrf the discriminating oracle
// is stronger than a residual in a second way -- the ELEMENTWISE PIVOT SEQUENCE
// against an independent host xGETRF, which is the only thing that catches a
// valid-but-different pivot choice. Every residual test passes such a choice.
// ---------------------------------------------------------------------------
template <typename T>
Event getrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<int64_t> pivots,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

template <typename T>
Event getrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             GetrfTrailingGemm<T> trailing_gemm = {},
                             GetrfPanelSolveTrsm<T> panel_trsm = {});

// Does an order-n matrix fit the resident leaf at this local-memory budget? The
// SAME predicate the launcher applies and the route table's capacity is derived
// from, exposed so those three cannot answer it differently. geqrf_cta.cc:343-352
// is the shape to copy, and its three-place application (the local_accessor
// allocation, the fits predicate, the capacity query) is what the 48 KB launch
// hole makes mandatory -- see the note below.
//
// THE 48 KB LAUNCH HOLE, AND WHY IT IS A LIVE RISK FOR getrf SPECIFICALLY. A
// partial-pivoting LU does a WORK-GROUP REDUCTION PER COLUMN to find the pivot,
// which is verbatim the condition WP4 recorded as reopening the defect
// (potrf_cta.cc:285-290: "a reduce_over_group ... reintroduces the hole"), and
// WP5 walked into it anyway because geqr2_panel_device has two per reflector.
//
// MEASURED FOR WP6, with a PAD= knob holding kernel, shape and work-group fixed
// and moving ONLY the declared byte count, one process per point:
//   * The hole is SPECIFIC BYTE COUNTS, not a range -- 49,024 B launches, 49,152 B
//     FAILS ("unknown internal error"), 49,280 B launches. An n ladder finds
//     nothing because it steps over 49,152.
//   * 5/5 deterministic across five separate processes, and independent of
//     work-group width (32/64/128/256/512 all fail).
//   * IT IS ATTRIBUTABLE TO sycl::reduce_over_group ALONE. At the IDENTICAL byte
//     count an explicit SLM tree argmax launches fine, as does the unpivoted arm.
//   * THE BAND IS WIDER FOR WIDE SCALARS: the collective also fails at 48,896 B
//     for double and cdouble but not for float or cfloat. The mechanism is that
//     the collective allocates local memory the local_accessor accounting cannot
//     see, sizeof(T)-dependent.
//
// RECOMMENDATION, and it is the same answer on speed: USE AN EXPLICIT SLM TREE
// ARGMAX. It is 1.5-4.7x FASTER than reduce_over_group for double and cdouble
// (double n=16 is 7.07x the unpivoted bound with the collective against pivman's
// 2.00x), a wash for float/cfloat (0.87-1.25x), and it sidesteps the hole
// entirely. If a group collective is ever wanted anyway, potrf_cta.cc:258-296's
// band-and-pad is MANDATORY in all three places that must agree, and the band must
// be WIDENED for 8- and 16-byte scalars. geqrf's constants are kHoleLo = 47104,
// kHoleHi = 49664, kHolePadTo = 49920 (geqrf_cta.cc:115-137); they do not cover
// 48,896 for double.
//
// A DISCRIMINATING TEST FOR THIS MUST BE DECLARED FIRST IN ITS FILE, with a
// comment saying why: the SLM attribute is STICKY PER CUfunction, so any earlier,
// larger launch in the same process raises the cap and hides the entire class BY
// EXECUTION ORDER.
template <typename T>
bool getrf_cta_fits(int n, std::size_t slm_budget_bytes);

// The same predicate for a RECTANGULAR panel, which is the shape the blocked
// driver's leaf is handed: (m - j0) x ib. getrf_cta_fits(n, b) IS
// getrf_leaf_fits(n, n, b) -- one function, so the square tier's ceiling and the
// blocked tier's per-panel residency choice cannot answer differently.
template <typename T>
bool getrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes);

// ---------------------------------------------------------------------------
// THE PANEL LEAF ITSELF -- the ONE decision site between the two residencies,
// and the symbol that forces getrf_cta.cc and getrf_blocked.cc into the SAME
// device-code cluster (src/extensions/CMakeLists.txt:29-42). geqrf_panel_
// factorize is the precedent, and it exists for the same reason: the blocked
// driver must not re-derive "does this panel fit local memory".
//
// `piv_ptr` is the PACKED int32 pivot buffer (see the PIVOT CONTRACT above),
// `piv_stride` its per-item stride (the ORDER of the whole matrix, never the
// panel width), and `piv_base` the panel's first global row index -- which is
// what makes the values LAPACK's GLOBAL 1-based ipiv and info the GLOBAL column
// index without a fix-up pass.
//
// `info_ptr` is READ as well as written (first-failure-wins across panels), so
// the caller must zero it before the first panel.
// ---------------------------------------------------------------------------
template <typename T>
Event getrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            int* piv_ptr, int piv_stride, int piv_base,
                            int32_t* info_ptr,
                            bool* used_resident_out);

// ---------------------------------------------------------------------------
// WHAT THE PIVOTING WILL COST, measured before any kernel exists so that a
// disappointing first number can be recognised as expected rather than as a bug.
//
// A standalone CTA-resident LU probe (experiments/wp6_lu/baseline/pivotcost.cpp),
// whole matrix in local memory at ld = n|1, wg = 256, batch 4096, four arms:
// nopiv (Doolittle, no search, no swap -- the LOWER BOUND), swaponly, an explicit
// SLM tree argmax, and reduce_over_group. Ratios against the unpivoted bound:
//
//     float  n=16  swap 1.20  tree 2.65  collective 3.30
//     float  n=64  swap 1.08  tree 1.52  collective 1.72
//     float  n=128 swap 1.02  tree 1.35  collective 1.38
//     double n=32  swap 1.04  tree 1.69  collective 4.25
//     double n=110 swap 1.03  tree 1.35  collective 1.49
//     cdouble n=64 swap 1.04  tree 1.32  collective 1.46
//
// THE SWAP IS NEARLY FREE (1.00-1.20x); THE SEARCH IS THE WHOLE COST, and it gets
// CHEAPER as n grows. Effort spent making the row exchange clever is spent on the
// 3% end. The ratios are FLAT IN BATCH (float n=64: 1.85, 1.72, 1.53, 1.57, 1.53,
// 1.52, 1.52, 1.52 at batch 128..16384), so this is a per-matrix property and not
// an occupancy artefact.
//
// AND THE UNPIVOTED ARM IS NOT A USABLE ALGORITHM, only a timing reference: on the
// row-permuted test matrix its residual is 1.5e-03 (float n=64) against the
// pivoted arms' 4.6e-07.
// ---------------------------------------------------------------------------

}  // namespace batchlas::sycl_getrf
