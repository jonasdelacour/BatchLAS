#pragma once

// Native batched GETRS -- declarations.
//
// STATUS: LIVE. getrs_blocked_available<T>() is true for every scalar type and
// getrs_blocked_dispatch is the driver in src/extensions/getrs_native.cc, so
// RouteTable<Op::getrs,T>::supports() admits the native arm for every square GPU
// shape and a vendor-free build takes it, at all three transA values. A
// vendor-PRESENT build is unchanged because preferred() is all-false. NOTE that
// the native arm is a CROSSOVER ON nrhs rather than a flat loss: geomean 0.32x at
// nrhs=1 rising monotonically to 1.36x at nrhs=128
// (experiments/wp6_lu/bench/README.md).
//
// THE SHAPE OF THE OP. getrs against a factored A = P L U is:
//     NoTrans            : apply P to B, solve L y = Pb (unit lower), solve U x = y
//     Trans / ConjTrans  : solve U^T/U^H y = b, solve L^T/L^H z = y, then apply
//                          P^T to z -- the two solves SWAP ORDER and the
//                          permutation moves to the OUTPUT, in REVERSE.
// Nothing in BatchLAS interprets transA today: enum_convert hands it straight to
// the vendor (cublas.cc:1478, rocsolver.cc:274-287, netlib_lapack.cc:1247-1249).
// The vendor is correct in all three modes, measured against a host LAPACKE oracle
// (residual max|op(A)X - B| at n=6 batch=3 nrhs=2: NoTrans 1.19e-07 / 6.66e-16 /
// 2.39e-07 / 8.88e-16; Trans 2.38e-07 / 3.33e-16 / 3.58e-07 / 8.88e-16; ConjTrans
// 3.58e-07 / 8.88e-16). So a native getrs has an EXACT ORACLE to hit, and the
// reverse-permutation trap is entirely on the native side.
//
// THIS FILE IS THE SMALL ONE. Both triangular solves are the ROUTED trsm,
// injected from the facade; the only kernel getrs owns is the row permutation.
// That is why getrs_native.cc sits in EXTENSIONS_FACTORIZATION_SOURCES and not in
// EXTENSIONS_CTA_SOURCES: it shares no device symbol with the getrf pair
// (src/extensions/CMakeLists.txt:77-86's cluster rule, and orgqr_blocked.cc's
// precedent at :62-66). If a future getrs ever calls the getrf CTA device body it
// MOVES, and the failure mode of getting that wrong is a hard
// `ptxas fatal: Unresolved extern function`, never a silent miscompile.
//
// ===========================================================================
// THE MEASUREMENT THAT MUST GOVERN THIS FILE, AND IT IS A NEGATIVE RESULT.
//
// Composed "row permutation + two routed trsm" against cublas?getrsBatched, at
// saturating batch, in process, against a host oracle
// (experiments/wp6_lu/baseline/grid_norm.csv, summary.txt):
//
//   nrhs = 1  : GEOMEAN 0.36x over 28 cells (4 types x 7 orders). 25 LOSSES.
//               n(batch)   float double cfloat cdouble
//                32(8192)   0.20   0.19   0.10   0.09
//               128(4096)   0.41   0.23   0.34   0.14
//               512(512)    0.66   0.32   0.59   0.26
//              2048(32)     0.94   1.14   0.87   1.07
//               The only wins are at n=2048, and that is against a vendor which
//               is NOT SATURATED there (64x the work for 1.52x the time from
//               batch 1 to 64).
//
//   nrhs = 64 : geomean 1.17x with a LAPACK-faithful interchange walk, 1.55x with
//               the permutation collapsed to a gather. 20 and 25 wins of 28.
//
// THE nrhs=1 LOSS IS STRUCTURAL, NOT A BAD KERNEL, and the permutation strategy is
// irrelevant to it (0.36x either way): at one right-hand side the permutation is a
// rounding error and the whole loss is in the triangular solves, because trsm's
// blocked driver amortises a panel over many columns and one column gives it
// nothing to amortise. So a native getrs needs EITHER a separate narrow-RHS path
// OR it ships route-neutral at small nrhs -- which is a legitimate outcome under
// the campaign's gate ("a native kernel that is correct but slower than the vendor
// ships route-neutral, it does not become the default"). Engineering around the
// number is not.
//
// nrhs IS AVAILABLE TO preferred() as GetrsShape::nrhs() (== s.n), which is
// precisely why route_getrs.hh maps it there rather than folding it into max_dim.
//
// AND getrs HAS NO INTERNAL CONSUMER AT ALL. src/extensions/inv.cc:48-49 calls
// getrf then getri; the public linalg layer calls getrs only through
// linalg::solve (include/batchlas/blas/linalg-ops.hh:343-344). Nothing in the
// library depends on this op being fast.
// ===========================================================================
//
// THE PERMUTATION IS HALF THE COMPOSED CALL, AND IT CAN BE COLLAPSED -- but not
// for free, which is where getrs differs from getri. Measured by an accidental
// break (BREAK=laswp, float n=128 nrhs=128 batch=256): getrs_trsm 0.4456 ->
// 0.2252 ms without the row exchange, i.e. 49% of the call. The cause is
// structural: LAPACK's ipiv is a SEQUENCE of interchanges so it must be walked in
// order, one work-item per column, and in column-major that puts consecutive
// work-items ldb apart -- 32 transactions per warp access. Applying the
// interchanges to an identity index array once and then GATHERING puts consecutive
// work-items on consecutive ROWS, which is contiguous. getri gets that for nothing
// (it writes P straight into C); getrs's gather needs an OUT-OF-PLACE RHS or an
// in-place cycle walk, and the out-of-place buffer is real workspace: 67,371,008 B
// at n=2048, nrhs=64, batch=32, against the vendor's 0 B. That trade is the design
// decision this file exists to make explicitly rather than by accident.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getrs {

// ---------------------------------------------------------------------------
// CAPABILITY. ONE flag, because getrs has ONE native arm -- there is no capacity
// to advertise: the two solves are the routed trsm, whose own tiers carry no upper
// bound on the order, and the permutation kernel is a strided copy with no
// resident tile.
//
// DEFINED IN src/extensions/getrs_native.cc, beside the driver, for
// potrf_native.hh:81-92's reason: these are full explicit specialisations and link
// from wherever they sit, so co-locating them is what makes "the flag is true" and
// "the file is compiled" the same fact. TRUE for all four types.
//
// 0/false is the agreed spelling of "this arm is not in this build"
// (TrsmShape::cta_max_n's convention, pinned by
// RouteTrsm.AbsentKernelIsUnsupportedRatherThanSelectable).
// ---------------------------------------------------------------------------
template <typename T>
bool getrs_blocked_available();

// ---------------------------------------------------------------------------
// WORKSPACE, in bytes, for the native arm.
//
// It must come from replaying the layout through BumpAllocator::measuring()
// (mempool.hh:186-190's workspace_bytes), NEVER hand-summed -- mempool.hh:96-104
// checks capacity as the alignment-rounded alloc_size from the UNALIGNED cursor
// while :111-113 advances by only the raw data extent, so an exactly-computed
// figure fails the allocator's own capacity check; and required_bytes()
// (mempool.hh:45-51) rounds to the coarsest quantum the sequence asked for,
// because callers re-serve the number through allocate<std::byte>(). It is also
// what makes the facade's max(native, vendor) safe: every term must be such a
// rounded figure.
//
// A ZERO ANSWER IS LEGITIMATE. An in-place interchange walk needs nothing at all;
// only the gather strategy needs the out-of-place RHS and the collapsed
// permutation. That is why the facade's consistency check is gated on
// `native_fired` and not on `native_need != 0`.
//
// IT MUST NOT DEREFERENCE ANY POINTER. Extents, ld, stride, batch and
// is_heterogeneous() are metadata and are safe; a data read is a segfault in a
// sizing path.
//
// IT TAKES B AND transA, not just A, and both are load-bearing: the workspace
// scales with nrhs (B.cols()) under the gather strategy, and transA decides
// whether the permutation is applied to the input or to the output -- which for an
// out-of-place gather is a different buffer. Sizing off A alone would be right only
// for the in-place walk, i.e. only for the strategy the measurement argues
// against.
// ---------------------------------------------------------------------------
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      Transpose transA);

// ---------------------------------------------------------------------------
// THE TRIANGULAR SOLVE, INJECTED rather than hardcoded.
//
// Both solves of a getrs go through the ROUTED trsm. Calling a native kernel
// entry point straight from a driver TU is the RECORDED DEFECT of WP3 step 16
// (trsm_native.hh:82-104, fix at level3.cc:186-231): it bypasses
// RouteTable<Op::trsm> entirely, so the driver would get the native kernel even on
// shapes measured to lose.
//
// It is ALSO the only way to reach trsm from here at all: this driver is
// instantiated per scalar type with NO Backend parameter, and trsm<B,T> needs one.
// Only the facade layer can name a routed entry point.
//
// The signature is the routed batchlas::trsm's positional form verbatim
// (functions/trsm.hh:100-108). NOTE THAT alpha COMES THIRD, not last: the old
// spelling is a DELETED overload at :121-138 precisely so a stale call cannot
// silently compile into a wrong answer.
//
// NO BUFFER-SIZE TWIN, unlike WP5's OrgqrApplyQ / OrgqrApplyQBufferSize pair
// (orgqr_native.hh:118, :137). The public trsm takes no workspace at all, so there
// is no size for the query and the call to disagree about -- which is the entire
// hazard that pair exists to close.
//
// AN ABSENT INJECTION THROWS rather than silently reaching for a native kernel.
// getrs's solves are the whole op; a driver that quietly bypassed the router here
// would be the WP3-step-16 defect re-created, and a direct caller (a test) injects
// trsm<Backend::CUDA, T> itself, which is still a call no vendor getrs can serve.
// ---------------------------------------------------------------------------
template <typename T>
using GetrsSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

// ---------------------------------------------------------------------------
// DIRECT-CALL ENTRY POINT.
//
// Exposed so tests can exercise the driver without going through dispatch, which
// is a correctness requirement and not a convenience (potrf_native.hh:126-141,
// tests/potrf_tests.cc:6-18): route_resolve.hh:165 gates the forced route on
// supports() and falls through to automatic() at :175, so a test that sets
// BATCHLAS_GETRS_ROUTE and gets one gate wrong runs cuBLAS and passes GREEN over a
// driver nothing executed.
//
// It must re-check every RouteTable<Op::getrs,T>::supports() gate and throw,
// because it is reachable WITHOUT the table. Today it throws unconditionally.
//
// AND ITS ORACLE IS THE HOST, NEVER THE VENDOR. A vendor reference is INERT in a
// vendor-free build -- the resolver falls back to the code under test and the test
// compares the kernel with itself (tests/geqrf_tests.cc:8-12). For getrs the host
// oracle is ||A X - B|| / ||A|| ||X|| in double regardless of T, on a matrix that
// has been ROW-PERMUTED so the pivot list is non-trivial by construction: on the
// obvious diagonally dominant matrix, dropping the row interchange entirely leaves
// the residual BIT-IDENTICAL (measured -- see getrf_native.hh's note 5).
// ---------------------------------------------------------------------------
template <typename T>
Event getrs_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             GetrsSolveTrsm<T> solve_trsm = {});

// ---------------------------------------------------------------------------
// THE PIVOT FORMAT IS getrf's, AND IT IS BACKEND-DEPENDENT. A native getrs reads
// the buffer a getrf wrote, and that buffer holds PACKED 1-BASED INT32 on CUDA and
// ROCm (cublas.cc:1476 and rocsolver.cc:227 both do pivots.as_span<int>()) and
// GENUINE 1-BASED INT64 on NETLIB (netlib_lapack.cc:1234-1241). The values are an
// INTERCHANGE LIST, not a permutation vector. The full measurement and the contract
// WP6 adopts are in getrf_native.hh's PIVOT CONTRACT section; do not re-derive them
// here, and do not let getrs pick a different convention from getrf -- the two ops
// have independent env variables and independent preferred() windows, so every
// mixture of native and vendor arms is reachable.
//
// AND THERE IS DEVICE-SIDE PIVOTING PRECEDENT IN THIS TREE: src/extensions/stein.cc
// implements tridiagonal LU with partial pivoting following LAPACK dgttrf (:42-48,
// :177-249), including a pivot floor built from ||T||_inf. Read it before inventing
// a policy.
// ---------------------------------------------------------------------------

}  // namespace batchlas::sycl_getrs
