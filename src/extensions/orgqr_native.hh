#pragma once

// Native batched ORGQR -- declarations.
//
// WP5. The driver is IMPLEMENTED (src/extensions/orgqr_blocked.cc) and
// orgqr_blocked_available<T>() is true for all four scalar types. It is
// ROUTE-NEUTRAL: RouteTable<Op::orgqr,T>::preferred() is still false everywhere,
// so a vendor-present build keeps taking cuSOLVER's per-item loop and this
// driver is reached only by a vendor-free build, by BATCHLAS_ORGQR_ROUTE, or by
// the direct entry point below. See geqrf_native.hh for why the interface landed
// before the body.
//
// ONE NATIVE TIER: Algorithm::Blocked, and it is ORMQR APPLIED TO AN IDENTITY.
// route_orgqr.hh carries the measurement that settles that design (correctness
// against cuSOLVER to 1.4e-15 / 6.9e-07 across 24 cells; 2.3-111x over a vendor
// that is a per-batch-item loop; and a 11x memory reduction because
// cublas.cc:1447 sizes the vendor workspace as single_ws * batch). Do not
// specialise before a cell measures it necessary -- the entire theoretical prize
// is the 1.5x flop ratio of applying Q to I (2n^3 against 4n^3/3).
//
// WHERE THE TU LIVES, AND WHY IT IS NOT WITH geqrf's. geqrf_cta.cc and
// geqrf_blocked.cc share a device function and therefore share a device-code
// cluster (EXTENSIONS_CTA_SOURCES; src/extensions/CMakeLists.txt:53-57's rule,
// W12). orgqr_blocked.cc shares device symbols with NEITHER -- its work is an
// identity fill plus a call back out through the injected apply-Q seam -- and it
// is built on ormqr_blocked, so it sits in EXTENSIONS_FACTORIZATION_SOURCES
// beside ormqr_blocked.cc. That mirrors the existing ormqr split
// (ormqr_cta.cc in the CTA list, ormqr_blocked.cc in the factorization list),
// which works precisely because those two share no device symbol. A wrong
// grouping is a hard `ptxas fatal: Unresolved extern function`, never a silent
// miscompile, so this decision fails the build immediately if it is wrong.
//
// INSTANTIATION SHAPE: per scalar type, NO Backend parameter -- potrf's shape
// (potrf_cta.cc:706-726, "no Backend cross-product ... pure cost" in a build that
// is device-link-bound). ormqr_blocked does the opposite
// (include/batchlas/internal/ormqr_blocked.hh:23-39 is template <Backend B,
// typename T>, a 4x4 cross-product), and building orgqr on it the naive way would
// inherit that. The apply-Q seam below is what avoids it, and it is the same
// device that lets potrf's type-only driver reach the ROUTED gemm and trsm.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_orgqr {

// Whether the native orgqr driver exists in this build. TRUE for all four
// scalar types as of WP5.
//
// DEFINED IN src/extensions/orgqr_blocked.cc, not here -- potrf_native.hh:81-92's
// placement rule: co-located, "the flag is true" and "the file is compiled" are
// the same fact, and no build can advertise a tier whose TU is missing from the
// CMake source list.
//
// It is NOT "is ormqr_blocked compiled". That is true already, and answering this
// question with it would make RouteTable<Op::orgqr,T> hand a vendor-free caller a
// route the facade cannot service.
template <typename T>
bool orgqr_blocked_available();


// The block width the driver WOULD use, for tests. 0 only for a degenerate
// extent.
//
// Same reason as geqrf_blocked_debug_params (potrf_native.hh:246-266): a test
// that must straddle a block boundary cannot see where the boundary is, and one
// that hardcodes the width keeps passing after the width moves.
//
// MEASURED, and it is NOT tuning::ormqr_block_size_for_n: that ladder
// (16/16/24/48/56 by A.rows(), ormqr.hh:220-228) was tuned on CUDA/float only and
// costs double 1.32-1.41x and cdouble 1.46-1.47x even in a vendor-present build.
// The measured best is 32 for float/cfloat/cdouble and 16 for double, and the
// width must stay a multiple of 16 and at least 32 for complex
// (gemm_kernels.cc:700's min_dim >= 32 wide-scalar gate). Full tables in
// docs/perf/qr.md#block-width-evidence.
template <typename T>
int orgqr_blocked_debug_block_size(Queue& ctx, int m, int n);

// ---------------------------------------------------------------------------
// THE APPLY-Q SEAM, INJECTED.
//
// The signature is the ROUTED batchlas::ormqr's positional entry point verbatim
// (include/batchlas/blas/functions/ormqr.hh:311-320), so neither side adapts:
//
//     ormqr(ctx, A, C, side, trans, tau, workspace, block_size_hint)
//
// ARGUMENT ORDER IS THE POSITIONAL ENTRY POINT'S, and that is worth stating:
// WP4's finding W13 records a compile error from copying an option struct's field
// order into an injected seam (TrsmOptions puts alpha first, options.hh:257-264,
// while functions/trsm.hh:100-108 puts it in position 4).
//
// AN EMPTY FUNCTION THROWS. The WP5 scaffolding said it would mean "use
// ormqr_blocked directly" -- the potrf/trsm convention, where an absent seam
// falls back to the native kernel and keeps the TU free of the dispatch layer.
// IT CANNOT MEAN THAT HERE, and the reason is the same one that makes the seam
// necessary in the first place: ormqr_blocked is
// template <Backend B, typename T> (internal/ormqr_blocked.hh:23-39) and this
// driver is instantiated per scalar type with NO Backend, so a "native fallback"
// would have to hardcode one -- Backend::CUDA, in a file that also builds for
// ROCm. Throwing names the requirement instead of guessing; a direct caller (a
// test) injects ormqr<Backend::CUDA, T> itself, which is still a call no vendor
// orgqr can serve. INJECTION IS THE POINT anyway: the
// facade passes a lambda calling ormqr<B,T>, so the apply goes through
// RouteTable<Op::ormqr> and honours BATCHLAS_ORMQR_ROUTE. Calling a native kernel
// entry point straight from a driver TU is the RECORDED DEFECT of WP3 step 16
// (trsm_native.hh:82-104): it bypasses the router and takes the native kernel
// even on shapes measured to lose.
//
// It is ALSO the only way to reach ormqr from here at all: batchlas::ormqr<B,T>
// needs a Backend and this family is instantiated per scalar type with none.
// ---------------------------------------------------------------------------
template <typename T>
using OrgqrApplyQ = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,   // A: the geqrf output, reflectors
    const MatrixView<T, MatrixFormat::Dense>&,   // C: the identity, overwritten by Q
    Side, Transpose,
    Span<T>,                                     // tau
    Span<std::byte>,                             // workspace
    int32_t)>;                                   // block_size_hint

// The apply-Q workspace query, injected for the same reason: the size and the
// call must come from the SAME resolution, and only the facade can ask the
// router. Mirrors batchlas::ormqr_buffer_size's positional signature.
//
// NOTE THE ANTI-PATTERN THIS AVOIDS, which is still in the tree:
// ormqr_buffer_size_dispatch (ormqr.hh:281-303) re-resolves the route at :295 and
// returns ONLY the chosen route's size at :298-302. The two agree today only
// because both reads of getenv return the same thing -- exactly the assumption
// factorization.cc:315-324 says may not be made. Do not copy it.
template <typename T>
using OrgqrApplyQBufferSize = std::function<std::size_t(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    Side, Transpose,
    Span<T>,
    int32_t)>;

// Workspace the native route needs, in bytes.
//
// It comes from replaying the driver's layout through BumpAllocator::measuring()
// (mempool.hh:185-190), never hand-summed -- see the same note in
// geqrf_native.hh, and potrf_native.hh:105-113 for the mechanism. The layout has
// two terms: the m x n identity the apply writes into, and whatever the apply
// itself demands for the chosen block width.
//
// IT TAKES tau AND THE INJECTED SIZE QUERY, WHICH THE WP5 SCAFFOLDING'S
// TWO-ARGUMENT FORM COULD NOT. The scaffolding asked that the apply's workspace
// "be computed from ONE pure function the driver also calls"; with only (Queue&,
// A) available that function would have to be a local copy of
// ormqr_blocked_buffer_size_impl's four allocation_size terms, which is the
// drift the request exists to prevent -- and it would be WRONG outright whenever
// the apply resolves to a vendor ormqr, whose workspace has a different formula
// entirely. So the ROUTED ormqr_buffer_size is injected into BOTH this query and
// the call, from the same resolution: they agree by construction rather than
// because getenv returned the same thing twice, which is exactly what
// ormqr_buffer_size_dispatch (ormqr.hh:281-303) relies on and must not be
// copied.
//
// `tau` is read for its SIZE only -- the routed ormqr_buffer_size validates
// tau.size() >= k * batch. NEITHER THIS QUERY NOR THE INJECTED ONE MAY
// DEREFERENCE A.data_ptr() OR tau.data(): orgqr is not sized against a null view
// anywhere in this tree today the way geqrf is (band_reduction.cc:1041-1044),
// but the facade's max(native, vendor) puts this on the same code path as
// geqrf's query and the rule costs nothing to keep. The C view this query hands
// the injected function is built over a NULLPTR for the same reason.
//
// An ABSENT injection throws rather than guessing. See the note on OrgqrApplyQ
// below for why there is no defaulted native fallback.
template <typename T>
std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau,
                                      OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

// DIRECT-CALL ENTRY POINT. Same reason as geqrf's and potrf's
// (potrf_native.hh:126-141, tests/potrf_tests.cc:6-18): a forced route that
// supports() rejects falls through to automatic() (route_resolve.hh:101, :111)
// and silently becomes the vendor, so a pinned-route test that is wrong about one
// gate passes green over cuSOLVER's numbers. A direct call cannot be served by a
// vendor.
//
// Re-checks every RouteTable<Op::orgqr,T>::supports() gate and throws, because it
// is reachable without the table. Today it throws unconditionally.
template <typename T>
Event orgqr_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             OrgqrApplyQ<T> apply_q = {},
                             OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

}  // namespace batchlas::sycl_orgqr
