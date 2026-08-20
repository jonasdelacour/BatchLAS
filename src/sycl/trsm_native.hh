#pragma once

// Native batched TRSM — declarations.
//
// See WP3_TRSM_SPEC_CORRECTIONS.md first, then WP3_TRSM_SPEC.md. The spec
// predates WP1 and WP2 and is stale in several places the corrections list;
// where they disagree the corrections win.
//
// TWO KERNELS, ONE RECURRENCE.
//
//   V1 (Algorithm::CTA)     — one work-group per matrix, one thread per
//                             INDEPENDENT SOLVE, the solution vector resident
//                             in that thread's registers as `T x[N]`, the
//                             triangle staged once into local memory and
//                             broadcast. Serves triangular order n <= the
//                             register capacity.
//   V2 (Algorithm::Blocked) — a host-blocked driver for larger n that calls V1
//                             as its diagonal-block solver and
//                             sycl_gemm::gemm_custom for the trailing update.
//                             V1 is literally V2's panel solve, so the
//                             crossover is a capacity, not a tuned guess.
//
// WHAT IS DELIBERATELY NOT HERE: diagonal-block inversion, at any tier. The
// spec rejects it (§2.4) and that rejection survived the verification pass. The
// short version: the "it is free for ortho" licence compares a trsm RESIDUAL
// bound against a CholQR ORTHOGONALITY bound, and pushed through the consumer
// the inverted variant contributes at the SAME order as the existing term with
// an unbounded constant. potrf is allowed to succeed to where that term is
// already O(1), so any constant above ~2 flips Chol2 from recovering to not.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <functional>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_trsm {

// The largest triangular order V1 can hold, per scalar type.
//
// ZERO MEANS "NO NATIVE TRSM KERNEL IN THIS BUILD", and that is the value every
// specialisation returns until the kernel itself lands. RouteTable<Op::trsm,T>
// reads it through TrsmShape::cta_max_n and reports BOTH native routes
// unsupported when it is zero, so a build in this state behaves exactly as it
// did before WP3 began — the vendor serves every call, and a vendor-free build
// throws the same NoRouteError it threw yesterday.
//
// WHY IT IS A FUNCTION AND NOT A constexpr LITERAL. route_gesvd.hh:64-71 makes
// the analogous per-type caps compile-time constants, but its four numbers come
// from a measured local-memory limit. TRSM's do not: the spec's
// {float 64, double 32, cfloat 32, cdouble 16} derive from a "256 B/thread
// register cliff" that src/sycl/gemm_kernels.cc:725-735 records as measured
// FALSE (an 8x8 double tile compiles to 208 registers and complex<float> to 247,
// both spill-free; the real wall is the 65,536-registers-per-BLOCK limit).
// Writing those four numbers into a header would launder four hypotheses into a
// compile-time constant. They get set here, in the same translation unit as the
// kernel, once scripts/register_probe.sh has measured the instantiations.
template <typename T>
int trsm_cta_max_n();

// Whether V2, the blocked driver for orders above trsm_cta_max_n<T>(), exists.
// False until it is written. Kept symmetric with the capacity above so that
// "the kernel is not in this build" is expressed the same way for both tiers
// and cannot be forgotten when one of them lands.
template <typename T>
bool trsm_blocked_available();

// V1, both sides. Direct-call entry: nothing in the library routes here yet.
// Exposed so tests can exercise the kernel before it is reachable through
// dispatch, which is what lets the register gate be answered before any routing
// decision depends on it.
template <typename T>
Event trsm_native_v1_dispatch(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& A,
                              const MatrixView<T, MatrixFormat::Dense>& B,
                              T alpha,
                              Side side,
                              Uplo uplo,
                              Transpose transA,
                              Diag diag);

// The trailing-update GEMM, injected rather than hardcoded.
//
// WHY. V2 called sycl_gemm::gemm_custom directly, which is the NATIVE kernel
// entry point and bypasses RouteTable<Op::gemm> entirely -- so the trailing
// updates always got the native kernel whether or not it was the better one.
// Measured, float, the six shapes V2 actually issues at order 512 (q=1024,
// batch=512), with the leading dimensions it actually passes (sub-views
// carrying the parent's ld=512, not ld==rows):
//
//   outer  m=128 n=1024 k=128/256/384   native 8.05 ms   vendor 3.89 ms
//   inner  m=32  n=1024 k=32/64/96      native 7.89 ms   vendor 3.98 ms  (x4 panels)
//
// i.e. cuBLAS is 1.6-2.3x faster on every one of them. The native GEMM is at
// parity when ld == rows and collapses when it is not; cuBLAS barely notices.
// Since a panel is ALWAYS a sub-view, the strided case is the only one trsm
// ever issues.
//
// The signature is deliberately identical to both sycl_gemm::gemm_custom and
// the routed batchlas::gemm, so the caller chooses without either side
// adapting. An EMPTY function means "use gemm_custom", which is what keeps the
// kernel layer standalone: tests and any direct caller get the native kernel
// with no dispatch dependency, and a vendor-free build is unaffected because
// the resolver falls back to native there anyway.
template <typename T>
using TrsmTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// V2, the blocked driver, for orders above trsm_cta_max_n<T>(). Calls V1 on
// each diagonal block and `trailing_gemm` (default: sycl_gemm::gemm_custom)
// for the trailing update.
template <typename T>
Event trsm_native_blocked(Queue& ctx,
                          const MatrixView<T, MatrixFormat::Dense>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B,
                          T alpha,
                          Side side,
                          Uplo uplo,
                          Transpose transA,
                          Diag diag,
                          TrsmTrailingGemm<T> trailing_gemm = {});

} // namespace batchlas::sycl_trsm
