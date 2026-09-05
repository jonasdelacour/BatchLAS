#pragma once

// Native batched ORGQR declarations: one tier, Algorithm::Blocked, which is ormqr
// applied to an identity. preferred() is false everywhere, so a vendor-present build
// still takes cuSOLVER's per-item loop. evidence: docs/perf/qr.md#orgqr-grid

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_orgqr {

// Defined in orgqr_blocked.cc, so the flag and the compiled TU are one fact. It is NOT
// "is ormqr_blocked compiled" -- that advertises a route the facade cannot service.
template <typename T>
bool orgqr_blocked_available();


// Test hook. The width must be a multiple of 16, and >= 32 for complex
// (gemm_kernels.cc's wide-scalar min_dim gate). evidence: docs/perf/qr.md#block-width-evidence
template <typename T>
int orgqr_blocked_debug_block_size(Queue& ctx, int m, int n);

// The apply-Q seam, injected to reach the ROUTED ormqr -- a native entry point called
// from a driver TU bypasses RouteTable<Op::ormqr>. Argument order is the positional
// ormqr entry point's. Absent injection throws; there is no Backend to fall back on.
template <typename T>
using OrgqrApplyQ = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,   // A: the geqrf output, reflectors
    const MatrixView<T, MatrixFormat::Dense>&,   // C: the identity, overwritten by Q
    Side, Transpose,
    Span<T>,                                     // tau
    Span<std::byte>,                             // workspace
    int32_t)>;                                   // block_size_hint

template <typename T>
using OrgqrApplyQBufferSize = std::function<std::size_t(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    Side, Transpose,
    Span<T>,
    int32_t)>;

// Workspace in bytes -- replay the layout through BumpAllocator::measuring(). The size
// query comes from the SAME resolution as the call; neither may dereference data.
template <typename T>
std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau,
                                      OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

// Direct-call entry: a forced route that supports() rejects falls back to automatic()
// and silently runs the vendor, so tests bypass the table here -- which means this
// entry point must re-check every RouteTable<Op::orgqr,T>::supports() gate itself.
template <typename T>
Event orgqr_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             OrgqrApplyQ<T> apply_q = {},
                             OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

}  // namespace batchlas::sycl_orgqr
