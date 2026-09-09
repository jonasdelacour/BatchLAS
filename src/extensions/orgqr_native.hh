#pragma once

// Native batched ORGQR declarations: one tier, Algorithm::Blocked, which is ormqr applied
// to an identity. preferred() is false everywhere. evidence: docs/perf/qr.md#orgqr-grid

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_orgqr {

template <typename T>
bool orgqr_blocked_available();

// Test hook. The width must be a multiple of 16, and >= 32 for complex (gemm_kernels.cc's
// wide-scalar min_dim gate). evidence: docs/perf/qr.md#block-width-evidence
template <typename T>
int orgqr_blocked_debug_block_size(Queue& ctx, int m, int n);

// Must be the ROUTED ormqr -- a native ormqr entry point called from a driver TU bypasses
// RouteTable<Op::ormqr>. Argument order is the positional entry point's; absent injection throws.
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

// Runs under BumpAllocator::measuring(): same resolution as the call, no data dereference.
template <typename T>
std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Span<T> tau,
                                      OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

// Reachable without the route table, so it must re-check every supports() gate of
// RouteTable<Op::orgqr,T> itself -- a rejected forced route silently runs the vendor.
template <typename T>
Event orgqr_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             OrgqrApplyQ<T> apply_q = {},
                             OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

}  // namespace batchlas::sycl_orgqr
