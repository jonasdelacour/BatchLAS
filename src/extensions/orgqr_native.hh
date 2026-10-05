#pragma once

// Native batched ORGQR: one tier, Algorithm::Blocked -- ormqr applied to an identity.
// tuned/orgqr.*.txt picks it to n = 512 on both extents, so it is the DEFAULT inside
// that window, not a vendor-free fallback. evidence: docs/perf/qr.md#the-shipped-orgqr-ceiling

#include "../util/internal-api.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_orgqr {

template <typename T>
BATCHLAS_INTERNAL_API bool orgqr_blocked_available();

// Must be the SELECTED ormqr: a native entry point called from a driver TU bypasses
// ormqr's choose(). Positional argument order; absent injection throws.
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
BATCHLAS_INTERNAL_API std::size_t orgqr_blocked_buffer_size(Queue& ctx,
                                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                                            Span<T> tau,
                                                            OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

// Callable directly (tests, benchmarks), bypassing select::choose, so it re-checks every
// can_run gate itself and throws rather than computing a wrong Q.
template <typename T>
BATCHLAS_INTERNAL_API Event orgqr_blocked_dispatch(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                                   Span<T> tau,
                                                   Span<std::byte> workspace,
                                                   OrgqrApplyQ<T> apply_q = {},
                                                   OrgqrApplyQBufferSize<T> apply_q_buffer_size = {});

}  // namespace batchlas::sycl_orgqr
