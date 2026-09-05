#pragma once

// Native batched GETRS declarations: the composed tier (row permutation + two
// routed trsm, getrs_native.cc) and the fused narrow-RHS tier (getrs_fused.cc).
// Routing windows ship in route_getrs.hh; evidence: docs/perf/lu.md

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getrs {

template <typename T>
bool getrs_blocked_available();

// Size via BumpAllocator::measuring(); a hand-summed figure fails the
// allocator's own capacity check. Zero is legitimate; dereferences nothing.
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      Transpose transA);

// Injected so both solves go through the ROUTED trsm, never a native trsm entry
// point. Positional form of batchlas::trsm -- alpha comes THIRD, not last.
template <typename T>
using GetrsSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

template <typename T>
Event getrs_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& B,
                             Transpose transA,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             GetrsSolveTrsm<T> solve_trsm = {});

// Per-call boundary on B.cols(): interchange walk below it, in-place gather at or
// above. Both spell the SAME permutation -- a speed choice, never correctness.
// evidence: docs/perf/lu.md#getrs-collapsed-permutation
inline constexpr int kGetrsPermGatherMinNrhs = 16;

// Test-only: 1 = gather, 0 = walk; the gather silently falls back to the walk
// when local memory cannot hold a column of B plus the index arrays.
template <typename T>
int getrs_perm_spelling_debug(Queue& ctx, int n, int nrhs);

// PIVOT FORMAT is getrf's and backend-dependent: packed 1-based int32 on CUDA and
// ROCm, genuine 1-based int64 on Netlib, and an INTERCHANGE LIST, not a
// permutation vector (getrf_native.hh's PIVOT CONTRACT). getrs must match getrf.

// The fused tier's ceiling is the RESIDENT RHS: capacity is a supports() question
// and not a preferred() one -- above it the kernel does not launch.

// A CAPABILITY, not a speed window: above this width the kernel is not instantiated.
inline constexpr int64_t kGetrsFusedMaxRhs = 8;

template <typename T>
bool getrs_fused_available();

// Ask the DEVICE for slm_budget_bytes, never device_limits.hh (hardcoded 49152,
// 2.06x wrong here), or supports() will claim a route that cannot launch.
template <typename T>
std::size_t getrs_fused_max_rhs_elems(std::size_t slm_budget_bytes);

template <typename T>
std::size_t getrs_fused_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    Transpose transA);

// Trans/ConjTrans swap the two solves and reverse-permute the OUTPUT instead.
template <typename T>
Event getrs_fused_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           Transpose transA,
                           Span<int64_t> pivots,
                           Span<std::byte> workspace);

}  // namespace batchlas::sycl_getrs
