#pragma once

// Native batched GETRS declarations: the composed tier (row permutation + two routed
// trsm) and the fused narrow-RHS tier; windows in route_getrs.hh. evidence: docs/perf/lu.md

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

// Size via BumpAllocator::measuring(), never a hand sum; zero is a legitimate size.
template <typename T>
std::size_t getrs_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      const MatrixView<T, MatrixFormat::Dense>& B,
                                      Transpose transA);

// Positional form of the ROUTED batchlas::trsm -- alpha comes THIRD, not last.
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

// Boundary on B.cols(): interchange walk below, in-place gather at or above; both spell
// the same permutation. evidence: docs/perf/lu.md#getrs-collapsed-permutation
inline constexpr int kGetrsPermGatherMinNrhs = 16;

// Test-only: 1 = gather, 0 = walk; the gather falls back to the walk when SLM is short.
template <typename T>
int getrs_perm_spelling_debug(Queue& ctx, int n, int nrhs);

// Pivots are getrf's: a 1-based INTERCHANGE LIST, not a permutation vector -- packed int32
// on CUDA/ROCm, int64 on Netlib (getrf_native.hh PIVOT CONTRACT). getrs must match getrf.

// A capability, not a speed window: above this width the kernel is not instantiated.
inline constexpr int64_t kGetrsFusedMaxRhs = 8;

template <typename T>
bool getrs_fused_available();

// Pass a DEVICE-queried slm_budget_bytes; device_limits.hh's constant admits a route that cannot launch.
template <typename T>
std::size_t getrs_fused_max_rhs_elems(std::size_t slm_budget_bytes);

template <typename T>
std::size_t getrs_fused_buffer_size(Queue& ctx,
                                    const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    Transpose transA);

template <typename T>
Event getrs_fused_dispatch(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           Transpose transA,
                           Span<int64_t> pivots,
                           Span<std::byte> workspace);

}  // namespace batchlas::sycl_getrs
