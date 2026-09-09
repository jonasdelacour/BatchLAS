#pragma once

// Native batched GETRI declarations: the arm writes P straight into C, then two
// routed trsm calls; zero workspace. evidence: docs/perf/lu.md#getri-window-evidence

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getri {

template <typename T>
bool getri_blocked_available();

// Zero is the expected answer. Runs under BumpAllocator::measuring(): no workspace
// access, no kernel launch, no A.data_ptr() dereference -- metadata only.
template <typename T>
std::size_t getri_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// Must be the ROUTED trsm -- a native trsm entry point called from a driver TU
// bypasses RouteTable<Op::trsm>. alpha comes THIRD, not last. Absent injection throws.
template <typename T>
using GetriSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

// Reachable without the route table, so it must re-check every
// RouteTable<Op::getri,T>::supports() gate and throw. A is read-only, A == C is
// unsupported, and info is exact-zero semantics, not a tolerance. pivots keep getrf's
// format: a 1-based interchange list, int32 packed into the int64 span on CUDA/ROCm.
template <typename T>
Event getri_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             GetriSolveTrsm<T> solve_trsm = {});

}  // namespace batchlas::sycl_getri
