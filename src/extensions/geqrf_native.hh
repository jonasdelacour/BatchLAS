#pragma once

// Native batched GEQRF: the CTA tier (panel resident in local memory) and the blocked driver,
// whose panel leaf IS the CTA device function -- both TUs must share one device-code cluster.
// Neither arm is preferred(): vendor-free or forced only. evidence: docs/perf/qr.md#route-arms

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_geqrf {

// slm_budget_bytes must equal the launch's local_accessor size (unsuffixed forms pass local_mem_size - 4096 B); 0 means the tier is absent.
template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int geqrf_cta_max_m();

template <typename T>
int64_t geqrf_cta_max_elems();

template <typename T>
bool geqrf_blocked_available();

// Sizes must come from a BumpAllocator::measuring() replay and be monotone in
// (rows, cols, batch); callers size with null data pointers, so never dereference them.
template <typename T>
std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
std::size_t geqrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// nb in the low 16 bits, the LEADING panel's leaf in the high 16 (1 = resident, 2 = global); 0 means the driver is absent.
template <typename T>
unsigned geqrf_blocked_debug_params(Queue& ctx, int m, int n);

// Empty means "use sycl_gemm::gemm_custom"; inject to route trailing updates through RouteTable<Op::gemm>.
template <typename T>
using GeqrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

template <typename T>
Event geqrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<T> tau,
                         Span<std::byte> workspace);

template <typename T>
Event geqrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau,
                             Span<std::byte> workspace,
                             GeqrfTrailingGemm<T> trailing_gemm = {});

// Raw pointers, not a MatrixView, because a slice carries the PARENT pointer array. tau is
// indexed tau_ptr[b * tau_batch_stride + tau_offset + j] with k = min(rows, cols) of the
// WHOLE matrix; a panel-derived stride scatters tau silently, and only for batch > 1.
template <typename T>
Event geqrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            T* tau_ptr, int tau_batch_stride, int tau_offset,
                            bool* used_resident_out = nullptr);

// The launcher applies this same predicate; forking it lets capacity and the driver's per-panel choice disagree.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes);

}  // namespace batchlas::sycl_geqrf
