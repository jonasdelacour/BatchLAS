#pragma once

// Native batched GEQRF: the CTA tier and the blocked driver, whose panel leaf IS the CTA device
// function -- both TUs must share one device-code cluster. evidence: docs/perf/qr.md#route-arms

#include "../util/internal-api.hh"
#include "../util/resident_capacity.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_geqrf {

// 2, not resident::kMinBlocksPerSm's 4: this ceiling is a ROUTING cut wearing a capacity's
// clothes, placed at the measured CTA-vs-Blocked crossover. GeqrfTest.OccupancyTargetIsPinned
// fails if it moves. evidence: docs/perf/qr.md#settling-the-target-on-the-tall-panels
inline constexpr int kGeqrfMinBlocksPerSm = 2;

// The WHOLE budget (0 = tier absent); min_blocks_per_sm cuts the slice the tier ADVERTISES.
template <typename T>
BATCHLAS_INTERNAL_API int geqrf_cta_max_m_for_slm(
    std::size_t slm_budget_bytes, int min_blocks_per_sm = kGeqrfMinBlocksPerSm);

template <typename T>
BATCHLAS_INTERNAL_API int64_t geqrf_cta_max_elems_for_slm(
    std::size_t slm_budget_bytes, int min_blocks_per_sm = kGeqrfMinBlocksPerSm);

template <typename T>
int geqrf_cta_max_m();

template <typename T>
int64_t geqrf_cta_max_elems();

template <typename T>
BATCHLAS_INTERNAL_API bool geqrf_blocked_available();

// TINY tier (square m == n <= 32): the ONE predicate for its ceiling -- the shape builder, the
// entry point and the tests all call it. evidence: docs/perf/qr.md#the-tiny-ceiling-predicate
template <typename T>
BATCHLAS_INTERNAL_API int geqrf_tiny_max_n_for_slm(std::size_t slm_budget_bytes);

template <typename T>
BATCHLAS_INTERNAL_API int geqrf_tiny_max_n();

// Zero and constant, hence monotone in (rows, cols, batch) as band_reduction.cc's sizing
// replay requires. Never dereferences A.data_ptr().
template <typename T>
BATCHLAS_INTERNAL_API std::size_t geqrf_tiny_buffer_size(
    Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API unsigned geqrf_tiny_debug_launch(Queue& ctx, int n);  // G | N<<16

template <typename T>
BATCHLAS_INTERNAL_API Event geqrf_tiny_dispatch(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A,
                                                Span<T> tau,
                                                Span<std::byte> workspace);

// A BumpAllocator::measuring() replay, monotone in (rows, cols, batch); deref neither A nor tau.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API std::size_t geqrf_blocked_buffer_size(Queue& ctx,
                                                            const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API unsigned geqrf_blocked_debug_params(
    Queue& ctx, int m, int n);  // nb | leaf<<16; high half 1 = resident, 2 = global, 0 = absent

// Empty means "use sycl_gemm::gemm_custom"; inject to route trailing updates through the table.
template <typename T>
using GeqrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

template <typename T>
BATCHLAS_INTERNAL_API Event geqrf_cta_dispatch(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               Span<T> tau,
                                               Span<std::byte> workspace);

template <typename T>
BATCHLAS_INTERNAL_API Event geqrf_blocked_dispatch(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                                   Span<T> tau,
                                                   Span<std::byte> workspace,
                                                   GeqrfTrailingGemm<T> trailing_gemm = {});

// Raw pointers, not a MatrixView: a slice carries the PARENT pointer array. tau is indexed
// tau_ptr[b * tau_batch_stride + tau_offset + j] with k = min(rows, cols) of the WHOLE matrix;
// a panel-derived stride scatters tau silently, and only for batch > 1.
template <typename T>
Event geqrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            T* tau_ptr, int tau_batch_stride, int tau_offset,
                            bool* used_resident_out = nullptr);

// The CTA TIER's predicate, occupancy-scaled: capacity, this gate and the launcher's are ONE.
template <typename T>
BATCHLAS_INTERNAL_API bool geqrf_cta_fits(
    int m, int n, std::size_t slm_budget_bytes,
    int min_blocks_per_sm = kGeqrfMinBlocksPerSm);

// The RESIDENCY predicate, at the whole budget: what chooses the resident leaf, not the tier.
template <typename T>
BATCHLAS_INTERNAL_API bool geqrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes);

template <typename T>
BATCHLAS_INTERNAL_API unsigned geqrf_cta_debug_launch(Queue& ctx, int m, int n);  // G | wg<<16

}  // namespace batchlas::sycl_geqrf
