#pragma once

// Native batched POTRF, declarations only: the route table and the vendor-free facade need no
// <sycl/sycl.hpp>. preferred() is false for all three tiers. EVERY *_dispatch re-applies
// supports()'s gates -- a rejected forced route silently runs the vendor. evidence: docs/perf/potrf.md

#include "../util/internal-api.hh"
#include "../util/resident_capacity.hh"
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_potrf {

// TINY tier (potrf_tiny.cc): it owns NO local memory, so the ceiling takes no budget argument
// and is a flat compile-time constant; 0 = absent. evidence: docs/perf/potrf.md#the-tiny-tier
template <typename T>
BATCHLAS_INTERNAL_API int potrf_tiny_max_n();

// NOT zero: an empty or SHORT caller `info` span means "not requested" and draws scratch.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t potrf_tiny_buffer_size(
    Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API unsigned potrf_tiny_debug_launch(Queue& ctx, int n);  // G | S<<16

template <typename T>
BATCHLAS_INTERNAL_API Event potrf_tiny_dispatch(Queue& ctx,
                                                const MatrixView<T, MatrixFormat::Dense>& A,
                                                Uplo uplo,
                                                Span<std::byte> workspace,
                                                Span<int32_t> info);

// Per-type CTA capacity for a budget in BYTES: the budget is a device property, and a
// hardcoded ceiling makes supports() promise an unlaunchable route. `min_blocks_per_sm`
// scales it to the ADVERTISED capacity; 1 asks the residency question, which has a different
// answer. evidence: docs/perf/potrf.md#the-occupancy-rule
template <typename T>
BATCHLAS_INTERNAL_API int potrf_cta_max_n_for_slm(
    std::size_t slm_budget_bytes,
    int min_blocks_per_sm = resident::kMinBlocksPerSm);

template <typename T>
int potrf_cta_max_n();

template <typename T>
BATCHLAS_INTERNAL_API bool potrf_blocked_available();

// Replay the layout through BumpAllocator::measuring(): a hand-summed exact figure fails it.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t potrf_cta_buffer_size(Queue& ctx,
                                                        const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API unsigned potrf_cta_debug_launch(
    Queue& ctx, int n, int batch,
    int min_blocks_per_sm = resident::kMinBlocksPerSm);  // G | L<<16, 0 if unfit

template <typename T>
BATCHLAS_INTERNAL_API Event potrf_cta_dispatch(Queue& ctx,
                                               const MatrixView<T, MatrixFormat::Dense>& A,
                                               Uplo uplo,
                                               Span<std::byte> workspace,
                                               Span<int32_t> info,
                                               int min_blocks_per_sm = resident::kMinBlocksPerSm);

// Trailing-update GEMM, injected to reach the ROUTED gemm; empty means gemm_custom.
template <typename T>
using PotrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// Injected likewise; empty means sycl_trsm::trsm_native_blocked. ALPHA IS IN POSITION 4.
template <typename T>
using PotrfPanelSolve = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,   // A: the ib x ib factored L11
    const MatrixView<T, MatrixFormat::Dense>&,   // B: the m2 x ib panel, in place
    T,                                           // alpha
    Side, Uplo, Transpose, Diag)>;

template <typename T>
BATCHLAS_INTERNAL_API std::size_t potrf_blocked_buffer_size(Queue& ctx,
                                                            const MatrixView<T, MatrixFormat::Dense>& A,
                                                            Uplo uplo);

template <typename T>
BATCHLAS_INTERNAL_API unsigned potrf_blocked_debug_params(Queue& ctx, int n);  // nb | W<<16

// Uplo::LOWER ONLY -- the right-looking schedule overwrites the wrong triangle for Upper,
// so it throws. `info` is LAPACK's (1-based, GLOBAL, first failure wins) while the leaf
// writes a sub-view-LOCAL index, so the driver translates and merges.
template <typename T>
BATCHLAS_INTERNAL_API Event potrf_blocked_dispatch(Queue& ctx,
                                                   const MatrixView<T, MatrixFormat::Dense>& A,
                                                   Uplo uplo,
                                                   Span<std::byte> workspace,
                                                   Span<int32_t> info,
                                                   PotrfTrailingGemm<T> trailing_gemm = {},
                                                   PotrfPanelSolve<T> panel_solve = {});

}  // namespace batchlas::sycl_potrf
