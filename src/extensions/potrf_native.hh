#pragma once

// Native batched POTRF -- declarations only, so the route table and the vendor-free
// facade need no <sycl/sycl.hpp>. Two tiers: CTA (one matrix resident in local
// memory) and Blocked (its leaf is that CTA kernel). preferred() is false for both.
// evidence: docs/perf/potrf.md

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_potrf {

// Per-type CTA capacity for a local-memory budget in BYTES; the budget is a device
// property, so a hardcoded ceiling makes supports() promise an unlaunchable route.
template <typename T>
int potrf_cta_max_n_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int potrf_cta_max_n();

template <typename T>
bool potrf_blocked_available();

// Workspace in bytes -- replay the layout through BumpAllocator::measuring();
// a hand-summed exact figure fails the allocator's own capacity check.
template <typename T>
std::size_t potrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

// Test hook: low 16 bits G (matrices/work-group), high 16 L (items/matrix); 0 if unfit.
template <typename T>
unsigned potrf_cta_debug_launch(Queue& ctx, int n, int batch);

// Direct-call entry: a forced route that supports() rejects falls back to
// automatic() and silently runs the vendor, so tests bypass the gate here.
template <typename T>
Event potrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Uplo uplo,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

// Trailing-update GEMM, injected to reach the ROUTED gemm; empty means gemm_custom.
template <typename T>
using PotrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// Panel solve, injected likewise; empty means sycl_trsm::trsm_native_blocked.
// Argument order is the positional trsm entry point's: ALPHA IS IN POSITION 4.
template <typename T>
using PotrfPanelSolve = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,   // A: the ib x ib factored L11
    const MatrixView<T, MatrixFormat::Dense>&,   // B: the m2 x ib panel, in place
    T,                                           // alpha
    Side, Uplo, Transpose, Diag)>;

template <typename T>
std::size_t potrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A,
                                      Uplo uplo);

// Test hook: low 16 bits nb (diagonal-block order), high 16 W (trailing-update width).
template <typename T>
unsigned potrf_blocked_debug_params(Queue& ctx, int n);

// Uplo::LOWER ONLY -- the right-looking schedule would overwrite the wrong triangle
// for Upper, so it throws. `info` is LAPACK's: 1-based, GLOBAL, first failure wins;
// the leaf writes a sub-view-LOCAL index, so the driver translates and merges.
template <typename T>
Event potrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Uplo uplo,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             PotrfTrailingGemm<T> trailing_gemm = {},
                             PotrfPanelSolve<T> panel_solve = {});

}  // namespace batchlas::sycl_potrf
