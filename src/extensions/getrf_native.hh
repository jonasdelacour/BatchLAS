#pragma once

// Native batched GETRF: capability queries, workspace sizing, routed seams, and
// direct-call entry points. The blocked driver's panel leaf IS the CTA tier's device
// function, so both TUs must sit in one device-code cluster. PIVOT CONTRACT: the
// int64_t span packs 1-based int32 in its first half (as_span<int>()) -- an
// interchange list, not a permutation; complex pivots on cabs1, not cuBLAS's modulus.
// evidence: docs/perf/lu.md#getrf-window-evidence

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_getrf {

// A RUNTIME local_mem_size budget, not device_limits.hh's build-time constant, and
// it must cover the pivot-search scratch as well as the tile. 0 = absent.
template <typename T>
int getrf_cta_max_n_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int getrf_cta_max_n();

template <typename T>
bool getrf_blocked_available();

// Zero is a legitimate size, and neither query may touch A.data_ptr().
template <typename T>
std::size_t getrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
std::size_t getrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// Test hook: low 16 bits nb, high 16 the leading panel's leaf (1 = local, 2 = global); 0 if absent.
template <typename T>
unsigned getrf_blocked_debug_params(Queue& ctx, int n);

// An empty seam means "use sycl_gemm::gemm_custom" rather than a routed gemm.
template <typename T>
using GetrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// Positional form of the routed batchlas::trsm -- alpha comes THIRD, not last.
template <typename T>
using GetrfPanelSolveTrsm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, Side, Uplo, Transpose, Diag)>;

template <typename T>
Event getrf_cta_dispatch(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         Span<int64_t> pivots,
                         Span<std::byte> workspace,
                         Span<int32_t> info);

template <typename T>
Event getrf_blocked_dispatch(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<int64_t> pivots,
                             Span<std::byte> workspace,
                             Span<int32_t> info,
                             GetrfTrailingGemm<T> trailing_gemm = {},
                             GetrfPanelSolveTrsm<T> panel_trsm = {});

// Budgets an explicit SLM tree argmax: sycl::reduce_over_group fails to launch at
// specific byte counts near 48 KB. evidence: docs/perf/lu.md#the-48-kb-launch-hole
template <typename T>
bool getrf_cta_fits(int n, std::size_t slm_budget_bytes);

template <typename T>
bool getrf_leaf_fits(int m, int n, std::size_t slm_budget_bytes);

// `piv_stride` is the matrix ORDER, never the panel width; `piv_base` is the panel's
// first global row. `info_ptr` is read as well as written, so zero it before panel 0.
template <typename T>
Event getrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            int* piv_ptr, int piv_stride, int piv_base,
                            int32_t* info_ptr,
                            bool* used_resident_out);

}  // namespace batchlas::sycl_getrf
