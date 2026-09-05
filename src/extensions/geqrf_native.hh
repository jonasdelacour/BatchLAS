#pragma once

// Native batched GEQRF: the CTA tier (panel resident in local memory) and the
// blocked driver, whose panel leaf IS the CTA device function -- so both TUs must
// share one device-code cluster. preferred() is false for both arms: they run only
// vendor-free or forced. evidence: docs/perf/qr.md#route-arms

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_geqrf {

// CTA capacity, area and height; unsuffixed, at the standard budget of local_mem_size
// less 4096 B. The budget is a parameter because device_limits.hh's constant is 2.06x
// wrong here, and must equal the launch's local_accessor. 0 means the tier is absent.
template <typename T>
int geqrf_cta_max_m_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int64_t geqrf_cta_max_elems_for_slm(std::size_t slm_budget_bytes);

template <typename T>
int geqrf_cta_max_m();

template <typename T>
int64_t geqrf_cta_max_elems();

// Specialised in geqrf_blocked.cc beside the driver, so it cannot advertise a tier
// whose TU is absent from the build.
template <typename T>
bool geqrf_blocked_available();

// Both sizes must come from a BumpAllocator::measuring() replay; a hand-summed
// figure fails the allocator's own capacity check. sytrd sizes these with null
// pointers (so never dereference A.data_ptr() or tau.data()) at (m_max x nb_max)
// but calls smaller, so both must be monotone in (rows, cols, batch).
template <typename T>
std::size_t geqrf_cta_buffer_size(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
std::size_t geqrf_blocked_buffer_size(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& A);

// nb in the low 16 bits; in the high 16, the leaf the LEADING panel takes
// (1 = resident, 2 = global), varying per panel; 0 for the whole word means the
// driver is absent. A short-final-panel test needs m > n, a middle panel, or
// complex. evidence: docs/perf/qr.md#the-short-final-panel-vacuity
template <typename T>
unsigned geqrf_blocked_debug_params(Queue& ctx, int m, int n);

// Injected so trailing updates resolve through RouteTable<Op::gemm> instead of the
// native kernel entry point; empty means "use sycl_gemm::gemm_custom".
template <typename T>
using GeqrfTrailingGemm = std::function<Event(
    Queue&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    const MatrixView<T, MatrixFormat::Dense>&,
    T, T, Transpose, Transpose, ComputePrecision)>;

// Direct calls no vendor can serve, so a facade test must compare element by
// element -- a residual passes on cuSOLVER too. Each re-checks every supports()
// gate and throws, being reachable without the table.
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

// The leaf both tiers share: in-place m x n panel factorisation, LAPACK ?GEQR2
// semantics, resident when geqrf_cta_fits<T>() and streamed otherwise
// (used_resident_out says which). Raw pointers, not a MatrixView, because a slice
// carries the PARENT pointer array and defaults stride to ld*cols. tau is indexed
// tau_ptr[b * tau_batch_stride + tau_offset + j] with k = min(rows, cols) of the
// WHOLE matrix; a panel-derived stride scatters tau, silently and only for batch > 1.
template <typename T>
Event geqrf_panel_factorize(Queue& ctx,
                            T* a_ptr, int ld, int stride,
                            int m, int n, int batch,
                            T* tau_ptr, int tau_batch_stride, int tau_offset,
                            bool* used_resident_out = nullptr);

// The same predicate the launcher applies, so the table's capacity and the driver's
// per-panel choice cannot disagree.
template <typename T>
bool geqrf_cta_fits(int m, int n, std::size_t slm_budget_bytes);

}  // namespace batchlas::sycl_geqrf
