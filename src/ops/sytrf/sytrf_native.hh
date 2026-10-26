#pragma once

// Native batched sytrf / sytrs (cta, blocked, wg) / fused sysv, declarations only. PIVOT CONTRACT: LAPACK's ipiv, 1-based
// int32 packed into the first half of the int64 span (as_span<int>()), item b at [b n, (b+1) n), the
// same packing getrf uses (getrf_native.hh). evidence: docs/perf/sytrf.md

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace batchlas::sycl_sytrf {

inline constexpr int kSytrfCtaMaxN = 32;  // one sub-group per matrix: the sub-group width is the ceiling
inline constexpr int kSytrfCtaWg = 32;

inline constexpr int kSytrfWgMax = 256;  // the widest work-group of the global-memory kernels

// Blocked tier: panel width nb (the trailing gemm's k is nb + 1 = 32, see ldl_panel) and the width
// of the diagonal blocks the native trailing kernel owns (the gemm takes the rest). Untimed.
inline constexpr int kSytrfBlockedNb = 31;
inline constexpr int kSytrfBlockedUb = 128;

// Mandatory: sytrf_blocked_dispatch throws on an empty seam (inject the public gemm).
template <typename T>
using SytrfTrailingGemm = std::function<Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                              const MatrixView<T, MatrixFormat::Dense>&,
                                              const MatrixView<T, MatrixFormat::Dense>&, T, T, Transpose, Transpose,
                                              ComputePrecision)>;

// The per-item status fallback when the caller's `info` span is short; zero bytes are never returned.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t sytrf_native_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrf_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                               Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrf_wg_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                              Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

template <typename T>
BATCHLAS_INTERNAL_API std::size_t sytrf_blocked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrf_blocked_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                                   Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info,
                                                   SytrfTrailingGemm<T> trailing_gemm);

// sytrs `wg`, one work-group per right-hand side. `skip_info` (per item, or null), every sytrs
// driver: a nonzero entry keeps that item's B, as LAPACK ?sysv after a singular ?sytrf.
template <typename T>
BATCHLAS_INTERNAL_API Event sytrs_native_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                  const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                                  Span<int64_t> pivots, const int32_t* skip_info = nullptr);

// sytrs `cta`: one work-group per item, the factor converted once in local memory, one sub-group
// per right-hand side (sytrs_device.hh). A capability ceiling: the register tile is R = n / 32 <= 4.
inline constexpr int kSytrsCtaMaxN = 128;
inline constexpr int kSytrsCtaMaxSg = 8;  // sub-groups per work-group, at most one per right-hand side

// Local memory of one `cta` work-group (the 48 KB launch hole padded, as getrs_fused's).
template <typename T>
constexpr std::size_t sytrs_cta_slm_bytes(int n) {
    const auto un = static_cast<std::size_t>(n);
    const std::size_t bytes = (un * static_cast<std::size_t>(n | 1) + un) * sizeof(T) + (3 * un + 1) * sizeof(int);
    return bytes > 47104 && bytes <= 49664 ? std::size_t(49920) : bytes;
}

template <typename T>
BATCHLAS_INTERNAL_API Event sytrs_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                               const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                               Span<int64_t> pivots, const int32_t* skip_info = nullptr);

// sytrs `sg`: one sub-group per (item, rhs slot), the factor read from A as stored (no local memory).
inline constexpr int kSytrsSgMaxN = 256;

template <typename T>
BATCHLAS_INTERNAL_API Event sytrs_sg_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                              const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                              Span<int64_t> pivots, const int32_t* skip_info = nullptr);

// Positional form of the routed batchlas::trsm (alpha third), injected by the facade.
template <typename T>
using SytrsSolveTrsm = std::function<Event(Queue&, const MatrixView<T, MatrixFormat::Dense>&,
                                           const MatrixView<T, MatrixFormat::Dense>&, T, Side, Uplo, Transpose, Diag)>;

// sytrs `blocked` (?sytrs2): converted factor and gathered B in workspace per chunk, two trsm.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t sytrs_blocked_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                            const MatrixView<T, MatrixFormat::Dense>& B);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrs_blocked_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                   const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                                   Span<int64_t> pivots, Span<std::byte> workspace,
                                                   const int32_t* skip_info, SytrsSolveTrsm<T> solve_trsm);

template <typename T>
BATCHLAS_INTERNAL_API Event sysv_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                              const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                              Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

}  // namespace batchlas::sycl_sytrf
