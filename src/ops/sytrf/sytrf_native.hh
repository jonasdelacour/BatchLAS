#pragma once

// Native batched sytrf / sytrs / fused sysv, declarations only. PIVOT CONTRACT: LAPACK's ipiv, 1-based
// int32 packed into the first half of the int64 span (as_span<int>()), item b at [b n, (b+1) n), the
// same packing getrf uses (getrf_native.hh). evidence: docs/perf/sytrf.md

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_sytrf {

inline constexpr int kSytrfCtaMaxN = 32;  // one sub-group per matrix: the sub-group width is the ceiling
inline constexpr int kSytrfCtaWg = 32;

inline constexpr int kSytrfWgMax = 256;  // the widest work-group of the global-memory kernels

// The per-item status fallback when the caller's `info` span is short; zero bytes are never returned.
template <typename T>
BATCHLAS_INTERNAL_API std::size_t sytrf_native_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrf_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                               Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

template <typename T>
BATCHLAS_INTERNAL_API Event sytrf_wg_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                              Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

// `skip_info` (one entry per item, or null): an item whose entry is nonzero keeps its B, as LAPACK
// ?sysv leaves B alone after a singular ?sytrf.
template <typename T>
BATCHLAS_INTERNAL_API Event sytrs_native_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                                  const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                                  Span<int64_t> pivots, const int32_t* skip_info = nullptr);

template <typename T>
BATCHLAS_INTERNAL_API Event sysv_cta_dispatch(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                              const MatrixView<T, MatrixFormat::Dense>& B, Uplo uplo,
                                              Span<int64_t> pivots, Span<std::byte> workspace, Span<int32_t> info);

}  // namespace batchlas::sycl_sytrf
