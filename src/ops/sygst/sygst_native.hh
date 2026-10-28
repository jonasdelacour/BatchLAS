#pragma once  // sygst blocked's diagonal-block kernel. evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace batchlas::sycl_sygst {

/// The largest diagonal block the kernel reduces: one work-item per column, the column in registers.
inline constexpr int kDiagMaxOrder = 32;

inline constexpr int kDiagWg = 128;              ///< the widest work-group: kb columns per item
inline constexpr std::size_t kDiagSlmCap = 32768;  ///< a group's shared memory target, bytes

template <typename T>  /// F's leading dimension: whole 16-byte packs per line, plus one pack against bank conflicts
constexpr int diag_ldf(int kb) {
    const int v = sizeof(T) >= 16 ? 1 : static_cast<int>(16 / sizeof(T));
    return (kb + v - 1) / v * v + v;
}

template <typename T>  /// shared-memory elements of one item: F and the transpose buffer (ld kb + 1)
constexpr std::size_t diag_item_elems(int kb) {
    return static_cast<std::size_t>(kb) * static_cast<std::size_t>(diag_ldf<T>(kb) + kb + 1);
}

template <typename T>  /// items per work-group at order kb: as many as kDiagWg columns and kDiagSlmCap hold
constexpr int diag_items_per_group(int kb) {
    const std::size_t per = diag_item_elems<T>(kb) * sizeof(T);
    const int g = static_cast<int>(std::min<std::size_t>(kDiagWg / (kb < 1 ? 1 : kb), kDiagSlmCap / per));
    return g < 1 ? 1 : g;
}

template <typename T>  /// shared memory of one work-group at order kb, in whole 16-byte packs
constexpr std::size_t diag_slm_bytes(int kb) {
    const std::size_t bytes = static_cast<std::size_t>(diag_items_per_group<T>(kb)) * diag_item_elems<T>(kb) * sizeof(T);
    return (bytes + 15) / 16 * 16;
}

// The kb x kb diagonal block of every item, in place, reading and writing only the uplo triangles:
// itype 1 C = F^-1 A F^-H, itypes 2, 3 C = F^H A F, with F = L (Lower) or U^H (Upper).
// With `full`, the whole Hermitian result (real diagonal) is also stored there. A null `mask` is every
// item; otherwise item b is skipped where mask[b] == 0 (both kernels).
template <typename T>
BATCHLAS_INTERNAL_API Event sygst_diag(Queue& ctx, T* a, int lda, std::ptrdiff_t stride_a, const T* b, int ldb,
                                       std::ptrdiff_t stride_b, int kb, int batch, int itype, Uplo uplo, T* full,
                                       int ldf, std::ptrdiff_t stride_f, const std::uint8_t* mask);

// full := the whole Hermitian kb x kb block whose uplo triangle a holds (the other is not read).
template <typename T>
BATCHLAS_INTERNAL_API Event sygst_mirror(Queue& ctx, const T* a, int lda, std::ptrdiff_t stride_a, int kb, int batch,
                                         Uplo uplo, T* full, int ldf, std::ptrdiff_t stride_f,
                                         const std::uint8_t* mask);

}  // namespace batchlas::sycl_sygst

namespace batchlas::ops::sygst {

template <class T>
using MV = MatrixView<T, MatrixFormat::Dense>;

template <class T>
constexpr Transpose kH = is_std_complex_v<T> ? Transpose::ConjTrans : Transpose::Trans;

// Blocked (kernels/sygst_blocked.cc): workspace bytes, and the reduction on an in-order queue.
template <class T>
std::size_t blocked_workspace(Queue& q, const MV<T>& A, int nb, int itype);
template <Backend B, class T>
Event blocked(Queue& q, const MV<T>& A, const MV<T>& Bf, int nb, int itype, Uplo uplo, Span<std::byte> ws);

// TwoSided (kernels/sygst_twosided.cc).
template <class T>
std::size_t twosided_workspace(Queue& q, const MV<T>& A, int itype);
template <Backend B, class T>
Event twosided(Queue& q, const MV<T>& A, const MV<T>& Bf, int itype, Uplo uplo, Span<std::byte> ws);

}  // namespace batchlas::ops::sygst
