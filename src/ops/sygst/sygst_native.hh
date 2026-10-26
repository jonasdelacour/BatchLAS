#pragma once  // sygst blocked's diagonal-block kernel. evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstddef>

namespace batchlas::sycl_sygst {

/// The largest diagonal block the kernel reduces: one work-item per column, the column in registers.
inline constexpr int kDiagMaxOrder = 32;

template <typename T>  /// shared memory of one work-group at order kb: F and a transpose buffer per item
constexpr std::size_t diag_slm_bytes(int kb) {
    const int items = kb >= 1 ? (kDiagMaxOrder / kb > 0 ? kDiagMaxOrder / kb : 1) : 1;
    return static_cast<std::size_t>(2 * items * kb * (kb + 1)) * sizeof(T);
}

// The kb x kb diagonal block of every item, in place, reading and writing only the uplo triangles:
// itype 1 C = F^-1 A F^-H, itypes 2, 3 C = F^H A F, with F = L (Lower) or U^H (Upper).
template <typename T>
BATCHLAS_INTERNAL_API Event sygst_diag(Queue& ctx, T* a, int lda, std::ptrdiff_t stride_a, const T* b, int ldb,
                                       std::ptrdiff_t stride_b, int kb, int batch, int itype, Uplo uplo);

}  // namespace batchlas::sycl_sygst
