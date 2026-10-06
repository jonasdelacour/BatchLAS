#pragma once

// Batch-shape helpers shared by the gemm entry point and the vendor TUs. No routing lives here:
// gemm's kernel choice is src/ops/gemm/gemm.cc (flat kernel selection).

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::backend {

template <typename T>
inline bool gemm_has_heterogeneous_batch(const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         const MatrixView<T, MatrixFormat::Dense>& C) {
    return A.is_heterogeneous() || B.is_heterogeneous() || C.is_heterogeneous();
}

// Every batch member's (m, n, k) agrees across A, B and C.
template <typename T>
inline bool gemm_batch_dimensions_compatible(const MatrixView<T, MatrixFormat::Dense>& A,
                                             const MatrixView<T, MatrixFormat::Dense>& B,
                                             const MatrixView<T, MatrixFormat::Dense>& C,
                                             Transpose transA,
                                             Transpose transB) {
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        return false;
    }

    for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
        const auto [m, k] = get_effective_dims(A, transA, batch_index);
        const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
        if (k != k_b) {
            return false;
        }
        if (C.rows(batch_index) != m || C.cols(batch_index) != n) {
            return false;
        }
        if (m < 0 || n < 0 || k < 0) {
            return false;
        }
    }

    return true;
}

} // namespace batchlas::backend
