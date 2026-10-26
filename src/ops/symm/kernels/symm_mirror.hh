// SPDX-License-Identifier: MIT
#pragma once

// The symm / hemm `mirror` family: C = alpha A B + beta C (Side::Left) or alpha B A + beta C
// (Side::Right) with A read from its `uplo` triangle only, inside the gemm tile loader. Hermitian
// (Herm) conjugates the mirrored element and takes the diagonal as real. No scratch.

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_symm {

/// @pre homogeneous views, C nonempty, A square of C's side extent; at most `chunk` items per launch.
template <typename T, bool Herm>
BATCHLAS_INTERNAL_API Event symm_mirror(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& B,
                                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side,
                                        Uplo uplo, int chunk);

}  // namespace batchlas::sycl_symm
