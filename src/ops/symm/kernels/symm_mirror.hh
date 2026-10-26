// SPDX-License-Identifier: MIT
#pragma once

// symm / hemm `mirror`: C = alpha A B + beta C (Left) or alpha B A + beta C (Right), A read from its
// `uplo` triangle only (Herm: conjugate mirror, real diagonal). At most `chunk` items per launch.

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_symm {

template <typename T, bool Herm>
BATCHLAS_INTERNAL_API Event symm_mirror(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& B,
                                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Side side,
                                        Uplo uplo, int chunk);

}  // namespace batchlas::sycl_symm
