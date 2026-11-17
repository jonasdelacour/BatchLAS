// SPDX-License-Identifier: MIT
#pragma once

// symm / hemm `mirror`: \f$C = \alpha A B + \beta C\f$ (Left) or \f$\alpha B A + \beta C\f$ (Right), A read from its
// `uplo` triangle only (Herm: conjugate mirror, real diagonal). At most `chunk` items per launch.

#include <src/util/internal-api.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::ops::symm {

template <typename Scalar, bool Herm>
BATCHLAS_INTERNAL_API Event symm_mirror(Queue& q, const MatrixView<Scalar, MatrixFormat::Dense>& A,
                                        const MatrixView<Scalar, MatrixFormat::Dense>& B,
                                        const MatrixView<Scalar, MatrixFormat::Dense>& C, Scalar alpha, Scalar beta,
                                        Side side, Uplo uplo, int chunk);

}  // namespace batchlas::ops::symm
