// SPDX-License-Identifier: MIT
// Eigenpair checks shared by the syev tests: orthonormal V and ||AV - V diag(w)|| through
// batchlas::verify, judged at its orthogonality_rotations and eigen_residual bounds.
#pragma once

#include "test_utils.hh"

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <span>

namespace test_utils {

/// V (n x n, one eigenvector per column) is orthonormal and (A0, V, W) an eigendecomposition, over
/// @p items of the batch (default: first, middle, last). W holds n values per item, packed.
template <typename Scalar>
void expect_eigenpairs(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                       const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& V,
                       batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n,
                       std::span<const int> items = {}) {
    using Real = typename batchlas::base_type<Scalar>::type;
    const batchlas::VectorView<Real> w(W, n, V.batch_size());
    const double ortho = batchlas::verify::orthogonality(V, items);
    const double resid = batchlas::verify::eigen_residual(A0, V, w, items);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, ortho);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::eigen_residual, n, resid);
}

template <typename Scalar>
void expect_eigenpairs(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                       const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& V,
                       batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n, int item) {
    expect_eigenpairs<Scalar>(A0, V, W, n, std::span<const int>(&item, 1));
}

}  // namespace test_utils
