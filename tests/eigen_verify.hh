// SPDX-License-Identifier: MIT
// Eigen checks shared by the syev tests through batchlas::verify: orthonormal V and ||AV - V diag(w)||
// (orthogonality_rotations, eigen_residual), and eigenvalues against LAPACKE (values).
#pragma once

#include "test_utils.hh"

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

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

/// W (n values per item, packed) against LAPACKE ?syevd / ?heevd of A0 (both triangles valid) in
/// double: max |w - w_ref| / ||A0||_2 at Check::values, over @p items (default: first, middle, last).
/// @p sorted_copy compares as a multiset, for a solver asked not to sort. Skips without LAPACKE.
template <typename Scalar>
void expect_eigenvalues_match_lapacke(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                                      const batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n,
                                      bool sorted_copy = false, std::span<const int> items = {}) {
#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    (void)A0, (void)W, (void)n, (void)sorted_copy, (void)items;
    GTEST_SKIP() << "no host LAPACKE reference in this build";
#else
    using Real = typename batchlas::base_type<Scalar>::type;
    const int batch = A0.batch_size();
    const std::vector<int> picked = items.empty() ? batchlas::verify::default_items(batch) : std::vector<int>(items.begin(), items.end());
    std::vector<Real> got(W.begin(), W.begin() + static_cast<std::ptrdiff_t>(n) * batch);
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch));
    double scale = 0;
    for (int b : picked) {
        auto a = batchlas::verify::copy_item(A0, b);
        ASSERT_TRUE(batchlas::verify::eigenvalues(n, a, ref[static_cast<std::size_t>(b)])) << "LAPACKE reference failed, item " << b;
        for (double l : ref[static_cast<std::size_t>(b)]) scale = batchlas::verify::nanmax(scale, std::fabs(l));
        if (sorted_copy) std::sort(got.begin() + static_cast<std::ptrdiff_t>(b) * n, got.begin() + static_cast<std::ptrdiff_t>(b + 1) * n);
    }
    const batchlas::VectorView<Real> w(got.data(), n, batch);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, n, batchlas::verify::values_error(w, ref, scale, picked));
#endif
}

/// W against @p W_ref, another result of the code under test (no LAPACKE): max |w - w_ref| / max |w_ref|
/// at Check::values over every item. @p reversed compares W with W_ref read back to front per item.
template <typename Scalar>
void expect_eigenvalues_agree(const batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W,
                              const batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W_ref, int n, int batch,
                              bool reversed = false) {
    using Real = typename batchlas::base_type<Scalar>::type;
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch), std::vector<double>(static_cast<std::size_t>(n)));
    double scale = 0;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) {
            const double r = W_ref[static_cast<std::size_t>(b) * n + (reversed ? n - 1 - i : i)];
            ref[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)] = r;
            scale = batchlas::verify::nanmax(scale, std::fabs(r));
        }
    const batchlas::VectorView<Real> w(const_cast<Real*>(W.data()), n, batch);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, n, batchlas::verify::values_error(w, ref, scale, batchlas::verify::all_items(batch)));
}

}  // namespace test_utils
