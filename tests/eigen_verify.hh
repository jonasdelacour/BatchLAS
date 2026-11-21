// SPDX-License-Identifier: MIT
#pragma once
// Eigen checks shared by the syev tests: orthonormal V and ||AV - V diag(w)|| (MatrixView statics)
// (orthogonality_rotations, eigen_residual), and eigenvalues against LAPACKE (values). Also holds the
// random upper-bidiagonal (d, e) inputs shared by the bdsqr and bdsdc tests.

#include <tests/test_utils.hh>

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace test_utils {

/// V (n x n, one eigenvector per column) is orthonormal and (A0, V, W) an eigendecomposition, over
/// @p items of the batch (default: first, middle, last). W holds n values per item, packed.
template <typename Scalar>
void expect_eigenpairs(batchlas::Queue& ctx, const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                       const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& V,
                       batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n,
                       std::span<const int> items = {}) {
    using Real = typename batchlas::base_type<Scalar>::type;
    using MV = batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>;
    const batchlas::VectorView<Real> w(W, n, V.batch_size());
    const double ortho = MV::orthogonality(ctx, V, items);
    const double resid = MV::eigen_residual(ctx, A0, V, w, items);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, ortho);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::eigen_residual, n, resid);
}

template <typename Scalar>
void expect_eigenpairs(batchlas::Queue& ctx, const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                       const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& V,
                       batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n, int item) {
    expect_eigenpairs<Scalar>(ctx, A0, V, W, n, std::span<const int>(&item, 1));
}

/// W (n values per item, packed) against LAPACKE ?syevd / ?heevd of A0 (both triangles valid) in
/// double: max |w - w_ref| / ||A0||_2 at Check::values, over @p items (default: first, middle, last).
/// @p sorted_copy compares as a multiset, for a solver asked not to sort; @p slack scales the bound.
/// Skips without LAPACKE.
template <typename Scalar>
void expect_eigenvalues_match_lapacke(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                                      const batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& W, int n,
                                      bool sorted_copy = false, std::span<const int> items = {},
                                      std::optional<batchlas::verify::Slack> slack = std::nullopt) {
#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    (void)A0, (void)W, (void)n, (void)sorted_copy, (void)items, (void)slack;
    GTEST_SKIP() << "no host LAPACKE reference in this build";
#else
    using Real = typename batchlas::base_type<Scalar>::type;
    const int batch = A0.batch_size();
    const std::vector<int> picked = items.empty() ? A0.default_items() : std::vector<int>(items.begin(), items.end());
    std::vector<Real> got(W.begin(), W.begin() + static_cast<std::ptrdiff_t>(n) * batch);
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch));
    double scale = 0;
    for (int b : picked) {
        ASSERT_TRUE(batchlas::verify::eigenvalues(A0, b, ref[static_cast<std::size_t>(b)])) << "LAPACKE reference failed, item " << b;
        for (double l : ref[static_cast<std::size_t>(b)]) scale = batchlas::verify::nanmax(scale, std::fabs(l));
        if (sorted_copy) std::sort(got.begin() + static_cast<std::ptrdiff_t>(b) * n, got.begin() + static_cast<std::ptrdiff_t>(b + 1) * n);
    }
    const batchlas::VectorView<Real> w(got.data(), n, batch);
    const double err = batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>::values_error(w, ref, scale, picked);
    if (slack) EXPECT_VERIFY_SLACK(Scalar, batchlas::verify::Check::values, n, err, *slack);
    else EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, n, err);
#endif
}

/// Descending singular values of the column-major r x c @p a (ld = r) by one-sided (Hestenes)
/// Jacobi in double: the vendor-free oracle, plain host code that shares nothing with a BatchLAS
/// route or with LAPACKE. A wide input is solved transposed; sweeps stop when every column pair is
/// orthogonal to 4 eps, or after 64 sweeps (false).
template <typename D>
bool jacobi_singular_values(int r, int c, std::vector<D> a, std::vector<double>& s) {
    if (r < c) {
        std::vector<D> t(a.size());
        for (int j = 0; j < c; ++j)
            for (int i = 0; i < r; ++i) {
                const D x = a[std::size_t(j) * r + i];
                if constexpr (std::is_same_v<D, double>) t[std::size_t(i) * c + j] = x;
                else t[std::size_t(i) * c + j] = std::conj(x);
            }
        a.swap(t);
        std::swap(r, c);
    }
    auto col = [&](int j) { return a.data() + std::size_t(j) * r; };
    bool converged = c <= 1;
    for (int sweep = 0; sweep < 64 && !converged; ++sweep) {
        converged = true;
        for (int p = 0; p < c - 1; ++p)
            for (int q = p + 1; q < c; ++q) {
                double alpha = 0, beta = 0;
                D gamma{};
                for (int i = 0; i < r; ++i) {
                    alpha += std::norm(col(p)[i]);
                    beta += std::norm(col(q)[i]);
                    if constexpr (std::is_same_v<D, double>) gamma += col(p)[i] * col(q)[i];
                    else gamma += std::conj(col(p)[i]) * col(q)[i];
                }
                const double g = std::abs(gamma);
                if (g == 0 || g <= 4 * std::numeric_limits<double>::epsilon() * std::sqrt(alpha * beta)) continue;
                converged = false;
                const D phase = gamma / g;  // h_q = g_q / phase makes g_p^H h_q = |gamma| real
                const double zeta = (beta - alpha) / (2 * g);
                const double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                const double cs = 1 / std::sqrt(1 + t * t), sn = cs * t;
                for (int i = 0; i < r; ++i) {
                    const D x = col(p)[i], y = col(q)[i] / phase;
                    col(p)[i] = cs * x - sn * y;
                    col(q)[i] = sn * x + cs * y;
                }
            }
    }
    s.assign(std::size_t(c), 0.0);
    for (int j = 0; j < c; ++j) {
        double nrm = 0;
        for (int i = 0; i < r; ++i) nrm += std::norm(col(j)[i]);
        s[std::size_t(j)] = std::sqrt(nrm);
    }
    std::sort(s.begin(), s.end(), std::greater<double>());
    return converged;
}

/// s (min(m, n) values per item, packed) against the singular values of A0 in double: LAPACKE's,
/// or jacobi_singular_values' in a build without LAPACKE. max |s - s_ref| / max s_ref at
/// Check::values over @p items (default: every item).
template <typename Scalar>
void expect_singular_values_match_lapacke(const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& A0,
                                          const batchlas::UnifiedVector<typename batchlas::base_type<Scalar>::type>& s,
                                          std::span<const int> items = {}) {
    using Real = typename batchlas::base_type<Scalar>::type;
    const int k = std::min(A0.rows(), A0.cols()), batch = A0.batch_size();
    const std::vector<int> picked = items.empty() ? A0.all_items() : std::vector<int>(items.begin(), items.end());
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch));
    double smax = 0;
    for (int b : picked) {
#if BATCHLAS_VERIFY_HAVE_LAPACKE
        ASSERT_TRUE(batchlas::verify::singular_values(A0, b, ref[static_cast<std::size_t>(b)])) << "LAPACKE reference failed, item " << b;
#else
        using P = typename batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>::promoted_type;
        ASSERT_TRUE(jacobi_singular_values<P>(A0.rows(), A0.cols(), batchlas::verify::detail::item_vector<P>(A0, b),
                                              ref[static_cast<std::size_t>(b)]))
            << "host Jacobi reference did not converge, item " << b;
#endif
        if (k > 0) smax = batchlas::verify::nanmax(smax, ref[static_cast<std::size_t>(b)].front());
    }
    const batchlas::VectorView<Real> sv(const_cast<Real*>(s.data()), k, batch);
    const double err = batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>::values_error(sv, ref, smax, picked);
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, k, err);
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
    const double err = batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>::values_error(w, ref, scale, w.all_items());
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::values, n, err);
}

/// Upper-bidiagonal (d, e) inputs, packed per batch item: d has n values, e has n-1.
struct Bidiagonal {
    std::vector<double> d, e;
};

/// Positive d and e in [0.3, 1.7] (e halved), both graded by dscale^i: a far easier class than gebrd emits.
inline Bidiagonal random_positive_bidiagonal(int n, int batch, unsigned seed, double dscale = 1.0) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.3, 1.7);
    std::vector<double> dh(static_cast<size_t>(n) * batch), eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) dh[static_cast<size_t>(b) * n + i] = dist(rng) * std::pow(dscale, i);
        for (int i = 0; i < n - 1; ++i) eh[static_cast<size_t>(b) * (n - 1) + i] = dist(rng) * 0.5 * std::pow(dscale, i);
    }
    return {std::move(dh), std::move(eh)};
}

/// Mixed signs, exact zeros (deflation) and entries over six decades (1e-3 .. 1e3), as gebrd emits. It
/// caught bdsqr stagnating when gesvd_blocked was first wired onto it.
inline Bidiagonal hostile_bidiagonal(int n, int batch, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> mag(-3.0, 3.0);
    std::uniform_int_distribution<int> sign(0, 1);
    std::uniform_int_distribution<int> zero(0, 9);
    std::vector<double> dh(static_cast<size_t>(n) * batch);
    std::vector<double> eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            const double m10 = std::pow(10.0, mag(rng));
            double v = m10 * (sign(rng) ? 1.0 : -1.0);
            if (zero(rng) == 0) v = 0.0;
            dh[static_cast<size_t>(b) * n + i] = v;
        }
        for (int i = 0; i < n - 1; ++i) {
            const double m10 = std::pow(10.0, mag(rng));
            double v = m10 * (sign(rng) ? 1.0 : -1.0);
            if (zero(rng) == 0) v = 0.0;
            eh[static_cast<size_t>(b) * (n - 1) + i] = v;
        }
    }
    return {std::move(dh), std::move(eh)};
}

}  // namespace test_utils
