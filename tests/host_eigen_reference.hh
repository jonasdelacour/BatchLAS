// SPDX-License-Identifier: MIT
#pragma once
// Host eigenvalue references in double: plain C++ with no LAPACKE and no BatchLAS route, so a test
// in a build without LAPACKE (vendor-free) checks a BatchLAS solver against code it shares nothing
// with. eigen_verify.hh falls back to these; the device tests include this header alone.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace test_utils {

/// Ascending eigenvalues of the symmetric tridiagonal (d, e) by Sturm-count bisection in double, to
/// eps ||T|| / 4 absolute. The counts are backward stable (LAPACK dlaebz's pivmin guard).
inline std::vector<double> host_tridiagonal_eigenvalues(const std::vector<double>& d, const std::vector<double>& e) {
    const int n = static_cast<int>(d.size());
    std::vector<double> w(static_cast<std::size_t>(n));
    if (n == 0) return w;
    double lo = d[0], hi = d[0], emax2 = 0;
    for (int i = 0; i < n; ++i) {
        const double r = (i > 0 ? std::fabs(e[i - 1]) : 0.0) + (i + 1 < n ? std::fabs(e[i]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
        if (i + 1 < n) emax2 = std::max(emax2, e[i] * e[i]);
    }
    const double pivmin = std::numeric_limits<double>::min() * std::max(1.0, emax2);
    const double tol = std::max(0.25 * std::numeric_limits<double>::epsilon() * std::max(-lo, hi), 4 * pivmin);
    lo -= tol;
    hi += tol;
    const auto below = [&](double x) {  // eigenvalues < x
        int c = 0;
        double q = 1;
        for (int i = 0; i < n; ++i) {
            q = d[i] - x - (i > 0 ? e[i - 1] * e[i - 1] / q : 0.0);
            if (std::fabs(q) < pivmin) q = -pivmin;
            c += q < 0;
        }
        return c;
    };
    for (int k = 0; k < n; ++k) {
        double a = lo, b = hi;  // below(a) <= k < below(b)
        for (int it = 0; it < 256 && b - a > tol; ++it) {
            const double m = 0.5 * (a + b);
            (below(m) > k ? b : a) = m;
        }
        w[static_cast<std::size_t>(k)] = 0.5 * (a + b);
    }
    return w;
}

/// Ascending eigenvalues of the n x n Hermitian @p a (column-major; the lower triangle is read, as
/// LAPACKE 'L' reads it), in double: Householder reduction to a real tridiagonal (|e| keeps the
/// spectrum), then host_tridiagonal_eigenvalues. D is double or std::complex<double>.
template <typename D>
std::vector<double> host_hermitian_eigenvalues(int n, std::vector<D> a) {
    const auto cj = [](D x) {
        if constexpr (std::is_same_v<D, double>) return x;
        else return std::conj(x);
    };
    const auto A = [&](int i, int j) -> D& { return a[static_cast<std::size_t>(j) * n + i]; };
    for (int j = 0; j < n; ++j) {
        A(j, j) = std::real(A(j, j));
        for (int i = j + 1; i < n; ++i) A(j, i) = cj(A(i, j));
    }
    std::vector<double> d(static_cast<std::size_t>(n)), e(static_cast<std::size_t>(std::max(n - 1, 0)));
    std::vector<D> v(static_cast<std::size_t>(n)), q(static_cast<std::size_t>(n));
    for (int k = 0; k + 1 < n; ++k) {
        double xnorm = 0;
        for (int i = k + 1; i < n; ++i) xnorm += std::norm(A(i, k));
        xnorm = std::sqrt(xnorm);
        e[static_cast<std::size_t>(k)] = xnorm;
        if (k + 2 == n || xnorm == 0) continue;  // a scalar or zero column needs no reflector
        const D x0 = A(k + 1, k);
        const D alpha = (std::abs(x0) == 0 ? D(1) : x0 / std::abs(x0)) * -xnorm;
        double vnorm = 0;
        for (int i = k + 1; i < n; ++i) {
            v[i] = A(i, k) - (i == k + 1 ? alpha : D(0));
            vnorm += std::norm(v[i]);
        }
        vnorm = std::sqrt(vnorm);
        for (int i = k + 1; i < n; ++i) v[i] /= vnorm;
        // H B H = B - 2 v q^H - 2 q v^H, with p = B v, q = p - (v^H p) v (H = I - 2 v v^H).
        D c = 0;
        for (int i = k + 1; i < n; ++i) {
            D p = 0;
            for (int j = k + 1; j < n; ++j) p += A(i, j) * v[j];
            q[i] = p;
            c += cj(v[i]) * p;
        }
        for (int i = k + 1; i < n; ++i) q[i] -= c * v[i];
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < n; ++i) A(i, j) -= 2.0 * (v[i] * cj(q[j]) + q[i] * cj(v[j]));
    }
    for (int i = 0; i < n; ++i) d[static_cast<std::size_t>(i)] = std::real(A(i, i));
    return host_tridiagonal_eigenvalues(d, e);
}

}  // namespace test_utils
