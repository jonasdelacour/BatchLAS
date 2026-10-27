// SPDX-License-Identifier: MIT
#pragma once
// Test inputs on the library's own generators (MatrixView::fill_*).
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cmath>
#include <type_traits>

namespace test_support {

/// The graded cases' condition number: 1e6 in double precision, 1e3 in single (potrs, potri, syfn).
template <typename T>
constexpr double cond_for() {
    return std::is_same_v<batchlas::float_t<T>, double> ? 1e6 : 1e3;
}

/// n x n HPD items, packed, one per item: graded to condition `cond` (fill_graded_hermitian) for cond > 1,
/// else diagonally dominant (fill_random_spd). For identical items draw one; test_support::store repeats it.
template <typename T>
batchlas::Matrix<T> hpd(batchlas::Queue& q, int n, int batch, double cond, unsigned seed) {
    batchlas::Matrix<T> m(n, n, batch);
    if (cond > 1) (void)m.view().fill_graded_hermitian(q, std::log10(cond), seed);
    else (void)m.view().fill_random_spd(q, seed);
    q.wait();
    return m;
}

}  // namespace test_support
