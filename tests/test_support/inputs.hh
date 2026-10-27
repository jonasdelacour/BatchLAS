// SPDX-License-Identifier: MIT
#pragma once
// Test inputs on the library's own generators (MatrixView::fill_*).
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

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

/// The Cholesky factor of a's uplo triangle in place, through the public potrf; info gets one
/// entry per item. False when the build has no potrf route for the shape.
template <batchlas::Backend B, typename T>
bool potrf_in_place(batchlas::Queue& q, const batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>& a,
                    batchlas::Uplo uplo, batchlas::Span<int32_t> info) {
    try {
        const std::size_t need = batchlas::potrf_buffer_size<B, T>(q, a, uplo);
        batchlas::UnifiedVector<std::byte> ws(std::max<std::size_t>(need, 1));
        (void)batchlas::potrf<B, T>(q, a, uplo, batchlas::Span<std::byte>(ws.data(), need), info);
    } catch (const batchlas::NoRouteError&) {
        return false;
    }
    q.wait();
    return true;
}

}  // namespace test_support

/// NETLIB serves potrf, trsm and the other dense factorizations only through LAPACKE/CBLAS, so a
/// vendor-free build has no route there: call from SetUp() of a typed fixture over backend BK.
#define TEST_SUPPORT_SKIP_NETLIB_WITHOUT_LAPACK(BK)                                                 \
    do {                                                                                            \
        if ((BK) == batchlas::Backend::NETLIB && !(BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS))     \
            GTEST_SKIP() << "NETLIB without LAPACKE/CBLAS has no dense factorization route";       \
    } while (0)
