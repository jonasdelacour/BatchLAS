// SPDX-License-Identifier: MIT
#pragma once
// Test inputs on the library's own generators (MatrixView::fill_*), and a host-side input hash.
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

/// A well-spread value in [-1, 1] per (item, element, salt), a splitmix-style hash on the host: distinct
/// pivots in every item of the getrf and getrs mask tests.
inline double spread(std::size_t item, std::size_t e, int salt) {
    std::uint64_t h = (item * 0x9E3779B97F4A7C15ull) ^ (e * 0xC2B2AE3D27D4EB4Full) ^ std::uint64_t(salt * 7919);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return double(int(h % 2001) - 1000) / 1000.0;
}

/// The graded cases' condition number: 1e6 in double precision, 1e3 in single (potrs, potri, syfn).
template <typename Scalar>
constexpr double cond_for() {
    return std::is_same_v<batchlas::float_t<Scalar>, double> ? 1e6 : 1e3;
}

/// n x n HPD items, packed, one per item: graded to condition `cond` (fill_graded_hermitian) for cond > 1,
/// else diagonally dominant (fill_random_spd). For identical items draw one; test_support::store repeats it.
template <typename Scalar>
batchlas::Matrix<Scalar> hpd(batchlas::Queue& q, int n, int batch, double cond, unsigned seed) {
    batchlas::Matrix<Scalar> m(n, n, batch);
    if (cond > 1) (void)m.view().fill_graded_hermitian(q, std::log10(cond), seed);
    else (void)m.view().fill_random_spd(q, seed);
    q.wait();
    return m;
}

/// The Cholesky factor of a's uplo triangle in place, through the public potrf; info gets one
/// entry per item. False when the build has no potrf route for the shape.
template <batchlas::Backend B, typename Scalar>
bool potrf_in_place(batchlas::Queue& q, const batchlas::MatrixView<Scalar, batchlas::MatrixFormat::Dense>& a,
                    batchlas::Uplo uplo, batchlas::Span<int32_t> info) {
    try {
        const std::size_t need = batchlas::potrf_buffer_size<B, Scalar>(q, a, uplo);
        batchlas::UnifiedVector<std::byte> ws(std::max<std::size_t>(need, 1));
        (void)batchlas::potrf<B, Scalar>(q, a, uplo, batchlas::Span<std::byte>(ws.data(), need), info);
        // Before ws dies: a host queue's kernels still write it after the free (a CUDA-default
        // context unmaps the pages: SIGSEGV), where a CUDA queue's free happens to synchronise.
        q.wait();
    } catch (const batchlas::NoRouteError&) {
        return false;
    }
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
