// Tests for bdsqr, the bidiagonal QR iteration.
//
// This file exists because bdsqr had ZERO callers anywhere in the tree (420
// lines of dead code). It is the accurate bidiagonal solver that gesvd should
// have been using instead of forming the tridiagonal of B^T B, so before wiring
// it into gesvd_blocked it needs evidence that it works at all -- dead code
// carries none.
//
// The reference is LAPACKE ?gesdd of the same bidiagonal matrix in double
// (batchlas::verify), not a second call to anything in this library.

#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"
#include "eigen_verify.hh"

#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <random>
#include <tuple>
#include <vector>

using namespace batchlas;

// Every typed config below is Backend::CUDA: without it the suite is empty and does not compile.
#if BATCHLAS_HAS_CUDA_BACKEND

namespace {

template <typename T, Backend B>
struct BdsqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <template <typename, Backend> class Config>
struct bdsqr_types {
    using tuple_type = decltype(std::tuple_cat(
#if BATCHLAS_HAS_CUDA_BACKEND
        std::tuple<Config<float, Backend::CUDA>,
                   Config<double, Backend::CUDA>>{},
#endif
        std::tuple<>{}));
    using type = typename test_utils::tuple_to_types<tuple_type>::type;
};

using BdsqrTestTypes = typename bdsqr_types<BdsqrConfig>::type;

template <typename Config>
class BdsqrTest : public test_utils::BatchLASTest<Config> {
protected:
    using Scalar = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    // Runs bdsqr on (d,e) and checks singular values against LAPACKE, plus -- when vectors are
    // requested -- that U S Vh reconstructs B and that U, Vh are orthogonal. U and Vh start at the
    // identity, so they accumulate the rotations of B itself.
    void run_and_check(const std::vector<double>& dh, const std::vector<double>& eh, int n, int batch,
                       bool vectors, const char* label) {
        using MV = MatrixView<Scalar, MatrixFormat::Dense>;
        SCOPED_TRACE(std::string(label) + " n=" + std::to_string(n));
        auto& ctx = *this->ctx;
        UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch);
        UnifiedVector<Scalar> e(static_cast<size_t>(std::max(1, n - 1)) * batch);
        Matrix<Scalar> B0 = Matrix<Scalar>::Zeros(n, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(dh[static_cast<size_t>(b) * n + i]);
                B0.at(i, i, b) = d[static_cast<size_t>(b) * n + i];
            }
            for (int i = 0; i < n - 1; ++i) {
                e[static_cast<size_t>(b) * (n - 1) + i] = static_cast<Scalar>(eh[static_cast<size_t>(b) * (n - 1) + i]);
                B0.at(i, i + 1, b) = e[static_cast<size_t>(b) * (n - 1) + i];
            }
        }

        VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
        VectorView<Scalar> ev(e.to_span(), std::max(0, n - 1), batch, 1, std::max(1, n - 1));
        UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch);

        const size_t ws_bytes = bdsqr_buffer_size<Scalar>(ctx, dv, ev, s.to_span());
        UnifiedVector<std::byte> ws(ws_bytes);

        auto U = Matrix<Scalar>::Identity(n, batch);
        auto Vh = Matrix<Scalar>::Identity(n, batch);
        if (vectors) {
            (void)bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), U, Vh, true);
        } else {
            (void)bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), true);
        }
        ctx.wait_and_throw();

        const auto items = B0.all_items();
        const VectorView<Scalar> sv(s.to_span(), n, batch, 1, n);
        if (vectors) {
            EXPECT_VERIFY(Scalar, batchlas::verify::Check::factorization, n,
                          MV::svd_residual(ctx, B0, U, sv, Vh, items));
            EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, U, items))
                << "U";
            EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, Vh, items))
                << "Vh (square: orthonormal columns iff orthonormal rows)";
        }

        test_utils::expect_singular_values_match_lapacke(B0, s, items);
    }

    void check(int n, int batch, unsigned seed, bool vectors, double dscale = 1.0) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(0.3, 1.7);
        std::vector<double> dh(static_cast<size_t>(n) * batch), eh(static_cast<size_t>(std::max(0, n - 1)) * batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) dh[static_cast<size_t>(b) * n + i] = dist(rng) * std::pow(dscale, i);
            for (int i = 0; i < n - 1; ++i) eh[static_cast<size_t>(b) * (n - 1) + i] = dist(rng) * 0.5 * std::pow(dscale, i);
        }
        run_and_check(dh, eh, n, batch, vectors, "random");
    }

// The generator above produces only POSITIVE d and e in [0.3, 1.7], which is a
// far easier class than anything gebrd actually emits. Real bidiagonal factors
// carry mixed signs, exact zeros (deflation), and entries spanning many orders
// of magnitude. This variant covers that; it is what caught bdsqr stagnating
// when gesvd_blocked was first wired onto it.
    void check_hostile(int n, int batch, unsigned seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> mag(-3.0, 3.0);   // 1e-3 .. 1e3
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
        run_and_check(dh, eh, n, batch, /*vectors=*/false, "hostile");
    }
};

TYPED_TEST_SUITE(BdsqrTest, BdsqrTestTypes);

TYPED_TEST(BdsqrTest, ValuesOnly) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 100u + n, /*vectors=*/false);
}

TYPED_TEST(BdsqrTest, WithVectors) {
    for (int n : {2, 4, 9, 16, 33, 64}) this->check(n, 3, 200u + n, /*vectors=*/true);
}

// Graded: the case where forming B^T B loses the small singular values and a
// direct bidiagonal method should not.
TYPED_TEST(BdsqrTest, Graded) {
    this->check(32, 2, 301u, /*vectors=*/true, /*dscale=*/0.8);
    this->check(64, 2, 302u, /*vectors=*/false, /*dscale=*/0.85);
}

TYPED_TEST(BdsqrTest, MixedSignsZerosAndWideRange) {
    for (int n : {8, 32, 64}) this->check_hostile(n, 16, 900u + n);
}

// A graded bidiagonal scaled by 2^(max_exponent/2): every d_i^2 overflows, so a shift formed from
// squared entries is inf. The scale is a power of two, so dividing s by it afterwards is exact and
// the checks run on the unscaled matrix.
TYPED_TEST(BdsqrTest, GradedNearOverflow) {
    using Scalar = typename TestFixture::Scalar;
    using MV = MatrixView<Scalar, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::B;
    auto& ctx = *this->ctx;
    const int n = 32, batch = 2;
    const Scalar scale = std::ldexp(Scalar(1), std::numeric_limits<Scalar>::max_exponent / 2);

    auto A0 = Matrix<Scalar>::Zeros(n, n, batch);
    UnifiedVector<Scalar> d(static_cast<size_t>(n) * batch), e(static_cast<size_t>(n - 1) * batch);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> dh(n), eh(n - 1);
        for (int i = 0; i < n; ++i) {
            dh[i] = static_cast<Scalar>((1.7 - 0.9 * ((i * 7 + b) % 5) / 4.0) * std::pow(0.8, i));
            if (i < n - 1) eh[i] = static_cast<Scalar>(0.5 * (0.6 + 0.2 * ((i * 3 + b) % 4)) * std::pow(0.8, i));
        }
        for (int i = 0; i < n; ++i) {
            A0.view()(i, i, b) = static_cast<Scalar>(dh[i]);
            d[static_cast<size_t>(b) * n + i] = static_cast<Scalar>(dh[i]) * scale;
            if (i < n - 1) {
                A0.view()(i, i + 1, b) = static_cast<Scalar>(eh[i]);
                e[static_cast<size_t>(b) * (n - 1) + i] = static_cast<Scalar>(eh[i]) * scale;
            }
        }
    }
    VectorView<Scalar> dv(d.to_span(), n, batch, 1, n);
    VectorView<Scalar> ev(e.to_span(), n - 1, batch, 1, n - 1);
    UnifiedVector<Scalar> s(static_cast<size_t>(n) * batch, Scalar(0));
    UnifiedVector<std::byte> ws(bdsqr_buffer_size<Scalar>(ctx, dv, ev, s.to_span()));
    auto U = Matrix<Scalar>::Identity(n, batch);
    auto Vh = Matrix<Scalar>::Identity(n, batch);
    (void)bdsqr<B, Scalar>(ctx, dv, ev, s.to_span(), ws.to_span(), U, Vh, true);
    ctx.wait_and_throw();
    for (auto& x : s) x /= scale;

    const VectorView<Scalar> sv(s.to_span(), n, batch, 1, n);
    test_utils::expect_singular_values_match_lapacke(A0, s, A0.all_items());
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::factorization, n,
                  MV::svd_residual(ctx, A0, U, sv, Vh, A0.all_items()));
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, U, U.all_items()));
    // Vh is square, so orthonormal columns mean orthonormal rows.
    EXPECT_VERIFY(Scalar, batchlas::verify::Check::orthogonality_rotations, n, MV::orthogonality(ctx, Vh, Vh.all_items()));
}


} // namespace
#endif  // BATCHLAS_HAS_CUDA_BACKEND
