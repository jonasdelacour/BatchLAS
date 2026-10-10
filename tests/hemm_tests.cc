#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <complex>
#include <cstdlib>
#include <string>

#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>

using namespace batchlas;

template <typename T, Backend B>
struct HemmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// hemm exists only for complex scalars: for a real matrix "Hermitian" and
// "symmetric" are the same statement, and BLAS has no real ?hemm.
using HemmTestTypes = typename test_utils::backend_types_complex<HemmConfig>::type;

template <typename Config>
class HemmTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(HemmTest, HemmTestTypes);

// HEMM must not reference the opposite triangle of A, and must take the
// diagonal to be real whatever imaginary part is stored there -- A = A^H says
// both, so neither is storage the caller has to supply.
//
// Building A already Hermitian and validating against a gemm on that same A
// cannot distinguish a real hemm from a plain gemm: the mirrored half already
// holds the conjugate and the diagonal is already real, so an implementation
// that reads the wrong triangle reads the right values anyway. This test
// poisons exactly the storage HEMM is forbidden to read -- the unreferenced
// triangle with a value nothing else could produce, and the diagonal with a
// nonzero imaginary part. The library reads A as a Hermitian matrix from the
// named triangle of the poisoned storage, with a real diagonal, as BLAS does.
//
// The shapes are ragged on purpose. The CUDA backend materialises the triangle
// into packed scratch a 32x32 tile at a time, so what matters is the sizes
// where that tiling does not divide evenly and where the scratch's leading
// dimension is not the matrix width. On CUDA the sweep then runs once per route
// with the choice pinned, because the two routes are separate implementations
// and which one a given shape picks is a tuning decision that is free to
// change; pinning keeps both covered whatever the heuristic later decides.
TYPED_TEST(HemmTest, IgnoresUnreferencedTriangleAndImaginaryDiagonal) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    struct Shape {
        int n;      // order of A
        int m;      // the other dimension of B and C
        int batch;
    };
    const Shape shapes[] = {{16, 24, 2}, {33, 61, 5}, {77, 13, 1}, {96, 64, 3}, {129, 48, 2}};

    const T alpha = T(1.25, -0.75);
    const T beta = T(-0.5, 0.25);

    auto sweep = [&](const char* route) {
        for (const auto& shape : shapes) {
            const int n = shape.n;

            for (auto side : {Side::Left, Side::Right}) {
                for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                    const int rows = side == Side::Left ? n : shape.m;
                    const int cols = side == Side::Left ? shape.m : n;

                    auto A_clean = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, shape.batch, 17);
                    A_clean.view().hermitize(*(this->ctx), uplo).wait();
                    auto A_poisoned = A_clean.clone();

                    auto B = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, shape.batch, 19);
                    auto C0 = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, shape.batch, 23);
                    this->ctx->wait();

                    // Use the element accessor, not raw data() arithmetic: the
                    // storage has its own leading dimension and batch stride, and
                    // assuming ld == n silently writes into the referenced triangle.
                    for (int b = 0; b < shape.batch; ++b) {
                        for (int col = 0; col < n; ++col) {
                            for (int row = 0; row < n; ++row) {
                                const bool referenced =
                                    (uplo == Uplo::Lower) ? (row > col) : (row < col);
                                if (referenced) {
                                    continue;
                                }
                                if (row == col) {
                                    A_poisoned(row, col, b) =
                                        T(A_clean(row, col, b).real(), real_t(555));
                                } else {
                                    A_poisoned(row, col, b) = T(1000, -777);
                                }
                            }
                        }
                    }
                    this->ctx->wait();

                    Matrix<T, MatrixFormat::Dense> C(rows, cols, shape.batch);
                    MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();

                    hemm(*(this->ctx), A_poisoned.view(), B.view(), C.view(),
                         {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();

                    const auto shape_a = uplo == Uplo::Lower ? batchlas::verify::Shape::hermitian_lower
                                                             : batchlas::verify::Shape::hermitian_upper;
                    const auto general = batchlas::verify::Shape::general;
                    const double err =
                        side == Side::Left
                            ? batchlas::verify::gemm_backward_error(A_poisoned.view(), shape_a, Transpose::NoTrans, B.view(), general,
                                                                    Transpose::NoTrans, C0.view(), C.view(), general, batchlas::verify::up(alpha),
                                                                    batchlas::verify::up(beta), batchlas::verify::all_items(shape.batch))
                            : batchlas::verify::gemm_backward_error(B.view(), general, Transpose::NoTrans, A_poisoned.view(), shape_a,
                                                                    Transpose::NoTrans, C0.view(), C.view(), general, batchlas::verify::up(alpha),
                                                                    batchlas::verify::up(beta), batchlas::verify::all_items(shape.batch));
                    EXPECT_VERIFY(T, batchlas::verify::Check::blas, n, err)
                        << "hemm read storage it must not touch: " << route << " route, n=" << n
                        << ", side=" << (side == Side::Left ? "Left" : "Right")
                        << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper");
                }
            }
        }
    };

    sweep("default");

    if constexpr (TestFixture::BackendType == Backend::CUDA) {
        {
            ScopedEnvVar route("BATCHLAS_EXPAND_ROUTE", "expand");
            sweep("expansion");
        }
        {
            ScopedEnvVar route("BATCHLAS_EXPAND_ROUTE", "loop");
            sweep("vendor-loop");
        }
    }
}

// The option struct must mean exactly what the positional call means. A bare
// `{}` in particular has to reach the option-struct overload -- the positional
// spelling takes eight arguments, so a five-argument call that resolved to it
// would not compile, but a defaulted option struct that disagreed with the
// positional defaults would silently compute something else.
TYPED_TEST(HemmTest, OptionStructMatchesPositional) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 32;
    const int batch = 3;

    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 5);
    auto B = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 7);

    Matrix<T, MatrixFormat::Dense> C_defaults(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_defaults_pos(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named_pos(n, n, batch);
    for (auto* C : {&C_defaults, &C_defaults_pos, &C_named, &C_named_pos}) {
        (void)C->view().fill_zeros(*(this->ctx));
    }
    this->ctx->wait();

    (void)hemm(*(this->ctx), A.view(), B.view(), C_defaults.view(), {});
    (void)hemm<Ba, T>(*(this->ctx), A.view(), B.view(), C_defaults_pos.view(),
                T(1), T(0), Side::Left, Uplo::Lower);

    (void)hemm(*(this->ctx), A.view(), B.view(), C_named.view(),
         {.alpha = T(1.5, 0.25), .side = Side::Right, .uplo = Uplo::Upper});
    (void)hemm<Ba, T>(*(this->ctx), A.view(), B.view(), C_named_pos.view(),
                T(1.5, 0.25), T(0), Side::Right, Uplo::Upper);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_EQ(C_defaults(i, j, b), C_defaults_pos(i, j, b))
                    << "hemm defaults at (" << i << "," << j << ") batch " << b;
                ASSERT_EQ(C_named(i, j, b), C_named_pos(i, j, b))
                    << "hemm designated initialisers at (" << i << "," << j << ") batch " << b;
            }
        }
    }
}

// x^H A x is real for every x when A is Hermitian. That is a property of the
// product rather than of the operand, so it catches a mirror that dropped the
// conjugate -- which would make the expanded matrix complex-symmetric instead,
// and x^H A x complex.
TYPED_TEST(HemmTest, QuadraticFormIsReal) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int n = 64;
    const int batch = 4;

    for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
        auto A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 11);
        auto X = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 13);
        Matrix<T, MatrixFormat::Dense> AX(n, n, batch);
        Matrix<T, MatrixFormat::Dense> XhAX(n, n, batch);
        (void)AX.view().fill_zeros(*(this->ctx));
        (void)XhAX.view().fill_zeros(*(this->ctx));
        this->ctx->wait();

        hemm(*(this->ctx), A.view(), X.view(), AX.view(), {.uplo = uplo}).wait();
        gemm(*(this->ctx), X.view(), AX.view(), XhAX.view(),
             {.transA = Transpose::ConjTrans}).wait();

        // Imaginary diagonal over the largest denominator of X^H (AX): two products of length n, k = 2n.
        const auto general = batchlas::verify::Shape::general;
        const auto shape_a = uplo == Uplo::Lower ? batchlas::verify::Shape::hermitian_lower : batchlas::verify::Shape::hermitian_upper;
        for (int b = 0; b < batch; ++b) {
            double worst = 0;
            for (int i = 0; i < n; ++i) worst = batchlas::verify::nanmax(worst, std::abs(double(XhAX(i, i, b).imag())));
            const double scale = batchlas::verify::gemm_max_denominator(X.view(), general, Transpose::ConjTrans, AX.view(), general,
                                                                        Transpose::NoTrans, XhAX.view(), 1.0, 0.0, b);
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, 2 * n, worst / scale)
                << "x^H A x is not real, so the expansion lost the conjugate: uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                << ", batch=" << b;
        }
        EXPECT_VERIFY(T, batchlas::verify::Check::blas, n,
                      batchlas::verify::gemm_backward_error(A.view(), shape_a, Transpose::NoTrans, X.view(), general, Transpose::NoTrans, AX.view(),
                                                            AX.view(), general, 1.0, 0.0, batchlas::verify::all_items(batch)))
            << "uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper");
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
