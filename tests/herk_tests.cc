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
struct HerkConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// herk exists only for complex scalars; its real spelling is syrk.
using HerkTestTypes = typename test_utils::backend_types_complex<HerkConfig>::type;

template <typename Config>
class HerkTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(HerkTest, HerkTestTypes);

// HERK owns exactly one triangle of C. The opposite one is not part of the
// result: it is neither read through beta nor written, and neither is the
// imaginary part of the diagonal, because C = C^H forces that to zero whatever
// the caller stored there.
//
// Checking only that the referenced triangle holds the right numbers would miss
// both halves of that. This test poisons the storage HERK may not touch -- the
// unreferenced triangle and the diagonal's imaginary part, each with a value
// nothing else could produce -- and then asserts three separate things: the
// referenced triangle matches alpha A A^H + beta C0 in double (C0 clean, with a
// real diagonal), the poison is still bit-for-bit intact afterwards, and the
// diagonal came out real.
//
// The shapes are ragged on purpose: the GEMM route folds an n x n product into
// C with a tiled elementwise kernel, so what matters is the sizes where the
// tiling does not divide evenly and where the product's leading dimension is
// not n. On CUDA the sweep runs once per route with the choice pinned, because
// the two routes are separate implementations and which one a shape picks is a
// tuning decision free to change -- left to the default, every shape here would
// take the loop and the fold would never be reached.
TYPED_TEST(HerkTest, IgnoresUnreferencedTriangleOfC) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    struct Shape {
        int n;
        int k;
        int batch;
    };
    const Shape shapes[] = {{16, 24, 2}, {33, 61, 5}, {77, 13, 1}, {96, 64, 3}, {129, 48, 2}};

    struct Scaling {
        real_t alpha;
        real_t beta;
    };
    // beta = 0 is its own path: C is not an input at all there.
    const Scaling scalings[] = {{real_t(1.25), real_t(-0.5)}, {real_t(0.75), real_t(0)}};

    const T poison = T(1000, -777);
    const real_t diagonal_poison = real_t(555);

    auto sweep = [&](const char* route) {
        for (const auto& shape : shapes) {
            const int n = shape.n;
            const int k = shape.k;
            for (auto trans : {Transpose::NoTrans, Transpose::ConjTrans}) {
                const int a_rows = trans == Transpose::NoTrans ? n : k;
                const int a_cols = trans == Transpose::NoTrans ? k : n;

                auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 17);
                auto C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, shape.batch, 23);
                this->ctx->wait();
                // A Hermitian C has a real diagonal: the reference scales its real part by beta.
                for (int b = 0; b < shape.batch; ++b)
                    for (int d = 0; d < n; ++d) C0(d, d, b) = T(C0(d, d, b).real(), real_t(0));

                for (const auto& scaling : scalings) {
                    for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                        auto C = C0.clone();
                        this->ctx->wait();

                        // Use the element accessor, not raw data() arithmetic:
                        // the storage has its own leading dimension and batch
                        // stride, and assuming ld == n silently poisons the
                        // referenced triangle instead.
                        for (int b = 0; b < shape.batch; ++b) {
                            for (int col = 0; col < n; ++col) {
                                for (int row = 0; row < n; ++row) {
                                    const bool referenced =
                                        (uplo == Uplo::Lower) ? (row > col) : (row < col);
                                    if (referenced) {
                                        continue;
                                    }
                                    if (row == col) {
                                        C(row, col, b) =
                                            T(C0(row, col, b).real(), diagonal_poison);
                                    } else {
                                        C(row, col, b) = poison;
                                    }
                                }
                            }
                        }
                        this->ctx->wait();

                        herk(*(this->ctx), A.view(), C.view(),
                             {.alpha = scaling.alpha,
                              .beta = scaling.beta,
                              .uplo = uplo,
                              .trans = trans}).wait();

                        const auto triangle = uplo == Uplo::Lower ? batchlas::verify::Shape::lower : batchlas::verify::Shape::upper;
                        const auto general = batchlas::verify::Shape::general;
                        const double err = batchlas::verify::gemm_backward_error(
                            A.view(), general, trans, A.view(), general,
                            trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans, C0.view(), C.view(), triangle,
                            batchlas::verify::up(T(scaling.alpha)), batchlas::verify::up(T(scaling.beta)),
                            batchlas::verify::all_items(shape.batch));
                        EXPECT_VERIFY(T, batchlas::verify::Check::blas, k, err)
                            << "herk read storage it must not touch: " << route << " route, n=" << n << ", k=" << k
                            << ", trans=" << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper") << ", beta=" << scaling.beta;

                        for (int b = 0; b < shape.batch; ++b) {
                            for (int j = 0; j < n; ++j) {
                                for (int i = 0; i < n; ++i) {
                                    const bool referenced = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                                    const T got = C(i, j, b);
                                    if (!referenced) {
                                        ASSERT_EQ(got, poison)
                                            << "herk wrote the unreferenced triangle: " << route
                                            << " route, n=" << n << ", k=" << k
                                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                            << ", batch=" << b << ", row=" << i << ", col=" << j;
                                    } else if (i == j) {
                                        // BLAS sets the diagonal's imaginary part to zero rather than leaving
                                        // whatever the arithmetic produced, so this is exact.
                                        ASSERT_EQ(got.imag(), real_t(0))
                                            << "herk left an imaginary part on the diagonal: " << route
                                            << " route, n=" << n << ", k=" << k << ", batch=" << b << ", index=" << i;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    sweep("default");

    if constexpr (TestFixture::BackendType == Backend::CUDA) {
        {
            ScopedEnvVar route("BATCHLAS_EXPAND_ROUTE", "expand");
            sweep("gemm");
        }
        {
            ScopedEnvVar route("BATCHLAS_EXPAND_ROUTE", "loop");
            sweep("vendor-loop");
        }
    }
}

// A A^H is Hermitian, so the lower triangle HERK writes for Uplo::Lower and the
// upper one it writes for Uplo::Upper must be conjugate transposes of each
// other. Each triangle is checked against alpha A A^H in double; the two
// references are conjugates of each other, so agreement follows.
TYPED_TEST(HerkTest, TrianglesAgreeAcrossUplo) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int n = 65;
    const int k = 40;
    const int batch = 4;

    for (auto trans : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const int a_rows = trans == Transpose::NoTrans ? n : k;
        const int a_cols = trans == Transpose::NoTrans ? k : n;
        auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 31);

        Matrix<T, MatrixFormat::Dense> C_lower(n, n, batch);
        Matrix<T, MatrixFormat::Dense> C_upper(n, n, batch);
        (void)C_lower.view().fill_zeros(*(this->ctx));
        (void)C_upper.view().fill_zeros(*(this->ctx));
        this->ctx->wait();

        herk(*(this->ctx), A.view(), C_lower.view(), {.uplo = Uplo::Lower, .trans = trans}).wait();
        herk(*(this->ctx), A.view(), C_upper.view(), {.uplo = Uplo::Upper, .trans = trans}).wait();

        const auto general = batchlas::verify::Shape::general;
        const Transpose other = trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans;
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            auto& C = uplo == Uplo::Lower ? C_lower : C_upper;
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                          batchlas::verify::gemm_backward_error(A.view(), general, trans, A.view(), general, other, C.view(), C.view(),
                                                                uplo == Uplo::Lower ? batchlas::verify::Shape::lower : batchlas::verify::Shape::upper,
                                                                1.0, 0.0, batchlas::verify::all_items(batch)))
                << "herk's " << (uplo == Uplo::Lower ? "lower" : "upper") << " triangle: trans="
                << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans");
        }
    }
}

// The option struct must mean exactly what the positional call means. A bare
// `{}` in particular has to reach the option-struct overload -- the positional
// spelling takes seven arguments, so a four-argument call that resolved to it
// would not compile, but a defaulted option struct that disagreed with the
// positional defaults would silently compute something else.
TYPED_TEST(HerkTest, OptionStructMatchesPositional) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 32;
    const int k = 24;
    const int batch = 3;

    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, k, false, batch, 5);

    Matrix<T, MatrixFormat::Dense> C_defaults(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_defaults_pos(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named_pos(n, n, batch);
    for (auto* C : {&C_defaults, &C_defaults_pos, &C_named, &C_named_pos}) {
        (void)C->view().fill_zeros(*(this->ctx));
    }
    this->ctx->wait();

    (void)herk(*(this->ctx), A.view(), C_defaults.view(), {});
    (void)herk<Ba, T>(*(this->ctx), A.view(), C_defaults_pos.view(),
                real_t(1), real_t(0), Uplo::Lower, Transpose::NoTrans);

    (void)herk(*(this->ctx), A.view(), C_named.view(),
         {.alpha = real_t(1.5), .uplo = Uplo::Upper});
    (void)herk<Ba, T>(*(this->ctx), A.view(), C_named_pos.view(),
                real_t(1.5), real_t(0), Uplo::Upper, Transpose::NoTrans);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_EQ(C_defaults(i, j, b), C_defaults_pos(i, j, b))
                    << "herk defaults at (" << i << "," << j << ") batch " << b;
                ASSERT_EQ(C_named(i, j, b), C_named_pos(i, j, b))
                    << "herk designated initialisers at (" << i << "," << j << ") batch " << b;
            }
        }
    }
}

// The two tests above check the *shape* of the answer -- that the untouched
// triangle stays untouched, and that the two uplo runs agree. Neither can catch
// conjugating the wrong operand, which is the one thing a shared syrk/herk
// kernel is most likely to get wrong: conjugating the row index instead of the
// column returns conj(C) rather than C, which is still Hermitian and still
// consistent across both triangles. Only a value comparison sees it.
//
// n is swept because the narrow kernel picks a different tile width -- and, for
// complex, a different thread tile -- at 32, 64 and 128, and k is deliberately
// not a multiple of the k chunk.
TYPED_TEST(HerkTest, MatchesGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int k = 200;
    const int batch = 3;
    const real_t alpha = real_t(0.9);
    const real_t beta = real_t(-0.35);

    auto sweep = [&](const char* route) {
    for (int n : {24, 32, 48, 64, 96, 128}) {
        for (auto transA : {Transpose::NoTrans, Transpose::ConjTrans}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                const int a_rows = transA == Transpose::NoTrans ? n : k;
                const int a_cols = transA == Transpose::NoTrans ? k : n;

                Matrix<T, MatrixFormat::Dense> A =
                    Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
                Matrix<T, MatrixFormat::Dense> C0 =
                    Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
                Matrix<T, MatrixFormat::Dense> C(n, n, batch);

                // BLAS does not reference the imaginary part of a Hermitian C's
                // diagonal, so herk drops it and the reference would scale it by
                // beta. A real diagonal makes the two comparable.
                for (int b = 0; b < batch; ++b) {
                    for (int d = 0; d < n; ++d) {
                        C0(d, d, b) = T(C0(d, d, b).real(), real_t(0));
                    }
                }
                this->ctx->wait();

                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();

                herk<Ba, T>(*(this->ctx), A.view(), C.view(), alpha, beta, uplo, transA).wait();

                // Only the requested triangle is the answer; its conjugate half is not computed.
                const auto general = batchlas::verify::Shape::general;
                EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                              batchlas::verify::gemm_backward_error(
                                  A.view(), general, transA, A.view(), general,
                                  transA == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans, C0.view(), C.view(),
                                  uplo == Uplo::Lower ? batchlas::verify::Shape::lower : batchlas::verify::Shape::upper,
                                  batchlas::verify::up(T(alpha)), batchlas::verify::up(T(beta)), batchlas::verify::all_items(batch)))
                    << "route=" << route << " n=" << n << " trans=" << static_cast<int>(transA) << " uplo=" << static_cast<int>(uplo);
            }
        }
    }
    };

    sweep("default");
    if constexpr (Ba == Backend::CUDA) {
        // The Gram kernel is not on herk's automatic path -- it loses to the
        // GEMM-plus-fold in complex -- so without pinning it the conjugation
        // this test exists to check would never run.
        ScopedEnvVar pin("BATCHLAS_SYRK_ROUTE", "gram");
        sweep("gram");
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
