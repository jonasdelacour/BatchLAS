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
struct Her2kConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// her2k exists only for complex scalars; its real spelling is syr2k.
using Her2kTestTypes = typename test_utils::backend_types_complex<Her2kConfig>::type;

template <typename Config>
class Her2kTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(Her2kTest, Her2kTestTypes);

// HER2K owns exactly one triangle of C. The opposite one is not part of the
// result: it is neither read through beta nor written, and neither is the
// imaginary part of the diagonal, because C = C^H forces that to zero whatever
// the caller stored there.
//
// The reference is alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H + beta C0 in
// double (the library's rank-2k check), with the conjugate on the second term
// explicit: the whole of the difference from SYR2K and the easiest thing to lose.
//
// The shapes are ragged on purpose, and on CUDA the sweep runs once per route
// with the choice pinned: left to the default, every shape here would take the
// vendor loop and the fold would never be reached.
TYPED_TEST(Her2kTest, IgnoresUnreferencedTriangleOfC) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    struct Shape {
        int n;
        int k;
        int batch;
    };
    const Shape shapes[] = {{16, 24, 2}, {33, 61, 5}, {77, 13, 1}, {96, 64, 3}, {129, 48, 2}};

    struct Scaling {
        T alpha;
        real_t beta;
    };
    // beta = 0 is its own path: C is not an input at all there. A complex alpha
    // is the point of the first case -- with a real alpha the two terms would
    // be conjugates whether or not the implementation conjugated the scalar.
    const Scaling scalings[] = {{T(1.25, -0.75), real_t(-0.5)}, {T(0.5, 2.0), real_t(0)}};

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
                auto B = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 19);
                auto C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, shape.batch, 23);
                this->ctx->wait();

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

                        her2k(*(this->ctx), A.view(), B.view(), C.view(),
                              {.alpha = scaling.alpha,
                               .beta = scaling.beta,
                               .uplo = uplo,
                               .trans = trans}).wait();

                        // C0's diagonal is read as real, as BLAS does.
                        EXPECT_VERIFY(T, batchlas::verify::Check::blas, 2 * k,
                                      batchlas::verify::rank2k_backward_error(A.view(), B.view(), trans, C0.view(), C.view(), uplo,
                                                                              batchlas::verify::up(scaling.alpha),
                                                                              batchlas::verify::up(T(scaling.beta)), true,
                                                                              batchlas::verify::all_items(shape.batch)))
                            << "her2k read storage it must not touch: " << route << " route, n=" << n << ", k=" << k
                            << ", trans=" << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper") << ", beta=" << scaling.beta;

                        for (int b = 0; b < shape.batch; ++b) {
                            for (int j = 0; j < n; ++j) {
                                for (int i = 0; i < n; ++i) {
                                    const bool referenced = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                                    const T got = C(i, j, b);
                                    if (!referenced) {
                                        ASSERT_EQ(got, poison)
                                            << "her2k wrote the unreferenced triangle: " << route
                                            << " route, n=" << n << ", k=" << k
                                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                            << ", batch=" << b << ", row=" << i << ", col=" << j;
                                    } else if (i == j) {
                                        // BLAS sets the diagonal's imaginary part to zero rather than leaving
                                        // whatever the arithmetic produced, so this is exact.
                                        ASSERT_EQ(got.imag(), real_t(0))
                                            << "her2k left an imaginary part on the diagonal: " << route
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

// alpha * A * B^H + conj(alpha) * B * A^H is Hermitian for every alpha, so the
// lower triangle HER2K writes for Uplo::Lower and the upper one it writes for
// Uplo::Upper must be conjugate transposes of each other. Losing the conjugate
// on the second term gives a complex-symmetric sum instead. Each triangle is
// checked against the rank-2k definition; the two references are conjugates.
TYPED_TEST(Her2kTest, TrianglesAgreeAcrossUplo) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int n = 65;
    const int k = 40;
    const int batch = 4;
    const T alpha = T(1.25, -0.75);

    for (auto trans : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const int a_rows = trans == Transpose::NoTrans ? n : k;
        const int a_cols = trans == Transpose::NoTrans ? k : n;
        auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 31);
        auto B = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 37);

        Matrix<T, MatrixFormat::Dense> C_lower(n, n, batch);
        Matrix<T, MatrixFormat::Dense> C_upper(n, n, batch);
        (void)C_lower.view().fill_zeros(*(this->ctx));
        (void)C_upper.view().fill_zeros(*(this->ctx));
        this->ctx->wait();

        her2k(*(this->ctx), A.view(), B.view(), C_lower.view(),
              {.alpha = alpha, .uplo = Uplo::Lower, .trans = trans}).wait();
        her2k(*(this->ctx), A.view(), B.view(), C_upper.view(),
              {.alpha = alpha, .uplo = Uplo::Upper, .trans = trans}).wait();

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const auto& C = uplo == Uplo::Lower ? C_lower : C_upper;
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, 2 * k,
                          batchlas::verify::rank2k_backward_error(A.view(), B.view(), trans, C.view(), C.view(), uplo,
                                                                  batchlas::verify::up(alpha), batchlas::verify::up(T(0)), true,
                                                                  batchlas::verify::all_items(batch)))
                << "her2k's " << (uplo == Uplo::Lower ? "lower" : "upper") << " triangle: trans="
                << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans");
        }
    }
}

// The option struct must mean exactly what the positional call means. A bare
// `{}` in particular has to reach the option-struct overload -- the positional
// spelling takes eight arguments, so a five-argument call that resolved to it
// would not compile, but a defaulted option struct that disagreed with the
// positional defaults would silently compute something else.
TYPED_TEST(Her2kTest, OptionStructMatchesPositional) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 32;
    const int k = 24;
    const int batch = 3;

    auto A = Matrix<T, MatrixFormat::Dense>::Random(n, k, false, batch, 5);
    auto B = Matrix<T, MatrixFormat::Dense>::Random(n, k, false, batch, 7);

    Matrix<T, MatrixFormat::Dense> C_defaults(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_defaults_pos(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named(n, n, batch);
    Matrix<T, MatrixFormat::Dense> C_named_pos(n, n, batch);
    for (auto* C : {&C_defaults, &C_defaults_pos, &C_named, &C_named_pos}) {
        (void)C->view().fill_zeros(*(this->ctx));
    }
    this->ctx->wait();

    (void)her2k(*(this->ctx), A.view(), B.view(), C_defaults.view(), {});
    (void)her2k<Ba, T>(*(this->ctx), A.view(), B.view(), C_defaults_pos.view(),
                 T(1), real_t(0), Uplo::Lower, Transpose::NoTrans);

    (void)her2k(*(this->ctx), A.view(), B.view(), C_named.view(),
          {.alpha = T(1.5, 0.25), .uplo = Uplo::Upper});
    (void)her2k<Ba, T>(*(this->ctx), A.view(), B.view(), C_named_pos.view(),
                 T(1.5, 0.25), real_t(0), Uplo::Upper, Transpose::NoTrans);
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_EQ(C_defaults(i, j, b), C_defaults_pos(i, j, b))
                    << "her2k defaults at (" << i << "," << j << ") batch " << b;
                ASSERT_EQ(C_named(i, j, b), C_named_pos(i, j, b))
                    << "her2k designated initialisers at (" << i << "," << j << ") batch " << b;
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
