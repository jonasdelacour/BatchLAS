#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

#include <complex>
#include <cstdlib>
#include <string>

#include "test_utils.hh"

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

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name_)) {
            old_value_ = old;
            had_old_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

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
// referenced triangle matches an independent GEMM taking beta from the clean C,
// the poison is still bit-for-bit intact afterwards, and the diagonal came out
// real.
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
            const real_t tol = test_utils::tolerance<T>() * real_t(64 * (n + k));

            for (auto trans : {Transpose::NoTrans, Transpose::ConjTrans}) {
                const int a_rows = trans == Transpose::NoTrans ? n : k;
                const int a_cols = trans == Transpose::NoTrans ? k : n;

                auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 17);
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

                        Matrix<T, MatrixFormat::Dense> R(n, n, shape.batch);
                        R.view().fill_zeros(*(this->ctx));
                        this->ctx->wait();

                        herk(*(this->ctx), A.view(), C.view(),
                             {.alpha = scaling.alpha,
                              .beta = scaling.beta,
                              .uplo = uplo,
                              .trans = trans}).wait();

                        // A A^H the long way round: a full GEMM writing both
                        // triangles, with beta applied on the host from the
                        // clean C rather than from the poisoned one.
                        gemm(*(this->ctx), A.view(), A.view(), R.view(),
                             {.alpha = T(scaling.alpha),
                              .beta = T(0),
                              .transA = trans,
                              .transB = trans == Transpose::NoTrans ? Transpose::ConjTrans
                                                                    : Transpose::NoTrans}).wait();

                        for (int b = 0; b < shape.batch; ++b) {
                            for (int j = 0; j < n; ++j) {
                                for (int i = 0; i < n; ++i) {
                                    const bool referenced =
                                        (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                                    const T got = C(i, j, b);

                                    if (!referenced) {
                                        ASSERT_EQ(got, poison)
                                            << "herk wrote the unreferenced triangle: " << route
                                            << " route, n=" << n << ", k=" << k
                                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                            << ", batch=" << b << ", row=" << i << ", col=" << j;
                                        continue;
                                    }

                                    const T product = R(i, j, b);
                                    const T prev = C0(i, j, b);
                                    // The diagonal of a Hermitian matrix is
                                    // real, so beta scales only its real part.
                                    const real_t want_real =
                                        product.real() + scaling.beta * prev.real();
                                    const real_t want_imag =
                                        product.imag() + scaling.beta * prev.imag();

                                    ASSERT_NEAR(got.real(), want_real, tol)
                                        << "herk read storage it must not touch: " << route
                                        << " route, n=" << n << ", k=" << k
                                        << ", trans="
                                        << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                                        << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                        << ", beta=" << scaling.beta
                                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                                    if (i == j) {
                                        // BLAS sets the diagonal's imaginary
                                        // part to zero rather than leaving
                                        // whatever the arithmetic produced, so
                                        // this is exact and not a tolerance.
                                        ASSERT_EQ(got.imag(), real_t(0))
                                            << "herk left an imaginary part on the diagonal: " << route
                                            << " route, n=" << n << ", k=" << k
                                            << ", trans="
                                            << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                            << ", beta=" << scaling.beta
                                            << ", batch=" << b << ", index=" << i;
                                    } else {
                                        ASSERT_NEAR(got.imag(), want_imag, tol)
                                            << "herk read storage it must not touch: " << route
                                            << " route, n=" << n << ", k=" << k
                                            << ", trans="
                                            << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                                            << ", uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                            << ", beta=" << scaling.beta
                                            << ", batch=" << b << ", row=" << i << ", col=" << j;
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
// other. That is a property of the result rather than of the reference used to
// check it, so it catches a fold that dropped a conjugate somewhere -- which
// against a GEMM reference computed the same way could otherwise cancel out.
TYPED_TEST(HerkTest, TrianglesAgreeAcrossUplo) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int n = 65;
    const int k = 40;
    const int batch = 4;
    const real_t tol = test_utils::tolerance<T>() * real_t(64 * (n + k));

    for (auto trans : {Transpose::NoTrans, Transpose::ConjTrans}) {
        const int a_rows = trans == Transpose::NoTrans ? n : k;
        const int a_cols = trans == Transpose::NoTrans ? k : n;
        auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 31);

        Matrix<T, MatrixFormat::Dense> C_lower(n, n, batch);
        Matrix<T, MatrixFormat::Dense> C_upper(n, n, batch);
        C_lower.view().fill_zeros(*(this->ctx));
        C_upper.view().fill_zeros(*(this->ctx));
        this->ctx->wait();

        herk(*(this->ctx), A.view(), C_lower.view(), {.uplo = Uplo::Lower, .trans = trans}).wait();
        herk(*(this->ctx), A.view(), C_upper.view(), {.uplo = Uplo::Upper, .trans = trans}).wait();

        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = j; i < n; ++i) {
                    const T lower = C_lower(i, j, b);
                    const T upper = C_upper(j, i, b);
                    ASSERT_NEAR(lower.real(), upper.real(), tol)
                        << "herk's two triangles disagree: trans="
                        << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                    ASSERT_NEAR(lower.imag(), -upper.imag(), tol)
                        << "herk's two triangles are not conjugates: trans="
                        << (trans == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                }
            }
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
        C->view().fill_zeros(*(this->ctx));
    }
    this->ctx->wait();

    herk(*(this->ctx), A.view(), C_defaults.view(), {});
    herk<Ba, T>(*(this->ctx), A.view(), C_defaults_pos.view(),
                real_t(1), real_t(0), Uplo::Lower, Transpose::NoTrans);

    herk(*(this->ctx), A.view(), C_named.view(),
         {.alpha = real_t(1.5), .uplo = Uplo::Upper});
    herk<Ba, T>(*(this->ctx), A.view(), C_named_pos.view(),
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
    const real_t tol = test_utils::tolerance<T>() * real_t(12 * k);

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
                Matrix<T, MatrixFormat::Dense> C_ref(n, n, batch);

                // BLAS does not reference the imaginary part of a Hermitian C's
                // diagonal, so herk drops it and a plain GEMM scales it by beta.
                // Starting from a real diagonal is what makes the two
                // comparable; without it the test fails on element (0,0) for a
                // reason that has nothing to do with the kernel.
                for (int b = 0; b < batch; ++b) {
                    for (int d = 0; d < n; ++d) {
                        C0(d, d, b) = T(C0(d, d, b).real(), real_t(0));
                    }
                }
                this->ctx->wait();

                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();
                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C_ref.view(), C0.view()).wait();

                herk<Ba, T>(*(this->ctx), A.view(), C.view(), alpha, beta, uplo, transA).wait();

                gemm(*(this->ctx), A.view(), A.view(), C_ref.view(),
                     {.alpha = T(alpha), .beta = T(beta), .transA = transA,
                      .transB = transA == Transpose::NoTrans ? Transpose::ConjTrans
                                                             : Transpose::NoTrans}).wait();

                // Mirror both, so the half herk was not asked for is compared
                // against the reference too rather than against its own input.
                C.view().symmetrize(*(this->ctx), uplo).wait();
                C_ref.view().symmetrize(*(this->ctx), uplo).wait();

                for (int b = 0; b < batch; ++b) {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < n; ++i) {
                            const T got = C(i, j, b);
                            const T want = C_ref(i, j, b);
                            ASSERT_NEAR(got.real(), want.real(), tol)
                                << "real route=" << route << " n=" << n
                                << " trans=" << static_cast<int>(transA)
                                << " uplo=" << static_cast<int>(uplo)
                                << " b=" << b << " row=" << i << " col=" << j;
                            ASSERT_NEAR(got.imag(), want.imag(), tol)
                                << "imag route=" << route << " n=" << n
                                << " trans=" << static_cast<int>(transA)
                                << " uplo=" << static_cast<int>(uplo)
                                << " b=" << b << " row=" << i << " col=" << j;
                        }
                    }
                }
            }
        }
    }
    };

    sweep("default");
    if constexpr (Ba == Backend::CUDA) {
        // The Gram kernel is not on herk's automatic path -- it loses to the
        // GEMM-plus-fold in complex -- so without pinning it the conjugation
        // this test exists to check would never run.
        ScopedEnvVar pin("BATCHLAS_SYRK_VARIANT", "gram");
        sweep("gram");
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
