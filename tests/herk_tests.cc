#include <gtest/gtest.h>
#include <batchlas/backend_config.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#include <tests/test_utils.hh>
#include <src/ops/syrk/kernels/syrk_gram_tiles.hh>

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

                        herk(*(this->ctx), A, C,
                             {.alpha = scaling.alpha,
                              .beta = scaling.beta,
                              .uplo = uplo,
                              .trans = trans}).wait();

                        using MV = MatrixView<T, MatrixFormat::Dense>;
                        const double err = MV::gemm_backward_error(
                            *(this->ctx), A, trans, A,
                            trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans, C0, C,
                            MV::up(T(scaling.alpha)), MV::up(T(scaling.beta)), C.all_items(), uplo);
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
        (void)C_lower.fill_zeros(*(this->ctx));
        (void)C_upper.fill_zeros(*(this->ctx));
        this->ctx->wait();

        herk(*(this->ctx), A, C_lower, {.uplo = Uplo::Lower, .trans = trans}).wait();
        herk(*(this->ctx), A, C_upper, {.uplo = Uplo::Upper, .trans = trans}).wait();

        using MV = MatrixView<T, MatrixFormat::Dense>;
        const Transpose other = trans == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans;
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            auto& C = uplo == Uplo::Lower ? C_lower : C_upper;
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                          MV::gemm_backward_error(*(this->ctx), A, trans, A, other, C, C, 1.0, 0.0,
                                                  C.all_items(), uplo))
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

    (void)herk(*(this->ctx), A, C_defaults, {});
    (void)herk<Ba, T>(*(this->ctx), A, C_defaults_pos,
                real_t(1), real_t(0), Uplo::Lower, Transpose::NoTrans);

    (void)herk(*(this->ctx), A, C_named,
         {.alpha = real_t(1.5), .uplo = Uplo::Upper});
    (void)herk<Ba, T>(*(this->ctx), A, C_named_pos,
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

                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C, C0).wait();

                herk<Ba, T>(*(this->ctx), A, C, alpha, beta, uplo, transA).wait();

                // Only the requested triangle is the answer; its conjugate half is not computed.
                using MV = MatrixView<T, MatrixFormat::Dense>;
                EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                              MV::gemm_backward_error(
                                  *(this->ctx), A, transA, A,
                                  transA == Transpose::NoTrans ? Transpose::ConjTrans : Transpose::NoTrans, C0, C,
                                  MV::up(T(alpha)), MV::up(T(beta)), C.all_items(), uplo))
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

// The gram tile kernel puts the batch in a grid dimension capped at 65535 and runs a batch past it
// as slices. With gram pinned, 65535, 65536 and 2 * 65535 + 1 items repeating 7 distinct problems
// (out of phase with every slice; padded lds and strides, beta != 0, complex data) run the gram
// kernel: every item is bit-identical to the gram kernel launched directly on the 7 (the vendor
// loop is another kernel), and right.
TYPED_TEST(HerkTest, GramBatchPastTheGridRunsAsSlices) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;
    if constexpr (Ba != Backend::CUDA || !BATCHLAS_HAS_CUBLAS) {
        GTEST_SKIP() << "the gram opt-in is cuBLAS's herk (no route vendor-free)";
    } else {
        using MV = MatrixView<T, MatrixFormat::Dense>;
        Queue& q = *(this->ctx);
        const ScopedEnvVar pin("BATCHLAS_SYRK_ROUTE", "gram");
        const int n = 8, k = 5, period = 7;
        const real_t alpha = real_t(0.75), beta = real_t(-0.5);
        auto A7 = Matrix<T, MatrixFormat::Dense>::Random(n, k, false, period, 31);
        auto C7 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, period, 37);
        q.wait();
        for (int r = 0; r < period; ++r)
            for (int d = 0; d < n; ++d) C7(d, d, r) = T(C7(d, d, r).real(), real_t(0));
        auto want = C7.clone();
        q.wait();
        (void)backend::detail::syrk_gram_tiles<T, true>(q, A7, want, T(alpha), T(beta), Uplo::Lower,
                                                        Transpose::NoTrans);
        q.wait();
        for (const int batch : {65535, 65536, 2 * 65535 + 1}) {
            Matrix<T, MatrixFormat::Dense> Ab(n + 3, k + 1, batch), Cb(n + 2, n + 1, batch);
            const MV A(Ab, n, k), C(Cb, n, n);
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i) {
                        if (j < k) Ab(i, j, b) = A7(i, j, b % period);
                        Cb(i, j, b) = C7(i, j, b % period);
                    }
            auto C0b = Cb.clone();
            q.wait();
            const MV C0(C0b, n, n);
            (void)herk<Ba, T>(q, A, C, alpha, beta, Uplo::Lower, Transpose::NoTrans);
            q.wait();
            int differ = 0;
            for (int b = 0; b < batch; ++b)
                for (int j = 0; j < n; ++j)
                    for (int i = j; i < n; ++i) differ += Cb(i, j, b) != want(i, j, b % period);
            EXPECT_EQ(differ, 0) << "batch " << batch << ": not the gram kernel on every item";
            std::vector<int> items{batch / 2, batch - 1};
            for (int r = 0; r < period; ++r) items.push_back(r);
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                          MV::gemm_backward_error(q, A, Transpose::NoTrans, A, Transpose::ConjTrans, C0, C,
                                                  MV::up(T(alpha)), MV::up(T(beta)), items,
                                                  Uplo::Lower))
                << "batch " << batch;
        }
    }
}

// The GEMM-plus-fold: its gemm and accumulate_hermitian put the batch in grid z (65535 on CUDA)
// and run a batch past it as slices, where expansion_fits used to refuse it. With the expansion
// pinned, 65535, 65536 and 2 * 65535 + 1 distinct items take it (its public gemm is in the select
// trace at the full batch; the per-item loop calls no gemm), at padded lds and strides, beta != 0,
// both triangles and transposes, C's input diagonal with a nonzero imaginary part (not read; the
// output's is exactly 0), and every item is right.
TYPED_TEST(HerkTest, ExpandBatchPastTheGridRunsAsSlices) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;
    if constexpr (Ba != Backend::CUDA || !BATCHLAS_HAS_CUBLAS) {
        GTEST_SKIP() << "the expansion is cuBLAS's herk (no route vendor-free)";
    } else {
        using MV = MatrixView<T, MatrixFormat::Dense>;
        Queue& q = *(this->ctx);
        const ScopedEnvVar route("BATCHLAS_EXPAND_ROUTE", "expand");
        const int n = 8, k = 5;
        const real_t alpha = real_t(0.75), beta = real_t(-0.5);
        for (const int batch : {65535, 65536, 2 * 65535 + 1}) {
            const bool odd = batch % 2;
            const Transpose trans = odd ? Transpose::ConjTrans : Transpose::NoTrans;
            const Uplo uplo = odd ? Uplo::Upper : Uplo::Lower;
            const int ar = odd ? k : n, ac = odd ? n : k;
            auto Ab = Matrix<T, MatrixFormat::Dense>::Random(ar + 3, ac + 1, false, batch, 41);
            auto Cb = Matrix<T, MatrixFormat::Dense>::Random(n + 2, n + 1, false, batch, 43);
            auto C0b = Cb.clone();
            q.wait();
            const MV A(Ab, ar, ac), C(Cb, n, n), C0(C0b, n, n);
            for (int b = 0; b < batch; ++b)
                for (int d = 0; d < n; ++d) C0b(d, d, b) = T(C0b(d, d, b).real(), real_t(0));
            const ScopedEnvVar trace("BATCHLAS_SELECT_TRACE", "1");
            ::testing::internal::CaptureStderr();
            (void)herk<Ba, T>(q, A, C, alpha, beta, uplo, trans);
            q.wait();
            const std::string err = ::testing::internal::GetCapturedStderr();
            EXPECT_NE(err.find("batch=" + std::to_string(batch) + " -> "), std::string::npos)
                << "batch " << batch << ": no gemm at the full batch, so not the expansion: " << err;
            int imaginary = 0;
            for (int b = 0; b < batch; ++b)
                for (int d = 0; d < n; ++d) imaginary += Cb(d, d, b).imag() != real_t(0);
            EXPECT_EQ(imaginary, 0) << "batch " << batch << ": an output diagonal kept an imaginary part";
            EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                          MV::gemm_backward_error(q, A, trans, A, odd ? Transpose::NoTrans : Transpose::ConjTrans,
                                                  C0, C, MV::up(T(alpha)), MV::up(T(beta)), C.all_items(), uplo))
                << "batch " << batch;
        }
    }
}
