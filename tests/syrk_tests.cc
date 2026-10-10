#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>

using namespace batchlas;

template <typename T, Backend B>
struct SyrkConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SyrkTestTypes = typename test_utils::backend_types_filtered<SyrkConfig, false>::type;

template <typename Config>
class SyrkTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyrkTest, SyrkTestTypes);

// The `uplo` triangle of C against alpha op(A) op(A)^T + beta C0 in double, every item (Check::blas, k the
// inner dimension). The other triangle is not the answer; the poison tests check that it is left alone.
template <typename T, class VA, class VC0, class VC>
void expect_syrk_matches(const VA& A, const VC0& C0, const VC& C, Uplo uplo, Transpose trans, T alpha, T beta, int k) {
    const auto general = batchlas::verify::Shape::general;
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, k,
                  batchlas::verify::gemm_backward_error(A, general, trans, A, general,
                                                        trans == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans, C0, C,
                                                        uplo == Uplo::Lower ? batchlas::verify::Shape::lower : batchlas::verify::Shape::upper,
                                                        batchlas::verify::up(alpha), batchlas::verify::up(beta),
                                                        batchlas::verify::all_items(C.batch_size())))
        << "trans=" << static_cast<int>(trans) << ", uplo=" << static_cast<int>(uplo);
}

TYPED_TEST(SyrkTest, MatchesGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 96;
    const int k = 64;
    const int batch = 3;
    const T alpha = T(0.9);
    const T beta = T(-0.35);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const int a_rows = transA == Transpose::NoTrans ? n : k;
            const int a_cols = transA == Transpose::NoTrans ? k : n;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
            Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
            Matrix<T, MatrixFormat::Dense> C(n, n, batch);

            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();

            syrk(*(this->ctx),
                     A.view(),
                     C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

            expect_syrk_matches(A.view(), C0.view(), C.view(), uplo, transA, alpha, beta, k);
        }
    }
}

// MatchesGemmReference above pins one shape, n = 96, which reaches exactly one
// of the narrow kernel's three tile widths. The other two -- and the wider
// range ortho actually asks for, where k is the block size and n is tens of
// vectors -- went unexercised, so the band-split bug that broke n = 96 could
// equally have hidden at n = 32 or 64 with nothing to catch it. Each n here
// selects a different instantiation: 32 and 24 the 32-wide tile, 64 and 48 the
// 64-wide one, 128 and 96 the 128-wide one, with the odd sizes checking the
// masking of a tile the matrix does not fill.
TYPED_TEST(SyrkTest, NarrowShapesMatchGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int k = 200;   // deliberately not a multiple of the k chunk
    const int batch = 3;
    const T alpha = T(0.9);
    const T beta = T(-0.35);

    for (int n : {24, 32, 48, 64, 96, 128}) {
        for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                const int a_rows = transA == Transpose::NoTrans ? n : k;
                const int a_cols = transA == Transpose::NoTrans ? k : n;

                Matrix<T, MatrixFormat::Dense> A =
                    Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
                Matrix<T, MatrixFormat::Dense> C0 =
                    Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
                Matrix<T, MatrixFormat::Dense> C(n, n, batch);

                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();

                syrk(*(this->ctx), A.view(), C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

                SCOPED_TRACE(::testing::Message() << "n=" << n);
                expect_syrk_matches(A.view(), C0.view(), C.view(), uplo, transA, alpha, beta, k);
            }
        }
    }
}

// The poison tests further down are CUDA-only: they force a route with
// BATCHLAS_SYRK_ROUTE and skip without a GPU, so no *backend* other than CUDA
// has ever had "syrk leaves the other triangle alone" checked. This one is typed
// over every backend the build has. It is the contract SyrkOptions::uplo names,
// and it is exactly what the generic gemm fallback in src/extensions/syrk.cc got
// wrong: it aimed one gemm at C, which writes both triangles, with `uplo` an
// unnamed parameter. That fallback is instantiated for Backend::MKL only, so
// this test cannot fail in a build without MKL.
TYPED_TEST(SyrkTest, LeavesTheOtherTriangleUntouched) {
    using T = typename TestFixture::ScalarType;

    const int n = 48;
    const int k = 32;
    const int batch = 3;
    const T alpha = T(0.9);
    const T poison = T(-12345);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto beta : {T(0), T(0.5)}) {
                const int a_rows = transA == Transpose::NoTrans ? n : k;
                const int a_cols = transA == Transpose::NoTrans ? k : n;

                auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
                Matrix<T, MatrixFormat::Dense> C(n, n, batch);
                for (int b = 0; b < batch; ++b)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const bool referenced =
                                uplo == Uplo::Upper ? (i <= j) : (i >= j);
                            C(i, j, b) = referenced ? T(0.25) * T(i + j) : poison;
                        }

                syrk(*(this->ctx), A.view(), C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

                for (int b = 0; b < batch; ++b)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const bool referenced =
                                uplo == Uplo::Upper ? (i <= j) : (i >= j);
                            if (referenced) continue;
                            ASSERT_EQ(C(i, j, b), poison)
                                << "the half syrk was not given was written: "
                                << "trans=" << static_cast<int>(transA)
                                << ", uplo=" << static_cast<int>(uplo)
                                << ", batch=" << b << ", row=" << i << ", col=" << j;
                        }
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#if BATCHLAS_HAS_CUDA_BACKEND
namespace {

// syrk names one triangle of C, and BLAS forbids the other one from being
// written. Building C symmetric and comparing against gemm cannot see that:
// both halves then hold the same numbers, so a route that computes and stores
// the whole n x n passes anyway. Poisoning the unreferenced half with values
// nothing in the problem could produce makes the difference observable, and
// the values are distinct per element so that writing the transposed position
// is caught too, not merely writing something.
float syrk_poison_value(int row, int col, int batch, int n) {
    return -static_cast<float>(1 + row + n * col + n * n * batch);
}

void poison_unreferenced_triangle(Matrix<float, MatrixFormat::Dense>& C, Uplo uplo) {
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (!referenced) {
                    C(i, j, b) = syrk_poison_value(i, j, b, n);
                }
            }
        }
    }
}

void expect_triangle_respected(Matrix<float, MatrixFormat::Dense>& A,
                               Matrix<float, MatrixFormat::Dense>& C,
                               Matrix<float, MatrixFormat::Dense>& C0,
                               Uplo uplo,
                               Transpose transA,
                               float alpha,
                               float beta,
                               int k) {
    expect_syrk_matches<float>(A.view(), C0.view(), C.view(), uplo, transA, alpha, beta, k);
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (!referenced) {
                    ASSERT_EQ(C(i, j, b), syrk_poison_value(i, j, b, n))
                        << "wrote outside the requested triangle: n=" << n
                        << ", trans=" << static_cast<int>(transA)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                }
            }
        }
    }
}

} // namespace

TEST(SyrkCudaCustomTest, TriangularTilesLeaveTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }

    struct Shape {
        int n;
        int k;
        int batch;
    };

    // 256x64 is whole 128 tiles with a k the 8-deep staging fills exactly, so
    // it takes the unpredicated path; 200x53 breaks both and takes the
    // predicated one, with a partial tile on the diagonal. 384x8 is the
    // shallowest k that runs, and wide enough to have a tile that is neither on
    // the diagonal nor next to it.
    const Shape shapes[] = {{256, 64, 8}, {200, 53, 5}, {384, 8, 3}};
    const float alpha = 0.9f;
    const float beta = -0.35f;

    for (const auto& shape : shapes) {
        for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
            const int a_rows = transA == Transpose::NoTrans ? shape.n : shape.k;
            const int a_cols = transA == Transpose::NoTrans ? shape.k : shape.n;
            Matrix<float, MatrixFormat::Dense> A =
                Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 17);

            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                Matrix<float, MatrixFormat::Dense> C0 =
                    Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 23);
                poison_unreferenced_triangle(C0, uplo);
                auto C_custom = C0.clone();

                {
                    ScopedEnvVar force_route("BATCHLAS_SYRK_ROUTE", "triangular");
                    syrk(ctx,
                         A.view(),
                         C_custom.view(),
                         {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
                }

                expect_triangle_respected(A, C_custom, C0, uplo, transA, alpha, beta, shape.k);
            }
        }
    }
}

TEST(SyrkCudaCustomTest, AutoAndNativeRoutesLeaveTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }

    struct Shape {
        int n;
        int k;
        int batch;
    };

    // What this guards is the routing, not any one kernel: whichever route a
    // shape picks, the unreferenced half of C belongs to the caller. The
    // shapes sit on both sides of the transcribed table's steps
    // (tuned/syrk.float.*.txt): 512x64 batch 32 is a tall triangular row,
    // 512x512 batch 4 and 256x256 squareish triangular rows, and the two
    // small ones gram rows.
    const Shape shapes[] = {{512, 64, 32}, {512, 512, 4}, {256, 256, 8},
                            {128, 128, 2}, {96, 96, 1}};
    const float alpha = 1.25f;
    const float beta = 0.5f;

    for (const auto& shape : shapes) {
        Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(
            shape.n, shape.k, false, shape.batch, 41);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C0 =
                Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
            poison_unreferenced_triangle(C0, uplo);
            auto C_auto = C0.clone();
            auto C_native = C0.clone();

            syrk(ctx,
                 A.view(),
                 C_auto.view(),
                 {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = Transpose::NoTrans}).wait();
            {
                // `native` is a tile kernel, never the both-triangles GEMM it used to reach.
                ScopedEnvVar native_route("BATCHLAS_SYRK_ROUTE", "native");
                syrk(ctx,
                     A.view(),
                     C_native.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = Transpose::NoTrans}).wait();
            }

            expect_triangle_respected(A, C_auto, C0, uplo, Transpose::NoTrans, alpha, beta, shape.k);
            expect_triangle_respected(A, C_native, C0, uplo, Transpose::NoTrans, alpha, beta, shape.k);
        }
    }
}
// BATCHLAS_SYRK_ROUTE takes only its own words; the removed legacy spellings (and any typo)
// throw rather than silently meaning Auto.
TEST(SyrkCudaCustomTest, RemovedRouteWordsThrow) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }
    Matrix<float, MatrixFormat::Dense> A(16, 8, 2), C(16, 16, 2);
    for (const char* word : {"tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm", "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled", "native:auto", "vendor:auto", "bogus", "expand", "cublasdx"}) {
        ScopedEnvVar route("BATCHLAS_SYRK_ROUTE", word);
        EXPECT_THROW(syrk(ctx, A.view(), C.view(), {.alpha = 1.0f, .beta = 0.0f}).wait(), std::invalid_argument) << word;
    }
}

#endif