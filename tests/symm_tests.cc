#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <cstdlib>
#include <string>
#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>
#include "../src/ops/symm/choice.hh"

// The forced-route tests pin through select::ScopedPin (docs/design/flat-kernel-selection.md §12).
using SymmPin = batchlas::select::ScopedPin<batchlas::ops::symm::SymmChoice>;

using namespace batchlas;

template <typename T, Backend B>
struct SymmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SymmTestTypes = typename test_utils::backend_types_filtered<SymmConfig, false>::type;

template <typename Config>
class SymmTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SymmTest, SymmTestTypes);

// C against alpha A B + beta C0 (Side::Left) or alpha B A + beta C0, with A the symmetric matrix the `uplo`
// triangle of its storage describes, in double on every item (Check::blas, k = the order of A).
template <typename T, class VA, class VB, class VC0, class VC>
void expect_symm_matches(const VA& A, const VB& B, const VC0& C0, const VC& C, Side side, Uplo uplo, T alpha, T beta, int n) {
    const auto sym = uplo == Uplo::Lower ? batchlas::verify::Shape::symmetric_lower : batchlas::verify::Shape::symmetric_upper;
    const auto general = batchlas::verify::Shape::general;
    const auto items = batchlas::verify::all_items(C.batch_size());
    const double err =
        side == Side::Left
            ? batchlas::verify::gemm_backward_error(A, sym, Transpose::NoTrans, B, general, Transpose::NoTrans, C0, C, general,
                                                    batchlas::verify::up(alpha), batchlas::verify::up(beta), items)
            : batchlas::verify::gemm_backward_error(B, general, Transpose::NoTrans, A, sym, Transpose::NoTrans, C0, C, general,
                                                    batchlas::verify::up(alpha), batchlas::verify::up(beta), items);
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, n, err)
        << "n=" << n << ", side=" << static_cast<int>(side) << ", uplo=" << static_cast<int>(uplo);
}

TYPED_TEST(SymmTest, MatchesSymmetrizedGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 96;
    const int m = 64;
    const int batch = 3;
    const T alpha = T(1.25);
    const T beta = T(-0.5);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const int rows = side == Side::Left ? n : m;
            const int cols = side == Side::Left ? m : n;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
            Matrix<T, MatrixFormat::Dense> B = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch);
            Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch);

            Matrix<T, MatrixFormat::Dense> C(rows, cols, batch);

            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();

            symm(*(this->ctx),
                     A.view(),
                     B.view(),
                     C.view(),
                     {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();

            expect_symm_matches(A.view(), B.view(), C0.view(), C.view(), side, uplo, alpha, beta, n);
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#if BATCHLAS_HAS_CUDA_BACKEND
TEST(SymmCudaCustomTest, ForcedExpandPathMatchesVendor) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }

    const int n = 128;
    const int batch = 64;
    const float alpha = 1.1f;
    const float beta = -0.3f;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 7);
    Matrix<float, MatrixFormat::Dense> B = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 11);
    Matrix<float, MatrixFormat::Dense> C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 13);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);

            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();

            {
                const SymmPin force_route("symm", batchlas::ops::symm::Expand{});
                symm(ctx,
                                    A.view(),
                                    B.view(),
                                    C_custom.view(),
                                    {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
            }

            expect_symm_matches(A.view(), B.view(), C0.view(), C_custom.view(), side, uplo, alpha, beta, n);
        }
    }
}

// The custom path expands the referenced triangle into scratch a 32x32 tile at
// a time, so the sizes that matter are the ones where that tiling is ragged and
// the ones where the storage's leading dimension is not the matrix width.
TEST(SymmCudaCustomTest, ForcedExpandPathIgnoresUnreferencedTriangle) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }

    const int m = 61;
    const int batch = 5;
    const float alpha = 1.25f;
    const float beta = -0.5f;

    for (int n : {16, 33, 77, 129}) {
        for (auto side : {Side::Left, Side::Right}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                const int rows = side == Side::Left ? n : m;
                const int cols = side == Side::Left ? m : n;

                auto A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 17);
                auto B = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, false, batch, 19);
                auto C0 = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, false, batch, 23);
                ctx.wait();

                // Use the element accessor, not raw data() arithmetic: the
                // storage has its own leading dimension and batch stride, and
                // assuming ld == n silently writes into the referenced triangle.
                for (int b = 0; b < batch; ++b) {
                    for (int col = 0; col < n; ++col) {
                        for (int row = 0; row < n; ++row) {
                            const bool referenced =
                                (uplo == Uplo::Lower) ? (row >= col) : (row <= col);
                            if (!referenced) {
                                A(row, col, b) = 1000.0f;
                            }
                        }
                    }
                }
                ctx.wait();

                Matrix<float, MatrixFormat::Dense> C_custom(rows, cols, batch);
                MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();

                {
                    const SymmPin force_route("symm", batchlas::ops::symm::Expand{});
                    symm(ctx, A.view(), B.view(), C_custom.view(),
                         {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
                }

                expect_symm_matches(A.view(), B.view(), C0.view(), C_custom.view(), side, uplo, alpha, beta, n);
            }
        }
    }
}

// The expansion and the GEMM that consumes it are ordered by the queue's native
// stream, which only exists on an in-order queue; the out-of-order case takes a
// different ordering path and is not otherwise exercised.
TEST(SymmCudaCustomTest, ForcedExpandPathOrdersExpansionOnOutOfOrderQueue) {
    Queue ordered;
    if (ordered.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }
    Queue ctx(ordered.device(), Backend::CUDA, /*in_order=*/false);

    const int n = 192;
    const int batch = 8;
    const float alpha = 1.25f;
    const float beta = -0.5f;

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            auto A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 29);
            auto B = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 31);
            auto C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 37);

            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();

            {
                const SymmPin force_route("symm", batchlas::ops::symm::Expand{});
                symm(ctx, A.view(), B.view(), C_custom.view(),
                     {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
            }
            ctx.wait();

            expect_symm_matches(A.view(), B.view(), C0.view(), C_custom.view(), side, uplo, alpha, beta, n);
        }
    }
}
// BATCHLAS_SYMM_ROUTE takes only its own words; the removed legacy spellings (and any typo)
// throw rather than silently meaning Auto.
TEST(SymmCudaCustomTest, RemovedRouteWordsThrow) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }
    const int n = 16;
    Matrix<float, MatrixFormat::Dense> A(n, n, 2), B(n, n, 2), C(n, n, 2);
    for (const char* word : {"tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm", "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled", "native:auto", "vendor:auto", "bogus", "triangular", "gram", "cublasdx"}) {
        ScopedEnvVar route("BATCHLAS_SYMM_ROUTE", word);
        EXPECT_THROW(symm(ctx, A.view(), B.view(), C.view(), {.alpha = 1.0f, .beta = 0.0f}).wait(), std::invalid_argument) << word;
    }
}

#endif