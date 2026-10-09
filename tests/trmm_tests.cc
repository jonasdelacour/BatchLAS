#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>
#include "../src/select/vendor.hh"

using namespace batchlas;

template <typename T, Backend B>
struct TrmmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using TrmmTestTypes = typename test_utils::backend_types<TrmmConfig>::type;

template <typename Config>
class TrmmTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(TrmmTest, TrmmTestTypes);

// Componentwise error of C = op(A) B (Left) or B op(A) (Right), A read as its triangle; k = A's order.
template <typename T>
double trmm_error(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C, Side side, Uplo uplo, Transpose trans, Diag diag) {
    using verify::Shape;
    const bool unit = diag == Diag::Unit;
    const Shape tri = uplo == Uplo::Lower ? (unit ? Shape::unit_lower : Shape::lower) : (unit ? Shape::unit_upper : Shape::upper);
    if (side == Side::Left)
        return verify::gemm_backward_error(A.view(), tri, trans, B.view(), Shape::general, Transpose::NoTrans, C.view(), C.view(),
                                           Shape::general, verify::promoted_t<T>(1), verify::promoted_t<T>(0));
    return verify::gemm_backward_error(B.view(), Shape::general, Transpose::NoTrans, A.view(), tri, trans, C.view(), C.view(),
                                       Shape::general, verify::promoted_t<T>(1), verify::promoted_t<T>(0));
}

TYPED_TEST(TrmmTest, AllCombinations) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend Ba = TestFixture::BackendType;

    // keep the problem size small so that iterating over all parameter combinations is feasible
    const int n         = 512;
    const int batchSize = 4;

    // reuse one random B matrix for all permutations
    Matrix<T> B = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batchSize);
    Matrix<T> C = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batchSize);
    // loop over every combination of transpose, side, uplo and diagonal
    for (auto trans : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        for (auto side : {Side::Right, Side::Left}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                for (auto diag : {Diag::NonUnit, Diag::Unit}) {
                    // generate A for the current uplo/diag
                    Matrix<T> A = Matrix<T, MatrixFormat::Dense>::RandomTriangular(n, uplo, diag, batchSize);
                    

                    // compute C = trmm(A.view(),B.view()) with the current combination
                    trmm(*(this->ctx),
                             A.view(),
                             B.view(),
                             C.view(),
                             {.side = side, .uplo = uplo, .trans = trans, .diag = diag}).wait();

                    EXPECT_VERIFY(T, verify::Check::blas, n, trmm_error(A, B, C, side, uplo, trans, diag))
                        << "Failed combination: trans=" << static_cast<int>(trans)
                        << ", side=" << static_cast<int>(side)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", diag=" << static_cast<int>(diag);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#if BATCHLAS_HAS_CUDA_BACKEND
// BATCHLAS_TRMM_ROUTE takes only its own words; the removed legacy spellings (and any typo)
// throw rather than silently meaning Auto.
TEST(TrmmCudaCustomTest, RemovedRouteWordsThrow) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom trmm test requires a GPU device";
    }
    Matrix<float, MatrixFormat::Dense> A(16, 16, 2), B(16, 4, 2), C(16, 4, 2);
    for (const char* word : {"tiles", "narrow", "gemm", "custom", "dx", "fused", "diag_full_gemm", "triangular_tiles", "gram_tiles", "expand_gemm", "fused_device", "register_tiled", "native:auto", "vendor:auto", "bogus", "gram", "cublasdx"}) {
        ScopedEnvVar route("BATCHLAS_TRMM_ROUTE", word);
        EXPECT_THROW(trmm(ctx, A.view(), B.view(), C.view(), {.alpha = 1.0f}).wait(), std::invalid_argument) << word;
    }
}

#endif

// TRMM must not reference the opposite triangle of A, nor its diagonal when
// Diag::Unit is requested.
//
// AllCombinations builds A with RandomTriangular (zeros in the unreferenced half, ones on a unit
// diagonal), so it cannot tell a real trmm from a plain gemm. This test poisons the storage TRMM
// may not read; the host reference reads only the referenced triangle, so a read of the poison fails.
//
// A ragged dimension and a non-square B are in the shapes because the CUDA
// backend materialises the triangle into packed scratch with a leading
// dimension of its own (the `expand` family). The sweep runs twice there: capping the scratch budget
// at zero bytes sends Side::Right down the vendor loop it otherwise only reaches
// when the expansion will not fit on the device, which no test shape does (so vendor builds only).
TYPED_TEST(TrmmTest, IgnoresUnreferencedTriangleAndUnitDiagonal) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    struct Shape {
        int rows;
        int cols;
        int batch;
    };
    const Shape shapes[] = {{64, 48, 2}, {129, 96, 5}, {300, 32, 1}};

    auto sweep = [&](const char* route) {
        for (const auto& shape : shapes) {
            for (auto side : {Side::Left, Side::Right}) {
                const int k = side == Side::Left ? shape.rows : shape.cols;
                for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                    for (auto diag : {Diag::NonUnit, Diag::Unit}) {
                        for (auto trans : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                            auto A_clean = Matrix<T, MatrixFormat::Dense>::RandomTriangular(k, uplo, diag, shape.batch);
                            auto B = Matrix<T, MatrixFormat::Dense>::Random(shape.rows, shape.cols, false, shape.batch);
                            auto A_poisoned = A_clean.clone();
                            this->ctx->wait();

                            // Use the element accessor, not raw data() arithmetic: the
                            // storage has its own leading dimension and batch stride, and
                            // assuming ld == n silently writes into the referenced triangle.
                            for (int b = 0; b < shape.batch; ++b) {
                                for (int col = 0; col < k; ++col) {
                                    for (int row = 0; row < k; ++row) {
                                        const bool referenced =
                                            (uplo == Uplo::Lower) ? (row > col) : (row < col);
                                        if (referenced) continue;
                                        if (row == col && diag == Diag::NonUnit) continue;
                                        A_poisoned(row, col, b) = T(1000);
                                    }
                                }
                            }
                            this->ctx->wait();

                            auto C = Matrix<T, MatrixFormat::Dense>::Zeros(shape.rows, shape.cols, shape.batch);
                            trmm(*(this->ctx), A_poisoned.view(), B.view(), C.view(),
                                 {.side = side, .uplo = uplo, .trans = trans, .diag = diag})
                                .wait();

                            EXPECT_VERIFY(T, verify::Check::blas, k, trmm_error(A_poisoned, B, C, side, uplo, trans, diag))
                                << "trmm read storage it must not touch: " << route << " route, "
                                << shape.rows << "x" << shape.cols << " batch " << shape.batch
                                << " side=" << (side == Side::Left ? "Left" : "Right")
                                << " uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                << " trans=" << static_cast<int>(trans)
                                << " diag=" << (diag == Diag::Unit ? "Unit" : "NonUnit");
                        }
                    }
                }
            }
        }
    };

    sweep("default");

    if constexpr (TestFixture::BackendType == Backend::CUDA && select::level3_vendor_available<Backend::CUDA>) {
        ScopedEnvVar no_scratch("BATCHLAS_EXPAND_MAX_BYTES", "0");
        sweep("no-scratch");
    }
}