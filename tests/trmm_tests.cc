// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include <tests/test_utils.hh>
#include <tests/test_support/storage.hh>
#include <src/select/vendor.hh>

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
double trmm_error(Queue& ctx, const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C, Side side, Uplo uplo, Transpose trans,
                  Diag diag) {
    using MV = MatrixView<T, MatrixFormat::Dense>;
    auto tri = A.clone();
    tri.triangularize(ctx, uplo, diag).wait();
    if (side == Side::Left)
        return MV::gemm_backward_error(ctx, tri, trans, B, Transpose::NoTrans, C, C, 1.0, 0.0);
    return MV::gemm_backward_error(ctx, B, Transpose::NoTrans, tri, trans, C, C, 1.0, 0.0);
}

// Every (trans, side, uplo, diag) at an odd order on padded storage: ld = n + 3 and a gap between
// items, every pad holding the poison, so a read through a natural ld or into a gap fails. The first
// call's C is all poison (beta = 0 must not read it). One A per (uplo, diag); complex T, complex data.
TYPED_TEST(TrmmTest, AllCombinations) {
    using T = typename TestFixture::ScalarType;
    auto& ctx = *this->ctx;
    const int n = 129;
    const int batchSize = 4;

    Matrix<T> B = test_support::padded<T>(n, n, batchSize);
    MatrixView<T>::copy(ctx, B, Matrix<T>::Random(n, n, false, batchSize)).wait();
    Matrix<T> A = test_support::padded<T>(n, n, batchSize);
    Matrix<T> C = test_support::padded<T>(n, n, batchSize);
    for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
        for (auto diag : {Diag::NonUnit, Diag::Unit}) {
            MatrixView<T>::copy(ctx, A, Matrix<T>::RandomTriangular(n, uplo, diag, batchSize)).wait();
            for (auto trans : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                for (auto side : {Side::Right, Side::Left}) {
                    trmm(ctx, A, B, C, {.side = side, .uplo = uplo, .trans = trans, .diag = diag}).wait();
                    EXPECT_VERIFY(T, verify::Check::blas, n, trmm_error(ctx, A, B, C, side, uplo, trans, diag))
                        << "trans=" << static_cast<int>(trans) << " side=" << static_cast<int>(side)
                        << " uplo=" << static_cast<int>(uplo) << " diag=" << static_cast<int>(diag);
                }
            }
        }
    }
}

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
                            trmm(*(this->ctx), A_poisoned, B, C,
                                 {.side = side, .uplo = uplo, .trans = trans, .diag = diag})
                                .wait();

                            EXPECT_VERIFY(T, verify::Check::blas, k, trmm_error(*(this->ctx), A_poisoned, B, C, side, uplo, trans, diag))
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