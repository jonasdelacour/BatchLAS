#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
#include <batchlas/verify/residuals.hh>
using namespace batchlas;
#if BATCHLAS_HAS_GPU_BACKEND

template <typename T>
class TransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }
    std::shared_ptr<Queue> ctx;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TransposeTest, TestTypes);

TYPED_TEST(TransposeTest, OrthoTransposeIdentity) {
    using T = TypeParam;
    constexpr int m = 8;
    constexpr int k = 4;
    constexpr int batch_size = 2;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(m, k, false, batch_size);
    size_t ws = ortho_buffer_size(*this->ctx, A.view(), Transpose::NoTrans, OrthoAlgorithm::SVQB);
    UnifiedVector<std::byte> workspace(ws);
    (void)ortho(*this->ctx, A.view(), Transpose::NoTrans, workspace.to_span(), OrthoAlgorithm::SVQB);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A.view());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Prod(k, k, batch_size);
    (void)gemm(*this->ctx, At.view(), A.view(), Prod.view(), {});
    this->ctx->wait();

    // Three facts make Prod the identity: the copy is exactly A^T, the gemm of the two is right, and
    // A is orthonormal (SVQB orthogonalizes the given columns directly, so Check::orthogonality).
    const auto items = batchlas::verify::all_items(batch_size);
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, m,
                  batchlas::verify::gemm_backward_error(At.view(), batchlas::verify::Shape::general, Transpose::NoTrans, A.view(),
                                                        batchlas::verify::Shape::general, Transpose::NoTrans, Prod.view(),
                                                        Prod.view(), batchlas::verify::Shape::general, 1.0, 0.0, items));
    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, m, batchlas::verify::orthogonality(A.view(), items));
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j) ASSERT_EQ(At(j, i, b), A(i, j, b)) << "batch " << b << " (" << i << "," << j << ")";
}

TYPED_TEST(TransposeTest, SimpleTranspose) {
    using T = TypeParam;
    constexpr int m = 256;
    constexpr int k = 128;
    constexpr int batch_size = 1024;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(m, k, false, batch_size);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A.view());
    this->ctx->wait();

    ASSERT_EQ(At.rows(), k);
    ASSERT_EQ(At.cols(), m);
    ASSERT_EQ(At.batch_size(), batch_size);

    // A transpose moves values and does no arithmetic: every element is bit-identical.
    for (int b = 0; b < batch_size; ++b)
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k; ++j) ASSERT_EQ(At(j, i, b), A(i, j, b)) << "batch " << b << " (" << i << "," << j << ")";
}

#endif // BATCHLAS_HAS_GPU_BACKEND
