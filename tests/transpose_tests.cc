#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <blas/extra.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"
using namespace batchlas;
#if BATCHLAS_HAS_GPU_BACKEND

template <typename T>
class TransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }
    std::shared_ptr<Queue> ctx;

    static auto tolerance() {
        return test_utils::tolerance<T>();
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TransposeTest, TestTypes);

TYPED_TEST(TransposeTest, OrthoTransposeIdentity) {
    using T = TypeParam;
    constexpr int m = 8;
    constexpr int k = 4;
    constexpr int batch_size = 2;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(m, k, false, batch_size);
    size_t ws = ortho_buffer_size<test_utils::gpu_backend, T>(*this->ctx, A.view(), Transpose::NoTrans, OrthoAlgorithm::SVQB);
    UnifiedVector<std::byte> workspace(ws);
    ortho<test_utils::gpu_backend, T>(*this->ctx, A.view(), Transpose::NoTrans, workspace.to_span(), OrthoAlgorithm::SVQB);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A.view());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Prod(k, k, batch_size);
    gemm<test_utils::gpu_backend>(*this->ctx, At.view(), A.view(), Prod.view(), T(1.0), T(0.0),
                       Transpose::NoTrans, Transpose::NoTrans);
    this->ctx->wait();

    auto prod_data = Prod.data();
    int stride = Prod.stride();
    T tol = TestFixture::tolerance();
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < k; ++j) {
                T expected = (i == j) ? T(1) : T(0);
                T val = prod_data[b * stride + j * Prod.ld() + i];
                ASSERT_NEAR(val, expected, tol);
            }
        }
    }
}

TYPED_TEST(TransposeTest, SimpleTranspose) {
    using T = TypeParam;
    constexpr int m = 8;
    constexpr int k = 4;
    constexpr int batch_size = 2;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(m, k, false, batch_size);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A.view());
    this->ctx->wait();

    ASSERT_EQ(At.rows(), k);
    ASSERT_EQ(At.cols(), m);
    ASSERT_EQ(At.batch_size(), batch_size);

    auto a_data = A.data();
    auto at_data = At.data();
    int a_stride = A.stride();
    int at_stride = At.stride();
    T tol = TestFixture::tolerance();

    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                T val_a = a_data[b * a_stride + j * A.ld() + i];
                T val_at = at_data[b * at_stride + i * At.ld() + j];
                ASSERT_NEAR(val_a, val_at, tol);
            }
        }
    }
}

#endif // BATCHLAS_HAS_GPU_BACKEND
