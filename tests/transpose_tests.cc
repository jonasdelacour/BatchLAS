#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <blas/extra.hh>
using namespace batchlas;

template <typename T>
class TransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }
    std::shared_ptr<Queue> ctx;

    static T tolerance() {
        if constexpr (std::is_same_v<T, float>) return T(1e-5);
        else return T(1e-9);
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
    size_t ws = ortho_buffer_size<Backend::CUDA, T>(*this->ctx, A.view(), Transpose::NoTrans);
    UnifiedVector<std::byte> workspace(ws);
    ortho<Backend::CUDA, T>(*this->ctx, A.view(), Transpose::NoTrans, workspace.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A.view());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Prod(k, k, batch_size);
    gemm<Backend::CUDA>(*this->ctx, At.view(), A.view(), Prod.view(), T(1.0), T(0.0),
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

