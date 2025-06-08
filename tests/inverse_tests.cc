#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>

using namespace batchlas;

TEST(InverseTest, InverseIdentityCheck) {
    auto ctx = std::make_shared<Queue>(Device::default_device());

    Matrix<float, MatrixFormat::Dense> A(2,2,1);
    auto Adata = A.data();
    Adata[0] = 4.0f; Adata[1] = 2.0f; // column 0
    Adata[2] = 7.0f; Adata[3] = 6.0f; // column 1

    Matrix<float, MatrixFormat::Dense> Ainverse(2,2,1);
    UnifiedVector<std::byte> ws(inv_buffer_size<Backend::CUDA>(*ctx, A.view()));
    inv<Backend::CUDA>(*ctx, A.view(), Ainverse.view(), ws);
    ctx->wait();

    Matrix<float, MatrixFormat::Dense> result(2,2,1);
    gemm<Backend::CUDA>(*ctx, A.view(), Ainverse.view(), result.view(), 1.0f, 0.0f,
                        Transpose::NoTrans, Transpose::NoTrans);
    ctx->wait();

    auto r = result.data();
    EXPECT_NEAR(r[0], 1.0f, 1e-3f);
    EXPECT_NEAR(r[1], 0.0f, 1e-3f);
    EXPECT_NEAR(r[2], 0.0f, 1e-3f);
    EXPECT_NEAR(r[3], 1.0f, 1e-3f);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
