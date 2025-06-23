#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"

using namespace batchlas;

#if BATCHLAS_HAS_GPU_BACKEND
#if BATCHLAS_HAS_GPU_BACKEND
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(40, 40, false, 2);
    auto Adata = A.data();
    
    Matrix<float, MatrixFormat::Dense> Ainverse(40,40,2);
    UnifiedVector<std::byte> ws(inv_buffer_size<test_utils::gpu_backend>(ctx, A.view()));
    inv<test_utils::gpu_backend>(ctx, A.view(), Ainverse.view(), ws);
    ctx.wait();

    Matrix<float, MatrixFormat::Dense> result(40,40,2);
    gemm<test_utils::gpu_backend>(ctx, A.view(), Ainverse.view(), result.view(), 1.0f, 0.0f,
                        Transpose::NoTrans, Transpose::NoTrans);
    ctx.wait();

    auto r = result.data();
    for (int b = 0; b < result.batch_size(); ++b) {
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) {
                if (i == j) {
                    EXPECT_NEAR(r[b * result.stride() + i * result.ld() + j], 1.0f, 1e-4);
                } else {
                    EXPECT_NEAR(r[b * result.stride() + i * result.ld() + j], 0.0f, 1e-4);
                }
            }
        }
    }
}
#endif // BATCHLAS_HAS_GPU_BACKEND

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
