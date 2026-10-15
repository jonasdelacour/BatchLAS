#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"

using namespace batchlas;

#if BATCHLAS_HAS_GPU_BACKEND
TEST(InverseTest, InverseIdentityCheck) {
    Queue ctx(Device::default_device());

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(40, 40, false, 2);
    const Matrix<float, MatrixFormat::Dense> A0 = A;

    Matrix<float, MatrixFormat::Dense> Ainverse(40,40,2);
    UnifiedVector<std::byte> ws(inv_buffer_size(ctx, A));
    (void)inv(ctx, A, Ainverse, ws);
    ctx.wait();

    // ||A X - I||_F / (||A||_F ||X||_F): an empty B0 is the identity.
    using MV = MatrixView<float, MatrixFormat::Dense>;
    EXPECT_VERIFY(float, verify::Check::solve, 40,
                  MV::solve_residual(ctx, A0, Ainverse, MV(nullptr, 0, 0, 1), A0.all_items()));
}
#endif // BATCHLAS_HAS_GPU_BACKEND
