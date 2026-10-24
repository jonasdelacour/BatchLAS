// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/backend_config.h>
#include <tests/test_utils.hh>
#include <batchlas/verify/tolerance.hh>
using namespace batchlas;
#if BATCHLAS_HAS_GPU_BACKEND

template <typename T>
class TransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // A GPU without fp64 (Metal) has no double kernels: double runs on the host, as NETLIB.
        if constexpr (test_utils::native_gpu_lacks_fp64 && !std::is_same_v<T, float>) {
#if !BATCHLAS_HAS_CPU_TARGET
            GTEST_SKIP() << "the GPU has no fp64 and the build no CPU target";
#endif
            ctx = std::make_shared<Queue>(Device("cpu"), Backend::NETLIB, true);
        } else {
            ctx = std::make_shared<Queue>(Device::default_device());
        }
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
    size_t ws = ortho_buffer_size(*this->ctx, A, Transpose::NoTrans, OrthoAlgorithm::SVQB);
    UnifiedVector<std::byte> workspace(ws);
    (void)ortho(*this->ctx, A, Transpose::NoTrans, workspace.to_span(), OrthoAlgorithm::SVQB);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A);
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Prod(k, k, batch_size);
    (void)gemm(*this->ctx, At, A, Prod, {});
    this->ctx->wait();

    // Three facts make Prod the identity: the copy is exactly A^T, the gemm of the two is right, and
    // A is orthonormal (SVQB orthogonalizes the given columns directly, so Check::orthogonality).
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const auto items = A.all_items();
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, m,
                  MV::gemm_backward_error(*this->ctx, At, Transpose::NoTrans, A, Transpose::NoTrans, Prod,
                                          Prod, 1.0, 0.0, items));
    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, m, MV::orthogonality(*this->ctx, A, items));
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

    Matrix<T, MatrixFormat::Dense> At = transpose(*this->ctx, A);
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
