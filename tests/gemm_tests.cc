#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cstdlib>
#include <string>

#include "../src/sycl/gemm_kernels.hh"
#include "test_utils.hh"

using namespace batchlas;

namespace {

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        const char* old = std::getenv(name_);
        if (old) {
            had_old_ = true;
            old_value_ = old;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_old_ = false;
    std::string old_value_;
};

template <typename T>
::testing::AssertionResult AssertBatchedBufferNear(const UnifiedVector<T>& actual,
                                                   const UnifiedVector<T>& expected,
                                                   size_t rows,
                                                   size_t cols,
                                                   size_t batch_size,
                                                   typename batchlas::base_type<T>::type tol) {
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t row = 0; row < rows; ++row) {
            for (size_t col = 0; col < cols; ++col) {
                const size_t index = b * rows * cols + row * cols + col;
                const auto actual_value = actual[index];
                const auto expected_value = expected[index];
                if constexpr (test_utils::is_complex<T>::value) {
                    if (std::abs(actual_value.real() - expected_value.real()) > tol ||
                        std::abs(actual_value.imag() - expected_value.imag()) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                } else {
                    if (std::abs(actual_value - expected_value) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                }
            }
        }
    }

    return ::testing::AssertionSuccess();
}

template <typename T>
::testing::AssertionResult AssertBatchedMatrixNear(const Matrix<T>& actual,
                                                   const Matrix<T>& expected,
                                                   int rows,
                                                   int cols,
                                                   int batch_size,
                                                   typename batchlas::base_type<T>::type tol) {
    for (int b = 0; b < batch_size; ++b) {
        for (int col = 0; col < cols; ++col) {
            for (int row = 0; row < rows; ++row) {
                const auto actual_value = actual(row, col, b);
                const auto expected_value = expected(row, col, b);
                if constexpr (test_utils::is_complex<T>::value) {
                    if (std::abs(actual_value.real() - expected_value.real()) > tol ||
                        std::abs(actual_value.imag() - expected_value.imag()) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                } else {
                    if (std::abs(actual_value - expected_value) > tol) {
                        return ::testing::AssertionFailure()
                               << "mismatch at batch=" << b << ", row=" << row << ", col=" << col
                               << ": actual=" << actual_value << ", expected=" << expected_value << ", tol=" << tol;
                    }
                }
            }
        }
    }

    return ::testing::AssertionSuccess();
}

template <typename ScalarType, Backend BackendType>
void RunForcedSyclGemmKernelCompare(Queue& ctx,
                                    const char* kernel_name,
                                    int m,
                                    int n,
                                    int k,
                                    int batch_size,
                                    Transpose transA,
                                    Transpose transB,
                                    typename batchlas::base_type<ScalarType>::type tol_scale = 75) {
    const int a_rows = transA == Transpose::NoTrans ? m : k;
    const int a_cols = transA == Transpose::NoTrans ? k : m;
    const int b_rows = transB == Transpose::NoTrans ? k : n;
    const int b_cols = transB == Transpose::NoTrans ? n : k;

    auto A = Matrix<ScalarType>::Random(a_rows, a_cols, false, batch_size);
    auto B = Matrix<ScalarType>::Random(b_rows, b_cols, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", kernel_name);
        gemm<BackendType>(ctx, A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          transA, transB, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(ctx, A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          transA, transB, ComputePrecision::Default);
    }

    ctx.wait();

    auto tol = test_utils::tolerance<ScalarType>() * tol_scale;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

batchlas::sycl_gemm::KernelVariant SelectSyclKernelVariantForTest(int m,
                                                                  int n,
                                                                  int k,
                                                                  Transpose transA,
                                                                  Transpose transB,
                                                                  int a_ld = 0,
                                                                  int b_ld = 0,
                                                                  int c_ld = 0) {
    const int a_rows = transA == Transpose::NoTrans ? m : k;
    const int a_cols = transA == Transpose::NoTrans ? k : m;
    const int b_rows = transB == Transpose::NoTrans ? k : n;
    const int b_cols = transB == Transpose::NoTrans ? n : k;

    Matrix<float> A(a_rows, a_cols, 1, a_ld);
    Matrix<float> B(b_rows, b_cols, 1, b_ld);
    Matrix<float> C(m, n, 1, c_ld);

    return batchlas::sycl_gemm::select_kernel_variant<float>(A.view(), B.view(), C.view(), transA, transB);
}

} // namespace

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using GemmTestTypes = typename test_utils::backend_types<TestConfig>::type;

template <typename Config>
class GemmTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    const int rows = 10;
    const int cols = 10;
    const int ld = 10;
    const int batch_size = 3;
    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> B_data;
    UnifiedVector<ScalarType> C_data;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        
        if (!this->ctx) {
            return;
        }

        // Initialize test matrices
        A_data = UnifiedVector<ScalarType>(rows * cols * batch_size);
        B_data = UnifiedVector<ScalarType>(cols * cols * batch_size);
        C_data = UnifiedVector<ScalarType>(rows * cols * batch_size, static_cast<ScalarType>(0));
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    A_data[b * rows * cols + i * cols + j] = static_cast<ScalarType>(i * cols + j);
                }
            }
        }
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    B_data[b * cols * cols + i * cols + j] = static_cast<ScalarType>(i == j ? 1.0 : 0.0);
                }
            }
        }
    }

    void printMatrix(UnifiedVector<ScalarType>& matrix_data, int rows, int cols, int ld){
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                std::cout << matrix_data[i * ld + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

TYPED_TEST_SUITE(GemmTest, GemmTestTypes);

TEST(GemmDispatchPolicyTest, SelectsAlignedS2U1ForSmallSquareFloatNN) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(128, 128, 128, Transpose::NoTrans, Transpose::NoTrans),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK32S2U1Aligned);
}

TEST(GemmDispatchPolicyTest, SelectsAlignedS2U1ForMediumSquareFloatNN) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(256, 256, 256, Transpose::NoTrans, Transpose::NoTrans),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK32S2U1Aligned);
}

TEST(GemmDispatchPolicyTest, SelectsS2U2ForLargeSquareFloatNN) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(512, 512, 512, Transpose::NoTrans, Transpose::NoTrans),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK32S2U2);
}

TEST(GemmDispatchPolicyTest, SelectsGenericS2U1ForMisalignedSquareFloatNN) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(256, 256, 256, Transpose::NoTrans, Transpose::NoTrans, 272),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK32S2U1Generic);
}

TEST(GemmDispatchPolicyTest, KeepsTransposeHeavyCasesOnK32TransposeAlias) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(256, 128, 256, Transpose::Trans, Transpose::NoTrans),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK32TN);
}

TEST(GemmDispatchPolicyTest, KeepsSkinnyTallNNOnLegacyK16PathUntilBenchmarked) {
    EXPECT_EQ(SelectSyclKernelVariantForTest(512, 64, 512, Transpose::NoTrans, Transpose::NoTrans),
              batchlas::sycl_gemm::KernelVariant::Tiled128x32RegisterK16);
}

// Test GEMM operation using identity matrix (C = A * I = A)
TYPED_TEST(GemmTest, GemmWithIdentityMatrix) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    // Create matrix views with the new matrix handle format - using default template parameters
    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->ld);
    MatrixView<ScalarType, MatrixFormat::Dense> B_view(this->B_data.data(), this->cols, this->cols, this->ld);
    MatrixView<ScalarType, MatrixFormat::Dense> C_view(this->C_data.data(), this->rows, this->cols, this->ld);
    
    // Perform C = A * B (which should equal A since B is identity)
    gemm<BackendType>(*(this->ctx), A_view, B_view, C_view, ScalarType(1.0), ScalarType(0.0),
                       Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>();
    ASSERT_TRUE(AssertBatchedBufferNear(this->C_data, this->A_data, this->rows, this->cols, 1, tol));
}

// Test batched GEMM operation
TYPED_TEST(GemmTest, BatchedGemm) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    // Create batched matrix views - using default template parameters
    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    MatrixView<ScalarType, MatrixFormat::Dense> B_view(this->B_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    MatrixView<ScalarType, MatrixFormat::Dense> C_view(this->C_data.data(), this->rows, this->cols, this->ld, this->rows * this->cols, this->batch_size);
    
    // Adding the ComputePrecision parameter
    gemm<BackendType>(*(this->ctx), A_view, B_view, C_view, ScalarType(1.0), ScalarType(0.0),
                      Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    
    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>();
    ASSERT_TRUE(AssertBatchedBufferNear(this->C_data, this->A_data, this->rows, this->cols, this->batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariant) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");

    constexpr int size = 32;
    constexpr int batch_size = 4;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Zeros(size, size, batch_size);
    auto C_ref = Matrix<ScalarType>::Zeros(size, size, batch_size);

    gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(0),
                      Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);

    ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
    gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(0),
                      Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantLargeSquare) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;
    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64 SYCL register kernel is only selected for float in this first slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg64");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64x16 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg64k16");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k16");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32s2u1");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U2Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u2 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32s2u2");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S1U1Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s1_u1 SYCL register kernel is only selected for float in this slice";
    }

    if constexpr (BackendType == Backend::CUDA) {
        GTEST_SKIP() << "128x32x32_s1_u1 is experimental-only until the single-stage K32 path is correct on CUDA";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s1_u1",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1AlignedKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1_aligned SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u1_aligned",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1GenericKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1_generic SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u1_generic",
                                                            130, 96, 130, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1LegacyAliasGenericFallback) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1 legacy alias is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u1",
                                                            130, 96, 130, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U2TT8x4Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u2_tt8x4 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u2_tt8x4",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U2TT4x8Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u2_tt4x8 SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u2_tt4x8",
                                                            128, 128, 128, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32PersistentKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_persistent SYCL register kernel is only selected for float in this slice";
    }

    ScopedEnvVar experimental("BATCHLAS_GEMM_EXPERIMENTAL", "1");
    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_persistent",
                                                            256, 256, 256, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32SplitK4Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_splitk4 SYCL register kernel is only selected for float in this slice";
    }

    ScopedEnvVar experimental("BATCHLAS_GEMM_EXPERIMENTAL", "1");
    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_splitk4",
                                                            256, 256, 256, 2,
                                                            Transpose::NoTrans, Transpose::NoTrans,
                                                            100);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16Kernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 128;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg32x128k16");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantTransposed) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int m = 24;
    constexpr int n = 20;
    constexpr int k = 16;
    constexpr int batch_size = 3;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclTiledVariantLargeTransposed) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int m = 96;
    constexpr int n = 80;
    constexpr int k = 64;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "tiled16");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 TT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k16tt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 TN SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k16tn");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K16NTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x16 NT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(m, k, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k16nt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 TN SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32tn");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32S2U1TNCanonicalAlias) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32_s2_u1_tn SYCL register kernel is only selected for float in this slice";
    }

    RunForcedSyclGemmKernelCompare<ScalarType, BackendType>(*(this->ctx), "128x32x32_s2_u1_tn",
                                                            128, 128, 128, 2,
                                                            Transpose::Trans, Transpose::NoTrans);
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32NTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 NT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(m, k, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32nt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16TNKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 TN SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg32x128k16tn");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister32x128K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "32x128x16 TT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg32x128k16tt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister64x64K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "64x64x16 TT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg64k16tt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x64K16TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x64x16 TT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x64k16tt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x32K32TTKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x32x32 TT SYCL register kernel is only selected for float in this slice";
    }

    constexpr int m = 128;
    constexpr int n = 128;
    constexpr int k = 128;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(n, k, false, batch_size);
    auto C = Matrix<ScalarType>::Random(m, n, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x32k32tt");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::Trans, Transpose::Trans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 75;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclRegister128x64K32LargeKernel) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    if constexpr (!std::is_same_v<ScalarType, float>) {
        GTEST_SKIP() << "128x64x32 large SYCL register kernel is only selected for float in this slice";
    }

    constexpr int size = 256;
    constexpr int batch_size = 2;
    auto A = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto B = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C = Matrix<ScalarType>::Random(size, size, false, batch_size);
    auto C_ref = C.clone();

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        ScopedEnvVar force_kernel("BATCHLAS_GEMM_SYCL_KERNEL", "reg128x64k32large");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(1),
                          Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 100;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, size, size, batch_size, tol));
}

TYPED_TEST(GemmTest, BatchedGemmForcedSyclVariantConjugateTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr Backend BackendType = TestFixture::BackendType;

    constexpr int m = 18;
    constexpr int n = 14;
    constexpr int k = 12;
    constexpr int batch_size = 2;

    auto A = Matrix<ScalarType>::Random(k, m, false, batch_size);
    auto B = Matrix<ScalarType>::Random(k, n, false, batch_size);
    auto C = Matrix<ScalarType>::Zeros(m, n, batch_size);
    auto C_ref = Matrix<ScalarType>::Zeros(m, n, batch_size);

    {
        ScopedEnvVar force_variant("BATCHLAS_GEMM_VARIANT", "sycl");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C.view(), ScalarType(1), ScalarType(0),
                          Transpose::ConjTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    {
        ScopedEnvVar vendor_variant("BATCHLAS_GEMM_VARIANT", "vendor");
        gemm<BackendType>(*(this->ctx), A.view(), B.view(), C_ref.view(), ScalarType(1), ScalarType(0),
                          Transpose::ConjTrans, Transpose::NoTrans, ComputePrecision::Default);
    }

    this->ctx->wait();

    auto tol = test_utils::tolerance<ScalarType>() * 50;
    ASSERT_TRUE(AssertBatchedMatrixNear(C, C_ref, m, n, batch_size, tol));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
