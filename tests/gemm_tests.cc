#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>

using namespace batchlas;

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"

using GemmTestTypes = typename test_utils::backend_types<TestConfig>::type;

template <typename Config>
class GemmTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        // Create a SYCL queue based on BackendType
        if constexpr (BackendType != Backend::NETLIB) {
            try {
                ctx = std::make_shared<Queue>("gpu");
                if (!(ctx->device().type == DeviceType::GPU)) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL did not select a GPU device. Skipping.";
                }
            } catch (const sycl::exception& e) {
                if (e.code() == sycl::errc::runtime || e.code() == sycl::errc::feature_not_supported) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL GPU queue creation failed: " << e.what() << ". Skipping.";
                } else {
                    throw;
                }
            } catch (const std::exception& e) {
                GTEST_SKIP() << "CUDA backend selected, but Queue construction failed: " << e.what() << ". Skipping.";
            }
        } else {
            ctx = std::make_shared<Queue>("cpu");
        }
        if (!ctx) {
            GTEST_FAIL() << "Queue context is null after setup.";
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
    
    void TearDown() override {
    }
    
    std::shared_ptr<Queue> ctx;
    const int rows = 10;
    const int cols = 10;
    const int ld = 10;
    const int batch_size = 5;
    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> B_data;
    UnifiedVector<ScalarType> C_data;

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
    
    // Verify result (C should be equal to A)
    ScalarType tol = std::is_same_v<ScalarType, float> ? ScalarType(1e-5) : ScalarType(1e-10);
    for (size_t i = 0; i < this->rows*this->cols; ++i) {
        EXPECT_NEAR(this->C_data[i], this->A_data[i], tol) << "Mismatch at index " << i;
    }
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

    ScalarType tol = std::is_same_v<ScalarType, float> ? ScalarType(1e-5) : ScalarType(1e-10);
    for (size_t b = 0; b < this->batch_size; ++b) {
        for (size_t i = 0; i < this->rows; ++i) {
            for (size_t j = 0; j < this->cols; ++j) {
                EXPECT_NEAR(this->C_data[b * this->rows * this->cols + i * this->cols + j],
                            this->A_data[b * this->rows * this->cols + i * this->cols + j], tol)
                << "Mismatch at batch " << b << ", index (" << i << ", " << j << ")";
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
