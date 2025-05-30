#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix_handle_new.hh> // Include the new matrix handle
#include <blas/cublas_matrixview.hh> // Include the header with MatrixView-compatible gemv
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace batchlas;
// Test fixture for BLAS operations
class BlasOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SYCL queue
        ctx = std::make_shared<Queue>(Device::default_device());
        
        // Initialize test matrices
        A_data = UnifiedVector<float>(rows * cols * batch_size);
        B_data = UnifiedVector<float>(cols * cols * batch_size);
        C_data = UnifiedVector<float>(rows * cols * batch_size, 0.0f);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    A_data[b * rows * cols + i * cols + j] = static_cast<float>(i * cols + j);
                }
            }
        }
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    B_data[b * cols * cols + i * cols + j] = static_cast<float>(i == j ? 1.0 : 0.0);
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
    UnifiedVector<float> A_data;
    UnifiedVector<float> B_data;
    UnifiedVector<float> C_data;

    void printMatrix(UnifiedVector<float>& matrix_data, int rows, int cols, int ld){
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                std::cout << matrix_data[i * ld + j] << " ";
            }
            std::cout << std::endl;
        }
    }
};

// Test GEMM operation using identity matrix (C = A * I = A)
TEST_F(BlasOperationsTest, GemmWithIdentityMatrix) {
    // Create matrix views with the new matrix handle format - using default template parameters
    MatrixView A_view(A_data.data(), rows, cols, ld);
    MatrixView B_view(B_data.data(), cols, cols, ld);
    MatrixView C_view(C_data.data(), rows, cols, ld);
    
    // Perform C = A * B (which should equal A since B is identity)
    gemm<Backend::CUDA>(*ctx, A_view, B_view, C_view, 1.0f, 0.0f,
                       Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    
    ctx->wait();
    
    // Verify result (C should be equal to A)
    for (size_t i = 0; i < rows*cols; ++i) {
        EXPECT_FLOAT_EQ(C_data[i], A_data[i]) << "Mismatch at index " << i;
    }
}

// Test batched GEMM operation
TEST_F(BlasOperationsTest, BatchedGemm) {
    // Create batched matrix views - using default template parameters
    MatrixView A_view(A_data.data(), rows, cols, ld, rows * cols, batch_size);
    MatrixView B_view(B_data.data(), rows, cols, ld, rows * cols, batch_size);
    MatrixView C_view(C_data.data(), rows, cols, ld, rows * cols, batch_size);
    
    // Adding the ComputePrecision parameter
    gemm<Backend::CUDA>(*ctx, A_view, B_view, C_view, 1.0f, 0.0f, 
                      Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    
    ctx->wait();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                EXPECT_FLOAT_EQ(C_data[b * rows * cols + i * cols + j], 
                                A_data[b * rows * cols + i * cols + j]) 
                << "Mismatch at batch " << b << ", index (" << i << ", " << j << ")";
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}