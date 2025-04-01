#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace batchlas;

// Test fixture for GEMV operations
class GemvOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SYCL queue
        ctx = std::make_shared<Queue>(Device::default_device());
        
        // Initialize test matrices and vectors
        A_data = UnifiedVector<float>(std::max(rows,cols) * cols * batch_size);
        x_data = UnifiedVector<float>(std::max(cols,cols) * batch_size);
        y_data = UnifiedVector<float>(std::max(rows,cols) * batch_size, 0.0f);
        y_expected = UnifiedVector<float>(std::max(rows,cols) * batch_size, 0.0f);
        
        // Initialize matrix with deterministic values
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    A_data[b * rows * cols + i * cols + j] = static_cast<float>(i + j + 1);
                }
            }
        }
        
        // Initialize x vector with sequential values
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < std::max(rows,cols); ++j) {
                x_data[b * std::max(cols,cols) + j] = static_cast<float>(j + 1);
            }
        }
    }
    
    void TearDown() override {
    }

    // Helper function to compute expected y = alpha*A*x + beta*y
    void computeExpectedGemv(float alpha, float beta, bool transpose) {
        auto  vec_stride = std::max(rows,cols);
        for (int b = 0; b < batch_size; ++b) {
            // For non-transposed case (y = alpha*A*x + beta*y)
            if (!transpose) {
                for (int i = 0; i < rows; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        // Column-major layout: A[j*rows + i]
                        sum += A_data[b * rows * cols + j * rows + i] * x_data[b * vec_stride + j];
                    }
                    y_expected[b * vec_stride + i] = alpha * sum + beta * y_data[b * vec_stride + i];
                }
            }
            // For transposed case (y = alpha*A^T*x + beta*y)
            else {
                for (int i = 0; i < rows; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        // Transpose of column-major is accessing A as A[i*rows + j]
                        sum += A_data[b * rows * cols + j * rows + i] * x_data[b * vec_stride + j];
                    }
                    y_expected[b * vec_stride + i] = alpha * sum + beta * y_data[b * vec_stride + i];
                }
            }
        }
    }
    
    std::shared_ptr<Queue> ctx;
    const int rows = 10;
    const int cols = 10;
    const int batch_size = 5;
    UnifiedVector<float> A_data;
    UnifiedVector<float> x_data;
    UnifiedVector<float> y_data;
    UnifiedVector<float> y_expected;
};

// Test single GEMV operation with no transpose
TEST_F(GemvOperationsTest, SingleGemvNoTranspose) {
    // Create vector and matrix handles
    DenseMatView<float, BatchType::Single> A_view(A_data.data(), rows, cols, rows);
    DenseVecHandle<float, BatchType::Single> x_vec(x_data.data(), cols, 1);
    DenseVecHandle<float, BatchType::Single> y_vec(y_data.data(), rows, 1);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute expected result (y = alpha * A * x + beta * y)
    computeExpectedGemv(alpha, beta, false);

    std::cout << A_view << std::endl;
    
    // Perform y = alpha*A*x + beta*y
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result
    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y_data[i], y_expected[i], 1e-5f) 
            << "Mismatch at index " << i;
    }
}

// Test single GEMV operation with transpose
TEST_F(GemvOperationsTest, SingleGemvWithTranspose) {
    // Create vector and matrix handles
    DenseMatView<float, BatchType::Single> A_view(A_data.data(), rows, cols, rows);
    DenseVecHandle<float, BatchType::Single> x_vec(x_data.data(), cols, 1);
    DenseVecHandle<float, BatchType::Single> y_vec(y_data.data(), rows, 1);
    
    float alpha = 2.0f;
    float beta = 0.0f;
    
    // Compute expected result (y = alpha * A^T * x + beta * y)
    computeExpectedGemv(alpha, beta, true);
    
    // Perform y = alpha*A^T*x + beta*y
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::Trans);
    
    ctx->wait();
    
    // Verify result
    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y_data[i], y_expected[i], 1e-5f) 
        << "Mismatch with transpose at index " << i;
    }
}
        
/* 
// Test single GEMV operation with non-zero beta
TEST_F(GemvOperationsTest, SingleGemvWithBeta) {
    // Create vector and matrix handles
    DenseMatView<float, BatchType::Single> A_view(A_data.data(), rows, cols, rows);
    DenseVecHandle<float, BatchType::Single> x_vec(x_data.data(), cols, 1);
    DenseVecHandle<float, BatchType::Single> y_vec(y_data.data(), rows, 1);
    
    // Initialize y with some values
    for (int i = 0; i < rows; ++i) {
        y_data[i] = static_cast<float>(i * 2);
    }
    
    float alpha = 1.0f;
    float beta = 1.5f;
    
    // Compute expected result (y = alpha * A * x + beta * y)
    computeExpectedGemv(alpha, beta, false);
    
    // Perform y = alpha*A*x + beta*y
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result
    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y_data[i], y_expected[i], 1e-5f) 
            << "Mismatch with beta at index " << i;
    }
}

// Test batched GEMV operation
TEST_F(GemvOperationsTest, BatchedGemv) {
    // Create batched handles
    DenseMatHandle<float, BatchType::Batched> A_handle(A_data.data(), rows, cols, rows, rows * cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> x_vec(x_data.data(), cols, 1, cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> y_vec(y_data.data(), rows, 1, rows, batch_size);
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Compute expected result (y = alpha * A * x + beta * y) for each batch
    computeExpectedGemv(alpha, beta, false);
    
    // Perform batched gemv
    gemv<Backend::CUDA>(*ctx, A_handle(), x_vec, y_vec, alpha, beta, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_NEAR(y_data[b * rows + i], y_expected[b * rows + i], 1e-5f) 
                << "Mismatch at batch " << b << ", index " << i;
        }
    }
}

// Test batched GEMV operation with transpose
TEST_F(GemvOperationsTest, BatchedGemvWithTranspose) {
    // Create batched handles
    DenseMatHandle<float, BatchType::Batched> A_handle(A_data.data(), rows, cols, rows, rows * cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> x_vec(x_data.data(), cols, 1, cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> y_vec(y_data.data(), rows, 1, rows, batch_size);
    
    float alpha = 2.5f;
    float beta = 0.0f;
    
    // Compute expected result (y = alpha * A^T * x + beta * y) for each batch
    computeExpectedGemv(alpha, beta, true);
    
    // Perform batched gemv with transpose
    gemv<Backend::CUDA>(*ctx, A_handle(), x_vec, y_vec, alpha, beta, Transpose::Trans);
    
    ctx->wait();
    
    // Verify result for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_NEAR(y_data[b * rows + i], y_expected[b * rows + i], 1e-5f) 
                << "Mismatch with transpose at batch " << b << ", index " << i;
        }
    }
}

// Test both alpha and beta in batched GEMV
TEST_F(GemvOperationsTest, BatchedGemvWithAlphaBeta) {
    // Create batched handles
    DenseMatHandle<float, BatchType::Batched> A_handle(A_data.data(), rows, cols, rows, rows * cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> x_vec(x_data.data(), cols, 1, cols, batch_size);
    DenseVecHandle<float, BatchType::Batched> y_vec(y_data.data(), rows, 1, rows, batch_size);
    
    // Initialize y with some values
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            y_data[b * rows + i] = static_cast<float>(b * 10 + i);
        }
    }
    
    float alpha = 1.5f;
    float beta = 0.8f;
    
    // Compute expected result (y = alpha * A * x + beta * y) for each batch
    computeExpectedGemv(alpha, beta, false);
    
    // Perform batched gemv with alpha and beta
    gemv<Backend::CUDA>(*ctx, A_handle(), x_vec, y_vec, alpha, beta, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_NEAR(y_data[b * rows + i], y_expected[b * rows + i], 1e-5f) 
                << "Mismatch with alpha/beta at batch " << b << ", index " << i;
        }
    }
}

// Test for edge case with very small matrix
TEST_F(GemvOperationsTest, SmallMatrixGemv) {
    const int small_rows = 2;
    const int small_cols = 3;
    
    // Create matrices with explicit initialization
    UnifiedVector<float> small_A(small_rows * small_cols);
    // Row 1
    small_A[0] = 1.0f;
    small_A[1] = 2.0f;
    small_A[2] = 3.0f;
    // Row 2
    small_A[3] = 4.0f;
    small_A[4] = 5.0f;
    small_A[5] = 6.0f;
    
    UnifiedVector<float> small_x(small_cols);
    small_x[0] = 1.0f;
    small_x[1] = 2.0f;
    small_x[2] = 3.0f;
    
    UnifiedVector<float> small_y(small_rows, 0.0f);
    
    UnifiedVector<float> expected(small_rows);
    expected[0] = 14.0f; // A*x calculated manually
    expected[1] = 32.0f;
    
    DenseMatView<float, BatchType::Single> A_view(small_A.data(), small_rows, small_cols, small_rows);
    DenseVecHandle<float, BatchType::Single> x_vec(small_x.data(), small_cols, 1);
    DenseVecHandle<float, BatchType::Single> y_vec(small_y.data(), small_rows, 1);
    
    // Perform y = A*x
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, 1.0f, 0.0f, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result
    for (int i = 0; i < small_rows; ++i) {
        EXPECT_NEAR(small_y[i], expected[i], 1e-5f) 
            << "Mismatch in small matrix test at index " << i;
    }
}

// Test double precision
TEST_F(GemvOperationsTest, DoublePrecisionGemv) {
    const int dp_rows = 4;
    const int dp_cols = 3;
    
    // Create matrices with explicit initialization
    UnifiedVector<double> dp_A(dp_rows * dp_cols);
    // Row 1
    dp_A[0] = 1.0;
    dp_A[1] = 2.0;
    dp_A[2] = 3.0;
    // Row 2
    dp_A[3] = 4.0;
    dp_A[4] = 5.0;
    dp_A[5] = 6.0;
    // Row 3
    dp_A[6] = 7.0;
    dp_A[7] = 8.0;
    dp_A[8] = 9.0;
    // Row 4
    dp_A[9] = 10.0;
    dp_A[10] = 11.0;
    dp_A[11] = 12.0;
    
    UnifiedVector<double> dp_x(dp_cols);
    dp_x[0] = 1.0;
    dp_x[1] = 2.0;
    dp_x[2] = 3.0;
    
    UnifiedVector<double> dp_y(dp_rows, 0.0);
    
    UnifiedVector<double> expected(dp_rows);
    expected[0] = 14.0; // A*x calculated manually
    expected[1] = 32.0;
    expected[2] = 50.0;
    expected[3] = 68.0;
    
    DenseMatView<double, BatchType::Single> A_view(dp_A.data(), dp_rows, dp_cols, dp_rows);
    DenseVecHandle<double, BatchType::Single> x_vec(dp_x.data(), dp_cols, 1);
    DenseVecHandle<double, BatchType::Single> y_vec(dp_y.data(), dp_rows, 1);
    
    // Perform y = A*x
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, 1.0, 0.0, Transpose::NoTrans);
    
    ctx->wait();
    
    // Verify result
    for (int i = 0; i < dp_rows; ++i) {
        EXPECT_NEAR(dp_y[i], expected[i], 1e-12) 
            << "Mismatch in double precision test at index " << i;
    }
} */

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}