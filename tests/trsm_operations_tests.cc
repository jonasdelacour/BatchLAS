#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix_handle_new.hh>
#include <blas/cublas_matrixview.hh>
#include <blas/functions.hh>
#include <blas/enums.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace batchlas;

// Test fixture for TRSM operations
class TrsmOperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SYCL queue
        ctx = std::make_shared<Queue>(Device::default_device());
        
        // Initialize test matrices
        // Create a lower triangular matrix for A
        A_data = UnifiedVector<float>(rows * cols * batch_size);
        // Create a dense matrix for B
        B_data_original = UnifiedVector<float>(rows * cols * batch_size);
        B_data = UnifiedVector<float>(rows * cols * batch_size);
        
        // Initialize matrix A as a lower triangular matrix with ones on the diagonal
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if (i == j) {
                        A_data[b * rows * cols + i * cols + j] = 1.0f; // Diagonal elements
                    } else if (i > j) {
                        A_data[b * rows * cols + i * cols + j] = 0.5f; // Lower triangular elements
                    } else {
                        A_data[b * rows * cols + i * cols + j] = 0.0f; // Upper triangular elements (zeros)
                    }
                }
            }
        }
        
        // Initialize matrix B with some test values
        std::mt19937 rng(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(1.0f, 10.0f);
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float val = dist(rng);
                    B_data_original[b * rows * cols + i * cols + j] = val;
                    B_data[b * rows * cols + i * cols + j] = val; // Create a copy to be modified
                }
            }
        }
    }
    
    // Verify that the TRSM solution satisfies A*X = B
    bool verifyTrsmResult(int batch_idx) {
        // First check if B was actually modified from original
        bool anyChanges = false;
        for (int i = 0; i < rows && !anyChanges; ++i) {
            for (int j = 0; j < cols && !anyChanges; ++j) {
                int idx = batch_idx * rows * cols + i * cols + j;
                if (std::abs(B_data[idx] - B_data_original[idx]) > 1e-6f) {
                    anyChanges = true;
                }
            }
        }
        
        if (!anyChanges) {
            return false;
        }
        
        // Now verify each element of the result by checking AX = B_original
        bool allMatch = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float expected = B_data_original[batch_idx * rows * cols + i * cols + j];
                float calculated = 0.0f;
                
                // Calculate the result of A*X for this position
                for (int k = 0; k < cols; ++k) {
                    calculated += A_data[batch_idx * rows * cols + i * cols + k] * 
                                  B_data[batch_idx * rows * cols + k * cols + j];
                }
                
                // Use a reasonable tolerance for floating point comparisons
                if (std::abs(calculated - expected) > 1e-2f) {
                    allMatch = false;
                    break;
                }
            }
            if (!allMatch) break;
        }
        return allMatch;
    }
    
    void TearDown() override {
        // Clean-up
    }
    
    std::shared_ptr<Queue> ctx;
    const int rows = 8;
    const int cols = 8;
    const int ld = 8;
    const int batch_size = 3;
    const float alpha = 1.0f; // Scale factor for B
    
    UnifiedVector<float> A_data;        // Triangular matrix
    UnifiedVector<float> B_data;        // Right-hand side matrix, will be overwritten with solution X
    UnifiedVector<float> B_data_original; // Original B values before solving
};

// Test TRSM operation with a lower triangular matrix
TEST_F(TrsmOperationsTest, LowerTriangularSolve) {
    // Make sure we're only testing with one batch for this single test
    const int single_batch_idx = 0;
    
    // Create matrices using the convenience factory methods
    auto A_matrix = Matrix<float, MatrixFormat::Dense>::Triangular(rows, Uplo::Lower, 1.0f, 0.5f);
    auto B_matrix = Matrix<float, MatrixFormat::Dense>::Random(rows, cols);
    
    // Keep original B for verification
    auto B_original = B_matrix.clone();
    
    // Convert to column-major format for BLAS operations
    auto A_colmajor = A_matrix.to_column_major();
    auto B_colmajor = B_matrix.to_column_major();
    
    // Create matrix views
    auto A_view = A_colmajor.view();
    auto B_view = B_colmajor.view();
    
    // Perform triangular solve: B = alpha * inv(A) * B
    try {
        trsm<Backend::CUDA>(
            *ctx, 
            A_view, 
            B_view, 
            Side::Left,
            Uplo::Lower,
            Transpose::NoTrans,
            Diag::NonUnit,
            alpha
        );
        
        // Wait for the operation to complete
        ctx->wait();
    } catch(const std::exception& e) {
        FAIL() << "TRSM operation failed with exception: " << e.what();
    }
    
    // Convert result back to row-major for verification
    auto B_result = B_colmajor.to_row_major();
    
    // Copy to our fixture's data for verification
    for (int i = 0; i < rows * cols; ++i) {
        B_data[single_batch_idx * rows * cols + i] = B_result.data()[i];
        A_data[single_batch_idx * rows * cols + i] = A_matrix.data()[i];
        B_data_original[single_batch_idx * rows * cols + i] = B_original.data()[i];
    }
    
    // Verify the result
    EXPECT_TRUE(verifyTrsmResult(single_batch_idx)) << "TRSM solution verification failed";
}

// Test batched TRSM operation
TEST_F(TrsmOperationsTest, BatchedLowerTriangularSolve) {
    // Create matrices using convenience factory methods
    auto A_matrix = Matrix<float, MatrixFormat::Dense>::Triangular(rows, Uplo::Lower, 1.0f, 0.5f, batch_size);
    auto B_matrix = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, batch_size);
    
    // Keep original B for verification
    auto B_original = B_matrix.clone();
    
    // Convert to column-major format
    auto A_colmajor = A_matrix.to_column_major();
    auto B_colmajor = B_matrix.to_column_major();
    
    // Create matrix views
    auto A_parent_view = A_colmajor.view();
    auto B_parent_view = B_colmajor.view();
    
    // Process each batch using batch_item
    for (int b = 0; b < batch_size; ++b) {
        auto A_view = A_parent_view.batch_item(b);
        auto B_view = B_parent_view.batch_item(b);
        
        try {
            // Perform TRSM for this batch
            trsm<Backend::CUDA>(
                *ctx, 
                A_view, 
                B_view, 
                Side::Left,
                Uplo::Lower,
                Transpose::NoTrans,
                Diag::NonUnit,
                alpha
            );
        } catch(const std::exception& e) {
            FAIL() << "TRSM operation failed for batch " << b << " with exception: " << e.what();
        }
    }
    
    // Wait for all operations to complete
    ctx->wait();
    
    // Convert results back to row-major
    auto B_result = B_colmajor.to_row_major();
    
    // Copy to our fixture's data for verification
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows * cols; ++i) {
            B_data[b * rows * cols + i] = B_result.data()[b * rows * cols + i];
            A_data[b * rows * cols + i] = A_matrix.data()[b * rows * cols + i];
            B_data_original[b * rows * cols + i] = B_original.data()[b * rows * cols + i];
        }
        
        // Verify each batch
        EXPECT_TRUE(verifyTrsmResult(b)) << "TRSM solution verification failed for batch " << b;
    }
}

// Test TRSM operation with an upper triangular matrix
TEST_F(TrsmOperationsTest, UpperTriangularSolve) {
    // Create matrices using convenience factory methods
    auto A_matrix = Matrix<float, MatrixFormat::Dense>::Triangular(rows, Uplo::Upper, 1.0f, 0.5f);
    auto B_matrix = Matrix<float, MatrixFormat::Dense>::Random(rows, cols);
    
    // Keep a copy of the original B for verification
    auto B_original = B_matrix.clone();
    
    // Convert to column-major format for TRSM
    auto A_colmajor = A_matrix.to_column_major();
    auto B_colmajor = B_matrix.to_column_major();
    
    // Create matrix views
    MatrixView A_view = A_colmajor.view();
    MatrixView B_view = B_colmajor.view();
    
    // Perform triangular solve with upper triangular matrix
    try {
        trsm<Backend::CUDA>(
            *ctx, 
            A_view, 
            B_view, 
            Side::Left,
            Uplo::Upper,
            Transpose::NoTrans,
            Diag::NonUnit,
            alpha
        );
        
        // Wait for the operation to complete
        ctx->wait();
    } catch(const std::exception& e) {
        FAIL() << "Upper triangular TRSM operation failed with exception: " << e.what();
    }
    
    // Convert result back to row-major for verification
    auto B_result = B_colmajor.to_row_major();
    
    // Copy to fixture's data for verification
    for (int i = 0; i < rows * cols; ++i) {
        B_data[i] = B_result.data()[i];
        A_data[i] = A_matrix.data()[i];
        B_data_original[i] = B_original.data()[i];
    }
    
    // Verify result
    EXPECT_TRUE(verifyTrsmResult(0)) << "Upper triangular TRSM solution verification failed";
}