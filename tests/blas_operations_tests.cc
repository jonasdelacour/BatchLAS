#include <gtest/gtest.h>
#include <blas/linalg.hh>
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
    
    // Helper function to check if a matrix is orthonormal (A^T * A ≈ I)
    void verifyOrthonormality(UnifiedVector<float>& matrix_data, int rows, int cols, int batch_size) {
        // Create temporary storage for result of A^T * A
        UnifiedVector<float> temp_matrix = matrix_data;
        
        // Copy matrix_data to temp_matrix so we don't modify the original
        //std::copy(matrix_data.begin(), matrix_data.end(), temp_matrix.begin());
        
        
        // For each batch, compute A^T * A and verify it's close to identity
        for (int b = 0; b < batch_size; ++b) {
            UnifiedVector<float> result(cols * cols, 0.0f);
            // Create views for this batch
            DenseMatView<float, BatchType::Single> A(temp_matrix.data() + b * rows * cols, rows, cols, rows);
            DenseMatView<float, BatchType::Single> ATA(result.data(), cols, cols, cols);
            
            // Compute A^T * A
            gemm<Backend::CUDA>(*ctx, A, A, ATA, 1.0f, 0.0f,
                             Transpose::Trans, Transpose::NoTrans);
            ctx->wait();
            
            // Verify the result is close to identity
            for (int i = 0; i < cols; ++i) {
                for (int j = 0; j < cols; ++j) {
                    float expected = (i == j) ? 1.0f : 0.0f;
                    float actual = result[i * cols + j];
                    // Use a tolerance for floating point comparisons
                    EXPECT_NEAR(actual, expected, 1e-5f) 
                        << "Orthonormality check failed at batch " << b 
                        << ", element (" << i << ", " << j << ")";
                }
            }
        }
    }
};

// Test GEMM operation using identity matrix (C = A * I = A)
TEST_F(BlasOperationsTest, GemmWithIdentityMatrix) {
    // Create matrix handles
    DenseMatView<float, BatchType::Single> A_view(A_data.data(), rows, cols, ld);
    DenseMatView<float, BatchType::Single> B_view(B_data.data(), cols, cols, ld);
    DenseMatView<float, BatchType::Single> C_view(C_data.data(), rows, cols, ld);
    
    // Perform C = A * B (which should equal A since B is identity)
    gemm<Backend::CUDA>(*ctx, A_view, B_view, C_view, 1.0f, 0.0f,
                                      Transpose::NoTrans, Transpose::NoTrans);
    
    ctx->wait();
    // Copy result back to host
    
    // Verify result (C should be equal to A)
    for (size_t i = 0; i < rows*cols; ++i) {
        EXPECT_FLOAT_EQ(C_data[i], A_data[i]) << "Mismatch at index " << i;
    }
}

// Test batched GEMM operation
TEST_F(BlasOperationsTest, BatchedGemm) {

    DenseMatHandle<float, BatchType::Batched> A_handle(A_data.data(), rows, cols, ld, rows * cols, batch_size);
    DenseMatHandle<float, BatchType::Batched> B_handle(B_data.data(), rows, cols, ld, rows * cols, batch_size);
    DenseMatHandle<float, BatchType::Batched> C_handle(C_data.data(), rows, cols, ld, rows * cols, batch_size);
    

    gemm<Backend::CUDA>(*ctx, A_handle(), B_handle(), C_handle(), 1.0f, 0.0f, 
                        Transpose::NoTrans, Transpose::NoTrans);
    
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

// Test orthonormalization of random matrices
TEST_F(BlasOperationsTest, OrthonormalizeRandomMatrix) {
    const int test_rows = 20;
    const int test_cols = 10; // Testing tall matrices (rows > cols)
    const int test_ld = test_rows;
    const int test_batch_size = 3;
    
    // Create random matrices
    UnifiedVector<float> random_matrices(test_rows * test_cols * test_batch_size);
    
    // Initialize with random values
    std::srand(42); // Fixed seed for reproducibility
    for (size_t i = 0; i < random_matrices.size(); ++i) {
        random_matrices[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    }
    
    // Create a batched handle
    DenseMatHandle<float, BatchType::Batched> matrices_handle(
        random_matrices.data(), test_rows, test_cols, test_ld, 
        test_rows * test_cols, test_batch_size);
    
    
    // Create workspace for orthonormalization
    UnifiedVector<std::byte> workspace(ortho_buffer_size<Backend::CUDA>(
        *ctx, matrices_handle(), Transpose::NoTrans, OrthoAlgorithm::ShiftChol3));
        // Orthonormalize the matrices


    ortho<Backend::CUDA>(*ctx, matrices_handle(), Transpose::NoTrans, workspace, OrthoAlgorithm::ShiftChol3);
    ctx->wait();
    // Verify that the matrices are now orthonormal
    verifyOrthonormality(random_matrices, test_rows, test_cols, test_batch_size);
    
    // Also test a single matrix case
    UnifiedVector<float> single_matrix(test_rows * test_cols);
    for (size_t i = 0; i < single_matrix.size(); ++i) {
        single_matrix[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) - 0.5f;
    }
    
    DenseMatView<float, BatchType::Single> single_matrix_view(
        single_matrix.data(), test_rows, test_cols, test_ld);
    
    UnifiedVector<std::byte> single_workspace(ortho_buffer_size<Backend::CUDA>(
        *ctx, single_matrix_view, Transpose::NoTrans, OrthoAlgorithm::ShiftChol3));

    ortho<Backend::CUDA>(*ctx, single_matrix_view, Transpose::NoTrans, single_workspace, OrthoAlgorithm::ShiftChol3);
    ctx->wait();
    
    // Verify single matrix orthonormality
    verifyOrthonormality(single_matrix, test_rows, test_cols, 1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}