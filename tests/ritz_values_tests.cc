#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <blas/matrix.hh>
#include <blas/extensions.hh>
#include "test_utils.hh"

using namespace batchlas;


// Test fixture for ritz_values operations
class RitzValuesTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>(Device::default_device());
    }
    
    void TearDown() override {
    }
    
    std::shared_ptr<Queue> ctx;
};

TEST_F(RitzValuesTest, DiagonalMatrix) {
    // Create a diagonal matrix with known eigenvalues
    constexpr int n = 5;
    constexpr int batch = 1;
    constexpr int k = n; // Use all eigenvectors
    UnifiedVector<float> diag_entries(n);
    for (int i = 0; i < n; ++i) {
        diag_entries[i] = float(i + 1); // Eigenvalues: 1, 2, 3, 4, 5
    }
    // Create diagonal matrix A with diagonal entries 1, 2, 3, 4, 5
    auto A = Matrix<float, MatrixFormat::Dense>::Diagonal(diag_entries, batch);

    // Create trial vectors V (identity matrix - the actual eigenvectors)
    auto V = Matrix<float, MatrixFormat::Dense>::Identity(n, batch);
    
    
    // Expected Ritz values are the diagonal entries
    std::vector<float> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = float(j + 1);
    }
    
    // Allocate output for Ritz values
    Vector<float> ritz_vals(k, batch);
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_workspace<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A, V, ritz_vals);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    ritz_values<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A, V, ritz_vals, workspace);
    ctx->wait();
    
    // Verify results

    auto tol = test_utils::tolerance<float>();    
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(ritz_vals(j), expected[j], tol)
            << "Ritz value mismatch at index " << j;
    }
}

TEST_F(RitzValuesTest, TridiagonalMatrix) {
    // Create a tridiagonal Toeplitz matrix with known eigenvalues
    // For a symmetric tridiagonal matrix with diagonal=a and off-diagonals=b,c
    // eigenvalues are: a - 2*sqrt(b*c)*cos(k*pi/(n+1)) for k=1..n
    constexpr int n = 10;
    constexpr int batch = 2;
    constexpr int k = 3; // Test with first 3 trial vectors
    
    float a = 2.0f, b = -1.0f, c = -1.0f;
    auto A = Matrix<float, MatrixFormat::Dense>::TriDiagToeplitz(n, a, b, c, batch);
    
    // Create trial vectors V using the analytical eigenvectors
    // For a tridiagonal Toeplitz matrix, eigenvector j has components:
    // v_i = sin(j*(i+1)*pi/(n+1))
    Matrix<float, MatrixFormat::Dense> V(n, k, batch);
    auto V_data = V.data().to_vector();
    
    for (int bat = 0; bat < batch; ++bat) {
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                // Eigenvector j+1 (1-indexed): v_i = sin((j+1)*(i+1)*pi/(n+1))
                V_data[bat * n * k + j * n + i] = std::sin((j + 1) * (i + 1) * M_PI / (n + 1));
            }
        }
    }
    V.data().copy_from(V_data.data(), V_data.size());
    
    // Compute expected Ritz values (eigenvalues)
    std::vector<float> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = a - 2.0f * std::sqrt(b * c) * std::cos((j + 1) * M_PI / (n + 1));
    }
    
    // Allocate output for Ritz values
    Vector<float> ritz_vals(k, batch);
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_workspace<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A, V, ritz_vals);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    ritz_values<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A, V, ritz_vals, workspace);
    ctx->wait();
    
    // Verify results
    auto tol = test_utils::tolerance<float>() * 10.0f; // Slightly relaxed tolerance
    
    for (int bat = 0; bat < batch; ++bat) {
        for (int j = 0; j < k; ++j) {
            EXPECT_NEAR(ritz_vals(j, bat), expected[j], tol)
                << "Ritz value mismatch at batch " << bat << ", index " << j;
        }
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
