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


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
