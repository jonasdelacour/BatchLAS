#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <blas/matrix.hh>
#include <blas/extensions.hh>
#include <batchlas/backend_config.h>
#include "test_utils.hh"

using namespace batchlas;

#if BATCHLAS_HAS_GPU_BACKEND

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

TEST_F(RitzValuesTest, TridiagonalMatrix) {
    // Create a simple tridiagonal matrix with known eigenvalues
    constexpr int n = 10;
    constexpr int batch = 2;
    constexpr int k = 3; // Number of trial vectors
    
    // Create tridiagonal matrix A with diagonal = 2, off-diagonals = -1
    // Eigenvalues: 2 - 2*cos(j*pi/(n+1)) for j=1..n
    Matrix<float, MatrixFormat::Dense> A(n, n, batch);
    auto A_data = A.data().to_vector();
    
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                float val = 0.0f;
                if (i == j) val = 2.0f;
                else if (std::abs(i - j) == 1) val = -1.0f;
                A_data[b * n * n + i * n + j] = val;
            }
        }
    }
    A.data().copy_from(A_data.data(), A_data.size());
    auto A_view = A.view();
    
    // Create trial vectors V (using the first k actual eigenvectors)
    Matrix<float, MatrixFormat::Dense> V(n, k, batch);
    auto V_data = V.data().to_vector();
    
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                // Eigenvector j+1: v_i = sin((j+1)*(i+1)*pi/(n+1))
                V_data[b * n * k + j * n + i] = std::sin((j + 1) * (i + 1) * M_PI / (n + 1));
            }
        }
    }
    V.data().copy_from(V_data.data(), V_data.size());
    auto V_view = V.view();
    
    // Compute expected Ritz values (eigenvalues)
    std::vector<float> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = 2.0f - 2.0f * std::cos((j + 1) * M_PI / (n + 1));
    }
    
    // Allocate output for Ritz values
    Vector<float> ritz_vals(k, batch);
    auto ritz_vals_view = ritz_vals.view();
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_workspace<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A_view, V_view, ritz_vals_view);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    ritz_values<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A_view, V_view, ritz_vals_view, workspace);
    ctx->wait();
    
    // Verify results
    auto ritz_vals_host = ritz_vals.data().to_vector();
    auto tol = test_utils::tolerance<float>() * 10.0f; // Slightly relaxed tolerance
    
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < k; ++j) {
            EXPECT_NEAR(ritz_vals_host[b * k + j], expected[j], tol)
                << "Ritz value mismatch at batch " << b << ", index " << j;
        }
    }
}

TEST_F(RitzValuesTest, DiagonalMatrix) {
    // Create a diagonal matrix with known eigenvalues
    constexpr int n = 5;
    constexpr int batch = 1;
    constexpr int k = n; // Use all eigenvectors
    
    // Create diagonal matrix A with diagonal entries 1, 2, 3, 4, 5
    Matrix<float, MatrixFormat::Dense> A(n, n, batch);
    auto A_data = A.data().to_vector();
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_data[i * n + j] = (i == j) ? float(i + 1) : 0.0f;
        }
    }
    A.data().copy_from(A_data.data(), A_data.size());
    auto A_view = A.view();
    
    // Create trial vectors V (identity matrix - the actual eigenvectors)
    Matrix<float, MatrixFormat::Dense> V(n, k, batch);
    auto V_data = V.data().to_vector();
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            V_data[j * n + i] = (i == j) ? 1.0f : 0.0f;
        }
    }
    V.data().copy_from(V_data.data(), V_data.size());
    auto V_view = V.view();
    
    // Expected Ritz values are the diagonal entries
    std::vector<float> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = float(j + 1);
    }
    
    // Allocate output for Ritz values
    Vector<float> ritz_vals(k, batch);
    auto ritz_vals_view = ritz_vals.view();
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_workspace<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A_view, V_view, ritz_vals_view);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    ritz_values<test_utils::gpu_backend, float, MatrixFormat::Dense>(
        *ctx, A_view, V_view, ritz_vals_view, workspace);
    ctx->wait();
    
    // Verify results
    auto ritz_vals_host = ritz_vals.data().to_vector();
    auto tol = test_utils::tolerance<float>();
    
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(ritz_vals_host[j], expected[j], tol)
            << "Ritz value mismatch at index " << j;
    }
}

TEST_F(RitzValuesTest, SparseCSRMatrix) {
    // Create a sparse tridiagonal matrix in CSR format
    constexpr int n = 10;
    constexpr int batch = 1;
    constexpr int k = 3;
    constexpr int nnz = 3 * n - 2; // Tridiagonal matrix nnz
    
    // Create CSR matrix A (tridiagonal: diagonal = 2, off-diagonals = -1)
    std::vector<float> values(nnz);
    std::vector<int> col_indices(nnz);
    std::vector<int> row_offsets(n + 1);
    
    int idx = 0;
    for (int i = 0; i < n; ++i) {
        row_offsets[i] = idx;
        if (i > 0) {
            values[idx] = -1.0f;
            col_indices[idx] = i - 1;
            idx++;
        }
        values[idx] = 2.0f;
        col_indices[idx] = i;
        idx++;
        if (i < n - 1) {
            values[idx] = -1.0f;
            col_indices[idx] = i + 1;
            idx++;
        }
    }
    row_offsets[n] = idx;
    
    Matrix<float, MatrixFormat::CSR> A(values.data(), row_offsets.data(), col_indices.data(),
                                       nnz, n, n, nnz, n + 1, batch);
    auto A_view = A.view();
    
    // Create trial vectors V (using the first k actual eigenvectors)
    Matrix<float, MatrixFormat::Dense> V(n, k, batch);
    auto V_data = V.data().to_vector();
    
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < n; ++i) {
            V_data[j * n + i] = std::sin((j + 1) * (i + 1) * M_PI / (n + 1));
        }
    }
    V.data().copy_from(V_data.data(), V_data.size());
    auto V_view = V.view();
    
    // Compute expected Ritz values (eigenvalues)
    std::vector<float> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = 2.0f - 2.0f * std::cos((j + 1) * M_PI / (n + 1));
    }
    
    // Allocate output for Ritz values
    Vector<float> ritz_vals(k, batch);
    auto ritz_vals_view = ritz_vals.view();
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_workspace<test_utils::gpu_backend, float, MatrixFormat::CSR>(
        *ctx, A_view, V_view, ritz_vals_view);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    ritz_values<test_utils::gpu_backend, float, MatrixFormat::CSR>(
        *ctx, A_view, V_view, ritz_vals_view, workspace);
    ctx->wait();
    
    // Verify results
    auto ritz_vals_host = ritz_vals.data().to_vector();
    auto tol = test_utils::tolerance<float>() * 10.0f;
    
    for (int j = 0; j < k; ++j) {
        EXPECT_NEAR(ritz_vals_host[j], expected[j], tol)
            << "Ritz value mismatch at index " << j;
    }
}

#endif // BATCHLAS_HAS_GPU_BACKEND

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
