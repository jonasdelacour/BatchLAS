#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix_handle_new.hh> // Include the new matrix handle
#include <blas/cublas_matrixview.hh> // Include the header with MatrixView-compatible gemv
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace batchlas;

// Test fixture for GEMV operations using MatrixView/VectorView
class GemvMatrixViewTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a SYCL queue
        ctx = std::make_shared<Queue>(Device::default_device());

        // Initialize test matrices and vectors
        A_data = UnifiedVector<float>(rows * cols * batch_size);
        x_data = UnifiedVector<float>(std::max(rows, cols) * batch_size); // Use max for transpose case
        y_data = UnifiedVector<float>(rows * batch_size, 0.0f);
        y_expected = UnifiedVector<float>(rows * batch_size, 0.0f);

        // Initialize matrix with deterministic values (Column Major for BLAS)
        for (int b = 0; b < batch_size; ++b) {
            for (int j = 0; j < cols; ++j) { // Column index
                for (int i = 0; i < rows; ++i) { // Row index
                    A_data[b * rows * cols + j * rows + i] = static_cast<float>(i + j * rows + 1 + b * 100); // Example init
                }
            }
        }

        // Initialize x vector with sequential values
        for (int b = 0; b < batch_size; ++b) {
            int vec_dim = cols; // Dimension for NoTrans
            for (int j = 0; j < vec_dim; ++j) {
                x_data[b * std::max(rows, cols) + j] = static_cast<float>(j + 1 + b * 10);
            }
        }
         // Initialize y vector (if needed, e.g., for beta != 0)
        for (int b = 0; b < batch_size; ++b) {
             for (int i = 0; i < rows; ++i) {
                 y_data[b * rows + i] = 0.0f; // Initialize y to zero for simplicity first
             }
        }
    }

    void TearDown() override {
    }

    // Helper function to compute expected y = alpha*A*x + beta*y
    void computeExpectedGemv(float alpha, float beta, Transpose transA) {
        for (int b = 0; b < batch_size; ++b) {
            const float* A_batch = A_data.data() + b * rows * cols;
            const float* x_batch = x_data.data() + b * std::max(rows, cols);
            float* y_batch_expected = y_expected.data() + b * rows;
            const float* y_batch_initial = y_data.data() + b * rows; // Initial y for beta calculation

            if (transA == Transpose::NoTrans) {
                 // y = alpha * A * x + beta * y
                 // A is rows x cols, x is cols x 1, y is rows x 1
                for (int i = 0; i < rows; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < cols; ++j) {
                        // A is column-major: A[i + j * rows]
                        sum += A_batch[i + j * rows] * x_batch[j];
                    }
                    y_batch_expected[i] = alpha * sum + beta * y_batch_initial[i];
                }
            } else { // Transpose::Trans or Transpose::ConjTrans (same for real numbers)
                 // y = alpha * A^T * x + beta * y
                 // A^T is cols x rows, x is rows x 1, y is cols x 1
                 // Note: The output vector y should have size 'cols' in this case.
                 // Let's adjust the test setup or assume y always has size 'rows' and x size 'rows' for Transpose case.
                 // Re-initializing y_expected and y_data for transpose case if needed.
                 // For simplicity, let's assume the test setup ensures correct dimensions.
                 // If A is rows x cols, A^T is cols x rows.
                 // If x has size 'rows', then y must have size 'cols'.
                 // Let's stick to the original setup where y has size 'rows'.
                 // This implies x must have size 'rows' for the transpose case.
                 // We need to adjust the x_data initialization or the test logic.

                 // Let's assume the test setup is correct: A is rows x cols, x is rows x 1, y is cols x 1
                 // We need a y_expected_transposed vector of size cols.
                 // This complicates the fixture. Let's keep y size as 'rows' and assume x size matches.
                 // If A is rows x cols, x is cols x 1, y is rows x 1 (NoTrans)
                 // If A is rows x cols, x is rows x 1, y is cols x 1 (Trans) -> y needs size cols.

                 // Let's redefine the test slightly:
                 // NoTrans: A(rows, cols), x(cols), y(rows)
                 // Trans:   A(rows, cols), x(rows), y(cols)
                 // We need separate y vectors/expected vectors or adjust dimensions dynamically.

                 // --- Simplified approach: Assume square matrices for simplicity ---
                 // If rows == cols, then dimensions match for both cases.
                 // Let's modify the fixture to use square matrices for now.
                 // ASSERT_EQ(rows, cols); // Add this assertion in tests needing transpose

                if (rows != cols) {
                    // Skip transpose test if not square for simplicity in this fixture setup
                    GTEST_SKIP() << "Transpose test skipped for non-square matrix in this fixture setup.";
                    return;
                }

                for (int j = 0; j < cols; ++j) { // Output index for transpose case
                    float sum = 0.0f;
                    for (int i = 0; i < rows; ++i) { // Inner loop over rows
                         // A is column-major: A[i + j * rows]
                        sum += A_batch[i + j * rows] * x_batch[i]; // x index corresponds to row index of A
                    }
                     // Assuming y_expected has size 'rows' (matching NoTrans case)
                     // This calculation is for y of size 'cols'. We need to store it correctly.
                     // Let's calculate into a temporary buffer if y_expected must remain size 'rows'.
                     // Or, assert rows == cols.
                    y_batch_expected[j] = alpha * sum + beta * y_batch_initial[j];
                }
            }
        }
    }

    std::shared_ptr<Queue> ctx;
    const int rows = 10; // Make rows == cols for easier transpose testing in this fixture
    const int cols = 10;
    const int batch_size = 5;
    UnifiedVector<float> A_data;
    UnifiedVector<float> x_data; // Size needs to accommodate both rows and cols if not square
    UnifiedVector<float> y_data; // Size 'rows' for NoTrans, 'cols' for Trans
    UnifiedVector<float> y_expected; // Size 'rows' for NoTrans, 'cols' for Trans
};

// Test single GEMV operation with no transpose using MatrixView
TEST_F(GemvMatrixViewTest, SingleGemvNoTranspose) {
    // Create MatrixView and VectorView for the first batch item
    MatrixView<float, MatrixFormat::Dense> A_view(A_data.data(), rows, cols, rows, 0); // Single batch item view
    VectorView<float> x_vec(x_data.data(), cols, 1); // Single vector view, add ld parameter
    VectorView<float> y_vec(y_data.data(), rows, 1); // Single vector view, add ld parameter

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute expected result for the first batch item
    computeExpectedGemv(alpha, beta, Transpose::NoTrans); // Computes for all batches, we check batch 0

    // Perform y = alpha*A*x + beta*y
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    ctx->wait();

    // Verify result for the first batch item
    for (int i = 0; i < rows; ++i) {
        EXPECT_NEAR(y_data[i], y_expected[i], 1e-4f) // Increased tolerance slightly
            << "Mismatch at index " << i;
    }
}

// Test single GEMV operation with transpose using MatrixView
TEST_F(GemvMatrixViewTest, SingleGemvWithTranspose) {
     ASSERT_EQ(rows, cols) << "Transpose test requires square matrix in this fixture setup.";

    // Create MatrixView and VectorView for the first batch item
    MatrixView<float, MatrixFormat::Dense> A_view(A_data.data(), rows, cols, rows, 0); // Single batch item view
    // For A^T * x, x must have size 'rows'
    VectorView<float> x_vec(x_data.data(), rows, 1); // Single vector view, size 'rows'
    // For A^T * x, y must have size 'cols'
    VectorView<float> y_vec(y_data.data(), cols, 1); // Single vector view, size 'cols'

    float alpha = 2.0f;
    float beta = 0.0f;

    // Compute expected result for the first batch item
    computeExpectedGemv(alpha, beta, Transpose::Trans); // Computes for all batches, we check batch 0

    // Perform y = alpha*A^T*x + beta*y
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::Trans);

    ctx->wait();

    // Verify result for the first batch item (output y has size 'cols')
    for (int i = 0; i < cols; ++i) {
        EXPECT_NEAR(y_data[i], y_expected[i], 1e-4f)
        << "Mismatch with transpose at index " << i;
    }
}


// Test batched GEMV operation using MatrixView
TEST_F(GemvMatrixViewTest, BatchedGemvNoTranspose) {
    // Create batched MatrixView and VectorView - use constructors that match the declarations
    MatrixView<float, MatrixFormat::Dense> A_view(A_data.data(), rows, cols, rows, 
                                                rows * cols, batch_size); // Fixed constructor arguments
    VectorView<float> x_vec(x_data.data(), cols, 1, cols, batch_size); // Reorder arguments to match constructor
    VectorView<float> y_vec(y_data.data(), rows, 1, rows, batch_size); // Reorder arguments to match constructor

    float alpha = 1.0f;
    float beta = 0.0f;

    // Compute expected result (y = alpha * A * x + beta * y) for all batches
    computeExpectedGemv(alpha, beta, Transpose::NoTrans);

    // Perform batched gemv
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    ctx->wait();

    // Verify result for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_NEAR(y_data[b * rows + i], y_expected[b * rows + i], 1e-4f)
                << "Mismatch at batch " << b << ", index " << i;
        }
    }
}

// Test batched GEMV operation with transpose using MatrixView
TEST_F(GemvMatrixViewTest, BatchedGemvWithTranspose) {
    ASSERT_EQ(rows, cols) << "Transpose test requires square matrix in this fixture setup.";

    // Create batched MatrixView and VectorView
    MatrixView<float, MatrixFormat::Dense> A_view(A_data.data(), rows, cols, rows,
                                                rows * cols, batch_size); // Fixed constructor arguments
    // For A^T * x, x must have size 'rows'
    VectorView<float> x_vec(x_data.data(), rows, 1, rows, batch_size);
    // For A^T * x, y must have size 'cols'
    VectorView<float> y_vec(y_data.data(), cols, 1, cols, batch_size);

    float alpha = 2.5f;
    float beta = 0.0f;

    // Compute expected result (y = alpha * A^T * x + beta * y) for all batches
    computeExpectedGemv(alpha, beta, Transpose::Trans);

    // Perform batched gemv with transpose
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::Trans);

    ctx->wait();

    // Verify result for each batch (output y has size 'cols')
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < cols; ++i) { // Iterate up to cols
            EXPECT_NEAR(y_data[b * cols + i], y_expected[b * cols + i], 1e-4f)
                << "Mismatch with transpose at batch " << b << ", index " << i;
        }
    }
}

// Test both alpha and beta in batched GEMV
TEST_F(GemvMatrixViewTest, BatchedGemvWithAlphaBeta) {
    // Create batched MatrixView and VectorView
    MatrixView<float, MatrixFormat::Dense> A_view(A_data.data(), rows, cols, rows, 
                                                rows * cols, batch_size); // Fixed constructor arguments
    VectorView<float> x_vec(x_data.data(), cols, 1, cols, batch_size);
    VectorView<float> y_vec(y_data.data(), rows, 1, rows, batch_size);

    // Initialize y with some values
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            y_data[b * rows + i] = static_cast<float>(b * 1.0f + i * 0.1f);
        }
    }
     // Copy initial y to expected y before calculation if beta != 0
     y_expected = y_data; // Start expected from initial y

    float alpha = 1.5f;
    float beta = 0.8f;

    // Compute expected result (y = alpha * A * x + beta * y) for all batches
    computeExpectedGemv(alpha, beta, Transpose::NoTrans); // Uses initial y_data for beta term

    // Perform batched gemv with alpha and beta
    gemv<Backend::CUDA>(*ctx, A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    ctx->wait();

    // Verify result for each batch
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < rows; ++i) {
            EXPECT_NEAR(y_data[b * rows + i], y_expected[b * rows + i], 1e-4f)
                << "Mismatch with alpha/beta at batch " << b << ", index " << i;
        }
    }
}


// ... (Keep main function) ...
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// Note: The commented-out tests (SmallMatrixGemv, DoublePrecisionGemv)
// would need similar updates to use MatrixView/VectorView and potentially
// adjustments to the fixture or test logic if they involve non-square matrices
// or different data types.