#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/matrix.hh> // Include the new matrix handle
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits> // Required for std::is_same_v

using namespace batchlas;

// Configuration struct for parameterized tests
template <typename T, batchlas::Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr batchlas::Backend BackendVal = B;
};

// Define the types to be tested
#include <batchlas/backend_config.h>
using MyTypes = ::testing::Types<
#if BATCHLAS_HAS_HOST_BACKEND
    TestConfig<float, batchlas::Backend::NETLIB>,
    TestConfig<double, batchlas::Backend::NETLIB>
#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND || BATCHLAS_HAS_MKL_BACKEND
    ,
#endif
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    TestConfig<float, batchlas::Backend::CUDA>,
    TestConfig<double, batchlas::Backend::CUDA>
#if BATCHLAS_HAS_ROCM_BACKEND || BATCHLAS_HAS_MKL_BACKEND
    ,
#endif
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    TestConfig<float, batchlas::Backend::ROCM>,
    TestConfig<double, batchlas::Backend::ROCM>
#if BATCHLAS_HAS_MKL_BACKEND
    ,
#endif
#endif
#if BATCHLAS_HAS_MKL_BACKEND
    TestConfig<float, batchlas::Backend::MKL>,
    TestConfig<double, batchlas::Backend::MKL>
#endif
>;

// Test fixture for GEMV operations using MatrixView/VectorView, now templated
template <typename Config>
class GemvMatrixViewTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr batchlas::Backend BackendType = Config::BackendVal;

    void SetUp() override {
        // Create a SYCL queue based on BackendType
        if constexpr (BackendType == batchlas::Backend::CUDA) {
            try {
                this->ctx = std::make_shared<Queue>("gpu");
                // Check if a GPU device was actually obtained
                if (!(this->ctx->device().type == DeviceType::GPU)) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL did not select a GPU device. Skipping.";
                }
            } catch (const sycl::exception& e) {
                // Common errors indicating no GPU or CUDA setup issues
                if (e.code() == sycl::errc::runtime || 
                    e.code() == sycl::errc::feature_not_supported) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL GPU queue creation failed: " << e.what() << ". Skipping.";
                } else {
                    throw; // Re-throw unexpected SYCL exceptions
                }
            } catch (const std::exception& e) { // Catch other potential errors from Queue constructor
                 GTEST_SKIP() << "CUDA backend selected, but Queue construction failed: " << e.what() << ". Skipping.";
            }
        } else { // For NETLIB, use CPU
            this->ctx = std::make_shared<Queue>("cpu");
        }
        if (!this->ctx) { // Should be caught by GTEST_SKIP or an exception, but as a fallback
            GTEST_FAIL() << "Queue context is null after setup.";
            return; // Avoid further execution if ctx is null
        }


        // Initialize test matrices and vectors
        this->A_data = UnifiedVector<ScalarType>(this->rows * this->cols * this->batch_size);
        this->x_data = UnifiedVector<ScalarType>(std::max(this->rows, this->cols) * this->batch_size); // Use max for transpose case
        this->y_data = UnifiedVector<ScalarType>(this->rows * this->batch_size, static_cast<ScalarType>(0.0));
        this->y_expected = UnifiedVector<ScalarType>(this->rows * this->batch_size, static_cast<ScalarType>(0.0));

        // Initialize matrix with deterministic values (Column Major for BLAS)
        for (int b = 0; b < this->batch_size; ++b) {
            for (int j = 0; j < this->cols; ++j) { // Column index
                for (int i = 0; i < this->rows; ++i) { // Row index
                    this->A_data[b * this->rows * this->cols + j * this->rows + i] = static_cast<ScalarType>(i + j * this->rows + 1 + b * 100);
                }
            }
        }

        // Initialize x vector with sequential values
        for (int b = 0; b < this->batch_size; ++b) {
            int vec_dim = this->cols; // Dimension for NoTrans
            for (int j = 0; j < vec_dim; ++j) {
                this->x_data[b * std::max(this->rows, this->cols) + j] = static_cast<ScalarType>(j + 1 + b * 10);
            }
        }
         // Initialize y vector (if needed, e.g., for beta != 0)
        for (int b = 0; b < this->batch_size; ++b) {
             for (int i = 0; i < this->rows; ++i) {
                 this->y_data[b * this->rows + i] = static_cast<ScalarType>(0.0); // Initialize y to zero for simplicity first
             }
        }
    }

    void TearDown() override {
    }

    // Helper function to compute expected y = alpha*A*x + beta*y
    void computeExpectedGemv(ScalarType alpha, ScalarType beta, Transpose transA) {
        for (int b = 0; b < this->batch_size; ++b) {
            const ScalarType* A_batch = this->A_data.data() + b * this->rows * this->cols;
            const ScalarType* x_batch = this->x_data.data() + b * std::max(this->rows, this->cols);
            ScalarType* y_batch_expected = this->y_expected.data() + b * this->rows;
            const ScalarType* y_batch_initial = this->y_data.data() + b * this->rows; // Initial y for beta calculation

            if (transA == Transpose::NoTrans) {
                for (int i = 0; i < this->rows; ++i) {
                    ScalarType sum = static_cast<ScalarType>(0.0);
                    for (int j = 0; j < this->cols; ++j) {
                        sum += A_batch[i + j * this->rows] * x_batch[j];
                    }
                    y_batch_expected[i] = alpha * sum + beta * y_batch_initial[i];
                }
            } else { 
                if (this->rows != this->cols) {
                    GTEST_SKIP() << "Transpose test skipped for non-square matrix in this fixture setup.";
                    return;
                }

                for (int j = 0; j < this->cols; ++j) { 
                    ScalarType sum = static_cast<ScalarType>(0.0);
                    for (int i = 0; i < this->rows; ++i) { 
                        sum += A_batch[i + j * this->rows] * x_batch[i]; 
                    }
                    y_batch_expected[j] = alpha * sum + beta * y_batch_initial[j];
                }
            }
        }
    }

    ScalarType get_tolerance() {
        if constexpr (std::is_same_v<ScalarType, float>) {
            return 1e-4f;
        } else {
            return 1e-7;
        }
    }
    ScalarType get_rel_error_floor() {
        if constexpr (std::is_same_v<ScalarType, float>) {
            return 1e-6f;
        } else {
            return 1e-9;
        }
    }


    std::shared_ptr<Queue> ctx;
    const int rows = 10; 
    const int cols = 10;
    const int batch_size = 5;
    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> x_data; 
    UnifiedVector<ScalarType> y_data; 
    UnifiedVector<ScalarType> y_expected; 
};

TYPED_TEST_SUITE(GemvMatrixViewTest, MyTypes);

// Test single GEMV operation with no transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, SingleGemvNoTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr batchlas::Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 0); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, 1); 
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, 1); 

    ScalarType alpha = static_cast<ScalarType>(1.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans); 

    gemv<BackendType>(*(this->ctx), A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    this->ctx->wait();
    
    ScalarType tol = this->get_tolerance();
    for (int i = 0; i < this->rows; ++i) {
        EXPECT_NEAR(this->y_data[i], this->y_expected[i], tol) 
            << "Mismatch at index " << i;
    }
}

// Test single GEMV operation with transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, SingleGemvWithTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr batchlas::Backend BackendType = TestFixture::BackendType;

    ASSERT_EQ(this->rows, this->cols) << "Transpose test requires square matrix in this fixture setup.";

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 0); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->rows, 1); 
    VectorView<ScalarType> y_vec(this->y_data.data(), this->cols, 1); 

    ScalarType alpha = static_cast<ScalarType>(2.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::Trans); 

    gemv<BackendType>(*(this->ctx), A_view, x_vec, y_vec, alpha, beta, Transpose::Trans);

    this->ctx->wait();

    ScalarType tol = this->get_tolerance();
    for (int i = 0; i < this->cols; ++i) {
        EXPECT_NEAR(this->y_data[i], this->y_expected[i], tol)
        << "Mismatch with transpose at index " << i;
    }
}


// Test batched GEMV operation using MatrixView
TYPED_TEST(GemvMatrixViewTest, BatchedGemvNoTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr batchlas::Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, 1, this->cols, this->batch_size); 
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, 1, this->rows, this->batch_size); 

    ScalarType alpha = static_cast<ScalarType>(1.0);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans);

    gemv<BackendType>(*(this->ctx), A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    this->ctx->wait();

    ScalarType tol = this->get_tolerance();
    ScalarType floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            auto rel_error = std::abs(this->y_data[b * this->rows + i] - this->y_expected[b * this->rows + i]) / std::max(std::abs(this->y_expected[b * this->rows + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<ScalarType>(0.0), tol)
                << "Mismatch at batch " << b << ", index " << i;
        }
    }
}

// Test batched GEMV operation with transpose using MatrixView
TYPED_TEST(GemvMatrixViewTest, BatchedGemvWithTranspose) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr batchlas::Backend BackendType = TestFixture::BackendType;

    ASSERT_EQ(this->rows, this->cols) << "Transpose test requires square matrix in this fixture setup.";

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows,
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->rows, 1, this->rows, this->batch_size);
    VectorView<ScalarType> y_vec(this->y_data.data(), this->cols, 1, this->cols, this->batch_size);

    ScalarType alpha = static_cast<ScalarType>(2.5);
    ScalarType beta = static_cast<ScalarType>(0.0);

    this->computeExpectedGemv(alpha, beta, Transpose::Trans);

    gemv<BackendType>(*(this->ctx), A_view, x_vec, y_vec, alpha, beta, Transpose::Trans);

    this->ctx->wait();

    ScalarType tol = this->get_tolerance();
    ScalarType floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->cols; ++i) { 
            auto rel_error = std::abs(this->y_data[b * this->cols + i] - this->y_expected[b * this->cols + i]) / std::max(std::abs(this->y_expected[b * this->cols + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<ScalarType>(0.0), tol)
                << "Mismatch with transpose at batch " << b << ", index " << i;
        }
    }
}

// Test both alpha and beta in batched GEMV
TYPED_TEST(GemvMatrixViewTest, BatchedGemvWithAlphaBeta) {
    using ScalarType = typename TestFixture::ScalarType;
    constexpr batchlas::Backend BackendType = TestFixture::BackendType;

    MatrixView<ScalarType, MatrixFormat::Dense> A_view(this->A_data.data(), this->rows, this->cols, this->rows, 
                                                this->rows * this->cols, this->batch_size); 
    VectorView<ScalarType> x_vec(this->x_data.data(), this->cols, 1, this->cols, this->batch_size);
    VectorView<ScalarType> y_vec(this->y_data.data(), this->rows, 1, this->rows, this->batch_size);

    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            this->y_data[b * this->rows + i] = static_cast<ScalarType>(b * 1.0 + i * 0.1);
        }
    }
     this->y_expected = this->y_data; 

    ScalarType alpha = static_cast<ScalarType>(1.5);
    ScalarType beta = static_cast<ScalarType>(0.8);

    this->computeExpectedGemv(alpha, beta, Transpose::NoTrans); 

    gemv<BackendType>(*(this->ctx), A_view, x_vec, y_vec, alpha, beta, Transpose::NoTrans);

    this->ctx->wait();

    ScalarType tol = this->get_tolerance();
    ScalarType floor_val = this->get_rel_error_floor();
    for (int b = 0; b < this->batch_size; ++b) {
        for (int i = 0; i < this->rows; ++i) {
            auto rel_error = std::abs(this->y_data[b * this->rows + i] - this->y_expected[b * this->rows + i]) / std::max(std::abs(this->y_expected[b * this->rows + i]), floor_val);
            EXPECT_NEAR(rel_error, static_cast<ScalarType>(0.0), tol)
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