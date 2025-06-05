#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <complex>
#include <cmath>
#include <vector>
#include <chrono>
#include <blas/extra.hh>

using namespace batchlas;

/**
 * Comprehensive test suite for the norm function.
 * 
 * This test suite covers all norm types (Frobenius, One, Inf, Max)
 * and works with batched matrices in both Dense and CSR formats.
 * Uses matrix factory methods to generate test matrices.
 */

// Test fixture for norm operations
template <typename T>
class NormTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = std::make_shared<Queue>();
    }
    
    void TearDown() override {
        if (ctx) {
            ctx->wait();
        }
    }
    
    // Helper function to compute expected Frobenius norm
    typename base_type<T>::type expected_frobenius_norm(const Matrix<T, MatrixFormat::Dense>& mat, int batch_idx = 0) {
        using real_t = typename base_type<T>::type;
        real_t sum = real_t(0);
        auto data = mat.data();
        int stride = mat.stride();
        int size = mat.rows() * mat.cols();
        
        for (int i = 0; i < size; ++i) {
            T val = data[batch_idx * stride + i];
            if constexpr (std::is_same_v<T, std::complex<float>> || 
                         std::is_same_v<T, std::complex<double>>) {
                sum += val.real() * val.real() + val.imag() * val.imag();
            } else {
                sum += val * val;
            }
        }
        return std::sqrt(sum);
    }
    
    // Helper function to compute expected one norm (max column sum)
    typename base_type<T>::type expected_one_norm(const Matrix<T, MatrixFormat::Dense>& mat, int batch_idx = 0) {
        using real_t = typename base_type<T>::type;
        auto data = mat.data();
        int rows = mat.rows();
        int cols = mat.cols();
        int ld = mat.ld();
        int stride = mat.stride();
        real_t max_sum = real_t(0);
        for (int j = 0; j < cols; ++j) {
            real_t col_sum = real_t(0);
            for (int i = 0; i < rows; ++i) {
                T val = data[batch_idx * stride + j * ld + i];
                if constexpr (std::is_same_v<T, std::complex<float>> || 
                             std::is_same_v<T, std::complex<double>>) {
                    col_sum += std::abs(val);
                } else {
                    col_sum += std::abs(val);
                }
            }
            max_sum = std::max(max_sum, col_sum);
        }
        return max_sum;
    }
    
    // Helper function to compute expected infinity norm (max row sum)
    typename base_type<T>::type expected_inf_norm(const Matrix<T, MatrixFormat::Dense>& mat, int batch_idx = 0) {
        using real_t = typename base_type<T>::type;
        auto data = mat.data();
        int rows = mat.rows();
        int cols = mat.cols();
        int ld = mat.ld();
        int stride = mat.stride();
        
        real_t max_sum = real_t(0);
        for (int i = 0; i < rows; ++i) {
            real_t row_sum = real_t(0);
            for (int j = 0; j < cols; ++j) {
                T val = data[batch_idx * stride + j * ld + i];
                if constexpr (std::is_same_v<T, std::complex<float>> || 
                             std::is_same_v<T, std::complex<double>>) {
                    row_sum += std::abs(val);
                } else {
                    row_sum += std::abs(val);
                }
            }
            max_sum = std::max(max_sum, row_sum);
        }
        return max_sum;
    }
    
    // Helper function to compute expected max norm (max element magnitude)
    typename base_type<T>::type expected_max_norm(const Matrix<T, MatrixFormat::Dense>& mat, int batch_idx = 0) {
        using real_t = typename base_type<T>::type;
        auto data = mat.data();
        int stride = mat.stride();
        int size = mat.rows() * mat.cols();
        
        real_t max_val = real_t(0);
        for (int i = 0; i < size; ++i) {
            T val = data[batch_idx * stride + i];
            if constexpr (std::is_same_v<T, std::complex<float>> || 
                         std::is_same_v<T, std::complex<double>>) {
                max_val = std::max(max_val, real_t(std::abs(val)));
            } else {
                max_val = std::max(max_val, std::abs(val));
            }
        }
        return max_val;
    }
    
    std::shared_ptr<Queue> ctx;
    
    // Test tolerances based on type
    static constexpr auto tolerance() {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
            return 1e-5f;
        } else {
            return 1e-10;
        }
    }

    // Small helper to map NormType to a string for readable failure messages
    static const char* norm_name(NormType t) {
        switch (t) {
            case NormType::Frobenius: return "Frobenius";
            case NormType::One:       return "One";
            case NormType::Inf:       return "Inf";
            case NormType::Max:       return "Max";
        }
        return "";
    }

    // Convenience array with all norm types
    static constexpr std::array<NormType, 4> all_norms = {
        NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max
    };

    // Run norm for all types and compare against the provided expected function
    template <typename ExpectedFn>
    void check_all_norms(const Matrix<T, MatrixFormat::Dense>& mat,
                         ExpectedFn&& expected_fn) {
        using real_t = typename base_type<T>::type;
        UnifiedVector<real_t> result(mat.batch_size());

        for (auto ntype : all_norms) {
            norm(*ctx, mat.view(), ntype, result.to_span());
            ctx->wait();

            for (int b = 0; b < mat.batch_size(); ++b) {
                auto expected = expected_fn(ntype, b);
                EXPECT_NEAR(result[b], expected, tolerance())
                    << "Batch " << b << " " << norm_name(ntype) << " norm mismatch";
            }
        }
    }

    // Check that all norms are strictly positive
    void check_all_norms_positive(const Matrix<T, MatrixFormat::Dense>& mat) {
        using real_t = typename base_type<T>::type;
        UnifiedVector<real_t> result(mat.batch_size());

        for (auto ntype : all_norms) {
            norm(*ctx, mat.view(), ntype, result.to_span());
            ctx->wait();

            for (int b = 0; b < mat.batch_size(); ++b) {
                EXPECT_GT(result[b], real_t(0))
                    << "Batch " << b << " " << norm_name(ntype)
                    << " norm should be positive";
            }
        }
    }
};

// Typed tests for different scalar types
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NormTest, TestTypes);

// Test all norm types with a random matrix
TYPED_TEST(NormTest, RandomMatrixAllNorms) {
    using T = TypeParam;
    const int rows = 5, cols = 4, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, batch_size, 123);

    auto expected_fn = [this, &mat](NormType ntype, int b) {
        switch (ntype) {
            case NormType::Frobenius: return this->expected_frobenius_norm(mat, b);
            case NormType::One:       return this->expected_one_norm(mat, b);
            case NormType::Inf:       return this->expected_inf_norm(mat, b);
            case NormType::Max:       return this->expected_max_norm(mat, b);
        }
        return typename base_type<T>::type(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with Identity matrix
TYPED_TEST(NormTest, NormsIdentityMatrix) {
    using T = TypeParam;
    const int n = 5, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Identity(n, batch_size);

    auto expected_fn = [n](NormType ntype, int) {
        using real_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return std::sqrt(static_cast<real_t>(n));
            case NormType::One:
            case NormType::Inf:
            case NormType::Max: return real_t(1);
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with Zero matrix
TYPED_TEST(NormTest, NormsZeroMatrix) {
    using T = TypeParam;
    const int rows = 4, cols = 3, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Zeros(rows, cols, batch_size);

    auto expected_fn = [](NormType, int) {
        using real_t = typename base_type<T>::type;
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with Ones matrix
TYPED_TEST(NormTest, NormsOnesMatrix) {
    using T = TypeParam;
    const int rows = 3, cols = 4, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Ones(rows, cols, batch_size);

    auto expected_fn = [rows, cols](NormType ntype, int) {
        using real_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return std::sqrt(static_cast<real_t>(rows * cols));
            case NormType::One:       return static_cast<real_t>(rows);
            case NormType::Inf:       return static_cast<real_t>(cols);
            case NormType::Max:       return real_t(1);
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with diagonal matrix
TYPED_TEST(NormTest, NormsDiagonalMatrix) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    const int n = 4, batch_size = 2;

    // Create diagonal matrix with values [1, 2, 3, 4]
    UnifiedVector<T> diag_vals(n);
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> ||
                     std::is_same_v<T, std::complex<double>>) {
            diag_vals[i] = T(static_cast<real_t>(i + 1), static_cast<real_t>(0));
        } else {
            diag_vals[i] = static_cast<T>(i + 1);
        }
    }

    auto mat = Matrix<T, MatrixFormat::Dense>::Diagonal(diag_vals.to_span(), batch_size);
    real_t expected_frob = std::sqrt(real_t(1 + 4 + 9 + 16));

    auto expected_fn = [expected_frob](NormType ntype, int) {
        using r_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return expected_frob;
            case NormType::One:
            case NormType::Inf:
            case NormType::Max: return r_t(4);
        }
        return r_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with triangular matrix
TYPED_TEST(NormTest, NormsTriangularMatrix) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    const int n = 4, batch_size = 2;
    
    // Create upper triangular matrix with diagonal=2, off-diagonal=1
    T diag_val, off_diag_val;
    if constexpr (std::is_same_v<T, std::complex<float>> || 
                 std::is_same_v<T, std::complex<double>>) {
        diag_val = T(real_t(2), real_t(0));
        off_diag_val = T(real_t(1), real_t(0));
    } else {
        diag_val = T(2);
        off_diag_val = T(1);
    }
    
    auto mat = Matrix<T, MatrixFormat::Dense>::Triangular(n, Uplo::Upper, diag_val, off_diag_val, batch_size);

    this->check_all_norms_positive(mat);
}

// Test single matrix (batch_size = 1)
TYPED_TEST(NormTest, SingleMatrixNorms) {
    using T = TypeParam;
    const int rows = 100, cols = 100, batch_size = 1;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, batch_size, 555);

    this->check_all_norms_positive(mat);
}

// Test large batch size
TYPED_TEST(NormTest, LargeBatchNorms) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    const int rows = 3, cols = 3, batch_size = 10;
    
    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, batch_size, 777);
    auto result = UnifiedVector<real_t>(batch_size);
    
    norm(*this->ctx, mat.view(), NormType::Frobenius, result.to_span());
    this->ctx->wait();
    
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_GT(result[b], real_t(0))
            << "Batch " << b << " norm should be positive";
    }
}

// Test different matrix sizes
TYPED_TEST(NormTest, DifferentMatrixSizes) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {2, 3}, {5, 2}, {10, 10}};
    
    for (auto [rows, cols] : sizes) {
        auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, 1, 888);
        auto result = UnifiedVector<real_t>(1);
        
        norm(*this->ctx, mat.view(), NormType::Frobenius, result.to_span());
        this->ctx->wait();
        
        EXPECT_GT(result[0], real_t(0))
            << "Size " << rows << "x" << cols << " norm should be positive";
    }
}

// Test norm consistency (compare with manual calculation)
TYPED_TEST(NormTest, NormConsistency) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    const int rows = 2, cols = 2, batch_size = 1;
    
    // Create a known matrix
    auto mat = Matrix<T, MatrixFormat::Dense>(rows, cols, batch_size);
    auto data = mat.data();
    
    if constexpr (std::is_same_v<T, std::complex<float>> || 
                 std::is_same_v<T, std::complex<double>>) {
        data[0] = T(real_t(1), real_t(0));   // (0,0)
        data[1] = T(real_t(2), real_t(0));   // (1,0)
        data[2] = T(real_t(3), real_t(0));   // (0,1)
        data[3] = T(real_t(4), real_t(0));   // (1,1)
    } else {
        data[0] = T(1);  // (0,0)
        data[1] = T(2);  // (1,0)
        data[2] = T(3);  // (0,1)
        data[3] = T(4);  // (1,1)
    }
    
    auto expected_fn = [](NormType ntype, int) {
        using real_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return std::sqrt(real_t(30));
            case NormType::One:       return real_t(7);
            case NormType::Inf:       return real_t(6);
            case NormType::Max:       return real_t(4);
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Performance/stress test with larger matrices
TYPED_TEST(NormTest, StressTestLargeMatrix) {
    using T = TypeParam;
    using real_t = typename base_type<T>::type;
    const int rows = 100, cols = 100, batch_size = 5;
    
    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, batch_size, 1111);
    auto result = UnifiedVector<real_t>(batch_size);
    
    auto start = std::chrono::high_resolution_clock::now();

    norm(*this->ctx, mat.view(), NormType::Frobenius, result.to_span());
    this->ctx->wait();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Verify results are reasonable
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_GT(result[b], real_t(0))
            << "Large matrix norm should be positive";
        EXPECT_LT(result[b], real_t(1000))
            << "Large matrix norm should be reasonable";
    }
    
    // Performance should be reasonable (less than 1 second for this size)
    EXPECT_LT(duration.count(), 1000)
        << "Large matrix norm computation should complete in reasonable time";
}