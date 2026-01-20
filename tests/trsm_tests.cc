#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>
#include "test_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using TrsmTestTypes = typename test_utils::backend_types<TestConfig>::type;

template<typename Config>
class TrsmOperationsTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    
    const int rows = 8;
    const int cols = 8;
    const int ld = 8;
    const int batch_size = 3;
    const ScalarType alpha = static_cast<ScalarType>(1.0);

    UnifiedVector<ScalarType> A_data;
    UnifiedVector<ScalarType> B_data;
    UnifiedVector<ScalarType> B_data_original;
    
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        
        if (!this->ctx) {
            return;
        }
        
        // Initialize test matrices
        A_data = UnifiedVector<ScalarType>(rows * cols * batch_size);
        B_data_original = UnifiedVector<ScalarType>(rows * cols * batch_size);
        B_data = UnifiedVector<ScalarType>(rows * cols * batch_size);
        
        // Initialize matrix A as a lower triangular matrix with ones on the diagonal
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    if (i == j) {
                        A_data[b * rows * cols + i * cols + j] = static_cast<ScalarType>(1.0);
                    } else if (i > j) {
                        A_data[b * rows * cols + i * cols + j] = static_cast<ScalarType>(0.5);
                    } else {
                        A_data[b * rows * cols + i * cols + j] = static_cast<ScalarType>(0.0);
                    }
                }
            }
        }
        
        // Initialize matrix B with some test values
        std::mt19937 rng(42);
        std::uniform_real_distribution<typename base_type<ScalarType>::type> dist(
            static_cast<typename base_type<ScalarType>::type>(1.0), 
            static_cast<typename base_type<ScalarType>::type>(10.0));
        
        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    ScalarType val = dist(rng);
                    B_data_original[b * rows * cols + i * cols + j] = val;
                    B_data[b * rows * cols + i * cols + j] = val;
                }
            }
        }
    }
    
    // Verify that the TRSM solution satisfies A*X = B or A^T*X = B depending on transpose
    bool verifyTrsmResult(int batch_idx, Transpose trans = Transpose::NoTrans) {
        const bool trace_enabled = []() {
            const char* v = std::getenv("BATCHLAS_TRSM_TRACE");
            if (!v) return false;
            return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
                    std::string(v) == "on" || std::string(v) == "ON");
        }();

        // First check if B was actually modified from original
        bool anyChanges = false;
        for (int i = 0; i < rows && !anyChanges; ++i) {
            for (int j = 0; j < cols && !anyChanges; ++j) {
                int idx = batch_idx * rows * cols + i * cols + j;
                if (std::abs(B_data[idx] - B_data_original[idx]) > test_utils::tolerance<ScalarType>()) {
                    anyChanges = true;
                }
            }
        }
        
        if (!anyChanges) {
            if (trace_enabled) {
                std::cerr << "TRSM TRACE: output appears unchanged for batch " << batch_idx
                          << " (trans=" << static_cast<int>(trans) << ")" << std::endl;
                for (int k = 0; k < std::min(rows * cols, 4); ++k) {
                    std::cerr << "  B[" << k << "]=" << B_data[batch_idx * rows * cols + k]
                              << " (orig=" << B_data_original[batch_idx * rows * cols + k] << ")" << std::endl;
                }
            }
            return false;
        }
        
        // Now verify each element of the result by checking AX = B_original or A^T*X = B_original
        bool allMatch = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ScalarType expected = B_data_original[batch_idx * rows * cols + i * cols + j];
                ScalarType calculated = static_cast<ScalarType>(0.0);
                
                // Calculate the result of A*X or A^T*X for this position
                for (int k = 0; k < cols; ++k) {
                    int a_row = (trans == Transpose::NoTrans) ? i : k;
                    int a_col = (trans == Transpose::NoTrans) ? k : i;
                    calculated += A_data[batch_idx * rows * cols + a_row * cols + a_col] * 
                                  B_data[batch_idx * rows * cols + k * cols + j];
                }
                
                // Use a reasonable tolerance for floating point comparisons
                auto tolerance = test_utils::tolerance<ScalarType>();
                if (std::abs(calculated - expected) > tolerance) {
                    if (trace_enabled) {
                        std::cerr << "TRSM TRACE: mismatch at (i=" << i << ", j=" << j << ") batch=" << batch_idx
                                  << " (trans=" << static_cast<int>(trans) << ")\n"
                                  << "  expected=" << expected << "\n"
                                  << "  calculated=" << calculated << "\n"
                                  << "  |diff|=" << std::abs(calculated - expected) << " tol=" << tolerance
                                  << std::endl;
                    }
                    allMatch = false;
                    break;
                }
            }
            if (!allMatch) break;
        }
        return allMatch;
    }
    
    void performTrsmTest(Uplo uplo, Transpose trans, int test_batch_size = 1) {
        // Create matrices using convenience factory methods
        auto A_matrix = Matrix<ScalarType, MatrixFormat::Dense>::Triangular(rows, uplo, static_cast<ScalarType>(1.0), static_cast<ScalarType>(0.5), test_batch_size);
        auto B_matrix = Matrix<ScalarType, MatrixFormat::Dense>::Random(rows, cols, false, test_batch_size);
        
        // Keep original B for verification
        auto B_original = B_matrix.clone();
        
        // Convert to column-major format
        auto A_colmajor = A_matrix.to_column_major();
        auto B_colmajor = B_matrix.to_column_major();
        
        // Create matrix views
        if (test_batch_size == 1) {
            auto A_view = A_colmajor.view();
            auto B_view = B_colmajor.view();
            
            try {
                trsm<BackendType>(
                    *(this->ctx),
                    A_view,
                    B_view,
                    Side::Left,
                    uplo,
                    trans,
                    Diag::NonUnit,
                    alpha
                );
                this->ctx->wait();
            } catch(const std::exception& e) {
                FAIL() << "TRSM operation failed with exception: " << e.what();
            }
        } else {
            auto A_parent_view = A_colmajor.view();
            auto B_parent_view = B_colmajor.view();
            
            // Process each batch using batch_item
            for (int b = 0; b < test_batch_size; ++b) {
                auto A_view = A_parent_view.batch_item(b);
                auto B_view = B_parent_view.batch_item(b);
                
                try {
                    trsm<BackendType>(
                        *(this->ctx),
                        A_view,
                        B_view,
                        Side::Left,
                        uplo,
                        trans,
                        Diag::NonUnit,
                        alpha
                    );
                } catch(const std::exception& e) {
                    FAIL() << "TRSM operation failed for batch " << b << " with exception: " << e.what();
                }
            }
            this->ctx->wait();
        }
        
        // Convert result back to row-major for verification
        auto B_result = B_colmajor.to_row_major();
        
        // Copy to our fixture's data for verification
        for (int b = 0; b < test_batch_size; ++b) {
            for (int i = 0; i < rows * cols; ++i) {
                B_data[b * rows * cols + i] = B_result.data()[b * rows * cols + i];
                A_data[b * rows * cols + i] = A_matrix.data()[b * rows * cols + i];
                B_data_original[b * rows * cols + i] = B_original.data()[b * rows * cols + i];
            }
            
            EXPECT_TRUE(verifyTrsmResult(b, trans)) << "TRSM solution verification failed for batch " << b;
        }
    }
};

TYPED_TEST_SUITE(TrsmOperationsTest, TrsmTestTypes);

// Test TRSM operation with a lower triangular matrix (no transpose)
TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, 1);
}

// Test TRSM operation with a lower triangular matrix (transpose)
TYPED_TEST(TrsmOperationsTest, LowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, 1);
}

// Test TRSM operation with an upper triangular matrix (no transpose)
TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, 1);
}

// Test TRSM operation with an upper triangular matrix (transpose)
TYPED_TEST(TrsmOperationsTest, UpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, 1);
}

// Test batched TRSM operation with lower triangular (no transpose)
TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::NoTrans, this->batch_size);
}

// Test batched TRSM operation with lower triangular (transpose)
TYPED_TEST(TrsmOperationsTest, BatchedLowerTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Lower, Transpose::Trans, this->batch_size);
}

// Test batched TRSM operation with upper triangular (no transpose)
TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveNoTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::NoTrans, this->batch_size);
}

// Test batched TRSM operation with upper triangular (transpose)
TYPED_TEST(TrsmOperationsTest, BatchedUpperTriangularSolveTrans) {
    this->performTrsmTest(Uplo::Upper, Transpose::Trans, this->batch_size);
}