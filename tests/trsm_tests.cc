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

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
    }
    
    // Verify that the TRSM solution satisfies A*X = B or A^T*X = B depending on transpose
    bool verifyTrsmResult(const MatrixView<ScalarType, MatrixFormat::Dense>& A,
                          const MatrixView<ScalarType, MatrixFormat::Dense>& B,
                          const MatrixView<ScalarType, MatrixFormat::Dense>& B_original,
                          int batch_idx,
                          Transpose trans = Transpose::NoTrans) {
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
                if (std::abs(B.at(i, j, batch_idx) - B_original.at(i, j, batch_idx)) > test_utils::tolerance<ScalarType>()) {
                    anyChanges = true;
                }
            }
        }
        
        if (!anyChanges) {
            if (trace_enabled) {
                std::cerr << "TRSM TRACE: output appears unchanged for batch " << batch_idx
                          << " (trans=" << static_cast<int>(trans) << ")" << std::endl;
                for (int k = 0; k < std::min(rows * cols, 4); ++k) {
                    int col = k / rows;
                    int row = k % rows;
                    std::cerr << "  B[" << k << "]=" << B.at(row, col, batch_idx)
                              << " (orig=" << B_original.at(row, col, batch_idx) << ")" << std::endl;
                }
            }
            return false;
        }
        
        // Now verify each element of the result by checking AX = B_original or A^T*X = B_original
        bool allMatch = true;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ScalarType expected = B_original.at(i, j, batch_idx);
                ScalarType calculated = static_cast<ScalarType>(0.0);
                
                // Calculate the result of A*X or A^T*X for this position
                for (int k = 0; k < cols; ++k) {
                    int a_row = (trans == Transpose::NoTrans) ? i : k;
                    int a_col = (trans == Transpose::NoTrans) ? k : i;
                    calculated += A.at(a_row, a_col, batch_idx) * B.at(k, j, batch_idx);
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
        // Create matrices and fill on host to avoid device-side state or kernel ordering issues
        Matrix<ScalarType, MatrixFormat::Dense> A_matrix(rows, rows, test_batch_size);
        Matrix<ScalarType, MatrixFormat::Dense> B_matrix(rows, cols, test_batch_size);

        std::mt19937 rng(42);
        std::uniform_real_distribution<batchlas::float_t<ScalarType>> dist(-1.0, 1.0);

        auto A_view_full = A_matrix.view();
        auto B_view_full = B_matrix.view();

        for (int b = 0; b < test_batch_size; ++b) {
            for (int j = 0; j < rows; ++j) {
                for (int i = 0; i < rows; ++i) {
                    if (i == j) {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(1.0);
                    } else if ((uplo == Uplo::Lower && i > j) || (uplo == Uplo::Upper && i < j)) {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(0.5);
                    } else {
                        A_view_full.at(i, j, b) = static_cast<ScalarType>(0.0);
                    }
                }
            }

            for (int j = 0; j < cols; ++j) {
                for (int i = 0; i < rows; ++i) {
                    if constexpr (std::is_same_v<ScalarType, std::complex<float>> ||
                                  std::is_same_v<ScalarType, std::complex<double>>) {
                        B_view_full.at(i, j, b) = ScalarType(dist(rng), dist(rng));
                    } else {
                        B_view_full.at(i, j, b) = static_cast<ScalarType>(dist(rng));
                    }
                }
            }
        }
        
        // Keep original B for verification
        auto B_original = B_matrix.clone();

        // Create matrix views (matrices are already column-major)
        if (test_batch_size == 1) {
            auto A_view = A_matrix.view();
            auto B_view = B_matrix.view();
            
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
            auto A_parent_view = A_matrix.view();
            auto B_parent_view = B_matrix.view();
            
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

        auto A_view = A_matrix.view();
        auto B_view = B_matrix.view();
        auto B_original_view = B_original.view();
        for (int b = 0; b < test_batch_size; ++b) {
            EXPECT_TRUE(verifyTrsmResult(A_view, B_view, B_original_view, b, trans))
                << "TRSM solution verification failed for batch " << b;
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