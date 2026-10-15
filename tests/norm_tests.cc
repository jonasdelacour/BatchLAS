#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <complex>
#include <cmath>
#include <vector>
#include <chrono>
#include <batchlas/blas/extra.hh>
#include "test_utils.hh"
#include <batchlas/verify/tolerance.hh>
#include <batchlas/no_route.hh>

using namespace batchlas;

/**
 * Comprehensive test suite for the norm function.
 * 
 * This test suite covers all norm types (Frobenius, One, Inf, Max)
 * and works with batched matrices in both Dense and CSR formats.
 * Uses matrix factory methods to generate test matrices.
 */

template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using NormTestTypes = typename test_utils::backend_types<TestConfig>::type;

// Test fixture for norm operations
template <typename Config>
class NormTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
    }
    
    // Small helper to map NormType to a string for readable failure messages
    static const char* norm_name(NormType t) {
        switch (t) {
            case NormType::Frobenius: return "Frobenius";
            case NormType::One:       return "One";
            case NormType::Inf:       return "Inf";
            case NormType::Max:       return "Max";
            case NormType::Spectral:  return "Spectral";
        }
        return "";
    }

    // A norm is a reduction, so its relative error is judged as Check::blas with k the number of terms
    // it adds (Frobenius: all of them, One: a column, Inf: a row, Max: none); an exact zero must be exact.
    static void expect_norm_close(NormType t, double got, double want, int rows, int cols) {
        const int k = t == NormType::Frobenius ? rows * cols : t == NormType::One ? rows : t == NormType::Inf ? cols : 1;
        if (want == 0.0) {
            EXPECT_EQ(got, 0.0) << norm_name(t) << " norm of a zero matrix";
        } else {
            EXPECT_VERIFY(ScalarType, batchlas::verify::Check::blas, k, std::abs(got - want) / want)
                << norm_name(t) << " norm, got " << got << " want " << want;
        }
    }

    // Convenience array with all norm types
    static constexpr std::array<NormType, 4> all_norms = {
        NormType::Frobenius, NormType::One, NormType::Inf, NormType::Max
    };

    // Run norm for all types and compare against the provided expected function
    template <typename ExpectedFn>
    void check_all_norms(const Matrix<ScalarType, MatrixFormat::Dense>& mat,
                         ExpectedFn&& expected_fn) {
        using real_t = typename base_type<ScalarType>::type;
        UnifiedVector<real_t> result(mat.batch_size());

        for (auto ntype : all_norms) {
            (void)norm(*(this->ctx), mat, ntype, result.to_span());
            this->ctx->wait();

            for (int b = 0; b < mat.batch_size(); ++b) {
                auto expected = expected_fn(ntype, b);
                SCOPED_TRACE(::testing::Message() << "batch " << b);
                expect_norm_close(ntype, result[b], expected, mat.rows(), mat.cols());
            }
        }
    }

    // Check that all norms are strictly positive
    void check_all_norms_positive(const Matrix<ScalarType, MatrixFormat::Dense>& mat) {
        using real_t = typename base_type<ScalarType>::type;
        UnifiedVector<real_t> result(mat.batch_size());

        for (auto ntype : all_norms) {
            (void)norm(*(this->ctx), mat, ntype, result.to_span());
            this->ctx->wait();

            for (int b = 0; b < mat.batch_size(); ++b) {
                EXPECT_GT(result[b], real_t(0))
                    << "Batch " << b << " " << norm_name(ntype)
                    << " norm should be positive";
            }
        }
    }
};

// Typed tests for different scalar types
TYPED_TEST_SUITE(NormTest, NormTestTypes);

// The random matrix's signs on closed-form moduli |a_ij| = 5 (i+1)(j+1)(b+1), exact in every T: a
// complex entry is (+-3 +-4i) or (+-4 +-3i) times (i+1)(j+1)(b+1), so |re| + |im| != |a|. With
// S1(n) = n(n+1)/2 and S2(n) = n(n+1)(2n+1)/6, per item: Frobenius 5 (b+1) sqrt(S2(rows) S2(cols)),
// One 5 (b+1) cols S1(rows), Inf 5 (b+1) rows S1(cols), Max 5 (b+1) rows cols.
TYPED_TEST(NormTest, RandomMatrixAllNorms) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    const int rows = 5, cols = 4, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch_size, 123);
    auto A = mat.view();
    for (int b = 0; b < batch_size; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                const T r = A(i, j, b);
                const real_t m = real_t((i + 1) * (j + 1) * (b + 1));
                if constexpr (is_std_complex_v<T>) {
                    const bool wide = std::abs(r.real()) >= std::abs(r.imag());
                    const real_t re = (r.real() < 0 ? -m : m) * real_t(wide ? 4 : 3);
                    const real_t im = (r.imag() < 0 ? -m : m) * real_t(wide ? 3 : 4);
                    A(i, j, b) = T(re, im);
                } else {
                    A(i, j, b) = r < 0 ? -5 * m : 5 * m;
                }
            }

    auto s1 = [](int n) { return n * (n + 1) / 2.0; };
    auto s2 = [](int n) { return n * (n + 1) * (2 * n + 1) / 6.0; };
    auto expected_fn = [&](NormType ntype, int b) -> double {
        const double c = 5.0 * (b + 1);
        switch (ntype) {
            case NormType::Frobenius: return c * std::sqrt(s2(rows) * s2(cols));
            case NormType::One:       return c * cols * s1(rows);
            case NormType::Inf:       return c * rows * s1(cols);
            case NormType::Max:       return c * rows * cols;
            default: std::cerr << "Unsupported norm type for random matrix" << std::endl; return 0.0;
        }
        return 0.0;
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with Identity matrix
TYPED_TEST(NormTest, NormsIdentityMatrix) {
    using T = typename TestFixture::ScalarType;
    const int n = 5, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Identity(n, batch_size);

    auto expected_fn = [n](NormType ntype, int) {
        using real_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return std::sqrt(static_cast<real_t>(n));
            case NormType::One:
            case NormType::Inf:
            case NormType::Max: return real_t(1);
            case NormType::Spectral: return real_t(1);
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with Zero matrix
TYPED_TEST(NormTest, NormsZeroMatrix) {
    using T = typename TestFixture::ScalarType;
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
    using T = typename TestFixture::ScalarType;
    const int rows = 3, cols = 4, batch_size = 2;

    auto mat = Matrix<T, MatrixFormat::Dense>::Ones(rows, cols, batch_size);

    auto expected_fn = [rows, cols](NormType ntype, int) {
        using real_t = typename base_type<T>::type;
        switch (ntype) {
            case NormType::Frobenius: return std::sqrt(static_cast<real_t>(rows * cols));
            case NormType::One:       return static_cast<real_t>(rows);
            case NormType::Inf:       return static_cast<real_t>(cols);
            case NormType::Max:       return real_t(1);
            default: std::cerr << "Unsupported norm type for rectangular matrix" << std::endl; return real_t(0);
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with diagonal matrix
TYPED_TEST(NormTest, NormsDiagonalMatrix) {
    using T = typename TestFixture::ScalarType;
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
            default: std::cerr << "Unsupported norm type for diagonal matrix" << std::endl; return r_t(0);
        }
        return r_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Test with triangular matrix
TYPED_TEST(NormTest, NormsTriangularMatrix) {
    using T = typename TestFixture::ScalarType;
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
    using T = typename TestFixture::ScalarType;
    const int rows = 100, cols = 100, batch_size = 1;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch_size, 555);

    this->check_all_norms_positive(mat);
}

// Test large batch size
TYPED_TEST(NormTest, LargeBatchNorms) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    const int rows = 3, cols = 3, batch_size = 10;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch_size, 777);
    auto result = UnifiedVector<real_t>(batch_size);
    
    (void)norm(*(this->ctx), mat, NormType::Frobenius, result.to_span());
    this->ctx->wait();
    
    for (int b = 0; b < batch_size; ++b) {
        EXPECT_GT(result[b], real_t(0))
            << "Batch " << b << " norm should be positive";
    }
}

// Test different matrix sizes
TYPED_TEST(NormTest, DifferentMatrixSizes) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    
    std::vector<std::pair<int, int>> sizes = {{1, 1}, {2, 3}, {5, 2}, {10, 10}};
    
    for (auto [rows, cols] : sizes) {
        auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, 1, 888);
        auto result = UnifiedVector<real_t>(1);
        
        (void)norm(*(this->ctx), mat, NormType::Frobenius, result.to_span());
        this->ctx->wait();
        
        EXPECT_GT(result[0], real_t(0))
            << "Size " << rows << "x" << cols << " norm should be positive";
    }
}

// Test norm consistency (compare with manual calculation)
TYPED_TEST(NormTest, NormConsistency) {
    using T = typename TestFixture::ScalarType;
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
            case NormType::Spectral:  return real_t(5.3722813233); // Approximate spectral norm
        }
        return real_t(0);
    };

    this->check_all_norms(mat, expected_fn);
}

// Performance/stress test with larger matrices
TYPED_TEST(NormTest, StressTestLargeMatrix) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    const int rows = 100, cols = 100, batch_size = 5;

    auto mat = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch_size, 1111);
    auto result = UnifiedVector<real_t>(batch_size);
    
    auto start = std::chrono::high_resolution_clock::now();

    (void)norm(*(this->ctx), mat, NormType::Frobenius, result.to_span());
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
// The spectral norm's scratch (copy, eigenvalues, workspace lease) is local to the call, so the call
// must have finished with it on return: on an out-of-order Queue, read the result with no wait.
// Item b is (b+1) * ones(n, n), whose spectral norm is n (b+1). A contract test: the vendor syev
// path already synchronises the host, so the early free cannot be shown red on CUDA.
TYPED_TEST(NormTest, SpectralOutOfOrder) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    const int n = 64, batch_size = 64;
    Queue oq(*this->ctx, false);
    auto mat = Matrix<T, MatrixFormat::Dense>::Ones(n, n, batch_size);
    for (int b = 0; b < batch_size; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) mat.view()(i, j, b) = T(real_t(b + 1));
    UnifiedVector<real_t> result(batch_size, std::numeric_limits<real_t>::quiet_NaN());
    try {
        (void)norm(oq, mat, NormType::Spectral, result.to_span());
    } catch (const batchlas::NoRouteError& e) {
        GTEST_SKIP() << "the spectral norm needs syev, which has no route in this build: " << e.what();
    }
    for (int b = 0; b < batch_size; ++b) {
        const double want = double(n) * (b + 1);
        EXPECT_VERIFY(T, batchlas::verify::Check::values, n, std::abs(double(result[b]) - want) / want) << "item " << b;
    }
    oq.wait();
}

// One/Inf accumulate per column/row in local memory sized cols/rows; with 4096 of them the zeroing
// must stride past the work-group. A first launch on 1e6 entries leaves large sums behind.
TYPED_TEST(NormTest, OneInfWiderThanWorkgroup) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    const int wide = 4096, narrow = 2, batch_size = 8;
    for (NormType t : {NormType::One, NormType::Inf}) {
        const int rows = t == NormType::One ? narrow : wide, cols = t == NormType::One ? wide : narrow;
        auto poison = Matrix<T, MatrixFormat::Dense>::Ones(rows, cols, batch_size);
        for (int b = 0; b < batch_size; ++b)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i) poison.view()(i, j, b) = T(real_t(1e6));
        auto mat = Matrix<T, MatrixFormat::Dense>::Ones(rows, cols, batch_size);
        UnifiedVector<real_t> result(batch_size);
        (void)norm(*(this->ctx), poison, t, result.to_span());
        (void)norm(*(this->ctx), mat, t, result.to_span());
        this->ctx->wait();
        for (int b = 0; b < batch_size; ++b) {
            SCOPED_TRACE(::testing::Message() << this->norm_name(t) << " item " << b);
            this->expect_norm_close(t, result[b], double(narrow), rows, cols);
        }
    }
}
