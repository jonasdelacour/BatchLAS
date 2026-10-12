#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <iostream>
#include <vector>
#include <cmath>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/extensions.hh>
#include "test_utils.hh"
#include <batchlas/verify/tolerance.hh>
#include <batchlas/util/mempool.hh>

using namespace batchlas;


template <typename T, Backend B>
struct TestConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using RitzTestTypes = typename test_utils::backend_types<TestConfig>::type;

// Test fixture for ritz_values operations
template <typename Config>
class RitzValuesTest : public test_utils::BatchLASTest<Config> {
protected:
    using ScalarType = typename Config::ScalarType;
    using RealType = typename batchlas::base_type<ScalarType>::type;
    static constexpr Backend BackendType = Config::BackendVal;

    static ScalarType make_scalar(RealType value) {
        if constexpr (test_utils::is_complex<ScalarType>::value) {
            return ScalarType(value, RealType(0));
        } else {
            return static_cast<ScalarType>(value);
        }
    }
};

TYPED_TEST_SUITE(RitzValuesTest, RitzTestTypes);

TYPED_TEST(RitzValuesTest, DiagonalMatrix) {
    using ScalarType = typename TestFixture::ScalarType;
    using RealType = typename TestFixture::RealType;
    constexpr Backend BackendType = TestFixture::BackendType;
    if (!this->ctx) {
        return;
    }
    // Create a diagonal matrix with known eigenvalues
    constexpr int n = 5;
    constexpr int batch = 1;
    constexpr int k = n; // Use all eigenvectors
    UnifiedVector<ScalarType> diag_entries(n);
    for (int i = 0; i < n; ++i) {
        diag_entries[i] = TestFixture::make_scalar(RealType(i + 1)); // Eigenvalues: 1, 2, 3, 4, 5
    }
    // Create diagonal matrix A with diagonal entries 1, 2, 3, 4, 5
    auto A = Matrix<ScalarType, MatrixFormat::Dense>::Diagonal(diag_entries, batch);

    // Create trial vectors V (identity matrix - the actual eigenvectors)
    auto V = Matrix<ScalarType, MatrixFormat::Dense>::Identity(n, batch);
    
    
    // Expected Ritz values are the diagonal entries
    std::vector<RealType> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = RealType(j + 1);
    }
    
    // Allocate output for Ritz values
    Vector<RealType> ritz_vals(k, batch);
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_buffer_size<BackendType, ScalarType, MatrixFormat::Dense>(
        *(this->ctx), A, V, ritz_vals);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    (void)ritz_values<BackendType, ScalarType, MatrixFormat::Dense>(
        *(this->ctx), A, V, ritz_vals, workspace);
    this->ctx->wait();
    
    // Ritz values of exact eigenvectors are eigenvalues: Check::values against the closed form,
    // ||A||_2 = 5. The kind's float bound is 4.8e-5 absolute, so the Slack keeps the old power.
    const std::vector<std::vector<double>> ref{std::vector<double>(expected.begin(), expected.end())};
    using MV = MatrixView<ScalarType, MatrixFormat::Dense>;
    EXPECT_VERIFY_SLACK(ScalarType, verify::Check::values, n, MV::values_error(VectorView<RealType>(ritz_vals), ref, 5.0),
                        verify::Slack{0.25, "old bound 1e-5 absolute float (test_utils::tolerance), ||A||_2 = 5"});
}

TYPED_TEST(RitzValuesTest, TridiagonalMatrix) {
    using ScalarType = typename TestFixture::ScalarType;
    using RealType = typename TestFixture::RealType;
    constexpr Backend BackendType = TestFixture::BackendType;
    if (!this->ctx) {
        return;
    }
    // Create a tridiagonal Toeplitz matrix with known eigenvalues
    // For a symmetric tridiagonal matrix with diagonal=a and off-diagonals=b,c
    // eigenvalues are: a - 2*sqrt(b*c)*cos(k*pi/(n+1)) for k=1..n
    constexpr int n = 10;
    constexpr int batch = 2;
    constexpr int k = 3; // Test with first 3 trial vectors
    
    RealType a = RealType(2.0), b = RealType(-1.0), c = RealType(-1.0);
    auto A = Matrix<ScalarType, MatrixFormat::Dense>::TriDiagToeplitz(n, a, b, c, batch);
    
    // Create trial vectors V using the analytical eigenvectors
    // For a tridiagonal Toeplitz matrix, eigenvector j has components:
    // v_i = sin(j*(i+1)*pi/(n+1))
    Matrix<ScalarType, MatrixFormat::Dense> V(n, k, batch);
    
    for (int bat = 0; bat < batch; ++bat) {
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                // Eigenvector j+1 (1-indexed): v_i = sin((j+1)*(i+1)*pi/(n+1))
                V.view()(i, j, bat) = TestFixture::make_scalar(
                    std::sin((j + 1) * (i + 1) * M_PI / (n + 1)));
            }
        }
    }
    
    // Compute expected Ritz values (eigenvalues)
    std::vector<RealType> expected(k);
    for (int j = 0; j < k; ++j) {
        expected[j] = a - RealType(2.0) * std::sqrt(b * c) * std::cos((j + 1) * M_PI / (n + 1));
    }
    
    // Allocate output for Ritz values
    Vector<RealType> ritz_vals(k, batch);
    
    // Compute workspace size and allocate
    size_t workspace_size = ritz_values_buffer_size<BackendType, ScalarType, MatrixFormat::Dense>(
        *(this->ctx), A, V, ritz_vals);
    UnifiedVector<std::byte> workspace(workspace_size);
    
    // Compute Ritz values
    (void)ritz_values<BackendType, ScalarType, MatrixFormat::Dense>(
        *(this->ctx), A, V, ritz_vals, workspace);
    this->ctx->wait();
    
    const std::vector<std::vector<double>> ref(batch, std::vector<double>(expected.begin(), expected.end()));
    const double norm2 = double(a) + 2.0 * std::cos(M_PI / (n + 1));
    using MV = MatrixView<ScalarType, MatrixFormat::Dense>;
    const VectorView<RealType> w(ritz_vals);
    EXPECT_VERIFY(ScalarType, verify::Check::values, n, MV::values_error(w, ref, norm2, w.all_items()));
}


// CSR A: the spmm workspace is carved with allocate<std::byte>, which rounds, so the size query must
// equal the run path's measured sequence and an exactly sized workspace must be enough.
TYPED_TEST(RitzValuesTest, SparseWorkspaceMatchesRunPath) {
    using ScalarType = typename TestFixture::ScalarType;
    using RealType = typename TestFixture::RealType;
    constexpr Backend BackendType = TestFixture::BackendType;
    const int n = 37, k = 5, batch = 3;
    auto A = Matrix<ScalarType, MatrixFormat::CSR>::RandomSparseHermitian(n, 0.3f, batch, 5u);
    auto V = Matrix<ScalarType, MatrixFormat::Dense>::Random(n, k, false, batch, 6u);
    Vector<RealType> ritz_vals(k, batch);
    const size_t bytes = ritz_values_buffer_size<BackendType, ScalarType, MatrixFormat::CSR>(*(this->ctx), A, V, ritz_vals);

    auto pool = BumpAllocator::measuring();
    auto av = pool.allocate<ScalarType>(*(this->ctx), size_t(n) * k * batch);
    (void)pool.allocate<ScalarType*>(*(this->ctx), batch);
    const MatrixView<ScalarType, MatrixFormat::Dense> AV(av.data(), n, k, n, n * k, batch);
    (void)pool.allocate<std::byte>(*(this->ctx), spmm_buffer_size<BackendType>(*(this->ctx), A.view(), V.view(), AV, ScalarType(1),
                                                                                  ScalarType(0), Transpose::NoTrans, Transpose::NoTrans));
    EXPECT_EQ(bytes, pool.required_bytes());

    UnifiedVector<std::byte> workspace(bytes);
    auto run = [&] { (void)ritz_values<BackendType, ScalarType, MatrixFormat::CSR>(*(this->ctx), A, V, ritz_vals, workspace); };
    EXPECT_NO_THROW(run());
    this->ctx->wait();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
