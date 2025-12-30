#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <blas/extensions.hh>

using namespace batchlas;

template <typename T, Backend B>
struct SteqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using SteqrTestTypes = typename test_utils::backend_types<SteqrConfig>::type;

template <typename Config>
class SteqrTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(SteqrTest, SteqrTestTypes);

TYPED_TEST(SteqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    Vector<float_type> eigenvalues(n, batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }
    SteqrParams<float_type> params= {};
    params.sort = true;
    params.transpose_working_vectors = false;
    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    params.sort_order = SortOrder::Ascending;

    //VectorView<float_type>::copy(*this->ctx, VectorView(diag), VectorView(sub_diag)).wait();

    auto ws = UnifiedVector<std::byte>(steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));
    steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
        ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();
    //std::sort(eigenvalues.begin(), eigenvalues.end(), std::less<float_type>());
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(eigenvalues[i], expected_eigenvalues[i], 1e-5) << "Eigenvalue mismatch at index " << i;
    }

}

TYPED_TEST(SteqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 512;
    const int batch = 32;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::ones(n, batch);
    auto b = Vector<float_type>::ones(n - 1, batch);
    auto c = Vector<float_type>::zeros(n, batch);

    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params= {};
    params.block_size = 16;
    params.block_rotations = false;
    params.max_sweeps = 10;
    params.sort = true;

    UnifiedVector<std::byte> ws(steqr_buffer_size<float_type>(*this->ctx, a, b, c, JobType::EigenVectors, params), std::byte(0));

    steqr<B, float_type>(*this->ctx, a, b, c,
        ws.to_span(), JobType::EigenVectors, params, eigvects);
        
    this->ctx->wait();

    auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(1.0), float_type(1.0), float_type(1.0), batch);
    auto ritz_vals = ritz_values<B>(*this->ctx, dense_A, eigvects);
    
    this->ctx->wait();
    UnifiedVector<float> expected(n);
    for (int k = 1; k <= n; ++k) {
        expected[k-1] = float_type(1.0 - 2.0 * std::sqrt(1.0 * 1.0) * std::cos(double(k) * M_PI / double(n + 1)));
    }
    std::sort(expected.begin(), expected.end(), std::less<float>());

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(c(i, j), expected[i], 1e-3) << "Eigenvalue value mismatch at index " << i << ", batch " << j;
            ASSERT_NEAR(c(i, j), ritz_vals(i, j), 1e-3) << "Ritz value mismatch at index " << i << ", batch " << j;
        }
    }
}


TYPED_TEST(SteqrTest, BatchedRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 128;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    Vector<float_type> diag = Vector<float_type>::random(n, batch);
    Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);
    Vector<float_type> eigenvalues = Vector<float_type>::zeros(n, batch);
    auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
    dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();


    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params= {};
    params.block_rotations = false;
    params.max_sweeps = 10;
    params.sort = true;

    UnifiedVector<std::byte> ws(steqr_buffer_size<float_type>(*this->ctx,diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));

    steqr<B, float_type>(*this->ctx, diag, sub_diag, eigenvalues,
        ws.to_span(), JobType::EigenVectors, params, eigvects);
        
    this->ctx->wait();

    auto ritz_vals = ritz_values<B>(*this->ctx, dense_A, eigvects);

    UnifiedVector<float_type> ref_eigs(n * batch);
    
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B, float_type>(*this->ctx, dense_A, ref_eigs, JobType::NoEigenVectors, Uplo::Lower), std::byte(0));
    syev<B>(*this->ctx, dense_A, ref_eigs, JobType::NoEigenVectors, Uplo::Lower, syev_ws.to_span());
    
    this->ctx->wait();
    auto eps = test_utils::tolerance<float_type>();

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues(i, j), ritz_vals(i, j), std::numeric_limits<float_type>::epsilon()*5e2) << "Ritz value mismatch at index " << i << ", batch " << j;
            ASSERT_NEAR(eigenvalues(i, j), ref_eigs[i + j * n], std::numeric_limits<float_type>::epsilon()*5e2) << "Eigenvalue value mismatch at index " << i << ", batch " << j;
        }
    }
}

TYPED_TEST(SteqrTest, SteqrCtaSingleMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    Vector<float_type> eigenvalues(n, batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }

    SteqrParams<float_type> params = {};
    params.max_sweeps = 10;  // Per-eigenvalue iteration limit
    params.sort = true;  // Re-enable sorting to match test expectations
    params.transpose_working_vectors = false;
    params.sort_order = SortOrder::Ascending;

    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    auto ws = UnifiedVector<std::byte>(steqr_cta_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));

    steqr_cta<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(eigenvalues[i], expected_eigenvalues[i], 1e-5) << "Eigenvalue mismatch at index " << i;
    }
}

TYPED_TEST(SteqrTest, SteqrCtaBatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1280;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    Vector<float_type> eigenvalues(n, batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    // All matrices are identical, so expected eigenvalues are the same for each batch item
    for (int j = 0; j < batch; ++j) {
        for (int i = 1; i <= n; ++i) {
            expected_eigenvalues[j * n + i - 1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
        }
    }

    SteqrParams<float_type> params = {};
    params.max_sweeps = 10;
    params.sort = true;
    params.transpose_working_vectors = false;
    params.sort_order = SortOrder::Ascending;

    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    auto ws = UnifiedVector<std::byte>(steqr_cta_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));

    steqr_cta<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(eigenvalues(i, j), expected_eigenvalues[j * n + i], 1e-5) 
                << "Eigenvalue mismatch at index " << i << ", batch " << j;
        }
    }
}

TYPED_TEST(SteqrTest, SteqrCtaRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1280;

    Vector<float_type> diag = Vector<float_type>::random(n, batch);
    Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);
    Vector<float_type> eigenvalues = Vector<float_type>::zeros(n, batch);
    auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
    dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();

    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params = {};
    params.max_sweeps = 10;  // Per-eigenvalue iteration limit
    params.sort = true;
    params.transpose_working_vectors = false;
    params.sort_order = SortOrder::Ascending;

    auto ws = UnifiedVector<std::byte>(steqr_cta_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));

    steqr_cta<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    // Compare with reference SYEV implementation
    UnifiedVector<float_type> ref_eigs(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B, float_type>(*this->ctx, dense_A, ref_eigs, JobType::NoEigenVectors, Uplo::Lower), std::byte(0));
    syev<B>(*this->ctx, dense_A, ref_eigs, JobType::NoEigenVectors, Uplo::Lower, syev_ws.to_span());
    this->ctx->wait();

    // Validate eigenvalues (not eigenvectors with random matrices for now)
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues(i, j), ref_eigs[i + j * n], std::numeric_limits<float_type>::epsilon()*5e2) 
                << "Eigenvalue value mismatch at index " << i << ", batch " << j;
        }
    }
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
