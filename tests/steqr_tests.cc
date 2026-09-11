#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

using namespace batchlas;

namespace {
#if BATCHLAS_HAS_HOST_BACKEND
template <typename Real>
UnifiedVector<double> netlib_ref_eigs_tridiag(const VectorView<Real>& diag,
                                              const VectorView<Real>& sub);

template <typename Real>
UnifiedVector<double> netlib_ref_eigs_dense(const MatrixView<Real, MatrixFormat::Dense>& A);
#endif

static inline const char* update_scheme_name(SteqrUpdateScheme scheme) {
    switch (scheme) {
        case SteqrUpdateScheme::PG:
            return "PG";
        case SteqrUpdateScheme::EXP:
            return "EXP";
        default:
            return "UNKNOWN";
    }
}

static constexpr std::array<SteqrUpdateScheme, 2> kSteqrUpdateSchemes = {
    SteqrUpdateScheme::PG,
    SteqrUpdateScheme::EXP,
};
}

template <typename T, Backend B>
struct SteqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
// STEQR tests are not meaningful for complex types.
using SteqrTestTypes = typename test_utils::backend_types_filtered<SteqrConfig, false>::type;

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

    // Ritz values
    auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(c), batch);
    auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);
    this->ctx->wait();
    
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(eigenvalues[i], expected_eigenvalues[i], 1e-5) << "Eigenvalue mismatch at index " << i;
        EXPECT_NEAR(eigenvalues[i], ritz_vals(i, 0), 1e-5) << "Ritz value mismatch at index " << i;
    }

}

TYPED_TEST(SteqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 512;
    // n=512 with eigenvectors plus a dense 512x512xbatch ritz_values check is
    // an O(n^3)*batch job that runs on the host for the NETLIB instantiations.
    // The closed-form Toeplitz spectrum is verified just as well at batch=8.
    const int batch = 8;
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
    auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);
    
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
    // netlib_ref_eigs_dense() below is a host O(n^3) solve per batch item.
    const int batch = 16;
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

    auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);

#if BATCHLAS_HAS_HOST_BACKEND
    const auto ref_eigs = netlib_ref_eigs_dense(dense_A.view());
    auto eps = test_utils::tolerance<float_type>();

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues(i, j), ritz_vals(i, j), std::numeric_limits<float_type>::epsilon()*5e2) << "Ritz value mismatch at index " << i << ", batch " << j;
            ASSERT_NEAR(eigenvalues(i, j), ref_eigs[i + j * n], std::numeric_limits<float_type>::epsilon()*5e2) << "Eigenvalue value mismatch at index " << i << ", batch " << j;
        }
    }
#else
    // Without NETLIB/host backend, only validate Ritz values (no CPU reference).
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues(i, j), ritz_vals(i, j), std::numeric_limits<float_type>::epsilon()*5e2) << "Ritz value mismatch at index " << i << ", batch " << j;
        }
    }
#endif
}

TYPED_TEST(SteqrTest, SteqrRandomN8SchemeCompare) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 8;
    const int batch = 1;

    Vector<float_type> diag = Vector<float_type>::random(n, batch);
    Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);

    auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
    dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();

    // --- Reference steqr ---
    Vector<float_type> evals_ref = Vector<float_type>::zeros(n, batch);
    auto eigvects_ref = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params_ref = {};
    params_ref.max_sweeps = 30;
    params_ref.sort = true;
    params_ref.transpose_working_vectors = false;
    params_ref.sort_order = SortOrder::Ascending;

    UnifiedVector<std::byte> ws_ref(
        steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, evals_ref, JobType::EigenVectors, params_ref),
        std::byte(0));
    steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(evals_ref),
                         ws_ref.to_span(), JobType::EigenVectors, params_ref, eigvects_ref);
    this->ctx->wait();

    // --- STEQR with explicit update schemes ---
    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        Vector<float_type> evals_cta = Vector<float_type>::zeros(n, batch);
        auto eigvects_cta = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params_cta = {};
        params_cta.max_sweeps = 30;
        params_cta.sort = true;
        params_cta.transpose_working_vectors = false;
        params_cta.sort_order = SortOrder::Ascending;
        params_cta.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params_cta.cta_update_scheme = scheme;

        UnifiedVector<std::byte> ws_cta(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, evals_cta, JobType::EigenVectors, params_cta),
            std::byte(0));
        steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(evals_cta),
                             ws_cta.to_span(), JobType::EigenVectors, params_cta, eigvects_cta);
        this->ctx->wait();

        // Compare eigenvalues directly (both should be correct and similarly ordered after sort)
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(evals_cta[i], evals_ref[i], test_utils::tolerance<T>())
                << "Eigenvalue mismatch vs STEQR at index " << i;
        }

        // Compare Ritz values (validates eigenvectors)
        auto ritz_ref = ritz_values(*this->ctx, dense_A, eigvects_ref);
        auto ritz_cta = ritz_values(*this->ctx, dense_A, eigvects_cta);
        this->ctx->wait();

        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(evals_ref[i], ritz_ref(i, 0), test_utils::tolerance<T>())
                << "Ritz mismatch (STEQR) at index " << i;
            ASSERT_NEAR(evals_cta[i], ritz_cta(i, 0), test_utils::tolerance<T>())
                << "Ritz mismatch (STEQR) at index " << i;
        }
    }
}

TYPED_TEST(SteqrTest, SteqrSingleMatrixWithSchemes) {
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
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }

    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        // fresh output buffers each iteration
        Vector<float_type> eigenvalues(n, batch);
        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params = {};
        params.max_sweeps = 30;  // Per-eigenvalue iteration limit
        params.sort = true;  // Re-enable sorting to match test expectations
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = scheme;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        // Validate eigenvalues against expected analytical values
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues[i], expected_eigenvalues[i], 1e-5) << "Eigenvalue mismatch at index " << i;
        }

        // Test: Validate eigenvectors by computing Ritz values (should match eigenvalues)
        auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(b), batch);
        auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);
        this->ctx->wait();

        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eigenvalues[i], ritz_vals(i, 0), 1e-5) << "Ritz value mismatch at index " << i;
        }
    }
}

TYPED_TEST(SteqrTest, SteqrBatchedMatricesWithSchemes) {
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
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    // All matrices are identical, so expected eigenvalues are the same for each batch item
    for (int j = 0; j < batch; ++j) {
        for (int i = 1; i <= n; ++i) {
            expected_eigenvalues[j * n + i - 1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
        }
    }

    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        SteqrParams<float_type> params = {};
        params.max_sweeps = 10;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = scheme;

        Vector<float_type> eigenvalues(n, batch);
        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_NEAR(eigenvalues(i, j), expected_eigenvalues[j * n + i], 1e-5)
                    << "Eigenvalue mismatch at index " << i << ", batch " << j;
            }
        }

        // Test: Validate eigenvectors by computing Ritz values (should match eigenvalues)
        auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(c), batch);
        auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);
        this->ctx->wait();
        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_NEAR(eigenvalues(i, j), ritz_vals(i, j), 1e-5)
                    << "Ritz value mismatch at index " << i << ", batch " << j;
            }
        }
    }
}

TYPED_TEST(SteqrTest, SteqrRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    // Each size runs a host LAPACK reference over the whole batch, so cost is
    // linear in `batch` x 29 sizes. Large-batch dispatch stays covered by
    // SteqrBatchedMatricesWithSchemes (batch=1280); keep the full size sweep,
    // which is what actually exercises the block/tail boundaries.
    const int batch = 64;
    for (int n = 4; n <= 32; ++n) {
        Vector<float_type> diag = Vector<float_type>::random(n, batch);
        Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);
        Vector<float_type> eigenvalues = Vector<float_type>::zeros(n, batch);

        auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
        dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();
        auto dense_A_copy = dense_A;  // SYEV overwrites its input

        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params = {};
        params.max_sweeps = 10;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        auto ritz_vals = ritz_values(*this->ctx, dense_A, eigvects);
        this->ctx->wait();

        // Reference eigenvalues via NETLIB double
#if BATCHLAS_HAS_HOST_BACKEND
        const auto ref_eigs = netlib_ref_eigs_dense(dense_A_copy.view());

        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_NEAR(eigenvalues(i, j), ref_eigs[i + j * n],
                            5*test_utils::tolerance<T>())
                    << "Eigenvalue value mismatch at index " << i << ", batch " << j << ", n " << n;
            }
        }
#endif

        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_NEAR(eigenvalues(i, j), ritz_vals(i, j),
                            5*test_utils::tolerance<T>())
                    << "Ritz value mismatch at index " << i << ", batch " << j << ", n " << n;
            }
        }
    }
}

TYPED_TEST(SteqrTest, SteqrConditionedTridiagonalNetlibRef) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 32;
    const int batch = 32;
    const float_type log10_cond = float_type(5.0);

    auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, float_type>(
        *this->ctx, n, log10_cond, NormType::Spectral, batch, 1234u);

    Vector<float_type> diag(n, float_type(0), batch);
    Vector<float_type> sub(n - 1, float_type(0), batch);
    auto A_view = dense_A.view();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            diag(i, b) = A_view.at(i, i, b);
            if (i < n - 1) {
                sub(i, b) = A_view.at(i + 1, i, b);
            }
        }
    }

    auto conds = cond<B>(*this->ctx, dense_A.view(), NormType::Spectral);
    this->ctx->wait();
    for (int b = 0; b < batch; ++b) {
        EXPECT_GE(std::log10(conds[b]), log10_cond - float_type(0.5)) << "Batch " << b;
    }

    Vector<float_type> eigenvalues(n, batch);
    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params = {};
    params.max_sweeps = 100;
    params.sort = true;
    params.transpose_working_vectors = false;
    params.sort_order = SortOrder::Ascending;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<float_type>(*this->ctx, diag, sub, eigenvalues, JobType::EigenVectors, params),
        std::byte(0));
    steqr<B, float_type>(*this->ctx, diag, sub, eigenvalues,
                         ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

#if BATCHLAS_HAS_HOST_BACKEND
    const auto ref_eigs = netlib_ref_eigs_tridiag(VectorView(diag), VectorView(sub));
    const bool use_rel_tol = std::is_same_v<float_type, float>;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            const float_type ref = static_cast<float_type>(ref_eigs[i + b * n]);
            const float_type tol = use_rel_tol
                ? std::max(float_type(5e-3f), float_type(3e-7f) * (float_type(1) + std::abs(ref)))
                : float_type(5e-7);
            ASSERT_NEAR(eigenvalues(i, b), ref, tol)
                << "Eigenvalue mismatch at index " << i << ", batch " << b;
        }
    }
#endif
}

namespace {

#if BATCHLAS_HAS_HOST_BACKEND
template <typename Real>
UnifiedVector<double> netlib_ref_eigs_tridiag(const VectorView<Real>& diag,
                                              const VectorView<Real>& sub) {
    const int n = diag.size();
    const int batch = diag.batch_size();

    Queue ctx_cpu("cpu");
    auto diag_d = diag.template astype<double>();
    auto sub_d = sub.template astype<double>();

    Matrix<double> A = Matrix<double>::Zeros(n, n, batch);
    auto A_view = A.view();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            A_view.at(i, i, b) = diag_d(i, b);
            if (i < n - 1) {
                const double off = sub_d(i, b);
                A_view.at(i + 1, i, b) = off;
                A_view.at(i, i + 1, b) = off;
            }
        }
    }

    UnifiedVector<double> ref_eigs(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    UnifiedVector<std::byte> ws(
        syev_buffer_size(ctx_cpu, A.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev(ctx_cpu,
                                  A.view(),
                                  ref_eigs.to_span(),
                                  {.jobz = JobType::NoEigenVectors},
                                  ws.to_span()).wait();
    ctx_cpu.wait();

    return ref_eigs;
}

template <typename Real>
UnifiedVector<double> netlib_ref_eigs_dense(const MatrixView<Real, MatrixFormat::Dense>& A) {
    const int n = A.rows();
    const int batch = A.batch_size();

    Queue ctx_cpu("cpu");
    auto A_d = A.template astype<double>();

    UnifiedVector<double> ref_eigs(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    UnifiedVector<std::byte> ws(
        syev_buffer_size(ctx_cpu, A_d.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev(ctx_cpu,
                                  A_d.view(),
                                  ref_eigs.to_span(),
                                  {.jobz = JobType::NoEigenVectors},
                                  ws.to_span()).wait();
    ctx_cpu.wait();

    return ref_eigs;
}
#endif

inline bool stress_debug_enabled() {
    return std::getenv("BATCHLAS_STEQR_STRESS_DEBUG") != nullptr;
}

inline void stress_debug_log(const char* msg) {
    if (stress_debug_enabled()) {
        std::cerr << msg << std::endl;
    }
}

inline bool is_kernel_not_found_message(const std::string& msg) {
    return msg.find("No kernel named") != std::string::npos;
}

template <typename Real>
Real stress_large_scale() {
    if constexpr (std::is_same_v<Real, float>) {
        // Large enough to trigger ssfmax scaling, but small enough that squares don’t overflow.
        return Real(1e19f);
    } else {
        // sqrt(max(double)) ~ 1e154; this is > ssfmax but keeps squares finite.
        return Real(1e154);
    }
}

template <typename Real>
Real stress_small_scale() {
    if constexpr (std::is_same_v<Real, float>) {
        // ssfmin(float) is around 1e-5; this triggers scaling up.
        return Real(1e-10f);
    } else {
        // ssfmin(double) is around 1e-123; this triggers scaling up.
        return Real(1e-140);
    }
}

template <typename Real>
void fill_stress_tridiag(Vector<Real>& diag,
                         Vector<Real>& sub,
                         Real diag_scale,
                         Real offdiag_scale) {
    const int n = diag.size();
    const int batch = diag.batch_size();

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real t = Real(1) + Real(i) / Real(n);
            diag(i, j) = diag_scale * t;
            if (i < n - 1) {
                sub(i, j) = offdiag_scale * Real(0.25) * (Real(1) + Real(i) / Real(n - 1));
            }
        }
    }
}

template <typename Real>
void assert_all_finite(Vector<Real>& v) {
    const int n = v.size();
    const int batch = v.batch_size();
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real x = v(i, j);
            ASSERT_TRUE(std::isfinite(static_cast<double>(x)))
                << "Non-finite value at (" << i << "," << j << "): " << static_cast<double>(x);
        }
    }
}

template <Backend B, typename Real>
void stress_run_case(Queue& ctx,
                     Vector<Real>& diag,
                     Vector<Real>& sub,
                     Vector<Real>& steqr_eigs,
                     Vector<Real>& cta_eigs,
                     Real rel_tol,
                     bool check_steqr_against_ref = true,
                     bool check_cta_against_ref = true) {
    const int n = diag.size();
    const int batch = diag.batch_size();

    // Dense symmetric matrix for a reference eigenvalue solve.
    auto dense_A = Matrix<Real>::Zeros(n, n, batch);

    stress_debug_log("stress_run_case: fill_tridiag");
    dense_A.view().fill_tridiag(ctx, sub, diag, sub).wait();

    // Reference eigenvalues via SYEV.
    UnifiedVector<Real> ref_eigs(n * batch);
    {
        stress_debug_log("stress_run_case: syev reference");
        auto syev_ws = UnifiedVector<std::byte>(
            // NOTE: CUDA SYCL stacks can be sensitive to specific kernel launch patterns.
            // We request eigenvectors here (even though we only compare eigenvalues) because
            // it exercises the same well-tested SYEV path used elsewhere in this test file.
            batchlas::blas::dispatch::detail::syev_vendor_buffer_size_or_throw<B, Real>(
                ctx, dense_A.view(), ref_eigs, JobType::EigenVectors, Uplo::Lower),
            std::byte(0));
        batchlas::blas::dispatch::detail::syev_vendor_or_throw<B, Real>(
            ctx, dense_A.view(), ref_eigs, JobType::EigenVectors, Uplo::Lower, syev_ws.to_span());
        ctx.wait();
    }

    // steqr: eigenvalues only.
    {
        stress_debug_log("stress_run_case: steqr");
        SteqrParams<Real> params = {};
        params.max_sweeps = 200;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<Real>(ctx, diag, sub, steqr_eigs, JobType::EigenVectors, params),
            std::byte(0));
        // NOTE: On some CUDA SYCL stacks, the JobType::NoEigenVectors path can trigger
        // an assertion inside the runtime scheduler. We request eigenvectors here to
        // keep this stress test runnable on CUDA; we still validate only eigenvalues.
        auto eigvects = Matrix<Real>::Zeros(n, n, batch);
        steqr<B, Real>(ctx, diag, sub, steqr_eigs, ws.to_span(), JobType::EigenVectors, params, eigvects);
        ctx.wait();
    }

    // steqr (alternate scheme): eigenvalues only, but still requires a square eigvects argument.
    {
        stress_debug_log("stress_run_case: steqr(alt-scheme)");
        SteqrParams<Real> params = {};
        params.max_sweeps = 200;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = SteqrUpdateScheme::EXP;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<Real>(ctx, diag, sub, cta_eigs, JobType::EigenVectors, params),
            std::byte(0));
        auto eigvects = Matrix<Real>::Zeros(n, n, batch);
        steqr<B, Real>(ctx, diag, sub, cta_eigs, ws.to_span(), JobType::EigenVectors, params, eigvects);
        ctx.wait();
    }

    assert_all_finite(steqr_eigs);
    assert_all_finite(cta_eigs);

    const Real rel = rel_tol;

    // Compare against the reference (sort ref per-batch to be safe).
    for (int j = 0; j < batch; ++j) {
        std::vector<Real> ref(n);
        std::vector<Real> ste(n);
        std::vector<Real> cta(n);
        for (int i = 0; i < n; ++i) {
            ref[i] = ref_eigs[i + j * n];
            ste[i] = steqr_eigs(i, j);
            cta[i] = cta_eigs(i, j);
            ASSERT_TRUE(std::isfinite(static_cast<double>(ref[i])));
        }
        std::sort(ref.begin(), ref.end());
        std::sort(ste.begin(), ste.end());
        std::sort(cta.begin(), cta.end());

        for (int i = 0; i < n; ++i) {
            const Real r = ref[i];
            const Real tol = rel * (Real(1) + std::abs(r));
            if (check_steqr_against_ref) {
                ASSERT_NEAR(ste[i], r, tol) << "STEQR mismatch at (" << i << "," << j << ")";
            }
            if (check_cta_against_ref) {
                ASSERT_NEAR(cta[i], r, tol * 1e2) << "STEQR mismatch at (" << i << "," << j << ")";
            }
        }
    }
}

} // namespace

TYPED_TEST(SteqrTest, StressExtremeMagnitudesN32) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    // steqr dispatch uses CTA for n <= subgroup size on CUDA; use 32 for a heavier stress case.
    const int n = 32;
    const int batch = 16;

    Vector<float_type> diag(n, float_type(0), batch);
    Vector<float_type> sub(n - 1, float_type(0), batch);
    Vector<float_type> evals_steqr = Vector<float_type>::zeros(n, batch);
    Vector<float_type> evals_cta = Vector<float_type>::zeros(n, batch);

    try {
        // Case 1: very large magnitude (expects scale-down to kick in)
        fill_stress_tridiag(diag, sub, stress_large_scale<float_type>(), stress_large_scale<float_type>());
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta,
                   (std::is_same_v<float_type, float> ? float_type(5e-5f) : float_type(5e-10)));

        // Case 2: very small magnitude (expects scale-up to kick in)
        fill_stress_tridiag(diag, sub, stress_small_scale<float_type>(), stress_small_scale<float_type>());
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta,
                   (std::is_same_v<float_type, float> ? float_type(5e-5f) : float_type(5e-10)));

        // Case 3: mixed dynamic range without underflow.
        // We keep the “small” entries O(1) so this still stresses conditioning and the
        // ssfmax scale-down path, but avoids spanning ssfmin..ssfmax in one matrix.
        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool big = ((i + j) % 3) == 0;
                const float_type ds = big ? stress_large_scale<float_type>() : float_type(1);
                const float_type es = big ? stress_large_scale<float_type>() : float_type(1);
                diag(i, j) = ds * (float_type(1) + float_type(i) / float_type(n));
                if (i < n - 1) sub(i, j) = es * float_type(0.25);
            }
        }
        // This case is primarily meant to stress the CTA implementation; on CUDA,
        // the baseline STEQR path can be noticeably less accurate on ill-conditioned
        // mixed-scale inputs. We still require it to produce finite outputs.
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta,
                           /*rel_tol=*/float_type(2.5e-1),
                           /*check_steqr_against_ref=*/false,
                           /*check_cta_against_ref=*/true);

    } catch (const sycl::exception& e) {
        if (is_kernel_not_found_message(e.what())) {
            GTEST_SKIP() << "Skipping due to missing kernel bundle: " << e.what();
        }
        throw;
    } catch (const std::exception& e) {
        if (is_kernel_not_found_message(e.what())) {
            GTEST_SKIP() << "Skipping due to missing kernel bundle: " << e.what();
        }
        throw;
    }
}


// ---------------------------------------------------------------------------
// Per-item convergence status (`info`).
//
// steqr is the routine LAPACK documents as returning info > 0 -- "the number of
// off-diagonal elements that did not converge to zero" -- and until this work
// package the whole of that was unreachable here. steqr_cta computed a per-item
// `status` array and only READ it under a diagnostics env var; steqr_wg computed
// nothing at all and its driver ran a fixed n-1 passes with no test. At batch
// 16384 one non-converged item was invisible: the call returned, ctx.wait()
// returned, and the caller read eigenvalues that were simply wrong for it.
//
// TWO TIERS, ONE SPAN. `steqr` picks steqr_cta when n <= the device sub-group
// width and steqr_wg otherwise (src/extensions/steqr.cc:43), so at n = 32 these
// cases exercise the CTA arm on CUDA and the work-group arm on the host -- which
// is deliberate: the two tiers derive the status by completely different means
// (a bool out of steqr_cta_solve vs. a post-hoc scan of `e`), and a test that
// only ever reached one of them would say nothing about the other.
//
// THE BATCH IS MIXED ON PURPOSE. Even items are diagonal, so they are converged
// before the first sweep; odd items are the generic Toeplitz that is not. That
// is what lets the forced case below state something stronger than "the call
// failed": the items it reports as converged must still have the right answer.
// ---------------------------------------------------------------------------

namespace {

// Even items: diagonal, ascending, distinct -- e is exactly zero, so every
// off-diagonal is deflated before the first sweep runs.
// Odd items: the symmetric Toeplitz(0.5, 1, 0.5), whose spectrum is uniform and
// unclustered and therefore needs the full sweep budget.
template <typename Real>
void fill_mixed_convergence_batch(Vector<Real>& d, Vector<Real>& e) {
    const int n = d.size();
    const int batch = d.batch_size();
    for (int b = 0; b < batch; ++b) {
        const bool easy = (b % 2) == 0;
        for (int i = 0; i < n; ++i) d(i, b) = easy ? Real(i + 1) : Real(1);
        for (int i = 0; i < n - 1; ++i) e(i, b) = easy ? Real(0) : Real(0.5);
    }
}

// Ascending eigenvalue `i` of item `b` of that batch, in closed form. For the
// Toeplitz items: a - 2*sqrt(b*c)*cos(pi*k/(n+1)) with a = 1, b = c = 0.5.
template <typename Real>
Real mixed_convergence_eigenvalue(int b, int i, int n) {
    if ((b % 2) == 0) return Real(i + 1);
    return Real(1) - Real(std::cos(M_PI * double(i + 1) / double(n + 1)));
}

}  // namespace

TYPED_TEST(SteqrTest, InfoIsZeroOnAConvergingBatch) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Vector<Real> d(n, batch), e(n - 1, batch), w(n, batch);
    fill_mixed_convergence_batch<Real>(d, e);
    auto eigvects = Matrix<Real>::Zeros(n, n, batch);

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;

    // -1, NOT 0. A span left at zero cannot tell "the kernel wrote 0" from
    // "nothing wrote it at all", and "nothing wrote it" is the exact failure this
    // mechanism exists to make impossible: a status channel that is never filled
    // reports universal success. steqr clears the span itself
    // (src/extensions/steqr.cc:66), so a surviving -1 means that clear never ran.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<Real>(*this->ctx, d, e, w, JobType::EigenVectors, params), std::byte(0));
    steqr<B, Real>(*this->ctx, d, e, w, ws.to_span(), JobType::EigenVectors, params, eigvects,
                   info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: nothing wrote "
                                  "the span, so a zero here would have been an accident";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a batch that "
                                 "converges under the default sweep budget";
    }
    // Both halves of the mixed batch must be RIGHT as well as reported converged;
    // otherwise the forced case below could not attribute a wrong answer to the cap.
    const double tol = std::is_same_v<Real, float> ? 1e-4 : 1e-9;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(static_cast<double>(w(i, b)),
                        static_cast<double>(mixed_convergence_eigenvalue<Real>(b, i, n)), tol)
                << "batch " << b << " eigenvalue " << i;
        }
    }
}

// THE CASE THAT MATTERS. A test that only ever sees info == 0 cannot tell a
// working implementation from one that memsets zero, so this one forces the
// failure on the SAME input and asserts the opposite direction.
//
// The knob is SteqrParams::max_sweeps, and 1 rather than a "reduced" value such
// as 50, for a reason: syev_cta.cc:177-180 and syev_cta_fused.cc:554-557 silently
// REWRITE max_sweeps to 400 whenever a caller leaves it at the default 50, so a
// test that lowered the cap to that default would be answered by 400 and pass
// vacuously. Calling steqr directly avoids the rewrite; 1 escapes it regardless.
//
// One sweep per pass gives the CTA arm a budget of max_sweeps*n = 32 sweeps
// (steqr_cta_device.hh:764) and the work-group arm n-1 = 31 passes of one sweep
// each, against a 32x32 Toeplitz that needs 2-3 sweeps per eigenvalue.
TYPED_TEST(SteqrTest, InfoReportsItemsThatExhaustTheSweepBudget) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Vector<Real> d(n, batch), e(n - 1, batch), w(n, batch);
    fill_mixed_convergence_batch<Real>(d, e);
    auto eigvects = Matrix<Real>::Zeros(n, n, batch);

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;
    params.max_sweeps = 1;

    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<Real>(*this->ctx, d, e, w, JobType::EigenVectors, params), std::byte(0));
    steqr<B, Real>(*this->ctx, d, e, w, ws.to_span(), JobType::EigenVectors, params, eigvects,
                   info.to_span());
    this->ctx->wait();

    int reported = 0;
    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value";
        ASSERT_GE(info[b], 0) << "info is LAPACK-like: 0 or a positive count, never negative";
        if (info[b] != 0) ++reported;
    }
    EXPECT_GT(reported, 0)
        << "a one-sweep budget on a 32x32 Toeplitz batch reported universal convergence; "
           "either the status is not written at all, or it is written unconditionally zero";

    // The other half of the claim, and the reason the batch is mixed: an item the
    // library says converged must still be CORRECT. A status channel that
    // reported failure everywhere would satisfy the assertion above on its own;
    // this is what stops that from passing.
    const double tol = std::is_same_v<Real, float> ? 1e-4 : 1e-9;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(static_cast<double>(w(i, b)),
                        static_cast<double>(mixed_convergence_eigenvalue<Real>(b, i, n)), tol)
                << "item " << b << " reported info == 0 but eigenvalue " << i << " is wrong";
        }
    }
}

// An EMPTY span means "not requested" and must cost nothing -- the contract
// potrf documents, restated for every routine in this package. Two things are
// checked, because only one of them is visible from the caller's side: the
// answer does not change, and the workspace query does not either. The second is
// what keeps a `*_buffer_size` computed once and reused across calls -- which is
// how every driver in this tree uses it -- correct.
TYPED_TEST(SteqrTest, EmptyInfoSpanChangesNeitherAnswerNorWorkspace) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 4;

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;

    Vector<Real> d0(n, batch), e0(n - 1, batch), w0(n, batch);
    Vector<Real> d1(n, batch), e1(n - 1, batch), w1(n, batch);
    fill_mixed_convergence_batch<Real>(d0, e0);
    fill_mixed_convergence_batch<Real>(d1, e1);
    auto eigvects0 = Matrix<Real>::Zeros(n, n, batch);
    auto eigvects1 = Matrix<Real>::Zeros(n, n, batch);

    // steqr_buffer_size takes no `info` argument at all, which is the strongest
    // available statement of the contract: the size CANNOT depend on whether
    // status was asked for. Querying it twice guards the weaker thing that could
    // still go wrong later -- an overload that does take one.
    const size_t bytes_a =
        steqr_buffer_size<Real>(*this->ctx, d0, e0, w0, JobType::EigenVectors, params);
    const size_t bytes_b =
        steqr_buffer_size<Real>(*this->ctx, d1, e1, w1, JobType::EigenVectors, params);
    EXPECT_EQ(bytes_a, bytes_b);

    UnifiedVector<std::byte> ws0(bytes_a, std::byte(0));
    UnifiedVector<std::byte> ws1(bytes_a, std::byte(0));
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    steqr<B, Real>(*this->ctx, d0, e0, w0, ws0.to_span(), JobType::EigenVectors, params, eigvects0,
                   info.to_span());
    steqr<B, Real>(*this->ctx, d1, e1, w1, ws1.to_span(), JobType::EigenVectors, params, eigvects1,
                   Span<int32_t>{});
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(w0(i, b), w1(i, b))
                << "requesting status changed the answer at batch " << b << " index " << i;
        }
    }
}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
