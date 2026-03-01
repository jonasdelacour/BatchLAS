#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include "test_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
struct StedcConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using StedcTestTypes = typename test_utils::backend_types<StedcConfig>::type;

template <typename Config>
class StedcTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(StedcTest, StedcTestTypes);

TYPED_TEST(StedcTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 512;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::ones(n, batch);
    auto b = Vector<float_type>::ones(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 32};

    UnifiedVector<std::byte> ws(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params));

    stedc<B>(*this->ctx, a, b, eigvals,
                      ws, JobType::EigenVectors, params, eigvects);
    
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);

    Matrix<float_type> reconstructed = Matrix<float_type>::TriDiagToeplitz(n, float_type(1), float_type(1), float_type(1), batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B>(*(this->ctx), reconstructed, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));

    auto ritz_vals = ritz_values<B, float_type>(*this->ctx, reconstructed, eigvects);
    syev<B>(*(this->ctx), reconstructed, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower, syev_ws);
    this->ctx->wait();
    auto ref_view = VectorView<float_type>(ref_eigvals, n, batch);

    auto tol = 1e-3f;
    if (!VectorView<float_type>::all_close(*(this->ctx), eigvals, ref_view, tol)) {
        FAIL() << "Eigenvalues do not match reference within tolerance " << tol;
    }

    if (!VectorView<float_type>::all_close(*(this->ctx), eigvals, ritz_vals, tol)) {
        FAIL() << "Eigenvalues do not match Ritz values within tolerance " << tol;
    }
}

TYPED_TEST(StedcTest, BatchedRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 1024;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::random(n, batch);
    auto b = Vector<float_type>::random(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 16};

    UnifiedVector<std::byte> ws(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params));

    Matrix<float_type> reconstructed = Matrix<float_type>::Zeros(n, n, batch);
    reconstructed.view().fill_tridiag(*this->ctx, b, a, b).wait();
    this->ctx->wait();
    
    stedc<B>(*this->ctx, a, b, eigvals,
                      ws, JobType::EigenVectors, params, eigvects);
    
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);

    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B>(*(this->ctx), reconstructed, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));

    auto ritz_vals = ritz_values<B, float_type>(*this->ctx, reconstructed, eigvects);
    syev<B>(*(this->ctx), reconstructed, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower, syev_ws);
    this->ctx->wait();

    auto ref_view = VectorView<float_type>(ref_eigvals, n, batch);
    auto diff_vect = Vector<float_type>::zeros(n, batch);
    
    VectorView<float_type>::add(*(this->ctx), float_type(1.0), float_type(-1.0), eigvals, ref_view, diff_vect).wait();

    auto tol = std::numeric_limits<float_type>::epsilon()*5e2;
    for (int j = 0; j < batch; j++) {
        for (int i = 0; i < n; i++) {
            float_type diff = std::abs(eigvals(i, j) - ref_view(i, j));
            if (diff > tol) {
                FAIL() << "Eigenvalue mismatch at index " << i << " in batch " << j << ": computed " << eigvals(i, j) << ", reference " << ref_view(i, j) << ", diff " << diff << " exceeds tol " << tol;
            }
        }
    }
    
    /* if (!VectorView<float_type>::all_close(*(this->ctx), eigvals, ref_view, tol)) {
        FAIL() << "Eigenvalues do not match reference within tolerance \n" <<
        eigvals << "\n vs \n" << ref_view << "\n";
    }   
    if (!VectorView<float_type>::all_close(*(this->ctx), eigvals, ritz_vals, tol)) {
        FAIL() << "Eigenvalues do not match Ritz values within tolerance \n" <<
        eigvals << "\n vs \n" << ritz_vals << "\n";
    } */
}

TYPED_TEST(StedcTest, FlatMatchesRecursive) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 128;
    const int batch = 16;
    using float_type = typename base_type<T>::type;

    // Generate identical inputs for both paths (stedc mutates its inputs)
    auto a_ref = Vector<float_type>::random(n, batch);
    auto b_ref = Vector<float_type>::random(n - 1, batch);
    auto a_flat = a_ref;
    auto b_flat = b_ref;
    


    auto eigvals_ref = Vector<float_type>::zeros(n, batch);
    auto eigvals_flat = Vector<float_type>::zeros(n, batch);
    auto eigvecs_ref = Matrix<float_type>::Identity(n, batch);
    auto eigvecs_flat = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params{.recursion_threshold = 16};
    UnifiedVector<std::byte> ws_ref(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params));
    UnifiedVector<std::byte> ws_flat(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params));

    // Baseline recursive path
    stedc<B>(*this->ctx, a_ref, b_ref, eigvals_ref, ws_ref, JobType::EigenVectors, params, eigvecs_ref);

    // Flattened path
    stedc_flat<B>(*this->ctx, a_flat, b_flat, eigvals_flat, ws_flat, JobType::EigenVectors, params, eigvecs_flat);

    this->ctx->wait();

    auto tol = std::numeric_limits<float_type>::epsilon() * float_type(1e3);
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            float_type diff = std::abs(eigvals_ref(i, j) - eigvals_flat(i, j));
            if (diff > tol) {
                FAIL() << "Eigenvalue mismatch at (" << i << ", batch " << j << ") : recursive="
                       << eigvals_ref(i, j) << " flat=" << eigvals_flat(i, j) << " diff=" << diff
                       << " tol=" << tol;
            }
        }
    }
}

TYPED_TEST(StedcTest, FusedMergeMatchesBaseline) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 128;
    const int batch = 64;
    using float_type = typename base_type<T>::type;

    // Generate identical inputs for both paths (stedc mutates its inputs)
    auto a_base = Vector<float_type>::random(n, batch);
    auto b_base = Vector<float_type>::random(n - 1, batch);
    auto a_fused = a_base;
    auto b_fused = b_base;

    auto eigvals_base = Vector<float_type>::zeros(n, batch);
    auto eigvals_fused = Vector<float_type>::zeros(n, batch);
    auto eigvecs_base = Matrix<float_type>::Identity(n, batch);
    auto eigvecs_fused = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params_base{.recursion_threshold = 16};
    StedcParams<float_type> params_fused{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::Fused,
        .enable_rescale = true,
    };

    UnifiedVector<std::byte> ws_base(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_base));
    UnifiedVector<std::byte> ws_fused(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_fused));

    stedc<B>(*this->ctx, a_base, b_base, eigvals_base, ws_base, JobType::EigenVectors, params_base, eigvecs_base);
    stedc<B>(*this->ctx, a_fused, b_fused, eigvals_fused, ws_fused, JobType::EigenVectors, params_fused, eigvecs_fused);
    this->ctx->wait();

    auto tol = std::numeric_limits<float_type>::epsilon() * float_type(5e3);
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            float_type diff = std::abs(eigvals_base(i, j) - eigvals_fused(i, j));
            if (diff > tol) {
                FAIL() << "FusedMerge eigenvalue mismatch at (" << i << ", batch " << j << ") : baseline="
                       << eigvals_base(i, j) << " fused=" << eigvals_fused(i, j) << " diff=" << diff
                       << " tol=" << tol;
            }
        }
    }
}

TYPED_TEST(StedcTest, FusedCtaMergeMatchesReference) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "CTA merge is GPU-only"; }
    const int n = 64;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    auto a_cta = Vector<float_type>::random(n, batch);
    auto b_cta = Vector<float_type>::random(n - 1, batch);

    // Build dense tridiagonal for syev reference
    Matrix<float_type> T_mat = Matrix<float_type>::Zeros(n, n, batch);
    T_mat.view().fill_tridiag(*this->ctx, b_cta, a_cta, b_cta).wait();
    this->ctx->wait();

    auto eigvals_cta = Vector<float_type>::zeros(n, batch);
    auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params_cta{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = 32,
    };

    UnifiedVector<std::byte> ws_cta(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    stedc<B>(*this->ctx, a_cta, b_cta, eigvals_cta, ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta);
    this->ctx->wait();

    // syev reference eigenvalues
    UnifiedVector<float_type> ref_eigvals(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower, syev_ws);
    this->ctx->wait();
    auto ref_view = VectorView<float_type>(ref_eigvals, n, batch);

    // CTA solver uses origin-shifted quadratic interpolation adapted from the ROC solver.
    auto tol = std::is_same_v<float_type, float> ? float_type(1e-4) : float_type(1e-9);
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            float_type diff = std::abs(ref_view(i, j) - eigvals_cta(i, j));
            if (diff > tol) {
                FAIL() << "FusedCta eigenvalue mismatch vs syev at (" << i << ", batch " << j << ") : ref="
                       << ref_view(i, j) << " cta=" << eigvals_cta(i, j) << " diff=" << diff
                       << " tol=" << tol;
            }
        }
    }
}

TYPED_TEST(StedcTest, FusedCtaPartitionWidths) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "CTA merge is GPU-only"; }
    const int n = 64;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    auto a_saved = Vector<float_type>::random(n, batch);
    auto b_saved = Vector<float_type>::random(n - 1, batch);

    // Build dense tridiagonal for syev reference
    Matrix<float_type> T_mat = Matrix<float_type>::Zeros(n, n, batch);
    T_mat.view().fill_tridiag(*this->ctx, b_saved, a_saved, b_saved).wait();
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower, syev_ws);
    this->ctx->wait();
    auto ref_view = VectorView<float_type>(ref_eigvals, n, batch);

    auto tol_vs_ref = std::is_same_v<float_type, float> ? float_type(1e-4) : float_type(1e-9);

    // Run each partition width and check against syev reference
    for (int P : {4, 8, 16, 32}) {
        auto a_cta = a_saved;
        auto b_cta = b_saved;
        auto eigvals_cta = Vector<float_type>::zeros(n, batch);
        auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

        StedcParams<float_type> params_cta{
            .recursion_threshold = 16,
            .merge_variant = StedcMergeVariant::FusedCta,
            .enable_rescale = true,
            .secular_threads_per_root = P,
        };

        UnifiedVector<std::byte> ws_cta(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
        stedc<B>(*this->ctx, a_cta, b_cta, eigvals_cta, ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta);
        this->ctx->wait();

        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                float_type diff = std::abs(ref_view(i, j) - eigvals_cta(i, j));
                if (diff > tol_vs_ref) {
                    FAIL() << "FusedCta P=" << P << " eigenvalue mismatch vs syev at (" << i << ", batch " << j
                           << ") : ref=" << ref_view(i, j) << " cta=" << eigvals_cta(i, j)
                           << " diff=" << diff << " tol=" << tol_vs_ref;
                }
            }
        }
    }
}

TYPED_TEST(StedcTest, FusedCtaFallsBackToNonChunkedWhenRequestedExceedsMaxSubgroup) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "CTA merge is GPU-only"; }
    const int n = 64;
    const int batch = 128;
    using float_type = typename base_type<T>::type;

    constexpr int forced_threads_per_root = 1024;

    auto a_cta = Vector<float_type>::random(n, batch);
    auto b_cta = Vector<float_type>::random(n - 1, batch);

    Matrix<float_type> T_mat = Matrix<float_type>::Zeros(n, n, batch);
    T_mat.view().fill_tridiag(*this->ctx, b_cta, a_cta, b_cta).wait();
    this->ctx->wait();

    auto eigvals_cta = Vector<float_type>::zeros(n, batch);
    auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params_cta{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = forced_threads_per_root,
    };

    UnifiedVector<std::byte> ws_cta(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    stedc<B>(*this->ctx, a_cta, b_cta, eigvals_cta, ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta);
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev<B>(*(this->ctx), T_mat, ref_eigvals, JobType::NoEigenVectors, Uplo::Lower, syev_ws);
    this->ctx->wait();
    auto ref_view = VectorView<float_type>(ref_eigvals, n, batch);

    auto tol = std::is_same_v<float_type, float> ? float_type(1e-4) : float_type(1e-9);
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            float_type diff = std::abs(ref_view(i, j) - eigvals_cta(i, j));
            if (diff > tol) {
                FAIL() << "FusedCta non-chunked fallback mismatch vs syev at (" << i << ", batch " << j
                       << ") : ref=" << ref_view(i, j) << " cta=" << eigvals_cta(i, j)
                       << " diff=" << diff << " tol=" << tol;
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
