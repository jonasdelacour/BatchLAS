#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <filesystem>
#include <fstream>
#include "test_utils.hh"
#include "../src/queue.hh"
#include "../src/extensions/stedc_flattened.hh"
#include "../src/util/kernel-trace.hh"

using namespace batchlas;

namespace {

std::size_t count_trace_entries(const std::string& trace, const std::string& needle) {
    std::size_t count = 0;
    std::size_t pos = 0;
    while ((pos = trace.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

void reset_kernel_trace(const std::string& path) {
    {
        std::lock_guard<std::mutex> lock(batchlas_kernel_trace::g_mu);
        batchlas_kernel_trace::g_records.clear();
    }
    batchlas_kernel_trace::g_submit_counter.store(0);
    batchlas_kernel_trace::g_enabled.store(false);
    batchlas_kernel_trace::g_initialized.store(false);
    setenv("BATCHLAS_KERNEL_TRACE", "1", 1);
    setenv("BATCHLAS_KERNEL_TRACE_PATH", path.c_str(), 1);
    setenv("BATCHLAS_EXPERIMENTAL_STEDC_FLAT", "1", 1);
    std::filesystem::remove(path);
}

void disable_kernel_trace() {
    {
        std::lock_guard<std::mutex> lock(batchlas_kernel_trace::g_mu);
        batchlas_kernel_trace::g_records.clear();
    }
    batchlas_kernel_trace::g_submit_counter.store(0);
    batchlas_kernel_trace::g_enabled.store(false);
    batchlas_kernel_trace::g_initialized.store(false);
    unsetenv("BATCHLAS_KERNEL_TRACE");
    unsetenv("BATCHLAS_KERNEL_TRACE_PATH");
    unsetenv("BATCHLAS_EXPERIMENTAL_STEDC_FLAT");
}

std::string read_trace_file(const std::string& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

template <Backend B, typename T>
void expect_flat_matches_recursive_case(Queue& ctx,
                                        int n,
                                        int batch,
                                        JobType jobz,
                                        StedcParams<T> params,
                                        T eigenvalue_tol,
                                        bool check_ritz = false,
                                        T ritz_tol = T(0)) {
    auto diag_input = Vector<T>::random(n, batch);
    auto offdiag_input = Vector<T>::random(n - 1, batch);
    auto diag_ref = diag_input;
    auto offdiag_ref = offdiag_input;
    auto diag_flat = diag_input;
    auto offdiag_flat = offdiag_input;

    auto eigvals_ref = Vector<T>::zeros(n, batch);
    auto eigvals_flat = Vector<T>::zeros(n, batch);
    auto eigvecs_ref = Matrix<T>::Identity(n, batch);
    auto eigvecs_flat = Matrix<T>::Identity(n, batch);

    UnifiedVector<std::byte> ws_ref(stedc_workspace_size<B>(ctx, n, batch, jobz, params));
    UnifiedVector<std::byte> ws_flat(stedc_flat_workspace_size<B>(ctx, n, batch, jobz, params));

    stedc<B>(ctx, diag_ref, offdiag_ref, eigvals_ref, ws_ref, jobz, params, eigvecs_ref);
    stedc_flat<B>(ctx, diag_flat, offdiag_flat, eigvals_flat, ws_flat, jobz, params, eigvecs_flat);
    ctx.wait();

    const T effective_eigenvalue_tol = std::is_same_v<T, double>
        ? std::max<T>(eigenvalue_tol, std::numeric_limits<T>::epsilon() * T(1.2e5))
        : eigenvalue_tol;

    for (int batch_ix = 0; batch_ix < batch; ++batch_ix) {
        for (int i = 0; i < n; ++i) {
            const T diff = std::abs(eigvals_ref(i, batch_ix) - eigvals_flat(i, batch_ix));
            ASSERT_LE(diff, effective_eigenvalue_tol)
                << "Flat eigenvalue mismatch at n=" << n
                << ", batch=" << batch_ix
                << ", index=" << i
                << ": recursive=" << eigvals_ref(i, batch_ix)
                << " flat=" << eigvals_flat(i, batch_ix)
                << " diff=" << diff
                << " tol=" << effective_eigenvalue_tol;
        }
    }

    if (!check_ritz || jobz != JobType::EigenVectors) {
        return;
    }

    const T effective_ritz_tol = std::is_same_v<T, double>
        ? std::max<T>(ritz_tol, std::numeric_limits<T>::epsilon() * T(3e6))
        : ritz_tol;

    Matrix<T> reconstructed = Matrix<T>::Zeros(n, n, batch);
    reconstructed.view().fill_tridiag(ctx, offdiag_input, diag_input, offdiag_input).wait();
    ctx.wait();
    auto flat_ritz = ritz_values<B, T>(ctx, reconstructed, eigvecs_flat);
    ctx.wait();

    for (int batch_ix = 0; batch_ix < batch; ++batch_ix) {
        for (int i = 0; i < n; ++i) {
            const T diff = std::abs(flat_ritz(i, batch_ix) - eigvals_flat(i, batch_ix));
            ASSERT_LE(diff, effective_ritz_tol)
                << "Flat Ritz mismatch at n=" << n
                << ", batch=" << batch_ix
                << ", index=" << i
                << ": ritz=" << flat_ritz(i, batch_ix)
                << " eig=" << eigvals_flat(i, batch_ix)
                << " diff=" << diff
                << " tol=" << effective_ritz_tol;
        }
    }
}

template <Backend B, typename T>
void expect_flat_trace_counts(Queue& ctx,
                              int n,
                              int batch,
                              StedcParams<T> params,
                              std::size_t expected_merge_depths,
                              const std::string& trace_path) {
    (void)ctx;
    reset_kernel_trace(trace_path);

    Queue trace_ctx("gpu", true);

    auto diag = Vector<T>::random(n, batch);
    auto offdiag = Vector<T>::random(n - 1, batch);
    auto eigvals = Vector<T>::zeros(n, batch);
    auto eigvecs = Matrix<T>::Identity(n, batch);

    UnifiedVector<std::byte> ws_flat(stedc_flat_workspace_size<B>(trace_ctx, n, batch, JobType::EigenVectors, params));
    stedc_flat<B>(trace_ctx, diag, offdiag, eigvals, ws_flat, JobType::EigenVectors, params, eigvecs);
    trace_ctx.wait();
    batchlas_kernel_trace::flush();

    const auto trace = read_trace_file(trace_path);
    disable_kernel_trace();
    ASSERT_FALSE(trace.empty()) << "Expected a kernel trace file at " << trace_path;
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:leaf_pack\""), 1u);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:leaf_steqr\""), 1u);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:parent_pack\""), expected_merge_depths);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:build_v\""), expected_merge_depths);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:deflation\""), expected_merge_depths);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:merge_fused_cta\""), expected_merge_depths);
    EXPECT_EQ(count_trace_entries(trace, "\"name\":\"stedc_flat:update_q\""), expected_merge_depths);
}

} // namespace

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

    auto tol = std::is_same_v<float_type, double>
        ? std::numeric_limits<float_type>::epsilon() * 1e7
        : std::numeric_limits<float_type>::epsilon() * 1e5;
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
    using float_type = typename base_type<T>::type;
    StedcParams<float_type> params{
        .recursion_threshold = 32,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = 32,
    };
    expect_flat_matches_recursive_case<B, float_type>(
        *this->ctx,
        128,
        16,
        JobType::EigenVectors,
        params,
        std::numeric_limits<float_type>::epsilon() * float_type(5e3),
        true,
        std::numeric_limits<float_type>::epsilon() * float_type(2e4));
}

TYPED_TEST(StedcTest, FlatRaggedMatchesRecursive) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    using float_type = typename base_type<T>::type;

    StedcParams<float_type> params{
        .recursion_threshold = 32,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = 32,
    };

    for (int n : {70, 100, 129}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);
        expect_flat_matches_recursive_case<B, float_type>(
            *this->ctx,
            n,
            8,
            JobType::EigenVectors,
            params,
            std::numeric_limits<float_type>::epsilon() * float_type(5e3),
            true,
            std::numeric_limits<float_type>::epsilon() * float_type(2e4));
    }
}

TYPED_TEST(StedcTest, FlatMatchesRecursiveAcrossMergeVariants) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "Explicit GPU merge variant coverage is GPU-only";
    } else {
        using float_type = typename base_type<T>::type;
        const auto eig_tol = std::numeric_limits<float_type>::epsilon() * float_type(5e3);
        const auto ritz_tol = std::numeric_limits<float_type>::epsilon() * float_type(2e4);
        for (auto variant : {StedcMergeVariant::Baseline, StedcMergeVariant::Fused, StedcMergeVariant::FusedCta}) {
            SCOPED_TRACE(::testing::Message() << "merge_variant=" << static_cast<int>(variant));
            StedcParams<float_type> params{
                .recursion_threshold = 32,
                .merge_variant = variant,
                .enable_rescale = true,
                .secular_threads_per_root = 32,
            };
            expect_flat_matches_recursive_case<B, float_type>(
                *this->ctx,
                128,
                8,
                JobType::EigenVectors,
                params,
                eig_tol,
                true,
                ritz_tol);
        }
    }
}

TYPED_TEST(StedcTest, FlatNoEigenVectorsMatchesRecursiveAcrossMergeVariants) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "Explicit GPU merge variant coverage is GPU-only";
    } else {
        using float_type = typename base_type<T>::type;
        const auto eig_tol = std::numeric_limits<float_type>::epsilon() * float_type(5e3);
        for (auto variant : {StedcMergeVariant::Baseline, StedcMergeVariant::Fused, StedcMergeVariant::FusedCta}) {
            SCOPED_TRACE(::testing::Message() << "merge_variant=" << static_cast<int>(variant));
            StedcParams<float_type> params{
                .recursion_threshold = 32,
                .merge_variant = variant,
                .enable_rescale = true,
                .secular_threads_per_root = 32,
            };
            expect_flat_matches_recursive_case<B, float_type>(
                *this->ctx,
                129,
                8,
                JobType::NoEigenVectors,
                params,
                eig_tol);
        }
    }
}

TYPED_TEST(StedcTest, FlatTraceCollapsesByDepth) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "Flat trace collapse is GPU-only";
    } else {
        bool has32 = false;
        for (auto sgs : (*this->ctx)->get_device().template get_info<sycl::info::device::sub_group_sizes>()) {
            if (static_cast<int32_t>(sgs) == 32) {
                has32 = true;
                break;
            }
        }
        if (!has32) {
            GTEST_SKIP() << "Trace collapse test requires subgroup size 32 for deterministic CTA leaf path";
        }

        using float_type = typename base_type<T>::type;
        StedcParams<float_type> params{
            .recursion_threshold = 32,
            .merge_variant = StedcMergeVariant::FusedCta,
            .enable_rescale = true,
            .secular_threads_per_root = 32,
        };
        const auto schedule = build_flat_schedule(128, params.recursion_threshold);
        expect_flat_trace_counts<B, float_type>(
            *this->ctx,
            128,
            4,
            params,
            schedule.levels.size() - 1,
            "/tmp/batchlas_stedc_flat_trace_balanced.json");
    }
}

TYPED_TEST(StedcTest, FlatTraceCollapsesByDepthRagged) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) {
        GTEST_SKIP() << "Flat trace collapse is GPU-only";
    } else {
        bool has32 = false;
        for (auto sgs : (*this->ctx)->get_device().template get_info<sycl::info::device::sub_group_sizes>()) {
            if (static_cast<int32_t>(sgs) == 32) {
                has32 = true;
                break;
            }
        }
        if (!has32) {
            GTEST_SKIP() << "Trace collapse test requires subgroup size 32 for deterministic CTA leaf path";
        }

        using float_type = typename base_type<T>::type;
        StedcParams<float_type> params{
            .recursion_threshold = 32,
            .merge_variant = StedcMergeVariant::FusedCta,
            .enable_rescale = true,
            .secular_threads_per_root = 32,
        };
        const auto schedule = build_flat_schedule(129, params.recursion_threshold);
        expect_flat_trace_counts<B, float_type>(
            *this->ctx,
            129,
            4,
            params,
            schedule.levels.size() - 1,
            "/tmp/batchlas_stedc_flat_trace_ragged.json");
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

    StedcParams<float_type> params_base{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::Baseline,
    };
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

TYPED_TEST(StedcTest, FusedCtaFallsBackToWgWhenRequestedExceedsMaxSubgroup) {
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
