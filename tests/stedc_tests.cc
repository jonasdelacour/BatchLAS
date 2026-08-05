#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include "test_utils.hh"
#include "../src/queue.hh"
#include "../src/extensions/stedc_levels_plan.hh"

using namespace batchlas;

namespace {

} // namespace

// ---------------------------------------------------------------------------
// Level-plan shape. Host-only, no device: a bad leaf produces perfectly correct
// eigenvalues and only shows up as lost throughput, so the numerical tests
// below cannot catch it and these have to assert on the plan itself.
// ---------------------------------------------------------------------------

// The invariant that matters. `steqr` dispatches to the fast `steqr_cta` only
// for n <= the device sub-group width, and the tuned STEDC threshold *is* that
// width -- so a leaf above it silently falls back to `steqr_wg`, which measured
// ~14x slower one step over the edge (n=32: 0.26us -> n=36: 3.76us).
TEST(StedcLevelPlan, LeafNeverExceedsThreshold) {
    for (int64_t threshold : {8, 16, 32, 64}) {
        for (int64_t n = 2; n <= 4096; ++n) {
            const auto plan = plan_stedc_levels(n, threshold);
            if (plan.levels == 0) continue;  // caller falls back to the recursive driver
            EXPECT_LE(plan.leaf, threshold)
                << "leaf above the steqr_cta cap, n=" << n << " threshold=" << threshold;
            EXPECT_EQ(plan.padded_n, plan.leaf << plan.levels)
                << "padded_n inconsistent with the tree, n=" << n;
            EXPECT_GE(plan.padded_n, n)
                << "plan drops part of the problem, n=" << n;
        }
    }
}

// The two sizes PR #55 regressed. Both admit an exactly-fitting tree at leaf 40
// and at leaf 20; the scoring used to prefer 40 because it sits nearer the
// threshold, which drove the leaf solve off the steqr_cta cliff (syev n=320:
// 74.5 -> 242.4 us/matrix, n=640: 727.5 -> 1095.9).
TEST(StedcLevelPlan, NonPowerOfTwoPicksNarrowLeaf) {
    const auto p320 = plan_stedc_levels(320, 32);
    EXPECT_EQ(p320.leaf, 20);
    EXPECT_EQ(p320.levels, 4);
    EXPECT_EQ(p320.padded_n, 320) << "n=320 should still need no padding";

    const auto p640 = plan_stedc_levels(640, 32);
    EXPECT_EQ(p640.leaf, 20);
    EXPECT_EQ(p640.levels, 5);
    EXPECT_EQ(p640.padded_n, 640) << "n=640 should still need no padding";
}

// The power-of-two sizes must be unchanged: leaf lands exactly on the threshold
// with no padding. This is the regime the level driver was tuned and measured
// in, and the cap above must not perturb it.
TEST(StedcLevelPlan, PowerOfTwoIsUnchanged) {
    for (int64_t n : {64, 128, 256, 512, 1024, 2048}) {
        const auto plan = plan_stedc_levels(n, 32);
        EXPECT_EQ(plan.leaf, 32) << "n=" << n;
        EXPECT_EQ(plan.padded_n, n) << "n=" << n << " should need no padding";
    }
}

template <typename T, Backend B>
struct StedcConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// Every STEDC test body operates purely on `base_type<T>::type` (the real
// scalar), so the complex instantiations re-ran the real ones bit-for-bit.
// Drop them: they doubled the file's runtime for zero extra coverage.
using StedcTestTypes = typename test_utils::backend_types_filtered<StedcConfig, false>::type;

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
    // Large-batch behaviour is covered by the n=64/batch=128 fused-CTA tests
    // below; here `n` is what drives the divide-and-conquer merge depth, and
    // the batch dimension only multiplies the dense reference solve.
    const int batch = 8;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::ones(n, batch);
    auto b = Vector<float_type>::ones(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 32};

    UnifiedVector<std::byte> ws(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params));

    stedc(*this->ctx, a.view(), b.view(), eigvals.view(),
                      ws, JobType::EigenVectors, params, eigvects.view());
    
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);

    Matrix<float_type> reconstructed = Matrix<float_type>::TriDiagToeplitz(n, float_type(1), float_type(1), float_type(1), batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size(*(this->ctx), reconstructed.view(), ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));

    auto ritz_vals = ritz_values<B, float_type>(*this->ctx, reconstructed, eigvects);
    syev(*(this->ctx), reconstructed.view(), ref_eigvals, {.jobz = JobType::NoEigenVectors}, syev_ws);
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
    // This case builds a dense n x n x batch reference (537 MB at batch=128 in
    // float) and runs a full SYEV over it -- on the host for the NETLIB
    // instantiations, where it dominates this whole binary. Keep the deep
    // recursion that n=1024 exercises (6 merge levels at recursion_threshold
    // =16); drop the batch multiplicity, which added no coverage. Batched
    // behaviour at large n is still covered by BatchedMatrices above.
    const int batch = 4;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::random(n, batch);
    auto b = Vector<float_type>::random(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 16};

    UnifiedVector<std::byte> ws(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params));

    Matrix<float_type> reconstructed = Matrix<float_type>::Zeros(n, n, batch);
    reconstructed.view().fill_tridiag(*this->ctx, b, a, b).wait();
    this->ctx->wait();
    
    stedc(*this->ctx, a.view(), b.view(), eigvals.view(),
                      ws, JobType::EigenVectors, params, eigvects.view());
    
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);

    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size(*(this->ctx), reconstructed.view(), ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));

    auto ritz_vals = ritz_values<B, float_type>(*this->ctx, reconstructed, eigvects);
    syev(*(this->ctx), reconstructed.view(), ref_eigvals, {.jobz = JobType::NoEigenVectors}, syev_ws);
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

// The level-synchronous driver is the default; this pins it against the
// recursive one it replaced. n = 128 needs no padding (32 * 2^2 = 128), n = 100
// and n = 129 both do, which is where the padded diagonal tail and the
// leading-block extraction get exercised.
TYPED_TEST(StedcTest, LevelsMatchesRecursive) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "Level driver is GPU-only"; }
    using float_type = typename base_type<T>::type;
    const int batch = 8;

    // 320 and 640 are the sizes whose tree shape the leaf cap changes; 100 and
    // 129 cover padding and an odd n.
    for (int n : {128, 100, 129, 320, 640}) {
        auto a = Vector<float_type>::random(n, batch);
        auto b = Vector<float_type>::random(n - 1, batch);

        Matrix<float_type> dense = Matrix<float_type>::Zeros(n, n, batch);
        dense.view().fill_tridiag(*this->ctx, b, a, b).wait();
        this->ctx->wait();

        // merge_variant is pinned rather than left on Auto so the driver is the
        // only variable between the two arms. (It was pinned off Fused's
        // sibling because FusedCta used to deadlock; that is fixed, and Auto
        // resolves back to FusedCta -- but pinning is still the right thing for
        // an A/B of the drivers.)
        StedcParams<float_type> params_rec{
            .recursion_threshold = 32,
            .algorithm = StedcAlgorithm::Recursive,
            .merge_variant = StedcMergeVariant::Fused,
        };
        StedcParams<float_type> params_lvl{
            .recursion_threshold = 32,
            .algorithm = StedcAlgorithm::Levels,
            .merge_variant = StedcMergeVariant::Fused,
        };

        auto a_rec = a; auto b_rec = b;
        auto a_lvl = a; auto b_lvl = b;
        auto eigvals_rec = Vector<float_type>::zeros(n, batch);
        auto eigvals_lvl = Vector<float_type>::zeros(n, batch);
        auto eigvecs_rec = Matrix<float_type>::Identity(n, batch);
        auto eigvecs_lvl = Matrix<float_type>::Identity(n, batch);

        UnifiedVector<std::byte> ws_rec(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_rec));
        UnifiedVector<std::byte> ws_lvl(stedc_workspace_size<B>(*this->ctx, n, batch, JobType::EigenVectors, params_lvl));

        stedc<B>(*this->ctx, a_rec, b_rec, eigvals_rec, ws_rec, JobType::EigenVectors, params_rec, eigvecs_rec);
        stedc<B>(*this->ctx, a_lvl, b_lvl, eigvals_lvl, ws_lvl, JobType::EigenVectors, params_lvl, eigvecs_lvl);
        this->ctx->wait();

        const auto tol = std::numeric_limits<float_type>::epsilon() * float_type(5e3)
                       * std::max(float_type(1), std::abs(eigvals_rec(n - 1, 0)));
        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                const float_type diff = std::abs(eigvals_rec(i, j) - eigvals_lvl(i, j));
                ASSERT_LE(diff, tol) << "n=" << n << " eigenvalue mismatch at (" << i << ", batch " << j
                                     << "): recursive=" << eigvals_rec(i, j) << " levels=" << eigvals_lvl(i, j);
            }
        }

        // Eigenvectors are only defined up to sign, so compare Ritz values
        // rather than the columns themselves.
        auto ritz = ritz_values<B, float_type>(*this->ctx, dense, eigvecs_lvl);
        this->ctx->wait();
        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                const float_type diff = std::abs(ritz(i, j) - eigvals_lvl(i, j));
                ASSERT_LE(diff, tol) << "n=" << n << " Ritz mismatch at (" << i << ", batch " << j
                                     << "): ritz=" << ritz(i, j) << " eig=" << eigvals_lvl(i, j);
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

    StedcParams<float_type> params_base{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::Baseline,
    };
    StedcParams<float_type> params_fused{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::Fused,
        .enable_rescale = true,
    };

    UnifiedVector<std::byte> ws_base(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params_base));
    UnifiedVector<std::byte> ws_fused(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params_fused));

    stedc(*this->ctx, a_base.view(), b_base.view(), eigvals_base.view(), ws_base, JobType::EigenVectors, params_base, eigvecs_base.view());
    stedc(*this->ctx, a_fused.view(), b_fused.view(), eigvals_fused.view(), ws_fused, JobType::EigenVectors, params_fused, eigvecs_fused.view());
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

    UnifiedVector<std::byte> ws_cta(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    stedc(*this->ctx, a_cta.view(), b_cta.view(), eigvals_cta.view(), ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta.view());
    this->ctx->wait();

    // syev reference eigenvalues
    UnifiedVector<float_type> ref_eigvals(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size(*(this->ctx), T_mat.view(), ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev(*(this->ctx), T_mat.view(), ref_eigvals, {.jobz = JobType::NoEigenVectors}, syev_ws);
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

// Regression: a merge subproblem of size dd == 1 used to make the extremal-root
// secular solver index d_prob(dd - 2) == d_prob(-1). Because d_prob aliases the
// shared-memory d_local through a generic pointer, that negative offset faults
// with CUDA_ERROR_ILLEGAL_ADDRESS rather than reading harmless garbage.
//
// Plain random tridiagonals essentially never deflate that far, which is why
// FusedCtaPartitionWidths below did not catch it. Conditioned matrices from
// random_hermitian_tridiagonal_with_log10_cond_metric do -- this mirrors the
// benchmarks/stedc_acc case that originally exposed the bug.
TYPED_TEST(StedcTest, FusedCtaConditionedHeavyDeflation) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "CTA merge is GPU-only"; }
    using float_type = typename base_type<T>::type;

    const int n = 64;
    const int batch = 64;

    for (float_type log10_cond : {float_type(1), float_type(3), float_type(5)}) {
        auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, float_type>(
            *this->ctx, n, log10_cond, NormType::Spectral, batch, 1234u);
        this->ctx->wait();

        Vector<float_type> diag(n, float_type(0), batch);
        Vector<float_type> sub(n - 1, float_type(0), batch);
        auto A_view = dense_A.view();
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                diag(i, b) = A_view.at(i, i, b);
                if (i < n - 1) sub(i, b) = A_view.at(i + 1, i, b);
            }
        }

        // secular_threads_per_root = 4 is what the tuning tables select for
        // n <= 64, giving parts_per_wg = 8.
        for (int P : {4, 8, 16, 32}) {
            auto a_cta = diag;
            auto b_cta = sub;
            auto eigvals = Vector<float_type>::zeros(n, batch);
            auto eigvecs = Matrix<float_type>::Identity(n, batch);

            StedcParams<float_type> params{
                .recursion_threshold = 16,
                .merge_variant = StedcMergeVariant::FusedCta,
                .enable_rescale = true,
                .secular_threads_per_root = P,
            };

            UnifiedVector<std::byte> ws(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params));
            stedc(*this->ctx, a_cta.view(), b_cta.view(), eigvals.view(), ws, JobType::EigenVectors, params, eigvecs.view());
            this->ctx->wait();

            for (int j = 0; j < batch; ++j) {
                for (int i = 0; i < n; ++i) {
                    const float_type got = eigvals(i, j);
                    ASSERT_EQ(got, got) << "NaN eigenvalue, log10cond=" << log10_cond
                                        << " P=" << P << " at (" << i << ", batch " << j << ")";
                    ASSERT_TRUE(std::isfinite(got)) << "non-finite eigenvalue, log10cond=" << log10_cond
                                                    << " P=" << P << " at (" << i << ", batch " << j << ")";
                }
                for (int i = 0; i + 1 < n; ++i) {
                    ASSERT_LE(eigvals(i, j), eigvals(i + 1, j))
                        << "eigenvalues not sorted, log10cond=" << log10_cond << " P=" << P
                        << " at (" << i << ", batch " << j << ")";
                }
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
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size(*(this->ctx), T_mat.view(), ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev(*(this->ctx), T_mat.view(), ref_eigvals, {.jobz = JobType::NoEigenVectors}, syev_ws);
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

        UnifiedVector<std::byte> ws_cta(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
        stedc(*this->ctx, a_cta.view(), b_cta.view(), eigvals_cta.view(), ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta.view());
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

    UnifiedVector<std::byte> ws_cta(stedc_workspace_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    stedc(*this->ctx, a_cta.view(), b_cta.view(), eigvals_cta.view(), ws_cta, JobType::EigenVectors, params_cta, eigvecs_cta.view());
    this->ctx->wait();

    UnifiedVector<float_type> ref_eigvals(n * batch);
    auto syev_ws = UnifiedVector<std::byte>(syev_buffer_size(*(this->ctx), T_mat.view(), ref_eigvals, JobType::NoEigenVectors, Uplo::Lower));
    syev(*(this->ctx), T_mat.view(), ref_eigvals, {.jobz = JobType::NoEigenVectors}, syev_ws);
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
