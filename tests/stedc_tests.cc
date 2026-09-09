#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include "test_utils.hh"
#include "../src/queue.hh"
#include "../src/extensions/stedc_levels_plan.hh"

// Named rather than inherited transitively: the `info` cases below use M_PI,
// std::cos and std::is_same_v, and nothing else in this file did.
#include <cmath>
#include <cstdint>
#include <type_traits>

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

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));

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

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));

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

        UnifiedVector<std::byte> ws_rec(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_rec));
        UnifiedVector<std::byte> ws_lvl(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_lvl));

        stedc(*this->ctx, a_rec.view(), b_rec.view(), eigvals_rec.view(), ws_rec, JobType::EigenVectors, params_rec, eigvecs_rec.view());
        stedc(*this->ctx, a_lvl.view(), b_lvl.view(), eigvals_lvl.view(), ws_lvl, JobType::EigenVectors, params_lvl, eigvecs_lvl.view());
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

    UnifiedVector<std::byte> ws_base(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_base));
    UnifiedVector<std::byte> ws_fused(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_fused));

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

    UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
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

            UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
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

        UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
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

    UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
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


// ---------------------------------------------------------------------------
// Per-item convergence status (`info`).
//
// stedc is one of the five routines LAPACK gives an info > 0 to, and the one
// where the status has the furthest to travel: a divide-and-conquer solve is a
// tree of leaf steqr calls and secular-equation merges, and either level can
// fail to converge on one batch item while every other item is fine.
//
// WHICH KNOB FORCES A FAILURE, AND WHICH ONLY LOOKS LIKE IT DOES.
//
// stedc has two caps and they are NOT equally reachable, which is the trap in
// writing the forced case:
//
//   * `leaf_steqr_params.max_sweeps` is real. It caps the leaf steqr, and the
//     level-synchronous driver -- the default -- folds each leaf's status back
//     into the caller's span (src/extensions/stedc.cc:921-936; the fold is
//     needed because that one steqr call solves leaves*batch problems, so its
//     batch axis is LONGER than `info`). This is what the forced case below
//     uses, and it is also the failure LAPACK's `?stedc` reports info > 0 for.
//   * `max_sec_iter` is NOT reachable on the arms that run by default:
//     src/extensions/stedc_secular.cc:297 and :521 hardcode `i < 50` and :700
//     hardcodes `iter >= 100`, so the parameter never reaches the loop. Only
//     Fused/FusedCta honour it (stedc_merge_cta.cc:1005). A test that set
//     max_sec_iter, ran the default arm and watched nothing fail would look
//     exactly like a working guard while testing a value that never left the
//     parameter struct -- so it is not written here.
// ---------------------------------------------------------------------------

TYPED_TEST(StedcTest, InfoIsZeroOnAConvergingBatch) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int n = 128;
    const int batch = 4;

    // The same Toeplitz(1, 1, 1) the other cases in this file solve, at a size
    // that still forces a real merge tree (recursion_threshold = 32 gives two
    // levels) without paying for the dense reference solve they do.
    auto d = Vector<float_type>::ones(n, batch);
    auto e = Vector<float_type>::ones(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params = {.recursion_threshold = 32};

    // -1, NOT 0: a span left at zero cannot tell "the solver wrote 0" from
    // "nothing wrote it at all". The entry point is required to clear the span
    // once (the accumulator rule in src/extensions/info_span.hh), so a surviving
    // -1 means that clear never ran and every zero in this array would have been
    // an accident of initialisation.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
    stedc(*this->ctx, d.view(), e.view(), eigvals.view(), ws, JobType::EigenVectors, params,
          eigvects.view(), info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: the span was "
                                  "never written, so a zero here would prove nothing";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a Toeplitz batch "
                                 "that every other case in this file solves";
    }

    // Reported converged AND correct. The closed form for Toeplitz(1, 1, 1) is
    // 1 + 2*cos(k*pi/(n+1)), ascending in k measured from the far end.
    const double tol = std::is_same_v<float_type, float> ? 2e-3 : 1e-8;
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        for (int i = 0; i < n; ++i) {
            const double expected = 1.0 + 2.0 * std::cos(M_PI * double(n - i) / double(n + 1));
            EXPECT_NEAR(static_cast<double>(eigvals(i, b)), expected, tol)
                << "batch " << b << " eigenvalue " << i;
        }
    }
}

// THE CASE THAT MATTERS: the same batch, one sweep allowed in the leaf solves.
//
// A test that only ever observes info == 0 cannot distinguish a working
// implementation from one that memsets the span to zero, so this forces the
// failure on the SAME input the case above solves cleanly. `max_sweeps = 1`
// rather than a "reduced" 50 because 50 is the default, and the CTA tiers rewrite
// the default to 400 (syev_cta.cc:177-180) -- a habit worth not relying on the
// absence of.
//
// It also exercises the part of the plumbing that is unique to stedc: the leaf
// axis is not the batch axis. One steqr call solves 2^L * batch_size leaf
// problems, and each leaf's status has to be folded back onto the batch item it
// belongs to. A fold that got the divisor wrong would report failures against
// the wrong items -- or, with batch_size = 1, look perfect while being wrong, so
// this case uses a batch of 4.
TYPED_TEST(StedcTest, InfoReportsLeafSolvesThatExhaustTheirSweepBudget) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int n = 128;
    const int batch = 4;

    auto d = Vector<float_type>::ones(n, batch);
    auto e = Vector<float_type>::ones(n - 1, batch);
    auto eigvals = Vector<float_type>::zeros(n, batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params = {.recursion_threshold = 32};
    params.leaf_steqr_params.max_sweeps = 1;

    UnifiedVector<int32_t> info(batch, int32_t(-1));

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
    stedc(*this->ctx, d.view(), e.view(), eigvals.view(), ws, JobType::EigenVectors, params,
          eigvects.view(), info.to_span());
    this->ctx->wait();

    int reported = 0;
    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value";
        ASSERT_GE(info[b], 0) << "info is LAPACK-like: 0 or a positive count, never negative";
        if (info[b] != 0) ++reported;
    }
    EXPECT_GT(reported, 0)
        << "a one-sweep leaf budget on a 128-wide Toeplitz batch reported universal "
           "convergence; either the leaf status is not folded into the caller's span, or it "
           "is written unconditionally zero";
}

// The empty-span half of the contract: "not requested" must cost nothing and
// change nothing. stedc_buffer_size takes no `info` argument, so the size cannot
// depend on it by construction; what is checked here is that the ANSWER does not
// either -- i.e. that the status path is a write to caller memory and not an
// extra pool draw that shifts every later allocation.
TYPED_TEST(StedcTest, EmptyInfoSpanChangesNeitherAnswerNorWorkspace) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int n = 128;
    const int batch = 4;

    StedcParams<float_type> params = {.recursion_threshold = 32};
    const size_t bytes_a = stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params);
    const size_t bytes_b = stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params);
    EXPECT_EQ(bytes_a, bytes_b);

    auto d0 = Vector<float_type>::ones(n, batch);
    auto e0 = Vector<float_type>::ones(n - 1, batch);
    auto w0 = Vector<float_type>::zeros(n, batch);
    auto z0 = Matrix<float_type>::Identity(n, batch);
    auto d1 = Vector<float_type>::ones(n, batch);
    auto e1 = Vector<float_type>::ones(n - 1, batch);
    auto w1 = Vector<float_type>::zeros(n, batch);
    auto z1 = Matrix<float_type>::Identity(n, batch);

    UnifiedVector<std::byte> ws0(bytes_a);
    UnifiedVector<std::byte> ws1(bytes_a);
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    stedc(*this->ctx, d0.view(), e0.view(), w0.view(), ws0, JobType::EigenVectors, params,
          z0.view(), info.to_span());
    stedc(*this->ctx, d1.view(), e1.view(), w1.view(), ws1, JobType::EigenVectors, params,
          z1.view(), Span<int32_t>{});
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
