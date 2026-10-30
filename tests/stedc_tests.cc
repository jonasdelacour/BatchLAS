// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <tests/test_utils.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/verify/tolerance.hh>
#include <tests/eigen_verify.hh>
#include <src/queue.hh>
#include <src/ops/syev/stedc/stedc_levels_plan.hh>
#include <batchlas/util/group-collectives.hh>

// Named rather than inherited transitively: the `info` cases below use M_PI,
// std::cos and std::is_same_v, and nothing else in this file did.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
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

namespace {

// One scan per work-group of 128 (the deflation kernel's width), in place and out of place.
// Returns the number of wrong elements; the first few are reported.
int check_joint_exclusive_scan(Queue& ctx, int n, const char* where) {
    constexpr int kWg = 128;
    constexpr int kBatch = 64;
    constexpr int32_t kInit = 3;
    UnifiedVector<int32_t> in(static_cast<size_t>(n) * kBatch), inplace(in.size()), sep(in.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = static_cast<int32_t>((i * 7919u + 13u) % 11u) - 4;
    for (size_t i = 0; i < in.size(); ++i) inplace[i] = in[i];
    const int32_t* pin = in.data();
    int32_t* pip = inplace.data();
    int32_t* psep = sep.data();
    ctx->parallel_for(sycl::nd_range<1>(kBatch * kWg, kWg), [=](sycl::nd_item<1> it) {
        const auto g = it.get_group();
        const size_t off = g.get_group_linear_id() * static_cast<size_t>(n);
        batchlas::portable::joint_exclusive_scan(g, pip + off, pip + off + n, pip + off, kInit, sycl::plus<int32_t>());
        batchlas::portable::joint_exclusive_scan(g, pin + off, pin + off + n, psep + off, kInit, sycl::plus<int32_t>());
    });
    ctx.wait_and_throw();
    int bad = 0;
    for (int b = 0; b < kBatch; ++b) {
        int32_t acc = kInit;
        for (int i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>(b) * n + i;
            if (bad < 4) {
                EXPECT_EQ(inplace[k], acc) << where << " in place, n=" << n << " group " << b << " index " << i;
                EXPECT_EQ(sep[k], acc) << where << " out of place, n=" << n << " group " << b << " index " << i;
            }
            bad += (inplace[k] != acc) + (sep[k] != acc);
            acc += in[k];
        }
    }
    return bad;
}

void check_joint_exclusive_scan_on(Queue& ctx, const char* where) {
    for (int n : {1, 2, 3, 127, 128, 129, 130, 131, 255, 256, 257, 258, 640, 1025}) {
        check_joint_exclusive_scan(ctx, n, where);
    }
}

} // namespace

// The deflation kernel scans keep/exclude flags IN PLACE over n > its work-group width. AdaptiveCpp
// 25.10's own joint_exclusive_scan writes result[i + 1] in the chunk that reads first[i], so every
// later chunk's first input is overwritten before it is read (stedc lost eigenvalues for merges of
// size > 129), and with two elements it drops `init`. batchlas::portable's version must do neither.
TEST(StedcDeflationScan, PortableJointExclusiveScanInPlaceAcrossChunks) {
    Queue gpu(Device::default_device());
    check_joint_exclusive_scan_on(gpu, "default device");
#if BATCHLAS_HAS_CPU_TARGET
    std::unique_ptr<Queue> cpu;
    try {
        cpu = std::make_unique<Queue>(Device("cpu"));
    } catch (const std::exception& e) {
        std::cout << "[ INFO ] cpu leg not run: " << e.what() << std::endl;
        return;
    }
    check_joint_exclusive_scan_on(*cpu, "cpu device");
#endif
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
    const int n = 512;
    // Large-batch behaviour is covered by the n=64/batch=128 fused-CTA tests
    // below; here `n` is what drives the divide-and-conquer merge depth, and
    // the batch dimension only multiplies the dense reference solve.
    const int batch = 8;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::ones(n, batch);
    auto b = Vector<float_type>::ones(n - 1, batch);
    UnifiedVector<float_type> eigvals(static_cast<size_t>(n) * batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 32};
    const Matrix<float_type> dense = Matrix<float_type>::TriDiagToeplitz(n, float_type(1), float_type(1), float_type(1), batch);

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
    (void)stedc(*this->ctx, a, b, VectorView<float_type>(eigvals, n, batch),
                      ws, JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    test_utils::expect_eigenvalues_match_lapacke<float_type>(dense, eigvals, n);
    test_utils::expect_eigenpairs<float_type>(*this->ctx, dense, eigvects, eigvals, n);
}

TYPED_TEST(StedcTest, BatchedRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    const int n = 1024;
    // This case builds a dense n x n x batch reference (537 MB at batch=128 in
    // float) and checks three items of it on the host. Keep the deep
    // recursion that n=1024 exercises (6 merge levels at recursion_threshold
    // =16); drop the batch multiplicity, which added no coverage. Batched
    // behaviour at large n is still covered by BatchedMatrices above.
    const int batch = 4;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::random(n, batch);
    auto b = Vector<float_type>::random(n - 1, batch);
    UnifiedVector<float_type> eigvals(static_cast<size_t>(n) * batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    StedcParams<float_type> params= {.recursion_threshold = 16};

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));

    Matrix<float_type> dense = Matrix<float_type>::Zeros(n, n, batch);
    dense.fill_tridiag(*this->ctx, b, a, b).wait();
    (void)stedc(*this->ctx, a, b, VectorView<float_type>(eigvals, n, batch),
                      ws, JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    test_utils::expect_eigenvalues_match_lapacke<float_type>(dense, eigvals, n);
    test_utils::expect_eigenpairs<float_type>(*this->ctx, dense, eigvects, eigvals, n);
}

// The deflation scan (merge order >= 512) reads the kept widths from the CALLER's workspace;
// a device-only one must work. A host read of it segfaulted (gesvd n=256's bdsdc, syev n=512).
TYPED_TEST(StedcTest, DeviceWorkspaceAtTheDeflationScanOrder) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int n = 512, batch = 2;
    auto a = Vector<float_type>::random(n, batch);
    auto b = Vector<float_type>::random(n - 1, batch);
    UnifiedVector<float_type> eigvals(static_cast<size_t>(n) * batch);
    auto eigvects = Matrix<float_type>::Identity(n, batch);
    Matrix<float_type> dense = Matrix<float_type>::Zeros(n, n, batch);
    dense.fill_tridiag(*this->ctx, b, a, b).wait();
    const StedcParams<float_type> params{};
    const size_t bytes = stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params);
    std::byte* dev = sycl::malloc_device<std::byte>(bytes, (*this->ctx)->get_device(), (*this->ctx)->get_context());
    ASSERT_NE(dev, nullptr);
    (void)stedc(*this->ctx, a, b, VectorView<float_type>(eigvals, n, batch), Span<std::byte>(dev, bytes),
                JobType::EigenVectors, params, eigvects);
    this->ctx->wait();
    sycl::free(dev, (*this->ctx)->get_context());
    test_utils::expect_eigenvalues_match_lapacke<float_type>(dense, eigvals, n);
    test_utils::expect_eigenpairs<float_type>(*this->ctx, dense, eigvects, eigvals, n);
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
        dense.fill_tridiag(*this->ctx, b, a, b).wait();
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
        UnifiedVector<float_type> eigvals_rec(static_cast<size_t>(n) * batch), eigvals_lvl(eigvals_rec.size());
        auto eigvecs_rec = Matrix<float_type>::Identity(n, batch);
        auto eigvecs_lvl = Matrix<float_type>::Identity(n, batch);

        UnifiedVector<std::byte> ws_rec(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_rec));
        UnifiedVector<std::byte> ws_lvl(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_lvl));

        (void)stedc(*this->ctx, a_rec, b_rec, VectorView<float_type>(eigvals_rec, n, batch), ws_rec,
                    JobType::EigenVectors, params_rec, eigvecs_rec);
        (void)stedc(*this->ctx, a_lvl, b_lvl, VectorView<float_type>(eigvals_lvl, n, batch), ws_lvl,
                    JobType::EigenVectors, params_lvl, eigvecs_lvl);
        this->ctx->wait();

        SCOPED_TRACE("n=" + std::to_string(n));
        test_utils::expect_eigenvalues_agree<float_type>(eigvals_lvl, eigvals_rec, n, batch);
        test_utils::expect_eigenpairs<float_type>(*this->ctx, dense, eigvecs_lvl, eigvals_lvl, n);
    }
}

TYPED_TEST(StedcTest, FusedMergeMatchesBaseline) {
    using T = typename TestFixture::ScalarType;
    const int n = 128;
    const int batch = 64;
    using float_type = typename base_type<T>::type;

    // Generate identical inputs for both paths (stedc mutates its inputs)
    auto a_base = Vector<float_type>::random(n, batch);
    auto b_base = Vector<float_type>::random(n - 1, batch);
    auto a_fused = a_base;
    auto b_fused = b_base;

    UnifiedVector<float_type> eigvals_base(static_cast<size_t>(n) * batch), eigvals_fused(eigvals_base.size());
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

    (void)stedc(*this->ctx, a_base, b_base, VectorView<float_type>(eigvals_base, n, batch), ws_base,
                JobType::EigenVectors, params_base, eigvecs_base);
    (void)stedc(*this->ctx, a_fused, b_fused, VectorView<float_type>(eigvals_fused, n, batch), ws_fused,
                JobType::EigenVectors, params_fused, eigvecs_fused);
    this->ctx->wait();

    test_utils::expect_eigenvalues_agree<float_type>(eigvals_fused, eigvals_base, n, batch);
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
    T_mat.fill_tridiag(*this->ctx, b_cta, a_cta, b_cta).wait();
    this->ctx->wait();

    UnifiedVector<float_type> eigvals_cta(static_cast<size_t>(n) * batch);
    auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params_cta{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = 32,
    };

    UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    (void)stedc(*this->ctx, a_cta, b_cta, VectorView<float_type>(eigvals_cta, n, batch), ws_cta,
                JobType::EigenVectors, params_cta, eigvecs_cta);
    this->ctx->wait();

    const std::vector<int> all_items = T_mat.all_items();
    test_utils::expect_eigenvalues_match_lapacke<float_type>(T_mat, eigvals_cta, n, false, all_items);
    test_utils::expect_eigenpairs<float_type>(*this->ctx, T_mat, eigvecs_cta, eigvals_cta, n, all_items);
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

        // The tuned default (STEDC_THREADS_PER_ROOT_*, 8 in every bucket) is one
        // of these; parts_per_wg = wg_size / P varies across the sweep.
        for (int P : {4, 8, 16, 32}) {
            SCOPED_TRACE("log10cond=" + std::to_string(log10_cond) + " P=" + std::to_string(P));
            auto a_cta = diag;
            auto b_cta = sub;
            UnifiedVector<float_type> W(static_cast<size_t>(n) * batch);
            VectorView<float_type> eigvals(W, n, batch);
            auto eigvecs = Matrix<float_type>::Identity(n, batch);

            StedcParams<float_type> params{
                .recursion_threshold = 16,
                .merge_variant = StedcMergeVariant::FusedCta,
                .enable_rescale = true,
                .secular_threads_per_root = P,
            };

            UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
            (void)stedc(*this->ctx, a_cta, b_cta, eigvals, ws, JobType::EigenVectors, params, eigvecs);
            this->ctx->wait();

            const std::vector<int> all_items = dense_A.all_items();
            test_utils::expect_eigenvalues_match_lapacke<float_type>(dense_A, W, n, false, all_items);
            test_utils::expect_eigenpairs<float_type>(*this->ctx, dense_A, eigvecs, W, n, all_items);
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

    Matrix<float_type> T_mat = Matrix<float_type>::Zeros(n, n, batch);
    T_mat.fill_tridiag(*this->ctx, b_saved, a_saved, b_saved).wait();

    for (int P : {4, 8, 16, 32}) {
        SCOPED_TRACE("secular_threads_per_root=" + std::to_string(P));
        auto a_cta = a_saved;
        auto b_cta = b_saved;
        UnifiedVector<float_type> eigvals_cta(static_cast<size_t>(n) * batch);
        auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

        StedcParams<float_type> params_cta{
            .recursion_threshold = 16,
            .merge_variant = StedcMergeVariant::FusedCta,
            .enable_rescale = true,
            .secular_threads_per_root = P,
        };

        UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
        (void)stedc(*this->ctx, a_cta, b_cta, VectorView<float_type>(eigvals_cta, n, batch), ws_cta,
                    JobType::EigenVectors, params_cta, eigvecs_cta);
        this->ctx->wait();

        const std::vector<int> all_items = T_mat.all_items();
        test_utils::expect_eigenvalues_match_lapacke<float_type>(T_mat, eigvals_cta, n, false, all_items);
        test_utils::expect_eigenpairs<float_type>(*this->ctx, T_mat, eigvecs_cta, eigvals_cta, n, all_items);
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
    T_mat.fill_tridiag(*this->ctx, b_cta, a_cta, b_cta).wait();

    UnifiedVector<float_type> eigvals_cta(static_cast<size_t>(n) * batch);
    auto eigvecs_cta = Matrix<float_type>::Identity(n, batch);

    StedcParams<float_type> params_cta{
        .recursion_threshold = 16,
        .merge_variant = StedcMergeVariant::FusedCta,
        .enable_rescale = true,
        .secular_threads_per_root = forced_threads_per_root,
    };

    UnifiedVector<std::byte> ws_cta(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params_cta));
    (void)stedc(*this->ctx, a_cta, b_cta, VectorView<float_type>(eigvals_cta, n, batch), ws_cta,
                JobType::EigenVectors, params_cta, eigvecs_cta);
    this->ctx->wait();

    const std::vector<int> all_items = T_mat.all_items();
    test_utils::expect_eigenvalues_match_lapacke<float_type>(T_mat, eigvals_cta, n, false, all_items);
    test_utils::expect_eigenpairs<float_type>(*this->ctx, T_mat, eigvecs_cta, eigvals_cta, n, all_items);
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
//     into the caller's span (src/ops/syev/stedc/stedc.cc:922-937; the fold is
//     needed because that one steqr call solves leaves*batch problems, so its
//     batch axis is LONGER than `info`). This is what the forced case below
//     uses, and it is also the failure LAPACK's `?stedc` reports info > 0 for.
//   * `max_sec_iter` is NOT reachable on the arms that run by default:
//     src/ops/syev/stedc/stedc_secular.cc:298 and :521 hardcode `i < 50` and :700
//     hardcodes `iter >= 100`, so the parameter never reaches the loop. Only
//     Fused/FusedCta honour it (stedc_merge_cta.cc:1006). A test that set
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
    // once (the accumulator rule in src/kernels/common/info_span.hh), so a surviving
    // -1 means that clear never ran and every zero in this array would have been
    // an accident of initialisation.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
    (void)stedc(*this->ctx, d, e, eigvals, ws, JobType::EigenVectors, params,
          eigvects, info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: the span was "
                                  "never written, so a zero here would prove nothing";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a Toeplitz batch "
                                 "that every other case in this file solves";
    }

    // Reported converged AND correct. The closed form for Toeplitz(1, 1, 1) is
    // 1 + 2*cos(k*pi/(n+1)), ascending in k measured from the far end.
    std::vector<double> spectrum(n);
    for (int i = 0; i < n; ++i) spectrum[i] = 1.0 + 2.0 * std::cos(M_PI * double(n - i) / double(n + 1));
    const std::vector<std::vector<double>> ref(batch, spectrum);
    EXPECT_VERIFY(float_type, verify::Check::values, n,
                  MatrixView<float_type>::values_error(eigvals, ref, 3.0, eigvals.all_items()));
}

// THE CASE THAT MATTERS: the same batch, one sweep allowed in the leaf solves.
//
// A test that only ever observes info == 0 cannot distinguish a working
// implementation from one that memsets the span to zero, so this forces the
// failure on the SAME input the case above solves cleanly. `max_sweeps = 1`
// rather than a "reduced" 50 because 50 is the default, and the CTA tiers rewrite
// the default to 400 (syev_cta.cc:178-181) -- a habit worth not relying on the
// absence of.
//
// It also exercises the part of the plumbing that is unique to stedc: the leaf
// axis is not the batch axis. One steqr call solves 2^L * batch_size leaf
// problems, and each leaf's status has to be folded back onto the batch item it
// belongs to. A fold that got the divisor wrong would report failures against
// the wrong items -- or, with batch_size = 1, look perfect while being wrong, so
// this case uses a batch of 4.
// Two 32-wide diagonal leaves of equal value coupled by one off-diagonal: the two nonzero z
// components sit on equal poles, so deflation leaves a one-pole secular problem (dd == 1), which
// the fused merges solve in closed form. That arm must report convergence; it reported failure
// (heavy deflation reaches dd == 1 on the syev tuner's graded jobz=V n=448 cells). P = 8 is the
// partition path, P = 64 the work-group path. Spectrum {1 - c, 1 (n - 2 times), 1 + c}.
TYPED_TEST(StedcTest, InfoIsZeroWhenDeflationLeavesOnePole) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "the fused merges are GPU-only"; }
    using float_type = typename base_type<T>::type;
    const int n = 64;
    const int batch = 8;
    const double c = 0.5;

    for (int P : {8, 64}) {
        SCOPED_TRACE("secular_threads_per_root=" + std::to_string(P));
        auto d = Vector<float_type>::ones(n, batch);
        auto e = Vector<float_type>::zeros(n - 1, batch);
        for (int b = 0; b < batch; ++b) e(n / 2 - 1, b) = float_type(c);
        auto eigvals = Vector<float_type>::zeros(n, batch);
        auto eigvects = Matrix<float_type>::Identity(n, batch);
        StedcParams<float_type> params{
            .recursion_threshold = n / 2,
            .merge_variant = StedcMergeVariant::FusedCta,
            .secular_threads_per_root = P,
        };
        UnifiedVector<int32_t> info(batch, int32_t(-1));
        UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
        (void)stedc(*this->ctx, d, e, eigvals, ws, JobType::EigenVectors, params,
                    eigvects, info.to_span());
        this->ctx->wait();

        for (int b = 0; b < batch; ++b) EXPECT_EQ(info[b], 0) << "item " << b;
        std::vector<double> spectrum(n, 1.0);
        spectrum.front() = 1.0 - c;
        spectrum.back() = 1.0 + c;
        const std::vector<std::vector<double>> ref(batch, spectrum);
        std::vector<int> items(batch);
        for (int b = 0; b < batch; ++b) items[b] = b;
        EXPECT_VERIFY(float_type, verify::Check::values, n,
                      MatrixView<float_type>::values_error(eigvals, ref, 1.0 + c, items));
    }
}

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
    (void)stedc(*this->ctx, d, e, eigvals, ws, JobType::EigenVectors, params,
          eigvects, info.to_span());
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

    (void)stedc(*this->ctx, d0, e0, w0, ws0, JobType::EigenVectors, params,
          z0, info.to_span());
    (void)stedc(*this->ctx, d1, e1, w1, ws1, JobType::EigenVectors, params,
          z1, Span<int32_t>{});
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(w0(i, b), w1(i, b))
                << "requesting status changed the answer at batch " << b << " index " << i;
        }
    }
}
// The Fused merge at n = 64 in float, one merge of two 32-leaves, at two work-group widths: the
// Loewner product of up to 64 ratios (d_i - lambda_j) / (d_i - d_j) is carried as mantissa and
// exponent (maybe_rescale_vectors), as in the CTA merges. eigen_residual and orthogonality on the
// dense tridiagonal.
TYPED_TEST(StedcTest, StedcFusedN64FloatNoOverflow) {
    using Real = typename base_type<typename TestFixture::ScalarType>::type;
    using MV = MatrixView<Real, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "the fused merge is a GPU kernel"; }
    const int n = 64, batch = 32;
    for (const Real log10_cond : {Real(1), Real(5)}) {
        auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, Real>(*this->ctx, n, log10_cond, NormType::Spectral, batch, 1234u);
        this->ctx->wait();
        Vector<Real> diag(n, Real(0), batch), sub(n - 1, Real(0), batch);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i) {
                diag(i, b) = dense_A.at(i, i, b);
                if (i < n - 1) sub(i, b) = dense_A.at(i + 1, i, b);
            }
        for (const int threads : {8, 128}) {
            SCOPED_TRACE(::testing::Message() << "log10_cond " << log10_cond << ", merge_threads " << threads);
            auto d = diag;
            auto e = sub;
            auto w = Vector<Real>::zeros(n, batch);
            auto Q = Matrix<Real>::Identity(n, batch);
            StedcParams<Real> params{.recursion_threshold = 32, .merge_variant = StedcMergeVariant::Fused, .merge_threads = threads, .enable_rescale = true};
            UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
            (void)stedc(*this->ctx, d, e, w, ws, JobType::EigenVectors, params, Q);
            this->ctx->wait();
            EXPECT_VERIFY(Real, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, dense_A, Q, w, dense_A.all_items()));
            EXPECT_VERIFY(Real, verify::Check::orthogonality, n, MV::orthogonality(*this->ctx, Q, Q.all_items()));
        }
    }
}

// The Fused merge at n = 512 (top merge dd up to 512) with 128- and 256-wide work-groups: every
// lane's Loewner mantissa is in [0.5, 1), so one group-wide product of them leaves float range
// (about 2^-0.5dd). maybe_rescale_vectors must renormalize the reduction in rounds.
TYPED_TEST(StedcTest, StedcFusedWideWorkgroupRescale) {
    using Real = typename base_type<typename TestFixture::ScalarType>::type;
    using MV = MatrixView<Real, MatrixFormat::Dense>;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) { GTEST_SKIP() << "the fused merge is a GPU kernel"; }
    const int n = 512, batch = 4;
    auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, Real>(*this->ctx, n, Real(1), NormType::Spectral, batch, 4321u);
    this->ctx->wait();
    Vector<Real> diag(n, Real(0), batch), sub(n - 1, Real(0), batch);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) {
            diag(i, b) = dense_A.at(i, i, b);
            if (i < n - 1) sub(i, b) = dense_A.at(i + 1, i, b);
        }
    for (const int threads : {128, 256}) {
        SCOPED_TRACE(::testing::Message() << "merge_threads " << threads);
        auto d = diag;
        auto e = sub;
        auto w = Vector<Real>::zeros(n, batch);
        auto Q = Matrix<Real>::Identity(n, batch);
        StedcParams<Real> params{.recursion_threshold = 32, .merge_variant = StedcMergeVariant::Fused, .merge_threads = threads, .enable_rescale = true};
        UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::EigenVectors, params));
        (void)stedc(*this->ctx, d, e, w, ws, JobType::EigenVectors, params, Q);
        this->ctx->wait();
        EXPECT_VERIFY(Real, verify::Check::eigen_residual, n, MV::eigen_residual(*this->ctx, dense_A, Q, w, dense_A.all_items()));
        EXPECT_VERIFY(Real, verify::Check::orthogonality, n, MV::orthogonality(*this->ctx, Q, Q.all_items()));
    }
}

// Values only (jobz = NoEigenVectors). Every merge builds its secular vector from the last row of
// the left child's eigenvectors and the first row of the right child's (stedc_merge_step), so a
// driver that skips the leaf vectors under NoEigenVectors merges from zeros: every z component
// deflates and the result is the sorted torn diagonal, finite and plausible. The workspace is NaN
// poison and the eigenvector matrix a large finite one, so nothing upstream hides a missed write.
// Four different items per batch: graded (cond ~1e12), glued Wilkinson clusters, alternating-sign
// off-diagonals (rho < 0 at every split) and random. n = 180 and 600 recurse three and four times
// at recursion_threshold 32 and pad on the level driver; 256 is unpadded.
namespace {
template <typename Real>
void stedc_values_only_items(Vector<Real>& d, Vector<Real>& e, int n) {
    for (int i = 0; i < n; ++i) {
        const double t = n > 1 ? double(i) / double(n - 1) : 0.0;
        d(i, 0) = Real(std::pow(10.0, -6.0 + 12.0 * t));
        d(i, 1) = Real(std::fabs(10.0 - double(i % 21)));
        d(i, 2) = Real(2);
        d(i, 3) = Real(std::sin(1.7 * i + 0.3));
    }
    for (int i = 0; i + 1 < n; ++i) {
        e(i, 0) = Real(0.25 * std::sqrt(double(d(i, 0)) * double(d(i + 1, 0))));
        e(i, 1) = Real((i % 21) == 20 ? 1e-5 : 1.0);
        e(i, 2) = Real((i % 2) ? -1 : 1);
        e(i, 3) = Real(std::cos(0.9 * i + 1.1));
    }
}
} // namespace

TYPED_TEST(StedcTest, NoEigenVectorsMatchesReferenceOnPoisonedWorkspace) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int batch = 4;
    const std::vector<int> items = {0, 1, 2, 3};
    for (StedcAlgorithm algorithm : {StedcAlgorithm::Recursive, StedcAlgorithm::Auto}) {
        for (int n : {180, 256, 600}) {
            SCOPED_TRACE(::testing::Message() << "n " << n << " algorithm " << int(algorithm));
            auto d = Vector<float_type>::zeros(n, batch);
            auto e = Vector<float_type>::zeros(n - 1, batch);
            stedc_values_only_items(d, e, n);
            Matrix<float_type> dense = Matrix<float_type>::Zeros(n, n, batch);
            dense.fill_tridiag(*this->ctx, e, d, e).wait();

            StedcParams<float_type> params{.recursion_threshold = 32, .algorithm = algorithm};
            UnifiedVector<float_type> eigvals(static_cast<size_t>(n) * batch, float_type(1e30));
            auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
            eigvects.fill(*this->ctx, float_type(1e30)).wait();
            UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::NoEigenVectors, params),
                                        std::byte{0xFF});
            UnifiedVector<int32_t> info(batch, int32_t(-1));
            (void)stedc(*this->ctx, d, e, VectorView<float_type>(eigvals, n, batch), ws, JobType::NoEigenVectors,
                        params, eigvects, info.to_span());
            this->ctx->wait();

            for (int b = 0; b < batch; ++b) EXPECT_EQ(info[b], 0) << "item " << b;
            test_utils::expect_eigenvalues_match_lapacke<float_type>(dense, eigvals, n, false, items);
            // Values only leaves the caller's matrix alone, as LAPACK does with COMPZ = 'N'.
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(eigvects(n - 1, 0, b), float_type(1e30)) << "item " << b;
                EXPECT_EQ(eigvects(0, n - 1, b), float_type(1e30)) << "item " << b;
            }
        }
    }
}

// The same values-only call with no eigenvector matrix at all, the natural values-only spelling:
// a driver that still needs one as working storage reads through a null view.
TYPED_TEST(StedcTest, NoEigenVectorsWithoutAnEigenvectorMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    const int batch = 4;
    const std::vector<int> items = {0, 1, 2, 3};
    for (StedcAlgorithm algorithm : {StedcAlgorithm::Recursive, StedcAlgorithm::Auto}) {
        for (int n : {180, 256}) {
            SCOPED_TRACE(::testing::Message() << "n " << n << " algorithm " << int(algorithm));
            auto d = Vector<float_type>::zeros(n, batch);
            auto e = Vector<float_type>::zeros(n - 1, batch);
            stedc_values_only_items(d, e, n);
            Matrix<float_type> dense = Matrix<float_type>::Zeros(n, n, batch);
            dense.fill_tridiag(*this->ctx, e, d, e).wait();

            StedcParams<float_type> params{.recursion_threshold = 32, .algorithm = algorithm};
            UnifiedVector<float_type> eigvals(static_cast<size_t>(n) * batch, float_type(1e30));
            UnifiedVector<std::byte> ws(stedc_buffer_size(*this->ctx, n, batch, JobType::NoEigenVectors, params),
                                        std::byte{0xFF});
            (void)stedc(*this->ctx, d, e, VectorView<float_type>(eigvals, n, batch), ws, JobType::NoEigenVectors,
                        params, MatrixView<float_type, MatrixFormat::Dense>());
            this->ctx->wait();

            test_utils::expect_eigenvalues_match_lapacke<float_type>(dense, eigvals, n, false, items);
        }
    }
}
