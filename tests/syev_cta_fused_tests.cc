// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <tests/eigen_verify.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

template <typename T, Backend B>
struct SyevCtaFusedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevCtaFusedTestTypes = ::testing::Types<
    SyevCtaFusedConfig<float, Backend::CUDA>,
    SyevCtaFusedConfig<double, Backend::CUDA>,
    SyevCtaFusedConfig<std::complex<float>, Backend::CUDA>,
    SyevCtaFusedConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SyevCtaFusedTestTypes = ::testing::Types<
    SyevCtaFusedConfig<float, Backend::ROCM>,
    SyevCtaFusedConfig<double, Backend::ROCM>,
    SyevCtaFusedConfig<std::complex<float>, Backend::ROCM>,
    SyevCtaFusedConfig<std::complex<double>, Backend::ROCM>>;
#elif BATCHLAS_HAS_SYCL_BACKEND
// AdaptiveCpp Metal: the native kernels in float and complex<float>; double needs device fp64.
using SyevCtaFusedTestTypes = ::testing::Types<
    SyevCtaFusedConfig<float, Backend::SYCL>,
#if BATCHLAS_HAS_DEVICE_FP64
    SyevCtaFusedConfig<double, Backend::SYCL>,
    SyevCtaFusedConfig<std::complex<double>, Backend::SYCL>,
#endif
    SyevCtaFusedConfig<std::complex<float>, Backend::SYCL>>;
#else
using SyevCtaFusedTestTypes = ::testing::Types<SyevCtaFusedConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SyevCtaFusedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevCtaFusedTest, SyevCtaFusedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND || BATCHLAS_HAS_SYCL_BACKEND

// The fused kernel must produce a valid eigendecomposition for every partition
// width it dispatches (P = 4, 8, 16, 32) and for both triangles, including the
// sizes that only partly fill a partition.
TYPED_TEST(SyevCtaFusedTest, EigenvectorsAllSizesBothTrianglesResidualAndOrtho) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 8;

    for (int n : {1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 31, 32}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << (uplo == Uplo::Lower ? "L" : "U"));

            auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch,
                                                                 /*seed=*/1000u + n);
            auto A = A0;
            auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

            syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, uplo).wait();

            for (int b = 0; b < batch; ++b) {
                // Sorted ascending by default.
                for (int i = 1; i < n; ++i) {
                    EXPECT_LE(W[static_cast<std::size_t>(b) * n + i - 1],
                              W[static_cast<std::size_t>(b) * n + i])
                        << "eigenvalues not ascending at i=" << i;
                }
                test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, b);
            }
        }
    }
}

// Eigenvalues must match LAPACKE.
TYPED_TEST(SyevCtaFusedTest, EigenvaluesOnlyMatchesNetlib) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 16;
    const int batch = 128;

    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
    auto A_fused = A0;

    auto W_fused = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

    syev_cta_fused<B, Scalar>(*this->ctx, A_fused, W_fused.to_span(), JobType::NoEigenVectors,
                              Uplo::Lower)
        .wait();

    // Every item against LAPACKE in double (not a BatchLAS syev on this queue).
    test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_fused, n, false, A0.all_items());

    // jobz == NoEigenVectors must leave A alone (unlike syev_cta, which leaves
    // the reduction's reflectors behind).
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                ASSERT_EQ(A_fused.view()(i, j, b), A0.view()(i, j, b))
                    << "A modified at (" << i << "," << j << ") batch " << b;
            }
        }
    }
}

// The fused kernel and the three-kernel pipeline run the same algorithm on the
// same data, so their eigenvalues must agree to round-off.
TYPED_TEST(SyevCtaFusedTest, AgreesWithPartitionedSyevCta) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 16;

    for (int n : {4, 8, 16, 32}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);

        auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch,
                                                              /*seed=*/77u + n);
        auto A_fused = A0;
        auto A_pipe = A0;

        auto W_fused = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);
        auto W_pipe = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

        SteqrParams<Scalar> p;
        auto ws = UnifiedVector<std::byte>(
            syev_cta_buffer_size<B, Scalar>(*this->ctx, A_pipe, JobType::EigenVectors, p));
        syev_cta<B, Scalar>(*this->ctx, A_pipe, W_pipe.to_span(), JobType::EigenVectors,
                            Uplo::Lower, ws.to_span(), p)
            .wait();

        syev_cta_fused<B, Scalar>(*this->ctx, A_fused, W_fused.to_span(), JobType::EigenVectors,
                                  Uplo::Lower)
            .wait();

        test_utils::expect_eigenvalues_agree<Scalar>(W_fused, W_pipe, n, batch);
    }
}

TYPED_TEST(SyevCtaFusedTest, NearDegenerateSpectrum) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 24;
    const int batch = 4;

    Matrix<Scalar, MatrixFormat::Dense> A0(n, n, batch);
    A0.fill_near_degenerate_hermitian(*this->ctx, 1e-3, /*seed=*/1337).wait();
    auto A = A0;
    auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

    syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower).wait();

    test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, A0.all_items());
}

// Descending order and the work-group multiplier are both wired through to the
// same kernel, so a smoke test on each is enough.
TYPED_TEST(SyevCtaFusedTest, SortOrderAndWgMultiplier) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 16;
    const int batch = 64;

    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/909);

    for (size_t mult : {1u, 2u, 4u}) {
        SCOPED_TRACE(::testing::Message() << "wg_multiplier=" << mult);

        auto A = A0;
        auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

        SteqrParams<Scalar> p;
        p.sort_order = SortOrder::Descending;

        syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower,
                                  Span<std::byte>(), p, mult)
            .wait();

        for (int b = 0; b < batch; ++b) {
            for (int i = 1; i < n; ++i) {
                ASSERT_GE(W[static_cast<std::size_t>(b) * n + i - 1], W[static_cast<std::size_t>(b) * n + i])
                    << "eigenvalues not descending at i=" << i;
            }
        }
        test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, 0);
    }
}

// A requested multiplier is clamped to local memory in whole steps of lcm(P, 32) lanes = 32/P
// problems. The old clamp counted problems, so at P = 16 it admitted twice the bytes and the
// launch failed (DPC++: cdouble n=16 multiplier 8 asks 108,544 B of 101,376; acpp's 48 KiB:
// cfloat 8, cdouble 4). Complex only: a real type's clamped multiplier is a work-group too wide
// for its registers, a separate cap.
TYPED_TEST(SyevCtaFusedTest, WgMultiplierIsClampedToLocalMemoryInWholeSteps) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!test_utils::is_complex<Scalar>::value) {
        GTEST_SKIP() << "real types: the clamped work-group exceeds the register file first";
    } else {
        const int n = 16;
        const int batch = 203;  // ragged final work-group at every clamped width
        auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/4242);
        for (size_t mult : {8u, 16u, 32u}) {
            SCOPED_TRACE(::testing::Message() << "wg_multiplier=" << mult);
            auto A = A0;
            auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);
            auto run = [&] {
                syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower,
                                          Span<std::byte>(), SteqrParams<Scalar>{}, mult)
                    .wait();
            };
            ASSERT_NO_THROW(run());
            const int items[] = {0, batch / 2, batch - 1};
            test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, items);
        }
    }
}

// A full-size batch whose items grade in alternating directions, so the steqr
// solve inside the kernel mixes QL and QR blocks within every warp, and 4099
// leaves a ragged final work-group. Every item is checked: the eigenvector
// readout reads the accumulator tile across lanes after the solve.
TYPED_TEST(SyevCtaFusedTest, MixedDirectionLargeBatch) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 4099;

    for (int n : {8, 16}) {
        SCOPED_TRACE(::testing::Message() << "n=" << n);
        auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/4099u + n);
        const Real g = std::pow(Real(1e-2), Real(1) / Real(n - 1));
        for (int b = 0; b < batch; ++b) {
            const int kind = b % 3;  // 0: large end first, 1: small end first, 2: unscaled
            if (kind == 2) continue;
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    const int ki = kind == 0 ? i : n - 1 - i;
                    const int kj = kind == 0 ? j : n - 1 - j;
                    A0.view()(i, j, b) *= std::pow(g, Real(ki + kj));
                }
            }
        }
        auto A = A0;
        auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

        syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower).wait();

        test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, A0.all_items());
    }
}

// A batch one past a whole number of work-groups: every chunk but one of the
// final work-group is dead. Dead chunks run the kernel on a zero matrix aliased
// to item 0 rather than returning, so an ungated store would overwrite item 0
// long after its own work-group finished; k is large enough that it does.
TYPED_TEST(SyevCtaFusedTest, RaggedTailBatch) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    for (int n : {4, 8, 12, 32}) {
        const int k = n == 32 ? 4096 : 8192;  // more work-groups than fit resident
        const int P = n <= 4 ? 4 : (n <= 8 ? 8 : (n <= 16 ? 16 : 32));
        for (int mult : {1, 2}) {
            const int probs_per_wg = 32 * mult / P;
            if (probs_per_wg == 1) continue;  // no dead chunk
            const int batch = probs_per_wg * k + 1;
            SCOPED_TRACE(::testing::Message() << "n=" << n << " wg_multiplier=" << mult << " batch=" << batch);

            auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/77u + n);
            auto A = A0;
            auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);
            auto info = UnifiedVector<int32_t>(static_cast<std::size_t>(batch), int32_t(0));

            syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower,
                                      Span<std::byte>(), SteqrParams<Scalar>{}, static_cast<size_t>(mult),
                                      info.to_span())
                .wait();

            for (int b : {0, 1, batch - 2, batch - 1}) {
                SCOPED_TRACE(::testing::Message() << "item " << b);
                EXPECT_EQ(info[static_cast<std::size_t>(b)], 0);
                test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A, W, n, b);
            }
        }
    }
}

TYPED_TEST(SyevCtaFusedTest, RequiresNoWorkspace) {
    using Scalar = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(8, 8, /*hermitian=*/true, 2, /*seed=*/5);
    const size_t ws_bytes = syev_cta_fused_buffer_size<B, Scalar>(*this->ctx, A, JobType::EigenVectors);
    EXPECT_EQ(ws_bytes, 0u);
}


// A NaN entry in item 0: the rank sort must still be a bijection (NaN last, ties on index), so no
// slot of W keeps the 12345 it was poisoned with and A's columns are a permutation; items 1.. are
// unaffected.
TYPED_TEST(SyevCtaFusedTest, NaNInputLeavesNoStaleSlot) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16, batch = 4;
    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/77);
    A0.view()(3, 3, 0) = Scalar(std::numeric_limits<Real>::quiet_NaN());
    for (const auto order : {SortOrder::Ascending, SortOrder::Descending}) {
        for (const JobType jobz : {JobType::NoEigenVectors, JobType::EigenVectors}) {
            SCOPED_TRACE(::testing::Message() << "descending " << (order == SortOrder::Descending) << ", vectors " << (jobz == JobType::EigenVectors));
            auto A = A0;
            UnifiedVector<Real> W(static_cast<std::size_t>(n) * batch, Real(12345));
            SteqrParams<Scalar> params{};
            params.sort_order = order;
            syev_cta_fused<B, Scalar>(*this->ctx, A, W.to_span(), jobz, Uplo::Lower, Span<std::byte>(), params).wait();
            for (int i = 0; i < n; ++i) EXPECT_NE(W[i], Real(12345)) << "slot " << i << " of the NaN item was never written";
            const std::vector<int> clean{1, 2, 3};
            test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W, n, /*sorted_copy=*/true, clean);
        }
    }
}

#endif // CUDA || ROCM
