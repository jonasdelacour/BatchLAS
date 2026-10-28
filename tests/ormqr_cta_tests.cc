// SPDX-License-Identifier: MIT
#include <array>
#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

using namespace batchlas;

namespace {

inline bool is_missing_subgroup32_message(const std::string& msg) {
    return msg.find("subgroup size 32") != std::string::npos;
}

} // namespace

#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>
#include <tests/ormqr_verify.hh>

namespace ts = test_support;

#if BATCHLAS_HAS_CUDA_BACKEND

template <typename T, Backend B>
struct OrmqrCtaConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using OrmqrCtaTestTypes = ::testing::Types<OrmqrCtaConfig<float, Backend::CUDA>, OrmqrCtaConfig<double, Backend::CUDA>>;

template <typename Config>
class OrmqrCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(OrmqrCtaTest, OrmqrCtaTestTypes);

TYPED_TEST(OrmqrCtaTest, MatchesNetlibOrmqrLeftRightTrans) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 16;
    const int batch = 4;
    const int k = n;

    // Build a QR factorization (A_fact, tau) on the CPU queue: the reflector layout the CTA kernel reads.
    Queue ctx_cpu("cpu");
    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2025);
    Matrix<T, MatrixFormat::Dense> A_fact = A0;

    UnifiedVector<T> tau(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(ctx_cpu, A_fact, tau.to_span()), std::byte{0});
    geqrf(ctx_cpu, A_fact, tau.to_span(), ws_geqrf.to_span()).wait();
    ctx_cpu.wait();

    VectorView<T> tau_view(tau.to_span(), /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

    std::string skip_reason;
    auto run_case = [&](Side side, Transpose trans) -> bool {
        Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/false, batch,
                                                       /*seed=*/(side == Side::Left ? 11 : 22) + (trans == Transpose::NoTrans ? 0 : 1));
        Matrix<T, MatrixFormat::Dense> C_cta = C0;

        UnifiedVector<std::byte> ws_dummy(1, std::byte{0});
        try {
            ormqx_cta(*this->ctx,
                         A_fact,
                         tau_view,
                         C_cta,
                         Uplo::Upper, //QR factorization
                         side,
                         trans,
                         k,
                         ws_dummy.to_span(),
                         /*cta_wg_size_multiplier=*/1)
                .wait();
            this->ctx->wait();
        } catch (const sycl::exception& e) {
            const std::string msg = e.what();
            if (test_utils::may_skip_missing_kernel(msg) || is_missing_subgroup32_message(msg)) {
                skip_reason = msg;
                return false;
            }
            throw;
        } catch (const std::exception& e) {
            const std::string msg = e.what();
            if (test_utils::may_skip_missing_kernel(msg) || is_missing_subgroup32_message(msg)) {
                skip_reason = msg;
                return false;
            }
            throw;
        }

        EXPECT_VERIFY(T, batchlas::verify::Check::blas, n, test_utils::ormqr_apply_error(*this->ctx, A_fact, tau_view, C0, C_cta, side, trans))
            << "side " << int(side) << " trans " << int(trans);
        return true;
    };

    if (!run_case(Side::Left, Transpose::NoTrans) ||
        !run_case(Side::Left, Transpose::Trans) ||
        !run_case(Side::Right, Transpose::NoTrans) ||
        !run_case(Side::Right, Transpose::Trans)) {
        GTEST_SKIP() << "Skipping ormq_cta due to runtime limitation: " << skip_reason;
    }
}

TYPED_TEST(OrmqrCtaTest, ThrowsOnNTooLarge) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 33;
    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    Matrix<T, MatrixFormat::Dense> C = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    Vector<T> tau(n, /*batch=*/1);
    UnifiedVector<std::byte> ws_dummy(1, std::byte{0});

    EXPECT_THROW(
        ((void)ormqx_cta(*this->ctx,
                    A,
                    VectorView<T>(tau),
                    C,
                    Uplo::Upper, //QR factorization
                    Side::Left,
                    Transpose::NoTrans,
                    /*k=*/n,
                    ws_dummy.to_span(),
                    1),
         this->ctx->wait()),
        std::exception);
}

// A batch tail: batch 1025 at n = 13 (P = 16) leaves dead partitions in the last work-group,
// beside the live one in its sub-group (multiplier 1) and as whole dead warps (multiplier 4).
// Both kernels (Left: C in registers; Right: C in local memory) run a dead partition to the
// end and store nothing (known-defects.md, defect 23). Every item of C is held BITWISE to
// batch 1024 at multiplier 1 (no dead partition); no slot outside the items may change.
TYPED_TEST(OrmqrCtaTest, BatchTailStoresNothingAndKeepsItemsBitwise) {
    using T = typename TestFixture::ScalarType;
    constexpr int n = 13, ld = n + 3, stride = ld * n + 5, pad = 3;
    const T poison = ts::poison<T>();

    Queue ctx_cpu("cpu");
    Matrix<T, MatrixFormat::Dense> F = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, 1, /*seed=*/2026);
    UnifiedVector<T> tau1(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(ctx_cpu, F, tau1.to_span()), std::byte{0});
    geqrf(ctx_cpu, F, tau1.to_span(), ws_geqrf.to_span()).wait();
    const Matrix<T, MatrixFormat::Dense> C1 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, 1, /*seed=*/7);

    UnifiedVector<std::byte> ws(1, std::byte{0});
    const auto run = [&](int batch, Side side, size_t mul) {
        UnifiedVector<T> a(static_cast<size_t>(batch) * stride, poison), c(static_cast<size_t>(batch + pad) * stride, poison);
        UnifiedVector<T> tau(static_cast<size_t>(batch) * (n + 1), poison);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                tau[static_cast<size_t>(b) * (n + 1) + j] = tau1[j];
                for (int i = 0; i < n; ++i) {
                    a[static_cast<size_t>(b) * stride + i + j * ld] = F.view()(i, j, 0);
                    c[static_cast<size_t>(b) * stride + i + j * ld] = C1.view()(i, j, 0);
                }
            }
        }
        MatrixView<T, MatrixFormat::Dense> A(a.data(), n, n, ld, stride, batch), C(c.data(), n, n, ld, stride, batch);
        VectorView<T> tv(tau.data(), n, batch, 1, n + 1);
        (void)ormqx_cta(*this->ctx, A, tv, C, Uplo::Upper, side, Transpose::Trans, n, ws.to_span(), mul);
        this->ctx->wait();
        return c;
    };
    for (const Side side : {Side::Left, Side::Right}) {
        const auto ref = run(1024, side, 1);
        for (const size_t mul : {size_t(1), size_t(4)}) {
            SCOPED_TRACE(::testing::Message() << "side " << int(side) << ", cta_wg_size_multiplier " << mul);
            const auto got = run(1025, side, mul);
            const auto view = [&](const UnifiedVector<T>& v, int off, int batch) {
                return ts::MV<T>(const_cast<T*>(v.data()) + static_cast<size_t>(off) * stride, n, n, ld, stride, batch);
            };
            // items 0..1023 and the last live item against the batch-1024 reference, whole stride
            EXPECT_TRUE(ts::items_bit_identical_to(view(got, 0, 1024), view(ref, 0, 1024)));
            EXPECT_TRUE(ts::items_bit_identical_to(view(got, 1024, 1), view(ref, 0, 1)));
            // the gaps of the live items and every slot of the items past the batch stay poison
            EXPECT_TRUE(ts::unwritten_outside(view(got, 0, 1025), ts::Live::block()));
            const std::array<int, 3> tail{1025, 1026, 1027};
            EXPECT_TRUE(ts::unwritten_outside(view(got, 0, 1025 + pad), ts::Live::nothing(), tail));
        }
    }
}

// Batch 0: no launch, no throw, nothing written (acpp 25.10 launches an empty nd_range).
TYPED_TEST(OrmqrCtaTest, EmptyBatchIsANoOp) {
    using T = typename TestFixture::ScalarType;
    const int n = 16;
    UnifiedVector<T> sentinel(3 * n * n, T(7));
    MatrixView<T, MatrixFormat::Dense> A(sentinel.data(), n, n, n + 1, (n + 1) * n, 0);
    MatrixView<T, MatrixFormat::Dense> C(sentinel.data() + n * n + n, n, n, n, n * n, 0);
    VectorView<T> tau(sentinel.data() + 2 * n * n + n, n, 0);
    UnifiedVector<std::byte> ws(1, std::byte{0});
    for (const Side side : {Side::Left, Side::Right}) {
        const auto run = [&] {
            (void)ormqx_cta(*this->ctx, A, tau, C, Uplo::Upper, side, Transpose::Trans, n, ws.to_span(), 1);
            this->ctx->wait();
        };
        EXPECT_NO_THROW(run());
    }
    for (size_t i = 0; i < sentinel.size(); ++i) ASSERT_EQ(sentinel[i], T(7)) << "index " << i;
}

#endif
