#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct OrgqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using OrgqrTestTypes = typename test_utils::backend_types<OrgqrConfig>::type;

template <typename Config>
class OrgqrTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    std::shared_ptr<Queue> ctx;

    void SetUp() override {
        if constexpr (BackendType != Backend::NETLIB) {
            try {
                ctx = std::make_shared<Queue>("gpu");
                if (!(ctx->device().type == DeviceType::GPU)) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL did not select a GPU device. Skipping.";
                }
            } catch (const sycl::exception& e) {
                if (e.code() == sycl::errc::runtime || e.code() == sycl::errc::feature_not_supported) {
                    GTEST_SKIP() << "CUDA backend selected, but SYCL GPU queue creation failed: " << e.what() << ". Skipping.";
                } else {
                    throw;
                }
            } catch (const std::exception& e) {
                GTEST_SKIP() << "CUDA backend selected, but Queue construction failed: " << e.what() << ". Skipping.";
            }
        } else {
            ctx = std::make_shared<Queue>("cpu");
        }
    }
};

TYPED_TEST_SUITE(OrgqrTest, OrgqrTestTypes);

TYPED_TEST(OrgqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
    geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
    orgqr<B>(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n);
    gemm<B>(*this->ctx, A.view(), A.view(), Result.view(), T(1), T(0), Transpose::Trans, Transpose::NoTrans);
    this->ctx->wait();

    auto r = Result.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T expected = (i == j) ? T(1) : T(0);
            EXPECT_NEAR(r[i * Result.ld() + j], expected, T(1e-4));
        }
    }
}

TYPED_TEST(OrgqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
    UnifiedVector<T> tau(n * batch);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
    geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
    orgqr<B>(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n, batch);
    gemm<B>(*this->ctx, A.view(), A.view(), Result.view(), T(1), T(0), Transpose::Trans, Transpose::NoTrans);
    this->ctx->wait();

    auto r = Result.data();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T expected = (i == j) ? T(1) : T(0);
                EXPECT_NEAR(r[b * Result.stride() + i * Result.ld() + j], expected, T(1e-4));
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

