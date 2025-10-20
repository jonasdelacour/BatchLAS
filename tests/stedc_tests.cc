#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct StedcConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using StedcTestTypes = typename test_utils::backend_types<StedcConfig>::type;

template <typename Config>
class StedcTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    std::shared_ptr<Queue> ctx;
    Transpose trans = test_utils::is_complex<ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;

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
    auto tol = test_utils::tolerance<float_type>();
    for (int i = 0; i < n * batch; ++i) {
        EXPECT_NEAR(eigvals[i], ref_eigvals[i], 1e-2);
        EXPECT_NEAR(eigvals[i], ritz_vals[i], 1e-3);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
