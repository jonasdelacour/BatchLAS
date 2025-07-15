#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct SteqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using SteqrTestTypes = typename test_utils::backend_types<SteqrConfig>::type;

template <typename Config>
class SteqrTest : public ::testing::Test {
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

TYPED_TEST_SUITE(SteqrTest, SteqrTestTypes);

TYPED_TEST(SteqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    UnifiedVector<float_type> diag(n, float_type(a));
    UnifiedVector<float_type> sub_diag(n, float_type(b));
    UnifiedVector<float_type> eigenvalues(n);
    UnifiedVector<float_type> expected_eigenvalues(n);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }

    auto ws = UnifiedVector<std::byte>(tridiagonal_solver_buffer_size<B, float_type>(*this->ctx, n, 1, JobType::NoEigenVectors));
    tridiagonal_solver<B>(*this->ctx, diag.to_span(), sub_diag.to_span(), eigenvalues.to_span(), ws.to_span(), JobType::NoEigenVectors, MatrixView<float_type, MatrixFormat::Dense>(nullptr, n, n ,n), n, 1);
    this->ctx->wait();
    std::sort(eigenvalues.begin(), eigenvalues.end(), std::less<float_type>());
    for (int i = 0; i < n; ++i) {
        EXPECT_NEAR(eigenvalues[i], expected_eigenvalues[i], 1e-5) << "Eigenvalue mismatch at index " << i;
    }

}

TYPED_TEST(SteqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;


}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
