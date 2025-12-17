#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <type_traits>

using namespace batchlas;

template <typename T, Backend B>
struct SyevConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using SyevTestTypes = typename test_utils::backend_types<SyevConfig>::type;

template <typename Config>
class SyevTest : public test_utils::BatchLASTest<Config> {
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

TYPED_TEST_SUITE(SyevTest, SyevTestTypes);

TYPED_TEST(SyevTest, DiagTest) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    auto diag = UnifiedVector<T>(n);
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<typename base_type<T>::type> dis(-10.0, 10.0);

    // Populate diagonal with random values
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            diag[i] = {dis(gen), 0.0};
        } else if constexpr (std::is_floating_point_v<T>) {
            diag[i] = dis(gen);
        }
    }
    
    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Diagonal(diag.to_span());
    auto A_view = A.view();

    auto W = UnifiedVector<typename base_type<T>::type>(n);
    auto workspace = UnifiedVector<std::byte>(syev_buffer_size<B>(*this->ctx, A_view, W.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    std::sort(diag.begin(), diag.end(), [](const T& a, const T& b) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            return std::real(a) < std::real(b);
        } else if constexpr (std::is_floating_point_v<T>) {
            return a < b;
        }
    });
    syev<B>(*this->ctx, A_view, W.to_span(), JobType::NoEigenVectors, Uplo::Lower, workspace.to_span());
    (*this->ctx).wait();
    for (int i = 0; i < n; ++i) {
        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            EXPECT_NEAR(std::real(diag[i]), W[i], 1e-5);
        } else if constexpr (std::is_floating_point_v<T>) {
            EXPECT_NEAR(diag[i], W[i], 1e-5);
        }
    }
}