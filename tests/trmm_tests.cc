#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/extra.hh>
#include <util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct TrmmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using TrmmTestTypes = typename test_utils::backend_types<TrmmConfig>::type;

template <typename Config>
class TrmmTest : public ::testing::Test {
protected:
    using ScalarType = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;
    std::shared_ptr<Queue> ctx;
    Transpose trans = test_utils::is_complex<ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;

    void SetUp() override {
        if constexpr (BackendType != Backend::NETLIB) {
            try {
                ctx = std::make_shared<Queue>("gpu", false);
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
            ctx = std::make_shared<Queue>("cpu", false);
        }
    }
};

TYPED_TEST_SUITE(TrmmTest, TrmmTestTypes);

TYPED_TEST(TrmmTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;
    const int n = 1024;
    const int batch_size = 16;

    auto A = Matrix<T, MatrixFormat::Dense>::RandomTriangular(n, Uplo::Lower, Diag::NonUnit, batch_size);
    auto B = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch_size);
    auto C = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch_size);
    trmm<Ba>(*(this->ctx), A.view(), B.view(), C.view(), T(1.0), Side::Left, Uplo::Lower, Transpose::NoTrans, Diag::NonUnit).wait();
    gemm<Ba>(*(this->ctx), A.view(), B.view(), C.view(), T(1.0), T(-1.0), Transpose::NoTrans, Transpose::NoTrans).wait();

    std::cout << norm(*(this->ctx), C.view()) << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}