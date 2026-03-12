#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <cstdlib>
#include <string>
#include "test_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
struct SymmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SymmTestTypes = typename test_utils::backend_types_filtered<SymmConfig, false>::type;

template <typename Config>
class SymmTest : public test_utils::BatchLASTest<Config> {};

class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name_)) {
            old_value_ = old;
            had_old_value_ = true;
        }
        setenv(name_, value, 1);
    }

    ~ScopedEnvVar() {
        if (had_old_value_) {
            setenv(name_, old_value_.c_str(), 1);
        } else {
            unsetenv(name_);
        }
    }

private:
    const char* name_;
    std::string old_value_;
    bool had_old_value_ = false;
};

TYPED_TEST_SUITE(SymmTest, SymmTestTypes);

TYPED_TEST(SymmTest, MatchesSymmetrizedGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 96;
    const int m = 64;
    const int batch = 3;
    const T alpha = T(1.25);
    const T beta = T(-0.5);
    const real_t tol = test_utils::tolerance<T>() * real_t(12 * n);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const int rows = side == Side::Left ? n : m;
            const int cols = side == Side::Left ? m : n;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
            Matrix<T, MatrixFormat::Dense> B = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch);
            Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, batch);

            Matrix<T, MatrixFormat::Dense> A_ref(n, n, batch);
            Matrix<T, MatrixFormat::Dense> C(rows, cols, batch);
            Matrix<T, MatrixFormat::Dense> C_ref(rows, cols, batch);

            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), A_ref.view(), A.view()).wait();
            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();
            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C_ref.view(), C0.view()).wait();

            A_ref.view().symmetrize(*(this->ctx), uplo).wait();

            symm<Ba>(*(this->ctx), A.view(), B.view(), C.view(), alpha, beta, side, uplo).wait();

            if (side == Side::Left) {
                gemm<Ba>(*(this->ctx),
                         A_ref.view(),
                         B.view(),
                         C_ref.view(),
                         alpha,
                         beta,
                         Transpose::NoTrans,
                         Transpose::NoTrans).wait();
            } else {
                gemm<Ba>(*(this->ctx),
                         B.view(),
                         A_ref.view(),
                         C_ref.view(),
                         alpha,
                         beta,
                         Transpose::NoTrans,
                         Transpose::NoTrans).wait();
            }

            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < cols; ++j) {
                    for (int i = 0; i < rows; ++i) {
                        ASSERT_NEAR(C(i, j, b), C_ref(i, j, b), tol)
                            << "side=" << static_cast<int>(side)
                            << ", uplo=" << static_cast<int>(uplo)
                            << ", batch=" << b
                            << ", row=" << i
                            << ", col=" << j;
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#if BATCHLAS_HAS_CUDA_BACKEND
TEST(SymmCudaCustomTest, ForcedCuBLASDxPathMatchesVendor) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }

    const int n = 128;
    const int batch = 64;
    const float alpha = 1.1f;
    const float beta = -0.3f;
    const float tol = test_utils::tolerance<float>() * 2048.0f;

    Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 7);
    Matrix<float, MatrixFormat::Dense> B = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 11);
    Matrix<float, MatrixFormat::Dense> C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 13);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);
            Matrix<float, MatrixFormat::Dense> C_vendor(n, n, batch);

            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_vendor.view(), C0.view()).wait();

            {
                ScopedEnvVar force_variant("BATCHLAS_SYMM_VARIANT", "cublasdx");
                symm<Backend::CUDA>(ctx, A.view(), B.view(), C_custom.view(), alpha, beta, side, uplo).wait();
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYMM_VARIANT", "vendor");
                symm<Backend::CUDA>(ctx, A.view(), B.view(), C_vendor.view(), alpha, beta, side, uplo).wait();
            }

            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        ASSERT_NEAR(C_custom(i, j, b), C_vendor(i, j, b), tol)
                            << "side=" << static_cast<int>(side)
                            << ", uplo=" << static_cast<int>(uplo)
                            << ", batch=" << b
                            << ", row=" << i
                            << ", col=" << j;
                    }
                }
            }
        }
    }
}
#endif