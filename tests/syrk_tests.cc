#include <gtest/gtest.h>

#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

#include <cstdlib>
#include <string>

#include "test_utils.hh"

using namespace batchlas;

template <typename T, Backend B>
struct SyrkConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SyrkTestTypes = typename test_utils::backend_types_filtered<SyrkConfig, false>::type;

template <typename Config>
class SyrkTest : public test_utils::BatchLASTest<Config> {};

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

TYPED_TEST_SUITE(SyrkTest, SyrkTestTypes);

TYPED_TEST(SyrkTest, MatchesGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 96;
    const int k = 64;
    const int batch = 3;
    const T alpha = T(0.9);
    const T beta = T(-0.35);
    const real_t tol = test_utils::tolerance<T>() * real_t(12 * k);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const int a_rows = transA == Transpose::NoTrans ? n : k;
            const int a_cols = transA == Transpose::NoTrans ? k : n;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
            Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
            Matrix<T, MatrixFormat::Dense> C(n, n, batch);
            Matrix<T, MatrixFormat::Dense> C_ref(n, n, batch);

            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();
            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C_ref.view(), C0.view()).wait();

            syrk<Ba>(*(this->ctx), A.view(), C.view(), alpha, beta, uplo, transA).wait();

            gemm<Ba>(*(this->ctx),
                     A.view(),
                     A.view(),
                     C_ref.view(),
                     alpha,
                     beta,
                     transA,
                     transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans).wait();

            C.view().symmetrize(*(this->ctx), uplo).wait();
            C_ref.view().symmetrize(*(this->ctx), uplo).wait();

            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        ASSERT_NEAR(C(i, j, b), C_ref(i, j, b), tol)
                            << "trans=" << static_cast<int>(transA)
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
TEST(SyrkCudaCustomTest, ForcedCuBLASDxPathMatchesVendor) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }

    const int n = 128;
    const int k = 96;
    const int batch = 64;
    const float alpha = 1.05f;
    const float beta = -0.2f;
    const float tol = test_utils::tolerance<float>() * 2048.0f;

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        const int a_rows = transA == Transpose::NoTrans ? n : k;
        const int a_cols = transA == Transpose::NoTrans ? k : n;

        Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 17);
        Matrix<float, MatrixFormat::Dense> C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 23);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);
            Matrix<float, MatrixFormat::Dense> C_vendor(n, n, batch);

            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_vendor.view(), C0.view()).wait();

            {
                ScopedEnvVar force_variant("BATCHLAS_SYRK_VARIANT", "cublasdx");
                syrk<Backend::CUDA>(ctx, A.view(), C_custom.view(), alpha, beta, uplo, transA).wait();
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYRK_VARIANT", "vendor");
                syrk<Backend::CUDA>(ctx, A.view(), C_vendor.view(), alpha, beta, uplo, transA).wait();
            }

            C_custom.view().symmetrize(ctx, uplo).wait();
            C_vendor.view().symmetrize(ctx, uplo).wait();

            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        ASSERT_NEAR(C_custom(i, j, b), C_vendor(i, j, b), tol)
                            << "trans=" << static_cast<int>(transA)
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