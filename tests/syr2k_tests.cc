#include <gtest/gtest.h>

#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "test_utils.hh"

#if BATCHLAS_HAS_CUDA_BACKEND
#include "../src/backends/syr2k_cublasdx_fused.hh"
#endif

using namespace batchlas;

template <typename T, Backend B>
struct Syr2kConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using Syr2kTestTypes = typename test_utils::backend_types_filtered<Syr2kConfig, false>::type;

template <typename Config>
class Syr2kTest : public test_utils::BatchLASTest<Config> {};

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

TYPED_TEST_SUITE(Syr2kTest, Syr2kTestTypes);

TYPED_TEST(Syr2kTest, MatchesGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;
    constexpr Backend Ba = TestFixture::BackendType;

    const int n = 96;
    const int k = 64;
    const int batch = 3;
    const T alpha = T(0.85);
    const T beta = T(-0.15);
    const real_t tol = test_utils::tolerance<T>() * real_t(20 * k);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            const int a_rows = transA == Transpose::NoTrans ? n : k;
            const int a_cols = transA == Transpose::NoTrans ? k : n;

            Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 11);
            Matrix<T, MatrixFormat::Dense> B = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 29);
            Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch, 41);
            Matrix<T, MatrixFormat::Dense> C(n, n, batch);
            Matrix<T, MatrixFormat::Dense> C_ref(n, n, batch);

            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();
            MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C_ref.view(), C0.view()).wait();

            syr2k(*(this->ctx),
                      A.view(),
                      B.view(),
                      C.view(),
                      {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

            const Transpose transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans;
            gemm(*(this->ctx),
                     A.view(),
                     B.view(),
                     C_ref.view(),
                     {.alpha = alpha, .beta = beta, .transA = transA, .transB = transB}).wait();
            gemm(*(this->ctx),
                     B.view(),
                     A.view(),
                     C_ref.view(),
                     {.alpha = alpha, .beta = T(1), .transA = transA, .transB = transB}).wait();

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
TEST(Syr2kCudaCustomTest, ForcedCuBLASDxPathMatchesVendor) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    const int n = 128;
    const int k = 96;
    const int batch = 64;
    const float alpha = 0.95f;
    const float beta = -0.1f;
    const float tol = test_utils::tolerance<float>() * 4096.0f;

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        const int a_rows = transA == Transpose::NoTrans ? n : k;
        const int a_cols = transA == Transpose::NoTrans ? k : n;
        Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 13);
        Matrix<float, MatrixFormat::Dense> B = Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 31);
        Matrix<float, MatrixFormat::Dense> C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 43);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);
            Matrix<float, MatrixFormat::Dense> C_vendor(n, n, batch);

            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_vendor.view(), C0.view()).wait();

            {
                ScopedEnvVar force_variant("BATCHLAS_SYR2K_VARIANT", "cublasdx");
                try {
                    syr2k(ctx,
                                         A.view(),
                                         B.view(),
                                         C_custom.view(),
                                         {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
                } catch (const std::runtime_error& err) {
                    EXPECT_FALSE(batchlas::backend::syr2k_cublasdx::available());
                    EXPECT_NE(std::string(err.what()).find("BATCHLAS_SYR2K_VARIANT=cublasdx"), std::string::npos);
                    return;
                }
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYR2K_VARIANT", "vendor");
                syr2k(ctx,
                                     A.view(),
                                     B.view(),
                                     C_vendor.view(),
                                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
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

// SYR2K names one triangle of C and the other one is the caller's storage.
// Both tests above symmetrize C before comparing, so neither can see a route
// that computes and stores the whole n x n -- and that is what the custom
// route does: it decomposes into two batched GEMMs written straight into C.
//
// Poisoning the unreferenced half with per-element sentinels makes it
// observable. The values are distinct per element so that writing the
// transposed position is caught too, not merely writing something.
//
// Left DISABLED rather than red, because fixing it is a separate decision with
// no cheap answer. The two routes that respect the triangle are a host loop
// over cublasSsyr2k -- measured here at 16.6 ms for n=512 batch=512 against
// the decomposition's 6.0 -- or a tile-masked kernel of the kind syrk now has,
// which does not exist for SYR2K's two operands.
//
// Recording it because the same poisoning turned up a live SYRK bug, and SYRK
// and SYR2K share the shape of the mistake.
TEST(Syr2kCudaCustomTest, DISABLED_AutoRouteLeavesTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    const int n = 128;
    const int k = 96;
    const int batch = 64;
    const float alpha = 0.95f;
    const float beta = -0.1f;
    const float tol = test_utils::tolerance<float>() * 4096.0f;

    auto poison = [n](int row, int col, int b) {
        return -static_cast<float>(1 + row + n * col + n * n * b);
    };

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        const int a_rows = transA == Transpose::NoTrans ? n : k;
        const int a_cols = transA == Transpose::NoTrans ? k : n;
        Matrix<float, MatrixFormat::Dense> A =
            Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 13);
        Matrix<float, MatrixFormat::Dense> B =
            Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch, 31);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_auto =
                Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 43);
            Matrix<float, MatrixFormat::Dense> C_vendor =
                Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 43);

            // The element accessor, not raw data() arithmetic: the storage has
            // its own leading dimension and batch stride.
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                        if (!referenced) {
                            C_auto(i, j, b) = poison(i, j, b);
                            C_vendor(i, j, b) = poison(i, j, b);
                        }
                    }
                }
            }

            syr2k(ctx,
                  A.view(),
                  B.view(),
                  C_auto.view(),
                  {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYR2K_VARIANT", "vendor");
                syr2k(ctx,
                      A.view(),
                      B.view(),
                      C_vendor.view(),
                      {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
            }

            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < n; ++j) {
                    for (int i = 0; i < n; ++i) {
                        const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                        if (referenced) {
                            ASSERT_NEAR(C_auto(i, j, b), C_vendor(i, j, b), tol)
                                << "trans=" << static_cast<int>(transA)
                                << ", uplo=" << static_cast<int>(uplo)
                                << ", batch=" << b << ", row=" << i << ", col=" << j;
                        } else {
                            ASSERT_EQ(C_auto(i, j, b), poison(i, j, b))
                                << "wrote outside the requested triangle: trans="
                                << static_cast<int>(transA)
                                << ", uplo=" << static_cast<int>(uplo)
                                << ", batch=" << b << ", row=" << i << ", col=" << j;
                        }
                    }
                }
            }
        }
    }
}
#endif