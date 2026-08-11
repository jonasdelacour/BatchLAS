#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
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

            symm(*(this->ctx),
                     A.view(),
                     B.view(),
                     C.view(),
                     {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();

            if (side == Side::Left) {
                gemm(*(this->ctx),
                         A_ref.view(),
                         B.view(),
                         C_ref.view(),
                         {.alpha = alpha, .beta = beta}).wait();
            } else {
                gemm(*(this->ctx),
                         B.view(),
                         A_ref.view(),
                         C_ref.view(),
                         {.alpha = alpha, .beta = beta}).wait();
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
                symm(ctx,
                                    A.view(),
                                    B.view(),
                                    C_custom.view(),
                                    {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYMM_VARIANT", "vendor");
                symm(ctx,
                                    A.view(),
                                    B.view(),
                                    C_vendor.view(),
                                    {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
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

// The custom path expands the referenced triangle into scratch a 32x32 tile at
// a time, so the sizes that matter are the ones where that tiling is ragged and
// the ones where the storage's leading dimension is not the matrix width.
TEST(SymmCudaCustomTest, ForcedCuBLASDxPathIgnoresUnreferencedTriangle) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }

    const int m = 61;
    const int batch = 5;
    const float alpha = 1.25f;
    const float beta = -0.5f;

    for (int n : {16, 33, 77, 129}) {
        const float tol = test_utils::tolerance<float>() * 4096.0f * static_cast<float>(n);

        for (auto side : {Side::Left, Side::Right}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                const int rows = side == Side::Left ? n : m;
                const int cols = side == Side::Left ? m : n;

                auto A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 17);
                auto B = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, false, batch, 19);
                auto C0 = Matrix<float, MatrixFormat::Dense>::Random(rows, cols, false, batch, 23);
                ctx.wait();

                // Use the element accessor, not raw data() arithmetic: the
                // storage has its own leading dimension and batch stride, and
                // assuming ld == n silently writes into the referenced triangle.
                for (int b = 0; b < batch; ++b) {
                    for (int col = 0; col < n; ++col) {
                        for (int row = 0; row < n; ++row) {
                            const bool referenced =
                                (uplo == Uplo::Lower) ? (row >= col) : (row <= col);
                            if (!referenced) {
                                A(row, col, b) = 1000.0f;
                            }
                        }
                    }
                }
                ctx.wait();

                Matrix<float, MatrixFormat::Dense> C_custom(rows, cols, batch);
                Matrix<float, MatrixFormat::Dense> C_vendor(rows, cols, batch);
                MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();
                MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_vendor.view(), C0.view()).wait();

                {
                    ScopedEnvVar force_variant("BATCHLAS_SYMM_VARIANT", "cublasdx");
                    symm(ctx, A.view(), B.view(), C_custom.view(),
                         {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
                }
                {
                    ScopedEnvVar vendor_variant("BATCHLAS_SYMM_VARIANT", "vendor");
                    symm(ctx, A.view(), B.view(), C_vendor.view(),
                         {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
                }

                for (int b = 0; b < batch; ++b) {
                    for (int j = 0; j < cols; ++j) {
                        for (int i = 0; i < rows; ++i) {
                            ASSERT_NEAR(C_custom(i, j, b), C_vendor(i, j, b), tol)
                                << "n=" << n
                                << ", side=" << static_cast<int>(side)
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
}

// The expansion and the GEMM that consumes it are ordered by the queue's native
// stream, which only exists on an in-order queue; the out-of-order case takes a
// different ordering path and is not otherwise exercised.
TEST(SymmCudaCustomTest, ForcedCuBLASDxPathOrdersExpansionOnOutOfOrderQueue) {
    Queue ordered;
    if (ordered.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom symm test requires a GPU device";
    }
    Queue ctx(ordered.device(), Backend::CUDA, /*in_order=*/false);

    const int n = 192;
    const int batch = 8;
    const float alpha = 1.25f;
    const float beta = -0.5f;
    const float tol = test_utils::tolerance<float>() * 4096.0f * static_cast<float>(n);

    for (auto side : {Side::Left, Side::Right}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            auto A = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 29);
            auto B = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 31);
            auto C0 = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 37);

            Matrix<float, MatrixFormat::Dense> C_custom(n, n, batch);
            Matrix<float, MatrixFormat::Dense> C_vendor(n, n, batch);
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_custom.view(), C0.view()).wait();
            MatrixView<float, MatrixFormat::Dense>::copy(ctx, C_vendor.view(), C0.view()).wait();

            {
                ScopedEnvVar force_variant("BATCHLAS_SYMM_VARIANT", "cublasdx");
                symm(ctx, A.view(), B.view(), C_custom.view(),
                     {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
            }
            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYMM_VARIANT", "vendor");
                symm(ctx, A.view(), B.view(), C_vendor.view(),
                     {.alpha = alpha, .beta = beta, .side = side, .uplo = uplo}).wait();
            }
            ctx.wait();

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