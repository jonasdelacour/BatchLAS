#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>

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

            syrk(*(this->ctx),
                     A.view(),
                     C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

            gemm(*(this->ctx),
                     A.view(),
                     A.view(),
                     C_ref.view(),
                     {.alpha = alpha, .beta = beta, .transA = transA, .transB = transA == Transpose::NoTrans ? Transpose::Trans : Transpose::NoTrans}).wait();

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

// MatchesGemmReference above pins one shape, n = 96, which reaches exactly one
// of the narrow kernel's three tile widths. The other two -- and the wider
// range ortho actually asks for, where k is the block size and n is tens of
// vectors -- went unexercised, so the band-split bug that broke n = 96 could
// equally have hidden at n = 32 or 64 with nothing to catch it. Each n here
// selects a different instantiation: 32 and 24 the 32-wide tile, 64 and 48 the
// 64-wide one, 128 and 96 the 128-wide one, with the odd sizes checking the
// masking of a tile the matrix does not fill.
TYPED_TEST(SyrkTest, NarrowShapesMatchGemmReference) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    const int k = 200;   // deliberately not a multiple of the k chunk
    const int batch = 3;
    const T alpha = T(0.9);
    const T beta = T(-0.35);
    const real_t tol = test_utils::tolerance<T>() * real_t(12 * k);

    for (int n : {24, 32, 48, 64, 96, 128}) {
        for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                const int a_rows = transA == Transpose::NoTrans ? n : k;
                const int a_cols = transA == Transpose::NoTrans ? k : n;

                Matrix<T, MatrixFormat::Dense> A =
                    Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
                Matrix<T, MatrixFormat::Dense> C0 =
                    Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
                Matrix<T, MatrixFormat::Dense> C(n, n, batch);
                Matrix<T, MatrixFormat::Dense> C_ref(n, n, batch);

                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C.view(), C0.view()).wait();
                MatrixView<T, MatrixFormat::Dense>::copy(*(this->ctx), C_ref.view(), C0.view()).wait();

                syrk(*(this->ctx), A.view(), C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

                gemm(*(this->ctx), A.view(), A.view(), C_ref.view(),
                     {.alpha = alpha, .beta = beta, .transA = transA,
                      .transB = transA == Transpose::NoTrans ? Transpose::Trans
                                                            : Transpose::NoTrans}).wait();

                // Comparing the untouched half would only compare C0 with
                // itself, so both sides are mirrored from the half syrk was
                // asked for -- that way every element of the answer is checked
                // against the reference, including the ones a wrongly indexed
                // thread tile would have left at their input value.
                C.view().symmetrize(*(this->ctx), uplo).wait();
                C_ref.view().symmetrize(*(this->ctx), uplo).wait();

                for (int b = 0; b < batch; ++b) {
                    for (int j = 0; j < n; ++j) {
                        for (int i = 0; i < n; ++i) {
                            ASSERT_NEAR(C(i, j, b), C_ref(i, j, b), tol)
                                << "n=" << n
                                << ", trans=" << static_cast<int>(transA)
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

// The poison tests further down are CUDA-only: they force a route with
// BATCHLAS_SYRK_VARIANT and skip without a GPU, so no *backend* other than CUDA
// has ever had "syrk leaves the other triangle alone" checked. This one is typed
// over every backend the build has. It is the contract SyrkOptions::uplo names,
// and it is exactly what the generic gemm fallback in src/extensions/syrk.cc got
// wrong: it aimed one gemm at C, which writes both triangles, with `uplo` an
// unnamed parameter. That fallback is instantiated for Backend::MKL only, so
// this test cannot fail in a build without MKL.
TYPED_TEST(SyrkTest, LeavesTheOtherTriangleUntouched) {
    using T = typename TestFixture::ScalarType;

    const int n = 48;
    const int k = 32;
    const int batch = 3;
    const T alpha = T(0.9);
    const T poison = T(-12345);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            for (auto beta : {T(0), T(0.5)}) {
                const int a_rows = transA == Transpose::NoTrans ? n : k;
                const int a_cols = transA == Transpose::NoTrans ? k : n;

                auto A = Matrix<T, MatrixFormat::Dense>::Random(a_rows, a_cols, false, batch);
                Matrix<T, MatrixFormat::Dense> C(n, n, batch);
                for (int b = 0; b < batch; ++b)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const bool referenced =
                                uplo == Uplo::Upper ? (i <= j) : (i >= j);
                            C(i, j, b) = referenced ? T(0.25) * T(i + j) : poison;
                        }

                syrk(*(this->ctx), A.view(), C.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

                for (int b = 0; b < batch; ++b)
                    for (int j = 0; j < n; ++j)
                        for (int i = 0; i < n; ++i) {
                            const bool referenced =
                                uplo == Uplo::Upper ? (i <= j) : (i >= j);
                            if (referenced) continue;
                            ASSERT_EQ(C(i, j, b), poison)
                                << "the half syrk was not given was written: "
                                << "trans=" << static_cast<int>(transA)
                                << ", uplo=" << static_cast<int>(uplo)
                                << ", batch=" << b << ", row=" << i << ", col=" << j;
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
                syrk(ctx,
                                    A.view(),
                                    C_custom.view(),
                                    {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYRK_VARIANT", "vendor");
                syrk(ctx,
                                    A.view(),
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

namespace {

// syrk names one triangle of C, and BLAS forbids the other one from being
// written. Building C symmetric and comparing against gemm cannot see that:
// both halves then hold the same numbers, so a route that computes and stores
// the whole n x n passes anyway. Poisoning the unreferenced half with values
// nothing in the problem could produce makes the difference observable, and
// the values are distinct per element so that writing the transposed position
// is caught too, not merely writing something.
float syrk_poison_value(int row, int col, int batch, int n) {
    return -static_cast<float>(1 + row + n * col + n * n * batch);
}

void poison_unreferenced_triangle(Matrix<float, MatrixFormat::Dense>& C, Uplo uplo) {
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (!referenced) {
                    C(i, j, b) = syrk_poison_value(i, j, b, n);
                }
            }
        }
    }
}

void expect_triangle_respected(Matrix<float, MatrixFormat::Dense>& C,
                               Matrix<float, MatrixFormat::Dense>& C_ref,
                               Uplo uplo,
                               Transpose transA,
                               float tol) {
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (referenced) {
                    ASSERT_NEAR(C(i, j, b), C_ref(i, j, b), tol)
                        << "n=" << n
                        << ", trans=" << static_cast<int>(transA)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                } else {
                    ASSERT_EQ(C(i, j, b), syrk_poison_value(i, j, b, n))
                        << "wrote outside the requested triangle: n=" << n
                        << ", trans=" << static_cast<int>(transA)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                }
            }
        }
    }
}

} // namespace

TEST(SyrkCudaCustomTest, TriangularTilesLeaveTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }

    struct Shape {
        int n;
        int k;
        int batch;
    };

    // 256x64 is whole 128 tiles with a k the 8-deep staging fills exactly, so
    // it takes the unpredicated path; 200x53 breaks both and takes the
    // predicated one, with a partial tile on the diagonal. 384x8 is the
    // shallowest k that runs, and wide enough to have a tile that is neither on
    // the diagonal nor next to it.
    const Shape shapes[] = {{256, 64, 8}, {200, 53, 5}, {384, 8, 3}};
    const float alpha = 0.9f;
    const float beta = -0.35f;

    for (const auto& shape : shapes) {
        const float tol = test_utils::tolerance<float>() * 64.0f * static_cast<float>(shape.k);
        for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
            const int a_rows = transA == Transpose::NoTrans ? shape.n : shape.k;
            const int a_cols = transA == Transpose::NoTrans ? shape.k : shape.n;
            Matrix<float, MatrixFormat::Dense> A =
                Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 17);

            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                Matrix<float, MatrixFormat::Dense> C_custom =
                    Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 23);
                Matrix<float, MatrixFormat::Dense> C_vendor =
                    Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 23);
                poison_unreferenced_triangle(C_custom, uplo);
                poison_unreferenced_triangle(C_vendor, uplo);

                {
                    ScopedEnvVar force_variant("BATCHLAS_SYRK_VARIANT", "triangular");
                    syrk(ctx,
                         A.view(),
                         C_custom.view(),
                         {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
                }
                {
                    ScopedEnvVar vendor_variant("BATCHLAS_SYRK_VARIANT", "vendor");
                    syrk(ctx,
                         A.view(),
                         C_vendor.view(),
                         {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
                }

                expect_triangle_respected(C_custom, C_vendor, uplo, transA, tol);
            }
        }
    }
}

TEST(SyrkCudaCustomTest, AutoRouteLeavesTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syrk test requires a GPU device";
    }

    struct Shape {
        int n;
        int k;
        int batch;
    };

    // What this guards is the routing, not any one kernel: whichever route a
    // shape picks, the unreferenced half of C belongs to the caller. The
    // shapes straddle every threshold the router has -- 512x64 batch 32 is
    // past both of the triangular kernel's, 512x512 batch 4 clears the tile
    // width but not the block count, 256x256 clears neither, and the two small
    // ones sit where the launch is the whole cost.
    const Shape shapes[] = {{512, 64, 32}, {512, 512, 4}, {256, 256, 8},
                            {128, 128, 2}, {96, 96, 1}};
    const float alpha = 1.25f;
    const float beta = 0.5f;

    for (const auto& shape : shapes) {
        const float tol =
            test_utils::tolerance<float>() * 64.0f * static_cast<float>(shape.k);

        Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(
            shape.n, shape.k, false, shape.batch, 41);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_auto =
                Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
            Matrix<float, MatrixFormat::Dense> C_vendor =
                Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
            poison_unreferenced_triangle(C_auto, uplo);
            poison_unreferenced_triangle(C_vendor, uplo);

            syrk(ctx,
                 A.view(),
                 C_auto.view(),
                 {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = Transpose::NoTrans}).wait();

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYRK_VARIANT", "vendor");
                syrk(ctx,
                     A.view(),
                     C_vendor.view(),
                     {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = Transpose::NoTrans}).wait();
            }

            expect_triangle_respected(C_auto, C_vendor, uplo, Transpose::NoTrans, tol);
        }
    }
}
#endif