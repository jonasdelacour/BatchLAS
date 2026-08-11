#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <limits>
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
    // A null value unsets the variable for the duration, which is how a test
    // asks for the automatic route regardless of what the environment already
    // said.
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name_)) {
            old_value_ = old;
            had_old_value_ = true;
        }
        if (value) {
            setenv(name_, value, 1);
        } else {
            unsetenv(name_);
        }
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

namespace {

struct Syr2kShape {
    int n;
    int k;
    int batch;
};

// SYR2K names one triangle of C, and BLAS forbids the other one from being
// written. Comparing against a symmetrized reference cannot see that: both
// halves then hold the same numbers, so a route that computes and stores the
// whole n x n passes anyway. Poisoning the unreferenced half with values
// nothing in the problem could produce makes the difference observable, and the
// values are distinct per element so that writing the transposed position is
// caught too, not merely writing something.
float syr2k_poison_value(int row, int col, int batch, int n) {
    return -static_cast<float>(1 + row + n * col + n * n * batch);
}

void poison_unreferenced_triangle(Matrix<float, MatrixFormat::Dense>& C, Uplo uplo) {
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (!referenced) {
                    C(i, j, b) = syr2k_poison_value(i, j, b, n);
                }
            }
        }
    }
}

void expect_triangle_respected(Matrix<float, MatrixFormat::Dense>& C,
                               Matrix<float, MatrixFormat::Dense>& C_ref,
                               Uplo uplo,
                               Transpose transA,
                               int k,
                               float tol) {
    const int n = C.rows();
    for (int b = 0; b < C.batch_size(); ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                if (referenced) {
                    ASSERT_NEAR(C(i, j, b), C_ref(i, j, b), tol)
                        << "n=" << n << ", k=" << k
                        << ", trans=" << static_cast<int>(transA)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                } else {
                    ASSERT_EQ(C(i, j, b), syr2k_poison_value(i, j, b, n))
                        << "wrote outside the requested triangle: n=" << n << ", k=" << k
                        << ", trans=" << static_cast<int>(transA)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", batch=" << b << ", row=" << i << ", col=" << j;
                }
            }
        }
    }
}

// One shape under both the route being exercised and the vendor, comparing the
// referenced triangle and requiring the poison back bit-exact from the other.
// A null variant means the automatic route, with the environment variable
// genuinely absent rather than spelled out.
void expect_route_respects_triangle(Queue& ctx,
                                    const char* variant,
                                    const Syr2kShape& shape,
                                    float alpha,
                                    float beta) {
    const float tol = test_utils::tolerance<float>() * 64.0f * static_cast<float>(shape.k);

    for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
        const int a_rows = transA == Transpose::NoTrans ? shape.n : shape.k;
        const int a_cols = transA == Transpose::NoTrans ? shape.k : shape.n;
        Matrix<float, MatrixFormat::Dense> A =
            Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 13);
        Matrix<float, MatrixFormat::Dense> B =
            Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 31);

        for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
            Matrix<float, MatrixFormat::Dense> C_custom =
                Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
            Matrix<float, MatrixFormat::Dense> C_vendor =
                Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
            poison_unreferenced_triangle(C_custom, uplo);
            poison_unreferenced_triangle(C_vendor, uplo);

            {
                ScopedEnvVar route_variant("BATCHLAS_SYR2K_VARIANT", variant);
                syr2k(ctx,
                      A.view(),
                      B.view(),
                      C_custom.view(),
                      {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
            }

            {
                ScopedEnvVar vendor_variant("BATCHLAS_SYR2K_VARIANT", "vendor");
                syr2k(ctx,
                      A.view(),
                      B.view(),
                      C_vendor.view(),
                      {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
            }

            ASSERT_NO_FATAL_FAILURE(
                expect_triangle_respected(C_custom, C_vendor, uplo, transA, shape.k, tol));
        }
    }
}

} // namespace

TEST(Syr2kCudaCustomTest, TriangularTilesLeaveTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    // 256x64 is whole 128 tiles with a k the 8-deep staging fills exactly, so
    // it takes the unpredicated path; 200x53 breaks both and takes the
    // predicated one, with a partial tile on the diagonal. 384x8 is the
    // shallowest k that runs, and wide enough to have a tile that is neither on
    // the diagonal nor next to it.
    const Syr2kShape shapes[] = {{256, 64, 8}, {200, 53, 5}, {384, 8, 3}};

    for (const auto& shape : shapes) {
        // beta = 0 takes the epilogue branch that never reads prior C, which is
        // the one path where the mask is not what keeps the other half intact.
        for (const auto beta : {-0.35f, 0.0f}) {
            ASSERT_NO_FATAL_FAILURE(
                expect_route_respects_triangle(ctx, "triangular", shape, 0.9f, beta));
        }
    }
}

TEST(Syr2kCudaCustomTest, AutoRouteLeavesTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    // What this guards is the routing, not any one kernel: whichever route a
    // shape picks, the unreferenced half of C belongs to the caller. The router
    // turns on one thing, the batch, and these straddle it in both directions at
    // three sizes, over kernel paths that differ in every other way.
    //
    //   512x64  b32   whole 128 tiles, wide enough to have tiles that are
    //                 neither on nor next to the diagonal
    //   424x53  b16   the same but ragged in n and off the staging's k step
    //   256x4   b8    a k shallower than one staging step
    //   200x53  b2    the smallest batch that leaves the vendor
    //   200x53  b1    one under it, so the same shape goes to the vendor
    //   128x128 b8    a tile grid that is nothing but diagonal
    //   512x512 b1    a large shape that still goes to the vendor
    //   24x24   b4    an n smaller than a single tile
    //   24x24   b1    the same at the batch the vendor keeps
    const Syr2kShape shapes[] = {{512, 64, 32}, {424, 53, 16},  {256, 4, 8},
                                 {200, 53, 2},  {200, 53, 1},   {128, 128, 8},
                                 {512, 512, 1}, {24, 24, 4},    {24, 24, 1}};

    for (const auto& shape : shapes) {
        ASSERT_NO_FATAL_FAILURE(
            expect_route_respects_triangle(ctx, nullptr, shape, 1.25f, 0.5f));
    }
}

// Shapes deliberately outside the two tests above, chosen against the router
// and the kernel's own branches rather than against each other: an n one row
// past a tile boundary, an n one row short of one, a k of one, a batch on each
// side of the condition that decides the route, and the two scalar values that
// change which epilogue branch runs.
TEST(Syr2kCudaCustomTest, AdversarialShapesLeaveTheOtherHalfUntouched) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    struct Case {
        Syr2kShape shape;
        float alpha;
        float beta;
    };

    const Case cases[] = {
        {{129, 9, 2}, 1.25f, 0.5f},     // one row into the second tile row
        {{257, 16, 3}, 1.25f, 0.5f},    // a one-row diagonal tile two tiles along
        {{127, 8, 2}, 1.25f, 0.5f},     // one row short of a whole tile
        {{640, 24, 2}, 1.25f, 0.5f},    // five whole tiles a side
        {{1, 5, 4}, 1.25f, 0.5f},       // a single element of C
        {{384, 1, 2}, 1.25f, 0.5f},     // k of one
        {{256, 256, 2}, 0.9f, 1.0f},    // beta exactly one
        {{200, 53, 2}, 0.0f, 0.5f},     // alpha of zero: C is only scaled
        {{256, 64, 2}, 1.0f, 0.0f},     // beta of zero on the whole-tile path
        {{129, 7, 1}, 1.25f, 0.5f},     // under the batch condition, ragged
        {{1024, 1024, 1}, 1.25f, 0.5f}, // under it at a size that fills the card
    };

    for (const auto& c : cases) {
        ASSERT_NO_FATAL_FAILURE(
            expect_route_respects_triangle(ctx, nullptr, c.shape, c.alpha, c.beta));
    }
}

// Writing outside the triangle is observable by poisoning; reading outside it
// is not, because a route can read a value it never writes back. A NaN in the
// unreferenced half makes the read observable instead: beta * NaN poisons
// whatever element it lands in, so a referenced half that comes back finite is
// proof that the beta term never crossed the boundary. Only a diagonal tile can
// cross it, which is why both a whole-tile and a ragged shape are here.
TEST(Syr2kCudaCustomTest, BetaNeverReadsOutsideTheTriangle) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom syr2k test requires a GPU device";
    }

    const Syr2kShape shapes[] = {{256, 64, 4}, {200, 53, 3}, {129, 9, 2}};
    const float alpha = 1.25f;
    const float beta = 0.5f;
    const float nan_value = std::numeric_limits<float>::quiet_NaN();

    for (const auto& shape : shapes) {
        const float tol = test_utils::tolerance<float>() * 64.0f * static_cast<float>(shape.k);

        for (auto transA : {Transpose::NoTrans, Transpose::Trans}) {
            const int a_rows = transA == Transpose::NoTrans ? shape.n : shape.k;
            const int a_cols = transA == Transpose::NoTrans ? shape.k : shape.n;
            Matrix<float, MatrixFormat::Dense> A =
                Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 13);
            Matrix<float, MatrixFormat::Dense> B =
                Matrix<float, MatrixFormat::Dense>::Random(a_rows, a_cols, false, shape.batch, 31);

            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                Matrix<float, MatrixFormat::Dense> C_custom =
                    Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
                Matrix<float, MatrixFormat::Dense> C_vendor =
                    Matrix<float, MatrixFormat::Dense>::Random(shape.n, shape.n, false, shape.batch, 43);
                for (int b = 0; b < shape.batch; ++b) {
                    for (int j = 0; j < shape.n; ++j) {
                        for (int i = 0; i < shape.n; ++i) {
                            const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                            if (!referenced) {
                                C_custom(i, j, b) = nan_value;
                            }
                        }
                    }
                }

                syr2k(ctx,
                      A.view(),
                      B.view(),
                      C_custom.view(),
                      {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();

                {
                    ScopedEnvVar vendor_variant("BATCHLAS_SYR2K_VARIANT", "vendor");
                    syr2k(ctx,
                          A.view(),
                          B.view(),
                          C_vendor.view(),
                          {.alpha = alpha, .beta = beta, .uplo = uplo, .trans = transA}).wait();
                }

                for (int b = 0; b < shape.batch; ++b) {
                    for (int j = 0; j < shape.n; ++j) {
                        for (int i = 0; i < shape.n; ++i) {
                            const bool referenced = uplo == Uplo::Lower ? i >= j : i <= j;
                            if (!referenced) {
                                continue;
                            }
                            ASSERT_NEAR(C_custom(i, j, b), C_vendor(i, j, b), tol)
                                << "read outside the requested triangle: n=" << shape.n
                                << ", k=" << shape.k
                                << ", trans=" << static_cast<int>(transA)
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