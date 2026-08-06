#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/extra.hh>
#include <util/sycl-device-queue.hh>

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "test_utils.hh"

#if BATCHLAS_HAS_CUDA_BACKEND
#include "../src/backends/trmm_cublasdx_fused.hh"
#endif

using namespace batchlas;

template <typename T, Backend B>
struct TrmmConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using TrmmTestTypes = typename test_utils::backend_types<TrmmConfig>::type;

template <typename Config>
class TrmmTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

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

TYPED_TEST_SUITE(TrmmTest, TrmmTestTypes);

TYPED_TEST(TrmmTest, AllCombinations) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend Ba = TestFixture::BackendType;

    // keep the problem size small so that iterating over all parameter combinations is feasible
    const int n         = 512;
    const int batchSize = 4;

    // reuse one random B matrix for all permutations
    Matrix<T> B = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batchSize);
    Matrix<T> C = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batchSize);
    // loop over every combination of transpose, side, uplo and diagonal
    for (auto trans : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        for (auto side : {Side::Right, Side::Left}) {
            for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                for (auto diag : {Diag::NonUnit, Diag::Unit}) {
                    // generate A for the current uplo/diag
                    Matrix<T> A = Matrix<T, MatrixFormat::Dense>::RandomTriangular(n, uplo, diag, batchSize);
                    

                    // compute C = trmm(A.view(),B.view()) with the current combination
                    trmm(*(this->ctx),
                             A.view(),
                             B.view(),
                             C.view(),
                             {.side = side, .uplo = uplo, .trans = trans, .diag = diag}).wait();

                    // subtract the full matrix product from C to obtain the residual
                    if (side == Side::Right) {
                        gemm(*(this->ctx),
                                 B.view(),
                                 A.view(),
                                 C.view(),
                                 {.beta = T(-1.0), .transB = trans}).wait();
                    } else {
                        gemm(*(this->ctx),
                                 A.view(),
                                 B.view(),
                                 C.view(),
                                 {.beta = T(-1.0), .transA = trans}).wait();
                    }

                    // the residual should be close to zero for a correct implementation
                    auto   diffNorm = norm(*(this->ctx), C.view());
                    using real_t   = typename base_type<T>::type;
                    real_t tol     = test_utils::tolerance<T>() * real_t(n);
                    for (auto norm : diffNorm) {
                        // check if the norm is within the tolerance
                        EXPECT_LE(norm, tol)
                        << "Failed combination: trans=" << static_cast<int>(trans)
                        << ", side=" << static_cast<int>(side)
                        << ", uplo=" << static_cast<int>(uplo)
                        << ", diag=" << static_cast<int>(diag);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

#if BATCHLAS_HAS_CUDA_BACKEND
TEST(TrmmCudaCustomTest, ForcedCuBLASDxPathMatchesVendor) {
    Queue ctx;
    if (ctx.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "CUDA custom trmm test requires a GPU device";
    }

    const int n = 128;
    const int batch = 64;
    const float alpha = 0.9f;
    const float tol = test_utils::tolerance<float>() * 4096.0f;

    for (auto diag : {Diag::NonUnit, Diag::Unit}) {
        Matrix<float> A = Matrix<float, MatrixFormat::Dense>::RandomTriangular(n, Uplo::Lower, diag, batch, 7);
        Matrix<float> B = Matrix<float, MatrixFormat::Dense>::Random(n, n, false, batch, 19);
        Matrix<float> C_custom(n, n, batch);
        Matrix<float> C_vendor(n, n, batch);

        {
            ScopedEnvVar force_variant("BATCHLAS_TRMM_VARIANT", "cublasdx");
            try {
                trmm(ctx,
                                    A.view(),
                                    B.view(),
                                    C_custom.view(),
                                    {.alpha = alpha, .diag = diag}).wait();
            } catch (const std::runtime_error& err) {
                EXPECT_FALSE(batchlas::backend::trmm_cublasdx::available());
                EXPECT_NE(std::string(err.what()).find("BATCHLAS_TRMM_VARIANT=cublasdx"), std::string::npos);
                return;
            }
        }

        {
            ScopedEnvVar vendor_variant("BATCHLAS_TRMM_VARIANT", "vendor");
            trmm(ctx,
                                A.view(),
                                B.view(),
                                C_vendor.view(),
                                {.alpha = alpha, .diag = diag}).wait();
        }

        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    ASSERT_NEAR(C_custom(i, j, b), C_vendor(i, j, b), tol)
                        << "diag=" << static_cast<int>(diag)
                        << ", batch=" << b
                        << ", row=" << i
                        << ", col=" << j;
                }
            }
        }
    }
}
#endif

// TRMM must not reference the opposite triangle of A, nor its diagonal when
// Diag::Unit is requested.
//
// Every other test in this file builds A with RandomTriangular -- already
// materialised with zeros in the unreferenced half and ones on a unit
// diagonal -- and then validates against a full gemm on that same A. That
// comparison cannot distinguish a real trmm from a plain gemm, so it passes
// for an implementation that ignores uplo and diag entirely.
//
// This test poisons the storage TRMM is forbidden to read. The reference is a
// gemm against the clean A, so a correct trmm is unaffected while an
// implementation that reads the poisoned entries is not.
//
// A ragged dimension and a non-square B are in the shapes because the CUDA
// backend materialises the triangle into packed scratch with a leading
// dimension of its own. The sweep runs twice there: capping the scratch budget
// at zero bytes sends it down the no-scratch route it otherwise only reaches
// when the expansion will not fit on the device, which no test shape does.
TYPED_TEST(TrmmTest, IgnoresUnreferencedTriangleAndUnitDiagonal) {
    using T = typename TestFixture::ScalarType;
    using real_t = typename base_type<T>::type;

    struct Shape {
        int rows;
        int cols;
        int batch;
    };
    const Shape shapes[] = {{64, 48, 2}, {129, 96, 5}, {300, 32, 1}};

    auto sweep = [&](const char* route) {
        for (const auto& shape : shapes) {
            for (auto side : {Side::Left, Side::Right}) {
                const int k = side == Side::Left ? shape.rows : shape.cols;
                for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
                    for (auto diag : {Diag::NonUnit, Diag::Unit}) {
                        for (auto trans : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                            auto A_clean = Matrix<T, MatrixFormat::Dense>::RandomTriangular(k, uplo, diag, shape.batch);
                            auto B = Matrix<T, MatrixFormat::Dense>::Random(shape.rows, shape.cols, false, shape.batch);
                            auto A_poisoned = A_clean.clone();
                            this->ctx->wait();

                            // Use the element accessor, not raw data() arithmetic: the
                            // storage has its own leading dimension and batch stride, and
                            // assuming ld == n silently writes into the referenced triangle.
                            for (int b = 0; b < shape.batch; ++b) {
                                for (int col = 0; col < k; ++col) {
                                    for (int row = 0; row < k; ++row) {
                                        const bool referenced =
                                            (uplo == Uplo::Lower) ? (row > col) : (row < col);
                                        if (referenced) continue;
                                        if (row == col && diag == Diag::NonUnit) continue;
                                        A_poisoned(row, col, b) = T(1000);
                                    }
                                }
                            }
                            this->ctx->wait();

                            auto C = Matrix<T, MatrixFormat::Dense>::Zeros(shape.rows, shape.cols, shape.batch);
                            trmm(*(this->ctx), A_poisoned.view(), B.view(), C.view(),
                                 {.side = side, .uplo = uplo, .trans = trans, .diag = diag})
                                .wait();

                            // Subtract the product against the clean A; the residual must vanish.
                            if (side == Side::Left) {
                                gemm(*(this->ctx), A_clean.view(), B.view(), C.view(),
                                     {.beta = T(-1.0), .transA = trans}).wait();
                            } else {
                                gemm(*(this->ctx), B.view(), A_clean.view(), C.view(),
                                     {.beta = T(-1.0), .transB = trans}).wait();
                            }

                            const real_t tol = test_utils::tolerance<T>() * real_t(k);
                            for (auto residual : norm(*(this->ctx), C.view())) {
                                EXPECT_LE(residual, tol)
                                    << "trmm read storage it must not touch: " << route << " route, "
                                    << shape.rows << "x" << shape.cols << " batch " << shape.batch
                                    << " side=" << (side == Side::Left ? "Left" : "Right")
                                    << " uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper")
                                    << " trans=" << static_cast<int>(trans)
                                    << " diag=" << (diag == Diag::Unit ? "Unit" : "NonUnit");
                            }
                        }
                    }
                }
            }
        }
    };

    sweep("default");

    if constexpr (TestFixture::BackendType == Backend::CUDA) {
        ScopedEnvVar no_scratch("BATCHLAS_EXPAND_MAX_BYTES", "0");
        sweep("no-scratch");
    }
}