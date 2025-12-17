#include <gtest/gtest.h>
#include <blas/linalg.hh>
#include <blas/extra.hh>
#include <util/sycl-device-queue.hh>
#include "test_utils.hh"

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
                    

                    // compute C = trmm(A,B) with the current combination
                    trmm<Ba>(*(this->ctx), A.view(), B.view(), C.view(), T(1.0), side, uplo, trans, diag).wait();

                    // subtract the full matrix product from C to obtain the residual
                    if (side == Side::Right) {
                        gemm<Ba>(*(this->ctx), B.view(), A.view(), C.view(), T(1.0), T(-1.0),
                                 Transpose::NoTrans, trans).wait();
                    } else {
                        gemm<Ba>(*(this->ctx), A.view(), B.view(), C.view(), T(1.0), T(-1.0),
                                 trans, Transpose::NoTrans).wait();
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