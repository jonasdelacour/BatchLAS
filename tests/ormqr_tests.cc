#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/util/sycl-device-queue.hh>

using namespace batchlas;

template <typename T, Backend B>
struct OrmqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
#include "ormqr_verify.hh"
using OrmqrTestTypes = typename test_utils::backend_types<OrmqrConfig>::type;

template <typename Config>
class OrmqrTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(OrmqrTest, OrmqrTestTypes);

TYPED_TEST(OrmqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n);
    UnifiedVector<std::byte> ws_ormqr(ormqr_buffer_size(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
    (void)ormqr(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ormqr.to_span());
    this->ctx->wait();

    EXPECT_VERIFY(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q, Transpose::NoTrans));
    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q.view()));
}

TYPED_TEST(OrmqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(n * batch);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    UnifiedVector<std::byte> ws_ormqr(ormqr_buffer_size(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
    (void)ormqr(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ormqr.to_span());
    this->ctx->wait();

    EXPECT_VERIFY(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q, Transpose::NoTrans));
    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q.view()));
}

// The two routing regressions that lived here (an unmatched forced provider whose size and
// call disagreed; a forced blocked on complex Trans) are ported to ormqr_candidates_tests:
// BufferSizeIsTheChosenFamilysAndRuns and CanRunFalsePinsThrow.

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
