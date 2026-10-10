#include <gtest/gtest.h>

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <type_traits>

#include "test_utils.hh"
#include "ormqr_verify.hh"
#include "../src/ops/ormqr/vendor.hh"

using namespace batchlas;

namespace {

template <typename T, Backend B>
struct OrmqrBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// The GPU backend is the one this repo setup expects to be working, but the
// host backend runs the same blocked algorithm over different BLAS routes --
// the WY update's trmm among them -- so it is covered too wherever it is built
// rather than only when no GPU backend is.
#if BATCHLAS_HAS_CUDA_BACKEND
using OrmqrBlockedTestTypes = ::testing::Types<OrmqrBlockedConfig<float, Backend::CUDA>,
                                              OrmqrBlockedConfig<double, Backend::CUDA>,
                                              OrmqrBlockedConfig<std::complex<float>, Backend::CUDA>,
                                              OrmqrBlockedConfig<std::complex<double>, Backend::CUDA>
#if BATCHLAS_HAS_HOST_BACKEND
                                              ,
                                              OrmqrBlockedConfig<float, Backend::NETLIB>,
                                              OrmqrBlockedConfig<double, Backend::NETLIB>
#endif
                                              >;
#elif BATCHLAS_HAS_ROCM_BACKEND
using OrmqrBlockedTestTypes = ::testing::Types<OrmqrBlockedConfig<float, Backend::ROCM>,
                                              OrmqrBlockedConfig<double, Backend::ROCM>,
                                              OrmqrBlockedConfig<std::complex<float>, Backend::ROCM>,
                                              OrmqrBlockedConfig<std::complex<double>, Backend::ROCM>>;
#elif BATCHLAS_HAS_HOST_BACKEND
using OrmqrBlockedTestTypes = ::testing::Types<OrmqrBlockedConfig<float, Backend::NETLIB>,
                                              OrmqrBlockedConfig<double, Backend::NETLIB>>;
#else
// No valid GPU backend; provide a placeholder so TYPED_TEST_SUITE compiles.
using OrmqrBlockedTestTypes = ::testing::Types<OrmqrBlockedConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class OrmqrBlockedTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    Transpose trans_h() const {
        if constexpr (test_utils::is_complex<T>::value) {
            return Transpose::ConjTrans;
        }
        return Transpose::Trans;
    }
};

TYPED_TEST_SUITE(OrmqrBlockedTest, OrmqrBlockedTestTypes);

namespace {
inline int32_t get_block_size_or_default(int32_t def) {
    if (const char* p = std::getenv("BATCHLAS_ORMQR_BLOCK_SIZE")) {
        try {
            const int v = std::stoi(std::string(p));
            if (v > 0) return static_cast<int32_t>(v);
        } catch (...) {
        }
    }
    return def;
}
} // namespace

// Float keeps the old entrywise 1e-5 against the reference ormqr. One entry of Q column j off by d moves
// ||A0 - QR||_F / ||A0||_F by d ||R(j,:)|| / ||A0||_F, about d / sqrt(n) for an average row: 1.25e-6 at
// n = 64, 1/49 of the float bound 16 n u = 6.1e-5. 1/64 catches it wherever ||R(j,:)|| >= 0.76 of the
// average (j <= 45 of 64 on random input); measured need 0.007 of the unslacked bound.
template <typename T>
batchlas::verify::Slack recon_slack() {
    if constexpr (std::is_same_v<batchlas::verify::real_t<T>, float>)
        return {1.0 / 64, "the old 1e-5 entrywise tolerance vs the reference ormqr (1e-5/sqrt(64) = 1/49 of the bound)"};
    else
        return {1.0, "the library bound (double's old 1e-10 entrywise is looser)"};
}

// ormqr_blocked and the backend ormqr are each judged against Q formed on the host from the reflectors.
TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceSingle) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
        (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A.view(),
                                                                        Q_blk.view(),
                                                                        Side::Left,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q_blk.view()));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_ref, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_ref";
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceSingleTrans) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
        (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, this->trans_h(), tau.to_span()));
        (void)batchlas::blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, this->trans_h(), tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A.view(),
                                                                        Q_blk.view(),
                                                                        Side::Left,
                                                                        this->trans_h(),
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q_blk.view()));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_blk, this->trans_h()),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_ref, this->trans_h()),
                        recon_slack<T>())
        << "Q_ref";
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceRightSingle) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
        (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Right, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Right, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A.view(),
                                                                        Q_blk.view(),
                                                                        Side::Right,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Right, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q_blk.view()));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_ref, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_ref";
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceRightSingleTrans) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
        (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(
            batchlas::blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Right, this->trans_h(), tau.to_span()));
        (void)batchlas::blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Right, this->trans_h(), tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A.view(),
                                                                        Q_blk.view(),
                                                                        Side::Right,
                                                                        this->trans_h(),
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Right, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q_blk.view()));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_blk, this->trans_h()),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_ref, this->trans_h()),
                        recon_slack<T>())
        << "Q_ref";
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceBatched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 8;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    const Matrix<T, MatrixFormat::Dense> A0 = A;
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
        (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::blas::dispatch::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::blas::dispatch::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A.view(),
                                                                        Q_blk.view(),
                                                                        Side::Left,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q_blk.view()));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(A0, A, Q_ref, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_ref";
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
