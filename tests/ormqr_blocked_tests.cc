#include <gtest/gtest.h>

#include <blas/functions.hh>
#include <blas/linalg.hh>
#include <blas/matrix.hh>
#include <internal/ormqr_blocked.hh>
#include <util/sycl-device-queue.hh>

#include <cstdlib>
#include <type_traits>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename T, Backend B>
struct OrmqrBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// CUDA-only: this repo setup expects CUDA to be the working backend here.
#if BATCHLAS_HAS_CUDA_BACKEND
using OrmqrBlockedTestTypes = ::testing::Types<OrmqrBlockedConfig<float, Backend::CUDA>,
                                              OrmqrBlockedConfig<double, Backend::CUDA>,
                                              OrmqrBlockedConfig<std::complex<float>, Backend::CUDA>,
                                              OrmqrBlockedConfig<std::complex<double>, Backend::CUDA>>;
#else
using OrmqrBlockedTestTypes = ::testing::Types<>;
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

// This test is expected to FAIL until ormqr_blocked is fixed.
// It compares ormqr_blocked against the backend ormqr (CUSOLVER) implementation.
TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceSingle) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
        geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(backend::ormqr_vendor_buffer_size<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
        backend::ormqr_vendor<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
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
        ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    // Check orthonormality for the blocked output.
    Matrix<T, MatrixFormat::Dense> QtQ = Matrix<T, MatrixFormat::Dense>::Zeros(n, n, batch);
    gemm<B>(*this->ctx, Q_blk.view(), Q_blk.view(), QtQ.view(), T(1), T(0), this->trans_h(), Transpose::NoTrans);
    this->ctx->wait();

    auto r = QtQ.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const T expected = (i == j) ? T(1) : T(0);
            test_utils::expect_near(r[i * QtQ.ld() + j], expected);
        }
    }

    // Compare blocked vs reference.
    auto ref = Q_ref.data();
    auto blk = Q_blk.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            test_utils::expect_near(blk[i * Q_blk.ld() + j], ref[i * Q_ref.ld() + j]);
        }
    }
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceSingleTrans) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
        geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(backend::ormqr_vendor_buffer_size<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, this->trans_h(), tau.to_span()));
        backend::ormqr_vendor<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, this->trans_h(), tau.to_span(), ws_ref.to_span());
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
        ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    auto ref = Q_ref.data();
    auto blk = Q_blk.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            test_utils::expect_near(blk[i * Q_blk.ld() + j], ref[i * Q_ref.ld() + j]);
        }
    }
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceRightSingle) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
        geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(backend::ormqr_vendor_buffer_size<B>(*this->ctx, A.view(), Q_ref.view(), Side::Right, Transpose::NoTrans, tau.to_span()));
        backend::ormqr_vendor<B>(*this->ctx, A.view(), Q_ref.view(), Side::Right, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
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
        ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Right, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    auto ref = Q_ref.data();
    auto blk = Q_blk.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            test_utils::expect_near(blk[i * Q_blk.ld() + j], ref[i * Q_ref.ld() + j]);
        }
    }
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceRightSingleTrans) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
        geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(
            backend::ormqr_vendor_buffer_size<B>(*this->ctx, A.view(), Q_ref.view(), Side::Right, this->trans_h(), tau.to_span()));
        backend::ormqr_vendor<B>(*this->ctx, A.view(), Q_ref.view(), Side::Right, this->trans_h(), tau.to_span(), ws_ref.to_span());
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
        ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Right, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    auto ref = Q_ref.data();
    auto blk = Q_blk.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            test_utils::expect_near(blk[i * Q_blk.ld() + j], ref[i * Q_ref.ld() + j]);
        }
    }
}

TYPED_TEST(OrmqrBlockedTest, MatchesOrmqrReferenceBatched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;

    const int n = 64;
    const int batch = 8;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*symmetric=*/false, batch);
    UnifiedVector<T> tau(static_cast<size_t>(n) * static_cast<size_t>(batch));

    {
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size<B>(*this->ctx, A.view(), tau.to_span()));
        geqrf<B>(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(backend::ormqr_vendor_buffer_size<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
        backend::ormqr_vendor<B>(*this->ctx, A.view(), Q_ref.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
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
        ormqr_blocked<B, T>(*this->ctx, A.view(), Q_blk.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    // Compare blocked vs reference per batch item.
    auto ref = Q_ref.data();
    auto blk = Q_blk.data();
    for (int b = 0; b < batch; ++b) {
        const size_t off_ref = static_cast<size_t>(b) * static_cast<size_t>(Q_ref.stride());
        const size_t off_blk = static_cast<size_t>(b) * static_cast<size_t>(Q_blk.stride());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                test_utils::expect_near(blk[off_blk + static_cast<size_t>(i) * static_cast<size_t>(Q_blk.ld()) + static_cast<size_t>(j)],
                                        ref[off_ref + static_cast<size_t>(i) * static_cast<size_t>(Q_ref.ld()) + static_cast<size_t>(j)]);
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
