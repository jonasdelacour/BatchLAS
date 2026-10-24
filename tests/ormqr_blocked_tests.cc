// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/env.hh>
#include <batchlas/tuning_params.hh>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include <tests/test_utils.hh>
#include <tests/ormqr_verify.hh>
#include <src/ops/ormqr/vendor.hh>
#include <src/ops/ormqr/choice.hh>
#include <src/select/select.hh>

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
    if constexpr (std::is_same_v<batchlas::float_t<T>, float>)
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
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A, tau.to_span()));
        (void)geqrf(*this->ctx, A, tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::ops::ormqr::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::ops::ormqr::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A,
                                                                        Q_blk,
                                                                        Side::Left,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A, Q_blk, Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, (MatrixView<T, MatrixFormat::Dense>::orthogonality(*this->ctx, Q_blk)));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_ref, Transpose::NoTrans),
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
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A, tau.to_span()));
        (void)geqrf(*this->ctx, A, tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::ops::ormqr::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, this->trans_h(), tau.to_span()));
        (void)batchlas::ops::ormqr::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, this->trans_h(), tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A,
                                                                        Q_blk,
                                                                        Side::Left,
                                                                        this->trans_h(),
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A, Q_blk, Side::Left, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, (MatrixView<T, MatrixFormat::Dense>::orthogonality(*this->ctx, Q_blk)));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_blk, this->trans_h()),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_ref, this->trans_h()),
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
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A, tau.to_span()));
        (void)geqrf(*this->ctx, A, tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::ops::ormqr::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Right, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::ops::ormqr::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Right, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A,
                                                                        Q_blk,
                                                                        Side::Right,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A, Q_blk, Side::Right, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, (MatrixView<T, MatrixFormat::Dense>::orthogonality(*this->ctx, Q_blk)));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_ref, Transpose::NoTrans),
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
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A, tau.to_span()));
        (void)geqrf(*this->ctx, A, tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(
            batchlas::ops::ormqr::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Right, this->trans_h(), tau.to_span()));
        (void)batchlas::ops::ormqr::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Right, this->trans_h(), tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A,
                                                                        Q_blk,
                                                                        Side::Right,
                                                                        this->trans_h(),
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A, Q_blk, Side::Right, this->trans_h(), tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, (MatrixView<T, MatrixFormat::Dense>::orthogonality(*this->ctx, Q_blk)));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_blk, this->trans_h()),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_ref, this->trans_h()),
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
        UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A, tau.to_span()));
        (void)geqrf(*this->ctx, A, tau.to_span(), ws_geqrf.to_span());
        this->ctx->wait();
    }

    Matrix<T, MatrixFormat::Dense> Q_ref = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    Matrix<T, MatrixFormat::Dense> Q_blk = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);

    {
        UnifiedVector<std::byte> ws_ref(batchlas::ops::ormqr::detail::ormqr_vendor_buffer_size_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, Transpose::NoTrans, tau.to_span()));
        (void)batchlas::ops::ormqr::detail::ormqr_vendor_or_throw<B, T>(*this->ctx, A, Q_ref, Side::Left, Transpose::NoTrans, tau.to_span(), ws_ref.to_span());
        this->ctx->wait();
    }

    {
        const int32_t block_size = get_block_size_or_default(32);
        UnifiedVector<std::byte> ws_blk(ormqr_blocked_buffer_size<B, T>(*this->ctx,
                                                                        A,
                                                                        Q_blk,
                                                                        Side::Left,
                                                                        Transpose::NoTrans,
                                                                        tau.to_span(),
                                                                        block_size));
        (void)ormqr_blocked<B, T>(*this->ctx, A, Q_blk, Side::Left, Transpose::NoTrans, tau.to_span(), ws_blk.to_span(), block_size);
        this->ctx->wait();
    }

    EXPECT_VERIFY(T, batchlas::verify::Check::orthogonality, n, (MatrixView<T, MatrixFormat::Dense>::orthogonality(*this->ctx, Q_blk)));
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_blk, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_blk";
    EXPECT_VERIFY_SLACK(T, batchlas::verify::Check::factorization, n, test_utils::ormqr_q_error(*this->ctx, A0, A, Q_ref, Transpose::NoTrans),
                        recon_slack<T>())
        << "Q_ref";
}

// L2 batch chunking fixtures (docs/perf/qr.md#qr-l2-batch-chunking): k = 69 reflectors of order 73
// (three blocks of 32) at a padded ld and odd stride, factored by the public geqrf.
template <typename T>
struct ChunkProblem {
    static constexpr int nq = 73, k = 69, q = 37;
    int batch = 0, lda = nq + 5, sa = (nq + 5) * k + 11;
    UnifiedVector<T> a, tau;
    UnifiedVector<T*> ptrs;   // the vendor geqrf is a pointer-array API
    MatrixView<T, MatrixFormat::Dense> A() { return {a.data(), nq, k, lda, sa, batch, ptrs.data()}; }
    VectorView<T> tau_view() { return VectorView<T>(tau.data(), k, batch, 1, k); }
};

template <typename T>
ChunkProblem<T> make_chunk_problem(Queue& ctx, int batch, bool identical) {
    ChunkProblem<T> p;
    p.batch = batch;
    p.a = UnifiedVector<T>(size_t(p.sa) * batch, T(-7));
    p.tau = UnifiedVector<T>(size_t(p.k) * batch);
    p.ptrs = UnifiedVector<T*>(size_t(batch), nullptr);
    const auto R = Matrix<T, MatrixFormat::Dense>::Random(p.nq, p.k, false, identical ? 1 : batch, 4711u);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < p.k; ++j)
            for (int i = 0; i < p.nq; ++i) p.a[size_t(b) * p.sa + size_t(j) * p.lda + i] = R(i, j, identical ? 0 : b);
    const ScopedEnvVar whole("BATCHLAS_TUNE_L2_CHUNK_ITEMS", nullptr);
    auto A = p.A();
    UnifiedVector<std::byte> ws(std::max<size_t>(1, geqrf_buffer_size(ctx, A, p.tau.to_span())));
    (void)geqrf(ctx, A, p.tau.to_span(), ws.to_span());
    ctx.wait();
    return p;
}

// C (Left: nq x q, Right: q x nq) at its own padded ld and odd stride, random with a nonzero imaginary part.
template <typename T>
std::pair<UnifiedVector<T>, MatrixView<T, MatrixFormat::Dense>> make_chunk_c(int rows, int cols, int batch,
                                                                            unsigned seed, bool identical) {
    const int ldc = rows + 3, sc = ldc * cols + 7;
    UnifiedVector<T> c(size_t(sc) * batch, T(5));
    const auto R = Matrix<T, MatrixFormat::Dense>::Random(rows, cols, false, identical ? 1 : batch, seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) c[size_t(b) * sc + size_t(j) * ldc + i] = R(i, j, identical ? 0 : b);
    MatrixView<T, MatrixFormat::Dense> v(c.data(), rows, cols, ldc, sc, batch);
    return {std::move(c), v};
}

// A chunk forced to 7 over 23 items (7 + 7 + 9: the last part takes the remainder), both sides and
// both transposes, in order through the driver and out of order through the pinned facade. Each item
// is judged alone against op(Q) of its own reflectors: a tau or C sub-range at the wrong offset moves
// only the items past the first part.
TYPED_TEST(OrmqrBlockedTest, BlockedChunksApplyEachItemsOwnQ) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    const int batch = 23;
    const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "7");
    ASSERT_EQ(tuning::blocked_chunk_items(batch, 1, 1, 1), 7) << "the override did not reach the accessor";
    auto p = make_chunk_problem<T>(*this->ctx, batch, false);
    const auto A = p.A();
    const bool gpu = this->ctx->device().type == DeviceType::GPU;
    Queue ooo(*this->ctx, /*in_order=*/false);
    for (Side side : {Side::Left, Side::Right})
        for (Transpose tr : {Transpose::NoTrans, this->trans_h()})
            for (bool facade : {false, true}) {
                if (facade && !gpu) continue;   // the blocked pin is GPU-only (can_run)
                const int rows = side == Side::Left ? p.nq : p.q, cols = side == Side::Left ? p.q : p.nq;
                auto [c, C] = make_chunk_c<T>(rows, cols, batch, 99u + unsigned(side == Side::Left) * 7u + unsigned(tr), false);
                const UnifiedVector<T> c0(c);
                const MatrixView<T, MatrixFormat::Dense> C0(const_cast<T*>(c0.data()), rows, cols, C.ld(), C.stride(), batch);
                if (!facade) {
                    UnifiedVector<std::byte> ws(std::max<size_t>(
                        1, ormqr_blocked_buffer_size<B, T>(*this->ctx, A, C, side, tr, p.tau.to_span(), 32)));
                    (void)ormqr_blocked<B, T>(*this->ctx, A, C, side, tr, p.tau.to_span(), ws.to_span(), 32);
                    this->ctx->wait();
                } else {
                    const select::ScopedPin<ops::ormqr::OrmqrChoice> pin("ormqr", ops::ormqr::Blocked{});
                    UnifiedVector<std::byte> ws(std::max<size_t>(1, ormqr_buffer_size<B, T>(ooo, A, C, side, tr, p.tau.to_span())));
                    (void)ormqr<B, T>(ooo, A, C, side, tr, p.tau.to_span(), ws.to_span());
                    ooo.wait();
                }
                for (int b = 0; b < batch; ++b) {
                    const VectorView<T> tb(p.tau.data() + size_t(b) * p.k, p.k, 1, 1, p.k);
                    EXPECT_VERIFY(T, batchlas::verify::Check::blas, p.nq,
                                  test_utils::ormqr_apply_error<T>(*this->ctx, A.batch_range(b, 1), tb, C0.batch_range(b, 1),
                                                                   C.batch_range(b, 1), side, tr))
                        << (side == Side::Left ? "Left " : "Right ") << (tr == Transpose::NoTrans ? "NoTrans" : "ConjTrans")
                        << (facade ? ", facade out of order" : ", driver in order") << ": item " << b << " (part "
                        << std::min(b / 7, 2) << ")";
                }
                if (this->HasFailure()) return;
            }
}

// SATURATING: 1024 copies of one factor and one C in four chunks of 256; every item's result must be
// bit-identical to item 0's, so parts that overlap or skip items show.
TYPED_TEST(OrmqrBlockedTest, BlockedChunksAtSaturatingBatchAreBitIdentical) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    const int batch = 1024;
    const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "256");
    auto p = make_chunk_problem<T>(*this->ctx, batch, true);
    const auto A = p.A();
    for (Side side : {Side::Left, Side::Right}) {
        const int rows = side == Side::Left ? p.nq : p.q, cols = side == Side::Left ? p.q : p.nq;
        auto [c, C] = make_chunk_c<T>(rows, cols, batch, 321u, true);
        const UnifiedVector<T> c0(c);
        UnifiedVector<std::byte> ws(std::max<size_t>(
            1, ormqr_blocked_buffer_size<B, T>(*this->ctx, A, C, side, this->trans_h(), p.tau.to_span(), 32)));
        (void)ormqr_blocked<B, T>(*this->ctx, A, C, side, this->trans_h(), p.tau.to_span(), ws.to_span(), 32);
        this->ctx->wait();
        int wrong = 0, first = -1;
        for (int b = 1; b < batch; ++b) {
            bool same = true;
            for (int j = 0; j < cols && same; ++j)
                same = std::memcmp(c.data() + size_t(b) * C.stride() + size_t(j) * C.ld(), c.data() + size_t(j) * C.ld(),
                                   sizeof(T) * size_t(rows)) == 0;
            if (!same && wrong++ == 0) first = b;
        }
        EXPECT_EQ(wrong, 0) << (side == Side::Left ? "Left" : "Right") << ": items differing from item 0; first at " << first;
        const MatrixView<T, MatrixFormat::Dense> C0(const_cast<T*>(c0.data()), rows, cols, C.ld(), C.stride(), 1);
        const VectorView<T> t0(p.tau.data(), p.k, 1, 1, p.k);
        EXPECT_VERIFY(T, batchlas::verify::Check::blas, p.nq,
                      test_utils::ormqr_apply_error<T>(*this->ctx, A.batch_range(0, 1), t0, C0, C.batch_range(0, 1), side,
                                                       this->trans_h()));
    }
}

// Past the grid: one part of 65537 items puts 65537 work-groups in pack_v's grid z (65535 on CUDA).
// The chunk is forced above the batch, so only select::grid_capped_chunk splits it. Every item is a
// copy of item 0 (reflectors of order 17, k = 17, C 17 x 3 or 3 x 17) and must be bit-identical to it.
TYPED_TEST(OrmqrBlockedTest, BlockedPastTheGridLimitIsBitIdentical) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the grid-z ceiling is a GPU limit";
    const int batch = 65537, nq = 17, k = 17, q = 3, lda = nq + 2, sa = lda * k + 3;
    UnifiedVector<T> a(size_t(sa) * batch), tau(size_t(k) * batch);
    {
        const auto R = Matrix<T, MatrixFormat::Dense>::Random(nq, k, false, 1, 8121u);
        for (int j = 0; j < k; ++j)
            for (int i = 0; i < nq; ++i) a[size_t(j) * lda + i] = R(i, j, 0);
        MatrixView<T, MatrixFormat::Dense> A1(a.data(), nq, k, lda, sa, 1);
        UnifiedVector<std::byte> ws(std::max<size_t>(1, geqrf_buffer_size(*this->ctx, A1, tau.to_span())));
        (void)geqrf(*this->ctx, A1, tau.to_span(), ws.to_span());
        this->ctx->wait();
    }
    for (int b = 1; b < batch; ++b) {
        std::copy_n(a.begin(), size_t(sa), a.begin() + size_t(b) * sa);
        std::copy_n(tau.begin(), size_t(k), tau.begin() + size_t(b) * k);
    }
    const MatrixView<T, MatrixFormat::Dense> A(a.data(), nq, k, lda, sa, batch);
    const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "70000");
    for (Side side : {Side::Left, Side::Right}) {
        const int rows = side == Side::Left ? nq : q, cols = side == Side::Left ? q : nq;
        auto [c, C] = make_chunk_c<T>(rows, cols, batch, 331u, true);
        UnifiedVector<T> c0(size_t(C.stride()));
        std::copy_n(c.begin(), size_t(C.stride()), c0.begin());
        UnifiedVector<std::byte> ws(std::max<size_t>(
            1, ormqr_blocked_buffer_size<B, T>(*this->ctx, A, C, side, this->trans_h(), tau.to_span(), 32)));
        ASSERT_NO_THROW(((void)ormqr_blocked<B, T>(*this->ctx, A, C, side, this->trans_h(), tau.to_span(), ws.to_span(), 32)))
            << (side == Side::Left ? "Left" : "Right");
        this->ctx->wait();
        int wrong = 0, first = -1;
        for (int b = 1; b < batch; ++b) {
            const bool same = std::memcmp(c.data() + size_t(b) * C.stride(), c.data(), sizeof(T) * size_t(C.stride())) == 0;
            if (!same && wrong++ == 0) first = b;
        }
        EXPECT_EQ(wrong, 0) << (side == Side::Left ? "Left" : "Right") << ": items differing from item 0; first at " << first;
        const MatrixView<T, MatrixFormat::Dense> C0(const_cast<T*>(c0.data()), rows, cols, C.ld(), C.stride(), 1);
        const VectorView<T> t0(tau.data(), k, 1, 1, k);
        EXPECT_VERIFY(T, batchlas::verify::Check::blas, nq,
                      test_utils::ormqr_apply_error<T>(*this->ctx, A.batch_range(0, 1), t0, C0, C.batch_range(0, 1), side,
                                                       this->trans_h()));
    }
}

// sytrd sizes ormqr once at its largest operands and calls smaller sub-views on that workspace: the
// larger shape's chunked part holds less than this shape's one pass, so the facade must let the
// driver fit its part to what it holds instead of refusing the call.
TYPED_TEST(OrmqrBlockedTest, BlockedChunksFitAWorkspaceSizedAtLargerOperands) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the blocked pin is GPU-only (can_run)";
    const int batch = 23;
    const select::ScopedPin<ops::ormqr::OrmqrChoice> pin("ormqr", ops::ormqr::Blocked{});
    auto p = make_chunk_problem<T>(*this->ctx, batch, false);
    const auto A = p.A();
    auto [c, C] = make_chunk_c<T>(p.nq, p.q, batch, 555u, false);
    const UnifiedVector<T> c0(c);
    std::size_t ws_large = 0;
    {
        const ScopedEnvVar chunk("BATCHLAS_TUNE_L2_CHUNK_ITEMS", "3");
        const MatrixView<T, MatrixFormat::Dense> bigA(nullptr, p.nq + 64, p.k, p.nq + 64, (p.nq + 64) * p.k, batch);
        const MatrixView<T, MatrixFormat::Dense> bigC(nullptr, p.nq + 64, p.q + 32, p.nq + 64, (p.nq + 64) * (p.q + 32), batch);
        ws_large = ormqr_buffer_size<B, T>(*this->ctx, bigA, bigC, Side::Left, this->trans_h(), p.tau.to_span());
    }
    const std::size_t ws_own = ormqr_buffer_size<B, T>(*this->ctx, A, C, Side::Left, this->trans_h(), p.tau.to_span());
    ASSERT_LT(ws_large, ws_own)
        << "the larger operands' part must hold less than this call's single pass, or nothing is fitted";
    UnifiedVector<std::byte> ws(ws_large);
    auto run = [&] { (void)ormqr<B, T>(*this->ctx, A, C, Side::Left, this->trans_h(), p.tau.to_span(), ws.to_span()); };
    ASSERT_NO_THROW(run());
    this->ctx->wait();
    const MatrixView<T, MatrixFormat::Dense> C0(const_cast<T*>(c0.data()), p.nq, p.q, C.ld(), C.stride(), batch);
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, p.nq,
                  test_utils::ormqr_apply_error<T>(*this->ctx, A, p.tau_view(), C0, C, Side::Left, this->trans_h()));
}

} // namespace
