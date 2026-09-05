#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cstdlib>
#include <string>

using namespace batchlas;

namespace {
class ScopedEnvVar {
public:
    ScopedEnvVar(const char* name, const char* value) : name_(name) {
        if (const char* old = std::getenv(name_)) { had_old_ = true; old_value_ = old; }
        ::setenv(name_, value, 1);
    }
    ~ScopedEnvVar() {
        if (had_old_) ::setenv(name_, old_value_.c_str(), 1);
        else ::unsetenv(name_);
    }
private:
    const char* name_;
    bool had_old_ = false;
    std::string old_value_;
};
} // namespace

template <typename T, Backend B>
struct OrmqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
using OrmqrTestTypes = typename test_utils::backend_types<OrmqrConfig>::type;

template <typename Config>
class OrmqrTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(OrmqrTest, OrmqrTestTypes);

TYPED_TEST(OrmqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n);
    UnifiedVector<std::byte> ws_ormqr(ormqr_buffer_size(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
    ormqr(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ormqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n);
    gemm(*this->ctx, Q.view(), Q.view(), Result.view(), {.transA = this->trans});
    this->ctx->wait();

    auto r = Result.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T expected = (i == j) ? T(1) : T(0);
            test_utils::assert_near(r[i * Result.ld() + j], expected);
        }
    }
}

TYPED_TEST(OrmqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
    UnifiedVector<T> tau(n * batch);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n, batch);
    UnifiedVector<std::byte> ws_ormqr(ormqr_buffer_size(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span()));
    ormqr(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span(), ws_ormqr.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Result(n, n, batch);
    gemm(*this->ctx, Q.view(), Q.view(), Result.view(), {.transA = this->trans});
    this->ctx->wait();

    auto r = Result.data();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                T expected = (i == j) ? T(1) : T(0);
                test_utils::assert_near(r[b * Result.stride() + i * Result.ld() + j], expected);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Routing regressions.
//
// Both of these were live defects in choose_ormqr_provider, which returned a
// forced provider without ever checking it against ormqr_supports_blocked.
// ---------------------------------------------------------------------------

// A forced provider that names NEITHER of ormqr's two real routes.
// `cta`, `two_stage` and `jacobi` all parse but are matched by no branch, so
// ormqr_dispatch fell into its `else` arm and ran on the vendor while
// ormqr_buffer_size_dispatch fell past its single `if` and returned the BLOCKED
// size. Sizing a workspace with the public ormqr_buffer_size and handing it to
// the public ormqr therefore raised "insufficient workspace for chosen
// provider" -- from the call it had just been sized for.
//
// The property asserted is that the two AGREE, checked against whichever route
// actually resolves. Pinning a particular route here would be pinning the
// fallback rule rather than the defect: an unsupported forced route falls back
// to the ordinary automatic choice (see route_resolve.hh), which for ormqr on a
// GPU is the blocked path, and would have been a different answer earlier in
// this work package without the bug being any less fixed.
TYPED_TEST(OrmqrTest, BufferSizeAgreesWithDispatchUnderAnUnmatchedForcedRoute) {
    using T = typename TestFixture::ScalarType;
    const int n = 4;

    ScopedEnvVar force("BATCHLAS_ORMQR_PROVIDER", "cta");

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    Matrix<T, MatrixFormat::Dense> Q = Matrix<T, MatrixFormat::Dense>::Identity(n);
    const size_t unmatched_size = ormqr_buffer_size(
        *this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span());

    // The size reported must be the size the route that actually runs needs.
    // Stated against the two candidate sizes rather than as a bare no-throw,
    // which would be vacuous whenever the reported size happens to be the
    // larger of the two.
    //
    // The gap is not a near-miss: measured on this tree at n=4, RTX 4090, the
    // vendor wants 276480 bytes (float) and blocked 2560. The old code reported
    // the blocked size here and then demanded the vendor size, a deterministic
    // 108x under-size on every GPU instantiation. (The host instantiations
    // report 0 for both and neither pass nor fail on the size.)
    size_t vendor_size = 0;
    {
        ScopedEnvVar force_vendor("BATCHLAS_ORMQR_PROVIDER", "vendor");
        vendor_size = ormqr_buffer_size(
            *this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span());
    }
    size_t blocked_size = 0;
    {
        ScopedEnvVar force_blocked("BATCHLAS_ORMQR_PROVIDER", "blocked");
        blocked_size = ormqr_buffer_size(
            *this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans, tau.to_span());
    }
    const auto resolved = batchlas::blas::dispatch::detail::ormqr_route<T>(
        *this->ctx, A.view(), Side::Left, Transpose::NoTrans);
    const size_t expected_size =
        batchlas::dispatch::is_vendor(resolved) ? vendor_size : blocked_size;
    EXPECT_EQ(unmatched_size, expected_size)
        << "the reported size must be the one the resolved route needs"
        << " (resolved=" << batchlas::dispatch::to_string(resolved.origin)
        << ":" << batchlas::dispatch::to_string(resolved.algo)
        << ", vendor=" << vendor_size << ", blocked=" << blocked_size << ")";

    UnifiedVector<std::byte> ws(unmatched_size);
    ASSERT_NO_THROW(ormqr(*this->ctx, A.view(), Q.view(), Side::Left, Transpose::NoTrans,
                          tau.to_span(), ws.to_span()));
    this->ctx->wait();

    // ...and the answer is still Q, not merely a call that did not throw.
    Matrix<T, MatrixFormat::Dense> Result(n, n);
    gemm(*this->ctx, Q.view(), Q.view(), Result.view(), {.transA = this->trans});
    this->ctx->wait();
    auto r = Result.data();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            test_utils::assert_near(r[i * Result.ld() + j], (i == j) ? T(1) : T(0));
        }
    }
}

// Forcing must not be able to select a route that cannot serve the shape.
// ormqr_supports_blocked excludes complex with a plain Trans, but the chooser
// returned the forced value unchecked and ormqr_dispatch's tail is
// `if (vendor) ... else blocked` -- so BATCHLAS_ORMQR_PROVIDER=blocked ran the
// blocked path on exactly the inputs that predicate exists to exclude.
//
// Asserted on the resolved route rather than by running it: the point is that
// the unsupported kernel is never selected, and executing the case would
// additionally depend on whether the vendor accepts a non-conjugate transpose
// for complex, which is a different question.
TYPED_TEST(OrmqrTest, ForcingCannotSelectAnUnsupportedRoute) {
    using T = typename TestFixture::ScalarType;
    namespace d = batchlas::dispatch;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);

    ScopedEnvVar force("BATCHLAS_ORMQR_PROVIDER", "blocked");
    const d::Route trans_route = batchlas::blas::dispatch::detail::ormqr_route<T>(
        *this->ctx, A.view(), Side::Left, Transpose::Trans);

    if constexpr (test_utils::is_complex<T>()) {
        EXPECT_TRUE(d::is_vendor(trans_route))
            << "complex with a plain Trans is unsupported by the blocked path";
    } else if (this->ctx->device().type == DeviceType::GPU) {
        EXPECT_TRUE(d::is_native(trans_route));
    }

    // The supported case is unaffected: forcing still gets what it asked for.
    const d::Route notrans_route = batchlas::blas::dispatch::detail::ormqr_route<T>(
        *this->ctx, A.view(), Side::Left, Transpose::NoTrans);
    if (this->ctx->device().type == DeviceType::GPU) {
        EXPECT_TRUE(d::is_native(notrans_route));
        EXPECT_EQ(notrans_route.algo, d::Algorithm::Blocked);
    } else {
        // The blocked path is GPU-only, so a host queue must yield regardless.
        EXPECT_TRUE(d::is_vendor(notrans_route));
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
