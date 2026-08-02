#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

// The band path accumulates rounding through two reductions (sy2sb + BANDR1)
// before STEDC ever runs, so it needs a looser tolerance than syev_blocked.
template <typename Real>
Real tol_eig_for() {
    if constexpr (std::is_same_v<Real, float>) return Real(5e-3f);
    return Real(1e-8);
}

// RAII env override so a failing assertion cannot leak state into later tests.
class ScopedEnv {
public:
    ScopedEnv(const char* key, const std::string& value) : key_(key) {
        const char* old = std::getenv(key);
        had_old_ = (old != nullptr);
        if (had_old_) old_ = old;
        setenv(key, value.c_str(), 1);
    }
    ~ScopedEnv() {
        if (had_old_) {
            setenv(key_, old_.c_str(), 1);
        } else {
            unsetenv(key_);
        }
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv& operator=(const ScopedEnv&) = delete;

private:
    const char* key_;
    bool had_old_ = false;
    std::string old_;
};

template <typename T, Backend B>
struct SyevBandConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// Reference eigenvalues from the CPU LAPACK path.
template <typename Scalar>
UnifiedVector<typename base_type<Scalar>::type> netlib_eigenvalues(
    Queue& ctx,
    const Matrix<Scalar, MatrixFormat::Dense>& A0,
    int n,
    int batch) {
    using Real = typename base_type<Scalar>::type;
    Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;
    UnifiedVector<Real> W(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    auto ws = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(
        ctx, A_ref.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower));
    syev<Backend::NETLIB>(ctx, A_ref.view(), W.to_span(), JobType::NoEigenVectors,
                          Uplo::Lower, ws.to_span())
        .wait();
    return W;
}

// Run syev_band and return its eigenvalues.
template <typename Scalar, Backend B>
UnifiedVector<typename base_type<Scalar>::type> band_eigenvalues(
    Queue& ctx,
    const Matrix<Scalar, MatrixFormat::Dense>& A0,
    int n,
    int batch,
    SyevBandParams params = SyevBandParams()) {
    using Real = typename base_type<Scalar>::type;
    Matrix<Scalar, MatrixFormat::Dense> A = A0;
    UnifiedVector<Real> W(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

    StedcParams<Real> sp;
    sp.recursion_threshold = 32;

    auto ws = UnifiedVector<std::byte>(syev_band_buffer_size<B, Scalar>(
        ctx, A.view(), JobType::NoEigenVectors, Uplo::Lower, sp, params));
    syev_band<B, Scalar>(ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower,
                         ws.to_span(), sp, params)
        .wait();
    return W;
}

template <typename Real>
void expect_eigenvalues_match(const UnifiedVector<Real>& got,
                              const UnifiedVector<Real>& want,
                              int n,
                              int batch,
                              Real tol,
                              const char* what) {
    // Scale the tolerance by the spectral radius: a relative test.
    for (int b = 0; b < batch; ++b) {
        Real scale = Real(1);
        for (int i = 0; i < n; ++i) {
            scale = std::max(scale, std::abs(want[i + b * n]));
        }
        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(got[i + b * n], want[i + b * n], tol * scale)
                << what << " at (i,b)= (" << i << "," << b << ")";
        }
    }
}

} // namespace

using SyevBandTestTypes = typename test_utils::backend_types<SyevBandConfig>::type;

template <typename Config>
class SyevBandTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevBandTest, SyevBandTestTypes);

// Core correctness: eigenvalues from the band pipeline must match LAPACK.
TYPED_TEST(SyevBandTest, EigenvaluesMatchNetlib) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 128;
    const int batch = 4;

    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 4242);
    auto W_ref = netlib_eigenvalues<Scalar>(*this->ctx, A0, n, batch);
    auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch);

    expect_eigenvalues_match(W_band, W_ref, n, batch, tol_eig_for<Real>(), "syev_band");
}

// Small n exercises the kd clamp and the short-sweep schedule.
TYPED_TEST(SyevBandTest, EigenvaluesMatchNetlibSmall) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    for (int n : {2, 5, 17, 33}) {
        const int batch = 3;
        auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 100 + n);
        auto W_ref = netlib_eigenvalues<Scalar>(*this->ctx, A0, n, batch);
        auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch);
        expect_eigenvalues_match(W_band, W_ref, n, batch, tol_eig_for<Real>(), "syev_band small");
    }
}

// The result must not depend on kd: every kd gives the same spectrum.
TYPED_TEST(SyevBandTest, SpectrumIsInvariantToBandwidth) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 96;
    const int batch = 2;
    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 777);
    auto W_ref = netlib_eigenvalues<Scalar>(*this->ctx, A0, n, batch);

    for (int kd : {4, 8, 16, 32}) {
        SyevBandParams p;
        p.kd = kd;
        auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch, p);
        expect_eigenvalues_match(W_band, W_ref, n, batch, tol_eig_for<Real>(),
                                 (std::string("kd=") + std::to_string(kd)).c_str());
    }
}

// ...nor on the chase block size nb.
TYPED_TEST(SyevBandTest, SpectrumIsInvariantToBlockSize) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 96;
    const int batch = 2;
    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 888);
    auto W_ref = netlib_eigenvalues<Scalar>(*this->ctx, A0, n, batch);

    for (int nb : {1, 4, 16, 32}) {
        SyevBandParams p;
        p.kd = 16;
        p.bandr_explicit = true;
        p.bandr.d_seq = {0};
        p.bandr.block_size_seq = {nb};
        p.bandr.max_sweeps = -1;
        p.bandr.kd_work = 0;
        auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch, p);
        expect_eigenvalues_match(W_band, W_ref, n, batch, tol_eig_for<Real>(),
                                 (std::string("nb=") + std::to_string(nb)).c_str());
    }
}

// The env-var tuning knobs must be honoured and must not change the answer.
TYPED_TEST(SyevBandTest, EnvOverridesAreHonoured) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 96;
    const int batch = 2;
    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 999);
    auto W_ref = netlib_eigenvalues<Scalar>(*this->ctx, A0, n, batch);

    ScopedEnv kd_env("BATCHLAS_SYEV_BAND_KD", "12");
    ScopedEnv nb_env("BATCHLAS_SYEV_BAND_NB", "8");
    auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch);
    expect_eigenvalues_match(W_band, W_ref, n, batch, tol_eig_for<Real>(), "env override");
}

// Eigenvectors are not representable through the band path; the API must say so
// rather than returning silently wrong vectors.
TYPED_TEST(SyevBandTest, EigenvectorsAreRejected) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 32;
    auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, 1, 5);
    UnifiedVector<Real> W(static_cast<std::size_t>(n));

    // Lambdas keep the comma in `<B, Scalar>` out of the macro argument list.
    auto query_size = [&] {
        (void)syev_band_buffer_size<B, Scalar>(*this->ctx, A.view(), JobType::EigenVectors,
                                               Uplo::Lower);
    };
    auto run_solver = [&] {
        syev_band<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower,
                             Span<std::byte>());
    };

    EXPECT_THROW(query_size(), std::invalid_argument);
    EXPECT_THROW(run_solver(), std::invalid_argument);
}

// syev_band must agree with the pipeline it is meant to replace, not just with
// LAPACK -- this is the comparison the benchmark reports speedups against.
TYPED_TEST(SyevBandTest, AgreesWithSyevBlocked) {
    using Scalar = typename TestFixture::ScalarType;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 160;
    const int batch = 4;
    auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 31337);

    UnifiedVector<Real> W_blk(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    {
        Matrix<Scalar, MatrixFormat::Dense> A_blk = A0;
        StedcParams<Real> sp;
        sp.recursion_threshold = 32;
        auto ws = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(
            *this->ctx, A_blk.view(), JobType::NoEigenVectors, Uplo::Lower, sp));
        syev_blocked<B, Scalar>(*this->ctx, A_blk.view(), W_blk.to_span(),
                                JobType::NoEigenVectors, Uplo::Lower, ws.to_span(), sp)
            .wait();
    }

    auto W_band = band_eigenvalues<Scalar, B>(*this->ctx, A0, n, batch);
    expect_eigenvalues_match(W_band, W_blk, n, batch, tol_eig_for<Real>(), "band vs blocked");
}
