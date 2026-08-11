#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

#if BATCHLAS_HAS_HOST_BACKEND
template <typename Real>
UnifiedVector<double> netlib_ref_eigs_dense(const MatrixView<Real, MatrixFormat::Dense>& A) {
    const int n = A.rows();
    const int batch = A.batch_size();

    Queue ctx_cpu("cpu");
    auto A_d = A.template astype<double>();

    UnifiedVector<double> ref_eigs(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    const size_t ws_bytes = backend::syev_vendor_buffer_size<Backend::NETLIB, double>(
        ctx_cpu, A_d.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});
    backend::syev_vendor<Backend::NETLIB, double>(
        ctx_cpu, A_d.view(), ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span()).wait();
    ctx_cpu.wait();

    return ref_eigs;
}
#endif

#if BATCHLAS_HAS_HOST_BACKEND
template <typename Scalar>
UnifiedVector<typename base_type<Scalar>::type> netlib_ref_eigs_dense_native(const MatrixView<Scalar, MatrixFormat::Dense>& A) {
    using Real = typename base_type<Scalar>::type;

    Queue ctx_cpu("cpu");
    UnifiedVector<Real> ref_eigs(static_cast<std::size_t>(A.rows()) * static_cast<std::size_t>(A.batch_size()));
    const size_t ws_bytes = backend::syev_vendor_buffer_size<Backend::NETLIB, Scalar>(
        ctx_cpu, A, ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});
    backend::syev_vendor<Backend::NETLIB, Scalar>(
        ctx_cpu, A, ref_eigs.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws.to_span()).wait();
    ctx_cpu.wait();

    return ref_eigs;
}
#endif

template <typename T, Backend B>
struct SytrdBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
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

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdBlockedTestTypes = ::testing::Types<SytrdBlockedConfig<float, Backend::CUDA>, SytrdBlockedConfig<double, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SytrdBlockedTestTypes = ::testing::Types<SytrdBlockedConfig<float, Backend::ROCM>, SytrdBlockedConfig<double, Backend::ROCM>>;
#else
using SytrdBlockedTestTypes = ::testing::Types<SytrdBlockedConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SytrdBlockedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdBlockedTest, SytrdBlockedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 128;
    const int batch = 128;
    const int nb = 32;
    const double eig_tol = (std::is_same_v<Real, float> ? 10000.0 : 100.0) * test_utils::tolerance<double>();

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/789);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    // Validate a few representative batch items fully. (Validating all 128 would be expensive.)
    std::vector<int> batch_items;
    batch_items.push_back(0);
    if (batch > 1) batch_items.push_back(batch / 2);
    if (batch > 2) batch_items.push_back(batch - 1);

    Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
    Tmat.view().fill_tridiag(*this->ctx, e, d, e).wait();

#if BATCHLAS_HAS_HOST_BACKEND
    const auto eig_ref = netlib_ref_eigs_dense(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense(Tmat.view());

    for (int b : batch_items) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = eig_ref[base + static_cast<std::size_t>(i)];
            double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            if constexpr (std::is_same_v<Real, float>) {
                err_tol = std::max(err_tol, 3e-6);
            }
            EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
#endif
}

// The blocked trailing update (A22 -= V W^H + W V^H) only runs when the trailing
// block is wider than 128; below that sytrd_blocked takes update_vw_lower_small
// instead. Every other case in this file is n <= 128, so none of them reach it --
// which matters now that the trailing update defaults to syr2k rather than the
// GEMM pair on CUDA/float.
//
// n = 320 with nb = 32 leaves n2 = 288 on the first panel and stays above 128 for
// six of them. Both routes are run against each other as well as against the
// reference, so a divergence points at the route rather than at sytrd generally.
TYPED_TEST(SytrdBlockedTest, TrailingUpdateRoutesAgree) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 320;
    const int batch = 8;
    const int nb = 32;
    const double eig_tol = (std::is_same_v<Real, float> ? 10000.0 : 100.0) * test_utils::tolerance<double>();

    Matrix<Real, MatrixFormat::Dense> A0 =
        Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/20260806);

    // Returns the tridiagonal (d, e) produced under the given trailing-update route.
    auto run_route = [&](const char* route) {
        ScopedEnvVar mode("BATCHLAS_SYTRD_TRAILING_UPDATE", route);

        Matrix<Real, MatrixFormat::Dense> A = A0;
        Vector<Real> d(n, batch);
        Vector<Real> e(n - 1, batch);
        Vector<Real> tau(n - 1, batch);

        const size_t ws_bytes =
            sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
        UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

        sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

        Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
        Tmat.view().fill_tridiag(*this->ctx, e, d, e).wait();
        return Tmat;
    };

    Matrix<Real, MatrixFormat::Dense> T_gemm = run_route("gemm");
    Matrix<Real, MatrixFormat::Dense> T_syr2k = run_route("syr2k");

#if BATCHLAS_HAS_HOST_BACKEND
    const auto eig_ref = netlib_ref_eigs_dense(A0.view());
    const auto eig_gemm = netlib_ref_eigs_dense(T_gemm.view());
    const auto eig_syr2k = netlib_ref_eigs_dense(T_syr2k.view());

    // Both routes perform the same rank-2 update, but sum it in a different
    // order, so they agree only to rounding. Judge them by how far each lands
    // from the reference spectrum rather than from each other: the question is
    // whether syr2k is as accurate as the GEMM pair, not whether it is
    // bit-identical to it.
    double worst_gemm = 0.0;
    double worst_syr2k = 0.0;
    double spectral_radius = 0.0;
    for (int b = 0; b < batch; ++b) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const std::size_t ix = base + static_cast<std::size_t>(i);
            spectral_radius = std::max(spectral_radius, std::abs(eig_ref[ix]));
            worst_gemm = std::max(worst_gemm, std::abs(eig_gemm[ix] - eig_ref[ix]));
            worst_syr2k = std::max(worst_syr2k, std::abs(eig_syr2k[ix] - eig_ref[ix]));
        }
    }

    // Backward error of a Householder tridiagonalisation is O(n * eps * ||A||).
    const double eps = static_cast<double>(std::numeric_limits<Real>::epsilon());
    const double backward_err_tol = 4.0 * static_cast<double>(n) * eps * spectral_radius;
    const double err_tol = std::max(eig_tol * std::max(1.0, spectral_radius), backward_err_tol);

    EXPECT_LT(worst_syr2k, err_tol)
        << "syr2k trailing update lost accuracy: worst eigenvalue error " << worst_syr2k
        << " (GEMM route: " << worst_gemm << ", spectral radius " << spectral_radius << ")";

    // The substitution is only justified if it is no worse than what it
    // replaced. This is the assertion with the teeth: the backward-error bound
    // above is ~1000x looser than the error either route actually incurs
    // (measured 2.6e-6 against a 3.2e-3 bound at n=320, float), so on its own it
    // would pass almost anything. A factor of 4 plus a few ulps of the spectrum
    // leaves room for the different summation order and nothing else.
    const double route_tol = 4.0 * worst_gemm + 8.0 * eps * spectral_radius;
    EXPECT_LT(worst_syr2k, route_tol)
        << "syr2k route is materially less accurate than the GEMM pair: "
        << worst_syr2k << " vs " << worst_gemm;
#endif
}

TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower33) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 33;
    const int batch = 64;
    const int nb = 8;
    const double eig_tol = (std::is_same_v<Real, float> ? 10000.0 : 100.0) * test_utils::tolerance<double>();

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/1337);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    std::vector<int> batch_items;
    batch_items.push_back(0);
    if (batch > 1) batch_items.push_back(batch / 2);
    if (batch > 2) batch_items.push_back(batch - 1);

    Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
    Tmat.view().fill_tridiag(*this->ctx, e, d, e).wait();

#if BATCHLAS_HAS_HOST_BACKEND
    const auto eig_ref = netlib_ref_eigs_dense(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense(Tmat.view());

    for (int b : batch_items) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = eig_ref[base + static_cast<std::size_t>(i)];
            double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            if constexpr (std::is_same_v<Real, float>) {
                err_tol = std::max(err_tol, 3e-6);
            }
            EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
#endif
}
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
TEST(SytrdBlockedFloatCudaTest, Syr2kTrailingUpdateMatchesNetlibReference) {
    using Real = float;
    constexpr Backend B = Backend::CUDA;

    const double eig_tol = 10000.0 * test_utils::tolerance<double>();
    const double experiment_floor = 2.5e-6;
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "SYTRD SYR2K trailing-update test requires a GPU device";
    }

    for (const int n : {192, 256}) {
        const int batch = 32;
        const int nb = 32;

        auto ctx = std::make_shared<Queue>(Device("gpu"), true);

        Matrix<Real, MatrixFormat::Dense> A0 =
            Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/1000 + n);
        Matrix<Real, MatrixFormat::Dense> A = A0;
        Vector<Real> d(n, batch);
        Vector<Real> e(n - 1, batch);
        Vector<Real> tau(n - 1, batch);

        const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*ctx, A.view(), d, e, tau, Uplo::Lower, nb);
        UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

        {
            ScopedEnvVar trailing_update("BATCHLAS_SYTRD_TRAILING_UPDATE", "syr2k");
            sytrd_blocked<B, Real>(*ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();
        }

        std::vector<int> batch_items{0};
        if (batch > 1) batch_items.push_back(batch / 2);
        if (batch > 2) batch_items.push_back(batch - 1);

        Matrix<Real, MatrixFormat::Dense> Tmat = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, batch);
        Tmat.view().fill_tridiag(*ctx, e, d, e).wait();

        const auto eig_ref = netlib_ref_eigs_dense(A0.view());
        const auto eig_trd = netlib_ref_eigs_dense(Tmat.view());

        for (int b : batch_items) {
            const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
            for (int i = 0; i < n; ++i) {
                const double ref = eig_ref[base + static_cast<std::size_t>(i)];
                const double err_tol = std::max(eig_tol * std::max(1.0, std::abs(ref)), experiment_floor);
                EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                    << "eigenvalue mismatch at n=" << n << ", i=" << i << ", batch=" << b;
            }
        }
    }
}

TEST(SytrdBlockedComplexDoubleCudaTest, TridiagonalSpectrumMatchesNetlibReference) {
    using Scalar = std::complex<double>;
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = Backend::CUDA;

    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "Complex<double> SYTRD blocked test requires a GPU device";
    }

    const int n = 96;
    const int batch = 16;
    const int nb = 32;
    const double eig_tol = 100.0 * test_utils::tolerance<double>();

    auto ctx = std::make_shared<Queue>(Device("gpu"), true);

    Matrix<Scalar, MatrixFormat::Dense> A0 =
        Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/4242);
    Matrix<Scalar, MatrixFormat::Dense> A = A0;
    Vector<Scalar> d(n, batch);
    Vector<Scalar> e(n - 1, batch);
    Vector<Scalar> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    Matrix<Scalar, MatrixFormat::Dense> Tmat = Matrix<Scalar, MatrixFormat::Dense>::Zeros(n, n, batch);
    auto Tview = Tmat.view();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            Tview.template at<MatrixFormat::Dense>(i, i, b) = Scalar(d(i, b).real(), 0.0);
            if (i < n - 1) {
                const Scalar sub = e(i, b);
                Tview.template at<MatrixFormat::Dense>(i + 1, i, b) = sub;
                Tview.template at<MatrixFormat::Dense>(i, i + 1, b) = std::conj(sub);
            }
        }
    }

    const auto eig_ref = netlib_ref_eigs_dense_native(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense_native(Tmat.view());

    std::vector<int> batch_items{0};
    if (batch > 1) batch_items.push_back(batch / 2);
    if (batch > 2) batch_items.push_back(batch - 1);

    for (int b : batch_items) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = eig_ref[base + static_cast<std::size_t>(i)];
            const double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            EXPECT_NEAR(eig_trd[base + static_cast<std::size_t>(i)], ref, err_tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
}
#endif

#if BATCHLAS_HAS_CUDA_BACKEND && BATCHLAS_HAS_HOST_BACKEND
// The grid LATRD path (BATCHLAS_LATRD_IMPL=grid) only engages when
// MAX_COMPUTE_UNITS / batch >= 2, i.e. in the small-batch regime that no other
// test in this file covers. It also runs the same shapes through the legacy
// path so a divergence is attributable.
template <typename Scalar>
void run_latrd_grid_case(int n, int batch, int nb, const char* impl, double eig_tol_scale) {
    using Real = typename base_type<Scalar>::type;
    constexpr Backend B = Backend::CUDA;

    auto ctx = std::make_shared<Queue>(Device("gpu"), true);

    Matrix<Scalar, MatrixFormat::Dense> A0 =
        Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/4242 + n + batch);
    Matrix<Scalar, MatrixFormat::Dense> A = A0;
    Vector<Scalar> d(n, batch);
    Vector<Scalar> e(n - 1, batch);
    Vector<Scalar> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    {
        ScopedEnvVar latrd_impl("BATCHLAS_LATRD_IMPL", impl);
        sytrd_blocked<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();
    }
    ctx->wait();

    Matrix<Scalar, MatrixFormat::Dense> Tmat = Matrix<Scalar, MatrixFormat::Dense>::Zeros(n, n, batch);
    constexpr bool kIsComplex = !std::is_same_v<Scalar, Real>;
    if constexpr (kIsComplex) {
        auto Tview = Tmat.view();
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                Tview.template at<MatrixFormat::Dense>(i, i, b) = Scalar(d(i, b).real(), Real(0));
                if (i < n - 1) {
                    const Scalar sub = e(i, b);
                    Tview.template at<MatrixFormat::Dense>(i + 1, i, b) = sub;
                    Tview.template at<MatrixFormat::Dense>(i, i + 1, b) = std::conj(sub);
                }
            }
        }
        ctx->wait();
    } else {
        Tmat.view().fill_tridiag(*ctx, e, d, e).wait();
    }

    const auto eig_ref = netlib_ref_eigs_dense_native<Scalar>(A0.view());
    const auto eig_trd = netlib_ref_eigs_dense_native<Scalar>(Tmat.view());

    const double eig_tol = eig_tol_scale * test_utils::tolerance<double>();
    for (int b = 0; b < batch; ++b) {
        const std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(n);
        for (int i = 0; i < n; ++i) {
            const double ref = static_cast<double>(eig_ref[base + static_cast<std::size_t>(i)]);
            double err_tol = eig_tol * std::max(1.0, std::abs(ref));
            if constexpr (std::is_same_v<Real, float>) {
                // Single precision tridiagonalization of an n=256 matrix loses
                // this much on the legacy path too; the floor is not specific
                // to the grid path.
                err_tol = std::max(err_tol, 3e-4);
            }
            ASSERT_NEAR(static_cast<double>(eig_trd[base + static_cast<std::size_t>(i)]), ref, err_tol)
                << "impl=" << impl << " n=" << n << " batch=" << batch
                << " eigenvalue mismatch at i=" << i << ", batch item=" << b;
        }
    }
}

TEST(SytrdBlockedLatrdGridCudaTest, SmallBatchMatchesNetlibReference) {
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }

    for (const int batch : {1, 2, 8}) {
        for (const int n : {65, 96, 129, 256}) {
            for (const int nb : {8, 16, 32}) {
                for (const char* impl : {"grid", "legacy"}) {
                    run_latrd_grid_case<float>(n, batch, nb, impl, 20000.0);
                    run_latrd_grid_case<double>(n, batch, nb, impl, 200.0);
                }
            }
        }
    }
}

// n=1024, batch=1 is the regime the grid path targets: 32 work-groups of 32
// work-items per matrix instead of a single work-group.
TEST(SytrdBlockedLatrdGridCudaTest, LargeNBatchOneMatchesNetlibReference) {
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }
    for (const char* impl : {"grid", "legacy"}) {
        run_latrd_grid_case<double>(1024, 1, 32, impl, 4000.0);
    }
}

TEST(SytrdBlockedLatrdGridCudaTest, SmallBatchComplexMatchesNetlibReference) {
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }

    for (const int batch : {1, 8}) {
        for (const int n : {96, 257}) {
            run_latrd_grid_case<std::complex<double>>(n, batch, 32, "grid", 200.0);
        }
    }
}

TEST(SytrdBlockedLatrdGridCudaTest, GridMatchesLegacyTridiagonal) {
    using Scalar = double;
    constexpr Backend B = Backend::CUDA;
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }

    auto ctx = std::make_shared<Queue>(Device("gpu"), true);
    // n=1024/batch=1 is the target regime for the grid path (G == 32 groups of
    // 32 work-items per matrix) and also the deadlock smoke test.
    for (const int n : {96, 129, 1024}) {
        for (const int batch : (n >= 1024 ? std::vector<int>{1, 8} : std::vector<int>{1, 4})) {
            for (const int nb : (n >= 1024 ? std::vector<int>{32} : std::vector<int>{8, 16, 32})) {
                for (const int seed : {456, 789}) {
                    Matrix<Scalar, MatrixFormat::Dense> A0 =
                        Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, seed);
                    UnifiedVector<Scalar> ds[2], es[2];
                    for (int k = 0; k < 2; ++k) {
                        Matrix<Scalar, MatrixFormat::Dense> A = A0;
                        Vector<Scalar> d(n, batch), e(n - 1, batch), tau(n - 1, batch);
                        const size_t wsb = sytrd_blocked_buffer_size<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, nb);
                        // Deliberately NOT zero-initialized: sytrd's W workspace
                        // comes from a shared BumpAllocator in syev_blocked, so
                        // any read of an unwritten W entry must be caught here.
                        UnifiedVector<std::byte> ws(wsb, std::byte{0x7f});
                        {
                            ScopedEnvVar impl("BATCHLAS_LATRD_IMPL", k == 0 ? "legacy" : "grid");
                            sytrd_blocked<B, Scalar>(*ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();
                        }
                        ctx->wait();
                        ds[k] = UnifiedVector<Scalar>(static_cast<std::size_t>(n) * batch);
                        es[k] = UnifiedVector<Scalar>(static_cast<std::size_t>(n - 1) * batch);
                        for (int b = 0; b < batch; ++b) {
                            for (int i = 0; i < n; ++i) ds[k][b * n + i] = d(i, b);
                            for (int i = 0; i < n - 1; ++i) es[k][b * (n - 1) + i] = e(i, b);
                        }
                    }
                    // The grid path reduces per work-group and then combines the
                    // G partials, so rounding differs from the legacy single
                    // group tree reduction; the difference accumulates over the
                    // n sequential reflector steps. Scale with n accordingly.
                    const double elem_tol = 1e-11 * n;
                    for (std::size_t i = 0; i < ds[0].size(); ++i) {
                        ASSERT_NEAR(ds[1][i], ds[0][i], elem_tol * std::max(1.0, std::abs(ds[0][i])))
                            << "d mismatch n=" << n << " batch=" << batch << " nb=" << nb
                            << " seed=" << seed << " i=" << i;
                    }
                    for (std::size_t i = 0; i < es[0].size(); ++i) {
                        ASSERT_NEAR(std::abs(es[1][i]), std::abs(es[0][i]),
                                    elem_tol * std::max(1.0, std::abs(es[0][i])))
                            << "e mismatch n=" << n << " batch=" << batch << " nb=" << nb
                            << " seed=" << seed << " i=" << i;
                    }
                }
            }
        }
    }
}

// End-to-end: syev_blocked must produce the same spectrum whichever LATRD
// implementation runs underneath.
TEST(SytrdBlockedLatrdGridCudaTest, SyevBlockedSpectrumMatchesLegacy) {
    using Scalar = double;
    constexpr Backend B = Backend::CUDA;
    Queue probe;
    if (probe.device().type != DeviceType::GPU) GTEST_SKIP();
    auto ctx = std::make_shared<Queue>(Device("gpu"), true);

    for (const int batch : {1, 16}) {
        for (const auto jobz : {JobType::NoEigenVectors, JobType::EigenVectors}) {
            const int n = 96;
            Matrix<Scalar, MatrixFormat::Dense> A0 =
                Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 456);
            UnifiedVector<Scalar> W[2];
            for (int k = 0; k < 2; ++k) {
                Matrix<Scalar, MatrixFormat::Dense> A = A0;
                W[k] = UnifiedVector<Scalar>(static_cast<std::size_t>(n) * batch);
                StedcParams<Scalar> params;
                params.recursion_threshold = 32;
                UnifiedVector<std::byte> ws(
                    syev_blocked_buffer_size<B, Scalar>(*ctx, A.view(), jobz, Uplo::Lower, params));
                ScopedEnvVar impl("BATCHLAS_LATRD_IMPL", k == 0 ? "legacy" : "grid");
                syev_blocked<B, Scalar>(*ctx, A.view(), W[k].to_span(), jobz, Uplo::Lower,
                                        ws.to_span(), params).wait();
                ctx->wait();
            }
            double maxdiff = 0;
            for (std::size_t i = 0; i < W[0].size(); ++i)
                maxdiff = std::max(maxdiff, std::abs(W[1][i] - W[0][i]));
            EXPECT_LT(maxdiff, 1e-9) << "batch=" << batch
                                     << " jobz=" << (jobz == JobType::EigenVectors ? "EV" : "NoEV");
        }
    }
}

TEST(SytrdBlockedLatrdGridCudaTest, ForcedGroupCountsAgree) {
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }

    // Exercise group counts / work-group sizes the heuristic would not pick,
    // including partitions where trailing work-groups end up empty.
    for (const char* groups : {"2", "3", "7", "16", "64"}) {
        for (const char* wgs : {"32", "128"}) {
            ScopedEnvVar g("BATCHLAS_LATRD_GRID_GROUPS", groups);
            ScopedEnvVar w("BATCHLAS_LATRD_GRID_WG", wgs);
            run_latrd_grid_case<double>(129, 1, 32, "grid", 200.0);
        }
    }
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
