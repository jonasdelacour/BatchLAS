#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "test_utils.hh"
#include "eigen_verify.hh"

using namespace batchlas;

namespace {

// The tridiagonalization's backward error seen through its spectrum: max |eig(T) - eig(A0)| / ||A0||_2
// over `items` (default: first, middle, last), both from LAPACKE in double. T's spectrum comes from
// (Re d, |e|): a unitary diagonal similarity makes a Hermitian tridiagonal's spectrum the real one's.
template <typename Scalar>
double spectrum_error(const MatrixView<Scalar, MatrixFormat::Dense>& A0, Vector<Scalar>& d, Vector<Scalar>& e,
                      std::span<const int> items = {}) {
    const int n = A0.rows();
    const int batch = A0.batch_size();
    const std::vector<int> picked = items.empty() ? verify::default_items(batch) : std::vector<int>(items.begin(), items.end());
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch));
    std::vector<double> got(static_cast<std::size_t>(n) * batch, 0.0);
    double scale = 0;
    for (int b : picked) {
        auto a = verify::copy_item(A0, b);
        std::vector<double> dd(n), ee(std::max(0, n - 1));
        for (int i = 0; i < n; ++i) dd[i] = std::real(verify::up(d(i, b)));
        for (int i = 0; i + 1 < n; ++i) ee[i] = verify::abs(verify::up(e(i, b)));
        if (!verify::eigenvalues(n, a, ref[static_cast<std::size_t>(b)]) || !verify::tridiagonal_eigenvalues(dd, ee))
            return std::numeric_limits<double>::quiet_NaN();
        std::copy(dd.begin(), dd.end(), got.begin() + static_cast<std::ptrdiff_t>(b) * n);
        for (double l : ref[static_cast<std::size_t>(b)]) scale = verify::nanmax(scale, std::fabs(l));
    }
    return verify::values_error(VectorView<double>(got.data(), n, batch), ref, scale, picked);
}

// float only. The old float bounds, 1e4 x test_utils::tolerance<double>() = 1e-6 relative per value
// (floors 2.5e-6 / 3e-6 / 3e-4 absolute), sat near n u ||A||_2, far under the kind's 32 n u; these
// keep that power. double keeps the kind: its old 1e-8 relative was looser.
template <typename Real>
std::optional<verify::Slack> float_slack(verify::Slack s) {
    if constexpr (std::is_same_v<Real, float>) return s;
    return std::nullopt;
}
// factor = 1e-6 / (32 n u); measured float errors are 0.1-0.2 of these.
const verify::Slack kSlackN128{0.004, "old float bound 1e-6 relative (1e4 x test_utils::tolerance<double>), n = 128"};
const verify::Slack kSlackN33{0.016, "old float bound 1e-6 relative (1e4 x test_utils::tolerance<double>), n = 33"};
const verify::Slack kSlackN192{0.0027, "old float bound 1e-6 relative (1e4 x test_utils::tolerance<double>), n = 192"};
const verify::Slack kSlackN256{0.002, "old float bound 1e-6 relative (1e4 x test_utils::tolerance<double>), n = 256"};
const verify::Slack kSlackLatrd{0.05, "old float floor 3e-4 absolute over n = 65..256; measured <= 0.0015"};

// spectrum_error at Check::values (n = the order); skips without LAPACKE.
template <typename Scalar>
void expect_spectrum(const MatrixView<Scalar, MatrixFormat::Dense>& A0, Vector<Scalar>& d, Vector<Scalar>& e,
                     std::optional<verify::Slack> slack = std::nullopt, std::span<const int> items = {}) {
#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    (void)A0, (void)d, (void)e, (void)slack, (void)items;
    GTEST_SKIP() << "no host LAPACKE reference in this build";
#else
    const double err = spectrum_error(A0, d, e, items);
    if (slack) EXPECT_VERIFY_SLACK(Scalar, verify::Check::values, A0.rows(), err, *slack);
    else EXPECT_VERIFY(Scalar, verify::Check::values, A0.rows(), err);
#endif
}

template <typename T, Backend B>
struct SytrdBlockedConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
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

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/789);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    expect_spectrum<Real>(A0.view(), d, e, float_slack<Real>(kSlackN128));
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
    Matrix<Real, MatrixFormat::Dense> A0 =
        Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/20260806);
    const auto items = verify::all_items(batch);

    // Runs one trailing-update route; returns its spectrum_error, checked at Check::values.
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
        expect_spectrum<Real>(A0.view(), d, e, float_slack<Real>({0.25, "old bound 4 n eps(2^-23) ||A|| = 8 n u ||A||"}), items);
        return spectrum_error(A0.view(), d, e, items);
    };

    const double worst_gemm = run_route("gemm");
    const double worst_syr2k = run_route("syr2k");

    // Both routes perform the same rank-2 update, summed in a different order, so they agree only
    // to rounding. The kind's bound is ~1000x the error either route incurs (measured 2.6e-6
    // absolute at n=320, float), so this is the assertion with teeth: syr2k no worse than the GEMM
    // pair, 4x plus 16 u of the spectrum (both errors are relative to ||A||_2).
    EXPECT_LT(worst_syr2k, 4.0 * worst_gemm + 16.0 * verify::eps<Real>())
        << "syr2k route is materially less accurate than the GEMM pair: " << worst_syr2k << " vs " << worst_gemm;
}

TYPED_TEST(SytrdBlockedTest, RandomSymmetricLower33) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 33;
    const int batch = 64;
    const int nb = 8;

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/1337);
    Matrix<Real, MatrixFormat::Dense> A = A0;
    Vector<Real> d(n, batch);
    Vector<Real> e(n - 1, batch);
    Vector<Real> tau(n - 1, batch);

    const size_t ws_bytes = sytrd_blocked_buffer_size<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, nb);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_blocked<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), nb).wait();

    expect_spectrum<Real>(A0.view(), d, e, float_slack<Real>(kSlackN33));
}
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
TEST(SytrdBlockedFloatCudaTest, Syr2kTrailingUpdateMatchesNetlibReference) {
    using Real = float;
    constexpr Backend B = Backend::CUDA;

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

        expect_spectrum<Real>(A0.view(), d, e, n == 192 ? kSlackN192 : kSlackN256);
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

    expect_spectrum<Scalar>(A0.view(), d, e);
}
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
// The grid LATRD path (BATCHLAS_LATRD_IMPL=grid) only engages when
// MAX_COMPUTE_UNITS / batch >= 2, i.e. in the small-batch regime that no other
// test in this file covers. It also runs the same shapes through the legacy
// path so a divergence is attributable.
template <typename Scalar>
void run_latrd_grid_case(int n, int batch, int nb, const char* impl) {
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

    SCOPED_TRACE(::testing::Message() << "impl=" << impl << " n=" << n << " batch=" << batch << " nb=" << nb);
    expect_spectrum<Scalar>(A0.view(), d, e, float_slack<Real>(kSlackLatrd), verify::all_items(batch));
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
                    run_latrd_grid_case<float>(n, batch, nb, impl);
                    run_latrd_grid_case<double>(n, batch, nb, impl);
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
        run_latrd_grid_case<double>(1024, 1, 32, impl);
    }
}

TEST(SytrdBlockedLatrdGridCudaTest, SmallBatchComplexMatchesNetlibReference) {
    Queue probe;
    if (probe.device().type != DeviceType::GPU) {
        GTEST_SKIP() << "LATRD grid path test requires a GPU device";
    }

    for (const int batch : {1, 8}) {
        for (const int n : {96, 257}) {
            run_latrd_grid_case<std::complex<double>>(n, batch, 32, "grid");
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
            // Two results of the code under test (grid vs legacy LATRD), not a reference.
            SCOPED_TRACE(::testing::Message() << "batch=" << batch << " jobz=" << (jobz == JobType::EigenVectors ? "EV" : "NoEV"));
            test_utils::expect_eigenvalues_agree<Scalar>(W[1], W[0], n, batch);
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
            run_latrd_grid_case<double>(129, 1, 32, "grid");
        }
    }
}
#endif

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
