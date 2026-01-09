#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <blas/functions.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <lapacke.h>
#include <limits>
#include <type_traits>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename Real>
Real tol_for() {
    if constexpr (std::is_same_v<Real, float>) return Real(10) * Real(test_utils::tolerance<float>());
    return Real(10) * Real(test_utils::tolerance<double>());
}

template <typename T>
void fill_lower_band_from_dense(const MatrixView<T, MatrixFormat::Dense>& A,
                                MatrixView<T, MatrixFormat::Dense> AB,
                                int n,
                                int kd) {
    // AB is (kd+1) x n, lower band: AB(r,j) = A(j+r, j)
    const int batch = A.batch_size();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            const int rmax = std::min(kd, n - 1 - j);
            for (int r = 0; r <= rmax; ++r) {
                AB(r, j, b) = A(j + r, j, b);
            }
            for (int r = rmax + 1; r <= kd; ++r) {
                AB(r, j, b) = T(0);
            }
        }
    }
}

template <typename T, Backend B>
struct SytrdSb2stConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename Real>
int lapack_sterf(int n, Real* d, Real* e);

template <>
int lapack_sterf<float>(int n, float* d, float* e) {
    return LAPACKE_ssterf(static_cast<lapack_int>(n), d, e);
}

template <>
int lapack_sterf<double>(int n, double* d, double* e) {
    return LAPACKE_dsterf(static_cast<lapack_int>(n), d, e);
}

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdSb2stTestTypes = ::testing::Types<
    SytrdSb2stConfig<float, Backend::CUDA>,
    SytrdSb2stConfig<double, Backend::CUDA>,
    SytrdSb2stConfig<std::complex<float>, Backend::CUDA>,
    SytrdSb2stConfig<std::complex<double>, Backend::CUDA>>;
#else
using SytrdSb2stTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SytrdSb2stTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdSb2stTest, SytrdSb2stTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SytrdSb2stTest, MatchesDenseSyevSpectrum) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    #if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
    const int n = 12;
    const int kd = 3;
    #else
    const int n = 1024;
    const int kd = 6;
    #endif
    const int batch = 5;
    const int block_size = 16;
    const Real tol = tol_for<Real>();

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/7);

    // Make A0 strictly banded (|i-j| <= kd) so that the dense reference spectrum
    // matches the band-storage input to SB2ST.
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                if (std::abs(i - j) > kd) {
                    A0(i, j, b) = T(0);
                }
            }
        }
    }

    Matrix<T, MatrixFormat::Dense> AB(kd + 1, n, batch);
    fill_lower_band_from_dense<T>(A0, AB, n, kd);

    // Reference eigenvalues from dense SYEV.
    UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev(
        syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    // Under test: SB2ST on band storage.
    Vector<Real> d_out(n, batch);
    Vector<Real> e_out(std::max(0, n - 1), batch);
    Vector<T> tau_out(std::max(0, n - 1), batch);

    UnifiedVector<std::byte> ws(
        sytrd_sb2st_buffer_size<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, block_size));
    sytrd_sb2st<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, ws.to_span(), block_size).wait();

    // SB2ST outputs tridiagonal (d,e). Use host LAPACK STERF to compute its eigenvalues.
    // Note: STERF overwrites (d,e), so copy into scratch buffers.
    UnifiedVector<Real> d_tri(static_cast<size_t>(n));
    UnifiedVector<Real> e_tri(static_cast<size_t>(std::max(0, n - 1)));

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) d_tri[static_cast<size_t>(i)] = d_out(i, b);
        for (int i = 0; i < n - 1; ++i) e_tri[static_cast<size_t>(i)] = e_out(i, b);

        ASSERT_EQ(lapack_sterf<Real>(n, d_tri.data(), e_tri.data()), 0)
            << "LAPACKE_xsterf failed for SB2ST tridiagonal (batch=" << b << ")";
        std::sort(d_tri.begin(), d_tri.end());

        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(eig_ref[static_cast<size_t>(i + b * n)], d_tri[static_cast<size_t>(i)], tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
}
#endif
