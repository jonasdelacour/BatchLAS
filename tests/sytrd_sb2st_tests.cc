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
#include <random>
#include <string>
#include <cstring>
#include <lapacke.h>
#include <limits>
#include <type_traits>

#include "test_utils.hh"

using namespace batchlas;

namespace {

struct ScopedEnvVar {
    std::string key;
    bool had_old = false;
    std::string old;

    ScopedEnvVar(const char* k, const char* v) : key(k) {
        if (const char* prev = std::getenv(k)) {
            had_old = true;
            old = prev;
        }
        ::setenv(k, v, 1);
    }

    ~ScopedEnvVar() {
        if (had_old) {
            ::setenv(key.c_str(), old.c_str(), 1);
        } else {
            ::unsetenv(key.c_str());
        }
    }
};

template <typename Real>
Real tol_for() {
    if constexpr (std::is_same_v<Real, float>) return Real(10) * Real(test_utils::tolerance<float>());
    return Real(10) * Real(test_utils::tolerance<double>());
}

template <typename U>
inline U conj_if_needed(const U& x) {
    if constexpr (std::is_same_v<U, std::complex<float>> || std::is_same_v<U, std::complex<double>>) {
        return std::conj(x);
    } else {
        return x;
    }
}

template <typename T>
static void dense_from_lower_band_work(const std::vector<T>& ABw,
                                       std::vector<T>& A,
                                       int n,
                                       int kd_work,
                                       int batch,
                                       int ldab,
                                       int lda) {
    // ABw layout matches the device band layout: rows = kd_work+1, cols = n, batch stride = ldab*n
    for (int b = 0; b < batch; ++b) {
        T* A_b = A.data() + static_cast<size_t>(b) * static_cast<size_t>(lda) * static_cast<size_t>(n);
        const T* AB_b = ABw.data() + static_cast<size_t>(b) * static_cast<size_t>(ldab) * static_cast<size_t>(n);

        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A_b[i + static_cast<size_t>(j) * static_cast<size_t>(lda)] = T(0);
            }
        }

        for (int j = 0; j < n; ++j) {
            for (int r = 0; r <= kd_work; ++r) {
                const int i = j + r;
                if (i >= n) break;
                const T aij = AB_b[r + static_cast<size_t>(j) * static_cast<size_t>(ldab)];
                A_b[i + static_cast<size_t>(j) * static_cast<size_t>(lda)] = aij;
                if (i != j) {
                    A_b[j + static_cast<size_t>(i) * static_cast<size_t>(lda)] = conj_if_needed(aij);
                }
            }
        }
    }
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

template <typename T>
::testing::AssertionResult expect_lower_band_unchanged(const MatrixView<T, MatrixFormat::Dense>& AB_before,
                                                       const MatrixView<T, MatrixFormat::Dense>& AB_after,
                                                       int n,
                                                       int kd,
                                                       typename base_type<T>::type tol) {
    if (AB_before.rows() != AB_after.rows()) {
        return ::testing::AssertionFailure() << "row mismatch: before=" << AB_before.rows() << " after=" << AB_after.rows();
    }
    if (AB_before.cols() != AB_after.cols()) {
        return ::testing::AssertionFailure() << "col mismatch: before=" << AB_before.cols() << " after=" << AB_after.cols();
    }
    if (AB_before.batch_size() != AB_after.batch_size()) {
        return ::testing::AssertionFailure() << "batch mismatch: before=" << AB_before.batch_size() << " after=" << AB_after.batch_size();
    }
    const int batch = AB_before.batch_size();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            const int rmax = std::min(kd, n - 1 - j);
            for (int r = 0; r <= kd; ++r) {
                const auto before = AB_before(r, j, b);
                const auto after = AB_after(r, j, b);
                if (r <= rmax) {
                    const auto diff = static_cast<typename base_type<T>::type>(std::abs(before - after));
                    if (diff > tol) {
                        return ::testing::AssertionFailure()
                               << "AB changed at (r=" << r << ", j=" << j << ") batch=" << b
                               << " diff=" << diff << " tol=" << tol;
                    }
                } else {
                    // Rows beyond the stored band should stay exactly zero (by construction).
                    const auto mag = static_cast<typename base_type<T>::type>(std::abs(after));
                    if (mag > tol) {
                        return ::testing::AssertionFailure()
                               << "AB had unexpected fill at (r=" << r << ", j=" << j << ") batch=" << b
                               << " |after|=" << mag << " tol=" << tol;
                    }
                }
            }
        }
    }

    return ::testing::AssertionSuccess();
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

    // Ensure this test actually exercises the CTA/sub-group implementation in
    // src/extensions/sytrd_sb2st_cta.cc. If the runtime/device doesn't support
    // sub_group_size=32, SB2ST will throw when forced on; in that case, skip.
    ScopedEnvVar force_subgroup("BATCHLAS_SB2ST_SUBGROUP", "1");

    #if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
    const int n = 12;
    const int kd = 3;
    #else
    const int n = 256;
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
    try {
        sytrd_sb2st<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, ws.to_span(), block_size).wait();
    } catch (const std::exception& ex) {
        if (std::strstr(ex.what(), "sub_group_size=32") != nullptr) {
            GTEST_SKIP() << ex.what();
        }
        throw;
    }

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

TYPED_TEST(SytrdSb2stTest, BandReductionMatchesDenseSyevSpectrum) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    #if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
    const int n = 12;
    const int kd = 3;
    #else
    const int n = 256;
    const int kd = 6;
    #endif
    const int batch = 5;
    const int block_size = 16;
    const Real tol = tol_for<Real>();

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/9);

    // Make A0 strictly banded so the dense reference spectrum matches the band-storage input.
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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    // Copy input AB to validate that band reduction does not mutate the input storage.
    Matrix<T, MatrixFormat::Dense> AB_before(kd + 1, n, batch);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r < kd + 1; ++r) {
                AB_before(r, j, b) = AB(r, j, b);
            }
        }
    }

    // Reference eigenvalues from dense SYEV.
    UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev(
        syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    // Under test: BANDR1-style band reduction on band storage.
    Vector<Real> d_out(n, batch);
    Vector<Real> e_out(std::max(0, n - 1), batch);
    Vector<T> tau_out(std::max(0, n - 1), batch);

    UnifiedVector<std::byte> ws(
        sytrd_band_reduction_buffer_size<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, block_size));

    sytrd_band_reduction<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, ws.to_span(), block_size).wait();

    // BANDR1 implementation uses internal workspace; input AB should remain unchanged.
    ASSERT_TRUE(expect_lower_band_unchanged<T>(AB_before, AB, n, kd, tol));

    // Sanity check: schedule-parameter overload accepts non-default d/max_sweeps/kd_work.
    // Different schedules should still preserve eigenvalues (similarity transform).
    Vector<Real> d_out2(n, batch);
    Vector<Real> e_out2(std::max(0, n - 1), batch);
    Vector<T> tau_out2(std::max(0, n - 1), batch);
    SytrdBandReductionParams params;
    params.block_size = block_size;
    params.d = 2;
    params.max_sweeps = std::max(1, kd - 1);
    params.kd_work = 3 * kd;

    UnifiedVector<std::byte> ws2(
        sytrd_band_reduction_buffer_size<B, T>(ctx, AB, d_out2, e_out2, tau_out2, Uplo::Lower, kd, params));
    sytrd_band_reduction<B, T>(ctx, AB, d_out2, e_out2, tau_out2, Uplo::Lower, kd, ws2.to_span(), params).wait();

    // Compute eigenvalues of returned tridiagonal (d,e) via host LAPACK STERF.
    UnifiedVector<Real> d_tri(static_cast<size_t>(n));
    UnifiedVector<Real> e_tri(static_cast<size_t>(std::max(0, n - 1)));

    auto check_spectrum = [&](Vector<Real>& d_vec, Vector<Real>& e_vec, const char* label) -> ::testing::AssertionResult {
        for (int bb = 0; bb < batch; ++bb) {
            for (int i = 0; i < n; ++i) d_tri[static_cast<size_t>(i)] = d_vec(i, bb);
            for (int i = 0; i < n - 1; ++i) e_tri[static_cast<size_t>(i)] = e_vec(i, bb);

            const int sterf_info = lapack_sterf<Real>(n, d_tri.data(), e_tri.data());
            if (sterf_info != 0) {
                return ::testing::AssertionFailure()
                       << "LAPACKE_xsterf failed for band_reduction tridiagonal (" << label
                       << ", batch=" << bb << ") info=" << sterf_info;
            }
            std::sort(d_tri.begin(), d_tri.end());

            for (int i = 0; i < n; ++i) {
                const Real diff = std::abs(eig_ref[static_cast<size_t>(i + bb * n)] - d_tri[static_cast<size_t>(i)]);
                if (diff > tol) {
                    return ::testing::AssertionFailure()
                           << "eigenvalue mismatch at i=" << i << ", batch=" << bb << " (" << label << ")"
                           << " diff=" << diff << " tol=" << tol;
                }
            }
        }

        return ::testing::AssertionSuccess();
    };

    ASSERT_TRUE(check_spectrum(d_out, e_out, "default"));
    ASSERT_TRUE(check_spectrum(d_out2, e_out2, "d=2"));
}

TYPED_TEST(SytrdSb2stTest, BandReductionSpectrumSmallSweep) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 64;
    const int batch = 3;
    const Real tol = tol_for<Real>();

    for (int kd : {2, 4, 8, 12}) {
        if (kd >= n) continue;
        for (int block_size : {8, 16}) {
            Matrix<T, MatrixFormat::Dense> A0 =
                Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/17 + kd + block_size);

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
            fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

            UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
            UnifiedVector<std::byte> ws_syev(
                syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
            syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
            for (int b = 0; b < batch; ++b) {
                std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                          eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
            }

            Vector<Real> d_out(n, batch);
            Vector<Real> e_out(std::max(0, n - 1), batch);
            Vector<T> tau_out(std::max(0, n - 1), batch);
            UnifiedVector<std::byte> ws(
                sytrd_band_reduction_buffer_size<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, block_size));
            sytrd_band_reduction<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, ws.to_span(), block_size).wait();

            UnifiedVector<Real> d_tri(static_cast<size_t>(n));
            UnifiedVector<Real> e_tri(static_cast<size_t>(std::max(0, n - 1)));

            for (int b = 0; b < batch; ++b) {
                for (int i = 0; i < n; ++i) d_tri[static_cast<size_t>(i)] = d_out(i, b);
                for (int i = 0; i < n - 1; ++i) e_tri[static_cast<size_t>(i)] = e_out(i, b);

                ASSERT_EQ(lapack_sterf<Real>(n, d_tri.data(), e_tri.data()), 0)
                    << "LAPACKE_xsterf failed (kd=" << kd << ", block_size=" << block_size << ", batch=" << b << ")";
                std::sort(d_tri.begin(), d_tri.end());

                for (int i = 0; i < n; ++i) {
                    ASSERT_NEAR(eig_ref[static_cast<size_t>(i + b * n)], d_tri[static_cast<size_t>(i)], tol)
                        << "eigenvalue mismatch at i=" << i
                        << ", kd=" << kd
                        << ", block_size=" << block_size
                        << ", batch=" << b;
                }
            }
        }
    }
}

TYPED_TEST(SytrdSb2stTest, BandReductionSingleStepBandContainment) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 64;
    const int kd = 8;
    const int kd_work = 3 * kd;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/21);
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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);

    SytrdBandReductionParams params;
    params.block_size = 8;
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d = 0;

    UnifiedVector<std::byte> ws(
        sytrd_band_reduction_single_step_buffer_size<B, T>(ctx, AB, ABw, Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step<B, T>(ctx, AB, ABw, Uplo::Lower, kd, ws.to_span(), params).wait();

    const Real tol = tol_for<Real>();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            const int rmax = std::min(kd_work, n - 1 - j);
            for (int r = rmax + 1; r <= kd_work; ++r) {
                // Rows that would map past the last matrix row must remain exactly zero.
                ASSERT_NEAR(static_cast<Real>(std::abs(ABw(r, j, b))), Real(0), tol)
                    << "ABw had unexpected fill at (r=" << r << ", j=" << j << ") batch=" << b;
            }
        }
    }
}

TYPED_TEST(SytrdSb2stTest, BandReductionDumpEvolution64) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 64;
    const int kd = 8;
    const int block_size = 16;
    const int batch = 1;

    Matrix<T, MatrixFormat::Dense> A0 =
        Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);

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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    using Real = typename base_type<T>::type;
    Vector<Real> d_out(n, batch);
    Vector<Real> e_out(std::max(0, n - 1), batch);
    Vector<T> tau_out(std::max(0, n - 1), batch);
    UnifiedVector<std::byte> ws(
        sytrd_band_reduction_buffer_size<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, block_size));

    // Intended for debug-dump visualization; correctness is covered elsewhere.
    sytrd_band_reduction<B, T>(ctx, AB, d_out, e_out, tau_out, Uplo::Lower, kd, ws.to_span(), block_size).wait();
    SUCCEED();
}

TYPED_TEST(SytrdSb2stTest, BandReductionSingleStepSpectrumPreservation) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 32;
    const int kd = 6;
    const int kd_work = 3 * kd;
    const int batch = 2;
    const Real tol = tol_for<Real>();

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/33);
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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    // Reference eigenvalues from dense SYEV on A0.
    UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev(
        syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    // Run exactly one BANDR1 chase step and reconstruct dense A_after.
    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);
    SytrdBandReductionParams params;
    params.block_size = 8;
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d = 0;

    UnifiedVector<std::byte> ws_step(
        sytrd_band_reduction_single_step_buffer_size<B, T>(ctx, AB, ABw, Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step<B, T>(ctx, AB, ABw, Uplo::Lower, kd, ws_step.to_span(), params).wait();

    std::vector<T> ABw_host(static_cast<size_t>(batch) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n));
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r <= kd_work; ++r) {
                ABw_host[static_cast<size_t>(b) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n) +
                         static_cast<size_t>(r) + static_cast<size_t>(j) * static_cast<size_t>(kd_work + 1)] =
                    ABw(r, j, b);
            }
        }
    }

    std::vector<T> A_after_host(static_cast<size_t>(batch) * static_cast<size_t>(n) * static_cast<size_t>(n));
    dense_from_lower_band_work(ABw_host, A_after_host, n, kd_work, batch, kd_work + 1, n);

    Matrix<T, MatrixFormat::Dense> A_after(n, n, batch);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A_after(i, j, b) =
                    A_after_host[static_cast<size_t>(b) * static_cast<size_t>(n) * static_cast<size_t>(n) +
                                 static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(n)];
            }
        }
    }

    UnifiedVector<Real> eig_after(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev2(
        syev_buffer_size<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower, ws_syev2.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_after.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_after.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eig_ref[static_cast<size_t>(i + b * n)], eig_after[static_cast<size_t>(i + b * n)], tol)
                << "eigenvalue mismatch at i=" << i << ", batch=" << b;
        }
    }
}

TYPED_TEST(SytrdSb2stTest, BandReductionMultiStepSpectrumPreservation) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 64;
    const int kd = 8;
    const int kd_work = 3 * kd;
    const int batch = 2;
    const Real tol0 = tol_for<Real>();

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/35);
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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    // Reference eigenvalues from dense SYEV on A0.
    UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev(
        syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);

    constexpr int kMax = 16;
    for (int k = 1; k <= kMax; ++k) {
        SytrdBandReductionParams params;
        params.block_size = 8;
        params.kd_work = kd_work;
        params.max_sweeps = 1;
        params.d = 0;
        params.max_steps = k;

        UnifiedVector<std::byte> ws_step(
            sytrd_band_reduction_single_step_buffer_size<B, T>(ctx, AB, ABw, Uplo::Lower, kd, params));
        sytrd_band_reduction_single_step<B, T>(ctx, AB, ABw, Uplo::Lower, kd, ws_step.to_span(), params).wait();

        std::vector<T> ABw_host(static_cast<size_t>(batch) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n));
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int r = 0; r <= kd_work; ++r) {
                    ABw_host[static_cast<size_t>(b) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n) +
                             static_cast<size_t>(r) + static_cast<size_t>(j) * static_cast<size_t>(kd_work + 1)] =
                        ABw(r, j, b);
                }
            }
        }

        std::vector<T> A_after_host(static_cast<size_t>(batch) * static_cast<size_t>(n) * static_cast<size_t>(n));
        dense_from_lower_band_work(ABw_host, A_after_host, n, kd_work, batch, kd_work + 1, n);

        Matrix<T, MatrixFormat::Dense> A_after(n, n, batch);
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    A_after(i, j, b) =
                        A_after_host[static_cast<size_t>(b) * static_cast<size_t>(n) * static_cast<size_t>(n) +
                                     static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(n)];
                }
            }
        }

        UnifiedVector<Real> eig_after(static_cast<size_t>(n) * static_cast<size_t>(batch));
        UnifiedVector<std::byte> ws_syev2(
            syev_buffer_size<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower));
        syev<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower, ws_syev2.to_span()).wait();
        for (int b = 0; b < batch; ++b) {
            std::sort(eig_after.begin() + static_cast<ptrdiff_t>(b) * n,
                      eig_after.begin() + static_cast<ptrdiff_t>(b + 1) * n);
        }

        const Real tol = tol0 * static_cast<Real>(std::max(1, k));
        bool ok = true;
        Real max_abs_diff = Real(0);
        int max_b = 0;
        int max_i = 0;
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) {
                const Real diff = std::abs(eig_ref[static_cast<size_t>(i + b * n)] -
                                           eig_after[static_cast<size_t>(i + b * n)]);
                if (diff > max_abs_diff) {
                    max_abs_diff = diff;
                    max_b = b;
                    max_i = i;
                }
                if (diff > tol) {
                    ok = false;
                }
            }
        }

        if (!ok) {
            // Re-run once with dumps enabled to support Python comparison.
            const std::string dir = std::string("output/bandr1_dumps/gtest_multistep_k") + std::to_string(k);
            ScopedEnvVar dump_on("BATCHLAS_DUMP_BANDR1_STEP", "1");
            ScopedEnvVar dump_batch("BATCHLAS_DUMP_BANDR1_BATCH", std::to_string(max_b).c_str());
            ScopedEnvVar dump_dir("BATCHLAS_DUMP_BANDR1_DIR", dir.c_str());
            sytrd_band_reduction_single_step<B, T>(ctx, AB, ABw, Uplo::Lower, kd, ws_step.to_span(), params).wait();

            ADD_FAILURE() << "Spectrum not preserved after k=" << k
                          << " chase steps. max_abs_diff=" << max_abs_diff
                          << " at (i=" << max_i << ", batch=" << max_b << ")"
                          << ". BANDR1 dumps written to: " << dir;
            break;
        }
    }
}

namespace {

// Count how many QR-chase steps the BANDR1 schedule executes in a single sweep
// starting from bandwidth b = kd.
inline int bandr1_count_steps_one_sweep(int n, int kd, int block_size, int d_per_sweep) {
    if (n <= 0 || kd <= 1) return 0;
    int b = kd;

    const int d_red = (d_per_sweep > 0) ? std::min(d_per_sweep, b - 1)
                                        : std::max(1, b - std::min(block_size, b - 1));
    const int b_tilde = b - d_red;
    const int nb = std::min(std::max(1, block_size), b_tilde);

    int steps = 0;
    for (int j1 = 0; j1 < std::max(0, n - b_tilde); j1 += nb) {
        const int j2 = std::min(j1 + nb - 1, n - 1);
        int i1 = j1 + b_tilde;
        int i2 = std::min(j1 + b + nb - 1, n - 1);

        while (i1 < n) {
            if (i1 > i2) {
                i1 = i2 + 1;
                i2 = std::min(i1 + b - 1, n - 1);
                continue;
            }

            const int m = i2 - i1 + 1;
            const int r = j2 - j1 + 1;
            if (m <= 0 || r <= 0) {
                i1 = i2 + 1;
                i2 = std::min(i1 + b - 1, n - 1);
                continue;
            }

            ++steps;
            i1 = i2 + 1;
            i2 = std::min(i1 + b - 1, n - 1);
        }
    }
    return steps;
}

} // namespace

TYPED_TEST(SytrdSb2stTest, BandReductionOneSweepSpectrumPreservation) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;

    const int n = 64;
    const int kd = 8;
    const int kd_work = 3 * kd;
    const int batch = 2;
    const Real tol0 = tol_for<Real>();

    Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/71);
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
    fill_lower_band_from_dense<T>(A0.view(), AB, n, kd);

    UnifiedVector<Real> eig_ref(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev(
        syev_buffer_size<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A0.view(), eig_ref, JobType::NoEigenVectors, Uplo::Lower, ws_syev.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_ref.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_ref.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);

    SytrdBandReductionParams params;
    params.block_size = 8;
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d = 0;
    params.max_steps = bandr1_count_steps_one_sweep(n, kd, params.block_size, params.d);

    ASSERT_GT(params.max_steps, 0) << "unexpected: sweep step count is 0";

    UnifiedVector<std::byte> ws_step(
        sytrd_band_reduction_single_step_buffer_size<B, T>(ctx, AB, ABw, Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step<B, T>(ctx, AB, ABw, Uplo::Lower, kd, ws_step.to_span(), params).wait();

    std::vector<T> ABw_host(static_cast<size_t>(batch) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n));
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int r = 0; r <= kd_work; ++r) {
                ABw_host[static_cast<size_t>(b) * static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n) +
                         static_cast<size_t>(r) + static_cast<size_t>(j) * static_cast<size_t>(kd_work + 1)] =
                    ABw(r, j, b);
            }
        }
    }

    std::vector<T> A_after_host(static_cast<size_t>(batch) * static_cast<size_t>(n) * static_cast<size_t>(n));
    dense_from_lower_band_work(ABw_host, A_after_host, n, kd_work, batch, kd_work + 1, n);

    Matrix<T, MatrixFormat::Dense> A_after(n, n, batch);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                A_after(i, j, b) =
                    A_after_host[static_cast<size_t>(b) * static_cast<size_t>(n) * static_cast<size_t>(n) +
                                 static_cast<size_t>(i) + static_cast<size_t>(j) * static_cast<size_t>(n)];
            }
        }
    }

    UnifiedVector<Real> eig_after(static_cast<size_t>(n) * static_cast<size_t>(batch));
    UnifiedVector<std::byte> ws_syev2(
        syev_buffer_size<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower));
    syev<B, T>(ctx, A_after.view(), eig_after, JobType::NoEigenVectors, Uplo::Lower, ws_syev2.to_span()).wait();
    for (int b = 0; b < batch; ++b) {
        std::sort(eig_after.begin() + static_cast<ptrdiff_t>(b) * n,
                  eig_after.begin() + static_cast<ptrdiff_t>(b + 1) * n);
    }

    const Real tol = tol0;
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            ASSERT_NEAR(eig_ref[static_cast<size_t>(i + b * n)], eig_after[static_cast<size_t>(i + b * n)], tol)
                << "eigenvalue mismatch after one full sweep at i=" << i << ", batch=" << b;
        }
    }
}
#endif
