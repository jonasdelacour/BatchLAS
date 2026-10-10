#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/blas/functions.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <string>
#include <cstring>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/scalar.hh>
#include <batchlas/verify/tolerance.hh>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

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
                    A_b[j + static_cast<size_t>(i) * static_cast<size_t>(lda)] = batchlas::verify::conj(aij);
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
                                                       int kd) {
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
                    // The input band is read-only: it must come back bit for bit.
                    if (!(before == after)) {
                        return ::testing::AssertionFailure()
                               << "AB changed at (r=" << r << ", j=" << j << ") batch=" << b
                               << " before=" << before << " after=" << after;
                    }
                } else {
                    // Rows beyond the stored band should stay exactly zero (by construction).
                    if (!(after == T(0))) {
                        return ::testing::AssertionFailure()
                               << "AB had unexpected fill at (r=" << r << ", j=" << j << ") batch=" << b
                               << " after=" << after;
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

// LAPACKE spectra (ascending, in double) per batch item: of the Hermitian A (both triangles valid) and of
// the tridiagonals (d, e). Both are independent of any BatchLAS operation.
using Spectra = std::vector<std::vector<double>>;

template <class View>
bool dense_spectra(const View& A, Spectra& out) {
    out.assign(static_cast<std::size_t>(A.batch_size()), {});
    for (int b = 0; b < A.batch_size(); ++b) {
        auto a = batchlas::verify::copy_item(A, b);
        if (!batchlas::verify::eigenvalues(A.rows(), a, out[static_cast<std::size_t>(b)])) return false;
    }
    return true;
}

template <class Real>
bool tridiagonal_spectra(Vector<Real>& d, Vector<Real>& e, int n, int batch, Spectra& out) {
    out.assign(static_cast<std::size_t>(batch), {});
    for (int b = 0; b < batch; ++b) {
        std::vector<double> dd(static_cast<std::size_t>(n)), ee(static_cast<std::size_t>(std::max(0, n - 1)));
        for (int i = 0; i < n; ++i) dd[static_cast<std::size_t>(i)] = d(i, b);
        for (int i = 0; i < n - 1; ++i) ee[static_cast<std::size_t>(i)] = e(i, b);
        if (!batchlas::verify::tridiagonal_eigenvalues(dd, ee)) return false;
        out[static_cast<std::size_t>(b)] = std::move(dd);
    }
    return true;
}

// max |got - ref| / max|ref| over the batch (the library's values_error); *worst_item is the item attaining it.
inline double spectra_error(const Spectra& got, const Spectra& ref, int* worst_item = nullptr) {
    const int batch = static_cast<int>(ref.size());
    const int n = static_cast<int>(ref.front().size());
    std::vector<double> packed;
    double scale = 0;
    for (int b = 0; b < batch; ++b) {
        packed.insert(packed.end(), got[static_cast<std::size_t>(b)].begin(), got[static_cast<std::size_t>(b)].end());
        for (double l : ref[static_cast<std::size_t>(b)]) scale = batchlas::verify::nanmax(scale, std::fabs(l));
    }
    const VectorView<double> w(packed.data(), n, batch);
    double worst = 0;
    for (int b = 0; b < batch; ++b) {
        const double err = batchlas::verify::values_error(w, ref, scale, std::span<const int>(&b, 1));
        if (worst_item && !(err <= worst)) *worst_item = b;
        worst = batchlas::verify::nanmax(worst, err);
    }
    return worst;
}

// Spectrum comparisons (Check::values): float keeps the old 10 x tolerance<float>() = 1e-4 absolute, about
// 3e-5 of the spectral radius (3.3 at n = 256), 1/16 of the kind's bound; measured need 0.009 of the kind's
// bound. Double's old 1e-9 absolute is looser than the kind's bound, which it meets (need 0.017). @p steps
// scales the bound for that many accumulated chase steps, as the old tol0 * k.
template <class T>
batchlas::verify::Slack spectra_slack(int steps = 1) {
    if constexpr (std::is_same_v<typename base_type<T>::type, float>)
        return {0.0625 * std::max(1, steps), "the old 1e-4 absolute spectrum tolerance (about 3e-5 of the spectral radius); measured need 0.009"};
    else
        return {1.0 * std::max(1, steps), "the library bound, tighter than the old 1e-9 absolute; measured need 0.017"};
}

template <class T>
::testing::AssertionResult spectra_match(const Spectra& got, const Spectra& ref, int n) {
    return test_utils::verify_pass<T>(batchlas::verify::Check::values, n, spectra_error(got, ref), spectra_slack<T>());
}

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdSb2stTestTypes = ::testing::Types<
    SytrdSb2stConfig<float, Backend::CUDA>,
    SytrdSb2stConfig<double, Backend::CUDA>,
    SytrdSb2stConfig<std::complex<float>, Backend::CUDA>,
    SytrdSb2stConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SytrdSb2stTestTypes = ::testing::Types<
    SytrdSb2stConfig<float, Backend::ROCM>,
    SytrdSb2stConfig<double, Backend::ROCM>,
    SytrdSb2stConfig<std::complex<float>, Backend::ROCM>,
    SytrdSb2stConfig<std::complex<double>, Backend::ROCM>>;
#else
using SytrdSb2stTestTypes = ::testing::Types<SytrdSb2stConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SytrdSb2stTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdSb2stTest, SytrdSb2stTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SytrdSb2stTest, MatchesDenseSyevSpectrum) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

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

    Spectra eig_ref;
    ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

    // Under test: SB2ST on band storage.
    Vector<Real> d_out(n, batch);
    Vector<Real> e_out(std::max(0, n - 1), batch);
    Vector<T> tau_out(std::max(0, n - 1), batch);

    UnifiedVector<std::byte> ws(
        sytrd_sb2st_buffer_size(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, block_size));
    try {
        sytrd_sb2st(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, ws.to_span(), block_size).wait();
    } catch (const std::exception& ex) {
        if (std::strstr(ex.what(), "sub_group_size=32") != nullptr) {
            GTEST_SKIP() << ex.what();
        }
        throw;
    }

    // SB2ST outputs tridiagonal (d,e): its LAPACKE eigenvalues must be those of A0.
    Spectra tri;
    ASSERT_TRUE(tridiagonal_spectra(d_out, e_out, n, batch, tri)) << "LAPACKE_dsterf failed for SB2ST tridiagonal";
    EXPECT_TRUE(spectra_match<T>(tri, eig_ref, n)) << "SB2ST";
}

TYPED_TEST(SytrdSb2stTest, BandReductionMatchesDenseSyevSpectrum) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

    #if defined(BATCHLAS_SB2ST_DEBUG_PRINTF)
    const int n = 12;
    const int kd = 3;
    #else
    const int n = 256;
    const int kd = 6;
    #endif
    const int batch = 5;
    const int block_size = 16;

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

    Spectra eig_ref;
    ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

    // Under test: BANDR1-style band reduction on band storage.
    Vector<Real> d_out(n, batch);
    Vector<Real> e_out(std::max(0, n - 1), batch);
    Vector<T> tau_out(std::max(0, n - 1), batch);

    UnifiedVector<std::byte> ws(
        sytrd_band_reduction_buffer_size(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, block_size));

    sytrd_band_reduction(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, ws.to_span(), block_size).wait();

    // BANDR1 implementation uses internal workspace; input AB should remain unchanged.
    ASSERT_TRUE(expect_lower_band_unchanged<T>(AB_before, AB, n, kd));

    // Sanity check: schedule-parameter overload accepts non-default d/max_sweeps/kd_work.
    // Different schedules should still preserve eigenvalues (similarity transform).
    Vector<Real> d_out2(n, batch);
    Vector<Real> e_out2(std::max(0, n - 1), batch);
    Vector<T> tau_out2(std::max(0, n - 1), batch);
    SytrdBandReductionParams params;
    params.block_size_seq = {block_size};
    params.d_seq = {2};
    params.max_sweeps = std::max(1, kd - 1);
    params.kd_work = 3 * kd;

    UnifiedVector<std::byte> ws2(
        sytrd_band_reduction_buffer_size(ctx, AB.view(), d_out2.view(), e_out2.view(), tau_out2.view(), Uplo::Lower, kd, params));
    sytrd_band_reduction(ctx, AB.view(), d_out2.view(), e_out2.view(), tau_out2.view(), Uplo::Lower, kd, ws2.to_span(), params).wait();

    // The LAPACKE eigenvalues of each returned tridiagonal (d,e) must be those of A0.
    auto check_spectrum = [&](Vector<Real>& d_vec, Vector<Real>& e_vec, const char* label) -> ::testing::AssertionResult {
        Spectra tri;
        if (!tridiagonal_spectra(d_vec, e_vec, n, batch, tri))
            return ::testing::AssertionFailure() << "LAPACKE_dsterf failed for band_reduction tridiagonal (" << label << ")";
        return spectra_match<T>(tri, eig_ref, n) << " (" << label << ")";
    };

    ASSERT_TRUE(check_spectrum(d_out, e_out, "default"));
    ASSERT_TRUE(check_spectrum(d_out2, e_out2, "d=2"));
}

TYPED_TEST(SytrdSb2stTest, BandReductionSpectrumSmallSweep) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

    const int n = 64;
    const int batch = 3;

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

            Spectra eig_ref;
            ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

            Vector<Real> d_out(n, batch);
            Vector<Real> e_out(std::max(0, n - 1), batch);
            Vector<T> tau_out(std::max(0, n - 1), batch);
            UnifiedVector<std::byte> ws(
                sytrd_band_reduction_buffer_size(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, block_size));
            sytrd_band_reduction(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, ws.to_span(), block_size).wait();

            Spectra tri;
            ASSERT_TRUE(tridiagonal_spectra(d_out, e_out, n, batch, tri))
                << "LAPACKE_dsterf failed (kd=" << kd << ", block_size=" << block_size << ")";
            ASSERT_TRUE(spectra_match<T>(tri, eig_ref, n)) << "kd=" << kd << ", block_size=" << block_size;
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
    params.block_size_seq = {8};
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d_seq = {0};

    UnifiedVector<std::byte> ws(
        sytrd_band_reduction_single_step_buffer_size(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, ws.to_span(), params).wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            const int rmax = std::min(kd_work, n - 1 - j);
            for (int r = rmax + 1; r <= kd_work; ++r) {
                // Rows that would map past the last matrix row must remain exactly zero.
                ASSERT_TRUE(ABw(r, j, b) == T(0))
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
        sytrd_band_reduction_buffer_size(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, block_size));

    // Intended for debug-dump visualization; correctness is covered elsewhere.
    sytrd_band_reduction(ctx, AB.view(), d_out.view(), e_out.view(), tau_out.view(), Uplo::Lower, kd, ws.to_span(), block_size).wait();
    SUCCEED();
}

TYPED_TEST(SytrdSb2stTest, BandReductionSingleStepSpectrumPreservation) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

    const int n = 32;
    const int kd = 6;
    const int kd_work = 3 * kd;
    const int batch = 2;

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

    Spectra eig_ref;
    ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

    // Run exactly one BANDR1 chase step and reconstruct dense A_after.
    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);
    SytrdBandReductionParams params;
    params.block_size_seq = {8};
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d_seq = {0};

    UnifiedVector<std::byte> ws_step(
        sytrd_band_reduction_single_step_buffer_size(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, ws_step.to_span(), params).wait();

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

    Spectra eig_after;
    ASSERT_TRUE(dense_spectra(A_after.view(), eig_after)) << "LAPACKE spectrum of the reduced band failed";

    ASSERT_TRUE(spectra_match<T>(eig_after, eig_ref, n));
}

TYPED_TEST(SytrdSb2stTest, BandReductionMultiStepSpectrumPreservation) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    auto& ctx = *this->ctx;
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

    const int n = 64;
    const int kd = 8;
    const int kd_work = 3 * kd;
    const int batch = 2;

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

    Spectra eig_ref;
    ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);

    constexpr int kMax = 16;
    for (int k = 1; k <= kMax; ++k) {
        SytrdBandReductionParams params;
        params.block_size_seq = {8};
        params.kd_work = kd_work;
        params.max_sweeps = 1;
        params.d_seq = {0};
        params.max_steps = k;

        UnifiedVector<std::byte> ws_step(
            sytrd_band_reduction_single_step_buffer_size(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, params));
        sytrd_band_reduction_single_step(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, ws_step.to_span(), params).wait();

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

        Spectra eig_after;
        ASSERT_TRUE(dense_spectra(A_after.view(), eig_after)) << "LAPACKE spectrum of the reduced band failed";

        int max_b = 0;
        const double err = spectra_error(eig_after, eig_ref, &max_b);
        const bool ok = batchlas::verify::pass<T>(batchlas::verify::Check::values, n, err, spectra_slack<T>(k));
        if (!ok) {
            // Re-run once with dumps enabled to support Python comparison.
            const std::string dir = std::string("output/bandr1_dumps/gtest_multistep_k") + std::to_string(k);
            ScopedEnvVar dump_on("BATCHLAS_DUMP_BANDR1_STEP", "1");
            ScopedEnvVar dump_batch("BATCHLAS_DUMP_BANDR1_BATCH", std::to_string(max_b).c_str());
            ScopedEnvVar dump_dir("BATCHLAS_DUMP_BANDR1_DIR", dir.c_str());
            sytrd_band_reduction_single_step(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, ws_step.to_span(), params).wait();

            ADD_FAILURE() << "Spectrum not preserved after k=" << k
                          << " chase steps. error=" << err << " in batch=" << max_b
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
    if (!BATCHLAS_VERIFY_HAVE_LAPACKE) GTEST_SKIP() << "no host LAPACKE reference in this build";

    const int n = 64;
    const int kd = 8;
    const int kd_work = 3 * kd;
    const int batch = 2;

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

    Spectra eig_ref;
    ASSERT_TRUE(dense_spectra(A0.view(), eig_ref)) << "LAPACKE reference failed";

    Matrix<T, MatrixFormat::Dense> ABw(kd_work + 1, n, batch);

    SytrdBandReductionParams params;
    params.block_size_seq = {8};
    params.kd_work = kd_work;
    params.max_sweeps = 1;
    params.d_seq = {0};
    params.max_steps = bandr1_count_steps_one_sweep(n, kd, params.block_size_seq.front(), params.d_seq.front());

    ASSERT_GT(params.max_steps, 0) << "unexpected: sweep step count is 0";

    UnifiedVector<std::byte> ws_step(
        sytrd_band_reduction_single_step_buffer_size(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, params));
    sytrd_band_reduction_single_step(ctx, AB.view(), ABw.view(), Uplo::Lower, kd, ws_step.to_span(), params).wait();

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

    Spectra eig_after;
    ASSERT_TRUE(dense_spectra(A_after.view(), eig_after)) << "LAPACKE spectrum of the reduced band failed";

    ASSERT_TRUE(spectra_match<T>(eig_after, eig_ref, n)) << "after one full sweep";
}
#endif
