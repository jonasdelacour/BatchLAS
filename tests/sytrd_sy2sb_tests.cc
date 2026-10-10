#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

#include <batchlas/verify/norms.hh>
#include <batchlas/verify/scalar.hh>

using namespace batchlas;

namespace {

// ||tril(B) - band(AB)||_F / ||A0||_F: the lower triangle of B = Q^T A0 Q must be AB inside the band
// (AB(i - j, j) = B(i, j)) and zero below it.
template <typename Real>
double lower_banded_error(const Matrix<Real, MatrixFormat::Dense>& A0, const MatrixView<Real, MatrixFormat::Dense>& B,
                          const MatrixView<Real, MatrixFormat::Dense>& AB, int n, int kd) {
    std::vector<double> D(static_cast<std::size_t>(n) * n, 0.0);
    for (int j = 0; j < n; ++j) {
        for (int i = j; i < n; ++i) {
            const double band = i - j <= kd ? batchlas::verify::up(AB(i - j, j, 0)) : 0.0;
            D[static_cast<std::size_t>(i) + j * n] = batchlas::verify::up(B(i, j, 0)) - band;
        }
    }
    return batchlas::verify::frobenius(batchlas::verify::view(D.data(), n, n, n), 0) / batchlas::verify::frobenius(A0.view(), 0);
}

template <typename Real, Backend B = test_utils::gpu_backend>
void apply_sy2sb_reflectors_to_trailing(Queue& ctx,
                                       const MatrixView<Real, MatrixFormat::Dense>& A_sy2sb,
                                       const VectorView<Real>& tau,
                                       MatrixView<Real, MatrixFormat::Dense> A_work,
                                       int n,
                                       int kd) {
    // Apply Q^T * A_work * Q (real) where Q is defined by the stored Householders.
    //
    // Important: each block reflector acts on the trailing index set [i+kd, n),
    // so it must be applied to:
    //  - rows [i+kd, n) of ALL columns (left application), and
    //  - columns [i+kd, n) of ALL rows (right application).
    for (int i = 0; i <= n - kd - 1; i += kd) {
        const int pn = n - i - kd;
        if (pn <= 0) break;
        const int pk = std::min(pn, kd);

        auto V = A_sy2sb({i + kd, SliceEnd()}, {i, i + pk}).batch_item(0);
        auto A_rows = A_work({i + kd, SliceEnd()}, {0, SliceEnd()}).batch_item(0);
        auto A_cols = A_work({0, SliceEnd()}, {i + kd, SliceEnd()}).batch_item(0);

        Vector<Real> tau_panel(pk, /*batch=*/1);
        for (int j = 0; j < pk; ++j) {
            tau_panel(j, 0) = tau(i + j, 0);
        }

        UnifiedVector<std::byte> ws_l(
            ormqr_buffer_size(ctx, V, A_rows, Side::Left, Transpose::Trans, tau_panel.data()));
        ormqr(ctx, V, A_rows, Side::Left, Transpose::Trans, tau_panel.data(), ws_l.to_span()).wait();

        UnifiedVector<std::byte> ws_r(
            ormqr_buffer_size(ctx, V, A_cols, Side::Right, Transpose::NoTrans, tau_panel.data()));
        ormqr(ctx, V, A_cols, Side::Right, Transpose::NoTrans, tau_panel.data(), ws_r.to_span()).wait();
    }
}

template <typename T, Backend B>
struct SytrdSy2sbConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdSy2sbTestTypes = ::testing::Types<SytrdSy2sbConfig<float, Backend::CUDA>, SytrdSy2sbConfig<double, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SytrdSy2sbTestTypes = ::testing::Types<SytrdSy2sbConfig<float, Backend::ROCM>, SytrdSy2sbConfig<double, Backend::ROCM>>;
#else
using SytrdSy2sbTestTypes = ::testing::Types<SytrdSy2sbConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SytrdSy2sbTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdSy2sbTest, SytrdSy2sbTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SytrdSy2sbTest, RandomSymmetricLowerBandMatchesExplicitSimilarity) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int batch = 1;

    // n % kd matters: the final panel is only kd columns wide when kd divides n.
    // A short final panel used to skip part of the two-sided update, so the
    // n % kd >= 2 cases below are the regression coverage for that.
    const std::vector<std::pair<int, int>> cases = {
        {16, 8}, {17, 8}, {18, 8}, {23, 8}, {32, 12}, {40, 16}, {33, 16}, {48, 20}};

    for (const auto& nk : cases) {
        const int n = nk.first;
        const int kd = nk.second;
        SCOPED_TRACE("n=" + std::to_string(n) + " kd=" + std::to_string(kd));

        Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2024);
        Matrix<Real, MatrixFormat::Dense> A = A0;

        Matrix<Real, MatrixFormat::Dense> AB(kd + 1, n, batch);
        Vector<Real> tau(n - kd, batch);

        const size_t ws_bytes = sytrd_sy2sb_buffer_size<B, Real>(*this->ctx, A.view(), AB.view(), tau, Uplo::Lower, kd);
        UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

        sytrd_sy2sb<B, Real>(*this->ctx, A.view(), AB.view(), tau, Uplo::Lower, kd, ws.to_span()).wait();

        // Compute B = Q^T * A0 * Q by applying stored reflectors to the trailing blocks.
        Matrix<Real, MatrixFormat::Dense> Bwork = A0;
        apply_sy2sb_reflectors_to_trailing<Real, B>(*this->ctx, A.view(), static_cast<VectorView<Real>>(tau).batch_item(0), Bwork.view(), n, kd);

        // Validate AB matches the lower band of B.
        EXPECT_VERIFY(Real, batchlas::verify::Check::factorization, n, lower_banded_error(A0, Bwork.view(), AB.view(), n, kd));
    }
}
#endif
