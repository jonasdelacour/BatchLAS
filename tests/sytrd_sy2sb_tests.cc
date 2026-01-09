#include <gtest/gtest.h>

#include <blas/enums.hh>
#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "test_utils.hh"

using namespace batchlas;

namespace {

template <typename Real>
Real tol_for() {
    if constexpr (std::is_same_v<Real, float>) return Real(5) * Real(test_utils::tolerance<float>());
    return Real(5) * Real(test_utils::tolerance<double>());
}

template <typename Real>
void expect_lower_banded_matches_ab(const MatrixView<Real, MatrixFormat::Dense>& B,
                                   const MatrixView<Real, MatrixFormat::Dense>& AB,
                                   int n,
                                   int kd,
                                   Real tol) {
    const int ldab = AB.ld();

    for (int j = 0; j < n; ++j) {
        const int i_max = std::min(n - 1, j + kd);
        for (int i = j; i <= i_max; ++i) {
            const int r = i - j;
            EXPECT_NEAR(B(i, j, 0), AB(r, j, 0), tol) << "AB mismatch at (i,j)= (" << i << "," << j << ")";
        }
    }

    const Real ztol = tol * Real(50);
    for (int j = 0; j < n; ++j) {
        for (int i = j + kd + 1; i < n; ++i) {
            EXPECT_NEAR(B(i, j, 0), Real(0), ztol) << "Not banded below at (" << i << "," << j << ")";
        }
    }
}

template <typename Real>
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
            ormqr_buffer_size<Backend::CUDA>(ctx, V, A_rows, Side::Left, Transpose::Trans, tau_panel.data()));
        ormqr<Backend::CUDA>(ctx, V, A_rows, Side::Left, Transpose::Trans, tau_panel.data(), ws_l.to_span()).wait();

        UnifiedVector<std::byte> ws_r(
            ormqr_buffer_size<Backend::CUDA>(ctx, V, A_cols, Side::Right, Transpose::NoTrans, tau_panel.data()));
        ormqr<Backend::CUDA>(ctx, V, A_cols, Side::Right, Transpose::NoTrans, tau_panel.data(), ws_r.to_span()).wait();
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
#else
using SytrdSy2sbTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SytrdSy2sbTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdSy2sbTest, SytrdSy2sbTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SytrdSy2sbTest, RandomSymmetricLowerBandMatchesExplicitSimilarity) {
    using Real = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 16;
    const int kd = 8;
    const int batch = 1;
    const Real tol = tol_for<Real>();

    Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2024);
    Matrix<Real, MatrixFormat::Dense> A = A0;

    Matrix<Real, MatrixFormat::Dense> AB(kd + 1, n, batch);
    Vector<Real> tau(n - kd, batch);

    const size_t ws_bytes = sytrd_sy2sb_buffer_size<B, Real>(*this->ctx, A.view(), AB.view(), tau, Uplo::Lower, kd);
    UnifiedVector<std::byte> ws(ws_bytes, std::byte{0});

    sytrd_sy2sb<B, Real>(*this->ctx, A.view(), AB.view(), tau, Uplo::Lower, kd, ws.to_span()).wait();

    std::cout << "AB: \n"; AB.print(std::cout, n, n);

    // Compute B = Q^T * A0 * Q by applying stored reflectors to the trailing blocks.
    Matrix<Real, MatrixFormat::Dense> Bwork = A0;
    apply_sy2sb_reflectors_to_trailing<Real>(*this->ctx, A.view(), static_cast<VectorView<Real>>(tau).batch_item(0), Bwork.view(), n, kd);

    // Validate AB matches the lower band of B.
    expect_lower_banded_matches_ab(Bwork.view(), AB.view(), n, kd, tol);
}
#endif
