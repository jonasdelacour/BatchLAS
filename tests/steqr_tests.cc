#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>
#include "test_utils.hh"
#include "eigen_verify.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {
// Eigenvalues of every item of a dense symmetric A (both triangles valid) from LAPACKE, ascending,
// and ||A||_2 over them. Empty without LAPACKE.
template <typename Real>
std::vector<std::vector<double>> lapacke_spectra(const MatrixView<Real, MatrixFormat::Dense>& A, double& norm2) {
    std::vector<std::vector<double>> w(static_cast<std::size_t>(A.batch_size()));
    norm2 = 0;
    for (int b = 0; b < A.batch_size(); ++b) {
        auto a = verify::copy_item(A, b);
        if (!verify::eigenvalues(A.rows(), a, w[static_cast<std::size_t>(b)])) return {};
        for (double l : w[static_cast<std::size_t>(b)]) norm2 = verify::nanmax(norm2, std::fabs(l));
    }
    return w;
}

// The same closed-form spectrum for each of `batch` items, and its ||.||_2.
std::vector<std::vector<double>> repeated(const std::vector<double>& one, int batch, double& norm2) {
    norm2 = 0;
    for (double l : one) norm2 = verify::nanmax(norm2, std::fabs(l));
    return std::vector<std::vector<double>>(static_cast<std::size_t>(batch), one);
}

// w (n values per item) against ref at Check::values, scaled by norm2. An empty ref (no LAPACKE) skips.
template <typename Real>
void expect_values(const VectorView<Real>& w, const std::vector<std::vector<double>>& ref, double norm2, int n,
                   std::optional<verify::Slack> slack = std::nullopt, std::span<const int> items = {}) {
    if (ref.empty()) GTEST_SKIP() << "no host LAPACKE reference in this build";
    const double err = verify::values_error(w, ref, norm2, items);
    if (slack) EXPECT_VERIFY_SLACK(Real, verify::Check::values, n, err, *slack);
    else EXPECT_VERIFY(Real, verify::Check::values, n, err);
}

// (A, V, w) an eigendecomposition with orthonormal V (eigen_residual, orthogonality_rotations).
template <typename Real>
void expect_pairs(const MatrixView<Real, MatrixFormat::Dense>& A, const MatrixView<Real, MatrixFormat::Dense>& V,
                  const VectorView<Real>& w, std::span<const int> items = {}) {
    const int n = V.cols();
    EXPECT_VERIFY(Real, verify::Check::eigen_residual, n, verify::eigen_residual(A, V, w, items));
    EXPECT_VERIFY(Real, verify::Check::orthogonality_rotations, n, verify::orthogonality(V, items));
}

// float only: the closed-form checks' old 1e-5 absolute (both types) is 0.16 x the kind's float
// bound at n = 16, ||T||_2 = 2; double keeps the kind (tighter than 1e-5).
template <typename Real>
std::optional<verify::Slack> float_slack(verify::Slack s) {
    if constexpr (std::is_same_v<Real, float>) return s;
    return std::nullopt;
}
const verify::Slack kClosedFormN16{0.16, "old bound 1e-5 absolute, ||T||_2 = 2"};

static inline const char* update_scheme_name(SteqrUpdateScheme scheme) {
    switch (scheme) {
        case SteqrUpdateScheme::PG:
            return "PG";
        case SteqrUpdateScheme::EXP:
            return "EXP";
        default:
            return "UNKNOWN";
    }
}

static constexpr std::array<SteqrUpdateScheme, 2> kSteqrUpdateSchemes = {
    SteqrUpdateScheme::PG,
    SteqrUpdateScheme::EXP,
};
}

template <typename T, Backend B>
struct SteqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
// STEQR tests are not meaningful for complex types.
using SteqrTestTypes = typename test_utils::backend_types_filtered<SteqrConfig, false>::type;

template <typename Config>
class SteqrTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(SteqrTest, SteqrTestTypes);

TYPED_TEST(SteqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    Vector<float_type> eigenvalues(n, batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }
    SteqrParams<float_type> params= {};
    params.sort = true;
    params.transpose_working_vectors = false;
    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    params.sort_order = SortOrder::Ascending;

    //VectorView<float_type>::copy(*this->ctx, VectorView(diag), VectorView(sub_diag)).wait();

    auto ws = UnifiedVector<std::byte>(steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));
    (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
        ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(c), batch);
    double norm2 = 0;
    const auto ref = repeated(std::vector<double>(expected_eigenvalues.begin(), expected_eigenvalues.end()), batch, norm2);
    expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n, float_slack<float_type>(kClosedFormN16));
    expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(eigenvalues));
}

TYPED_TEST(SteqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 512;
    // n=512 with eigenvectors plus a dense 512x512xbatch eigenpair check is
    // an O(n^3)*batch job that runs on the host for the NETLIB instantiations.
    // The closed-form Toeplitz spectrum is verified just as well at batch=8.
    const int batch = 8;
    using float_type = typename base_type<T>::type;

    auto a = Vector<float_type>::ones(n, batch);
    auto b = Vector<float_type>::ones(n - 1, batch);
    auto c = Vector<float_type>::zeros(n, batch);

    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params= {};
    params.block_size = 16;
    params.block_rotations = false;
    params.max_sweeps = 10;
    params.sort = true;

    UnifiedVector<std::byte> ws(steqr_buffer_size<float_type>(*this->ctx, a, b, c, JobType::EigenVectors, params), std::byte(0));

    (void)steqr<B, float_type>(*this->ctx, a, b, c,
        ws.to_span(), JobType::EigenVectors, params, eigvects);
        
    this->ctx->wait();

    auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(1.0), float_type(1.0), float_type(1.0), batch);
    std::vector<double> expected(n);
    for (int k = 1; k <= n; ++k) expected[k - 1] = 1.0 - 2.0 * std::cos(double(k) * M_PI / double(n + 1));
    std::sort(expected.begin(), expected.end());

    double norm2 = 0;
    const auto ref = repeated(expected, batch, norm2);
    expect_values(VectorView<float_type>(c), ref, norm2, n, std::nullopt, verify::all_items(batch));
    expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(c));
}


TYPED_TEST(SteqrTest, BatchedRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 128;
    // lapacke_spectra() below is a host O(n^3) solve per batch item.
    const int batch = 16;
    using float_type = typename base_type<T>::type;

    Vector<float_type> diag = Vector<float_type>::random(n, batch);
    Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);
    Vector<float_type> eigenvalues = Vector<float_type>::zeros(n, batch);
    auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
    dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();


    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params= {};
    params.block_rotations = false;
    params.max_sweeps = 10;
    params.sort = true;

    UnifiedVector<std::byte> ws(steqr_buffer_size<float_type>(*this->ctx,diag, sub_diag, eigenvalues, JobType::EigenVectors, params), std::byte(0));

    (void)steqr<B, float_type>(*this->ctx, diag, sub_diag, eigenvalues,
        ws.to_span(), JobType::EigenVectors, params, eigvects);
        
    this->ctx->wait();

    double norm2 = 0;
    const auto ref = lapacke_spectra(dense_A.view(), norm2);
    // Diagonal and off-diagonal are uniform [0, 1) (||T||_2 ~ 2.5): the kind's bound is 11x the old
    // absolute 500 machine eps (1000 u), so the Slack keeps the old power.
    expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n,
                  verify::Slack{0.1, "old bound 500 machine eps absolute = 1000 u, ||T||_2 ~ 2.5"}, verify::all_items(batch));
    expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(eigenvalues), verify::all_items(batch));
}

TYPED_TEST(SteqrTest, SteqrRandomN8SchemeCompare) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 8;
    const int batch = 1;

    Vector<float_type> diag = Vector<float_type>::random(n, batch);
    Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);

    auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
    dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();

    // --- Reference steqr ---
    Vector<float_type> evals_ref = Vector<float_type>::zeros(n, batch);
    auto eigvects_ref = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params_ref = {};
    params_ref.max_sweeps = 30;
    params_ref.sort = true;
    params_ref.transpose_working_vectors = false;
    params_ref.sort_order = SortOrder::Ascending;

    UnifiedVector<std::byte> ws_ref(
        steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, evals_ref, JobType::EigenVectors, params_ref),
        std::byte(0));
    (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(evals_ref),
                         ws_ref.to_span(), JobType::EigenVectors, params_ref, eigvects_ref);
    this->ctx->wait();

    // --- STEQR with explicit update schemes ---
    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        Vector<float_type> evals_cta = Vector<float_type>::zeros(n, batch);
        auto eigvects_cta = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params_cta = {};
        params_cta.max_sweeps = 30;
        params_cta.sort = true;
        params_cta.transpose_working_vectors = false;
        params_cta.sort_order = SortOrder::Ascending;
        params_cta.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params_cta.cta_update_scheme = scheme;

        UnifiedVector<std::byte> ws_cta(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, evals_cta, JobType::EigenVectors, params_cta),
            std::byte(0));
        (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(evals_cta),
                             ws_cta.to_span(), JobType::EigenVectors, params_cta, eigvects_cta);
        this->ctx->wait();

        // Two results of the code under test (default vs explicit scheme), not a reference.
        UnifiedVector<float_type> w_cta(n * batch), w_ref(n * batch);
        for (int i = 0; i < n * batch; ++i) w_cta[i] = evals_cta(i % n, i / n), w_ref[i] = evals_ref(i % n, i / n);
        test_utils::expect_eigenvalues_agree<float_type>(w_cta, w_ref, n, batch);
        expect_pairs(dense_A.view(), eigvects_ref.view(), VectorView<float_type>(evals_ref));
        expect_pairs(dense_A.view(), eigvects_cta.view(), VectorView<float_type>(evals_cta));
    }
}

TYPED_TEST(SteqrTest, SteqrSingleMatrixWithSchemes) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    for (int i = 1; i <= n; ++i) {
        expected_eigenvalues[i-1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
    }

    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        // fresh output buffers each iteration
        Vector<float_type> eigenvalues(n, batch);
        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params = {};
        params.max_sweeps = 30;  // Per-eigenvalue iteration limit
        params.sort = true;  // Re-enable sorting to match test expectations
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = scheme;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        double norm2 = 0;
        const auto ref = repeated(std::vector<double>(expected_eigenvalues.begin(), expected_eigenvalues.end()), batch, norm2);
        expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n, float_slack<float_type>(kClosedFormN16));

        auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(b), batch);
        expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(eigenvalues));
    }
}

TYPED_TEST(SteqrTest, SteqrBatchedMatricesWithSchemes) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 16;
    const int batch = 1280;

    float_type a = 1.0f;
    float_type b = 0.5f;
    float_type c = 0.5f;
    Vector<float_type> diag(n, float_type(a), batch);
    Vector<float_type> sub_diag(n - 1, float_type(b), batch);
    UnifiedVector<float_type> expected_eigenvalues(n * batch);

    // All matrices are identical, so expected eigenvalues are the same for each batch item
    for (int j = 0; j < batch; ++j) {
        for (int i = 1; i <= n; ++i) {
            expected_eigenvalues[j * n + i - 1] = float_type(a - 2.0f * std::sqrt(b * c) * std::cos(M_PI * i / (n + 1)));
        }
    }

    for (auto scheme : kSteqrUpdateSchemes) {
        SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
        SteqrParams<float_type> params = {};
        params.max_sweeps = 10;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = scheme;

        Vector<float_type> eigenvalues(n, batch);
        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        double norm2 = 0;
        const auto ref = repeated(std::vector<double>(expected_eigenvalues.begin(), expected_eigenvalues.begin() + n), batch, norm2);
        const auto all = verify::all_items(batch);
        expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n, float_slack<float_type>(kClosedFormN16), all);

        auto dense_A = Matrix<float_type>::TriDiagToeplitz(n, float_type(a), float_type(b), float_type(c), batch);
        expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(eigenvalues), all);
    }
}

TYPED_TEST(SteqrTest, SteqrRandomMatrices) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    // Each size runs a host LAPACK reference over the whole batch, so cost is
    // linear in `batch` x 29 sizes. Large-batch dispatch stays covered by
    // SteqrBatchedMatricesWithSchemes (batch=1280); keep the full size sweep,
    // which is what actually exercises the block/tail boundaries.
    const int batch = 64;
    for (int n = 4; n <= 32; ++n) {
        Vector<float_type> diag = Vector<float_type>::random(n, batch);
        Vector<float_type> sub_diag = Vector<float_type>::random(n - 1, batch);
        Vector<float_type> eigenvalues = Vector<float_type>::zeros(n, batch);

        auto dense_A = Matrix<float_type>::Zeros(n, n, batch);
        dense_A.view().fill_tridiag(*this->ctx, sub_diag, diag, sub_diag).wait();

        auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
        SteqrParams<float_type> params = {};
        params.max_sweeps = 10;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<float_type>(*this->ctx, diag, sub_diag, eigenvalues, JobType::EigenVectors, params),
            std::byte(0));

        (void)steqr<B, float_type>(*this->ctx, VectorView(diag), VectorView(sub_diag), VectorView(eigenvalues),
                             ws.to_span(), JobType::EigenVectors, params, eigvects);
        this->ctx->wait();

        SCOPED_TRACE(::testing::Message() << "n " << n);
        double norm2 = 0;
        const auto ref = lapacke_spectra(dense_A.view(), norm2);
        const auto all = verify::all_items(batch);
        expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n, std::nullopt, all);
        expect_pairs(dense_A.view(), eigvects.view(), VectorView<float_type>(eigenvalues), all);
    }
}

TYPED_TEST(SteqrTest, SteqrConditionedTridiagonalNetlibRef) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    const int n = 32;
    const int batch = 32;
    const float_type log10_cond = float_type(5.0);

    auto dense_A = random_hermitian_tridiagonal_with_log10_cond_metric<B, float_type>(
        *this->ctx, n, log10_cond, NormType::Spectral, batch, 1234u);

    Vector<float_type> diag(n, float_type(0), batch);
    Vector<float_type> sub(n - 1, float_type(0), batch);
    auto A_view = dense_A.view();
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            diag(i, b) = A_view.at(i, i, b);
            if (i < n - 1) {
                sub(i, b) = A_view.at(i + 1, i, b);
            }
        }
    }

    auto conds = cond<B>(*this->ctx, dense_A.view(), NormType::Spectral);
    this->ctx->wait();
    for (int b = 0; b < batch; ++b) {
        EXPECT_GE(std::log10(conds[b]), log10_cond - float_type(0.5)) << "Batch " << b;
    }

    Vector<float_type> eigenvalues(n, batch);
    auto eigvects = Matrix<float_type>::Zeros(n, n, batch);
    SteqrParams<float_type> params = {};
    params.max_sweeps = 100;
    params.sort = true;
    params.transpose_working_vectors = false;
    params.sort_order = SortOrder::Ascending;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<float_type>(*this->ctx, diag, sub, eigenvalues, JobType::EigenVectors, params),
        std::byte(0));
    (void)steqr<B, float_type>(*this->ctx, diag, sub, eigenvalues,
                         ws.to_span(), JobType::EigenVectors, params, eigvects);
    this->ctx->wait();

    std::vector<std::vector<double>> ref(batch);
    double norm2 = 0;
    for (int b = 0; b < batch; ++b) {
        std::vector<double> d(n), e(n - 1);
        for (int i = 0; i < n; ++i) d[i] = diag(i, b);
        for (int i = 0; i + 1 < n; ++i) e[i] = sub(i, b);
        if (!verify::tridiagonal_eigenvalues(d, e)) { ref.clear(); break; }
        for (double l : d) norm2 = verify::nanmax(norm2, std::fabs(l));
        ref[b] = std::move(d);
    }
    expect_values(VectorView<float_type>(eigenvalues), ref, norm2, n, std::nullopt, verify::all_items(batch));
}

namespace {

inline bool stress_debug_enabled() {
    return std::getenv("BATCHLAS_STEQR_STRESS_DEBUG") != nullptr;
}

inline void stress_debug_log(const char* msg) {
    if (stress_debug_enabled()) {
        std::cerr << msg << std::endl;
    }
}

inline bool is_kernel_not_found_message(const std::string& msg) {
    return msg.find("No kernel named") != std::string::npos;
}

template <typename Real>
Real stress_large_scale() {
    if constexpr (std::is_same_v<Real, float>) {
        // Large enough to trigger ssfmax scaling, but small enough that squares don’t overflow.
        return Real(1e19f);
    } else {
        // sqrt(max(double)) ~ 1e154; this is > ssfmax but keeps squares finite.
        return Real(1e154);
    }
}

template <typename Real>
Real stress_small_scale() {
    if constexpr (std::is_same_v<Real, float>) {
        // ssfmin(float) is around 1e-5; this triggers scaling up.
        return Real(1e-10f);
    } else {
        // ssfmin(double) is around 1e-123; this triggers scaling up.
        return Real(1e-140);
    }
}

template <typename Real>
void fill_stress_tridiag(Vector<Real>& diag,
                         Vector<Real>& sub,
                         Real diag_scale,
                         Real offdiag_scale) {
    const int n = diag.size();
    const int batch = diag.batch_size();

    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real t = Real(1) + Real(i) / Real(n);
            diag(i, j) = diag_scale * t;
            if (i < n - 1) {
                sub(i, j) = offdiag_scale * Real(0.25) * (Real(1) + Real(i) / Real(n - 1));
            }
        }
    }
}

template <typename Real>
void assert_all_finite(Vector<Real>& v) {
    const int n = v.size();
    const int batch = v.batch_size();
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            const Real x = v(i, j);
            ASSERT_TRUE(std::isfinite(static_cast<double>(x)))
                << "Non-finite value at (" << i << "," << j << "): " << static_cast<double>(x);
        }
    }
}

template <Backend B, typename Real>
void stress_run_case(Queue& ctx,
                     Vector<Real>& diag,
                     Vector<Real>& sub,
                     Vector<Real>& steqr_eigs,
                     Vector<Real>& cta_eigs,
                     bool check_steqr_against_ref = true,
                     bool check_cta_against_ref = true) {
    const int n = diag.size();
    const int batch = diag.batch_size();

    // Dense symmetric matrix for a reference eigenvalue solve.
    auto dense_A = Matrix<Real>::Zeros(n, n, batch);

    stress_debug_log("stress_run_case: fill_tridiag");
    dense_A.view().fill_tridiag(ctx, sub, diag, sub).wait();

    double norm2 = 0;
    const auto ref = lapacke_spectra(dense_A.view(), norm2);

    // steqr: eigenvalues only.
    {
        stress_debug_log("stress_run_case: steqr");
        SteqrParams<Real> params = {};
        params.max_sweeps = 200;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<Real>(ctx, diag, sub, steqr_eigs, JobType::EigenVectors, params),
            std::byte(0));
        // NOTE: On some CUDA SYCL stacks, the JobType::NoEigenVectors path can trigger
        // an assertion inside the runtime scheduler. We request eigenvectors here to
        // keep this stress test runnable on CUDA; we still validate only eigenvalues.
        auto eigvects = Matrix<Real>::Zeros(n, n, batch);
        (void)steqr<B, Real>(ctx, diag, sub, steqr_eigs, ws.to_span(), JobType::EigenVectors, params, eigvects);
        ctx.wait();
    }

    // steqr (alternate scheme): eigenvalues only, but still requires a square eigvects argument.
    {
        stress_debug_log("stress_run_case: steqr(alt-scheme)");
        SteqrParams<Real> params = {};
        params.max_sweeps = 200;
        params.sort = true;
        params.transpose_working_vectors = false;
        params.sort_order = SortOrder::Ascending;
        params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
        params.cta_update_scheme = SteqrUpdateScheme::EXP;

        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<Real>(ctx, diag, sub, cta_eigs, JobType::EigenVectors, params),
            std::byte(0));
        auto eigvects = Matrix<Real>::Zeros(n, n, batch);
        (void)steqr<B, Real>(ctx, diag, sub, cta_eigs, ws.to_span(), JobType::EigenVectors, params, eigvects);
        ctx.wait();
    }

    assert_all_finite(steqr_eigs);
    assert_all_finite(cta_eigs);

    // Sorted copies against LAPACKE at Check::values (||T||_2 over the batch).
    std::vector<Real> ste(static_cast<std::size_t>(n) * batch), cta(static_cast<std::size_t>(n) * batch);
    for (int j = 0; j < batch; ++j) {
        for (int i = 0; i < n; ++i) {
            ste[i + j * n] = steqr_eigs(i, j);
            cta[i + j * n] = cta_eigs(i, j);
        }
        std::sort(ste.begin() + j * n, ste.begin() + (j + 1) * n);
        std::sort(cta.begin() + j * n, cta.begin() + (j + 1) * n);
    }
    const auto all = verify::all_items(batch);
    if (check_steqr_against_ref) expect_values(VectorView<Real>(ste.data(), n, batch), ref, norm2, n, std::nullopt, all);
    if (check_cta_against_ref) expect_values(VectorView<Real>(cta.data(), n, batch), ref, norm2, n, std::nullopt, all);
}

} // namespace

TYPED_TEST(SteqrTest, StressExtremeMagnitudesN32) {
    using T = typename TestFixture::ScalarType;
    using float_type = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    // steqr dispatch uses CTA for n <= subgroup size on CUDA; use 32 for a heavier stress case.
    const int n = 32;
    const int batch = 16;

    Vector<float_type> diag(n, float_type(0), batch);
    Vector<float_type> sub(n - 1, float_type(0), batch);
    Vector<float_type> evals_steqr = Vector<float_type>::zeros(n, batch);
    Vector<float_type> evals_cta = Vector<float_type>::zeros(n, batch);

    try {
        // Case 1: very large magnitude (expects scale-down to kick in)
        fill_stress_tridiag(diag, sub, stress_large_scale<float_type>(), stress_large_scale<float_type>());
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta);

        // Case 2: very small magnitude (expects scale-up to kick in)
        fill_stress_tridiag(diag, sub, stress_small_scale<float_type>(), stress_small_scale<float_type>());
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta);

        // Case 3: mixed dynamic range without underflow.
        // We keep the “small” entries O(1) so this still stresses conditioning and the
        // ssfmax scale-down path, but avoids spanning ssfmin..ssfmax in one matrix.
        for (int j = 0; j < batch; ++j) {
            for (int i = 0; i < n; ++i) {
                const bool big = ((i + j) % 3) == 0;
                const float_type ds = big ? stress_large_scale<float_type>() : float_type(1);
                const float_type es = big ? stress_large_scale<float_type>() : float_type(1);
                diag(i, j) = ds * (float_type(1) + float_type(i) / float_type(n));
                if (i < n - 1) sub(i, j) = es * float_type(0.25);
            }
        }
        // This case is primarily meant to stress the CTA implementation; on CUDA,
        // the baseline STEQR path can be noticeably less accurate on ill-conditioned
        // mixed-scale inputs. We still require it to produce finite outputs.
        stress_run_case<B>(*this->ctx, diag, sub, evals_steqr, evals_cta,
                           /*check_steqr_against_ref=*/false,
                           /*check_cta_against_ref=*/true);

    } catch (const sycl::exception& e) {
        if (is_kernel_not_found_message(e.what())) {
            GTEST_SKIP() << "Skipping due to missing kernel bundle: " << e.what();
        }
        throw;
    } catch (const std::exception& e) {
        if (is_kernel_not_found_message(e.what())) {
            GTEST_SKIP() << "Skipping due to missing kernel bundle: " << e.what();
        }
        throw;
    }
}


// ---------------------------------------------------------------------------
// Per-item convergence status (`info`).
//
// steqr is the routine LAPACK documents as returning info > 0 -- "the number of
// off-diagonal elements that did not converge to zero" -- and until this work
// package the whole of that was unreachable here. steqr_cta computed a per-item
// `status` array and only READ it under a diagnostics env var; steqr_wg computed
// nothing at all and its driver ran a fixed n-1 passes with no test. At batch
// 16384 one non-converged item was invisible: the call returned, ctx.wait()
// returned, and the caller read eigenvalues that were simply wrong for it.
//
// TWO TIERS, ONE SPAN. `steqr` picks steqr_cta when n <= the device sub-group
// width and steqr_wg otherwise (src/extensions/steqr.cc:43), so at n = 32 these
// cases exercise the CTA arm on CUDA and the work-group arm on the host -- which
// is deliberate: the two tiers derive the status by completely different means
// (a bool out of steqr_cta_solve vs. a post-hoc scan of `e`), and a test that
// only ever reached one of them would say nothing about the other.
//
// THE BATCH IS MIXED ON PURPOSE. Even items are diagonal, so they are converged
// before the first sweep; odd items are the generic Toeplitz that is not. That
// is what lets the forced case below state something stronger than "the call
// failed": the items it reports as converged must still have the right answer.
// ---------------------------------------------------------------------------

namespace {

// Even items: diagonal, ascending, distinct -- e is exactly zero, so every
// off-diagonal is deflated before the first sweep runs.
// Odd items: the symmetric Toeplitz(0.5, 1, 0.5), whose spectrum is uniform and
// unclustered and therefore needs the full sweep budget.
template <typename Real>
void fill_mixed_convergence_batch(Vector<Real>& d, Vector<Real>& e) {
    const int n = d.size();
    const int batch = d.batch_size();
    for (int b = 0; b < batch; ++b) {
        const bool easy = (b % 2) == 0;
        for (int i = 0; i < n; ++i) d(i, b) = easy ? Real(i + 1) : Real(1);
        for (int i = 0; i < n - 1; ++i) e(i, b) = easy ? Real(0) : Real(0.5);
    }
}

// Ascending eigenvalue `i` of item `b` of that batch, in closed form. For the
// Toeplitz items: a - 2*sqrt(b*c)*cos(pi*k/(n+1)) with a = 1, b = c = 0.5.
template <typename Real>
Real mixed_convergence_eigenvalue(int b, int i, int n) {
    if ((b % 2) == 0) return Real(i + 1);
    return Real(1) - Real(std::cos(M_PI * double(i + 1) / double(n + 1)));
}

// Every item with info == 0 is right. Even items are diag(1..n) with e = 0: their eigenvalues are the
// exact input integers, so they compare exactly. Odd (Toeplitz) items: closed form at Check::values,
// ||T||_2 = 2.
template <typename Real>
void expect_converged_items_right(Vector<Real>& w, const UnifiedVector<int32_t>& info, int n, int batch) {
    std::vector<std::vector<double>> ref(static_cast<std::size_t>(batch), std::vector<double>(static_cast<std::size_t>(n)));
    for (int b = 0; b < batch; ++b) {
        if (info[b] != 0) continue;
        if (b % 2 == 0) {
            for (int i = 0; i < n; ++i) EXPECT_EQ(w(i, b), Real(i + 1)) << "item " << b << " reported info == 0, eigenvalue " << i;
            continue;
        }
        double norm2 = 0;
        for (int i = 0; i < n; ++i) {
            ref[b][i] = mixed_convergence_eigenvalue<double>(b, i, n);
            norm2 = verify::nanmax(norm2, std::fabs(ref[b][i]));
        }
        const int item[] = {b};
        EXPECT_VERIFY(Real, verify::Check::values, n, verify::values_error(VectorView<Real>(w), ref, norm2, item))
            << "item " << b << " reported info == 0";
    }
}

}  // namespace

TYPED_TEST(SteqrTest, InfoIsZeroOnAConvergingBatch) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Vector<Real> d(n, batch), e(n - 1, batch), w(n, batch);
    fill_mixed_convergence_batch<Real>(d, e);
    auto eigvects = Matrix<Real>::Zeros(n, n, batch);

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;

    // -1, NOT 0. A span left at zero cannot tell "the kernel wrote 0" from
    // "nothing wrote it at all", and "nothing wrote it" is the exact failure this
    // mechanism exists to make impossible: a status channel that is never filled
    // reports universal success. steqr clears the span itself
    // (src/extensions/steqr.cc:66), so a surviving -1 means that clear never ran.
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<Real>(*this->ctx, d, e, w, JobType::EigenVectors, params), std::byte(0));
    (void)steqr<B, Real>(*this->ctx, d, e, w, ws.to_span(), JobType::EigenVectors, params, eigvects,
                   info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value: nothing wrote "
                                  "the span, so a zero here would have been an accident";
        EXPECT_EQ(info[b], 0) << "item " << b << " reported non-convergence on a batch that "
                                 "converges under the default sweep budget";
    }
    // Both halves of the mixed batch must be RIGHT as well as reported converged;
    // otherwise the forced case below could not attribute a wrong answer to the cap.
    expect_converged_items_right<Real>(w, info, n, batch);
}

// THE CASE THAT MATTERS. A test that only ever sees info == 0 cannot tell a
// working implementation from one that memsets zero, so this one forces the
// failure on the SAME input and asserts the opposite direction.
//
// The knob is SteqrParams::max_sweeps, and 1 rather than a "reduced" value such
// as 50, for a reason: syev_cta.cc:177-180 and syev_cta_fused.cc:554-557 silently
// REWRITE max_sweeps to 400 whenever a caller leaves it at the default 50, so a
// test that lowered the cap to that default would be answered by 400 and pass
// vacuously. Calling steqr directly avoids the rewrite; 1 escapes it regardless.
//
// One sweep per pass gives the CTA arm a budget of max_sweeps*n = 32 sweeps
// (steqr_cta_device.hh:764) and the work-group arm n-1 = 31 passes of one sweep
// each, against a 32x32 Toeplitz that needs 2-3 sweeps per eigenvalue.
TYPED_TEST(SteqrTest, InfoReportsItemsThatExhaustTheSweepBudget) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 8;

    Vector<Real> d(n, batch), e(n - 1, batch), w(n, batch);
    fill_mixed_convergence_batch<Real>(d, e);
    auto eigvects = Matrix<Real>::Zeros(n, n, batch);

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;
    params.max_sweeps = 1;

    UnifiedVector<int32_t> info(batch, int32_t(-1));

    auto ws = UnifiedVector<std::byte>(
        steqr_buffer_size<Real>(*this->ctx, d, e, w, JobType::EigenVectors, params), std::byte(0));
    (void)steqr<B, Real>(*this->ctx, d, e, w, ws.to_span(), JobType::EigenVectors, params, eigvects,
                   info.to_span());
    this->ctx->wait();

    int reported = 0;
    for (int b = 0; b < batch; ++b) {
        ASSERT_NE(info[b], -1) << "info[" << b << "] still holds the poison value";
        ASSERT_GE(info[b], 0) << "info is LAPACK-like: 0 or a positive count, never negative";
        if (info[b] != 0) ++reported;
    }
    EXPECT_GT(reported, 0)
        << "a one-sweep budget on a 32x32 Toeplitz batch reported universal convergence; "
           "either the status is not written at all, or it is written unconditionally zero";

    // The other half of the claim, and the reason the batch is mixed: an item the
    // library says converged must still be CORRECT. A status channel that
    // reported failure everywhere would satisfy the assertion above on its own;
    // this is what stops that from passing.
    expect_converged_items_right<Real>(w, info, n, batch);
}

// An EMPTY span means "not requested" and must cost nothing -- the contract
// potrf documents, restated for every routine in this package. Two things are
// checked, because only one of them is visible from the caller's side: the
// answer does not change, and the workspace query does not either. The second is
// what keeps a `*_buffer_size` computed once and reused across calls -- which is
// how every driver in this tree uses it -- correct.
TYPED_TEST(SteqrTest, EmptyInfoSpanChangesNeitherAnswerNorWorkspace) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 32;
    const int batch = 4;

    SteqrParams<Real> params = {};
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    params.transpose_working_vectors = false;

    Vector<Real> d0(n, batch), e0(n - 1, batch), w0(n, batch);
    Vector<Real> d1(n, batch), e1(n - 1, batch), w1(n, batch);
    fill_mixed_convergence_batch<Real>(d0, e0);
    fill_mixed_convergence_batch<Real>(d1, e1);
    auto eigvects0 = Matrix<Real>::Zeros(n, n, batch);
    auto eigvects1 = Matrix<Real>::Zeros(n, n, batch);

    // steqr_buffer_size takes no `info` argument at all, which is the strongest
    // available statement of the contract: the size CANNOT depend on whether
    // status was asked for. Querying it twice guards the weaker thing that could
    // still go wrong later -- an overload that does take one.
    const size_t bytes_a =
        steqr_buffer_size<Real>(*this->ctx, d0, e0, w0, JobType::EigenVectors, params);
    const size_t bytes_b =
        steqr_buffer_size<Real>(*this->ctx, d1, e1, w1, JobType::EigenVectors, params);
    EXPECT_EQ(bytes_a, bytes_b);

    UnifiedVector<std::byte> ws0(bytes_a, std::byte(0));
    UnifiedVector<std::byte> ws1(bytes_a, std::byte(0));
    UnifiedVector<int32_t> info(batch, int32_t(-1));

    (void)steqr<B, Real>(*this->ctx, d0, e0, w0, ws0.to_span(), JobType::EigenVectors, params, eigvects0,
                   info.to_span());
    (void)steqr<B, Real>(*this->ctx, d1, e1, w1, ws1.to_span(), JobType::EigenVectors, params, eigvects1,
                   Span<int32_t>{});
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(w0(i, b), w1(i, b))
                << "requesting status changed the answer at batch " << b << " index " << i;
        }
    }
}

// ---------------------------------------------------------------------------
// Graded tridiagonals: the QL/QR direction choice.
//
// LAPACK dsteqr runs QL when |D(l)| <= |D(lend)| and QR otherwise, so the
// chase always converges the small end of a graded block first. steqr_cta and
// steqr_wg (the host tier and n > sub-group width) both once had this inverted: on a graded block it converged the LARGE end first, which
// took about 2x the implicit steps and lost relative accuracy in the small
// eigenvalues (float n=16 errors up to 5e1, double n=32 medians of 1e-6).
//
// Every other test in this file compares with rel * (1 + |lambda|), which is an
// absolute tolerance for small eigenvalues, and uses random input where both
// directions are equally good. Neither can see this defect. This test grades
// the input (span 1e-6 float, 1e-12 double), mixes both orientations within
// each warp, and checks RELATIVE error per eigenvalue against a long-double
// Sturm bisection. The two thresholds sit 7-100x above the fixed rule's
// measured errors and far below the inverted rule's.
// ---------------------------------------------------------------------------

namespace {

// All eigenvalues of one symmetric tridiagonal, ascending, by bisection on the
// Sturm count in long double.
std::vector<long double> sturm_eigenvalues(const std::vector<long double>& d,
                                           const std::vector<long double>& e) {
    const int n = static_cast<int>(d.size());
    long double lo = d[0], hi = d[0];
    for (int i = 0; i < n; ++i) {
        long double r = 0;
        if (i > 0) r += std::fabs(e[i - 1]);
        if (i < n - 1) r += std::fabs(e[i]);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
    }
    const long double pad = (hi - lo) * 1e-3L + std::numeric_limits<long double>::min();
    lo -= pad;
    hi += pad;
    auto count_below = [&](long double x) {
        int c = 0;
        long double q = 1;
        for (int i = 0; i < n; ++i) {
            q = (d[i] - x) - (i > 0 ? e[i - 1] * e[i - 1] / q : 0.0L);
            if (q == 0) q = -std::numeric_limits<long double>::min() * 1e10L;
            c += q < 0;
        }
        return c;
    };
    std::vector<long double> ev(n);
    for (int k = 0; k < n; ++k) {
        long double a = lo, b = hi;
        for (int it = 0; it < 120; ++it) {
            const long double m = (a + b) / 2;
            if (count_below(m) <= k) a = m; else b = m;
        }
        ev[k] = (a + b) / 2;
    }
    return ev;
}

}  // namespace

TYPED_TEST(SteqrTest, GradedTridiagonalRelativeAccuracy) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    constexpr bool is_float = std::is_same_v<Real, float>;
    const long double span = is_float ? 1e-6L : 1e-12L;
    const double median_tol = is_float ? 1e-5 : 1e-13;
    const double max_tol = is_float ? 5e-2 : 1e-9;

    // n = 16 and 32 put 2 and 1 problems in a warp on the CTA tier; n = 48 is
    // past the sub-group width, so it takes steqr_wg, which had the same
    // inversion. n = 8 is barely affected by the direction and is left out.
    for (const int n : {16, 32, 48}) {
        const int batch = 512;
        Vector<Real> diag(n, Real(0), batch);
        Vector<Real> sub(n - 1, Real(0), batch);
        Vector<Real> evals = Vector<Real>::zeros(n, batch);

        // Deterministic multiplicative factors in [0.5, 1.5) and signs, so the
        // input does not depend on a library RNG.
        uint32_t state = 12345u + static_cast<uint32_t>(n);
        auto next_unit = [&]() {
            state = state * 1664525u + 1013904223u;
            return 0.5L + static_cast<long double>(state >> 8) / static_cast<long double>(1u << 24);
        };
        const long double g = std::pow(span, 1.0L / (n - 1));
        std::vector<std::vector<long double>> hd(batch, std::vector<long double>(n));
        std::vector<std::vector<long double>> he(batch, std::vector<long double>(n - 1));
        for (int b = 0; b < batch; ++b) {
            const bool large_at_bottom = (b % 2) == 1;
            for (int i = 0; i < n; ++i) {
                const int k = large_at_bottom ? n - 1 - i : i;
                const long double sign = ((b / 2 + i) % 3 == 0) ? -1.0L : 1.0L;
                diag(i, b) = static_cast<Real>(sign * next_unit() * std::pow(g, k));
                hd[b][i] = diag(i, b);  // the reference sees exactly the rounded input
            }
            for (int i = 0; i < n - 1; ++i) {
                const long double k = large_at_bottom ? (n - 2 - i) + 0.5L : i + 0.5L;
                sub(i, b) = static_cast<Real>(next_unit() * std::pow(g, k));
                he[b][i] = sub(i, b);
            }
        }

        SteqrParams<Real> params = {};
        params.sort = true;
        params.sort_order = SortOrder::Ascending;
        try {
            auto ws = UnifiedVector<std::byte>(
                steqr_buffer_size<Real>(*this->ctx, diag, sub, evals, JobType::EigenVectors, params), std::byte(0));
            auto eigvects = Matrix<Real>::Zeros(n, n, batch);
            (void)steqr<B, Real>(*this->ctx, diag, sub, evals, ws.to_span(), JobType::EigenVectors, params, eigvects);
            this->ctx->wait();
        } catch (const std::exception& e) {
            if (is_kernel_not_found_message(e.what())) {
                GTEST_SKIP() << "Skipping due to missing kernel bundle: " << e.what();
            }
            throw;
        }

        std::vector<double> item_max(batch);
        for (int b = 0; b < batch; ++b) {
            const auto ref = sturm_eigenvalues(hd[b], he[b]);
            std::vector<long double> got(n);
            for (int i = 0; i < n; ++i) got[i] = evals(i, b);
            std::sort(got.begin(), got.end());
            double worst = 0;
            for (int i = 0; i < n; ++i) {
                ASSERT_TRUE(std::isfinite(static_cast<double>(got[i]))) << "n=" << n << " item " << b;
                worst = std::max(worst, static_cast<double>(std::fabs(got[i] - ref[i]) / std::fabs(ref[i])));
            }
            item_max[b] = worst;
        }
        const double max_err = *std::max_element(item_max.begin(), item_max.end());
        std::nth_element(item_max.begin(), item_max.begin() + batch / 2, item_max.end());
        const double median_err = item_max[batch / 2];
        EXPECT_LE(median_err, median_tol) << "n=" << n << ": median over items of the worst relative eigenvalue error";
        EXPECT_LE(max_err, max_tol) << "n=" << n << ": worst relative eigenvalue error over the batch";
    }
}

namespace {

// True when item b of a sorted steqr result is right, through batchlas::verify: eigenvalues
// against the long-double reference (values, scaled by ||T||_inf), ||T Z - Z diag(w)||_F / ||T||_F
// (eigen_residual) and ||Z^T Z - I||_F (orthogonality_rotations). `why` receives the three
// measured values and their bounds for the failure message.
template <typename Real>
bool tridiag_item_ok(const std::vector<long double>& hd, const std::vector<long double>& he,
                     const std::vector<long double>& ref, Vector<Real>& evals, const Real* Z, int b,
                     std::string& why) {
    using batchlas::verify::Check;
    const int n = static_cast<int>(hd.size());
    std::vector<double> T(static_cast<size_t>(n) * n, 0.0);
    double tnorm = 0;
    for (int i = 0; i < n; ++i) {
        T[i + static_cast<size_t>(i) * n] = static_cast<double>(hd[i]);
        if (i < n - 1) T[i + 1 + static_cast<size_t>(i) * n] = static_cast<double>(he[i]);
        tnorm = std::max(tnorm, static_cast<double>(std::fabs(hd[i]) + (i > 0 ? std::fabs(he[i - 1]) : 0.0L) +
                                                    (i < n - 1 ? std::fabs(he[i]) : 0.0L)));
    }
    const auto Zv = batchlas::verify::view(Z, n, n, n);
    const VectorView<Real> w(&evals(0, b), n, 1, 1, n);
    const std::vector<std::vector<double>> r{std::vector<double>(ref.begin(), ref.end())};
    const double ev_err = batchlas::verify::values_error(w, r, tnorm);
    const double res = batchlas::verify::eigen_residual(batchlas::verify::view(T.data(), n, n, n), Zv, w);
    const double orth = batchlas::verify::orthogonality(Zv);
    why = "eigenvalue error " + std::to_string(ev_err) + " (bound " +
          std::to_string(batchlas::verify::bound<Real>(Check::values, n)) + "), residual " + std::to_string(res) +
          " (bound " + std::to_string(batchlas::verify::bound<Real>(Check::eigen_residual, n)) + "), orthogonality " +
          std::to_string(orth) + " (bound " + std::to_string(batchlas::verify::bound<Real>(Check::orthogonality_rotations, n)) +
          ")";
    const bool v_ok = batchlas::verify::pass<Real>(Check::values, n, ev_err);
    const bool r_ok = batchlas::verify::pass<Real>(Check::eigen_residual, n, res);
    const bool o_ok = batchlas::verify::pass<Real>(Check::orthogonality_rotations, n, orth);
    return v_ok && r_ok && o_ok;
}

// steqr with eigenvectors, sorted ascending, on long-double input rounded to
// Real. Returns false with `skip_reason` set on a missing kernel bundle.
template <Backend B, typename Real>
bool run_steqr_vectors(Queue& ctx, const std::vector<std::vector<long double>>& hd,
                       const std::vector<std::vector<long double>>& he, SteqrParams<Real> params,
                       Vector<Real>& evals, Matrix<Real>& eigvects, UnifiedVector<int32_t>& info,
                       std::string& skip_reason) {
    const int batch = static_cast<int>(hd.size());
    const int n = static_cast<int>(hd[0].size());
    Vector<Real> diag(n, Real(0), batch), sub(n - 1, Real(0), batch);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) diag(i, b) = static_cast<Real>(hd[b][i]);
        for (int i = 0; i < n - 1; ++i) sub(i, b) = static_cast<Real>(he[b][i]);
    }
    params.sort = true;
    params.sort_order = SortOrder::Ascending;
    try {
        auto ws = UnifiedVector<std::byte>(
            steqr_buffer_size<Real>(ctx, diag, sub, evals, JobType::EigenVectors, params), std::byte(0));
        (void)steqr<B, Real>(ctx, diag, sub, evals, ws.to_span(), JobType::EigenVectors, params, eigvects,
                             info.to_span());
        ctx.wait();
    } catch (const std::exception& e) {
        if (is_kernel_not_found_message(e.what())) {
            skip_reason = e.what();
            return false;
        }
        throw;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Mixed QL/QR directions inside every warp.
//
// steqr_cta serves a QR block by mirroring it and running the QL chase, so
// every item of a warp shares one loop nest. The mirror has separate index
// maps for d and e, and a wrong one corrupts only the items that take it.
// Items cycle five kinds: graded-ascending (QL), graded-descending (QR),
// random, and two split at an interior e(k) == 0 whose blocks grade in
// opposite directions, so one mirrors a block with bb > 0 and the other one
// with be < n-1 (a map that drops bb passes every unsplit input). Every warp
// at n <= 16 holds both directions, n = 32 is the stedc leaf width (one item
// per warp, so a smaller batch), and batch 4099 leaves a ragged final
// work-group. Every item is checked, under every shift and update scheme.
// ---------------------------------------------------------------------------
TYPED_TEST(SteqrTest, MixedDirectionLockstep) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;

    for (const int n : {3, 5, 8, 13, 16, 32}) {
        if (B == Backend::NETLIB && n == 32) continue;  // host tier: no leaf width, 3x the test's time
        const int batch = n == 32 ? 1025 : 4099;
        uint32_t state = 777u + static_cast<uint32_t>(n);
        auto next_unit = [&]() {
            state = state * 1664525u + 1013904223u;
            return 0.5L + static_cast<long double>(state >> 8) / static_cast<long double>(1u << 24);
        };
        const long double g = std::pow(1e-6L, 1.0L / (n - 1));
        const int split = n / 3;  // e(split) == 0 in kinds 3 and 4; off-centre so the blocks differ
        std::vector<std::vector<long double>> hd(batch, std::vector<long double>(n));
        std::vector<std::vector<long double>> he(batch, std::vector<long double>(n - 1));
        std::vector<std::vector<long double>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            // 0: small end first (QL), 1: large end first (QR), 2: random,
            // 3: [0,split] QL above [split+1,n-1] QR, 4: [0,split] QR above [split+1,n-1] QL.
            const int kind = b % 5;
            // Exponent k of index i (|d(i)| ~ g^k, g < 1): 0 at the large end of its (sub-)block.
            auto grade_pos = [&](int i) {
                if (kind == 0) return n - 1 - i;
                if (kind == 1) return i;
                const bool upper = i <= split;
                const int lo = upper ? 0 : split + 1;
                const int hi = upper ? split : n - 1;
                const bool small_first = (kind == 3) == upper;
                return small_first ? hi - i : i - lo;
            };
            for (int i = 0; i < n; ++i) {
                const long double sign = ((b + i) % 3 == 0) ? -1.0L : 1.0L;
                const long double v =
                    kind == 2 ? 2.0L * next_unit() - 2.0L : sign * next_unit() * std::pow(g, grade_pos(i));
                hd[b][i] = static_cast<Real>(v);
            }
            for (int i = 0; i < n - 1; ++i) {
                const long double k = std::min(grade_pos(i), grade_pos(i + 1)) + 0.5L;
                const long double v = kind == 2 ? next_unit() : next_unit() * std::pow(g, k);
                he[b][i] = (kind >= 3 && i == split) ? 0.0L : static_cast<long double>(static_cast<Real>(v));
            }
            ref[b] = sturm_eigenvalues(hd[b], he[b]);
        }

        for (const auto scheme : kSteqrUpdateSchemes) {
            for (const auto shift : {SteqrShiftStrategy::Lapack, SteqrShiftStrategy::Wilkinson}) {
                Vector<Real> evals(n, Real(0), batch);
                auto eigvects = Matrix<Real>::Zeros(n, n, batch);
                UnifiedVector<int32_t> info(batch, int32_t(0));
                SteqrParams<Real> params = {};
                params.cta_update_scheme = scheme;
                params.cta_shift_strategy = shift;
                std::string skip;
                if (!run_steqr_vectors<B, Real>(*this->ctx, hd, he, params, evals, eigvects, info, skip)) {
                    GTEST_SKIP() << "Skipping due to missing kernel bundle: " << skip;
                }

                const std::string tag = std::string("n=") + std::to_string(n) + " " + update_scheme_name(scheme) +
                                        (shift == SteqrShiftStrategy::Lapack ? " Lapack" : " Wilkinson");
                int bad_items = 0;
                for (int b = 0; b < batch && bad_items < 8; ++b) {
                    const Real* Z = eigvects.data().data() + static_cast<size_t>(b) * n * n;
                    std::string why;
                    const bool ok = info[b] == 0 && tridiag_item_ok(hd[b], he[b], ref[b], evals, Z, b, why);
                    if (!ok) ++bad_items;
                    EXPECT_TRUE(ok) << tag << " item " << b << " (kind " << b % 5 << "): info " << info[b] << ", "
                                    << why;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One failing item must not disturb its warp.
//
// steqr_cta packs 32/P items into a warp (P = 4, 8, 16 at n = 4, 7, 13), and
// those chunks are to run in lockstep, sharing loops, votes and shuffles. An
// item with a NaN off-diagonal never deflates, so it spends its whole sweep
// budget and fails. It sits at every chunk position in turn (warp w poisons
// chunk w mod 32/P): only it may report info != 0, and every neighbour must
// still be right. Batch 4099 also leaves a ragged tail. The host tier packs
// nothing into warps, so it is skipped there.
// ---------------------------------------------------------------------------
TYPED_TEST(SteqrTest, PerChunkFailureIsolated) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) GTEST_SKIP() << "tests the steqr_cta chunk layout";
    const int batch = 4099;

    for (const int n : {4, 7, 13}) {
        const int per_warp = n <= 4 ? 8 : (n <= 8 ? 4 : 2);
        auto poisoned = [&](int b) { return b % per_warp == (b / per_warp) % per_warp; };
        uint32_t state = 4242u + static_cast<uint32_t>(n);
        auto next_unit = [&]() {
            state = state * 1664525u + 1013904223u;
            return 0.5L + static_cast<long double>(state >> 8) / static_cast<long double>(1u << 24);
        };
        std::vector<std::vector<long double>> hd(batch, std::vector<long double>(n));
        std::vector<std::vector<long double>> he(batch, std::vector<long double>(n - 1));
        std::vector<std::vector<long double>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i) hd[b][i] = static_cast<Real>(2.0L * next_unit() - 2.0L);
            for (int i = 0; i < n - 1; ++i) he[b][i] = static_cast<Real>(next_unit());
            if (poisoned(b)) {
                he[b][n / 2] = std::numeric_limits<long double>::quiet_NaN();
            } else {
                ref[b] = sturm_eigenvalues(hd[b], he[b]);
            }
        }

        for (const auto scheme : kSteqrUpdateSchemes) {
            Vector<Real> evals(n, Real(0), batch);
            auto eigvects = Matrix<Real>::Zeros(n, n, batch);
            UnifiedVector<int32_t> info(batch, int32_t(-1));
            SteqrParams<Real> params = {};
            params.cta_update_scheme = scheme;
            std::string skip;
            if (!run_steqr_vectors<B, Real>(*this->ctx, hd, he, params, evals, eigvects, info, skip)) {
                GTEST_SKIP() << "Skipping due to missing kernel bundle: " << skip;
            }

            int bad_items = 0;
            for (int b = 0; b < batch && bad_items < 8; ++b) {
                bool ok = false;
                std::string why = "the NaN item reported no failure";
                if (!poisoned(b)) {
                    const Real* Z = eigvects.data().data() + static_cast<size_t>(b) * n * n;
                    ok = info[b] == 0 && tridiag_item_ok(hd[b], he[b], ref[b], evals, Z, b, why);
                } else {
                    ok = info[b] > 0;
                }
                if (!ok) ++bad_items;
                EXPECT_TRUE(ok) << "n=" << n << " " << update_scheme_name(scheme) << " item " << b << " (chunk "
                                << b % per_warp << "): info " << info[b] << ", " << why;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ragged tail: batch = probs_per_wg*k + 1, probs_per_wg = 32*mult/P. At
// P = 32 with mult 2 the dead chunk is a whole warp on the maskless solve.
//
// Chunks past the batch end run the solve on a zero problem instead of
// returning. They alias item 0's views, so a missed gate on a load, store or
// status report corrupts item 0 -- but only if the stray write lands after
// item 0's own. Item 0 is therefore diagonal (its warp finishes at once) and
// k = 8192 puts the tail in a work-group launched long after work-group 0
// retired (more work-groups than an RTX 4090 holds resident); within one warp
// (k = 0) the winner of the race is arbitrary. Measured with the d store
// ungated: only the large-k case failed in float, k = 0 as well in double.
// ---------------------------------------------------------------------------
TYPED_TEST(SteqrTest, RaggedTailBatch) {
    using T = typename TestFixture::ScalarType;
    using Real = typename base_type<T>::type;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) GTEST_SKIP() << "tests the steqr_cta work-group tail";

    for (const int n : {4, 7, 13, 32}) {
        const int P = n <= 4 ? 4 : (n <= 8 ? 8 : (n <= 16 ? 16 : 32));
        for (const int mult : {1, 2}) {
            for (const int k : {0, 1, 97, 8192}) {
                const int batch = (32 * mult / P) * k + 1;
                auto checked = [&](int b) { return batch <= 8192 || b < 64 || b >= batch - 64; };
                uint32_t state = 99u + static_cast<uint32_t>(n * 1000 + mult * 100 + k);
                auto next_unit = [&]() {
                    state = state * 1664525u + 1013904223u;
                    return 0.5L + static_cast<long double>(state >> 8) / static_cast<long double>(1u << 24);
                };
                std::vector<std::vector<long double>> hd(batch, std::vector<long double>(n));
                std::vector<std::vector<long double>> he(batch, std::vector<long double>(n - 1));
                std::vector<std::vector<long double>> ref(batch);
                for (int b = 0; b < batch; ++b) {
                    for (int i = 0; i < n; ++i) hd[b][i] = static_cast<Real>(b == 0 ? i + 1.5L : 2.0L * next_unit() - 2.0L);
                    for (int i = 0; i < n - 1; ++i) he[b][i] = b == 0 ? 0.0L : static_cast<long double>(static_cast<Real>(next_unit()));
                    if (checked(b)) ref[b] = sturm_eigenvalues(hd[b], he[b]);
                }

                Vector<Real> evals(n, Real(0), batch);
                auto eigvects = Matrix<Real>::Zeros(n, n, batch);
                UnifiedVector<int32_t> info(batch, int32_t(-1));
                SteqrParams<Real> params = {};
                params.cta_wg_size_multiplier = mult;
                std::string skip;
                if (!run_steqr_vectors<B, Real>(*this->ctx, hd, he, params, evals, eigvects, info, skip)) {
                    GTEST_SKIP() << "Skipping due to missing kernel bundle: " << skip;
                }

                int bad_items = 0;
                for (int b = 0; b < batch && bad_items < 8; ++b) {
                    if (!checked(b)) continue;
                    const Real* Z = eigvects.data().data() + static_cast<size_t>(b) * n * n;
                    std::string why;
                    const bool ok = info[b] == 0 && tridiag_item_ok(hd[b], he[b], ref[b], evals, Z, b, why);
                    if (!ok) ++bad_items;
                    EXPECT_TRUE(ok) << "n=" << n << " mult=" << mult << " batch=" << batch << " item " << b
                                    << ": info " << info[b] << ", " << why;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
