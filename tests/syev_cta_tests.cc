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
#include <limits>
#include <type_traits>

using namespace batchlas;

namespace {

template <typename Real>
Real tol_eig_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(test_utils::tolerance<float>());
	return Real(test_utils::tolerance<double>());
}

template <typename Real>
Real tol_ortho_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(test_utils::tolerance<float>());
	return Real(test_utils::tolerance<double>());
}

template <typename Real>
Real tol_resid_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(test_utils::tolerance<float>() * 1e1);
	return Real(test_utils::tolerance<double>() * 1e1);
}

template <typename Scalar>
using RealOf = typename base_type<Scalar>::type;

template <typename Scalar>
static RealOf<Scalar> abs_val(const Scalar& x) {
	using Real = RealOf<Scalar>;
	return static_cast<Real>(std::abs(x));
}

template <typename Scalar>
static RealOf<Scalar> norm2_val(const Scalar& x) {
	using Real = RealOf<Scalar>;
	if constexpr (std::is_same_v<Scalar, Real>) {
		return x * x;
	} else {
		return static_cast<Real>(std::norm(x));
	}
}

template <typename Scalar>
static Scalar conj_val(const Scalar& x) {
	if constexpr (std::is_same_v<Scalar, RealOf<Scalar>>) {
		return x;
	} else {
		return std::conj(x);
	}
}

template <typename Scalar>
static void check_orthonormal_columns(const MatrixView<Scalar, MatrixFormat::Dense>& V,
									  const UnifiedVector<RealOf<Scalar>>& W /*unused*/, // keep signature uniform
									  RealOf<Scalar> tol) {
	static_cast<void>(W);
	using Real = RealOf<Scalar>;
	const int n = V.rows();
	Real max_err = Real(0);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			Scalar dot = Scalar(0);
			for (int r = 0; r < n; ++r) {
				dot += conj_val(V(r, i, 0)) * V(r, j, 0);
			}
			const Real target = (i == j) ? Real(1) : Real(0);
			max_err = std::max(max_err, abs_val(dot - Scalar(target)));
		}
	}
	EXPECT_LE(max_err, tol) << "max |V^H V - I| = " << max_err;
}

template <typename Scalar>
static void check_eigen_residual(const MatrixView<Scalar, MatrixFormat::Dense>& A0,
								 const MatrixView<Scalar, MatrixFormat::Dense>& V,
								 const UnifiedVector<RealOf<Scalar>>& W,
								 RealOf<Scalar> tol) {
	using Real = RealOf<Scalar>;
	const int n = A0.rows();

	Real a_norm2 = Real(0);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			a_norm2 += norm2_val(A0(i, j, 0));
		}
	}
	const Real a_norm = std::sqrt(a_norm2);

	Real r_norm2 = Real(0);
	for (int j = 0; j < n; ++j) {
		const Real wj = W[static_cast<std::size_t>(j)];
		for (int i = 0; i < n; ++i) {
			Scalar sum = Scalar(0);
			for (int k = 0; k < n; ++k) {
				sum += A0(i, k, 0) * V(k, j, 0);
			}
			sum -= Scalar(wj) * V(i, j, 0);
			r_norm2 += norm2_val(sum);
		}
	}

	const Real r_norm = std::sqrt(r_norm2);
	const Real denom = (a_norm > Real(0)) ? (a_norm * Real(n)) : Real(1);
	const Real rel = r_norm / denom;
	EXPECT_LE(rel, tol) << "relative residual ||AV - VΛ||/(||A||*n) = " << rel;
}

template <typename T, Backend B>
struct SyevCtaConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevCtaTestTypes = ::testing::Types<
	SyevCtaConfig<float, Backend::CUDA>,
	SyevCtaConfig<double, Backend::CUDA>,
	SyevCtaConfig<std::complex<float>, Backend::CUDA>,
	SyevCtaConfig<std::complex<double>, Backend::CUDA>>;
#else
using SyevCtaTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SyevCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevCtaTest, SyevCtaTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SyevCtaTest, EigenvaluesOnlyLowerMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 128;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n*batch));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n*batch));

	// Reference (CPU LAPACKE)
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}

	// CTA pipeline
	{
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::NoEigenVectors));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_cta.to_span()).wait();
	}

	const Real tol = tol_eig_for<Real>();
	for (int j = 0; j < batch; ++j) {
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(W_cta[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)],
						W_ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)],
						tol) << "eigenvalue mismatch i=" << i << " batch=" << j;
		}
	}
}

TYPED_TEST(SyevCtaTest, EigenvectorsLowerResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/456);
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n));

	// Reference eigenvalues (CPU LAPACKE). We compute eigenvectors too just to ensure it runs,
	// but we validate the CTA eigenvectors via residual + orthogonality.
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}

	{
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span()).wait();
	}

	const Real tol_w = tol_eig_for<Real>();
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(W_cta[static_cast<std::size_t>(i)], W_ref[static_cast<std::size_t>(i)], tol_w) << "eigenvalue mismatch i=" << i;
	}

	check_orthonormal_columns(A_cta.view(), W_cta, tol_ortho_for<Real>());
	check_eigen_residual(A0.view(), A_cta.view(), W_cta, tol_resid_for<Real>());
}

TYPED_TEST(SyevCtaTest, EigenvectorsUpperResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/789);
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n));

	// Reference (CPU LAPACKE)
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Upper));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Upper, ws_ref.to_span()).wait();
	}

	// CTA pipeline (Upper)
	{
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Upper, ws_cta.to_span()).wait();
	}

	const Real tol_w = tol_eig_for<Real>();
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(W_cta[static_cast<std::size_t>(i)], W_ref[static_cast<std::size_t>(i)], tol_w) << "eigenvalue mismatch i=" << i;
	}

	check_orthonormal_columns(A_cta.view(), W_cta, tol_ortho_for<Real>());
	check_eigen_residual(A0.view(), A_cta.view(), W_cta, tol_resid_for<Real>());
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
