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
	if constexpr (std::is_same_v<Real, float>) return Real(2e-3f);
	return Real(5e-10);
}

template <typename Real>
Real tol_ortho_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(5e-3f);
	return Real(5e-10);
}

template <typename Real>
Real tol_resid_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(2e-2f);
	return Real(5e-9);
}

template <typename Scalar>
using RealOf = typename base_type<Scalar>::type;

template <typename Scalar>
inline constexpr bool is_complex_scalar_v =
    std::is_same_v<std::remove_cv_t<std::remove_reference_t<Scalar>>, std::complex<RealOf<Scalar>>>;

template <typename Scalar>
static RealOf<Scalar> abs_val(const Scalar& x) {
	using Real = RealOf<Scalar>;
	return static_cast<Real>(std::abs(x));
}

template <typename Scalar>
static Scalar conj_val(const Scalar& x) {
	if constexpr (is_complex_scalar_v<Scalar>) {
		return std::conj(x);
	} else {
		return x;
	}
}

template <typename Scalar>
static void check_orthonormal_columns(const MatrixView<Scalar, MatrixFormat::Dense>& V,
										  const UnifiedVector<RealOf<Scalar>>& W,
										  RealOf<Scalar> tol) {
	using Real = RealOf<Scalar>;
	const int n = V.rows();

	// Check V^H V ~= I
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			Scalar dot = Scalar(0);
			for (int k = 0; k < n; ++k) {
				dot += conj_val(V(k, i)) * V(k, j);
			}
			const Scalar expected = (i == j) ? Scalar(1) : Scalar(0);
			EXPECT_LE(abs_val(dot - expected), tol) << "(i,j)= (" << i << "," << j << ")";
		}
	}

	(void)W;
}

template <typename Scalar>
static void check_eigen_residual(const MatrixView<Scalar, MatrixFormat::Dense>& A0,
									const MatrixView<Scalar, MatrixFormat::Dense>& V,
									const UnifiedVector<RealOf<Scalar>>& W,
									RealOf<Scalar> tol) {
	using Real = RealOf<Scalar>;
	const int n = A0.rows();

	// For each eigenpair: ||A*v - w*v|| / ||A||
	Real normA = Real(0);
	for (int c = 0; c < n; ++c) {
		for (int r = 0; r < n; ++r) {
			normA = std::max(normA, abs_val(A0(r, c)));
		}
	}
	if (normA == Real(0)) normA = Real(1);

	for (int j = 0; j < n; ++j) {
		const Real w = W[j];
		Real max_res = Real(0);
		for (int i = 0; i < n; ++i) {
			Scalar avi = Scalar(0);
			for (int k = 0; k < n; ++k) {
				avi += A0(i, k) * V(k, j);
			}
			const Scalar r = avi - Scalar(w) * V(i, j);
			max_res = std::max(max_res, abs_val(r));
		}
		EXPECT_LE(max_res / normA, tol) << "eigenvector col=" << j;
	}
}

template <typename T, Backend B>
struct SyevBlockedConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevBlockedTestTypes = ::testing::Types<
	SyevBlockedConfig<float, Backend::CUDA>,
	SyevBlockedConfig<double, Backend::CUDA>,
	SyevBlockedConfig<std::complex<float>, Backend::CUDA>,
	SyevBlockedConfig<std::complex<double>, Backend::CUDA>>;
#else
using SyevBlockedTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SyevBlockedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevBlockedTest, SyevBlockedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SyevBlockedTest, EigenvaluesOnlyLowerMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 96;      // ensure sytrd_blocked path
	const int batch = 16;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 123);
	Matrix<Scalar, MatrixFormat::Dense> A_blk = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_blk = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	// Reference (CPU LAPACKE)
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx,
															A_ref.view(),
															W_ref.to_span(),
															JobType::NoEigenVectors,
															Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}

	// Blocked pipeline
	{
		StedcParams<Real> params;
		params.recursion_threshold = 32;
		auto ws_blk = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(*this->ctx,
																A_blk.view(),
																JobType::NoEigenVectors,
																Uplo::Lower,
																/*sytrd_block_size=*/32,
																/*ormqr_block_size=*/32,
																params));
		syev_blocked<B, Scalar>(*this->ctx,
						A_blk.view(),
						W_blk.to_span(),
						JobType::NoEigenVectors,
						Uplo::Lower,
						ws_blk.to_span(),
						/*sytrd_block_size=*/32,
						/*ormqr_block_size=*/32,
						params).wait();
	}

	const Real tol = tol_eig_for<Real>();
	for (int j = 0; j < batch; ++j) {
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(W_blk[i + j * n], W_ref[i + j * n], tol) << "(i,b)= (" << i << "," << j << ")";
		}
	}
}

TYPED_TEST(SyevBlockedTest, EigenvectorsLowerResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 96;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 456);
	Matrix<Scalar, MatrixFormat::Dense> A_blk = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_blk = UnifiedVector<Real>(static_cast<std::size_t>(n));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n));

	// Reference eigenvalues (CPU LAPACKE)
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx,
															A_ref.view(),
															W_ref.to_span(),
															JobType::EigenVectors,
															Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}

	{
		StedcParams<Real> params;
		params.recursion_threshold = 32;
		auto ws_blk = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(*this->ctx,
																A_blk.view(),
																JobType::EigenVectors,
																Uplo::Lower,
																/*sytrd_block_size=*/32,
																/*ormqr_block_size=*/32,
																params));
		syev_blocked<B, Scalar>(*this->ctx,
						A_blk.view(),
						W_blk.to_span(),
						JobType::EigenVectors,
						Uplo::Lower,
						ws_blk.to_span(),
						/*sytrd_block_size=*/32,
						/*ormqr_block_size=*/32,
						params).wait();
	}

	const Real tol_w = tol_eig_for<Real>();
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(W_blk[i], W_ref[i], tol_w);
	}

	check_orthonormal_columns(A_blk.view(), W_blk, tol_ortho_for<Real>());
	check_eigen_residual(A0.view(), A_blk.view(), W_blk, tol_resid_for<Real>());
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
