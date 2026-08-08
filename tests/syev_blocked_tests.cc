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
#include <cstdlib>
#include <limits>
#include <string>
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

template <typename Scalar, Backend B>
typename base_type<Scalar>::type blocked_cuda_tolerance_floor_eig() {
	using Real = typename base_type<Scalar>::type;
	if constexpr (B == Backend::CUDA && std::is_same_v<Real, double>) {
		return Real(1e-8);
	}
	return Real(0);
}

template <typename Scalar, Backend B>
typename base_type<Scalar>::type blocked_cuda_tolerance_floor_ortho() {
	using Real = typename base_type<Scalar>::type;
	if constexpr (B == Backend::CUDA && std::is_same_v<Real, double>) {
		return Real(3e-8);
	}
	return Real(0);
}

template <typename Scalar, Backend B>
typename base_type<Scalar>::type blocked_cuda_tolerance_floor_resid() {
	using Real = typename base_type<Scalar>::type;
	if constexpr (B == Backend::CUDA && std::is_same_v<Real, double>) {
		return Real(1e-7);
	}
	return Real(0);
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

#include "test_utils.hh"
using SyevBlockedTestTypes = typename test_utils::backend_types<SyevBlockedConfig>::type;

template <typename Config>
class SyevBlockedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevBlockedTest, SyevBlockedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SyevBlockedTest, EigenvaluesOnlyLowerMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	// Values mode takes a different tridiagonal solver from eigenvector mode
	// (stebz, not stedc), so it needs its own shape coverage: n=8/32 below the
	// point where Auto would route here at all but reachable by direct call,
	// n=96 the historical case, n=320 the top of the blocked values-mode region
	// (syev_saturated_provider_for_n_values). Batch shrinks with n to keep the
	// dense host reference solve cheap.
	struct Shape { int n; int batch; };
	for (const Shape s : {Shape{8, 16}, Shape{32, 16}, Shape{96, 16}, Shape{320, 4}}) {
		const int n = s.n;
		const int batch = s.batch;
		SCOPED_TRACE("n=" + std::to_string(n) + " batch=" + std::to_string(batch));

		Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 123);
		Matrix<Scalar, MatrixFormat::Dense> A_blk = A0;
		Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

		auto W_blk = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
		auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

		// Reference (CPU LAPACKE)
		{
			auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
																A_ref.view(),
																W_ref.to_span(),
																JobType::NoEigenVectors,
																Uplo::Lower));
			syev(*this->ctx,
							A_ref.view(),
							W_ref.to_span(),
							{.jobz = JobType::NoEigenVectors},
							ws_ref.to_span()).wait();
		}

		// Blocked pipeline
		{
			StedcParams<Real> params;
			params.recursion_threshold = 32;
			auto ws_blk = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(*this->ctx,
																	A_blk.view(),
																	JobType::NoEigenVectors,
																	Uplo::Lower,
																	params));
			syev_blocked<B, Scalar>(*this->ctx,
							A_blk.view(),
							W_blk.to_span(),
							JobType::NoEigenVectors,
							Uplo::Lower,
							ws_blk.to_span(),
							params).wait();
		}

		// Element-by-element, so this doubles as the ordering test: stebz must
		// return the same ascending order stedc did.
		const Real tol = std::max(tol_eig_for<Real>(), blocked_cuda_tolerance_floor_eig<Scalar, B>());
		for (int j = 0; j < batch; ++j) {
			for (int i = 0; i < n; ++i) {
				EXPECT_NEAR(W_blk[i + j * n], W_ref[i + j * n], tol) << "(i,b)= (" << i << "," << j << ")";
			}
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
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
															A_ref.view(),
															W_ref.to_span(),
															JobType::EigenVectors,
															Uplo::Lower));
		syev(*this->ctx, A_ref.view(), W_ref.to_span(), {}, ws_ref.to_span()).wait();
	}

	{
		StedcParams<Real> params;
		params.recursion_threshold = 32;
		auto ws_blk = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(*this->ctx,
																A_blk.view(),
																JobType::EigenVectors,
																Uplo::Lower,
																params));
		syev_blocked<B, Scalar>(*this->ctx,
						A_blk.view(),
						W_blk.to_span(),
						JobType::EigenVectors,
						Uplo::Lower,
						ws_blk.to_span(),
						params).wait();
	}

	const Real tol_w = std::max(tol_eig_for<Real>(), blocked_cuda_tolerance_floor_eig<Scalar, B>());
	for (int i = 0; i < n; ++i) {
		EXPECT_NEAR(W_blk[i], W_ref[i], tol_w);
	}

	check_orthonormal_columns(A_blk.view(), W_blk, std::max(tol_ortho_for<Real>(), blocked_cuda_tolerance_floor_ortho<Scalar, B>()));
	check_eigen_residual(A0.view(), A_blk.view(), W_blk, std::max(tol_resid_for<Real>(), blocked_cuda_tolerance_floor_resid<Scalar, B>()));
}

TYPED_TEST(SyevBlockedTest, TwoStageProviderEigenvaluesOnlySmoke) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;

	const int n = 128;
	const int batch = 8;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 9876);
	Matrix<Scalar, MatrixFormat::Dense> A_two_stage = A0;
	auto W_two_stage = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	const char* old_provider = std::getenv("BATCHLAS_SYEV_PROVIDER");
	const std::string old_provider_value = old_provider ? std::string(old_provider) : std::string();
	setenv("BATCHLAS_SYEV_PROVIDER", "two_stage", 1);

	{
		auto ws_two_stage = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
																								  A_two_stage.view(),
																								  W_two_stage.to_span(),
																								  JobType::NoEigenVectors,
																								  Uplo::Lower));
		syev(*this->ctx,
                                 A_two_stage.view(),
                                 W_two_stage.to_span(),
                                 {.jobz = JobType::NoEigenVectors},
                                 ws_two_stage.to_span()).wait();
	}

	if (old_provider) {
		setenv("BATCHLAS_SYEV_PROVIDER", old_provider_value.c_str(), 1);
	} else {
		unsetenv("BATCHLAS_SYEV_PROVIDER");
	}

	for (int j = 0; j < batch; ++j) {
		for (int i = 0; i < n; ++i) {
			const Real wi = W_two_stage[i + j * n];
			EXPECT_TRUE(std::isfinite(wi)) << "non-finite eigenvalue at (i,b)= (" << i << "," << j << ")";
			if (i > 0) {
				EXPECT_LE(W_two_stage[(i - 1) + j * n], wi)
					<< "eigenvalues not sorted at (i,b)= (" << i << "," << j << ")";
			}
		}
	}
}

TYPED_TEST(SyevBlockedTest, TwoStageProviderEigenvectorsSmoke) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;

	const int n = 64;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 2468);
	Matrix<Scalar, MatrixFormat::Dense> A_two_stage = A0;
	auto W_two_stage = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	const char* old_provider = std::getenv("BATCHLAS_SYEV_PROVIDER");
	const std::string old_provider_value = old_provider ? std::string(old_provider) : std::string();
	setenv("BATCHLAS_SYEV_PROVIDER", "two_stage", 1);

	{
		auto ws_two_stage = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
															  A_two_stage.view(),
															  W_two_stage.to_span(),
															  JobType::EigenVectors,
															  Uplo::Lower));
		syev(*this->ctx,
                                 A_two_stage.view(),
                                 W_two_stage.to_span(),
                                 {},
                                 ws_two_stage.to_span()).wait();
	}

	if (old_provider) {
		setenv("BATCHLAS_SYEV_PROVIDER", old_provider_value.c_str(), 1);
	} else {
		unsetenv("BATCHLAS_SYEV_PROVIDER");
	}

	for (int i = 0; i < n; ++i) {
		EXPECT_TRUE(std::isfinite(W_two_stage[i])) << "non-finite eigenvalue at i=" << i;
	}

	const Real ortho_tol = std::max(tol_ortho_for<Real>(), Real(1e-7));
	const Real resid_tol = std::max(tol_resid_for<Real>(), Real(1e-7));
	check_orthonormal_columns(A_two_stage.view(), W_two_stage, ortho_tol);
	check_eigen_residual(A0.view(), A_two_stage.view(), W_two_stage, resid_tol);
}
// n = 320 is deliberate: it is inside the 256 < n <= 512 bucket where
// sytrd_block_size_default<T> now returns a different panel width for complex
// (32) than the tuning harness value used for real types (8). Every other
// eigen test in this file runs at n <= 96, so nothing here exercised that
// bucket at all -- the panel-width change and the workspace sizing that
// depends on it were both untested.
//
// This goes through the public `syev` on Auto rather than calling syev_blocked
// directly, so it also covers the per-type routing in
// syev_saturated_provider_for_n: at n = 320 that is blocked for float, double
// and complex<float>, and the vendor for complex<double>. Whichever provider
// Auto picks, the answer must satisfy the same residual and orthogonality
// bounds.
//
// The workspace is sized by syev_buffer_size, which re-derives the panel width
// through the same sytrd_block_size_default<T>. If the query and the solve ever
// disagreed about the width, this test would fail on the sizing check rather
// than silently under-allocating.
TYPED_TEST(SyevBlockedTest, AutoEigenvectorsAtRetunedPanelWidth) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;

	const int n = 320;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 =
		Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 1357);
	Matrix<Scalar, MatrixFormat::Dense> A = A0;
	auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

	{
		auto ws = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
															A.view(),
															W.to_span(),
															JobType::EigenVectors,
															Uplo::Lower));
		syev(*this->ctx, A.view(), W.to_span(), {}, ws.to_span()).wait();
	}

	for (std::size_t i = 0; i < W.size(); ++i) {
		ASSERT_TRUE(std::isfinite(W[i])) << "non-finite eigenvalue at i=" << i;
	}

	// Eigenvalues of a symmetric/Hermitian matrix are real and ascending.
	for (int b = 0; b < batch; ++b) {
		for (int i = 1; i < n; ++i) {
			EXPECT_LE(W[b * n + i - 1], W[b * n + i] + tol_eig_for<Real>())
				<< "eigenvalues not ascending at b=" << b << " i=" << i;
		}
	}

	const Real ortho_tol = std::max(tol_ortho_for<Real>(),
									blocked_cuda_tolerance_floor_ortho<Scalar, TestFixture::BackendType>());
	const Real resid_tol = std::max(tol_resid_for<Real>(),
									blocked_cuda_tolerance_floor_resid<Scalar, TestFixture::BackendType>());

	check_orthonormal_columns(A.view(), W, ortho_tol);
	check_eigen_residual(A0.view(), A.view(), W, resid_tol);
}

// The n <= 32 range, where Auto picks among the three CTA kernels rather than a
// provider. Complex now takes a different branch there than it used to:
// complex<float> uses syev_cta_fused for n <= 8 (it used syev_cta at every n),
// and complex<double> hands n > 24 to the vendor. Neither branch was reachable
// from Auto for complex before, so neither was covered.
//
// n = 6 and n = 28 sit one on each side of those two new boundaries. The sizes
// are driven through the public `syev` so that syev_dispatch's buffer-size query
// and its solve both run syev_choose_small_kernel -- that selector reads its env
// override fresh on every call and is documented as having to agree between the
// two, which is exactly the kind of disagreement a routing change can introduce.
TYPED_TEST(SyevBlockedTest, AutoEigenvectorsSmallNKernelBoundaries) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;

	for (const int n : {6, 28}) {
		const int batch = 1;

		Matrix<Scalar, MatrixFormat::Dense> A0 =
			Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 24680 + n);
		Matrix<Scalar, MatrixFormat::Dense> A = A0;
		auto W = UnifiedVector<Real>(static_cast<std::size_t>(n));

		{
			auto ws = UnifiedVector<std::byte>(syev_buffer_size(*this->ctx,
																A.view(),
																W.to_span(),
																JobType::EigenVectors,
																Uplo::Lower));
			syev(*this->ctx, A.view(), W.to_span(), {}, ws.to_span()).wait();
		}

		for (int i = 0; i < n; ++i) {
			ASSERT_TRUE(std::isfinite(W[i])) << "non-finite eigenvalue, n=" << n << " i=" << i;
		}
		for (int i = 1; i < n; ++i) {
			EXPECT_LE(W[i - 1], W[i] + tol_eig_for<Real>())
				<< "eigenvalues not ascending, n=" << n << " i=" << i;
		}

		const Real ortho_tol = std::max(tol_ortho_for<Real>(),
										blocked_cuda_tolerance_floor_ortho<Scalar, TestFixture::BackendType>());
		const Real resid_tol = std::max(tol_resid_for<Real>(),
										blocked_cuda_tolerance_floor_resid<Scalar, TestFixture::BackendType>());
		check_orthonormal_columns(A.view(), W, ortho_tol);
		check_eigen_residual(A0.view(), A.view(), W, resid_tol);
	}
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

// --- Uplo::Upper -------------------------------------------------------------
//
// Upper had NO coverage anywhere in the syev tests before this. It also had no
// implementation: sytrd_blocked threw on it, so every Upper call fell through to the vendor.
// syev_blocked/syev_two_stage now mirror the upper triangle into the lower one and run the
// ordinary Lower pipeline (src/extensions/uplo_mirror.hh), which is what lets Auto route
// Upper input to our own providers.
//
// The test that has teeth: build a matrix whose two triangles DISAGREE, so that reading the
// wrong one gives a different spectrum. Matrix::Random(..., /*symmetric=*/true) is symmetric,
// which would make Upper and Lower trivially interchangeable and the test vacuous. Here the
// strictly-lower entries are overwritten with garbage after the reference is taken from the
// upper triangle, so a solver that reads the lower triangle without mirroring gets the wrong
// answer.
TYPED_TEST(SyevBlockedTest, UpperMatchesNetlibWithDisagreeingTriangles) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 96;
	const int batch = 8;

	Matrix<Scalar, MatrixFormat::Dense> A0 =
		Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 4242);

	// Poison the strictly-lower triangle so it no longer mirrors the upper one.
	for (int b = 0; b < batch; ++b) {
		Scalar* Ab = A0.view().data().data() + static_cast<std::size_t>(b) * A0.view().stride();
		const int ld = static_cast<int>(A0.view().ld());
		for (int c = 0; c < n; ++c) {
			for (int r = c + 1; r < n; ++r) {
				Ab[r + c * ld] = Scalar(Real(-7.5));   // garbage, deliberately not symmetric
			}
		}
	}

	Matrix<Scalar, MatrixFormat::Dense> A_ours = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_ours = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	// Reference: CPU LAPACKE, reading the UPPER triangle.
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(
			*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Upper));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(),
							  JobType::NoEigenVectors, Uplo::Upper, ws_ref.to_span()).wait();
	}

	// PROVE THE FIXTURE HAS TEETH. If the poisoning above did not take effect the matrix is
	// still symmetric, Upper and Lower are interchangeable, and this test would pass even if
	// the mirror never ran. Solve the SAME matrix reading the LOWER triangle and require a
	// different spectrum -- that is what makes the Upper comparison below meaningful.
	{
		Matrix<Scalar, MatrixFormat::Dense> A_lo = A0;
		auto W_lo = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
		auto ws_lo = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(
			*this->ctx, A_lo.view(), W_lo.to_span(), JobType::NoEigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_lo.view(), W_lo.to_span(),
							  JobType::NoEigenVectors, Uplo::Lower, ws_lo.to_span()).wait();
		Real max_gap = Real(0);
		for (int j = 0; j < batch; ++j) {
			for (int i = 0; i < n; ++i) {
				max_gap = std::max(max_gap, std::abs(W_lo[i + j * n] - W_ref[i + j * n]));
			}
		}
		ASSERT_GT(max_gap, Real(1))
			<< "fixture is vacuous: the two triangles agree, so Upper vs Lower proves nothing";
	}

	// Ours: blocked path with Uplo::Upper, which must mirror before reducing.
	{
		StedcParams<Real> params;
		params.recursion_threshold = 32;
		auto ws = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(
			*this->ctx, A_ours.view(), JobType::NoEigenVectors, Uplo::Upper, params));
		syev_blocked<B, Scalar>(*this->ctx, A_ours.view(), W_ours.to_span(),
								JobType::NoEigenVectors, Uplo::Upper, ws.to_span(), params).wait();
	}

	const Real tol = std::max(tol_eig_for<Real>(), blocked_cuda_tolerance_floor_eig<Scalar, B>());
	for (int j = 0; j < batch; ++j) {
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(W_ours[i + j * n], W_ref[i + j * n], tol)
				<< "(i,b)= (" << i << "," << j << ")";
		}
	}
}

// Same, through the two-stage provider, which has its own mirror call site.
TYPED_TEST(SyevBlockedTest, UpperTwoStageMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	if constexpr (B == Backend::NETLIB) {
		GTEST_SKIP() << "two-stage is a GPU path";
	} else {
		const int n = 128;
		const int batch = 4;

		Matrix<Scalar, MatrixFormat::Dense> A0 =
			Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 909);
		for (int b = 0; b < batch; ++b) {
			Scalar* Ab = A0.view().data().data() + static_cast<std::size_t>(b) * A0.view().stride();
			const int ld = static_cast<int>(A0.view().ld());
			for (int c = 0; c < n; ++c) {
				for (int r = c + 1; r < n; ++r) {
					Ab[r + c * ld] = Scalar(Real(3.25));
				}
			}
		}

		Matrix<Scalar, MatrixFormat::Dense> A_ours = A0;
		Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;
		auto W_ours = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
		auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

		{
			auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(
				*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Upper));
			syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(),
								  JobType::NoEigenVectors, Uplo::Upper, ws_ref.to_span()).wait();
		}
		{
			StedcParams<Real> params;
			auto ws = UnifiedVector<std::byte>(syev_two_stage_buffer_size<B, Scalar>(
				*this->ctx, A_ours.view(), JobType::NoEigenVectors, Uplo::Upper, params));
			syev_two_stage<B, Scalar>(*this->ctx, A_ours.view(), W_ours.to_span(),
									  JobType::NoEigenVectors, Uplo::Upper, ws.to_span(), params).wait();
		}

		const Real tol = std::max(tol_eig_for<Real>(), blocked_cuda_tolerance_floor_eig<Scalar, B>());
		for (int j = 0; j < batch; ++j) {
			for (int i = 0; i < n; ++i) {
				EXPECT_NEAR(W_ours[i + j * n], W_ref[i + j * n], tol)
					<< "(i,b)= (" << i << "," << j << ")";
			}
		}
	}
}
