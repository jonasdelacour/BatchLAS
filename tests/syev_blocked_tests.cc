#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"
#include "eigen_verify.hh"

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

template <typename T, Backend B>
struct SyevBlockedConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

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
	// (jobz=N rows of tuned/syev.<dtype>.<device>.txt). Batch shrinks with n to keep the
	// dense host reference solve cheap.
	struct Shape { int n; int batch; };
	for (const Shape s : {Shape{8, 16}, Shape{32, 16}, Shape{96, 16}, Shape{320, 4}}) {
		const int n = s.n;
		const int batch = s.batch;
		SCOPED_TRACE("n=" + std::to_string(n) + " batch=" + std::to_string(batch));

		Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 123);
		Matrix<Scalar, MatrixFormat::Dense> A_blk = A0;

		auto W_blk = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

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

		// LAPACKE, never the queue-dispatching syev(): its table sends jobz=N, 32 < n <= 320 to
		// blocked, so it would compare syev_blocked with itself. n = 320 alone reaches stebz's
		// wg-clamped strided slot loop. Element-by-element: this is also the ordering test.
		// float n = 320: the kind's 32 n u ||A||_2 (||A||_2 ~ 20) is 6x the old 2e-3 absolute.
		std::optional<verify::Slack> slack;
		if (std::is_same_v<Real, float> && n == 320) slack = verify::Slack{0.25, "old float bound 2e-3 absolute, ||A||_2 ~ 20 at n = 320"};
		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0.view(), W_blk, n, false, verify::all_items(batch), slack);
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

	auto W_blk = UnifiedVector<Real>(static_cast<std::size_t>(n));

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

	test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0.view(), W_blk, n);
	test_utils::expect_eigenpairs<Scalar>(A0.view(), A_blk.view(), W_blk, n);
}

TYPED_TEST(SyevBlockedTest, TwoStageProviderEigenvaluesOnlySmoke) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	// A two_stage pin on NETLIB used to be ignored (NETLIB ran the vendor); it now throws (R6).
	if (TestFixture::BackendType == Backend::NETLIB) GTEST_SKIP() << "two-stage is a GPU path";

	const int n = 128;
	const int batch = 8;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 9876);
	Matrix<Scalar, MatrixFormat::Dense> A_two_stage = A0;
	auto W_two_stage = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	{
		ScopedEnvVar route("BATCHLAS_SYEV_ROUTE", "two_stage");
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
	// A two_stage pin on NETLIB used to be ignored (NETLIB ran the vendor); it now throws (R6).
	if (TestFixture::BackendType == Backend::NETLIB) GTEST_SKIP() << "two-stage is a GPU path";

	const int n = 64;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, true, batch, 2468);
	Matrix<Scalar, MatrixFormat::Dense> A_two_stage = A0;
	auto W_two_stage = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	{
		ScopedEnvVar route("BATCHLAS_SYEV_ROUTE", "two_stage");
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

	for (int i = 0; i < n; ++i) {
		EXPECT_TRUE(std::isfinite(W_two_stage[i])) << "non-finite eigenvalue at i=" << i;
	}

	test_utils::expect_eigenpairs<Scalar>(A0.view(), A_two_stage.view(), W_two_stage, n);
}
// n = 320 is deliberate: it is inside the 256 < n <= 512 bucket where
// sytrd_block_size_default<T> now returns a different panel width for complex
// (32) than the tuning harness value used for real types (8). Every other
// eigen test in this file runs at n <= 96, so nothing here exercised that
// bucket at all -- the panel-width change and the workspace sizing that
// depends on it were both untested.
//
// This goes through the public `syev` on Auto rather than calling syev_blocked
// directly, so it also covers the per-type rows of
// tuned/syev.<dtype>.<device>.txt: at n = 320 that is blocked for float, double
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
			EXPECT_LE(W[b * n + i - 1], W[b * n + i])
				<< "eigenvalues not ascending at b=" << b << " i=" << i;
		}
	}

	test_utils::expect_eigenpairs<Scalar>(A0.view(), A.view(), W, n);
}

// The n <= 32 range, where Auto picks among the three CTA kernels rather than a
// provider. Complex now takes a different branch there than it used to:
// complex<float> uses syev_cta_fused for n <= 8 (it used syev_cta at every n),
// and complex<double> hands n > 24 to the vendor. Neither branch was reachable
// from Auto for complex before, so neither was covered.
//
// n = 6 and n = 28 sit one on each side of those two new boundaries. The sizes
// are driven through the public `syev` so that its buffer-size query and its
// solve both run choose() (src/ops/syev/syev.cc) -- the two have to agree, which
// is exactly the kind of disagreement a routing change can introduce.
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
			EXPECT_LE(W[i - 1], W[i])
				<< "eigenvalues not ascending, n=" << n << " i=" << i;
		}

		test_utils::expect_eigenpairs<Scalar>(A0.view(), A.view(), W, n);
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
// A0 with its strictly-lower triangle replaced by the conjugate mirror of the upper one: the
// Hermitian matrix an Uplo::Upper solve of A0 must see, as LAPACKE's reference reads it.
template <typename Scalar>
Matrix<Scalar, MatrixFormat::Dense> upper_mirrored(const Matrix<Scalar, MatrixFormat::Dense>& A0) {
	Matrix<Scalar, MatrixFormat::Dense> M = A0;
	const int n = M.view().rows();
	const int ld = static_cast<int>(M.view().ld());
	for (int b = 0; b < M.view().batch_size(); ++b) {
		Scalar* Mb = M.view().data().data() + static_cast<std::size_t>(b) * M.view().stride();
		for (int c = 0; c < n; ++c)
			for (int r = c + 1; r < n; ++r) Mb[r + c * ld] = verify::conj(Mb[c + r * ld]);
	}
	return M;
}

// max over items of max_i |eig_i(A) - eig_i(B)|, both read from the lower triangle by LAPACKE.
template <typename Scalar>
double lapacke_spectrum_gap(const MatrixView<Scalar, MatrixFormat::Dense>& A, const MatrixView<Scalar, MatrixFormat::Dense>& B) {
	double gap = 0;
	for (int b = 0; b < A.batch_size(); ++b) {
		auto a = verify::copy_item(A, b);
		auto c = verify::copy_item(B, b);
		std::vector<double> wa, wc;
		if (!verify::eigenvalues(A.rows(), a, wa) || !verify::eigenvalues(B.rows(), c, wc)) return std::numeric_limits<double>::quiet_NaN();
		for (std::size_t i = 0; i < wa.size(); ++i) gap = verify::nanmax(gap, std::fabs(wa[i] - wc[i]));
	}
	return gap;
}

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
	const auto A_sym = upper_mirrored(A0);
	auto W_ours = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	// PROVE THE FIXTURE HAS TEETH. If the poisoning above did not take effect the matrix is
	// still symmetric, Upper and Lower are interchangeable, and this test would pass even if
	// the mirror never ran. The spectrum read from the LOWER triangle must differ from the
	// Upper one (LAPACKE on both) -- that is what makes the Upper comparison below meaningful.
#if BATCHLAS_VERIFY_HAVE_LAPACKE
	ASSERT_GT(lapacke_spectrum_gap(A0.view(), A_sym.view()), 1.0)
		<< "fixture is vacuous: the two triangles agree, so Upper vs Lower proves nothing";
#endif

	// Ours: blocked path with Uplo::Upper, which must mirror before reducing.
	{
		StedcParams<Real> params;
		params.recursion_threshold = 32;
		auto ws = UnifiedVector<std::byte>(syev_blocked_buffer_size<B, Scalar>(
			*this->ctx, A_ours.view(), JobType::NoEigenVectors, Uplo::Upper, params));
		syev_blocked<B, Scalar>(*this->ctx, A_ours.view(), W_ours.to_span(),
								JobType::NoEigenVectors, Uplo::Upper, ws.to_span(), params).wait();
	}

	test_utils::expect_eigenvalues_match_lapacke<Scalar>(A_sym.view(), W_ours, n, false, verify::all_items(batch));
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
		const auto A_sym = upper_mirrored(A0);
		auto W_ours = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

		{
			StedcParams<Real> params;
			auto ws = UnifiedVector<std::byte>(syev_two_stage_buffer_size<B, Scalar>(
				*this->ctx, A_ours.view(), JobType::NoEigenVectors, Uplo::Upper, params));
			syev_two_stage<B, Scalar>(*this->ctx, A_ours.view(), W_ours.to_span(),
									  JobType::NoEigenVectors, Uplo::Upper, ws.to_span(), params).wait();
		}

		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A_sym.view(), W_ours, n, false, verify::all_items(batch));
	}
}
