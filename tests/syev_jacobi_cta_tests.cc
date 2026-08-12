#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {
template <typename Scalar>
using RealOf = typename base_type<Scalar>::type;

template <typename Scalar>
static RealOf<Scalar> abs_val(const Scalar& x) {
	return static_cast<RealOf<Scalar>>(std::abs(x));
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
									  int n, int b, RealOf<Scalar> tol) {
	using Real = RealOf<Scalar>;
	Real max_err = Real(0);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			Scalar dot = Scalar(0);
			for (int r = 0; r < n; ++r) {
				dot += conj_val(V(r, i, b)) * V(r, j, b);
			}
			const Real target = (i == j) ? Real(1) : Real(0);
			max_err = std::max(max_err, abs_val(dot - Scalar(target)));
		}
	}
	EXPECT_LE(max_err, tol) << "max |V^H V - I| = " << max_err << " (batch " << b << ")";
}

template <typename Scalar>
static void check_eigen_residual(const MatrixView<Scalar, MatrixFormat::Dense>& A0,
								 const MatrixView<Scalar, MatrixFormat::Dense>& V,
								 const UnifiedVector<RealOf<Scalar>>& W,
								 int n, int b, RealOf<Scalar> tol) {
	using Real = RealOf<Scalar>;

	Real a_norm2 = Real(0);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			a_norm2 += norm2_val(A0(i, j, b));
		}
	}
	const Real a_norm = std::sqrt(a_norm2);

	Real r_norm2 = Real(0);
	for (int j = 0; j < n; ++j) {
		const Real wj = W[static_cast<std::size_t>(b) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j)];
		for (int i = 0; i < n; ++i) {
			Scalar sum = Scalar(0);
			for (int k = 0; k < n; ++k) {
				sum += A0(i, k, b) * V(k, j, b);
			}
			sum -= Scalar(wj) * V(i, j, b);
			r_norm2 += norm2_val(sum);
		}
	}

	const Real r_norm = std::sqrt(r_norm2);
	const Real denom = (a_norm > Real(0)) ? (a_norm * Real(n)) : Real(1);
	const Real rel = r_norm / denom;
	EXPECT_LE(rel, tol) << "relative residual ||AV - VW||/(||A||*n) = " << rel << " (batch " << b << ")";
}

template <typename Scalar>
static Matrix<Scalar, MatrixFormat::Dense> make_near_degenerate_hermitian(int n, int batch, unsigned seed,
																		 RealOf<Scalar> eps) {
	using Real = RealOf<Scalar>;

	// NOTE: Avoid Matrix::Zeros/Matrix::Diagonal here: those factory helpers launch
	// device kernels (often asynchronously) and can race with the host-side writes
	// below on USM memory.
	Matrix<Scalar, MatrixFormat::Dense> A(n, n, batch);

	std::minstd_rand rng(seed);
	std::uniform_real_distribution<Real> dist(Real(-1), Real(1));

	for (int b = 0; b < batch; ++b) {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i <= j; ++i) {
				Scalar z;
				if constexpr (std::is_same_v<Scalar, Real>) {
					z = Scalar(dist(rng));
				} else {
					z = Scalar(dist(rng), dist(rng));
				}

				if (i == j) {
					const Real base = Real(i / 4);
					const Real tiny = Real(i % 4) * Real(1e-4);
					if constexpr (std::is_same_v<Scalar, Real>) {
						A(i, j, b) = Scalar(base + tiny) + Scalar(eps) * z;
					} else {
						A(i, j, b) = Scalar(base + tiny) + Scalar(eps) * Scalar(Real(std::real(z)), Real(0));
					}
				} else {
					const Scalar v = Scalar(eps) * z;
					A(i, j, b) = v;
					A(j, i, b) = conj_val(v);
				}
			}
		}
	}

	return A;
}

// ---------------------------------------------------------------------------
// Independent CPU reference: cyclic-by-rows two-sided Jacobi in double.
//
// Used as the "truth" for the graded/badly-scaled accuracy test. A double
// LAPACK syev cannot serve that role: it tridiagonalizes, so its own error on
// the smallest eigenvalue of a strongly graded matrix is ~eps_double*||A||,
// which is far larger than the eigenvalue itself. Jacobi in double does have
// the relative-accuracy property, and this routine is deliberately simple and
// independent of the kernel under test.
// ---------------------------------------------------------------------------
static std::vector<double> reference_jacobi_eigenvalues(const std::vector<double>& A_in, int n) {
	std::vector<double> A = A_in; // column-major n x n
	const double tol = double(n) * std::numeric_limits<double>::epsilon();

	for (int sweep = 0; sweep < 100; ++sweep) {
		int rotations = 0;
		for (int p = 0; p < n - 1; ++p) {
			for (int q = p + 1; q < n; ++q) {
				const double apq = A[p + q * n];
				const double app = A[p + p * n];
				const double aqq = A[q + q * n];
				if (std::abs(apq) <= tol * std::sqrt(std::abs(app) * std::abs(aqq))) continue;
				if (apq == 0.0) continue;

				const double tau = (aqq - app) / (2.0 * apq);
				const double t = std::copysign(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
				const double c = 1.0 / std::sqrt(1.0 + t * t);
				const double s = t * c;

				// A <- J^T A J with J = [[c, s], [-s, c]] on rows/cols (p,q).
				for (int r = 0; r < n; ++r) {
					const double arp = A[r + p * n];
					const double arq = A[r + q * n];
					A[r + p * n] = c * arp - s * arq;
					A[r + q * n] = s * arp + c * arq;
				}
				for (int cc = 0; cc < n; ++cc) {
					const double apc = A[p + cc * n];
					const double aqc = A[q + cc * n];
					A[p + cc * n] = c * apc - s * aqc;
					A[q + cc * n] = s * apc + c * aqc;
				}
				A[p + q * n] = 0.0;
				A[q + p * n] = 0.0;
				++rotations;
			}
		}
		if (rotations == 0) break;
	}

	std::vector<double> w(static_cast<std::size_t>(n));
	for (int i = 0; i < n; ++i) w[static_cast<std::size_t>(i)] = A[i + i * n];
	std::sort(w.begin(), w.end());
	return w;
}

// Graded SPD matrix A = D * M * D with D = diag(2^{-grade*k}) and M a
// well-conditioned, diagonally dominant SPD matrix with dyadic entries.
//
// Every entry of A is a dyadic rational, so the matrix is represented exactly in
// both float and double: the float and double solvers see the *same* matrix and
// any difference in the result is entirely due to precision, not to input
// rounding. kappa(A) is astronomically large while kappa of the
// column-equilibrated matrix stays ~kappa(M), which is precisely the regime
// where Jacobi's relative-accuracy bound bites and a tridiagonalizing method
// loses the small eigenvalues.
static void make_graded_spd(int n, int grade, unsigned seed,
							std::vector<double>& out_double,
							std::vector<float>& out_float) {
	std::minstd_rand rng(seed);
	std::uniform_int_distribution<int> dist(-16, 16);

	std::vector<double> M(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
	for (int j = 0; j < n; ++j) {
		for (int i = j; i < n; ++i) {
			if (i == j) {
				M[i + j * n] = 1.0;
			} else {
				const double v = double(dist(rng)) / 256.0; // dyadic, |v| <= 1/16
				M[i + j * n] = v;
				M[j + i * n] = v;
			}
		}
	}

	std::vector<double> D(static_cast<std::size_t>(n));
	for (int i = 0; i < n; ++i) D[static_cast<std::size_t>(i)] = std::ldexp(1.0, -grade * i);

	out_double.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
	out_float.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0f);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			const double v = D[static_cast<std::size_t>(i)] * M[i + j * n] * D[static_cast<std::size_t>(j)];
			out_double[static_cast<std::size_t>(i + j * n)] = v;
			out_float[static_cast<std::size_t>(i + j * n)] = static_cast<float>(v);
		}
	}
}

// Tolerance for comparing against an independent eigensolver.
//
// Both solvers carry a backward error of order eps*||A||, so the *absolute*
// agreement between them scales with the spectral radius; a fixed absolute
// tolerance silently becomes a relative tolerance of eps only when ||A|| ~ 1.
// The scaling below is what makes the comparison size-independent.
// (Agreement at the eps*||A|| level is verified directly, against a double
// reference, in RandomSymmetricMatchesDoubleReference.)
template <typename Scalar>
static RealOf<Scalar> eig_compare_tol(const UnifiedVector<RealOf<Scalar>>& w_ref, int n, int batch) {
	using Real = RealOf<Scalar>;
	Real lambda_max = Real(1);
	for (int i = 0; i < n * batch; ++i) {
		lambda_max = std::max(lambda_max, std::abs(w_ref[static_cast<std::size_t>(i)]));
	}
	return test_utils::tolerance<Scalar>() * lambda_max;
}

// Random symmetric matrix with dyadic entries, so it is represented exactly in
// both float and double and the two precisions see identical input.
static void make_dyadic_symmetric(int n, unsigned seed,
								  std::vector<double>& out_double,
								  std::vector<float>& out_float) {
	std::minstd_rand rng(seed);
	std::uniform_int_distribution<int> dist(-1024, 1024);

	out_double.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0);
	out_float.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0f);
	for (int j = 0; j < n; ++j) {
		for (int i = j; i < n; ++i) {
			const double v = double(dist(rng)) / 1024.0; // dyadic in [-1, 1]
			out_double[static_cast<std::size_t>(i + j * n)] = v;
			out_double[static_cast<std::size_t>(j + i * n)] = v;
			out_float[static_cast<std::size_t>(i + j * n)] = static_cast<float>(v);
			out_float[static_cast<std::size_t>(j + i * n)] = static_cast<float>(v);
		}
	}
}

// Accumulated rounding over n(n-1)/2 rotations grows with problem size, so the
// fixed absolute tolerance from test_utils is scaled by sqrt(n). Measured
// values: ||V^H V - I|| ~ 86*eps at n=32 in single precision.
template <typename Scalar>
static RealOf<Scalar> vec_tol(int n, RealOf<Scalar> extra = RealOf<Scalar>(1)) {
	using Real = RealOf<Scalar>;
	return test_utils::tolerance<Scalar>() * std::sqrt(Real(n)) * extra;
}

template <typename T, Backend B>
struct SyevJacobiCtaConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevJacobiCtaTestTypes = ::testing::Types<
	SyevJacobiCtaConfig<float, Backend::CUDA>,
	SyevJacobiCtaConfig<double, Backend::CUDA>,
	SyevJacobiCtaConfig<std::complex<float>, Backend::CUDA>,
	SyevJacobiCtaConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SyevJacobiCtaTestTypes = ::testing::Types<
	SyevJacobiCtaConfig<float, Backend::ROCM>,
	SyevJacobiCtaConfig<double, Backend::ROCM>,
	SyevJacobiCtaConfig<std::complex<float>, Backend::ROCM>,
	SyevJacobiCtaConfig<std::complex<double>, Backend::ROCM>>;
#else
using SyevJacobiCtaTestTypes = ::testing::Types<SyevJacobiCtaConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SyevJacobiCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevJacobiCtaTest, SyevJacobiCtaTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND

TYPED_TEST(SyevJacobiCtaTest, EigenvaluesOnlyMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 64;

	for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
		SCOPED_TRACE(::testing::Message() << "uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper"));

		Matrix<Scalar, MatrixFormat::Dense> A0 =
			Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
		Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;
		Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

		auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
		auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

#if BATCHLAS_HAS_HOST_BACKEND
		{
			auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(
				*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, uplo));
			syev(*this->ctx,
                         A_ref.view(),
                         W_ref.to_span(),
                         {.jobz = JobType::NoEigenVectors, .uplo = uplo},
                         ws_ref.to_span()).wait();
		}
#endif

		syev_jacobi_cta<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(), JobType::NoEigenVectors, uplo).wait();

#if BATCHLAS_HAS_HOST_BACKEND
		const Real tol = eig_compare_tol<Scalar>(W_ref, n, batch);
		for (int b = 0; b < batch; ++b) {
			for (int i = 0; i < n; ++i) {
				const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
									  + static_cast<std::size_t>(i);
				ASSERT_NEAR(W_jac[idx], W_ref[idx], tol) << "eigenvalue mismatch i=" << i << " batch=" << b;
			}
		}
#endif
	}
}

TYPED_TEST(SyevJacobiCtaTest, EigenvaluesOnlyLeavesMatrixUntouched) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 12;
	const int batch = 2;

	Matrix<Scalar, MatrixFormat::Dense> A0 =
		Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/99);
	Matrix<Scalar, MatrixFormat::Dense> A = A0;
	auto W = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	syev_jacobi_cta<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower).wait();

	for (int b = 0; b < batch; ++b) {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				ASSERT_EQ(A.view()(i, j, b), A0.view()(i, j, b))
					<< "A was modified at (" << i << "," << j << "," << b << ") for NoEigenVectors";
			}
		}
	}
}

TYPED_TEST(SyevJacobiCtaTest, EigenvectorsResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int batch = 2;

	for (int n : {16, 32}) {
		for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
			SCOPED_TRACE(::testing::Message() << "n=" << n
							<< " uplo=" << (uplo == Uplo::Lower ? "Lower" : "Upper"));

			Matrix<Scalar, MatrixFormat::Dense> A0 =
				Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/456);
			Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;
			Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

			auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
			auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

#if BATCHLAS_HAS_HOST_BACKEND
			{
				auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(
					*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, uplo));
				syev(*this->ctx,
                          A_ref.view(),
                          W_ref.to_span(),
                          {.jobz = JobType::NoEigenVectors, .uplo = uplo},
                          ws_ref.to_span()).wait();
			}
#endif

			syev_jacobi_cta<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(), JobType::EigenVectors, uplo).wait();

#if BATCHLAS_HAS_HOST_BACKEND
			const Real tol_w = eig_compare_tol<Scalar>(W_ref, n, batch);
			for (int b = 0; b < batch; ++b) {
				for (int i = 0; i < n; ++i) {
					const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
										  + static_cast<std::size_t>(i);
					ASSERT_NEAR(W_jac[idx], W_ref[idx], tol_w) << "eigenvalue mismatch i=" << i << " batch=" << b;
				}
			}
#endif

			for (int b = 0; b < batch; ++b) {
				check_orthonormal_columns(A_jac.view(), n, b, vec_tol<Scalar>(n));
				check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, b, vec_tol<Scalar>(n));
			}
		}
	}
}

// Odd n exercises the padded round-robin schedule: the pivot index space is
// rounded up to n+1 and pairs touching the phantom index must be skipped.
TYPED_TEST(SyevJacobiCtaTest, OddAndSmallSizes) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int batch = 3;

	for (int n : {1, 2, 3, 5, 7, 13, 17, 31}) {
		SCOPED_TRACE(::testing::Message() << "n=" << n);

		Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(
			n, n, /*hermitian=*/true, batch, /*seed=*/static_cast<unsigned>(1000 + n));
		Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;
		Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

		auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
		auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

#if BATCHLAS_HAS_HOST_BACKEND
		{
			auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size(
				*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower));
			syev(*this->ctx,
                         A_ref.view(),
                         W_ref.to_span(),
                         {.jobz = JobType::NoEigenVectors},
                         ws_ref.to_span()).wait();
		}
#endif

		syev_jacobi_cta<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(), JobType::EigenVectors, Uplo::Lower).wait();

#if BATCHLAS_HAS_HOST_BACKEND
		const Real tol_w = eig_compare_tol<Scalar>(W_ref, n, batch);
		for (int b = 0; b < batch; ++b) {
			for (int i = 0; i < n; ++i) {
				const std::size_t idx = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
									  + static_cast<std::size_t>(i);
				ASSERT_NEAR(W_jac[idx], W_ref[idx], tol_w) << "eigenvalue mismatch i=" << i << " batch=" << b;
			}
		}
#endif

		for (int b = 0; b < batch; ++b) {
			check_orthonormal_columns(A_jac.view(), n, b, vec_tol<Scalar>(n));
			check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, b, vec_tol<Scalar>(n));
		}
	}
}

TYPED_TEST(SyevJacobiCtaTest, DescendingSortOrder) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 2;

	Matrix<Scalar, MatrixFormat::Dense> A0 =
		Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2468);
	Matrix<Scalar, MatrixFormat::Dense> A_asc = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_desc = A0;

	auto W_asc = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
	auto W_desc = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	JacobiParams<Scalar> p_desc;
	p_desc.sort_order = SortOrder::Descending;

	syev_jacobi_cta<B, Scalar>(*this->ctx, A_asc.view(), W_asc.to_span(), JobType::EigenVectors, Uplo::Lower).wait();
	syev_jacobi_cta<B, Scalar>(*this->ctx, A_desc.view(), W_desc.to_span(), JobType::EigenVectors, Uplo::Lower,
							   Span<std::byte>(), p_desc).wait();

	for (int b = 0; b < batch; ++b) {
		for (int i = 0; i < n; ++i) {
			const std::size_t ia = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
								 + static_cast<std::size_t>(i);
			const std::size_t id = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
								 + static_cast<std::size_t>(n - 1 - i);
			ASSERT_NEAR(W_asc[ia], W_desc[id], test_utils::tolerance<Scalar>())
				<< "descending order does not mirror ascending at i=" << i;
		}
		for (int i = 1; i < n; ++i) {
			const std::size_t prev = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
								   + static_cast<std::size_t>(i - 1);
			const std::size_t cur = static_cast<std::size_t>(b) * static_cast<std::size_t>(n)
								  + static_cast<std::size_t>(i);
			ASSERT_GE(W_desc[prev], W_desc[cur]) << "not descending at i=" << i;
		}
	}
}

TYPED_TEST(SyevJacobiCtaTest, NearDegenerateResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int batch = 1;

	for (int n : {24, 32}) {
		SCOPED_TRACE(::testing::Message() << "n=" << n);

		Matrix<Scalar, MatrixFormat::Dense> A0 = make_near_degenerate_hermitian<Scalar>(
			n, batch, /*seed=*/static_cast<unsigned>(1337 + n), Real(1e-3));
		Matrix<Scalar, MatrixFormat::Dense> A_jac = A0;

		auto W_jac = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

		syev_jacobi_cta<B, Scalar>(*this->ctx, A_jac.view(), W_jac.to_span(), JobType::EigenVectors, Uplo::Lower).wait();

		check_orthonormal_columns(A_jac.view(), n, 0, vec_tol<Scalar>(n, Real(10)));
		check_eigen_residual(A0.view(), A_jac.view(), W_jac, n, 0, vec_tol<Scalar>(n, Real(10)));
	}
}

TYPED_TEST(SyevJacobiCtaTest, RepeatedRunsDoNotHang) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 4;
	const int iters = 50;

	Matrix<Scalar, MatrixFormat::Dense> A0 =
		Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2025);
	auto W = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));

	for (int t = 0; t < iters; ++t) {
		Matrix<Scalar, MatrixFormat::Dense> A = A0;
		syev_jacobi_cta<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower).wait();
	}
}

// Verifies that the float kernel agrees with an exact-input double reference to
// within a small multiple of eps*||A||, i.e. that it is backward stable at the
// level theory predicts. This is what licenses the ||A||-scaled tolerance used
// when cross-checking against LAPACK above: the discrepancy there is ordinary
// float rounding shared by any backward-stable solver, not a deficiency of this
// kernel.
TEST(SyevJacobiCtaAccuracy, RandomSymmetricMatchesDoubleReference) {
#if BATCHLAS_HAS_CUDA_BACKEND
	constexpr Backend B = Backend::CUDA;
#else
	constexpr Backend B = Backend::ROCM;
#endif
	if (!test_utils::should_run_backend(B)) {
		GTEST_SKIP() << "Backend filtered by BATCHLAS_TEST_BACKEND environment variable";
	}
	if (!test_utils::should_run_float_type<float>()) {
		GTEST_SKIP() << "Float type filtered by BATCHLAS_TEST_FLOAT_TYPE environment variable";
	}
	auto ctx = Queue("gpu", true);
	if (ctx.device().type != DeviceType::GPU) {
		GTEST_SKIP() << "GPU backend requires GPU device, but none was selected";
	}

	for (int n : {8, 16, 31, 32}) {
		SCOPED_TRACE(::testing::Message() << "n=" << n);

		std::vector<double> A_d;
		std::vector<float> A_f;
		make_dyadic_symmetric(n, /*seed=*/static_cast<unsigned>(555 + n), A_d, A_f);

		const std::vector<double> w_true = reference_jacobi_eigenvalues(A_d, n);

		double a_fro = 0.0;
		for (double v : A_d) a_fro += v * v;
		a_fro = std::sqrt(a_fro);

		Matrix<float, MatrixFormat::Dense> A(n, n, 1);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				A.view()(i, j, 0) = A_f[static_cast<std::size_t>(i + j * n)];
			}
		}

		auto W = UnifiedVector<float>(static_cast<std::size_t>(n));
		syev_jacobi_cta(ctx, A.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower).wait();

		double max_abs = 0.0;
		for (int i = 0; i < n; ++i) {
			max_abs = std::max(max_abs,
							   std::abs(static_cast<double>(W[static_cast<std::size_t>(i)])
										- w_true[static_cast<std::size_t>(i)]));
		}

		// LAWN 169 Prop. 2.3 bounds the backward error of s cyclic sweeps by
		// O(s*n*eps); with s ~ 7 the measured constant here is ~24, so a bound of
		// 2*n leaves headroom while still catching an order-of-magnitude
		// regression.
		const double unit = static_cast<double>(std::numeric_limits<float>::epsilon()) * a_fro;
		std::cout << "  n=" << n << ": max |lambda_float - lambda_double| = " << max_abs
				  << " = " << (max_abs / unit) << " * eps*||A||_F\n";
		EXPECT_LE(max_abs, 2.0 * double(n) * unit)
			<< "max |lambda_float - lambda_double| = " << max_abs
			<< " = " << (max_abs / unit) << " * eps*||A||_F (||A||_F = " << a_fro << ")";
	}
}

// ---------------------------------------------------------------------------
// The payoff test: high relative accuracy on a graded SPD matrix.
//
// This is the reason the kernel exists. A = D*M*D with D spanning many decades
// has kappa(A) ~ 1e30 but kappa of the column-equilibrated matrix ~ kappa(M) ~ 1.
// Jacobi with the relative threshold must resolve the *smallest* eigenvalues to
// near-full relative precision; a tridiagonalization-based solver cannot, since
// its error is proportional to eps*||A|| which dwarfs them.
//
// Reference is an independent double-precision CPU Jacobi (see above). The
// matrix entries are dyadic, so float and double see bit-identical input.
// ---------------------------------------------------------------------------
TEST(SyevJacobiCtaAccuracy, GradedSpdRelativeAccuracy) {
#if BATCHLAS_HAS_CUDA_BACKEND
	constexpr Backend B = Backend::CUDA;
#else
	constexpr Backend B = Backend::ROCM;
#endif

	if (!test_utils::should_run_backend(B)) {
		GTEST_SKIP() << "Backend filtered by BATCHLAS_TEST_BACKEND environment variable";
	}
	if (!test_utils::should_run_float_type<float>()) {
		GTEST_SKIP() << "Float type filtered by BATCHLAS_TEST_FLOAT_TYPE environment variable";
	}
	auto ctx = Queue("gpu", true);
	if (ctx.device().type != DeviceType::GPU) {
		GTEST_SKIP() << "GPU backend requires GPU device, but none was selected";
	}

	const int n = 16;
	const int grade = 4; // D_ii = 2^{-4i}, so kappa(A) ~ 2^{120} ~ 1.3e36

	std::vector<double> A_d;
	std::vector<float> A_f;
	make_graded_spd(n, grade, /*seed=*/20240601, A_d, A_f);

	const std::vector<double> w_true = reference_jacobi_eigenvalues(A_d, n);

	// Sanity: the construction really is graded and positive definite.
	ASSERT_GT(w_true.front(), 0.0) << "reference says the test matrix is not positive definite";
	ASSERT_LT(w_true.front() / w_true.back(), 1e-20) << "test matrix is not graded enough to be discriminating";

	Matrix<float, MatrixFormat::Dense> A_jac(n, n, 1);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			A_jac.view()(i, j, 0) = A_f[static_cast<std::size_t>(i + j * n)];
		}
	}

	auto W = UnifiedVector<float>(static_cast<std::size_t>(n));
	syev_jacobi_cta(ctx, A_jac.view(), W.to_span(), JobType::NoEigenVectors, Uplo::Lower).wait();

	double max_rel = 0.0;
	for (int i = 0; i < n; ++i) {
		const double got = static_cast<double>(W[static_cast<std::size_t>(i)]);
		const double want = w_true[static_cast<std::size_t>(i)];
		const double rel = std::abs(got - want) / std::abs(want);
		max_rel = std::max(max_rel, rel);
		std::cout << "  lambda[" << i << "] jacobi=" << got << " ref=" << want << " rel=" << rel << "\n";
	}

	// Relative, not absolute: every eigenvalue including the smallest must be
	// accurate to a modest multiple of float eps. An absolute-accuracy solver
	// would show rel ~ 1 on the small end.
	const double bound = 200.0 * static_cast<double>(std::numeric_limits<float>::epsilon());
	EXPECT_LE(max_rel, bound) << "max relative eigenvalue error " << max_rel
							  << " exceeds " << bound
							  << " -- the relative-accuracy property is not being achieved";

	// Informational head-to-head against the tridiagonalization-based path on
	// the same input. Deliberately NOT asserted: this documents why the Jacobi
	// kernel exists without pinning a competitor's numbers into the test suite.
	{
		Matrix<float, MatrixFormat::Dense> A_tri(n, n, 1);
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				A_tri.view()(i, j, 0) = A_f[static_cast<std::size_t>(i + j * n)];
			}
		}
		auto W_tri = UnifiedVector<float>(static_cast<std::size_t>(n));
		SteqrParams<float> sp;
		auto ws = UnifiedVector<std::byte>(
			syev_cta_buffer_size(ctx, A_tri.view(), JobType::NoEigenVectors, sp));
		syev_cta(ctx, A_tri.view(), W_tri.to_span(), JobType::NoEigenVectors, Uplo::Lower,
						   ws.to_span(), sp).wait();

		std::vector<float> w_tri(W_tri.begin(), W_tri.begin() + n);
		std::sort(w_tri.begin(), w_tri.end());

		double tri_max_rel = 0.0;
		for (int i = 0; i < n; ++i) {
			const double want = w_true[static_cast<std::size_t>(i)];
			tri_max_rel = std::max(tri_max_rel,
								   std::abs(static_cast<double>(w_tri[static_cast<std::size_t>(i)]) - want)
									   / std::abs(want));
		}
		std::cout << "  [informational] max relative error: jacobi_cta=" << max_rel
				  << "  syev_cta(sytrd+steqr)=" << tri_max_rel << "\n";
	}
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
