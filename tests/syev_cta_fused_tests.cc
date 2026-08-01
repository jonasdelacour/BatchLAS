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

// max |V^H V - I| over one batch item.
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

// ||A V - V diag(w)||_F / (||A||_F * n)
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
		const Real wj = W[static_cast<std::size_t>(b) * n + j];
		for (int i = 0; i < n; ++i) {
			Scalar sum = Scalar(0);
			for (int k = 0; k < n; ++k) {
				sum += A0(i, k, b) * V(k, j, b);
			}
			sum -= Scalar(wj) * V(i, j, b);
			r_norm2 += norm2_val(sum);
		}
	}

	const Real rel = std::sqrt(r_norm2) / ((a_norm > Real(0)) ? (a_norm * Real(n)) : Real(1));
	EXPECT_LE(rel, tol) << "relative residual = " << rel << " (batch " << b << ")";
}

// Clustered spectrum with small Hermitian noise: exercises deflation and the
// tie-breaking in the fused kernel's rank-based ordering.
template <typename Scalar>
static Matrix<Scalar, MatrixFormat::Dense> make_near_degenerate_hermitian(int n, int batch, unsigned seed,
																		 RealOf<Scalar> eps) {
	using Real = RealOf<Scalar>;
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

template <typename T, Backend B>
struct SyevCtaFusedConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

#if BATCHLAS_HAS_CUDA_BACKEND
using SyevCtaFusedTestTypes = ::testing::Types<
	SyevCtaFusedConfig<float, Backend::CUDA>,
	SyevCtaFusedConfig<double, Backend::CUDA>,
	SyevCtaFusedConfig<std::complex<float>, Backend::CUDA>,
	SyevCtaFusedConfig<std::complex<double>, Backend::CUDA>>;
#elif BATCHLAS_HAS_ROCM_BACKEND
using SyevCtaFusedTestTypes = ::testing::Types<
	SyevCtaFusedConfig<float, Backend::ROCM>,
	SyevCtaFusedConfig<double, Backend::ROCM>,
	SyevCtaFusedConfig<std::complex<float>, Backend::ROCM>,
	SyevCtaFusedConfig<std::complex<double>, Backend::ROCM>>;
#else
using SyevCtaFusedTestTypes = ::testing::Types<SyevCtaFusedConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SyevCtaFusedTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevCtaFusedTest, SyevCtaFusedTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND

// The fused kernel must produce a valid eigendecomposition for every partition
// width it dispatches (P = 4, 8, 16, 32) and for both triangles, including the
// sizes that only partly fill a partition.
TYPED_TEST(SyevCtaFusedTest, EigenvectorsAllSizesBothTrianglesResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int batch = 8;
	const Real tol = test_utils::tolerance<Scalar>() * Real(5);

	for (int n : {1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 31, 32}) {
		for (auto uplo : {Uplo::Lower, Uplo::Upper}) {
			SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << (uplo == Uplo::Lower ? "L" : "U"));

			auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch,
																 /*seed=*/1000u + n);
			auto A = A0;
			auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

			syev_cta_fused<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, uplo).wait();

			for (int b = 0; b < batch; ++b) {
				// Sorted ascending by default.
				for (int i = 1; i < n; ++i) {
					EXPECT_LE(W[static_cast<std::size_t>(b) * n + i - 1],
							  W[static_cast<std::size_t>(b) * n + i])
						<< "eigenvalues not ascending at i=" << i;
				}
				check_orthonormal_columns(A.view(), n, b, tol);
				check_eigen_residual(A0.view(), A.view(), W, n, b, tol);
			}
		}
	}
}

// Eigenvalues must match the CPU reference.
TYPED_TEST(SyevCtaFusedTest, EigenvaluesOnlyMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 128;

	auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
	auto A_fused = A0;
	auto A_ref = A0;

	auto W_fused = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

#if BATCHLAS_HAS_HOST_BACKEND
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(
			*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::NoEigenVectors,
							  Uplo::Lower, ws_ref.to_span())
			.wait();
	}
#endif

	syev_cta_fused<B, Scalar>(*this->ctx, A_fused.view(), W_fused.to_span(), JobType::NoEigenVectors,
							  Uplo::Lower)
		.wait();

	const Real tol = test_utils::tolerance<Scalar>();
#if BATCHLAS_HAS_HOST_BACKEND
	for (int b = 0; b < batch; ++b) {
		for (int i = 0; i < n; ++i) {
			const std::size_t idx = static_cast<std::size_t>(b) * n + i;
			ASSERT_NEAR(W_fused[idx], W_ref[idx], tol) << "eigenvalue mismatch i=" << i << " batch=" << b;
		}
	}
#else
	(void)tol;
#endif

	// jobz == NoEigenVectors must leave A alone (unlike syev_cta, which leaves
	// the reduction's reflectors behind).
	for (int b = 0; b < batch; ++b) {
		for (int j = 0; j < n; ++j) {
			for (int i = 0; i < n; ++i) {
				ASSERT_EQ(A_fused.view()(i, j, b), A0.view()(i, j, b))
					<< "A modified at (" << i << "," << j << ") batch " << b;
			}
		}
	}
}

// The fused kernel and the three-kernel pipeline run the same algorithm on the
// same data, so their eigenvalues must agree to round-off.
TYPED_TEST(SyevCtaFusedTest, AgreesWithPartitionedSyevCta) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int batch = 16;

	for (int n : {4, 8, 16, 32}) {
		SCOPED_TRACE(::testing::Message() << "n=" << n);

		auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch,
															  /*seed=*/77u + n);
		auto A_fused = A0;
		auto A_pipe = A0;

		auto W_fused = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);
		auto W_pipe = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

		SteqrParams<Scalar> p;
		auto ws = UnifiedVector<std::byte>(
			syev_cta_buffer_size<B, Scalar>(*this->ctx, A_pipe.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_pipe.view(), W_pipe.to_span(), JobType::EigenVectors,
							Uplo::Lower, ws.to_span(), p)
			.wait();

		syev_cta_fused<B, Scalar>(*this->ctx, A_fused.view(), W_fused.to_span(), JobType::EigenVectors,
								  Uplo::Lower)
			.wait();

		const Real tol = test_utils::tolerance<Scalar>() * Real(5);
		for (int b = 0; b < batch; ++b) {
			for (int i = 0; i < n; ++i) {
				const std::size_t idx = static_cast<std::size_t>(b) * n + i;
				ASSERT_NEAR(W_fused[idx], W_pipe[idx], tol)
					<< "eigenvalue mismatch n=" << n << " i=" << i << " batch=" << b;
			}
		}
	}
}

TYPED_TEST(SyevCtaFusedTest, NearDegenerateSpectrum) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 24;
	const int batch = 4;

	auto A0 = make_near_degenerate_hermitian<Scalar>(n, batch, /*seed=*/1337, Real(1e-3));
	auto A = A0;
	auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

	syev_cta_fused<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower).wait();

	const Real tol = test_utils::tolerance<Scalar>() * Real(10);
	for (int b = 0; b < batch; ++b) {
		check_orthonormal_columns(A.view(), n, b, tol);
		check_eigen_residual(A0.view(), A.view(), W, n, b, tol);
	}
}

// Descending order and the work-group multiplier are both wired through to the
// same kernel, so a smoke test on each is enough.
TYPED_TEST(SyevCtaFusedTest, SortOrderAndWgMultiplier) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 64;
	const Real tol = test_utils::tolerance<Scalar>() * Real(5);

	auto A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/909);

	for (size_t mult : {1u, 2u, 4u}) {
		SCOPED_TRACE(::testing::Message() << "wg_multiplier=" << mult);

		auto A = A0;
		auto W = UnifiedVector<Real>(static_cast<std::size_t>(n) * batch);

		SteqrParams<Scalar> p;
		p.sort_order = SortOrder::Descending;

		syev_cta_fused<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower,
								  Span<std::byte>(), p, mult)
			.wait();

		for (int b = 0; b < batch; ++b) {
			for (int i = 1; i < n; ++i) {
				ASSERT_GE(W[static_cast<std::size_t>(b) * n + i - 1], W[static_cast<std::size_t>(b) * n + i])
					<< "eigenvalues not descending at i=" << i;
			}
		}
		check_orthonormal_columns(A.view(), n, 0, tol);
		check_eigen_residual(A0.view(), A.view(), W, n, 0, tol);
	}
}

TYPED_TEST(SyevCtaFusedTest, RequiresNoWorkspace) {
	using Scalar = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	auto A = Matrix<Scalar, MatrixFormat::Dense>::Random(8, 8, /*hermitian=*/true, 2, /*seed=*/5);
	const size_t ws_bytes = syev_cta_fused_buffer_size<B, Scalar>(*this->ctx, A.view(), JobType::EigenVectors);
	EXPECT_EQ(ws_bytes, 0u);
}

#endif // CUDA || ROCM
