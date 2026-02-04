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
#include <array>
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

template <typename Scalar>
static Matrix<Scalar, MatrixFormat::Dense> make_near_degenerate_hermitian(int n, int batch, unsigned seed, RealOf<Scalar> eps) {
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

				// Start from a clustered diagonal and add a small Hermitian noise.
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

static constexpr std::array<SteqrUpdateScheme, 2> kCtaUpdateSchemes = {
	SteqrUpdateScheme::PG,
	SteqrUpdateScheme::EXP,
};

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

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::NoEigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		const Real tol = test_utils::tolerance<Scalar>();
		for (int j = 0; j < batch; ++j) {
			for (int i = 0; i < n; ++i) {
				ASSERT_NEAR(W_cta[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)],
							W_ref[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(n)],
							tol) << "eigenvalue mismatch i=" << i << " batch=" << j;
			}
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

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		const Real tol_w = test_utils::tolerance<Scalar>();
		for (int i = 0; i < n; ++i) {
			ASSERT_NEAR(W_cta[static_cast<std::size_t>(i)], W_ref[static_cast<std::size_t>(i)], tol_w) << "eigenvalue mismatch i=" << i;
		}

		check_orthonormal_columns(A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
		check_eigen_residual(A0.view(), A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
	}
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

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Upper, ws_cta.to_span(), p).wait();

		const Real tol_w = test_utils::tolerance<Scalar>();
		for (int i = 0; i < n; ++i) {
			ASSERT_NEAR(W_cta[static_cast<std::size_t>(i)], W_ref[static_cast<std::size_t>(i)], tol_w) << "eigenvalue mismatch i=" << i;
		}

		check_orthonormal_columns(A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
		check_eigen_residual(A0.view(), A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
	}
}

TYPED_TEST(SyevCtaTest, EigenvectorsN32RandomLowerResidualAndOrtho) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 32;
	const int batch = 1;
	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/4242);
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n));

	// Reference (CPU LAPACKE)
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		// Compare as a multiset (CTA may intentionally skip internal sorting here).
		std::vector<Real> w_cta_sorted(static_cast<std::size_t>(n));
		std::vector<Real> w_ref_sorted(static_cast<std::size_t>(n));
		for (int i = 0; i < n; ++i) {
			w_cta_sorted[static_cast<std::size_t>(i)] = W_cta[static_cast<std::size_t>(i)];
			w_ref_sorted[static_cast<std::size_t>(i)] = W_ref[static_cast<std::size_t>(i)];
		}
		std::sort(w_cta_sorted.begin(), w_cta_sorted.end());
		std::sort(w_ref_sorted.begin(), w_ref_sorted.end());

		const Real tol_w = test_utils::tolerance<Scalar>() * Real(5);
		for (int i = 0; i < n; ++i) {
			ASSERT_NEAR(w_cta_sorted[static_cast<std::size_t>(i)], w_ref_sorted[static_cast<std::size_t>(i)], tol_w)
				<< "eigenvalue mismatch (sorted) i=" << i;
		}

		check_orthonormal_columns(A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
		check_eigen_residual(A0.view(), A_cta.view(), W_cta, test_utils::tolerance<Scalar>());
	}
}

TYPED_TEST(SyevCtaTest, EigenvectorsNearDegenerateLowerResidualAndOrtho_Stress) {
	const bool dbg = []() {
		const char* e = std::getenv("BATCHLAS_CTA_DEBUG_SYNC");
		return e && std::string(e) == "1";
	}();

	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 24;
	const int batch = 1;

	if (dbg) std::cerr << "[cta-test] building near-degenerate A" << std::endl;
	Matrix<Scalar, MatrixFormat::Dense> A0 = make_near_degenerate_hermitian<Scalar>(n, batch, /*seed=*/1337, Real(1e-3));
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	Matrix<Scalar, MatrixFormat::Dense> A_ref = A0;
	if (dbg) std::cerr << "[cta-test] A built" << std::endl;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));
	auto W_ref = UnifiedVector<Real>(static_cast<std::size_t>(n));

	if (dbg) std::cerr << "[cta-test] running NETLIB reference" << std::endl;
	{
		auto ws_ref = UnifiedVector<std::byte>(syev_buffer_size<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower));
		syev<Backend::NETLIB>(*this->ctx, A_ref.view(), W_ref.to_span(), JobType::EigenVectors, Uplo::Lower, ws_ref.to_span()).wait();
	}
	if (dbg) std::cerr << "[cta-test] NETLIB done" << std::endl;

	if (dbg) std::cerr << "[cta-test] running CTA" << std::endl;
	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.max_sweeps = 80;
		p.sort = false;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		// Compare as a multiset (CTA may intentionally skip internal sorting here).
		std::vector<Real> w_cta_sorted(static_cast<std::size_t>(n));
		std::vector<Real> w_ref_sorted(static_cast<std::size_t>(n));
		for (int i = 0; i < n; ++i) {
			w_cta_sorted[static_cast<std::size_t>(i)] = W_cta[static_cast<std::size_t>(i)];
			w_ref_sorted[static_cast<std::size_t>(i)] = W_ref[static_cast<std::size_t>(i)];
		}
		std::sort(w_cta_sorted.begin(), w_cta_sorted.end());
		std::sort(w_ref_sorted.begin(), w_ref_sorted.end());

		const Real tol_w = test_utils::tolerance<Scalar>() * Real(5);
		for (int i = 0; i < n; ++i) {
			ASSERT_NEAR(w_cta_sorted[static_cast<std::size_t>(i)], w_ref_sorted[static_cast<std::size_t>(i)], tol_w)
				<< "eigenvalue mismatch (sorted) i=" << i;
		}

		check_orthonormal_columns(A_cta.view(), W_cta, test_utils::tolerance<Scalar>() * Real(10));
		check_eigen_residual(A0.view(), A_cta.view(), W_cta, test_utils::tolerance<Scalar>() * Real(10));
	}
	if (dbg) std::cerr << "[cta-test] CTA done" << std::endl;
}

TYPED_TEST(SyevCtaTest, EigenvectorsNearDegenerateLowerResidualAndOrtho_N32_Stress) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 32;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0 = make_near_degenerate_hermitian<Scalar>(n, batch, /*seed=*/7331, Real(1e-3));
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.max_sweeps = 80;
		p.sort = false;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta.view(), JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta.view(), W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		check_orthonormal_columns(A_cta.view(), W_cta, test_utils::tolerance<Scalar>() * Real(10));
		check_eigen_residual(A0.view(), A_cta.view(), W_cta, test_utils::tolerance<Scalar>() * Real(10));
	}
}

TYPED_TEST(SyevCtaTest, RepeatedRunsDoNotHang) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 4;
	const int iters = 50;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/2025);
	auto W = UnifiedVector<Real>(static_cast<std::size_t>(n * batch));
	SteqrParams<Scalar> p;
	p.max_sweeps = 30;

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		p.cta_update_scheme = scheme;
		for (int t = 0; t < iters; ++t) {
			Matrix<Scalar, MatrixFormat::Dense> A = A0;
			auto ws = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A.view(), JobType::EigenVectors, p));
			syev_cta<B, Scalar>(*this->ctx, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), p).wait();
		}
	}
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
