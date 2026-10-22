// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>
#include <tests/eigen_verify.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

using namespace batchlas;

namespace {

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
#elif BATCHLAS_HAS_ROCM_BACKEND
using SyevCtaTestTypes = ::testing::Types<
    SyevCtaConfig<float, Backend::ROCM>,
    SyevCtaConfig<double, Backend::ROCM>,
    SyevCtaConfig<std::complex<float>, Backend::ROCM>,
    SyevCtaConfig<std::complex<double>, Backend::ROCM>>;
#elif BATCHLAS_HAS_SYCL_BACKEND
// AdaptiveCpp Metal: the native kernels in float and complex<float>; double needs device fp64.
using SyevCtaTestTypes = ::testing::Types<
    SyevCtaConfig<float, Backend::SYCL>,
#if BATCHLAS_HAS_DEVICE_FP64
    SyevCtaConfig<double, Backend::SYCL>,
    SyevCtaConfig<std::complex<double>, Backend::SYCL>,
#endif
    SyevCtaConfig<std::complex<float>, Backend::SYCL>>;
#else
using SyevCtaTestTypes = ::testing::Types<SyevCtaConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SyevCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevCtaTest, SyevCtaTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND || BATCHLAS_HAS_SYCL_BACKEND
TYPED_TEST(SyevCtaTest, EigenvaluesOnlyLowerMatchesNetlib) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 128;

	Matrix<Scalar, MatrixFormat::Dense> A0 = Matrix<Scalar, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n*batch));
	// Every item against LAPACKE in double (not a BatchLAS syev on this queue).
	const auto all = A0.all_items();

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::NoEigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::NoEigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_cta, n, false, all);
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

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_cta, n);

		test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A_cta, W_cta, n);
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

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::EigenVectors, Uplo::Upper, ws_cta.to_span(), p).wait();

		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_cta, n);

		test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A_cta, W_cta, n);
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

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		// Compare as a multiset (CTA may intentionally skip internal sorting here).
		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_cta, n, /*sorted_copy=*/true);

		test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A_cta, W_cta, n);
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
	Matrix<Scalar, MatrixFormat::Dense> A0(n, n, batch);
	A0.fill_near_degenerate_hermitian(*this->ctx, 1e-3, /*seed=*/1337).wait();
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;
	if (dbg) std::cerr << "[cta-test] A built" << std::endl;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));


	if (dbg) std::cerr << "[cta-test] running CTA" << std::endl;
	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.max_sweeps = 80;
		p.sort = false;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		// Compare as a multiset (CTA may intentionally skip internal sorting here).
		test_utils::expect_eigenvalues_match_lapacke<Scalar>(A0, W_cta, n, /*sorted_copy=*/true);

		test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A_cta, W_cta, n);
	}
	if (dbg) std::cerr << "[cta-test] CTA done" << std::endl;
}

TYPED_TEST(SyevCtaTest, EigenvectorsNearDegenerateLowerResidualAndOrtho_N32_Stress) {
	using Scalar = typename TestFixture::ScalarType;
	using Real = typename base_type<Scalar>::type;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 32;
	const int batch = 1;

	Matrix<Scalar, MatrixFormat::Dense> A0(n, n, batch);
	A0.fill_near_degenerate_hermitian(*this->ctx, 1e-3, /*seed=*/7331).wait();
	Matrix<Scalar, MatrixFormat::Dense> A_cta = A0;

	auto W_cta = UnifiedVector<Real>(static_cast<std::size_t>(n));

	for (auto scheme : kCtaUpdateSchemes) {
		SCOPED_TRACE(::testing::Message() << "update_scheme=" << update_scheme_name(scheme));
		A_cta = A0;
		SteqrParams<Scalar> p;
		p.max_sweeps = 80;
		p.sort = false;
		p.cta_update_scheme = scheme;
		auto ws_cta = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A_cta, JobType::EigenVectors, p));
		syev_cta<B, Scalar>(*this->ctx, A_cta, W_cta.to_span(), JobType::EigenVectors, Uplo::Lower, ws_cta.to_span(), p).wait();

		test_utils::expect_eigenpairs<Scalar>(*this->ctx, A0, A_cta, W_cta, n);
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
			auto ws = UnifiedVector<std::byte>(syev_cta_buffer_size<B, Scalar>(*this->ctx, A, JobType::EigenVectors, p));
			syev_cta<B, Scalar>(*this->ctx, A, W.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span(), p).wait();
		}
	}
}
#endif
