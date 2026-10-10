#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

using namespace batchlas;

namespace {

inline bool is_kernel_not_found_message(const std::string& msg) {
	// SYCL runtimes vary; keep this loose.
	return msg.find("No kernel named") != std::string::npos ||
		   msg.find("kernel bundle") != std::string::npos ||
		   msg.find("PI_ERROR_INVALID_KERNEL_NAME") != std::string::npos;
}

inline bool is_missing_subgroup32_message(const std::string& msg) {
	return msg.find("subgroup size 32") != std::string::npos;
}

} // namespace

#include "test_utils.hh"
#include "ormqr_verify.hh"

#if BATCHLAS_HAS_CUDA_BACKEND

template <typename T, Backend B>
struct OrmqrCtaConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

using OrmqrCtaTestTypes = ::testing::Types<OrmqrCtaConfig<float, Backend::CUDA>, OrmqrCtaConfig<double, Backend::CUDA>>;

template <typename Config>
class OrmqrCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(OrmqrCtaTest, OrmqrCtaTestTypes);

TYPED_TEST(OrmqrCtaTest, MatchesNetlibOrmqrLeftRightTrans) {
	using T = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 4;
	const int k = n;

	// Build a QR factorization (A_fact, tau) on the CPU queue: the reflector layout the CTA kernel reads.
	Queue ctx_cpu("cpu");
	Matrix<T, MatrixFormat::Dense> A0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/false, batch, /*seed=*/2025);
	Matrix<T, MatrixFormat::Dense> A_fact = A0;

	UnifiedVector<T> tau(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
	UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(ctx_cpu, A_fact.view(), tau.to_span()), std::byte{0});
	geqrf(ctx_cpu, A_fact.view(), tau.to_span(), ws_geqrf.to_span()).wait();
	ctx_cpu.wait();

	VectorView<T> tau_view(tau.to_span(), /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

	std::string skip_reason;
	auto run_case = [&](Side side, Transpose trans) -> bool {
		Matrix<T, MatrixFormat::Dense> C0 = Matrix<T, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/false, batch,
													   /*seed=*/(side == Side::Left ? 11 : 22) + (trans == Transpose::NoTrans ? 0 : 1));
		Matrix<T, MatrixFormat::Dense> C_cta = C0;

		UnifiedVector<std::byte> ws_dummy(1, std::byte{0});
		try {
			ormqx_cta(*this->ctx,
			             A_fact.view(),
			             tau_view,
			             C_cta.view(),
			             Uplo::Upper, //QR factorization
			             side,
			             trans,
			             k,
			             ws_dummy.to_span(),
			             /*cta_wg_size_multiplier=*/1)
				.wait();
			this->ctx->wait();
		} catch (const sycl::exception& e) {
			const std::string msg = e.what();
			if (is_kernel_not_found_message(msg) || is_missing_subgroup32_message(msg)) {
				skip_reason = msg;
				return false;
			}
			throw;
		} catch (const std::exception& e) {
			const std::string msg = e.what();
			if (is_kernel_not_found_message(msg) || is_missing_subgroup32_message(msg)) {
				skip_reason = msg;
				return false;
			}
			throw;
		}

		EXPECT_VERIFY(T, batchlas::verify::Check::blas, n, test_utils::ormqr_apply_error(A_fact, tau_view, C0, C_cta, side, trans))
			<< "side " << int(side) << " trans " << int(trans);
		return true;
	};

	if (!run_case(Side::Left, Transpose::NoTrans) ||
		!run_case(Side::Left, Transpose::Trans) ||
		!run_case(Side::Right, Transpose::NoTrans) ||
		!run_case(Side::Right, Transpose::Trans)) {
		GTEST_SKIP() << "Skipping ormq_cta due to runtime limitation: " << skip_reason;
	}
}

TYPED_TEST(OrmqrCtaTest, ThrowsOnNTooLarge) {
	using T = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 33;
	Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
	Matrix<T, MatrixFormat::Dense> C = Matrix<T, MatrixFormat::Dense>::Random(n, n);
	Vector<T> tau(n, /*batch=*/1);
	UnifiedVector<std::byte> ws_dummy(1, std::byte{0});

	EXPECT_THROW(
		((void)ormqx_cta(*this->ctx,
		            A.view(),
		            VectorView<T>(tau),
		            C.view(),
		            Uplo::Upper, //QR factorization
		            Side::Left,
		            Transpose::NoTrans,
		            /*k=*/n,
		            ws_dummy.to_span(),
		            1),
		 this->ctx->wait()),
		std::exception);
}

#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
