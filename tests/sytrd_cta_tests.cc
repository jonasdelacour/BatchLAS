#include <gtest/gtest.h>

#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include "test_utils.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
T abs_val(T x) {
	if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
		return std::abs(x);
	} else {
		return std::abs(static_cast<double>(x));
	}
}

template <typename Real>
Real tol_for() {
	if constexpr (std::is_same_v<Real, float>) return Real(2) * Real(test_utils::tolerance<float>());
	return Real(test_utils::tolerance<double>());
}

template <typename Real>
void ref_sytd2_upper(std::vector<Real>& a, int n, std::vector<Real>& d, std::vector<Real>& e, std::vector<Real>& tau) {
	// Unblocked SYTD2-style reference for Upper (0-based, column-major).
	// Produces d (diag), e (offdiag), tau (reflector scalars). Reflectors are stored in a(0:m-2, k) for k=1..n-1.
	auto idx = [&](int r, int c) { return r + c * n; };

	d.assign(static_cast<std::size_t>(n), Real(0));
	e.assign(static_cast<std::size_t>(std::max(0, n - 1)), Real(0));
	tau.assign(static_cast<std::size_t>(std::max(0, n - 1)), Real(0));

	if (n <= 0) return;
	if (n == 1) {
		d[0] = a[idx(0, 0)];
		return;
	}

	auto sign_nonzero = [](Real x) { return std::signbit(static_cast<double>(x)) ? Real(-1) : Real(1); };

	for (int k = n - 1; k >= 1; --k) {
		const int m = k;
		const int alpha_row = k - 1;
		const int col = k;

		Real alpha = a[idx(alpha_row, col)];
		Real xnorm2 = Real(0);
		for (int r = 0; r < m - 1; ++r) {
			const Real x = a[idx(r, col)];
			xnorm2 += x * x;
		}
		const Real xnorm = std::sqrt(xnorm2);

		Real taui = Real(0);
		Real beta = alpha;
		Real scale = Real(0);
		if (m <= 1 || xnorm == Real(0)) {
			taui = Real(0);
			beta = alpha;
			scale = Real(0);
		} else {
			beta = -sign_nonzero(alpha) * std::hypot(alpha, xnorm);
			taui = (beta - alpha) / beta;
			scale = Real(1) / (alpha - beta);
		}

		if (taui != Real(0)) {
			for (int r = 0; r < m - 1; ++r) {
				a[idx(r, col)] *= scale;
			}
		}
		a[idx(alpha_row, col)] = beta;

		e[k - 1] = beta;
		tau[k - 1] = taui;

		if (taui != Real(0)) {
			std::vector<Real> v(static_cast<std::size_t>(m), Real(0));
			for (int r = 0; r < m - 1; ++r) v[r] = a[idx(r, col)];
			v[m - 1] = Real(1);

			std::vector<Real> w(static_cast<std::size_t>(m), Real(0));
			for (int r = 0; r < m; ++r) {
				Real sum = Real(0);
				for (int c = 0; c < m; ++c) {
					sum += a[idx(r, c)] * v[c];
				}
				w[r] = taui * sum;
			}

			Real dot = Real(0);
			for (int r = 0; r < m; ++r) dot += w[r] * v[r];
			const Real alpha2 = Real(-0.5) * taui * dot;
			for (int r = 0; r < m; ++r) w[r] += alpha2 * v[r];

			for (int r = 0; r < m; ++r) {
				for (int c = 0; c < m; ++c) {
					a[idx(r, c)] -= v[r] * w[c] + w[r] * v[c];
				}
			}
		}
	}

	for (int i = 0; i < n; ++i) d[i] = a[idx(i, i)];
}

template <typename Real>
static std::vector<Real> extract_host_matrix_colmajor(const Matrix<Real, MatrixFormat::Dense>& A, int n) {
	std::vector<Real> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
	auto v = A.view();
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			out[static_cast<std::size_t>(i + j * n)] = v(i, j, 0);
		}
	}
	return out;
}

// Build Q from Householder vectors stored in A/tau, following the same loop order
// as sytrd_cta.cc.
template <Backend B, typename Real>
Matrix<Real, MatrixFormat::Dense> build_q_from_sytrd_cta(Queue& ctx,
												const Matrix<Real, MatrixFormat::Dense>& A_out,
												Vector<Real>& tau,
												int n,
												Uplo uplo) {
	// We build Q via BatchLAS `ormqr` by packing the Householder vectors into a QR-like
	// reflector matrix for a (n-1)x(n-1) subspace and embedding it into an n x n Q.
	//
	// - Lower: reflectors act on trailing submatrices, so subspace is rows/cols 1..n-1.
	// - Upper: reflectors act on leading submatrices with the implicit 1 at the bottom.
	//   We convert to QR form by reversing the subspace basis (a permutation similarity).
	if (n <= 1) return Matrix<Real, MatrixFormat::Dense>::Identity(n, /*batch_size=*/1);

	const int p = n - 1;
	auto Av = A_out.view();

	Matrix<Real, MatrixFormat::Dense> Aq(p, p, /*batch=*/1);
	Matrix<Real, MatrixFormat::Dense> Qsub = Matrix<Real, MatrixFormat::Dense>::Identity(p, /*batch_size=*/1);
	Vector<Real> tau_qr(p, /*batch=*/1);
	auto aq = Aq.view();

	// Zero Aq.
	for (int j = 0; j < p; ++j) {
		for (int i = 0; i < p; ++i) {
			aq(i, j, 0) = Real(0);
		}
	}

	if (uplo == Uplo::Lower) {
		// Pack reflectors for the subspace (global indices 1..n-1).
		for (int i = 0; i < p; ++i) {
			aq(i, i, 0) = Real(1);
			tau_qr(i, 0) = tau(i, 0);
			for (int r = i + 1; r < p; ++r) {
				// sub row r corresponds to global row (r+1)
				aq(r, i, 0) = Av(r + 1, i, 0);
			}
		}
	} else {
		// Upper: operate on subspace (global indices 0..n-2), reversed into QR form.
		// Reflector i (0..p-1) comes from column k=i+1 with implicit 1 at row i.
		// In reversed coordinates, it becomes a QR reflector starting at j = p-1-i,
		// with below-diagonal entries in reverse order.
		for (int i = 0; i < p; ++i) {
			const int j = (p - 1) - i;
			aq(j, j, 0) = Real(1);
			tau_qr(j, 0) = tau(i, 0);
			const int k = i + 1;
			// Fill below-diagonal entries for this reflector.
			for (int t = 1; t <= i; ++t) {
				// v_rev[t] = v[i - t], and v[r] is stored at A_out(r, k) for r=0..i-1.
				const int r_src = i - t;
				aq(j + t, j, 0) = Av(r_src, k, 0);
			}
		}
	}

	UnifiedVector<std::byte> ws_ormqr(
		ormqr_buffer_size<B>(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data()));

	// Apply Qsub := Qsub * Q (since Qsub starts as identity, this forms Qsub).
	// ormqr applies the product of Householder reflectors encoded in Aq/tau_qr.
	// We run it on the provided execution context.
	//
	// Some backends may require a non-empty workspace.
	ormqr<B>(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data(), ws_ormqr.to_span()).wait();

	Matrix<Real, MatrixFormat::Dense> Q = Matrix<Real, MatrixFormat::Dense>::Zeros(n, n, /*batch_size=*/1);
	auto Qv = Q.view();
	auto Qsv = Qsub.view();

	if (uplo == Uplo::Lower) {
		Qv(0, 0, 0) = Real(1);
		for (int r = 0; r < p; ++r) {
			for (int c = 0; c < p; ++c) {
				Qv(r + 1, c + 1, 0) = Qsv(r, c, 0);
			}
		}
	} else {
		// Undo the subspace reversal: Qsub_orig = J * Qsub_rev * J.
		// Also embed as a top-left block, leaving the last row/col fixed.
		for (int r = 0; r < p; ++r) {
			for (int c = 0; c < p; ++c) {
				Qv(r, c, 0) = Qsv((p - 1) - r, (p - 1) - c, 0);
			}
		}
		Qv(n - 1, n - 1, 0) = Real(1);
	}

	return Q;
}

template <typename Real>
void assert_tridiagonal_matches(const MatrixView<Real, MatrixFormat::Dense>& T,
								int n,
								Vector<Real>& d,
								Vector<Real>& e,
								Real tol) {
	for (int i = 0; i < n; ++i) {
		ASSERT_TRUE(std::isfinite(static_cast<double>(d(i, 0))));
		ASSERT_TRUE(std::isfinite(static_cast<double>(T(i, i, 0))));
		EXPECT_NEAR(T(i, i, 0), d(i, 0), tol) << "diag mismatch at i=" << i;
	}

	for (int i = 0; i < n - 1; ++i) {
		ASSERT_TRUE(std::isfinite(static_cast<double>(e(i, 0))));
		EXPECT_NEAR(T(i + 1, i, 0), e(i, 0), tol) << "offdiag mismatch at i=" << i;
		EXPECT_NEAR(T(i, i + 1, 0), e(i, 0), tol) << "offdiag mismatch at i=" << i;
	}

	// Everything beyond the first off-diagonal should be ~0.
	const Real ztol = tol * Real(50);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			if (std::abs(i - j) <= 1) continue;
			EXPECT_NEAR(T(i, j, 0), Real(0), ztol) << "non-tridiagonal at (" << i << "," << j << ")";
		}
	}
}

template <typename T, Backend B>
struct SytrdCtaConfig {
	using ScalarType = T;
	static constexpr Backend BackendVal = B;
};

} // namespace

#include "test_utils.hh"

#if BATCHLAS_HAS_CUDA_BACKEND
using SytrdCtaTestTypes = ::testing::Types<SytrdCtaConfig<float, Backend::CUDA>, SytrdCtaConfig<double, Backend::CUDA>>;
#else
using SytrdCtaTestTypes = ::testing::Types<>;
#endif

template <typename Config>
class SytrdCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdCtaTest, SytrdCtaTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND
TYPED_TEST(SytrdCtaTest, RandomSymmetricLower) {
	using Real = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;
	const Real tol = tol_for<Real>();

	Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
	Matrix<Real, MatrixFormat::Dense> A = A0;
	Vector<Real> d(n, batch);
	Vector<Real> e(n - 1, batch);
	Vector<Real> tau(n - 1, batch);
	UnifiedVector<std::byte> ws(1, std::byte{0});

    sytrd_cta<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), /*cta_wg_size_multiplier=*/1);
    this->ctx->wait();
	/* try {
	} catch (const sycl::exception& ex) {
		if (is_kernel_not_found_message(ex.what())) GTEST_SKIP() << "Missing kernel bundle: " << ex.what();
		throw;
	} catch (const std::exception& ex) {
		if (is_kernel_not_found_message(ex.what())) GTEST_SKIP() << "Missing kernel bundle: " << ex.what();
		throw;
	} */

	const auto Q = build_q_from_sytrd_cta<B>(*this->ctx, A, tau, n, Uplo::Lower);
	Matrix<Real, MatrixFormat::Dense> AQ(n, n, batch);
	Matrix<Real, MatrixFormat::Dense> Tmat(n, n, batch);
	gemm<B>(*this->ctx, A0, Q, AQ, Real(1), Real(0), Transpose::NoTrans, Transpose::NoTrans).wait();
	gemm<B>(*this->ctx, Q, AQ, Tmat, Real(1), Real(0), Transpose::Trans, Transpose::NoTrans).wait();
	this->ctx->wait();

	assert_tridiagonal_matches(Tmat.view(), n, d, e, tol);
}

TYPED_TEST(SytrdCtaTest, RandomSymmetricUpper) {
	using Real = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;
	const Real tol = tol_for<Real>();

	Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/456);
	Matrix<Real, MatrixFormat::Dense> A = A0;
	Vector<Real> d(n, batch);
	Vector<Real> e(n - 1, batch);
	Vector<Real> tau(n - 1, batch);
	UnifiedVector<std::byte> ws(1, std::byte{0});

    sytrd_cta<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Upper, ws.to_span(), /*cta_wg_size_multiplier=*/1);
    this->ctx->wait();
	/* try {
	} catch (const sycl::exception& ex) {
		if (is_kernel_not_found_message(ex.what())) GTEST_SKIP() << "Missing kernel bundle: " << ex.what();
		throw;
	} catch (const std::exception& ex) {
		if (is_kernel_not_found_message(ex.what())) GTEST_SKIP() << "Missing kernel bundle: " << ex.what();
		throw;
	} */

	// Sanity: compare d/e/tau against a CPU reference implementation.
	{
		std::vector<Real> a_ref = extract_host_matrix_colmajor(A0, n);
		std::vector<Real> d_ref, e_ref, tau_ref;
		ref_sytd2_upper(a_ref, n, d_ref, e_ref, tau_ref);
		for (int i = 0; i < n; ++i) {
			EXPECT_NEAR(d(i, 0), d_ref[static_cast<std::size_t>(i)], tol) << "d mismatch vs ref at i=" << i;
		}
		for (int i = 0; i < n - 1; ++i) {
			EXPECT_NEAR(e(i, 0), e_ref[static_cast<std::size_t>(i)], tol) << "e mismatch vs ref at i=" << i;
			EXPECT_NEAR(tau(i, 0), tau_ref[static_cast<std::size_t>(i)], tol) << "tau mismatch vs ref at i=" << i;
		}

		// Diagnostic: validate that the reflector storage in A matches the reference.
		// If d/e/tau match but these don't, Q reconstruction from A will be wrong.
		auto Aoutv = A.view();
		const Real atol = tol * Real(50);
		for (int k = 1; k < n; ++k) {
			// Reflector v is stored in column k, rows 0..k-2; implicit v[k-1] = 1.
			for (int r = 0; r < k - 1; ++r) {
				SCOPED_TRACE("k=" + std::to_string(k) + " r=" + std::to_string(r));
				EXPECT_NEAR(Aoutv(r, k, 0), a_ref[static_cast<std::size_t>(r + k * n)], atol) << "reflector entry mismatch";
			}
			// The reflector's beta is stored at (k-1, k).
			SCOPED_TRACE("k=" + std::to_string(k) + " beta");
			EXPECT_NEAR(Aoutv(k - 1, k, 0), a_ref[static_cast<std::size_t>((k - 1) + k * n)], atol) << "beta storage mismatch";
		}
	}

	const auto Q = build_q_from_sytrd_cta<B>(*this->ctx, A, tau, n, Uplo::Upper);
	Matrix<Real, MatrixFormat::Dense> AQ(n, n, batch);
	Matrix<Real, MatrixFormat::Dense> Tmat(n, n, batch);
	gemm<B>(*this->ctx, A0, Q, AQ, Real(1), Real(0), Transpose::NoTrans, Transpose::NoTrans).wait();
	gemm<B>(*this->ctx, Q, AQ, Tmat, Real(1), Real(0), Transpose::Trans, Transpose::NoTrans).wait();
	this->ctx->wait();

	assert_tridiagonal_matches(Tmat.view(), n, d, e, tol);
}
#endif

#if BATCHLAS_HAS_HOST_BACKEND
TEST(SytrdCtaTest, HostBackendThrowsWithoutSubgroup32) {
	// On typical CPU devices, subgroup size 32 is unsupported; sytrd_cta should
	// throw a clear runtime_error rather than misbehaving.
	Queue ctx("cpu");

	const int n = 8;
	Matrix<float, MatrixFormat::Dense> A = Matrix<float, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, /*batch=*/1, /*seed=*/7);
	Vector<float> d(n, 1);
	Vector<float> e(n - 1, 1);
	Vector<float> tau(n - 1, 1);
	UnifiedVector<std::byte> ws(1, std::byte{0});

	EXPECT_THROW(
		(sytrd_cta<Backend::NETLIB, float>(ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), 1), ctx.wait()),
		std::exception);
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

