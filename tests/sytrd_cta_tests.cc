#include <gtest/gtest.h>

#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include "test_utils.hh"

#include <batchlas/verify/norms.hh>
#include <batchlas/verify/residuals.hh>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

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
		ormqr_buffer_size(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data()));

	// Apply Qsub := Qsub * Q (since Qsub starts as identity, this forms Qsub).
	// ormqr applies the product of Householder reflectors encoded in Aq/tau_qr.
	// We run it on the provided execution context.
	//
	// Some backends may require a non-empty workspace.
	ormqr(ctx, Aq.view(), Qsub.view(), Side::Left, Transpose::NoTrans, tau_qr.data(), ws_ormqr.to_span()).wait();

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

// ||Tmat - tridiag(d, e)||_F / ||A0||_F over the whole matrix: the band must be (d, e) and everything
// outside it zero.
template <typename Real>
double tridiagonal_error(const Matrix<Real, MatrixFormat::Dense>& A0, const Matrix<Real, MatrixFormat::Dense>& Tmat,
						 int n, Vector<Real>& d, Vector<Real>& e) {
	std::vector<double> D(static_cast<std::size_t>(n) * n);
	for (int j = 0; j < n; ++j) {
		for (int i = 0; i < n; ++i) {
			double t = 0;
			if (i == j) t = batchlas::verify::up(d(i, 0));
			else if (std::abs(i - j) == 1) t = batchlas::verify::up(e(std::min(i, j), 0));
			D[static_cast<std::size_t>(i) + j * n] = batchlas::verify::up(Tmat.view()(i, j, 0)) - t;
		}
	}
	return batchlas::verify::frobenius(batchlas::verify::view(D.data(), n, n, n), 0) / batchlas::verify::frobenius(A0.view(), 0);
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
#elif BATCHLAS_HAS_ROCM_BACKEND
using SytrdCtaTestTypes = ::testing::Types<SytrdCtaConfig<float, Backend::ROCM>, SytrdCtaConfig<double, Backend::ROCM>>;
#else
using SytrdCtaTestTypes = ::testing::Types<SytrdCtaConfig<float, Backend::NETLIB>>;
#endif

template <typename Config>
class SytrdCtaTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SytrdCtaTest, SytrdCtaTestTypes);

#if BATCHLAS_HAS_CUDA_BACKEND || BATCHLAS_HAS_ROCM_BACKEND
TYPED_TEST(SytrdCtaTest, RandomSymmetricLower) {
	using Real = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;

	Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/123);
	Matrix<Real, MatrixFormat::Dense> A = A0;
	Vector<Real> d(n, batch);
	Vector<Real> e(n - 1, batch);
	Vector<Real> tau(n - 1, batch);
	UnifiedVector<std::byte> ws(1, std::byte{0});

    (void)sytrd_cta<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Lower, ws.to_span(), /*cta_wg_size_multiplier=*/1);
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
	gemm(*this->ctx, A0.view(), Q.view(), AQ.view(), {.alpha = Real(1), .beta = Real(0)}).wait();
	gemm(*this->ctx, Q.view(), AQ.view(), Tmat.view(), {.alpha = Real(1), .beta = Real(0), .transA = Transpose::Trans}).wait();
	this->ctx->wait();

	EXPECT_VERIFY(Real, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q.view()));
	EXPECT_VERIFY(Real, batchlas::verify::Check::factorization, n, tridiagonal_error(A0, Tmat, n, d, e));
}

TYPED_TEST(SytrdCtaTest, RandomSymmetricUpper) {
	using Real = typename TestFixture::ScalarType;
	constexpr Backend B = TestFixture::BackendType;

	const int n = 16;
	const int batch = 1;

	Matrix<Real, MatrixFormat::Dense> A0 = Matrix<Real, MatrixFormat::Dense>::Random(n, n, /*hermitian=*/true, batch, /*seed=*/456);
	Matrix<Real, MatrixFormat::Dense> A = A0;
	Vector<Real> d(n, batch);
	Vector<Real> e(n - 1, batch);
	Vector<Real> tau(n - 1, batch);
	UnifiedVector<std::byte> ws(1, std::byte{0});

    (void)sytrd_cta<B, Real>(*this->ctx, A.view(), d, e, tau, Uplo::Upper, ws.to_span(), /*cta_wg_size_multiplier=*/1);
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
		// d and e against the reference relative to ||A0||_F; tau and the stored reflectors are O(1) already.
		double dev_de = 0, dev_refl = 0;
		for (int i = 0; i < n; ++i) dev_de = batchlas::verify::nanmax(dev_de, std::abs(double(d(i, 0)) - double(d_ref[static_cast<std::size_t>(i)])));
		for (int i = 0; i < n - 1; ++i) {
			dev_de = batchlas::verify::nanmax(dev_de, std::abs(double(e(i, 0)) - double(e_ref[static_cast<std::size_t>(i)])));
			dev_refl = batchlas::verify::nanmax(dev_refl, std::abs(double(tau(i, 0)) - double(tau_ref[static_cast<std::size_t>(i)])));
		}
		// Diagnostic: the reflector storage in A must match the reference. If d/e/tau match but these
		// don't, Q reconstruction from A will be wrong.
		auto Aoutv = A.view();
		for (int k = 1; k < n; ++k) {
			// Reflector v is stored in column k, rows 0..k-2; implicit v[k-1] = 1. Its beta is stored at (k-1, k).
			for (int r = 0; r < k - 1; ++r)
				dev_refl = batchlas::verify::nanmax(dev_refl, std::abs(double(Aoutv(r, k, 0)) - double(a_ref[static_cast<std::size_t>(r + k * n)])));
			dev_de = batchlas::verify::nanmax(dev_de, std::abs(double(Aoutv(k - 1, k, 0)) - double(a_ref[static_cast<std::size_t>((k - 1) + k * n)])));
		}
		EXPECT_VERIFY(Real, batchlas::verify::Check::factorization, n, dev_de / batchlas::verify::frobenius(A0.view(), 0));
		EXPECT_VERIFY_SLACK(Real, batchlas::verify::Check::factorization, n, dev_refl,
							batchlas::verify::Slack{2.0, "unit-scale tau and reflector entries v = x / (alpha - beta), each rounded in both the kernel and the host sytd2: measured need 0.30 float, 0.51 double"});
	}

	const auto Q = build_q_from_sytrd_cta<B>(*this->ctx, A, tau, n, Uplo::Upper);
	Matrix<Real, MatrixFormat::Dense> AQ(n, n, batch);
	Matrix<Real, MatrixFormat::Dense> Tmat(n, n, batch);
	gemm(*this->ctx, A0.view(), Q.view(), AQ.view(), {.alpha = Real(1), .beta = Real(0)}).wait();
	gemm(*this->ctx, Q.view(), AQ.view(), Tmat.view(), {.alpha = Real(1), .beta = Real(0), .transA = Transpose::Trans}).wait();
	this->ctx->wait();

	EXPECT_VERIFY(Real, batchlas::verify::Check::orthogonality, n, batchlas::verify::orthogonality(Q.view()));
	EXPECT_VERIFY(Real, batchlas::verify::Check::factorization, n, tridiagonal_error(A0, Tmat, n, d, e));
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
		((void)sytrd_cta(ctx, A.view(), d.view(), e.view(), tau.view(), Uplo::Lower, ws.to_span(), 1), ctx.wait()),
		std::exception);
}
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}

