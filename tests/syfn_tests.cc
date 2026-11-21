// SPDX-License-Identifier: MIT
// syfn (fullerenes plan P2.2): F = V f(Lambda) V^H. Items are A = Q diag(d) Q^H with Q a product of
// two Householder reflectors (complex v for complex T) held in double, so each reference is f(d) on
// that Q, formed and compared by MatrixView's double-precision checks; the Loewdin case draws its
// graded A with fill_graded_hermitian. Operands sit in padded, poisoned storage (test_support), and
// syfn's unreferenced triangle is poisoned too.
// evidence: docs/perf/syfn.md#syfn-tests-and-breaks
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/syfn.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <random>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;
using verify::Check;
using batchlas::float_t;  // not math.h's

template <typename Scalar>
using MV = MatrixView<Scalar, MatrixFormat::Dense>;
template <typename Scalar>
using P = typename MV<Scalar>::promoted_type;
template <typename Scalar>
using Fn = SyfnFunction<float_t<Scalar>>;
using Spectra = std::vector<std::vector<double>>;  // one spectrum per item

// A vendor-free build's NETLIB items run syev, syrk/herk and gemm natively on the host
// (BATCHLAS_HAS_HOST_NATIVE), so every backend runs every case.
template <typename Config>
class SyfnTest : public test_utils::BatchLASTest<Config> {
protected:
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->IsSkipped()) return;
        if (!this->ctx->device().supports_fp64()) ref_.emplace(Device::default_device_for<double>());
    }
    // The queue of the double references (make_problem, qdq, rel_diff, MV<P<T>>): the test's, or
    // on a device without fp64 (Metal) one on Device::default_device_for<double>(), the host.
    Queue& rq() { return ref_ ? *ref_ : *this->ctx; }
    std::optional<Queue> ref_;
};

using SyfnTestTypes = typename test_utils::backend_types<ts::Config>::type;
TYPED_TEST_SUITE(SyfnTest, SyfnTestTypes);

template <typename Scalar>
Matrix<P<Scalar>> up(const MV<Scalar>& v) {
    return v.template astype<P<Scalar>>();
}

Spectra map(Spectra d, auto&& f) {
    for (auto& v : d)
        for (double& x : v) x = f(x);
    return d;
}

// Q diag(g_b) Q^H per item, through the checks' reference product.
template <typename Scalar>
Matrix<Scalar> qdq(Queue& q, const Matrix<Scalar>& Q, const Spectra& g) {
    const int n = Q.rows(), batch = Q.batch_size();
    Matrix<Scalar> D = Matrix<Scalar>::Zeros(n, n, batch), QD(n, n, batch), R(n, n, batch);
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < n; ++k) D(k, k, b) = Scalar(g[b][k]);
    (void)MV<Scalar>::product(q, Q, Transpose::NoTrans, D, Transpose::NoTrans, QD);
    (void)MV<Scalar>::product(q, QD, Transpose::NoTrans, Q, Transpose::ConjTrans, R);
    q.wait();
    return R;
}

// ||X - Y||_F / ||Y||_F, the worst item: qr_reconstruction with R = I.
template <typename Scalar>
double rel_diff(Queue& q, const Matrix<Scalar>& X, const Matrix<Scalar>& Y) {
    return MV<Scalar>::qr_reconstruction(q, Y, X, Matrix<Scalar>::Identity(Y.cols(), Y.batch_size()), Y.all_items());
}

// Item b: Q_b (double), d_b = spectrum(b), A = Q diag(d) Q^H in double and rounded to Scalar (At).
template <typename Scalar>
struct Problem {
    Matrix<P<Scalar>> Q, A;
    Matrix<Scalar> At;
    Spectra d;
};

template <typename Scalar>
Problem<Scalar> make_problem(Queue& q, int n, int batch, unsigned seed, auto&& spectrum) {
    using U = P<Scalar>;
    std::mt19937 gen(seed);
    std::normal_distribution<double> nd;
    Matrix<U> Q = Matrix<U>::Identity(n, batch);
    Spectra d;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < 2; ++r) {  // Q := Q (I - 2 v v^H / v^H v), row by row
            std::vector<U> v(n);
            double nv = 0;
            for (U& x : v) x = MV<U>::make(nd(gen), is_std_complex_v<Scalar> ? nd(gen) : 0.0), nv += std::norm(x);
            for (int i = 0; i < n; ++i) {
                U qv = 0;
                for (int k = 0; k < n; ++k) qv += U(Q(i, k, b)) * v[k];
                for (int j = 0; j < n; ++j) Q(i, j, b) -= 2.0 * qv * MV<U>::conj(v[j]) / nv;
            }
        }
        d.push_back(spectrum(b));
    }
    Matrix<U> A = qdq(q, Q, d);
    Matrix<Scalar> At = A.template astype<Scalar>();
    return {std::move(Q), std::move(A), std::move(At), std::move(d)};
}

// d_k = kappa^{-k/(n-1)}: cond kappa.
std::vector<double> graded(int n, double kappa) {
    std::vector<double> d(n);
    for (int k = 0; k < n; ++k) d[k] = std::pow(kappa, -double(k) / double(n - 1));
    return d;
}

// A's uplo triangle in poisoned storage at ld n + pad; a one-item A repeats into every item.
template <typename Scalar>
Matrix<Scalar> stored(const MV<Scalar>& A, Uplo uplo, int pad, int batch) {
    Matrix<Scalar> a = ts::padded<Scalar>(A.rows(), A.cols(), batch, pad);
    ts::store<Scalar>(a, A, Live::triangle(uplo));
    return a;
}

// F == F^H element for element (real Scalar: symmetric), every item.
template <typename Scalar>
bool exactly_hermitian(const MV<Scalar>& F) {
    for (int b = 0; b < F.batch_size(); ++b)
        for (int j = 0; j < F.cols(); ++j)
            for (int i = 0; i < F.rows(); ++i)
                if (!(Scalar(F(i, j, b)) == MV<Scalar>::conj(Scalar(F(j, i, b))))) return false;
    return true;
}

template <typename Scalar, Backend B>
std::vector<int32_t> run_syfn(Queue& q, const MV<Scalar>& A, const MV<Scalar>& F, Fn<Scalar> fn, Uplo uplo) {
    UnifiedVector<std::byte> ws(syfn_buffer_size<B, Scalar>(q, A, F, fn, uplo));
    UnifiedVector<int32_t> info(size_t(A.batch_size()), int32_t(-7));
    syfn<B, Scalar>(q, A, F, fn, uplo, ws.to_span(), info.to_span()).wait();
    return std::vector<int32_t>(info.begin(), info.end());
}

template <typename Scalar, Backend B>
std::vector<int32_t> run_from_eigen(Queue& q, const MV<Scalar>& V, UnifiedVector<float_t<Scalar>>& W,
                                    const MV<Scalar>& F, Fn<Scalar> fn) {
    UnifiedVector<std::byte> ws(syfn_from_eigen_buffer_size<B, Scalar>(q, V, W.to_span(), F, fn));
    UnifiedVector<int32_t> info(size_t(V.batch_size()), int32_t(-7));
    syfn_from_eigen<B, Scalar>(q, V, W.to_span(), F, fn, ws.to_span(), info.to_span()).wait();
    return std::vector<int32_t>(info.begin(), info.end());
}

// Loewdin acceptance: F = A^{-1/2} with ||F A F - I||_F / sqrt(n) at kappa * eps, F exactly
// Hermitian, info 0, A untouched. n = 12 runs syev's sub-group tier, 40 the blocked one.
TYPED_TEST(SyfnTest, InvSqrtOfGradedSpdIsLoewdin) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    const double kappa = ts::cond_for<T>();
    for (int n : {12, 40})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << ts::uplo_name(uplo));
            constexpr int batch = 3;
            const Matrix<T> A0 = ts::hpd<T>(q, n, batch, kappa, 11u + n);
            Matrix<T> a = stored<T>(A0, uplo, 3, batch), F = ts::padded<T>(n, n, batch, 5);
            const Matrix<T> before = a.clone();
            EXPECT_EQ((run_syfn<T, B>(q, a, F, Fn<T>::inv_sqrt(), uplo)), std::vector<int32_t>(batch, 0));
            EXPECT_TRUE(exactly_hermitian<T>(F));
            const double faf = MV<T>::similarity_residual(q, Matrix<T>::Identity(n, batch), F, A0, F.all_items());
            EXPECT_VERIFY(T, Check::factorization, n, faf / kappa);
            EXPECT_TRUE(ts::unwritten_outside<T>(F, Live::block()));
            EXPECT_TRUE(ts::items_bit_identical_to<T>(a, before));
        }
}

// exp against the closed form Q diag(e^d) Q^H, d in [-2, 2]; F is A itself (in place).
TYPED_TEST(SyfnTest, ExpMatchesClosedFormInPlace) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 24, batch = 4;
    Problem<T> p = make_problem<T>(this->rq(), n, batch, 5, [&](int b) {
        std::vector<double> d(n);
        for (int k = 0; k < n; ++k) d[k] = -2.0 + 4.0 * double((k * 7 + b) % n) / double(n - 1);
        return d;
    });
    Matrix<T> a = stored<T>(p.At, Uplo::Upper, 2, batch);
    EXPECT_EQ((run_syfn<T, B>(q, a, a, Fn<T>::exp(), Uplo::Upper)), std::vector<int32_t>(batch, 0));
    EXPECT_TRUE(exactly_hermitian<T>(a));
    const Matrix<P<T>> ref = qdq(this->rq(), p.Q, map(p.d, [](double x) { return std::exp(x); }));
    EXPECT_VERIFY(T, Check::factorization, n, rel_diff(this->rq(), up<T>(a), ref));
    EXPECT_TRUE(ts::unwritten_outside<T>(a, Live::block()));
}

// sqrt: F F = A; inverse of an indefinite A (the signed gemm form): F A = I.
TYPED_TEST(SyfnTest, SqrtAndInverseOfIndefinite) {
    using T = typename TestFixture::ScalarType;
    using U = P<T>;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 20, batch = 3;
    for (SyfnKind kind : {SyfnKind::Sqrt, SyfnKind::Inverse}) {
        SCOPED_TRACE(int(kind));
        Problem<T> p = make_problem<T>(this->rq(), n, batch, 9, [&](int b) {
            std::vector<double> d(n);
            for (int k = 0; k < n; ++k) {
                d[k] = 1.0 + 3.0 * double((k * 3 + b) % n) / double(n - 1);
                if (kind == SyfnKind::Inverse && k % 3 == 0) d[k] = -d[k];
            }
            return d;
        });
        Matrix<T> a = stored<T>(p.At, Uplo::Lower, 1, batch), F = ts::padded<T>(n, n, batch, 4);
        EXPECT_EQ((run_syfn<T, B>(q, a, F, Fn<T>{kind, {}}, Uplo::Lower)), std::vector<int32_t>(batch, 0));
        EXPECT_TRUE(exactly_hermitian<T>(F));
        const Matrix<U> f = up<T>(F), I = Matrix<U>::Identity(n, batch);
        if (kind == SyfnKind::Sqrt) {  // ||A - F F^H|| / ||A||, F exactly Hermitian
            EXPECT_VERIFY(T, Check::factorization, n, MV<U>::similarity_residual(this->rq(), p.A, f, I, p.A.all_items()));
        } else {
            Matrix<U> FA(n, n, batch);
            MV<U>::product(this->rq(), f, Transpose::NoTrans, p.A, Transpose::NoTrans, FA).wait();
            EXPECT_VERIFY(T, Check::solve, n, rel_diff(this->rq(), FA, I));
        }
        EXPECT_TRUE(ts::unwritten_outside<T>(F, Live::block()));
    }
}

// The per-iteration entry on a non-orthonormal n x m V (m < n): signed weights (gemm form) and
// inv_sqrt (herk form) against (V diag(f)) V^H in double.
TYPED_TEST(SyfnTest, FromEigenMatchesHostReference) {
    using T = typename TestFixture::ScalarType;
    using U = P<T>;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 33, m = 17, batch = 3;
    Matrix<T> V = ts::padded<T>(n, m, batch, 3);
    ts::store<T>(V, Matrix<T>::Random(n, m, false, batch, 3));
    std::mt19937 gen(3);
    std::normal_distribution<double> nd;
    UnifiedVector<float_t<T>> W(size_t(m) * batch), wts(size_t(m) * batch);
    for (size_t k = 0; k < W.size(); ++k) W[k] = float_t<T>(0.5 + std::fabs(nd(gen))), wts[k] = float_t<T>(nd(gen));
    for (bool weights : {true, false}) {
        SCOPED_TRACE(weights ? "weights" : "inv_sqrt");
        Matrix<T> F = ts::padded<T>(n, n, batch, 2);
        const auto fn = weights ? Fn<T>::with_weights(wts.to_span()) : Fn<T>::inv_sqrt();
        EXPECT_EQ((run_from_eigen<T, B>(q, V, W, F, fn)), std::vector<int32_t>(batch, 0));
        EXPECT_TRUE(exactly_hermitian<T>(F));
        const Matrix<U> v = up<T>(V);
        Matrix<U> vf = up<T>(V), R(n, n, batch);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < m; ++j) {
                const double f = weights ? double(wts[b * m + j]) : 1.0 / std::sqrt(double(W[b * m + j]));
                for (int i = 0; i < n; ++i) vf(i, j, b) *= f;
            }
        MV<U>::product(this->rq(), vf, Transpose::NoTrans, v, Transpose::ConjTrans, R).wait();
        EXPECT_VERIFY(T, Check::blas, m, rel_diff(this->rq(), up<T>(F), R));
        EXPECT_TRUE(ts::unwritten_outside<T>(F, Live::block()));
        EXPECT_TRUE(ts::unwritten_outside<T>(V, Live::block()));
    }
}

// syfn(A) with weights: weight j multiplies the eigenvector of the j-th smallest eigenvalue. The
// spectrum is distinct and stored unsorted, the weights signed and distinct. The deducing spelling
// gives the explicit one's result bit for bit.
TYPED_TEST(SyfnTest, WeightsPairWithAscendingEigenvalues) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 14, batch = 2;
    Problem<T> p = make_problem<T>(this->rq(), n, batch, 17, [&](int b) {
        std::vector<double> d(n);
        for (int k = 0; k < n; ++k) d[k] = -3.0 + 0.5 * double((k * 5 + b) % n);
        return d;
    });
    UnifiedVector<float_t<T>> wts(size_t(n) * batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j) wts[b * n + j] = float_t<T>((j % 2 ? -1.0 : 1.0) * (1.0 + j + 0.25 * b));
    Matrix<T> a = stored<T>(p.At, Uplo::Upper, 2, batch), F = ts::padded<T>(n, n, batch, 3),
              F2 = ts::padded<T>(n, n, batch, 3);
    const auto fn = Fn<T>::with_weights(wts.to_span());
    EXPECT_EQ((run_syfn<T, B>(q, a, F, fn, Uplo::Upper)), std::vector<int32_t>(batch, 0));
    Spectra g(batch, std::vector<double>(n));
    for (int b = 0; b < batch; ++b) {
        std::vector<int> order(n);
        for (int k = 0; k < n; ++k) order[k] = k;
        std::sort(order.begin(), order.end(), [&](int x, int y) { return p.d[b][x] < p.d[b][y]; });
        for (int j = 0; j < n; ++j) g[b][order[j]] = double(wts[b * n + j]);
    }
    EXPECT_TRUE(exactly_hermitian<T>(F));
    EXPECT_VERIFY(T, Check::factorization, n, rel_diff(this->rq(), up<T>(F), qdq(this->rq(), p.Q, g)));
    EXPECT_TRUE(ts::unwritten_outside<T>(F, Live::block()));
    UnifiedVector<std::byte> ws(syfn_buffer_size(q, a.view(), F2.view(), fn, Uplo::Upper));
    syfn(q, a.view(), F2.view(), fn, Uplo::Upper, ws.to_span(), Span<int32_t>{}).wait();
    EXPECT_TRUE(ts::items_bit_identical_to<T>(F2, F)) << "deducing spelling";
}

// Item 1 has two negative eigenvalues: info = {0, 2, 0}; items 0 and 2 are A^{-1/2}, item 1 is
// f on the positive eigenspace. Exact zeros through syfn_from_eigen: Inverse counts them, Sqrt
// takes them, InvSqrt counts them.
TYPED_TEST(SyfnTest, OutOfDomainEigenvaluesSetInfo) {
    using T = typename TestFixture::ScalarType;
    using R = float_t<T>;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 16, batch = 3;
    Problem<T> p = make_problem<T>(this->rq(), n, batch, 21, [&](int b) {
        std::vector<double> d = graded(n, 100.0);
        if (b == 1) d[3] = -0.5, d[9] = -2.0;
        return d;
    });
    Matrix<T> a = stored<T>(p.At, Uplo::Lower, 2, batch), F = ts::padded<T>(n, n, batch, 1);
    EXPECT_EQ((run_syfn<T, B>(q, a, F, Fn<T>::inv_sqrt(), Uplo::Lower)), (std::vector<int32_t>{0, 2, 0}));
    const Matrix<P<T>> ref = qdq(this->rq(), p.Q, map(p.d, [](double x) { return x > 0 ? 1.0 / std::sqrt(x) : 0.0; }));
    EXPECT_VERIFY_SLACK(T, Check::factorization, n, rel_diff(this->rq(), up<T>(F), ref),
                        verify::Slack{100.0, "forward error of A^{-1/2} scales with cond(A) = 100"});
    constexpr int m = 4;
    Matrix<T> V = ts::padded<T>(n, m, 1, 0), G = ts::padded<T>(n, n, 1, 0);
    ts::store<T>(V, Matrix<T>::Identity(n, 1));
    UnifiedVector<R> W(m);
    W[0] = R(0), W[1] = R(-1), W[2] = R(0), W[3] = R(16);
    const double tol = 8.0 * std::numeric_limits<R>::epsilon();
    auto diag = [&](int k) { return std::complex<double>(MV<T>::up(T(G(k, k, 0)))); };
    EXPECT_EQ((run_from_eigen<T, B>(q, V, W, G, Fn<T>::inverse())), std::vector<int32_t>{2});
    EXPECT_NEAR(diag(1).real(), -1.0, tol) << "inverse of -1";
    EXPECT_NEAR(diag(3).real(), 1.0 / 16.0, tol / 16.0);
    EXPECT_EQ(diag(0), 0.0) << "an out-of-domain value contributes 0";
    EXPECT_EQ((run_from_eigen<T, B>(q, V, W, G, Fn<T>::sqrt())), std::vector<int32_t>{1});
    EXPECT_NEAR(diag(3).real(), 4.0, 4.0 * tol);
    EXPECT_EQ(diag(1), 0.0);
    EXPECT_EQ((run_from_eigen<T, B>(q, V, W, G, Fn<T>::inv_sqrt())), std::vector<int32_t>{3});
    EXPECT_NEAR(diag(3).real(), 0.25, 0.25 * tol);
    EXPECT_EQ(diag(2), 0.0);
}

// 1024 identical items: every F bit-identical to item 0, item 0 correct.
TYPED_TEST(SyfnTest, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 16, batch = 1024;
    Problem<T> p = make_problem<T>(this->rq(), n, 1, 8, [&](int) { return graded(n, 1.0e2); });
    Matrix<T> a = stored<T>(p.At, Uplo::Lower, 1, batch), F = ts::padded<T>(n, n, batch, 3);
    run_syfn<T, B>(q, a, F, Fn<T>::inv_sqrt(), Uplo::Lower);
    EXPECT_TRUE(ts::items_bit_identical<T>(F));
    const Matrix<P<T>> ref = qdq(this->rq(), p.Q, map(p.d, [](double x) { return 1.0 / std::sqrt(x); }));
    EXPECT_VERIFY_SLACK(T, Check::factorization, n, rel_diff(this->rq(), up<T>(F.batch_item(0)), ref),
                        verify::Slack{100.0, "forward error of A^{-1/2} scales with cond(A) = 100"});
}

// An out-of-order queue gives the in-order result bit for bit. Its A is written by a copy still
// pending on that queue, and the batch is large enough that the internal copy, syev and product
// overlap if nothing orders them. Still a race: an unordered run can be lucky.
TYPED_TEST(SyfnTest, OutOfOrderQueueMatchesInOrder) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    constexpr int n = 48, batch = 256;
    Matrix<T> a = stored<T>(ts::hpd<T>(q, n, batch, 1.0e2, 4), Uplo::Lower, 1, batch);
    Matrix<T> a2 = ts::padded<T>(n, n, batch, 1), F1 = ts::padded<T>(n, n, batch, 1), F2 = ts::padded<T>(n, n, batch, 1);
    run_syfn<T, B>(q, a, F1, Fn<T>::inv_sqrt(), Uplo::Lower);
    Queue ooo(q.device(), B, false);
    (void)MV<T>::copy(ooo, a2, a);
    run_syfn<T, B>(ooo, a2, F2, Fn<T>::inv_sqrt(), Uplo::Lower);
    EXPECT_TRUE(ts::items_bit_identical_to<T>(F2, F1));
}

// m = 0 eigenpairs: F is exactly zero.
TYPED_TEST(SyfnTest, NoEigenpairsGivesZero) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Matrix<T> V = ts::padded<T>(6, 0, 2, 0), F = ts::padded<T>(6, 6, 2, 2);
    UnifiedVector<float_t<T>> W(1);
    EXPECT_EQ((run_from_eigen<T, B>(*this->ctx, V, W, F, Fn<T>::exp())), (std::vector<int32_t>{0, 0}));
    int nonzero = 0;
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < 6; ++j)
            for (int i = 0; i < 6; ++i) nonzero += !(T(F(i, j, b)) == T(0));
    EXPECT_EQ(nonzero, 0);
    EXPECT_TRUE(ts::unwritten_outside<T>(F, Live::block()));
}

TYPED_TEST(SyfnTest, RejectsBadArguments) {
    using T = typename TestFixture::ScalarType;
    using R = float_t<T>;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    Matrix<T> A(6, 6, 2), Rect(6, 5, 2), F(6, 6, 2), F3(6, 6, 3), V(6, 4, 2), Vw(4, 6, 2);
    UnifiedVector<R> W(8), Wshort(7);
    UnifiedVector<int32_t> info(2), infoshort(1);
    UnifiedVector<std::byte> ws(1 << 20);
    const auto inv = Fn<T>::inv_sqrt();
    auto f = [&](const MV<T>& a, const MV<T>& o, Fn<T> fn, Uplo u, size_t bytes, Span<int32_t> in) {
        syfn<B, T>(q, a, o, fn, u, ws.to_span().subspan(0, bytes), in).wait();
    };
    auto e = [&](const MV<T>& v, UnifiedVector<R>& w, const MV<T>& o, Fn<T> fn, size_t bytes) {
        syfn_from_eigen<B, T>(q, v, w.to_span(), o, fn, ws.to_span().subspan(0, bytes), Span<int32_t>{}).wait();
    };
    EXPECT_THROW(f(Rect, F, inv, Uplo::Lower, ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F3, inv, Uplo::Lower, ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, static_cast<Uplo>(77), ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, Uplo::Lower, ws.size(), infoshort.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, Fn<T>::with_weights(Wshort.to_span()), Uplo::Lower, ws.size(), info.to_span()),
                 batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, Fn<T>{static_cast<SyfnKind>(9), {}}, Uplo::Lower, ws.size(), info.to_span()),
                 batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, Uplo::Lower, 64, info.to_span()), batchlas::workspace_error);
    EXPECT_THROW(e(V, Wshort, F, inv, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(Vw, W, F, inv, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(V, W, Rect, inv, ws.size()), batchlas::invalid_argument);
    Matrix<T> Fv(6, 6, 2);
    EXPECT_THROW(e(MV<T>(Fv, 6, 4), W, Fv, inv, ws.size()), batchlas::invalid_argument) << "F aliases V";
    const MV<T> Fvl(Fv.data_ptr() + 2 * 6, 6, 4, 6, 36, 2);
    EXPECT_THROW(e(Fvl, W, Fv, inv, ws.size()), batchlas::invalid_argument) << "V is later columns of F";
    if (q.device().type == DeviceType::GPU) {
        std::vector<R> host_w(8, R(1));
        const auto hw = Fn<T>::with_weights(Span<R>(host_w.data(), host_w.size()));
        EXPECT_THROW(e(V, W, F, hw, ws.size()), batchlas::invalid_argument) << "host weights, explicit";
        EXPECT_THROW((void)syfn_from_eigen(q, MV<T>(V), W.to_span(), MV<T>(F), hw, ws.to_span(), Span<int32_t>{}),
                     batchlas::invalid_argument)
            << "host weights, deducing";
    }
    EXPECT_THROW(e(V, W, F, Fn<T>::with_weights(Wshort.to_span()), ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(V, W, F, inv, 16), batchlas::workspace_error);
}

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
// Vendor-free on a CPU queue (a CPU target without LAPACKE/CBLAS is always BATCHLAS_HAS_HOST_NATIVE):
// syfn sizes and runs syev, syrk and gemm natively. A = 4 I gives F = A^{-1/2} = I / 2, and
// syfn_from_eigen F = V exp(W) V^T = e I; info is 0 on every item.
TEST(SyfnVendorFree, HostWithoutLapackeRunsNatively) {
    if (BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS) GTEST_SKIP() << "host LAPACKE/CBLAS is built in";
    ASSERT_TRUE(BATCHLAS_HAS_HOST_NATIVE) << "a CPU target without LAPACKE/CBLAS is host-native";
    Queue q(Device("cpu"), Backend::NETLIB, true);
    Matrix<float> A = Matrix<float>::Identity(8, 3), F(8, 8, 3), V = Matrix<float>::Identity(8, 3);
    for (int b = 0; b < 3; ++b)
        for (int i = 0; i < 8; ++i) A(i, i, b) = 4.0f;
    UnifiedVector<float> W(24, 1.0f);
    UnifiedVector<int32_t> info(3, int32_t(-7));
    const auto inv = SyfnFunction<float>::inv_sqrt();
    const size_t need = syfn_buffer_size<Backend::NETLIB, float>(q, A, F, inv, Uplo::Lower);
    UnifiedVector<std::byte> ws(std::max<size_t>(need, 1));
    (void)syfn<Backend::NETLIB, float>(q, A, F, inv, Uplo::Lower, Span<std::byte>(ws.data(), need), info.to_span());
    q.wait();
    for (int b = 0; b < 3; ++b) {
        EXPECT_EQ(info[b], 0) << "item " << b;
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                EXPECT_NEAR(F(i, j, b), i == j ? 0.5f : 0.0f, 1e-5f) << "inv_sqrt " << i << "," << j << " item " << b;
    }
    EXPECT_EQ((run_from_eigen<float, Backend::NETLIB>(q, V, W, F, SyfnFunction<float>::exp())),
              std::vector<int32_t>(3, 0));
    for (int b = 0; b < 3; ++b)
        for (int j = 0; j < 8; ++j)
            for (int i = 0; i < 8; ++i)
                EXPECT_NEAR(F(i, j, b), i == j ? std::exp(1.0f) : 0.0f, 1e-5f) << i << "," << j << " item " << b;
}
#endif

}  // namespace
