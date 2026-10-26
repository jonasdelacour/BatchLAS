// syfn (fullerenes plan P2.2): F = V f(Lambda) V^H. Matrices are built on the host as
// A = Q diag(d) Q^H with Q a product of two Householder reflectors (complex v for complex T), so
// every reference is a closed form in double that shares no code with the library. Operands have
// a padded ld and a stride gap holding a large finite poison; syfn's unreferenced triangle is
// poisoned too.
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/syfn.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>

#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace batchlas;

namespace {

using verify::Check;
using cd = std::complex<double>;
using HostMat = std::vector<cd>;  // n x n column-major

template <typename T>
using Real = batchlas::float_t<T>;

template <typename T, Backend B>
struct SyfnConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SyfnTestTypes = typename test_utils::backend_types<SyfnConfig>::type;

constexpr bool kHostVendor = bool(BATCHLAS_HAS_LAPACKE) && bool(BATCHLAS_HAS_CBLAS);

// A vendor-free build has no host syev or syrk, so its NETLIB items are covered only by
// SyfnVendorFree.HostWithoutLapackeThrowsNoRoute.
template <typename Config>
class SyfnTest : public test_utils::BatchLASTest<Config> {
protected:
    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->IsSkipped()) return;
        if (test_utils::BatchLASTest<Config>::BackendType == Backend::NETLIB && !kHostVendor)
            GTEST_SKIP() << "no host LAPACKE/CBLAS: syfn has no route on NETLIB";
    }
};

TYPED_TEST_SUITE(SyfnTest, SyfnTestTypes);

constexpr double kPoison = 9.0e3;

template <typename T>
T to_t(cd z) {
    if constexpr (is_std_complex_v<T>) return T(Real<T>(z.real()), Real<T>(z.imag()));
    else return T(z.real());
}

template <typename T>
cd to_cd(T x) {
    if constexpr (is_std_complex_v<T>) return cd(double(x.real()), double(x.imag()));
    else return cd(double(x), 0.0);
}

template <typename T>
double kappa_for() { return std::is_same_v<Real<T>, float> ? 1.0e3 : 1.0e6; }

// Two Householder reflectors; v has a nonzero imaginary part for complex T.
template <typename T>
HostMat householder_q(int n, std::mt19937& gen) {
    std::normal_distribution<double> nd;
    HostMat Q(size_t(n) * n, cd(0));
    for (int i = 0; i < n; ++i) Q[i * n + i] = 1;
    for (int r = 0; r < 2; ++r) {
        std::vector<cd> v(n);
        double nv = 0;
        for (auto& x : v) {
            x = cd(nd(gen), is_std_complex_v<T> ? nd(gen) : 0.0);
            nv += std::norm(x);
        }
        HostMat Qn(Q);  // Q := Q (I - 2 v v^H / nv)
        for (int i = 0; i < n; ++i) {
            cd qv = 0;
            for (int k = 0; k < n; ++k) qv += Q[k * n + i] * v[k];
            for (int j = 0; j < n; ++j) Qn[j * n + i] -= 2.0 * qv * std::conj(v[j]) / nv;
        }
        Q = Qn;
    }
    return Q;
}

// Q diag(g) Q^H.
HostMat q_diag_qh(const HostMat& Q, const std::vector<double>& g, int n) {
    HostMat R(size_t(n) * n, cd(0));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            cd s = 0;
            for (int k = 0; k < n; ++k) s += Q[k * n + i] * g[k] * std::conj(Q[k * n + j]);
            R[j * n + i] = s;
        }
    return R;
}

HostMat mul(const HostMat& X, const HostMat& Y, int n) {
    HostMat R(size_t(n) * n, cd(0));
    for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
            for (int i = 0; i < n; ++i) R[j * n + i] += X[k * n + i] * Y[j * n + k];
    return R;
}

double frob_diff(const HostMat& X, const HostMat& Y) {
    double d = 0, y = 0;
    for (size_t k = 0; k < X.size(); ++k) d += std::norm(X[k] - Y[k]), y += std::norm(Y[k]);
    return std::sqrt(d / y);
}

// Padded storage: ld = rows + pad, stride ld * cols + 7, everything poisoned.
template <typename T>
struct Padded {
    Matrix<T> store;
    MatrixView<T> view;
    Padded(int rows, int cols, int pad, int batch)
        : store((rows + pad) * cols + 7, batch),
          view(store.data_ptr(), rows, cols, rows + pad, (rows + pad) * cols + 7, batch) {
        std::fill(store.data_ptr(), store.data_ptr() + int64_t((rows + pad) * cols + 7) * batch,
                  T(Real<T>(kPoison)));
    }
    bool pad_intact() const {
        const T* p = view.data_ptr();
        for (int b = 0; b < view.batch_size(); ++b)
            for (int k = 0; k < view.stride(); ++k) {
                const int j = k / view.ld(), i = k % view.ld();
                if (j < view.cols() && i < view.rows()) continue;
                if (p[int64_t(b) * view.stride() + k] != T(Real<T>(kPoison))) return false;
            }
        return true;
    }
    size_t elems() const { return size_t(view.stride()) * size_t(view.batch_size()); }
    std::vector<T> raw() const { return std::vector<T>(view.data_ptr(), view.data_ptr() + elems()); }
    HostMat item(int b) const {
        HostMat R(size_t(view.rows()) * view.cols());
        for (int j = 0; j < view.cols(); ++j)
            for (int i = 0; i < view.rows(); ++i) R[j * view.rows() + i] = to_cd(view(i, j, b));
        return R;
    }
};

// Writes the uplo triangle of H into item b of A; the other triangle keeps its poison.
template <typename T>
void store_triangle(MatrixView<T> A, int b, const HostMat& H, Uplo uplo) {
    const int n = A.rows();
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (uplo == Uplo::Lower ? i >= j : i <= j) A(i, j, b) = to_t<T>(H[j * n + i]);
}

template <typename T>
bool exactly_hermitian(const MatrixView<T>& F, int b) {
    for (int j = 0; j < F.cols(); ++j)
        for (int i = 0; i < F.rows(); ++i) {
            const T x = F(i, j, b), y = F(j, i, b);
            if constexpr (is_std_complex_v<T>) {
                if (x.real() != y.real() || x.imag() != -y.imag()) return false;
            } else if (x != y) {
                return false;
            }
        }
    return true;
}

template <typename T, Backend B>
std::vector<int32_t> run_syfn(Queue& q, const MatrixView<T>& A, const MatrixView<T>& F, SyfnFunction<Real<T>> fn,
                              Uplo uplo) {
    UnifiedVector<std::byte> ws(syfn_buffer_size<B, T>(q, A, F, fn, uplo));
    UnifiedVector<int32_t> info(size_t(A.batch_size()), int32_t(-7));
    syfn<B, T>(q, A, F, fn, uplo, ws.to_span(), info.to_span()).wait();
    return std::vector<int32_t>(info.begin(), info.end());
}

template <typename T, Backend B>
std::vector<int32_t> run_from_eigen(Queue& q, const MatrixView<T>& V, UnifiedVector<Real<T>>& W,
                                    const MatrixView<T>& F, SyfnFunction<Real<T>> fn) {
    UnifiedVector<std::byte> ws(syfn_from_eigen_buffer_size<B, T>(q, V, W.to_span(), F, fn));
    UnifiedVector<int32_t> info(size_t(V.batch_size()), int32_t(-7));
    syfn_from_eigen<B, T>(q, V, W.to_span(), F, fn, ws.to_span(), info.to_span()).wait();
    return std::vector<int32_t>(info.begin(), info.end());
}

// Graded spectra d_k = kappa^{-k/(n-1)} (cond kappa), A_b = Q_b diag(d) Q_b^H.
struct Problem {
    std::vector<HostMat> Q, A;
    std::vector<std::vector<double>> d;
};

template <typename T>
Problem make_problem(int n, int batch, unsigned seed, auto&& spectrum) {
    std::mt19937 gen(seed);
    Problem p;
    for (int b = 0; b < batch; ++b) {
        p.Q.push_back(householder_q<T>(n, gen));
        p.d.push_back(spectrum(b));
        p.A.push_back(q_diag_qh(p.Q.back(), p.d.back(), n));
    }
    return p;
}

std::vector<double> graded(int n, double kappa) {
    std::vector<double> d(n);
    for (int k = 0; k < n; ++k) d[k] = std::pow(kappa, -double(k) / double(n - 1));
    return d;
}

// Loewdin acceptance: F = A^{-1/2} with ||F A F - I||_F / sqrt(n) at kappa * eps, F exactly
// Hermitian, info 0, A untouched. n = 12 runs syev's sub-group tier, 40 the blocked one.
TYPED_TEST(SyfnTest, InvSqrtOfGradedSpdIsLoewdin) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const double kappa = kappa_for<T>();
    for (int n : {12, 40})
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            SCOPED_TRACE(::testing::Message() << "n=" << n << " uplo=" << int(uplo));
            constexpr int batch = 3;
            Problem p = make_problem<T>(n, batch, 11u + n, [&](int) { return graded(n, kappa); });
            Padded<T> A(n, n, 3, batch), F(n, n, 5, batch);
            for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b], uplo);
            const std::vector<T> A0 = A.raw();
            const std::vector<int32_t> info = run_syfn<T, B>(*this->ctx, A.view, F.view, SyfnFunction<Real<T>>::inv_sqrt(), uplo);
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(info[b], 0);
                EXPECT_TRUE(exactly_hermitian(F.view, b));
                const HostMat Fh = F.item(b), FAF = mul(mul(Fh, p.A[b], n), Fh, n);
                HostMat I(size_t(n) * n, cd(0));
                for (int i = 0; i < n; ++i) I[i * n + i] = 1;
                double r = 0;
                for (size_t k = 0; k < I.size(); ++k) r += std::norm(FAF[k] - I[k]);
                EXPECT_VERIFY(T, Check::factorization, n, std::sqrt(r / n) / kappa);
            }
            EXPECT_TRUE(F.pad_intact());
            EXPECT_TRUE(A0 == A.raw());
        }
}

// exp against the closed form Q diag(e^d) Q^H, d in [-2, 2]; F is A itself (in place).
TYPED_TEST(SyfnTest, ExpMatchesClosedFormInPlace) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 24, batch = 4;
    Problem p = make_problem<T>(n, batch, 5, [&](int b) {
        std::vector<double> d(n);
        for (int k = 0; k < n; ++k) d[k] = -2.0 + 4.0 * double((k * 7 + b) % n) / double(n - 1);
        return d;
    });
    Padded<T> A(n, n, 2, batch);
    for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b], Uplo::Upper);
    const std::vector<int32_t> info = run_syfn<T, B>(*this->ctx, A.view, A.view, SyfnFunction<Real<T>>::exp(), Uplo::Upper);
    for (int b = 0; b < batch; ++b) {
        std::vector<double> e(n);
        for (int k = 0; k < n; ++k) e[k] = std::exp(p.d[b][k]);
        EXPECT_EQ(info[b], 0);
        EXPECT_TRUE(exactly_hermitian(A.view, b));
        EXPECT_VERIFY(T, Check::factorization, n, frob_diff(A.item(b), q_diag_qh(p.Q[b], e, n)));
    }
    EXPECT_TRUE(A.pad_intact());
}

// sqrt: F F = A; inverse of an indefinite A (the signed gemm form): F A = I.
TYPED_TEST(SyfnTest, SqrtAndInverseOfIndefinite) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 20, batch = 3;
    for (SyfnKind kind : {SyfnKind::Sqrt, SyfnKind::Inverse}) {
        SCOPED_TRACE(int(kind));
        Problem p = make_problem<T>(n, batch, 9, [&](int b) {
            std::vector<double> d(n);
            for (int k = 0; k < n; ++k) {
                d[k] = 1.0 + 3.0 * double((k * 3 + b) % n) / double(n - 1);
                if (kind == SyfnKind::Inverse && k % 3 == 0) d[k] = -d[k];
            }
            return d;
        });
        Padded<T> A(n, n, 1, batch), F(n, n, 4, batch);
        for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b], Uplo::Lower);
        const std::vector<int32_t> info = run_syfn<T, B>(*this->ctx, A.view, F.view, SyfnFunction<Real<T>>{kind, {}}, Uplo::Lower);
        for (int b = 0; b < batch; ++b) {
            EXPECT_EQ(info[b], 0);
            EXPECT_TRUE(exactly_hermitian(F.view, b));
            const HostMat Fh = F.item(b);
            if (kind == SyfnKind::Sqrt) {
                EXPECT_VERIFY(T, Check::factorization, n, frob_diff(mul(Fh, Fh, n), p.A[b]));
            } else {
                HostMat I(size_t(n) * n, cd(0));
                for (int i = 0; i < n; ++i) I[i * n + i] = 1;
                EXPECT_VERIFY(T, Check::solve, n, frob_diff(mul(Fh, p.A[b], n), I));
            }
        }
        EXPECT_TRUE(F.pad_intact());
    }
}

// The per-iteration entry on a non-orthonormal n x m V (m < n): signed weights (gemm form) and
// inv_sqrt (herk form) against a host sum over j of f_j v_j v_j^H.
TYPED_TEST(SyfnTest, FromEigenMatchesHostReference) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 33, m = 17, batch = 3;
    std::mt19937 gen(3);
    std::normal_distribution<double> nd;
    Padded<T> V(n, m, 3, batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < n; ++i) V.view(i, j, b) = to_t<T>(cd(nd(gen), is_std_complex_v<T> ? nd(gen) : 0.0));
    UnifiedVector<Real<T>> W(size_t(m) * batch), wts(size_t(m) * batch);
    for (size_t k = 0; k < W.size(); ++k) W[k] = Real<T>(0.5 + std::fabs(nd(gen))), wts[k] = Real<T>(nd(gen));
    for (bool weights : {true, false}) {
        SCOPED_TRACE(weights ? "weights" : "inv_sqrt");
        Padded<T> F(n, n, 2, batch);
        const auto fn = weights ? SyfnFunction<Real<T>>::with_weights(wts.to_span()) : SyfnFunction<Real<T>>::inv_sqrt();
        const std::vector<int32_t> info = run_from_eigen<T, B>(*this->ctx, V.view, W, F.view, fn);
        for (int b = 0; b < batch; ++b) {
            HostMat R(size_t(n) * n, cd(0));
            for (int j = 0; j < m; ++j) {
                const double f = weights ? double(wts[b * m + j]) : 1.0 / std::sqrt(double(W[b * m + j]));
                for (int c = 0; c < n; ++c)
                    for (int r = 0; r < n; ++r) R[c * n + r] += f * to_cd(V.view(r, j, b)) * std::conj(to_cd(V.view(c, j, b)));
            }
            EXPECT_EQ(info[b], 0);
            EXPECT_TRUE(exactly_hermitian(F.view, b));
            EXPECT_VERIFY(T, Check::blas, m, frob_diff(F.item(b), R));
        }
        EXPECT_TRUE(F.pad_intact());
        EXPECT_TRUE(V.pad_intact());
    }
}

// syfn(A) with weights: weight j multiplies the eigenvector of the j-th smallest eigenvalue. The
// spectrum is distinct and stored unsorted, the weights signed and distinct. The deducing spelling
// gives the explicit one's result bit for bit.
TYPED_TEST(SyfnTest, WeightsPairWithAscendingEigenvalues) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 14, batch = 2;
    Problem p = make_problem<T>(n, batch, 17, [&](int b) {
        std::vector<double> d(n);
        for (int k = 0; k < n; ++k) d[k] = -3.0 + 0.5 * double((k * 5 + b) % n);
        return d;
    });
    UnifiedVector<Real<T>> wts(size_t(n) * batch);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j) wts[b * n + j] = Real<T>((j % 2 ? -1.0 : 1.0) * (1.0 + j + 0.25 * b));
    Padded<T> A(n, n, 2, batch), F(n, n, 3, batch), F2(n, n, 3, batch);
    for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b], Uplo::Upper);
    const auto fn = SyfnFunction<Real<T>>::with_weights(wts.to_span());
    const std::vector<int32_t> info = run_syfn<T, B>(*this->ctx, A.view, F.view, fn, Uplo::Upper);
    for (int b = 0; b < batch; ++b) {
        std::vector<int> order(n);
        for (int k = 0; k < n; ++k) order[k] = k;
        std::sort(order.begin(), order.end(), [&](int x, int y) { return p.d[b][x] < p.d[b][y]; });
        std::vector<double> g(n);
        for (int j = 0; j < n; ++j) g[order[j]] = double(wts[b * n + j]);
        EXPECT_EQ(info[b], 0);
        EXPECT_TRUE(exactly_hermitian(F.view, b));
        EXPECT_VERIFY(T, Check::factorization, n, frob_diff(F.item(b), q_diag_qh(p.Q[b], g, n)));
    }
    EXPECT_TRUE(F.pad_intact());
    UnifiedVector<std::byte> ws(syfn_buffer_size(*this->ctx, A.view, F2.view, fn, Uplo::Upper));
    syfn(*this->ctx, A.view, F2.view, fn, Uplo::Upper, ws.to_span(), Span<int32_t>{}).wait();
    EXPECT_EQ(std::memcmp(F.view.data_ptr(), F2.view.data_ptr(), sizeof(T) * F.elems()), 0) << "deducing spelling";
}

// Item 1 has two negative eigenvalues: info = {0, 2, 0}; items 0 and 2 are A^{-1/2}, item 1 is
// f on the positive eigenspace. Exact zeros through syfn_from_eigen: Inverse counts them, Sqrt
// takes them, InvSqrt counts them.
TYPED_TEST(SyfnTest, OutOfDomainEigenvaluesSetInfo) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 16, batch = 3;
    Problem p = make_problem<T>(n, batch, 21, [&](int b) {
        std::vector<double> d = graded(n, 100.0);
        if (b == 1) d[3] = -0.5, d[9] = -2.0;
        return d;
    });
    Padded<T> A(n, n, 2, batch), F(n, n, 1, batch);
    for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b], Uplo::Lower);
    const std::vector<int32_t> info = run_syfn<T, B>(*this->ctx, A.view, F.view, SyfnFunction<Real<T>>::inv_sqrt(), Uplo::Lower);
    EXPECT_EQ(info, (std::vector<int32_t>{0, 2, 0}));
    for (int b = 0; b < batch; ++b) {
        std::vector<double> g(n);
        for (int k = 0; k < n; ++k) g[k] = p.d[b][k] > 0 ? 1.0 / std::sqrt(p.d[b][k]) : 0.0;
        EXPECT_VERIFY_SLACK(T, Check::factorization, n, frob_diff(F.item(b), q_diag_qh(p.Q[b], g, n)),
                            verify::Slack{100.0, "forward error of A^{-1/2} scales with cond(A) = 100"});
    }
    constexpr int m = 4;
    Padded<T> V(n, m, 0, 1), G(n, n, 0, 1);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i) V.view(i, j, 0) = to_t<T>(cd(i == j ? 1.0 : 0.0, 0.0));
    UnifiedVector<Real<T>> W(m);
    W[0] = Real<T>(0), W[1] = Real<T>(-1), W[2] = Real<T>(0), W[3] = Real<T>(16);
    const double tol = 8.0 * std::numeric_limits<Real<T>>::epsilon();
    auto diag = [&](int k) { return to_cd(G.view(k, k, 0)); };
    EXPECT_EQ((run_from_eigen<T, B>(*this->ctx, V.view, W, G.view, SyfnFunction<Real<T>>::inverse())), std::vector<int32_t>{2});
    EXPECT_NEAR(diag(1).real(), -1.0, tol) << "inverse of -1";
    EXPECT_NEAR(diag(3).real(), 1.0 / 16.0, tol / 16.0);
    EXPECT_EQ(diag(0), cd(0.0)) << "an out-of-domain value contributes 0";
    EXPECT_EQ((run_from_eigen<T, B>(*this->ctx, V.view, W, G.view, SyfnFunction<Real<T>>::sqrt())), std::vector<int32_t>{1});
    EXPECT_NEAR(diag(3).real(), 4.0, 4.0 * tol);
    EXPECT_EQ(diag(1), cd(0.0));
    EXPECT_EQ((run_from_eigen<T, B>(*this->ctx, V.view, W, G.view, SyfnFunction<Real<T>>::inv_sqrt())), std::vector<int32_t>{3});
    EXPECT_NEAR(diag(3).real(), 0.25, 0.25 * tol);
    EXPECT_EQ(diag(2), cd(0.0));
}

// 1024 identical items: every F bit-identical to item 0, item 0 correct.
TYPED_TEST(SyfnTest, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 16, batch = 1024;
    std::mt19937 gen(8);
    const HostMat Q = householder_q<T>(n, gen);
    const std::vector<double> d = graded(n, 1.0e2);
    const HostMat Ah = q_diag_qh(Q, d, n);
    Padded<T> A(n, n, 1, batch), F(n, n, 3, batch);
    for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, Ah, Uplo::Lower);
    run_syfn<T, B>(*this->ctx, A.view, F.view, SyfnFunction<Real<T>>::inv_sqrt(), Uplo::Lower);
    int differing = 0;
    for (int b = 1; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            if (std::memcmp(&F.view(0, j, b), &F.view(0, j, 0), sizeof(T) * n) != 0) ++differing;
    EXPECT_EQ(differing, 0);
    std::vector<double> g(n);
    for (int k = 0; k < n; ++k) g[k] = 1.0 / std::sqrt(d[k]);
    EXPECT_VERIFY_SLACK(T, Check::factorization, n, frob_diff(F.item(0), q_diag_qh(Q, g, n)),
                        verify::Slack{100.0, "forward error of A^{-1/2} scales with cond(A) = 100"});
}

// An out-of-order queue gives the in-order result bit for bit. Its A is written by a copy still
// pending on that queue, and the batch is large enough that the internal copy, syev and product
// overlap if nothing orders them. Still a race: an unordered run can be lucky.
TYPED_TEST(SyfnTest, OutOfOrderQueueMatchesInOrder) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 48, distinct = 8, batch = 256;
    Problem p = make_problem<T>(n, distinct, 4, [&](int) { return graded(n, 1.0e2); });
    Padded<T> A(n, n, 1, batch), A2(n, n, 1, batch), F1(n, n, 1, batch), F2(n, n, 1, batch);
    for (int b = 0; b < batch; ++b) store_triangle<T>(A.view, b, p.A[b % distinct], Uplo::Lower);
    run_syfn<T, B>(*this->ctx, A.view, F1.view, SyfnFunction<Real<T>>::inv_sqrt(), Uplo::Lower);
    Queue ooo(this->ctx->device(), B, false);
    (void)MatrixView<T>::copy(ooo, A2.view, A.view);
    run_syfn<T, B>(ooo, A2.view, F2.view, SyfnFunction<Real<T>>::inv_sqrt(), Uplo::Lower);
    EXPECT_EQ(std::memcmp(F1.view.data_ptr(), F2.view.data_ptr(), sizeof(T) * F1.elems()), 0);
}

// m = 0 eigenpairs: F is exactly zero.
TYPED_TEST(SyfnTest, NoEigenpairsGivesZero) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Padded<T> V(6, 0, 0, 2), F(6, 6, 2, 2);
    UnifiedVector<Real<T>> W(1);
    const std::vector<int32_t> info = run_from_eigen<T, B>(*this->ctx, V.view, W, F.view, SyfnFunction<Real<T>>::exp());
    EXPECT_EQ(info, (std::vector<int32_t>{0, 0}));
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < 6; ++j)
            for (int i = 0; i < 6; ++i) EXPECT_EQ(to_cd(F.view(i, j, b)), cd(0.0));
    EXPECT_TRUE(F.pad_intact());
}

TYPED_TEST(SyfnTest, RejectsBadArguments) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    Matrix<T> A(6, 6, 2), R(6, 5, 2), F(6, 6, 2), F3(6, 6, 3), V(6, 4, 2), Vw(4, 6, 2);
    UnifiedVector<Real<T>> W(8), Wshort(7);
    UnifiedVector<int32_t> info(2), infoshort(1);
    UnifiedVector<std::byte> ws(1 << 20);
    const auto inv = SyfnFunction<Real<T>>::inv_sqrt();
    auto f = [&](const MatrixView<T>& a, const MatrixView<T>& o, SyfnFunction<Real<T>> fn, Uplo u, size_t bytes,
                 Span<int32_t> in) { syfn<B, T>(q, a, o, fn, u, ws.to_span().subspan(0, bytes), in).wait(); };
    auto e = [&](const MatrixView<T>& v, UnifiedVector<Real<T>>& w, const MatrixView<T>& o, SyfnFunction<Real<T>> fn,
                 size_t bytes) { syfn_from_eigen<B, T>(q, v, w.to_span(), o, fn, ws.to_span().subspan(0, bytes), Span<int32_t>{}).wait(); };
    EXPECT_THROW(f(R, F, inv, Uplo::Lower, ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F3, inv, Uplo::Lower, ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, static_cast<Uplo>(77), ws.size(), info.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, Uplo::Lower, ws.size(), infoshort.to_span()), batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, SyfnFunction<Real<T>>::with_weights(Wshort.to_span()), Uplo::Lower, ws.size(), info.to_span()),
                 batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, SyfnFunction<Real<T>>{static_cast<SyfnKind>(9), {}}, Uplo::Lower, ws.size(), info.to_span()),
                 batchlas::invalid_argument);
    EXPECT_THROW(f(A, F, inv, Uplo::Lower, 64, info.to_span()), batchlas::workspace_error);
    EXPECT_THROW(e(V, Wshort, F, inv, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(Vw, W, F, inv, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(V, W, R, inv, ws.size()), batchlas::invalid_argument);
    Matrix<T> Fv(6, 6, 2);
    EXPECT_THROW(e(MatrixView<T>(Fv, 6, 4), W, Fv, inv, ws.size()), batchlas::invalid_argument) << "F aliases V";
    const MatrixView<T> Fvl(Fv.data_ptr() + 2 * 6, 6, 4, 6, 36, 2);
    EXPECT_THROW(e(Fvl, W, Fv, inv, ws.size()), batchlas::invalid_argument) << "V is later columns of F";
    if (q.device().type == DeviceType::GPU) {
        std::vector<Real<T>> host_w(8, Real<T>(1));
        const auto hw = SyfnFunction<Real<T>>::with_weights(Span<Real<T>>(host_w.data(), host_w.size()));
        EXPECT_THROW(e(V, W, F, hw, ws.size()), batchlas::invalid_argument) << "host weights, explicit";
        EXPECT_THROW((void)syfn_from_eigen(q, MatrixView<T>(V), W.to_span(), MatrixView<T>(F), hw, ws.to_span(),
                                           Span<int32_t>{}),
                     batchlas::invalid_argument)
            << "host weights, deducing";
    }
    EXPECT_THROW(e(V, W, F, SyfnFunction<Real<T>>::with_weights(Wshort.to_span()), ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(e(V, W, F, inv, 16), batchlas::workspace_error);
}

#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
// Vendor-free on a CPU queue: syfn_buffer_size and syfn throw NoRouteError while sizing syev's
// workspace, before any kernel; syfn_from_eigen throws it from syrk after its own kernels, which it
// drains first. The queue stays usable.
TEST(SyfnVendorFree, HostWithoutLapackeThrowsNoRoute) {
    if (kHostVendor) GTEST_SKIP() << "host LAPACKE/CBLAS is built in";
    Queue q(Device("cpu"), Backend::NETLIB, true);
    Matrix<float> A = Matrix<float>::Identity(8, 3), F(8, 8, 3), V = Matrix<float>::Identity(8, 3);
    UnifiedVector<float> W(24, 1.0f);
    UnifiedVector<std::byte> ws(size_t(1) << 20);
    const auto inv = SyfnFunction<float>::inv_sqrt();
    EXPECT_THROW((void)(syfn_buffer_size<Backend::NETLIB, float>(q, A, F, inv, Uplo::Lower)), batchlas::NoRouteError);
    EXPECT_THROW((void)(syfn<Backend::NETLIB, float>(q, A, F, inv, Uplo::Lower, ws.to_span(), Span<int32_t>{})),
                 batchlas::NoRouteError);
    EXPECT_THROW((run_from_eigen<float, Backend::NETLIB>(q, V, W, F, SyfnFunction<float>::exp())), batchlas::NoRouteError);
    q.wait();
}
#endif

}  // namespace
