// Tests of batchlas::verify (docs/design/verification.md): tolerances, LAPACKE references, and
// MatrixView's verification scalars and item lists.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/tolerance.hh>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "test_utils.hh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <complex>
#include <fstream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::verify::Check;

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
template <class T> using MV = MatrixView<T, MatrixFormat::Dense>;
}

TEST(Scalar, NanmaxPropagates) {
    EXPECT_TRUE(std::isnan(batchlas::verify::nanmax(kNaN, 1.0)));
    EXPECT_TRUE(std::isnan(batchlas::verify::nanmax(1.0, kNaN)));
    EXPECT_EQ(batchlas::verify::nanmax(1.0, 2.0), 2.0);
}

TEST(Scalar, EpsIsUnitRoundoff) {
    EXPECT_EQ(batchlas::verify::eps<float>(), std::ldexp(1.0, -24));
    EXPECT_EQ(batchlas::verify::eps<std::complex<float>>(), std::ldexp(1.0, -24));
    EXPECT_EQ(batchlas::verify::eps<double>(), std::ldexp(1.0, -53));
    EXPECT_EQ(batchlas::verify::eps<std::complex<double>>(), std::ldexp(1.0, -53));
}

TEST(Scalar, MakeAndPromote) {
    using CF = MV<std::complex<float>>;
    EXPECT_EQ(MV<float>::make(1.5, 7.0), 1.5f);
    const auto z = CF::make(1.5, -2.0);
    static_assert(std::is_same_v<decltype(z), const std::complex<float>>);
    EXPECT_EQ(CF::up(z), std::complex<double>(1.5, -2.0));
    static_assert(std::is_same_v<decltype(MV<float>::up(1.0f)), double>);
    EXPECT_EQ(MV<std::complex<double>>::conj(std::complex<double>(3.0, 4.0)), std::complex<double>(3.0, -4.0));
    // conj and make keep the working type: a test builds T-valued data with them.
    static_assert(std::is_same_v<decltype(MV<float>::conj(1.0f)), float>);
    EXPECT_EQ(CF::conj(std::complex<float>(3.0f, 4.0f)), std::complex<float>(3.0f, -4.0f));
    EXPECT_EQ(CF::make(0.5, -0.25), std::complex<float>(0.5f, -0.25f));
    EXPECT_EQ(MV<double>::make(0.5, -0.25), 0.5);
}

TEST(Items, DefaultIncludesLastAndMiddle) {
    std::vector<double> buf(9, 0.0);
    const auto batch_of = [&](int b) { return MV<double>(buf.data(), 1, 1, 1, 1, b); };
    EXPECT_EQ(batch_of(1).default_items(), (std::vector<int>{0}));
    EXPECT_EQ(batch_of(2).default_items(), (std::vector<int>{0, 1}));
    EXPECT_EQ(batch_of(9).default_items(), (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(batch_of(3).all_items(), (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(batchlas::VectorView<double>(buf.data(), 2, 3, 1, 3).all_items(), (std::vector<int>{0, 1, 2}));
}

TEST(Tolerance, BoundAndPass) {
    using batchlas::verify::bound;
    using batchlas::verify::pass;
    EXPECT_EQ(bound<float>(Check::blas, 8), 8.0 * 10.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(bound<double>(Check::blas, 1), 8.0 * 3.0 * std::ldexp(1.0, -53));
    EXPECT_EQ(bound<double>(Check::blas, 0), 8.0 * 2.0 * std::ldexp(1.0, -53));
    EXPECT_EQ(bound<float>(Check::orthogonality, 5), 32.0 * 5.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(bound<float>(Check::orthogonality_rotations, 5), 256.0 * 5.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(bound<double>(Check::eigen_residual, 3), 32.0 * 3.0 * std::ldexp(1.0, -53));
    EXPECT_EQ(bound<double>(Check::solve, 0), bound<double>(Check::solve, 1));
    EXPECT_FALSE(pass<double>(Check::solve, 10, kNaN));
    const double b = bound<double>(Check::solve, 10);
    EXPECT_TRUE(pass<double>(Check::solve, 10, b));
    EXPECT_FALSE(pass<double>(Check::solve, 10, std::nextafter(b, 1.0)));
}

TEST(Tolerance, RecordsWhenAsked) {
    const std::string path = ::testing::TempDir() + "verify_record.txt";
    std::remove(path.c_str());
    ASSERT_EQ(setenv("BATCHLAS_VERIFY_RECORD", path.c_str(), 1), 0);
    (void)batchlas::verify::pass<std::complex<float>>(Check::orthogonality, 7, 1e-6);
    unsetenv("BATCHLAS_VERIFY_RECORD");
    (void)batchlas::verify::pass<float>(Check::solve, 7, 1e-6);  // unset: nothing appended

    std::ifstream in(path);
    std::string line;
    ASSERT_TRUE(std::getline(in, line));
    std::istringstream fields(line);
    std::string kind, dtype;
    int n = 0;
    double value = 0, bnd = 0, factor = 0;
    std::string reason;
    ASSERT_TRUE(bool(fields >> kind >> dtype >> n >> value >> bnd >> factor >> reason));
    EXPECT_EQ(kind, "orthogonality");
    EXPECT_EQ(dtype, "cfloat");
    EXPECT_EQ(n, 7);
    EXPECT_EQ(value, 1e-6);
    EXPECT_EQ(bnd, batchlas::verify::bound<std::complex<float>>(Check::orthogonality, 7));
    EXPECT_EQ(factor, 1.0);
    EXPECT_EQ(reason, "-");
    EXPECT_FALSE(std::getline(in, line));
    std::remove(path.c_str());
}

TEST(Tolerance, SlackScalesTheBoundEitherWay) {
    using batchlas::verify::bound;
    using batchlas::verify::Slack;
    const double b = bound<float>(Check::factorization, 12);
    EXPECT_EQ(bound<float>(Check::factorization, 12, Slack{4.0, "x"}), 4.0 * b);
    EXPECT_EQ(bound<float>(Check::factorization, 12, Slack{0.25, "x"}), 0.25 * b);
    EXPECT_TRUE(batchlas::verify::pass<float>(Check::factorization, 12, 3.0 * b, Slack{4.0, "x"}));
    EXPECT_FALSE(batchlas::verify::pass<float>(Check::factorization, 12, 3.0 * b));
    EXPECT_FALSE(batchlas::verify::pass<float>(Check::factorization, 12, 0.5 * b, Slack{0.25, "x"}));
    EXPECT_FALSE(batchlas::verify::pass<float>(Check::factorization, 12, kNaN, Slack{4.0, "x"}));
}

TEST(Tolerance, SlackNeedsAReasonAndAPositiveFactor) {
    using batchlas::verify::Slack;
    EXPECT_THROW(batchlas::verify::bound<double>(Check::solve, 3, Slack{2.0, ""}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::bound<double>(Check::solve, 3, Slack{2.0, nullptr}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::bound<double>(Check::solve, 3, Slack{0.0, "r"}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::bound<double>(Check::solve, 3, Slack{-1.0, "r"}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::bound<double>(Check::solve, 3, Slack{kNaN, "r"}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::pass<double>(Check::solve, 3, 0.0, Slack{2.0, ""}), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::within<double>(Check::solve, 3, 0.0, Slack{0.0, "r"}), std::invalid_argument);
}

TEST(Tolerance, WithinIsPassWithoutRecording) {
    using batchlas::verify::Slack;
    using batchlas::verify::within;
    const std::string path = ::testing::TempDir() + "within_never_records.txt";
    std::remove(path.c_str());
    ASSERT_EQ(setenv("BATCHLAS_VERIFY_RECORD", path.c_str(), 1), 0);
    const double b = batchlas::verify::bound<double>(Check::blas, 5);
    EXPECT_TRUE(within<double>(Check::blas, 5, b));
    EXPECT_FALSE(within<double>(Check::blas, 5, std::nextafter(b, 1.0)));
    EXPECT_FALSE(within<double>(Check::blas, 5, kNaN));
    EXPECT_TRUE(within<double>(Check::blas, 5, 1.5 * b, Slack{2.0, "r"}));
    EXPECT_FALSE(within<double>(Check::blas, 5, 1.5 * b, Slack{0.5, "r"}));
    unsetenv("BATCHLAS_VERIFY_RECORD");
    std::ifstream in(path);
    EXPECT_FALSE(in.good() && in.peek() != std::ifstream::traits_type::eof());
    std::remove(path.c_str());
}

TEST(Tolerance, RecordsSlackFactorReasonAndRawValue) {
    const std::string path = ::testing::TempDir() + "verify_record_slack.txt";
    std::remove(path.c_str());
    ASSERT_EQ(setenv("BATCHLAS_VERIFY_RECORD", path.c_str(), 1), 0);
    (void)batchlas::verify::pass<double>(Check::eigen_residual, 9, 3e-12, batchlas::verify::Slack{8.0, "graded spectrum, see docs"});
    unsetenv("BATCHLAS_VERIFY_RECORD");
    std::ifstream in(path);
    std::string line;
    ASSERT_TRUE(std::getline(in, line));
    std::istringstream fields(line);
    std::string kind, dtype, reason, extra;
    int n = 0;
    double value = 0, bnd = 0, factor = 0;
    ASSERT_TRUE(bool(fields >> kind >> dtype >> n >> value >> bnd >> factor >> reason));
    EXPECT_FALSE(bool(fields >> extra));
    EXPECT_EQ(kind, "eigen_residual");
    EXPECT_EQ(dtype, "double");
    EXPECT_EQ(value, 3e-12);
    EXPECT_EQ(bnd, batchlas::verify::bound<double>(Check::eigen_residual, 9));
    EXPECT_EQ(factor, 8.0);
    EXPECT_EQ(reason, "graded_spectrum,_see_docs");
    std::remove(path.c_str());
}

TEST(Tolerance, PivotRatioBound) {
    EXPECT_EQ(batchlas::verify::pivot_ratio_bound<float>(), 1.0 + 64.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(batchlas::verify::pivot_ratio_bound<std::complex<double>>(), 1.0 + 64.0 * std::ldexp(1.0, -53));
}

TEST(ExpectVerify, PassesWithinTheBound) {
    const double b = batchlas::verify::bound<float>(Check::solve, 4);
    EXPECT_VERIFY(float, Check::solve, 4, b);
    EXPECT_VERIFY_SLACK(float, Check::solve, 4, 2.0 * b, batchlas::verify::Slack{2.0, "r"});
}

TEST(ExpectVerify, FailureNamesValueBoundKindAndN) {
    // Through a variable: the failure also prints the expression, which must not supply the kind's name.
    const Check k = Check::orthogonality;
    EXPECT_NONFATAL_FAILURE(EXPECT_VERIFY(double, k, 11, 0.5), "value 0.5 exceeds bound");
    EXPECT_NONFATAL_FAILURE(EXPECT_VERIFY(double, k, 11, 0.5), "orthogonality");
    EXPECT_NONFATAL_FAILURE(EXPECT_VERIFY(double, k, 11, 0.5), "n=11");
    EXPECT_NONFATAL_FAILURE(EXPECT_VERIFY(std::complex<float>, Check::solve, 3, kNaN), "value nan");
    EXPECT_NONFATAL_FAILURE(EXPECT_VERIFY_SLACK(float, Check::blas, 2, 1.0, batchlas::verify::Slack{3.0, "wide k"}),
                            "slack 3 (wide k)");
}

TEST(Scalar, Cabs1AndFinite) {
    const auto cabs1 = [](auto x) { return MV<decltype(x)>::cabs1(x); };
    const auto finite = [](auto x) { return MV<decltype(x)>::finite(x); };
    EXPECT_EQ(cabs1(-2.5f), 2.5);
    EXPECT_EQ(cabs1(-2.5), 2.5);
    EXPECT_EQ(cabs1(std::complex<float>(3.0f, -4.0f)), 7.0);
    EXPECT_EQ(cabs1(std::complex<double>(-0.5, 0.25)), 0.75);
    const double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(finite(1.0f));
    EXPECT_FALSE(finite(float(kNaN)));
    EXPECT_FALSE(finite(-inf));
    EXPECT_TRUE(finite(std::complex<double>(1.0, -2.0)));
    EXPECT_FALSE(finite(std::complex<double>(1.0, kNaN)));
    EXPECT_FALSE(finite(std::complex<float>(float(inf), 0.0f)));
    EXPECT_TRUE(std::isnan(cabs1(std::complex<float>(0.0f, float(kNaN)))));
}

// 8200 items of 64 x 64 exceed the checks' 2^25-element group, so the check runs in two groups;
// a finite poison in the second group must still be seen (a wrong group offset reads group one twice).
TEST(Engine, SecondGroupKeepsItsItemIndices) {
    batchlas::Queue ctx;
    constexpr int n = 64, batch = 8200, poisoned = 8195;
    std::vector<float> buf(static_cast<std::size_t>(n) * n * batch, 0.0f);
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < n; ++i) buf[static_cast<std::size_t>(b) * n * n + static_cast<std::size_t>(i) * n + i] = 1.0f;
    MatrixView<float, MatrixFormat::Dense> Q(buf.data(), n, n, n, n * n, batch);
    const auto all = Q.all_items();
    EXPECT_EQ(MV<float>::orthogonality(ctx, Q, all), 0.0);
    buf[static_cast<std::size_t>(poisoned) * n * n + 5 * n + 3] = 1e3f;
    EXPECT_GT(MV<float>::orthogonality(ctx, Q, all), 1e3);
    const int clean[] = {0, batch / 2, batch - 1};
    EXPECT_EQ(MV<float>::orthogonality(ctx, Q, clean), 0.0);
}

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

template <class T> T sentinel() { return MV<T>::make(-777.0, -555.0); }

}  // namespace

// ---------------------------------------------------------------------------- reference.hh

TEST(Reference, AstypeItemIsPackedAndPromoted) {
    constexpr int m = 3, n = 2, ld = 5, batch = 2;
    constexpr long long stride = 17;
    std::vector<float> buf(stride * batch, sentinel<float>());
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) buf[b * stride + j * ld + i] = float(100 * b + 10 * j + i);
    MatrixView<float, MatrixFormat::Dense> A(buf.data(), m, n, ld, stride, batch);
    const int one = 1;
    auto d = A.astype<double>(std::span<const int>(&one, 1));
    static_assert(std::is_same_v<decltype(d), batchlas::Matrix<double, MatrixFormat::Dense>>);
    ASSERT_EQ(d.rows(), m);
    ASSERT_EQ(d.cols(), n);
    ASSERT_EQ(d.batch_size(), 1);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) EXPECT_EQ(d.data()[j * m + i], double(100 + 10 * j + i));
    const int bad = batch;
    EXPECT_THROW(A.astype<double>(std::span<const int>(&bad, 1)), std::out_of_range);
}

#if BATCHLAS_VERIFY_HAVE_LAPACKE

TEST(Reference, EigenvaluesOfDiagonal) {
    // Unsorted diagonal: the result must be ascending, not the diagonal order.
    std::vector<double> a(16, 0.0), w;
    const double diag[4] = {3.0, -2.0, 7.0, 0.5};
    for (int i = 0; i < 4; ++i) a[i * 4 + i] = diag[i];
    ASSERT_TRUE(batchlas::verify::eigenvalues(4, a, w));
    ASSERT_EQ(w.size(), 4u);
    const double want[4] = {-2.0, 0.5, 3.0, 7.0};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(w[i], want[i]);

    // [[2, -i], [i, 2]] is Hermitian with eigenvalues 1 and 3; the real part alone has 2, 2.
    std::vector<cdouble> h = {{2, 0}, {0, 1}, {0, -1}, {2, 0}};
    std::vector<double> hv;
    ASSERT_TRUE(batchlas::verify::eigenvalues(2, h, hv));
    EXPECT_NEAR(hv[0], 1.0, 1e-14);
    EXPECT_NEAR(hv[1], 3.0, 1e-14);
}

TEST(Reference, SingularValuesOfDiagonal) {
    // Tall, with a negative entry: descending, and the sign is dropped.
    std::vector<double> a(5 * 3, 0.0), s;
    a[0 * 5 + 0] = 2.0;
    a[1 * 5 + 1] = -5.0;
    a[2 * 5 + 2] = 3.0;
    ASSERT_TRUE(batchlas::verify::singular_values(5, 3, a, s));
    ASSERT_EQ(s.size(), 3u);
    EXPECT_NEAR(s[0], 5.0, 1e-14);
    EXPECT_NEAR(s[1], 3.0, 1e-14);
    EXPECT_NEAR(s[2], 2.0, 1e-14);

    // Wide complex: a unit-modulus phase does not change the singular value.
    std::vector<cdouble> z(2 * 4, 0.0);
    z[0 * 2 + 0] = cdouble(0.0, 4.0);
    z[1 * 2 + 1] = cdouble(3.0, 0.0);
    std::vector<double> zs;
    ASSERT_TRUE(batchlas::verify::singular_values(2, 4, z, zs));
    ASSERT_EQ(zs.size(), 2u);
    EXPECT_NEAR(zs[0], 4.0, 1e-14);
    EXPECT_NEAR(zs[1], 3.0, 1e-14);
}

TEST(Reference, GetrfPivotsOfPermutation) {
    // A(p[j], j) = 1: step by step the pivots are 3, 3, 4, 4 (1-based).
    const int p[4] = {2, 0, 3, 1};
    std::vector<double> a(16, 0.0);
    for (int j = 0; j < 4; ++j) a[j * 4 + p[j]] = 1.0;
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(4, 4, a, ipiv));
    ASSERT_EQ(ipiv.size(), 4u);
    const std::int32_t want[4] = {3, 3, 4, 4};
    for (int i = 0; i < 4; ++i) EXPECT_EQ(ipiv[i], want[i]);

    // Rectangular m > n: min(m, n) pivots.
    std::vector<double> t(3 * 2, 0.0);
    t[0 * 3 + 2] = 1.0;
    t[1 * 3 + 1] = 1.0;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(3, 2, t, ipiv));
    ASSERT_EQ(ipiv.size(), 2u);
    EXPECT_EQ(ipiv[0], 3);
    EXPECT_EQ(ipiv[1], 2);
}

TEST(Reference, GetrfPivotsUseCabs1ForComplex) {
    // Column 0 = (4, 2.7+2.7i): |z| picks row 0 (4 > 3.82), LAPACK's |re|+|im| picks row 1 (5.4 > 4).
    std::vector<cdouble> a = {{4, 0}, {2.7, 2.7}, {1, 0}, {1, 0}};
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(2, 2, a, ipiv));
    EXPECT_EQ(ipiv[0], 2);
    std::vector<cfloat> f = {{4, 0}, {2.7f, 2.7f}, {1, 0}, {1, 0}};
    ASSERT_TRUE(batchlas::verify::getrf_pivots(2, 2, f, ipiv));
    EXPECT_EQ(ipiv[0], 2);
}

namespace {
template <class T> void getrf_pivots_of_permutation() {
    const int p[4] = {2, 0, 3, 1};
    std::vector<T> a(16, T(0));
    for (int j = 0; j < 4; ++j) a[j * 4 + p[j]] = MV<T>::make(1.0, -1.0);
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(4, 4, a, ipiv));
    const std::vector<std::int32_t> want{3, 3, 4, 4};
    EXPECT_EQ(ipiv, want);
    EXPECT_EQ(a[0], MV<T>::make(1.0, -1.0));
}
}  // namespace

TEST(Reference, GetrfPivotsInWorkingPrecision) {
    getrf_pivots_of_permutation<float>();
    getrf_pivots_of_permutation<double>();
    getrf_pivots_of_permutation<cfloat>();
    getrf_pivots_of_permutation<cdouble>();
}

namespace {
// l = 1/3 rounds up in float and down in double, so u = 1 - 3l is -2^-23 in float but at most 2^-54
// in double, FMA or not. Row 3's 2^-40 lies between: no tie for any LAPACK to break differently.
const float kPivotSplit[9] = {3.0f, 1.0f, 0.0f, 3.0f, 1.0f, 0x1p-40f, 0.0f, 0.0f, 1.0f};
}  // namespace

TEST(Reference, GetrfPivotsOfFloatDataAreSgetrfs) {
    const float* col = kPivotSplit;
    const std::vector<std::int32_t> single{1, 2, 3}, promoted{1, 3, 3};
    std::vector<std::int32_t> ipiv;
    std::vector<float> f(col, col + 9);
    ASSERT_TRUE(batchlas::verify::getrf_pivots(3, 3, f, ipiv));
    EXPECT_EQ(ipiv, single);
    std::vector<cfloat> c(col, col + 9);
    ASSERT_TRUE(batchlas::verify::getrf_pivots(3, 3, c, ipiv));
    EXPECT_EQ(ipiv, single);
    std::vector<double> d(col, col + 9);
    ASSERT_TRUE(batchlas::verify::getrf_pivots(3, 3, d, ipiv));
    EXPECT_EQ(ipiv, promoted);
}

TEST(Reference, GetrfPivotsOfAViewItemStayInPrecision) {
    const float* col = kPivotSplit;
    constexpr int ld = 5, stride = 17;
    std::vector<float> buf(2 * stride, 1e30f);
    std::vector<cfloat> cbuf(2 * stride, {1e30f, -1e30f});
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            buf[stride + j * ld + i] = col[j * 3 + i];
            cbuf[stride + j * ld + i] = col[j * 3 + i];
        }
    const std::vector<std::int32_t> single{1, 2, 3};
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(MV<float>(buf.data(), 3, 3, ld, stride, 2), 1, ipiv));
    EXPECT_EQ(ipiv, single);
    ASSERT_TRUE(batchlas::verify::getrf_pivots(MV<cfloat>(cbuf.data(), 3, 3, ld, stride, 2), 1, ipiv));
    EXPECT_EQ(ipiv, single);
}

namespace {
// LAPACK's own factor and pivots of A0 must reproduce A0 through lu_solve_residual (X = I, B0 = A0):
// pins the library's interchange order to LAPACK's rather than to a test's own loop.
template <class T> void lu_convention_matches_lapack() {
    constexpr int n = 8, ld = 11;
    std::vector<T> a0(ld * n, sentinel<T>()), eye(ld * n, sentinel<T>());
    batchlas::Queue ctx;
    auto r = batchlas::Matrix<T>::Random(n, n, false, 1, 17);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            a0[j * ld + i] = r(i, j, 0);
            eye[j * ld + i] = T(i == j ? 1 : 0);
        }
    const MV<T> A0(a0.data(), n, n, ld, ld * n, 1);
    auto native = A0.template astype<T>();
    std::vector<T> f(native.data().begin(), native.data().end());
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(n, n, f, ipiv));
    int swaps = 0;
    for (int k = 0; k < n; ++k) swaps += ipiv[k] != k + 1;
    ASSERT_GE(swaps, 2) << "no interchanges to pin";
    const batchlas::VectorView<std::int32_t> p(ipiv.data(), n, 1);
    const double res = MV<T>::lu_solve_residual(ctx, MV<T>(f.data(), n, n, n, n * n, 1), p, batchlas::Transpose::NoTrans,
                                                MV<T>(eye.data(), n, n, ld, ld * n, 1), A0);
    EXPECT_TRUE(batchlas::verify::within<T>(Check::solve, n, res)) << "residual " << res;
}
}  // namespace

TEST(Reference, LuSolvePivotOrderIsLapacks) {
    lu_convention_matches_lapack<float>();
    lu_convention_matches_lapack<double>();
    lu_convention_matches_lapack<cfloat>();
    lu_convention_matches_lapack<cdouble>();
}

TEST(Reference, GeqrfTauKnownReflector) {
    // [3; 4]: beta = -5, tau = (beta - 3) / beta = 1.6, v = 4 / (3 - beta) = 0.5.
    std::vector<double> a = {3.0, 4.0}, tau;
    ASSERT_TRUE(batchlas::verify::geqrf_tau(2, 1, a, tau));
    ASSERT_EQ(tau.size(), 1u);
    EXPECT_NEAR(a[0], -5.0, 1e-15);
    EXPECT_NEAR(a[1], 0.5, 1e-15);
    EXPECT_NEAR(tau[0], 1.6, 1e-15);
    // [3i; 4]: beta = -5, tau = (1, 0.6), v = 4 / (3i + 5) = (20 - 12i) / 34.
    std::vector<cdouble> z = {{0.0, 3.0}, {4.0, 0.0}}, zt;
    ASSERT_TRUE(batchlas::verify::geqrf_tau(2, 1, z, zt));
    EXPECT_NEAR(std::abs(z[0] - cdouble(-5.0, 0.0)), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(z[1] - cdouble(20.0 / 34.0, -12.0 / 34.0)), 0.0, 1e-15);
    EXPECT_NEAR(std::abs(zt[0] - cdouble(1.0, 0.6)), 0.0, 1e-15);
}

TEST(Reference, GeqrfTauFeedsQrResidual) {
    batchlas::Queue ctx;
    for (const auto& [m, n] : {std::pair{7, 5}, std::pair{4, 6}}) {
        auto r = batchlas::Matrix<cdouble>::Random(m, n, false, 1, 5);
        std::vector<cdouble> a0(r.data().begin(), r.data().end());
        auto a = a0;
        std::vector<cdouble> tau;
        ASSERT_TRUE(batchlas::verify::geqrf_tau(m, n, a, tau));
        ASSERT_EQ(tau.size(), std::size_t(std::min(m, n)));
        const MV<cdouble> A0(a0.data(), m, n, m, m * n, 1), F(a.data(), m, n, m, m * n, 1);
        const batchlas::VectorView<cdouble> t(tau.data(), std::min(m, n), 1);
        EXPECT_LT(MV<cdouble>::qr_residual(ctx, A0, F, t), 1e-14);
    }
}

TEST(Reference, SterfOfKnownTridiagonal) {
    // 2 on the diagonal, -1 off it: eigenvalues 2 - sqrt(2), 2, 2 + sqrt(2), ascending, in d.
    std::vector<double> d = {2, 2, 2}, e = {-1, -1};
    ASSERT_TRUE(batchlas::verify::tridiagonal_eigenvalues(d, e));
    ASSERT_EQ(d.size(), 3u);
    EXPECT_NEAR(d[0], 2.0 - std::sqrt(2.0), 1e-14);
    EXPECT_NEAR(d[1], 2.0, 1e-14);
    EXPECT_NEAR(d[2], 2.0 + std::sqrt(2.0), 1e-14);
}

#else

TEST(Reference, WithoutLapackeReturnsFalse) {
    std::vector<double> a(4, 1.0), w, d = {1, 2}, e = {0.5};
    std::vector<std::int32_t> ipiv;
    EXPECT_FALSE(batchlas::verify::eigenvalues(2, a, w));
    EXPECT_FALSE(batchlas::verify::singular_values(2, 2, a, w));
    EXPECT_FALSE(batchlas::verify::getrf_pivots(2, 2, a, ipiv));
    EXPECT_FALSE(batchlas::verify::tridiagonal_eigenvalues(d, e));
    std::vector<float> f(4, 1.0f);
    EXPECT_FALSE(batchlas::verify::getrf_pivots(2, 2, f, ipiv));
    EXPECT_FALSE(batchlas::verify::getrf_pivots(MV<float>(f.data(), 2, 2, 2, 4, 1), 0, ipiv));
    std::vector<double> tau;
    EXPECT_FALSE(batchlas::verify::geqrf_tau(2, 2, a, tau));
    // Empty input too: a reference that is not there never reads as a pass.
    std::vector<double> none, none_e;
    EXPECT_FALSE(batchlas::verify::eigenvalues(0, none, w));
    EXPECT_FALSE(batchlas::verify::singular_values(0, 3, none, w));
    EXPECT_FALSE(batchlas::verify::getrf_pivots(0, 0, none, ipiv));
    EXPECT_FALSE(batchlas::verify::geqrf_tau(3, 0, none, tau));
    EXPECT_FALSE(batchlas::verify::tridiagonal_eigenvalues(none, none_e));
}

#endif
