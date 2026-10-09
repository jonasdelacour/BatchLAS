// Host-only tests of batchlas::verify (docs/design/verification.md): scalars, norms, items, tolerances, inputs.

#include <batchlas/verify/inputs.hh>
#include <batchlas/verify/items.hh>
#include <batchlas/verify/norms.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/scalar.hh>
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
#include <sstream>
#include <string>
#include <vector>

using batchlas::MatrixFormat;
using batchlas::MatrixView;
using batchlas::verify::Check;

namespace {
const double kNaN = std::numeric_limits<double>::quiet_NaN();
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

TEST(Scalar, RngMatchesFactorBench) {
    // First four next() values of Rng(12345), recorded from a copy of the Rng struct in
    // benchmarks/factor_bench.cc as of commit b4e9c1b2.
    batchlas::verify::Rng rng(12345);
    EXPECT_EQ(rng.next(), 0x1.0fc129b8p-1);
    EXPECT_EQ(rng.next(), -0x1.d47befbp-3);
    EXPECT_EQ(rng.next(), -0x1.5068e7bp-2);
    EXPECT_EQ(rng.next(), 0x1.4d723aap-1);
}

TEST(Scalar, MakeAndPromote) {
    EXPECT_EQ(batchlas::verify::make<float>(1.5, 7.0), 1.5f);
    const auto z = batchlas::verify::make<std::complex<float>>(1.5, -2.0);
    EXPECT_EQ(batchlas::verify::up(z), std::complex<double>(1.5, -2.0));
    EXPECT_EQ(batchlas::verify::abs(std::complex<double>(3.0, 4.0)), 5.0);
    EXPECT_EQ(batchlas::verify::conj(std::complex<double>(3.0, 4.0)), std::complex<double>(3.0, -4.0));
}

TEST(Items, DefaultIncludesLastAndMiddle) {
    using batchlas::verify::default_items;
    EXPECT_EQ(default_items(1), (std::vector<int>{0}));
    EXPECT_EQ(default_items(2), (std::vector<int>{0, 1}));
    EXPECT_EQ(default_items(9), (std::vector<int>{0, 4, 8}));
    EXPECT_EQ(batchlas::verify::all_items(3), (std::vector<int>{0, 1, 2}));
}

TEST(Norms, ReadThroughLdAndStride) {
    constexpr int rows = 3, cols = 2, ld = 5, stride = 17, batch = 3;
    std::vector<double> buf(stride * batch, 1e30);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) buf[b * stride + j * ld + i] = (b + 1) * (i + 1) * (j == 0 ? 1.0 : -1.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), rows, cols, ld, stride, batch);

    // Item 2 holds 3*(1,2,3) and -3*(1,2,3): sum of squares 2 * 9 * 14.
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 2), std::sqrt(2.0 * 9.0 * 14.0));
    EXPECT_DOUBLE_EQ(batchlas::verify::max_abs(A, 2), 9.0);
    EXPECT_DOUBLE_EQ(batchlas::verify::one_norm(A, 2), 18.0);

    buf[2 * stride + 1 * ld + 2] = kNaN;
    EXPECT_TRUE(std::isnan(batchlas::verify::frobenius(A, 2)));
    EXPECT_TRUE(std::isnan(batchlas::verify::max_abs(A, 2)));
    EXPECT_TRUE(std::isnan(batchlas::verify::one_norm(A, 2)));
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(2.0 * 14.0));
}

TEST(Norms, ComplexUsesModulus) {
    std::vector<std::complex<float>> buf{{3.0f, 4.0f}, {0.0f, 0.0f}, {0.0f, -12.0f}, {5.0f, 0.0f}};
    MatrixView<std::complex<float>, MatrixFormat::Dense> A(buf.data(), 2, 2, 2, 4, 1);
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(25.0 + 144.0 + 25.0));
    EXPECT_DOUBLE_EQ(batchlas::verify::max_abs(A, 0), 12.0);
    EXPECT_DOUBLE_EQ(batchlas::verify::one_norm(A, 0), 12.0 + 5.0);
}

TEST(Norms, KernelViewAccepted) {
    std::vector<double> buf{1.0, 2.0, 99.0, 3.0, 4.0, 99.0};
    batchlas::KernelMatrixView<double, MatrixFormat::Dense> A(buf.data(), 2, 2, 3, 6, 1);
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(A, 0), std::sqrt(30.0));
}

TEST(Tolerance, BoundAndPass) {
    using batchlas::verify::bound;
    using batchlas::verify::pass;
    EXPECT_EQ(bound<float>(Check::blas, 8), 4.0 * 10.0 * std::ldexp(1.0, -24));
    EXPECT_EQ(bound<double>(Check::blas, 1), 4.0 * 3.0 * std::ldexp(1.0, -53));
    EXPECT_EQ(bound<double>(Check::blas, 0), 4.0 * 2.0 * std::ldexp(1.0, -53));
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
    const std::string path = ::testing::TempDir() + "verify_within.txt";
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
    using batchlas::verify::cabs1;
    using batchlas::verify::finite;
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

TEST(View, LayoutAndDefaultStride) {
    std::vector<double> buf(40, 1e30);
    const auto v = batchlas::verify::view(buf.data(), 3, 2, 5, 0, 2);
    EXPECT_EQ(v.rows(), 3);
    EXPECT_EQ(v.cols(), 2);
    EXPECT_EQ(v.ld(), 5);
    EXPECT_EQ(v.stride(), 10);
    EXPECT_EQ(v.batch_size(), 2);
    const auto w = batchlas::verify::view(buf.data(), 3, 2, 5, 17, 2);
    EXPECT_EQ(w.stride(), 17);
    EXPECT_EQ(batchlas::verify::view(buf.data(), 4, 4, 4).batch_size(), 1);
    EXPECT_THROW(batchlas::verify::view(buf.data(), 6, 2, 5), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::view(buf.data(), 3, 2, 5, -1), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::view(buf.data(), 3, 2, 5, 0, 0), std::invalid_argument);
}

TEST(View, ConstOverloadFeedsTheChecks) {
    constexpr int ld = 4, stride = 11, batch = 3;
    std::vector<std::complex<float>> buf(stride * batch, {1e30f, -1e30f});
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 3; ++i) buf[b * stride + j * ld + i] = {float(b + 1), float(i - j)};
    const std::complex<float>* cbuf = buf.data();
    const auto v = batchlas::verify::view(cbuf, 3, 2, ld, stride, batch);
    // Item 2: six real parts 3, imaginary parts (0, 1, 2) and (-1, 0, 1).
    EXPECT_DOUBLE_EQ(batchlas::verify::frobenius(v, 2), std::sqrt(54.0 + 7.0));
    EXPECT_DOUBLE_EQ(batchlas::verify::max_abs(v, 2), std::abs(std::complex<double>(3.0, 2.0)));
}

// ----------------------------------------------------------------------------------- Inputs

namespace {

using cfloat = std::complex<float>;
using cdouble = std::complex<double>;

constexpr int kN = 7, kLd = 9, kBatch = 3;
constexpr long long kStride = 70;
constexpr std::uint64_t kSeed = 12345;

template <class T> T sentinel() { return batchlas::verify::make<T>(-777.0, -555.0); }

template <class T> std::vector<T> padded_buffer() { return std::vector<T>(kStride * kBatch, sentinel<T>()); }

template <class T> MatrixView<T, MatrixFormat::Dense> square_view(std::vector<T>& buf) {
    return MatrixView<T, MatrixFormat::Dense>(buf.data(), kN, kN, kLd, kStride, kBatch);
}

// FNV-1a 64 over the raw bytes of the whole ld x cols x stride x batch buffer, padding included.
template <class T> std::uint64_t fnv1a(const std::vector<T>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// Recorded from copies of fill_spd / fill_lu / fill_gauss in benchmarks/factor_bench.cc (last changed
// in commit b4e9c1b2) run on a std::vector, built with /opt/dpcpp-cuda clang++ (left-to-right argument
// evaluation, which the bench relies on): n=7, ld=9, stride=70, batch=3, seed 12345, padding
// pre-filled with make<T>(-777, -555).
template <class T> struct Golden;
template <> struct Golden<float> {
    static constexpr std::uint64_t spd = 0x750b3e8ce6cbdfbeULL, lu = 0xfc830640c86ea95bULL, gauss = 0x00731694f7177b43ULL;
};
template <> struct Golden<double> {
    static constexpr std::uint64_t spd = 0xec9db1114108317fULL, lu = 0xbc2af2584858c945ULL, gauss = 0x9115ccbde0ccdd8dULL;
};
template <> struct Golden<cfloat> {
    static constexpr std::uint64_t spd = 0xb5ea33f1e79a9fccULL, lu = 0x38a4780ba4d641c9ULL, gauss = 0x3b2bd1b5208c3c27ULL;
};
template <> struct Golden<cdouble> {
    static constexpr std::uint64_t spd = 0x369193e572a23086ULL, lu = 0xefd18494761318c5ULL, gauss = 0xfcebb90a62babe52ULL;
};

template <class T> void check_spd() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_spd(square_view(buf));
    EXPECT_EQ(fnv1a(buf), Golden<T>::spd);
}
template <class T> void check_lu() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_lu(square_view(buf), kSeed);
    EXPECT_EQ(fnv1a(buf), Golden<T>::lu);
}
template <class T> void check_gauss() {
    auto buf = padded_buffer<T>();
    batchlas::verify::fill_gauss(square_view(buf), kSeed);
    EXPECT_EQ(fnv1a(buf), Golden<T>::gauss);
}

// Every padding element (rows kN..kLd-1 of each column, the gap between items) still holds the sentinel.
template <class T> void expect_padding(const std::vector<T>& buf, int rows, int cols, int ld) {
    for (int b = 0; b < kBatch; ++b)
        for (long long o = 0; o < kStride; ++o) {
            const long long idx = b * kStride + o;
            const bool inside = (o % ld) < rows && (o / ld) < cols;
            const T s = sentinel<T>();
            if (!inside) ASSERT_EQ(std::memcmp(&buf[idx], &s, sizeof(T)), 0) << "padding written at " << idx;
        }
}

template <class T> void check_padding() {
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_spd(square_view(buf));
        expect_padding(buf, kN, kN, kLd);
    }
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_lu(square_view(buf), kSeed);
        expect_padding(buf, kN, kN, kLd);
    }
    {
        auto buf = padded_buffer<T>();
        batchlas::verify::fill_graded_hermitian(square_view(buf), 3.0, kSeed);
        expect_padding(buf, kN, kN, kLd);
    }
    {  // rectangular: rows 6, cols 4 inside ld 9
        auto buf = padded_buffer<T>();
        MatrixView<T, MatrixFormat::Dense> A(buf.data(), 6, 4, kLd, kStride, kBatch);
        batchlas::verify::fill_gauss(A, kSeed);
        expect_padding(buf, 6, 4, kLd);
        buf = padded_buffer<T>();
        MatrixView<T, MatrixFormat::Dense> R(buf.data(), 6, 4, kLd, kStride, kBatch);
        batchlas::verify::fill_random(R, kSeed);
        expect_padding(buf, 6, 4, kLd);
    }
    {  // reflectors: m=8, k=5, tau inc 2 stride 13
        std::vector<T> buf(kStride * kBatch, sentinel<T>());
        std::vector<T> tau(13 * kBatch, sentinel<T>());
        MatrixView<T, MatrixFormat::Dense> A(buf.data(), 8, 5, kLd, kStride, kBatch);
        batchlas::VectorView<T> t(tau.data(), 5, kBatch, 2, 13);
        batchlas::verify::fill_reflectors(A, t, kSeed);
        expect_padding(buf, 8, 5, kLd);
        for (int b = 0; b < kBatch; ++b)
            for (int o = 0; o < 13; ++o)
                if (const T s = sentinel<T>(); o % 2 != 0 || o / 2 >= 5) ASSERT_EQ(std::memcmp(&tau[b * 13 + o], &s, sizeof(T)), 0);
    }
}

}  // namespace

TEST(Inputs, SpdMatchesFactorBench) {
    check_spd<float>();
    check_spd<double>();
    check_spd<cfloat>();
    check_spd<cdouble>();
}

TEST(Inputs, LuMatchesFactorBench) {
    check_lu<float>();
    check_lu<double>();
    check_lu<cfloat>();
    check_lu<cdouble>();
}

TEST(Inputs, GaussMatchesFactorBench) {
    check_gauss<float>();
    check_gauss<double>();
    check_gauss<cfloat>();
    check_gauss<cdouble>();
}

TEST(Inputs, PaddingUntouched) {
    check_padding<float>();
    check_padding<double>();
    check_padding<cfloat>();
    check_padding<cdouble>();
}

TEST(Inputs, RandomIsItemMajorColumnMajor) {
    std::vector<cdouble> buf(kStride * kBatch, sentinel<cdouble>());
    MatrixView<cdouble, MatrixFormat::Dense> A(buf.data(), 3, 2, kLd, kStride, kBatch);
    batchlas::verify::fill_random(A, 7);
    batchlas::verify::Rng rng(7);
    for (int b = 0; b < kBatch; ++b)
        for (int c = 0; c < 2; ++c)
            for (int r = 0; r < 3; ++r) {
                const double re = rng.next();
                const double im = rng.next();
                EXPECT_EQ(buf[b * kStride + c * kLd + r], cdouble(re, im));
            }
}

TEST(Inputs, SquareOnlyGeneratorsThrow) {
    std::vector<double> buf(kStride * kBatch, 0.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), 5, 4, kLd, kStride, kBatch);
    EXPECT_THROW(batchlas::verify::fill_spd(A), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::fill_lu(A, 1), std::invalid_argument);
    EXPECT_THROW(batchlas::verify::fill_graded_hermitian(A, 2.0, 1), std::invalid_argument);
}

namespace {

template <class T> void check_graded_hermitian(double log10_cond, double tol) {
    constexpr int n = 9, ld = 12, batch = 3;
    constexpr long long stride = 130;
    std::vector<T> buf(stride * batch, sentinel<T>());
    MatrixView<T, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
    batchlas::verify::fill_graded_hermitian(A, log10_cond, 99);

    // lambda_i = 10^(-c i/(n-1)); trace = sum lambda and ||A||_F^2 = sum lambda^2 hold for any Q.
    double tr = 0, fro2 = 0;
    for (int i = 0; i < n; ++i) {
        const double l = std::pow(10.0, -log10_cond * i / (n - 1));
        tr += l;
        fro2 += l * l;
    }
    for (int b = 0; b < batch; ++b) {
        const T* p = buf.data() + b * stride;
        double rel = 0, trace = 0, imag_diag = 0;
        for (int c = 0; c < n; ++c) {
            const cdouble d = cdouble(batchlas::verify::up(p[c * ld + c]));
            trace += d.real();
            imag_diag = batchlas::verify::nanmax(imag_diag, std::abs(d.imag()));
            for (int r = 0; r < n; ++r)
                rel = batchlas::verify::nanmax(
                    rel, batchlas::verify::abs(batchlas::verify::up(p[c * ld + r]) - batchlas::verify::conj(batchlas::verify::up(p[r * ld + c]))));
        }
        const double fro = batchlas::verify::frobenius(A, b);
        EXPECT_LE(rel, 1e-12 * fro);
        EXPECT_EQ(imag_diag, 0.0);
        EXPECT_NEAR(trace, tr, tol * tr);
        EXPECT_NEAR(fro * fro, fro2, tol * fro2);
    }
    // Items differ (one shared Rng stream, item-major).
    EXPECT_NE(buf[0 * stride + 1], buf[1 * stride + 1]);
}

}  // namespace

TEST(Inputs, GradedHermitianIsHermitianWithSpectrum) {
    check_graded_hermitian<double>(6.0, 1e-12);
    check_graded_hermitian<cdouble>(6.0, 1e-12);
    check_graded_hermitian<float>(2.0, 1e-5);
    check_graded_hermitian<cfloat>(2.0, 1e-5);
}

TEST(Inputs, GradedHermitianHasRequestedCondition) {
    const int n = 9, ld = 12, batch = 3;
    const long long stride = 130;
    const double c = 6.0;
    bool ran = false;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> buf(stride * batch, sentinel<T>());
        MatrixView<T, MatrixFormat::Dense> A(buf.data(), n, n, ld, stride, batch);
        batchlas::verify::fill_graded_hermitian(A, c, 99);
        for (int b = 0; b < batch; ++b) {
            auto a = batchlas::verify::copy_item(A, b);
            std::vector<double> w;
            if (!batchlas::verify::eigenvalues(n, a, w)) return;
            ran = true;
            EXPECT_NEAR(w[n - 1] / w[0] / std::pow(10.0, c), 1.0, 0.01);
        }
    };
    run(double{});
    run(cdouble{});
    if (!ran) GTEST_SKIP() << "built without LAPACKE: eigenvalue-ratio half skipped";
}

namespace {

// H_i = I - tau_i v_i v_i^H, v_i = (0..0, 1, A(i+1:m, i)); unitary to roundoff for every larfg output.
template <class T> void check_reflectors(double tol) {
    constexpr int m = 8, k = 5, ld = 11, batch = 3, tinc = 2, tstride = 13;
    constexpr long long stride = 100;
    std::vector<T> buf(stride * batch, sentinel<T>());
    std::vector<T> tau(tstride * batch, sentinel<T>());
    MatrixView<T, MatrixFormat::Dense> A(buf.data(), m, k, ld, stride, batch);
    batchlas::VectorView<T> t(tau.data(), k, batch, tinc, tstride);
    batchlas::verify::fill_reflectors(A, t, 4242);
    using D = batchlas::verify::promoted_t<T>;
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < k; ++i) {
            std::vector<D> v(m, D(0));
            v[i] = D(1);
            for (int r = i + 1; r < m; ++r) v[r] = batchlas::verify::up(buf[b * stride + i * ld + r]);
            // Diagonal and above are large finite values nothing may read.
            for (int r = 0; r <= i; ++r) EXPECT_GE(batchlas::verify::abs(batchlas::verify::up(buf[b * stride + i * ld + r])), 32.0);
            const D ti = batchlas::verify::up(tau[b * tstride + i * tinc]);
            double worst = 0;
            for (int c = 0; c < m; ++c)
                for (int r = 0; r < m; ++r) {
                    // (H^H H)(r,c) = sum_q conj(H(q,r)) H(q,c)
                    D acc = 0;
                    for (int q = 0; q < m; ++q) {
                        const D hr = (q == r ? D(1) : D(0)) - ti * v[q] * batchlas::verify::conj(v[r]);
                        const D hc = (q == c ? D(1) : D(0)) - ti * v[q] * batchlas::verify::conj(v[c]);
                        acc += batchlas::verify::conj(hr) * hc;
                    }
                    worst = batchlas::verify::nanmax(worst, batchlas::verify::abs(acc - (r == c ? D(1) : D(0))));
                }
            EXPECT_LE(worst, tol) << "b=" << b << " i=" << i;
        }
}

}  // namespace

TEST(Inputs, ReflectorsHaveUnitTau) {
    check_reflectors<double>(1e-12);
    check_reflectors<cdouble>(1e-12);
    check_reflectors<float>(1e-5);
    check_reflectors<cfloat>(1e-5);
    // tau == 0 occurs for real data when the column below the diagonal is empty: last reflector of a square panel.
    std::vector<double> buf(kStride, 0.0), tau(4, -1.0);
    MatrixView<double, MatrixFormat::Dense> A(buf.data(), 4, 4, kLd, kStride, 1);
    batchlas::VectorView<double> t(tau.data(), 4, 1);
    batchlas::verify::fill_reflectors(A, t, 5);
    EXPECT_EQ(tau[3], 0.0);
}

// ---------------------------------------------------------------------------- reference.hh

TEST(Reference, CopyItemIsPackedAndPromoted) {
    constexpr int m = 3, n = 2, ld = 5, batch = 2;
    constexpr long long stride = 17;
    std::vector<float> buf(stride * batch, sentinel<float>());
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) buf[b * stride + j * ld + i] = float(100 * b + 10 * j + i);
    MatrixView<float, MatrixFormat::Dense> A(buf.data(), m, n, ld, stride, batch);
    const auto d = batchlas::verify::copy_item(A, 1);
    static_assert(std::is_same_v<decltype(d), const std::vector<double>>);
    ASSERT_EQ(d.size(), std::size_t(m * n));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) EXPECT_EQ(d[j * m + i], double(100 + 10 * j + i));
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
    for (int j = 0; j < 4; ++j) a[j * 4 + p[j]] = batchlas::verify::make<T>(1.0, -1.0);
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(4, 4, a, ipiv));
    const std::vector<std::int32_t> want{3, 3, 4, 4};
    EXPECT_EQ(ipiv, want);
    EXPECT_EQ(a[0], batchlas::verify::make<T>(1.0, -1.0));
}
}  // namespace

TEST(Reference, GetrfPivotsInWorkingPrecision) {
    static_assert(std::is_same_v<decltype(batchlas::verify::copy_item_native(
                                     std::declval<const MatrixView<float, MatrixFormat::Dense>&>(), 0)),
                                 std::vector<float>>);
    getrf_pivots_of_permutation<float>();
    getrf_pivots_of_permutation<double>();
    getrf_pivots_of_permutation<cfloat>();
    getrf_pivots_of_permutation<cdouble>();
}

// Found by search: the second pivot is a rounding tie that sgetrf breaks to row 2 and dgetrf (on the
// same, exactly promoted, data) to row 3. Pivots from a promoted copy would read {2, 3, 3}.
TEST(Reference, GetrfPivotsOfFloatDataAreSgetrfs) {
    const float col[9] = {0.0f, -1.0f, 0x1.555556p-1f, -1.0f, -4.5f, 2.0f, -1.0f, -0.25f, 2.5f};
    const std::vector<std::int32_t> single{2, 2, 3}, promoted{2, 3, 3};
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
    const float col[9] = {0.0f, -1.0f, 0x1.555556p-1f, -1.0f, -4.5f, 2.0f, -1.0f, -0.25f, 2.5f};
    constexpr int ld = 5, stride = 17;
    std::vector<float> buf(2 * stride, 1e30f);
    std::vector<cfloat> cbuf(2 * stride, {1e30f, -1e30f});
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < 3; ++i) {
            buf[stride + j * ld + i] = col[j * 3 + i];
            cbuf[stride + j * ld + i] = col[j * 3 + i];
        }
    const std::vector<std::int32_t> single{2, 2, 3};
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(batchlas::verify::view(buf.data(), 3, 3, ld, stride, 2), 1, ipiv));
    EXPECT_EQ(ipiv, single);
    ASSERT_TRUE(batchlas::verify::getrf_pivots(batchlas::verify::view(cbuf.data(), 3, 3, ld, stride, 2), 1, ipiv));
    EXPECT_EQ(ipiv, single);
}

namespace {
// LAPACK's own factor and pivots of A0 must reproduce A0 through lu_solve_residual (X = I, B0 = A0):
// pins the library's interchange order to LAPACK's rather than to a test's own loop.
template <class T> void lu_convention_matches_lapack() {
    constexpr int n = 8, ld = 11;
    std::vector<T> a0(ld * n, sentinel<T>()), eye(ld * n, sentinel<T>());
    batchlas::verify::Rng rng(17);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const double re = rng.next(), im = rng.next();
            a0[j * ld + i] = batchlas::verify::make<T>(re, im);
            eye[j * ld + i] = T(i == j ? 1 : 0);
        }
    const auto A0 = batchlas::verify::view(static_cast<const T*>(a0.data()), n, n, ld);
    auto f = batchlas::verify::copy_item_native(A0, 0);
    std::vector<std::int32_t> ipiv;
    ASSERT_TRUE(batchlas::verify::getrf_pivots(n, n, f, ipiv));
    int swaps = 0;
    for (int k = 0; k < n; ++k) swaps += ipiv[k] != k + 1;
    ASSERT_GE(swaps, 2) << "no interchanges to pin";
    const batchlas::VectorView<std::int32_t> p(ipiv.data(), n, 1);
    const double r = batchlas::verify::lu_solve_residual(batchlas::verify::view(static_cast<const T*>(f.data()), n, n, n), p,
                                                         batchlas::Transpose::NoTrans, batchlas::verify::view(eye.data(), n, n, ld), A0);
    EXPECT_TRUE(batchlas::verify::within<T>(Check::solve, n, r)) << "residual " << r;
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
    for (const auto& [m, n] : {std::pair{7, 5}, std::pair{4, 6}}) {
        std::vector<cdouble> a0(static_cast<std::size_t>(m) * n);
        batchlas::verify::Rng rng(5);
        for (auto& x : a0) x = cdouble(rng.next(), rng.next());
        auto a = a0;
        std::vector<cdouble> tau;
        ASSERT_TRUE(batchlas::verify::geqrf_tau(m, n, a, tau));
        ASSERT_EQ(tau.size(), std::size_t(std::min(m, n)));
        const auto A0 = batchlas::verify::view(static_cast<const cdouble*>(a0.data()), m, n, m);
        const auto F = batchlas::verify::view(static_cast<const cdouble*>(a.data()), m, n, m);
        const batchlas::VectorView<cdouble> t(tau.data(), std::min(m, n), 1);
        EXPECT_LT(batchlas::verify::qr_residual(A0, F, t), 1e-14);
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
    EXPECT_FALSE(batchlas::verify::getrf_pivots(batchlas::verify::view(f.data(), 2, 2, 2), 0, ipiv));
    std::vector<double> tau;
    EXPECT_FALSE(batchlas::verify::geqrf_tau(2, 2, a, tau));
}

#endif
